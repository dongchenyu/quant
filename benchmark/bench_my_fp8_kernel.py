import torch

# from runtime.sq_fp8_kernels import (
from my_fp8_kernels_dev import (
    my_f8f8bf16_tensorwise,
    my_f8f8bf16_rowwise,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

# Qwen2.5-0.5B Linear shape
# Linear weight logical shape = [N, K]
LINEAR_SHAPES = [
    ("attn_q_o",    896,  896),
    ("mlp_up_gate", 4864, 896),
    ("mlp_down",    896,  4864),
]

M_VALUES = [1, 16, 128]

def quant_per_tensor(x: torch.Tensor):
    x_fp32 = x.float()
    
    amax = x_fp32.abs().max()
    scale = torch.clamp(amax / FP8_MAX, min=1e-8)
    
    q = (x_fp32 / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    
    return q, scale

def quant_per_row(x: torch.Tensor):
    x_fp32 = x.float()
    
    amax = x_fp32.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(amax / FP8_MAX, min=1e-8)
    
    q = (x_fp32 / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    
    return q, scale

@torch.no_grad()
def benchmark_cuda(fn, warmup=50, iters=200):
    # 使用 CUDA Event 测 GPU elapsed time
    # return: average latency in microseconds
    for _ in range(warmup):
        fn()
        
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    for _ in range(iters):
        fn()

    end.record()
    
    torch.cuda.synchronize()
    
    total_ms = start.elapsed_time(end)

    avg_us = total_ms * 1000.0 / iters

    return avg_us

def calc_tflops(M, N, K, latency_us):
    # GEMM: [M,K] x [K,N]
    # FLOPs = 2 * M * N * K
    flops = 2.0 * M * N * K
    latency_s = latency_us * 1e-6
    return flops / latency_s / 1e12

# BF16 GEMM kernel vs FP8 Tensorwise GEMM kernel vs FP8 Rowwise GEMM kernel
@torch.no_grad()
def run_benchmark_case(M, N, K, name):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    
    weight_t = weight.t()
    
    qx_tensor, x_scale_tensor = quant_per_tensor(x)
    qw_tensor, w_scale_tensor = quant_per_tensor(weight)
    
    qw_tensor_t = qw_tensor.t()
    
    tensor_alpha = float((x_scale_tensor * w_scale_tensor).item())
    tensor_beta = 0.0
    
    qx_row, x_scale_row = quant_per_row(x)
    qw_row, w_scale_row = quant_per_row(weight)
    
    qw_row_t = qw_row.t()
    w_scale_row_t = w_scale_row.t()
    
    assert weight_t.shape == (K, N)

    assert qw_tensor_t.shape == (K, N)
    assert qw_row_t.shape == (K, N)
    
    assert qw_tensor_t.stride(0) == 1
    assert qw_row_t.stride(0) == 1
    
    # BF16 baseline
    # CUDA tensor 上的 torch.mm 底层走 cuBLAS / cuBLASLt
    def bf16_gemm():
        return torch.mm(x, weight_t)
    
    # FP8 Tensorwise CUTLASS
    def fp8_tensorwise_gemm():
        return my_f8f8bf16_tensorwise(qx_tensor, qw_tensor_t, None,
            tensor_alpha, tensor_beta, "bfloat16")
        
    # FP8 Rowwise CUTLASS
    def fp8_rowwise_gemm():
        return my_f8f8bf16_rowwise(qx_row, qw_row_t, None,
            x_scale_row, w_scale_row_t, True)
        
    # Benchmark
    bf16_us = benchmark_cuda(bf16_gemm)
    tensor_us = benchmark_cuda(fp8_tensorwise_gemm)
    row_us = benchmark_cuda(fp8_rowwise_gemm)
    
    # TFLOPS
    bf16_tflops = calc_tflops(M, N, K, bf16_us)
    tensor_tflops = calc_tflops(M, N, K, tensor_us)
    row_tflops = calc_tflops(M, N, K, row_us)
    
    # Speedup
    tensor_speedup = bf16_us / tensor_us
    row_speedup = bf16_us / row_us
    
    return [
        {
            "name": name, "M": M, "N": N, "K": K,
            "kernel": "BF16 cuBLAS", "latency_us": bf16_us, "tflops": bf16_tflops, "speedup": 1.0
        },
        {
            "name": name, "M": M, "N": N, "K": K,
            "kernel": "FP8 Tensor", "latency_us": tensor_us, "tflops": tensor_tflops, "speedup": tensor_speedup
        },
        {
            "name": name, "M": M, "N": N, "K": K,
            "kernel": "FP8 Row", "latency_us": row_us, "tflops": row_tflops, "speedup": row_speedup
        },
    ]
    
@torch.no_grad()
def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    assert torch.cuda.is_available()

    device_name = torch.cuda.get_device_name()
    
    print()
    print("=" * 105)
    print("FP8 Tensorwise / Rowwise vs BF16 cuBLAS Benchmark")
    print("=" * 105)
    print(f"GPU: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    try:
        blas = torch.backends.cuda.preferred_blas_library()
        print(f"Preferred BLAS: {blas}")
    except Exception:
        pass

    print("=" * 105)

    results = []
    
    for name, N, K in LINEAR_SHAPES:
        for M in M_VALUES:
            print(
                f"Running: "
                f"{name}, "
                f"M={M}, N={N}, K={K}"
            )
            case_results = run_benchmark_case(M=M, N=N, K=K, name=name)
            results.extend(case_results)
            
    print()
    print("=" * 105)
    print("Benchmark Results")
    print("=" * 105)

    header = (
        f"{'Layer':<14}"
        f"{'M':>6}"
        f"{'N':>7}"
        f"{'K':>7}"
        f"{'Kernel':>16}"
        f"{'Latency(us)':>16}"
        f"{'TFLOPS':>14}"
        f"{'Speedup':>12}"
    )

    print(header)
    print("-" * 105)

    for r in results:
        print(
            f"{r['name']:<14}"
            f"{r['M']:>6}"
            f"{r['N']:>7}"
            f"{r['K']:>7}"
            f"{r['kernel']:>16}"
            f"{r['latency_us']:>16.3f}"
            f"{r['tflops']:>14.3f}"
            f"{r['speedup']:>11.3f}x"
        )

    print("=" * 105)
    
if __name__ == "__main__":
    main()
    
    
'''
V1:
Tensorwise:
ThreadblockShape = <128, 128, 64>
WarpShape        = <64, 64, 64>
InstructionShape = <16, 8, 32>
Stages           = 3
Swizzle          = Identity

Rowwise:
ThreadblockShape = <32, 64, 128>
WarpShape        = <16, 64, 64>
InstructionShape = <16, 8, 32>
Stages           = 5
Swizzle          = StreamK
'''

# python runtime_refact/benchmark/bench_my_fp8_kernel.py