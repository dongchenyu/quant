import torch

from runtime.sq_fp8_kernels import (
    w8a8_int8_linear_bbf16_obf16_per_tensor
)


@torch.no_grad()
def quantize_per_tensor(x):
    """
    symmetric per-tensor INT8 quantization

    x ~= qx * scale
    """
    qmax = 127
    
    max_val = x.abs().max()
    scale = max_val / qmax
    scale = torch.clamp(scale, min=1e-5)
    
    qx = torch.round(x / scale)
    qx = torch.clamp(qx, -127, 127)
    qx = qx.to(torch.int8)
    
    return qx, scale

@torch.no_grad()
def main():
    torch.manual_seed(0)
    device = "cuda"
    
    # GEMM:
    #
    # A[M,K] @ W.T[K,N]
    #       ↓
    # C[M,N]
    
    M = 128
    N = 512
    K = 1024
    
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    W = torch.randn(N, K, device=device, dtype=torch.float16)
    
    bias = torch.randn(N, device=device, dtype=torch.float16)
    
    print("A: ", A.shape, A.dtype, A.stride())
    print("W: ", W.shape, W.dtype, W.stride())
    
    Aq, scale_a = quantize_per_tensor(A)
    Wq, scale_w = quantize_per_tensor(W)
    
    print("\nscale_a = ", scale_a)
    print("scale_w = ", scale_w)
    
    print("\nAq: ", Aq.shape, Aq.dtype, Aq.stride())
    print("Wq: ", Wq.shape, Wq.dtype, Wq.stride())
    
    Wq_t = Wq.t()
    
    print("\nWq_t")
    print("shape = ", Wq_t.shape)
    print("stride = ", Wq_t.stride())
    print("contiguous =", Wq_t.is_contiguous())
    
    alpha = (scale_a * scale_w).item()
    beta = 1.0
    
    print("\nalpha:", alpha)
    
    y_cutlass = w8a8_int8_linear_bbf16_obf16_per_tensor(Aq, Wq_t, bias, alpha, beta)
    
    acc_ref = Aq.float() @ Wq.float().t()
    y_quant_ref = acc_ref * alpha + bias.float()
    
    y_fp16_ref = A.float() @ W.float().t() + bias.float()
    
    kernel_error = (y_cutlass.float() - y_quant_ref).abs()
    quant_error = (y_cutlass.float() - y_fp16_ref).abs()
    
    print("max error: ", kernel_error.max().item())
    print("mean error:", kernel_error.mean().item())
    
    print("max error: ", quant_error.max().item())
    print("mean error:", quant_error.mean().item())
    
    print("\nfirst few values:")

    print("CUTLASS:", y_cutlass[0, :8])
    print("Quant Ref:", y_quant_ref[0, :8])
    print("FP Ref:", y_fp16_ref[0, :8])
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    