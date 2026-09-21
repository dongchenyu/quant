import torch
import torch.cuda.nvtx as nvtx

from my_fp8_kernels_dev import (
    my_f8f8bf16_tensorwise,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

def quant_per_tensor(x):
    x_fp32 = x.float()
    amax = x_fp32.abs().amax()

    scale = torch.clamp(amax / FP8_MAX, min=1e-8)

    q = ((x_fp32 / scale) .clamp(-FP8_MAX, FP8_MAX) .to(FP8_DTYPE))

    return q, scale

@torch.no_grad()
def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 最差 Tensorwise case
    M = 128
    N = 896
    K = 4864
    
    # BF16 original
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    
    qx, x_scale = quant_per_tensor(x)
    qw, w_scale = quant_per_tensor(weight)
    
    qw_t = qw.t()
    
    alpha = float((x_scale * w_scale).item())
    beta = 0.0
    
    for _ in range(10):
        my_f8f8bf16_tensorwise(qx, qw_t, None, alpha, beta, "bfloat16")
        
    torch.cuda.synchronize()
    
    for _ in range(5):
        out = my_f8f8bf16_tensorwise(qx, qw_t, None, alpha, beta, "bfloat16")
        
    torch.cuda.synchronize()
    
if __name__ == "__main__":
    main()
    
'''
ncu \
--section LaunchStats \
--section Occupancy \
--section SpeedOfLight \
--kernel-name regex:^Kernel$ \
env PYTHONPATH=. python runtime_refact/benchmark/ncu_tensorwise_case.py \
> tensorwise_ncu_clean_v2.txt
'''