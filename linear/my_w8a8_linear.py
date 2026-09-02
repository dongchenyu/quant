import torch
from typing import Tuple, Optional
from runtime.sq_fp8_kernels import cutlass_int8_gemm_per_tensor

INT8_MAX = 127

@torch.no_grad()
def quant_per_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax()
    scale = amax / INT8_MAX
    scale = torch.clamp(scale, min=1e-8)
    
    q = torch.round(x / scale)
    q = torch.clamp(q, -INT8_MAX, INT8_MAX).to(torch.int8)
    
    return q, scale.float()

def int8_linear_ref(
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qw: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype=torch.float16
) -> torch.Tensor:
    x_dequant = qx.to(out_dtype) * x_scale.to(out_dtype)
    w_dequant = qw.to(out_dtype) * w_scale.to(out_dtype)
    
    return torch.nn.functional.linear(
        x_dequant,
        w_dequant,
        bias
    )

class SimpleW8A8Linear(torch.nn.Module):
    """
    weight: 初始化时量化一次
            FP16/BF16 -> INT8 + weight_scale
    
    activation: 每次 forward 动态量化
            FP16/BF16 -> INT8 + input_scale
    
    output:
            INT8 GEMM -> INT32 accum
            -> * (input_scale * weight_scale)
            -> + bias
            -> FP16
    """
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        backend: str = "ref",
        out_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        
        assert backend in ("ref", "cuda")
        
        self.backend = backend
        self.out_dtype = out_dtype
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        
        qweight, weight_scale = quant_per_tensor(weight)
        
        self.register_buffer("qweight", qweight)
        self.register_buffer("weight_scale", weight_scale)
        
        if bias is None:
            self.bias = None
        else:
            self.register_buffer(
                "bias",
                bias.to(out_dtype),
            )
    def forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        
        qx, input_scale = quant_per_tensor(x_2d)
        
        y = int8_linear_ref(
            qx=qx,
            x_scale=input_scale,
            qw=self.qweight,
            w_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=self.out_dtype
        )
        
        return y.view(*x_shape[:-1], self.out_features)
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if cutlass_int8_gemm_per_tensor is None:
            raise RuntimeError("cutlass_int8_gemm_per_tensor don't bind python")
        
        x_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])        
        
        qx, input_scale = quant_per_tensor(x_2d)
        qweight_t = self.qweight.t()
        
        alpha = float((input_scale * self.weight_scale).item())
        beta = 1.0 if self.bias is not None else 0.0
        
        if self.bias is None:
            bias = torch.zeros(
                self.out_features,
                dtype=self.out_dtype,
                device=x.device
            )
        else:
            bias = self.bias

        y = cutlass_int8_gemm_per_tensor(
            qx,
            qweight_t,
            bias,
            alpha,
            beta,
        )        
        
        return y.view(*x_shape[:-1], self.out_features)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.backend == "cuda":
            return self.forward_cuda(x)
        return self.forward_ref(x)
    
if __name__ == "__main__":
    torch.manual_seed(0)
    M = 128
    K = 1024
    N = 512
    
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    
    y_fp16 = torch.nn.functional.linear(x, w, b)
    
    model_ref = SimpleW8A8Linear(w, b, backend="ref", out_dtype=torch.float16).cuda()
    
    y_ref = model_ref(x)
    
    print("INT8 ref vs FP16 max diff:",
        (y_ref - y_fp16).abs().max().item())
    
    if cutlass_int8_gemm_per_tensor is not None:
        model_cuda = SimpleW8A8Linear(w, b, backend="cuda", out_dtype=torch.float16).cuda()
        
        model_cuda.qweight.copy_(model_ref.qweight)
        model_cuda.weight_scale.copy_(model_ref.weight_scale)
        
        y_cuda = model_cuda(x)
        
        diff = (y_cuda.float() - y_ref.float()).abs()
        
        print("CUDA vs ref max diff :", diff.max().item())
        print("CUDA vs ref mean diff:", diff.mean().item())
    