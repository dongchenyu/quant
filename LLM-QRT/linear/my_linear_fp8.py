import torch
from typing import Tuple, Optional

from runtime.sq_fp8_kernels import (
    my_f8f8bf16_tensorwise,
    my_f8f8bf16_rowwise
)

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max

def quant_per_tensor(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().max()
    scale = amax / fp8_max
    scale = torch.clamp(scale, min=1e-8)
    q = (x / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scale.float()

def quant_per_row(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / fp8_max
    scale = torch.clamp(scale, min=1e-8)
    q = (x / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scale.float()

def quant_with_static_scale(x:torch.Tensor, scale:torch.Tensor) -> torch.Tensor:
    return (x / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)

def fp8_linear_ref(
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qw: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    x = qx.to(out_dtype) * x_scale.to(out_dtype)
    w = qw.to(out_dtype) * w_scale.to(out_dtype)
    return torch.nn.functional.linear(x, w, bias)

class SimpleFP8Linear(torch.nn.Module):
    def __init__(self, 
                weight: torch.Tensor, 
                bias:Optional[torch.Tensor] = None,
                granularity: str = "tensorwise",
                activation_mode: str = "dynamic",
                backend: str = "cuda",
                input_scale: Optional[torch.Tensor] = None,
                out_dtype:torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        assert granularity in ("tensorwise", "rowwise")
        assert activation_mode in ("dynamic", "static")
        
        if activation_mode == "static":
            assert granularity == "tensorwise"
            assert input_scale is not None
        
        self.granularity = granularity
        self.activation_mode = activation_mode
        self.out_dtype = out_dtype
        self.backend = backend
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        
        if granularity == "tensorwise":
            qw, w_scale = quant_per_tensor(weight)
        else:
            qw, w_scale = quant_per_row(weight)     
            
        self.register_buffer("qweight", qw)
        self.register_buffer("weight_scale", w_scale)
        
        if bias is None:
            self.bias = None 
        else:
            self.bias = torch.nn.Parameter(
                bias.to(out_dtype), requires_grad=False
            )  

        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.float())
        else:
            self.input_scale = None
    
    def quantize_activation(self, x):
        if self.activation_mode == "static":
            qx = quant_with_static_scale(x, self.input_scale)
            return qx, self.input_scale
        
        if self.granularity == "tensorwise":
            return quant_per_tensor(x)
        
        return quant_per_row(x)
    
    def forward_ref(self, x):
        qx, x_scale = self.quantize_activation(x)
        return fp8_linear_ref(
            qx=qx,
            x_scale=x_scale,
            qw=self.qweight,
            w_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=self.out_dtype
        )
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        assert self.qweight.is_cuda
        
        x_shape = x.shape
        qx, x_scale = self.quantize_activation(x)
        
        qx_2d = qx.reshape(-1, qx.shape[-1])
        
        qweight_t = self.qweight.t()
        
        if self.granularity == "tensorwise":
            alpha = float((x_scale * self.weight_scale).item())
            beta = 1.0 if self.bias is not None else 0.0
            
            output = my_f8f8bf16_tensorwise(
                qx_2d,
                qweight_t,
                self.bias,
                alpha,
                beta,
                "bfloat16" if self.out_dtype == torch.bfloat16 else "float16",
            )
            
        else:
            x_scale_2d = x_scale.reshape(-1, 1).float()
            w_scale_2d = self.weight_scale.t().contiguous().float()
            
            output = my_f8f8bf16_rowwise(
                qx_2d,
                qweight_t,
                self.bias,
                x_scale_2d,
                w_scale_2d,
                True
            )
        return output.view(*x_shape[:-1], self.out_features)
    
    def forward(self,x):
        if self.backend == "ref":
            return self.forward_ref(x)
        return self.forward_cuda(x)
           

if __name__ == "__main__":
    torch.manual_seed(0)
    M = 32
    K = 128
    N = 64
    
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    
    y_bf16 = torch.nn.functional.linear(x, w, b)
    
    tensor_model = SimpleFP8Linear(
            w, b, 
            granularity="tensorwise",
            activation_mode="dynamic",
            backend="cuda").cuda()

    #y_tensor_cuda: cuda kernel，
    #y_tensor_ref: pytorch fp8，
    # y_bf16: pytorch fp16
    
    y_tensor_cuda = tensor_model(x)
    y_tensor_ref = tensor_model.forward_ref(x)
    
    print("tensor CUDA vs ref:",
        (y_tensor_cuda - y_tensor_ref).abs().max().item())

    print("tensor ref vs bf16:",
        (y_tensor_ref - y_bf16).abs().max().item())

    print("tensor CUDA vs bf16:",
        (y_tensor_cuda - y_bf16).abs().max().item())
    
    row_model = SimpleFP8Linear(
                w, b, 
                granularity="rowwise",
                activation_mode="dynamic",
                backend="cuda").cuda()
    
    y_row_cuda = row_model(x)
    y_row_ref = row_model.forward_ref(x)
    
    print("row CUDA vs ref:", (y_row_cuda - y_row_ref).abs().max().item())

    print("row ref vs bf16:", (y_row_ref - y_bf16).abs().max().item())

    print("row CUDA vs bf16:", (y_row_cuda - y_bf16).abs().max().item())
    
    diff = (y_tensor_cuda.float() - y_tensor_ref.float()).abs()
    print("mean diff:", diff.mean().item())
    print("relative max:",
      (diff / y_tensor_ref.float().abs().clamp_min(1e-5)).max().item())
    
    
    
