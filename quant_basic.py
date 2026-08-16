import torch

def symmetric_quant(x):
    """
    FP32 -> INT8

    x:
        FP32 tensor

    return:
        q: int8 tensor
        scale: float
    """
    qmax = 127
    max_val = torch.max(torch.abs(x))
    
    scale = max_val / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -127, 127)
    return q.to(torch.int8), scale

def symmetric_dequant(q, scale):
    return q.float() * scale

def int8_linear(X, W):
    X_q, X_scale = symmetric_quant(X)
    W_q, W_scale = symmetric_quant(W)
    
    print("int8 quant X_q:")
    print(X_q)
    
    print("int8 quant W_q:")
    print(W_q)
    
    # 模拟int8量化,此处使用int32是考虑pytorch对int8支持有限, 其实正常用的是int8,只不过这里数值也等价于int8
    Y_int32 = (X_q.to(torch.int32) @ W_q.to(torch.int32).T)
    
    Y = Y_int32.float() * X_scale * W_scale
    
    return Y
    
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(16) * 5
    
    print("Original:")
    print(x)
    
    q, scale = symmetric_quant(x)
    
    print("INT8 quant: ")
    print(q)
    
    print("scale: ")
    print(scale)
    
    x_hat = symmetric_dequant(q, scale)
    
    error = (x - x_hat).abs()
    
    print("max error: ")
    print(error.max())
    print("mean error: ")
    print(error.mean())
    
    M = 4
    K = 8
    N = 6
    
    X = torch.randn(M, K)
    W = torch.randn(N, K)
    
    Y_ref = X @ W.T
    Y_int8_quant = int8_linear(X, W)
    
    print("FP32 result: ")
    print(Y_ref)
    
    print("INT8 quant result:")
    print(Y_int8_quant)
    
    diff = (Y_ref - Y_int8_quant).abs()
    
    print("max diff:")
    print(diff.max())
    
    print("mean diff:")
    print(diff.mean())
    
    relative_error = diff / (Y_ref.abs()+1e-8)

    print("max relative_error:")
    print(relative_error.max())
    
    print("mean relative_error:")
    print(relative_error.mean())

#ziEfO6PEuLYc4~