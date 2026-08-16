import torch

def symmetric_quant_tensor(x):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127
    
    q = torch.round(x / scale)
    q = torch.clamp(q, -127, 127)
    
    return q.to(torch.int8), scale

def symmetric_quant_per_channel(W):
    """
    每一行一个scale

    W:
        [out_features, in_features]

    """
    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
    scale = max_val / 127
    
    q = torch.round(W / scale)
    q = torch.clamp(q, -127, 127)
        
    return q.to(torch.int8), scale
    
def linear_Tensor(X, W):
    X_q, X_scale = symmetric_quant_tensor(X)
    W_q, W_scale = symmetric_quant_tensor(W)
    
    Y_int32 = (X_q.int() @ W_q.int().T)
    Y = Y_int32.float() * X_scale * W_scale
    
    return Y

def linear_per_channel(X, W):
    X_q, X_scale = symmetric_quant_per_channel(X)
    W_q, W_scale = symmetric_quant_per_channel(W)
    
    Y_int32 = (X_q.int() @ W_q.int().T)
    Y = Y_int32.float() * X_scale * W_scale.T
    
    return Y

torch.manual_seed(0)

M = 128
N = 4096
K = 4096

X=torch.randn(M,K)
W=torch.randn(N,K)

Y_ref=X @ W.T

Y_tensor = linear_Tensor(X,W)
Y_channel = linear_per_channel(X,W)

diff_Tensor = (Y_ref - Y_tensor).abs()

print("================")
print("Per Tensor Mean: ", diff_Tensor.mean())

print("Per Tensor MAX: ", diff_Tensor.max())

cos = torch.nn.functional.cosine_similarity(Y_ref.flatten(), Y_tensor.flatten(), dim=0)

print("cos:", cos)

diff_Per_Channel = (Y_ref - Y_channel).abs()

print("================")
print("Per Channel Mean: ", diff_Per_Channel.mean())

print("Per Channel MAX: ", diff_Per_Channel.max())

cos = torch.nn.functional.cosine_similarity(Y_ref.flatten(), Y_channel.flatten(), dim=0)

print("cos:", cos)