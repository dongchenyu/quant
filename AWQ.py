import torch

torch.manual_seed(0)

def quantize_int4_per_group(w, group_size=32):
    """
    w: [out_features, in_features]
    W -> INT4 -> dequant -> FP32
    """
    out_features, in_features = w.shape
    
    w_group = w.view(out_features, in_features // group_size, group_size)
    
    max_val = w_group.abs().amax(dim=-1, keepdim=True)
    
    scale = max_val / 7.0
    scale = torch.clamp(scale, min=1e-8)
    
    print("scale shape:", scale.shape)
    
    q = torch.round(w_group / scale)
    q = torch.clamp(q, -7, 7)
    
    w_dequant = q * scale
    
    return w_dequant.view(out_features, in_features)

num_token = 256
in_features = 128
out_features = 256

X = torch.randn(num_token, in_features)
W = torch.randn(out_features, in_features) * 0.1

X[:, 5] *= 10
X[:, 20] *= 8
X[:, 70] *= 6

Y_ref = X @ W.T

w_q_naive = quantize_int4_per_group(W, group_size=32)

Y_naive = X @ w_q_naive.T

naive_error = torch.mean((Y_ref - Y_naive) ** 2)

print("Naive INT4 MSE: ", naive_error.item())

act_scale = X.abs().mean(dim=0)

print("act_scale shape:", act_scale.shape)

print("\nTop activation channels:")

top_values, top_indices = torch.topk(act_scale, k=10)

print("top_values shape:", top_values.shape)
print("top_values:", top_values)

print("top_indices shape:", top_indices.shape)
print("top_indices:", top_indices)

best_alpha = None
best_error = float("inf")
best_scale = None
best_weight = None

for alpha in torch.linspace(0, 1, 21):
    # AWQ scaling: s_j = act_scale_j ^ alpha
    
    s = act_scale.pow(alpha)
    s = s / torch.sqrt(s.max() * s.min())
   
    # XW = (X / s) @ (W * s)^T
    
    X_scaled = X / s;
    W_scaled = W * s.unsqueeze(0)
    
    W_scaled_q = quantize_int4_per_group(W_scaled, group_size=32)
    
    Y_awq = X_scaled @ W_scaled_q.T
    
    error = torch.mean((Y_ref - Y_awq) ** 2)
    
    print(
        f"alpha={alpha.item():.2f}, "
        f"MSE={error.item():.8f}"
    )

    if error.item() < best_error:
        best_error = error.item()
        best_alpha = alpha.item()
        best_scale = s.clone()
        best_weight = W_scaled_q.clone()
        
print("\n========================")
print("Naive INT4 MSE :", naive_error.item())
print("Best AWQ MSE   :", best_error)
print("Best alpha     :", best_alpha)
print("Improvement   :", naive_error.item() / best_error)
    
