import torch

def quant_activation_per_token(X):
    max_val = torch.max(torch.abs(X), dim=1, keepdim=True).values
    
    scale = max_val / 127.0
    scale = torch.clamp(scale, min=1e-8)
    
    X_q = X / scale
    X_q = torch.clamp(X_q, -127, 127)
    
    return X_q.to(torch.int8), scale

def dequant_activation_per_token(X_q, scale):
    return X_q.float() * scale

X_normal = torch.tensor([[0.1, 0.3, -0.8, 1.0]], dtype=torch.float32)
X_outlier = torch.tensor([[0.1, 0.3, -0.8, 100]], dtype=torch.float32)

Xn_q, Xn_scale = quant_activation_per_token(X_normal)
Xo_q, Xo_scale = quant_activation_per_token(X_outlier)

Xn_hat = dequant_activation_per_token(Xn_q, Xn_scale)
Xo_hat = dequant_activation_per_token(Xo_q, Xo_scale)

print("===== Normal =====")
print("X:")
print(X_normal)

print("scale:")
print(Xn_scale)

print("X_q:")
print(Xn_q)

print("dequant:")
print(Xn_hat)

normal_error = (X_normal - Xn_hat).abs()
print("Normal mean error: ", normal_error.mean())

print("\n===== Outlier =====")
print("X:")
print(X_outlier)

print("scale:")
print(Xo_scale)

print("X_q:")
print(Xo_q)

print("dequant:")
print(Xo_hat)

outlier_error = (X_outlier - Xo_hat).abs()
print("Outlier mean error:", outlier_error.mean())