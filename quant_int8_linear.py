import torch

def quant_activation_per_token(X):
    max_val = torch.max(torch.abs(X), dim=1, keepdim=True).values
    
    X_scale = max_val / 127.0
    X_scale = torch.clamp(X_scale, min=1e-8)
    
    X_q = torch.round(X / X_scale)
    X_q = torch.clamp(X_q, -127, 127)
    
    return X_q.to(torch.int8), X_scale

def dequant(X_q, scale):
    return X_q.float() * scale

def quant_weight_per_channel(W):
    max_val = torch.max(torch.abs(W), dim=1, keepdim=True).values
    
    W_scale = max_val / 127.0
    W_scale = torch.clamp(W_scale, min=1e-8)
    
    W_q = torch.round(W / W_scale)
    W_q = torch.clamp(W_q, -127, 127)
    
    return W_q.to(torch.int8), W_scale

def int8_linear(X, W_q, W_scale):
    X_q, X_scale = quant_activation_per_token(X)
    
    # quant: INT8 * INT8 -> INT32 accumulate
    Y_int32 = X_q.int() @ W_q.int().T
    
    # dequant:
    Y = Y_int32.float() * X_scale * W_scale.T
    
    return Y, X_q, X_scale, Y_int32
    
torch.manual_seed(0)

# Model Weight: [2, 4]
W = torch.tensor([[0.2, -0.5, 1.0, 0.4],
                  [3.0, -1.0, 0.5, -2.0]],dtype=torch.float32)

# offline Weight quantization
W_q, W_scale = quant_weight_per_channel(W)

print("============ Weight ============")
print("W:")
print(W)

print("W_scale:")
print(W_scale)

print("W_q")
print(W_q)


# Runtime activation: 3 tokens [3,4]
X = torch.tensor([[0.1, 0.3, -0.8, 1.0],
                  [2.0, -1.0, 0.5, 4.0],
                  [-0.2, 0.1, 0.4, -0.3]],dtype=torch.float32)

Y_ref = X @ W.T

Y_quant, X_q, X_scale, Y_int32 = int8_linear(X, W_q, W_scale)

print("\n============ Activation ============")
print("X:")
print(X)

print("X_scale:")
print(X_scale)

print("X_q:")
print(X_q)

print("\n============ GEMM ============")
print("Y_int32:")
print(Y_int32)

print("\n============ Result ============")
print("FP32:")
print(Y_ref)

print("Int8 quant:")
print(Y_quant)

diff = torch.abs(Y_ref - Y_quant)

print("mean error:")
print(diff.mean())

print("max error:")
print(diff.max())

cos = torch.nn.functional.cosine_similarity(Y_ref.flatten(), Y_quant.flatten(), dim=0)

print("Cos similarity:")
print(cos)

X_normal = torch.tensor([
    [0.1, 0.3, -0.8, 1.0]
], dtype=torch.float32)

X_outlier = torch.tensor([
    [0.1, 0.3, -0.8, 100.0]
], dtype=torch.float32)

Y_ref_normal = X_normal @ W.T
Y_ref_outlier = X_outlier @ W.T

Y_quant_normal, _, _, _ = int8_linear(
    X_normal,
    W_q,
    W_scale
)

Y_quant_outlier, _, _, _ = int8_linear(
    X_outlier,
    W_q,
    W_scale
)
print("\n============ Outlier ============")
print("Normal diff:")
print((Y_ref_normal - Y_quant_normal).abs())

print("Outlier diff:")
print((Y_ref_outlier - Y_quant_outlier).abs())

