import torch

def symmetric_quant_per_tensor(X):
    max_val = torch.max(torch.abs(X))
    scale = max_val / 127
    
    q_X = torch.round(X / scale)
    q_X = torch.clamp(q_X, -127, 127)
    
    return q_X.to(torch.int8), scale

def naive_int8_linear(X, W):
    X_q, X_scale = symmetric_quant_per_tensor(X)
    W_q, W_scale = symmetric_quant_per_tensor(W)
    
    Y_int32 = X_q.int() @ W_q.int().T
    
    Y = Y_int32.float() * X_scale * W_scale
    return Y

def smooth_quant(X, W, alpha=0.5):
    # X:[M, K]  act_max:[K]
    # W:[N, K]  weight_max:[K]
    act_max = torch.max(torch.abs(X), dim=0).values
    weight_max = torch.max(torch.abs(W), dim=0).values
    
    act_max = torch.clamp(act_max, min=1e-5)
    weight_max = torch.clamp(weight_max, min=1e-5)
    
    s = act_max.pow(alpha) / weight_max.pow(1 - alpha)
    
    X_smooth = X / s
    W_smooth = W * s
    
    return X_smooth, W_smooth, s

def main():
    torch.manual_seed(0)
    
    M = 128
    K = 256
    N = 128
    
    X = torch.randn(M, K)
    W = torch.randn(N, K)
    
    X[:, 10] *= 20
    X[:, 50] *= 30
    X[:, 100] *= 40
    
    Y_ref = X @ W.T
    Y_naive = naive_int8_linear(X, W)
    
    X_smooth, W_smooth, s = smooth_quant(X, W, alpha=0.5)
    Y_smooth_fp32 = (X_smooth @ W_smooth.T)
    
    smooth_fp32_diff = torch.max(torch.abs(Y_ref - Y_smooth_fp32))
    
    print("Smooth FP32 max diff:", smooth_fp32_diff)
    
    Y_smooth_int8 = naive_int8_linear(X_smooth, W_smooth)
    
    mse_naive = torch.mean((Y_ref - Y_naive) ** 2)
    mse_smooth = torch.mean((Y_ref - Y_smooth_int8) ** 2)
    
    print("\nNaive INT8 MSE:")
    print(mse_naive)

    print("\nSmoothQuant INT8 MSE:")
    print(mse_smooth)
    
    print("\nNormal dim s[0]:")
    print(s[0])

    print("Outlier dim s[10]:")
    print(s[10])

    print("Outlier dim s[50]:")
    print(s[50])

    print("Outlier dim s[100]:")
    print(s[100])    
    
    print(torch.max(torch.abs(X), dim=0).values[[0, 10, 50, 100]])
    print(torch.max(torch.abs(X_smooth), dim=0).values[[0, 10, 50, 100]])
    print(torch.max(torch.abs(W), dim=0).values[[0, 10, 50, 100]])
    print(torch.max(torch.abs(W_smooth), dim=0).values[[0, 10, 50, 100]])

if __name__ == "__main__":
    main()
