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

def llm_int8_linear(X, W, threshold=6.0):
    outlier_mask = torch.any(torch.abs(X) > threshold, dim=0)
    normal_mask = ~outlier_mask
    
    print("outlier dims:")
    print(torch.where(outlier_mask)[0])
    print("normal dims:")
    print(torch.where(normal_mask)[0])
    
    # [2, 5]
    X_normal = X[:, normal_mask]
    W_normal = W[:, normal_mask]
    
    # [0, 1, 3, 4, 6, 7]
    X_outlier = X[:, outlier_mask]
    W_outlier = W[:, outlier_mask]
    
    if X_normal.shape[1] > 0:
        X_q, X_scale = symmetric_quant_per_tensor(X_normal)
        W_q, W_scale = symmetric_quant_per_tensor(W_normal)
        
        Y_int32 = X_q.int() @ W_q.int().T
        Y_normal = Y_int32.float() * X_scale * W_scale
        
    if X_outlier.shape[1] > 0:
        Y_outlier = X_outlier @ W_outlier.T
    
    Y = Y_normal + Y_outlier
    return Y
    
def main():
    torch.manual_seed(0)
    
    M = 4
    K = 8
    N = 6
    
    X = torch.randn(M, K)
    W = torch.randn(N, K)
    
    X[:, 2] *= 20
    X[:, 5] *= 15
    
    print("X:")
    print(X)
    
    Y_ref = X @ W.T
    Y_naive = naive_int8_linear(X, W)
    Y_llm_int8 = llm_int8_linear(X, W, threshold=6.0)
    
    mse_naive = torch.mean(Y_ref - Y_naive) ** 2
    mse_llm = torch.mean(Y_ref - Y_llm_int8) ** 2
    
    print("\nFP32:")
    print(Y_ref)

    print("\nNaive INT8:")
    print(Y_naive)

    print("\nLLM.int8:")
    print(Y_llm_int8)

    print("\nNaive INT8 MSE:")
    print(mse_naive)

    print("LLM.int8 MSE:")
    print(mse_llm)

    
if __name__ == "__main__":
    main()