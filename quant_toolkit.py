import torch

def calibrate_getvalue_per_tensor(activations):
    min_val = None
    max_val = None
    
    for X in activations:
        cur_min = torch.min(X)
        cur_max = torch.max(X)
        
        if min_val is None:
            min_val = cur_min
            max_val = cur_max
        else:
            min_val = torch.minimum(min_val, cur_min)
            max_val = torch.maximum(max_val, cur_max)
    
    return min_val, max_val

def calibrate_getvalue_per_channel(activations):
    min_val = None
    max_val = None
        
    for X in activations:
        cur_min = torch.min(X, dim=0, keepdim=True).values
        cur_max = torch.max(X, dim=0, keepdim=True).values
        
        if min_val is None:
            min_val = cur_min
            max_val = cur_max
        else:
            min_val = torch.minimum(min_val, cur_min) 
            max_val = torch.maximum(max_val, cur_max)
            
    return min_val, max_val
            
def symmetric_calibrate_per_tensor(activations):
    min_val, max_val = calibrate_getvalue_per_tensor(activations)
    
    max_abs = torch.maximum(torch.abs(min_val), torch.abs(max_val))
    scale = max_abs / 127
    zero_point = 0
    
    return scale, zero_point

def asymmetric_calibrate_per_tensor(activations):
    min_val, max_val = calibrate_getvalue_per_tensor(activations)
    
    scale = (max_val - min_val) / 255
    zero_point = torch.round((-1.0 * min_val) / scale)
    
    return scale, zero_point

def symmetric_calibrate_per_channel(activations):
    min_val, max_val = calibrate_getvalue_per_channel(activations)
    
    max_abs = torch.maximum(torch.abs(min_val), torch.abs(max_val))
    scale = max_abs / 127
    zero_point = 0
    
    return scale, zero_point

def asymmetric_calibrate_per_channel(activations):
    min_val, max_val = calibrate_getvalue_per_channel(activations)
    
    scale = (max_val - min_val) / 255
    zero_point = torch.round((-1.0 * min_val) / scale)
    
    return scale, zero_point

def symmetric_quant_per_tensor(X, scale):
    q_X = torch.round(X / scale)
    q_X = torch.clamp(q_X, -127, 127)
    
    return q_X.to(torch.int8)

def symmetric_dequant_per_tensor(q_X, scale):
    return q_X.float() * scale

def asymmetric_quant_per_tensor(X, scale, zero_point,):
    q_X = torch.round(X / scale + zero_point)
    q_X = torch.clamp(q_X, 0, 255)
    
    return q_X.to(torch.uint8)

def asymmetric_dequant_per_tensor(q_X, scale, zero_point):
    return ((q_X - zero_point) * scale).float()

def symmetric_quant_per_channel(X, scale):
    q_X = torch.round(X / scale)
    q_X = torch.clamp(q_X, -127, 127)
    
    return q_X.to(torch.int8)

def symmetric_dequant_per_channel(q_X, scale):
    return q_X.float() * scale

def asymmetric_quant_per_channel(X, scale, zero_point):
    q_X = torch.round(X / scale + zero_point)
    q_X = torch.clamp(q_X, 0, 255)
    
    return q_X.to(torch.uint8)

def asymmetric_dequant_per_channel(q_X, scale, zero_point):
    return ((q_X - zero_point) * scale).float()

def get_mse(X, X_hat):
    return torch.mean((X - X_hat) ** 2)

def run_quant(X):
    activations = [X]
    
    # symmetric + per-tensor
    scale, zero_point = symmetric_calibrate_per_tensor(activations)
    q_X = symmetric_quant_per_tensor(X, scale)
    X_hat = symmetric_dequant_per_tensor(q_X, scale)
    
    mse_s_tensor = get_mse(X, X_hat)
    
    print("============ symmetric + per-tensor ============")
    print("mse:")
    print(mse_s_tensor)
    
    print("max_error:")
    print(torch.max(torch.abs(X - X_hat))) 
    
    print("scale:")
    print(scale)   
    
    assert torch.all(torch.abs(X - X_hat) <= scale / 2 + 1e-6) 
    
    # asymmetric + per-tensor
    scale, zero_point = asymmetric_calibrate_per_tensor(activations)
    q_X = asymmetric_quant_per_tensor(X, scale, zero_point)
    X_hat = asymmetric_dequant_per_tensor(q_X, scale, zero_point)
    
    mse_as_tensor = get_mse(X, X_hat)
    
    print("============ asymmetric + per-tensor ============")
    print("mse:")
    print(mse_as_tensor)
        
    print("max_error:")
    print(torch.max(torch.abs(X - X_hat)))    
    
    print("scale:")
    print(scale)  
    
    assert torch.all(torch.abs(X - X_hat) <= scale / 2 + 1e-6)
    
    # symmetric + per-channel
    scale, zero_point = symmetric_calibrate_per_channel(activations)
    q_X = symmetric_quant_per_channel(X, scale)
    X_hat = symmetric_dequant_per_channel(q_X, scale)
    
    mse_s_channel = get_mse(X, X_hat)
        
    print("============ symmetric + per-channel ============")
    print("mse:")
    print(mse_s_channel)
        
    print("max_error:")
    print(torch.max(torch.abs(X - X_hat)))    
    
    print("scale:")
    print(scale)  
    
    assert torch.all(torch.abs(X - X_hat) <= scale / 2 + 1e-6)
    
    # asymmetric + per-channel
    scale, zero_point = asymmetric_calibrate_per_channel(activations)
    q_X = asymmetric_quant_per_channel(X, scale, zero_point)
    X_hat = asymmetric_dequant_per_channel(q_X, scale, zero_point)
    
    mse_as_channel = get_mse(X, X_hat)
    
    print("============ asymmetric + per-channel ============")
    print("mse:")
    print(mse_as_channel)
            
    print("max_error:")
    print(torch.max(torch.abs(X - X_hat)))    
    
    print("scale:")
    print(scale)  
    
    assert torch.all(torch.abs(X - X_hat) <= scale / 2 + 1e-6)
    
    assert mse_as_tensor < mse_s_tensor
    assert mse_as_channel < mse_s_channel
    
    print("PASS")
    
if __name__ == "__main__":
    X = torch.rand(4096, 4096) * 2 + 3
    
    X[:, 0] *= 10
    X[:, 1] *= 20
    X[:, 2] *= 50
    X[:, 3] *= 100
    run_quant(X)
    