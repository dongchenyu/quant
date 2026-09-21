import torch
import time

def benchmark_quant():
    device="cuda"

    shapes=[
        (1,896),
        (16,896),
        (128,896),
        (128,4864),
        (1,4864),
    ]

    scale = torch.tensor(0.01, device=device, dtype=torch.float32)
    
    for shape in shapes:
        x = torch.randn(shape, device=device, dtype=torch.bfloat16)

        # warmup
        for _ in range(20):
            q = (x / scale).to(torch.float8_e4m3fn)

        torch.cuda.synchronize()

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        start.record()

        for _ in range(1000):
            q = (x / scale).to(torch.float8_e4m3fn)

        end.record()

        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / 1000

        print(shape, f"{ms:.4f} ms", f"{ms*1000/1000:.3f} us")
        
if __name__=="__main__":
    benchmark_quant()