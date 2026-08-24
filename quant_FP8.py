import torch

fp8_dtype = torch.float8_e4m3fn

x1 = torch.tensor([1.00, 1.03, 1.06, 1.10, 1.13, 1.18, 1.24, 1.30, 1.37, 1.44, 1.52],dtype=torch.float32)

q = x1.to(fp8_dtype)

print("FP32:")
print(x1)

print("\ncast FP8 -> back FP32:")
print(q.float())

x = torch.tensor([0.12, 0.26, 1.17, 3.73, 10.8, 75.0, 300.0,], dtype=torch.float32)

amax = x.abs().max()

fp8_max = torch.finfo(fp8_dtype).max

print("amax = ", amax)
print("FP8 max = ", fp8_max)

scale = amax / fp8_max

print("scale =", scale)

x_scaled = x / scale

print("\nx_scaled:")
print(x_scaled)

x_fp8 = x_scaled.to(fp8_dtype)

print("\nx_fp8:")
print(x_fp8)

x_fp8_grid = x_fp8.float()

print("\nFP8 grid values:")
print(x_fp8_grid)

x_hat = x_fp8.float() * scale

print("\nOriginal:")
print(x)

print("\nReconstructed:")
print(x_hat)

print("\nError:")
print(x - x_hat)

# -----------------------------
# 6. 打印
# -----------------------------
print(f"amax     = {amax.item()}")
print(f"fp8_max  = {fp8_max}")
print(f"scale    = {scale.item()}")
print()

print(
    f"{'Original':>12} "
    f"{'After scale':>15} "
    f"{'FP8 quant':>15} "
    f"{'Dequant':>15}"
)

print("-" * 62)

for original, scaled, quant, dequant in zip(
    x, x_scaled, x_fp8_grid, x_hat
):
    print(
        f"{original.item():12.6f} "
        f"{scaled.item():15.6f} "
        f"{quant.item():15.6f} "
        f"{dequant.item():15.6f}"
    )