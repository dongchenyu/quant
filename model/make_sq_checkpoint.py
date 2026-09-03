import os
import json
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

SRC = "/root/models/Qwen2.5-0.5B-Instruct"
DST = "/root/models/Qwen2.5-0.5B-Instruct-sq"

# 1. 加载模型，只为了知道哪些 module 是 nn.Linear
print("[1] Loading model structure...")

model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu")

linear_names = {
    name
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
    and name != "lm_head"
}

print(f"Found {len(linear_names)} Linear modules")

for name in list(linear_names)[:10]:
    print(" ", name)
    
# 2. 拷贝 tokenizer/config 等文件
print("\n[2] Copying model directory...")

if os.path.exists(DST):
    shutil.rmtree(DST)

shutil.copytree(SRC, DST)

# 3. 读取原始 safetensors
src_weight = os.path.join(SRC, "model.safetensors")

print("\n[3] Loading original weights...")
state = load_file(src_weight)

new_state = {}

# 4. Linear weight:
# weight BF16/FP32 -> per-tensor absmax -> scale = absmax / 127 -> qweight INT8

# LLMQRT SqLinear 需要：
# qweight
# weight_scale
# input_scale

print("\n[4] Quantizing Linear weights...")

quantized_count = 0

for key, tensor in state.items():
    if key.endswith(".weight"):
        module_name = key[:-len(".weight")]

        if module_name in linear_names:
            w = tensor.float()

            absmax = w.abs().max()
            scale = absmax / 127.0

            if scale == 0:
                scale = torch.tensor(
                    1.0,
                    dtype=torch.float32
                )

            # 注意：下面这些必须在 if scale == 0 外面
            qweight = torch.round(w / scale)
            qweight = torch.clamp(qweight, -127, 127).to(torch.int8)

            new_state[module_name + ".qweight"] = qweight

            new_state[module_name + ".weight_scale"] = (scale.to(torch.float32))

            new_state[module_name + ".input_scale"] = torch.tensor(1.0,dtype=torch.float32)

            quantized_count += 1

            if quantized_count <= 5:
                print()
                print(module_name)
                print(" weight :", tuple(tensor.shape), tensor.dtype)
                print(" qweight:", tuple(qweight.shape), qweight.dtype)
                print(" scale  :",scale.item())

            continue

    new_state[key] = tensor
    
print(f"\nQuantized Linear count = {quantized_count}")

# 5. 保存新的 safetensors
dst_weight = os.path.join(DST, "model.safetensors")

print("\n[5] Saving quantized checkpoint...")

save_file(new_state, dst_weight)

# 6. 修改 config.json
config_path = os.path.join(DST, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)
    
config["quantization_config"] = {
    "quant_method": "sq",
    "zero_point": False,
    "group_size": 0,
    "bits": 8,
    "fp8_static_quant": False,
    "kv_cache_quant_layers": [],
    "modules_to_not_convert": ["lm_head"],
    "per_tensor": True
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    
print("\nDONE")
print("Output:", DST)
