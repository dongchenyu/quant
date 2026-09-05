import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil

import torch

from safetensors.torch import (load_file, save_file)

from transformers import AutoModelForCausalLM

SRC = Path("/root/models/Qwen2.5-0.5B-Instruct")
DST = Path("/root/models/Qwen2.5-0.5B-Instruct-fp8-static")

INPUT_SCALE_PATH = Path("/root/models/qwen2_fp8_static_input_scales.pt")

FP8 = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8)

print("========== LOAD MODEL STRUCTURE ==========")

model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu")

linear_names = {
    name
    for name, module in model.named_modules()
    if(isinstance(module, torch.nn.Linear) and name != "lm_head")
}

print("Linear modules:", len(linear_names))

print("========== LOAD INPUT SCALES ==========")

input_scales = torch.load(INPUT_SCALE_PATH, map_location="cpu")

print("Input scales:", len(input_scales))

assert (len(input_scales) == len(linear_names))

print("========== LOAD ORIGINAL CHECKPOINT ==========")

state = load_file(str(SRC / "model.safetensors"))

new_state = {}

quantized_count = 0

print("========== FP8 WEIGHT QUANT ==========")
for key, tensor in state.items():
    if key.endswith(".weight"):
        module_name = key[:-len(".weight")]
        
        if module_name in linear_names:
            w = tensor.float()
            
            # per-tensor weight scale
            amax = w.abs().max()
            weight_scale = (amax.clamp(min=1e-12) / FP8_INFO.max)
            
            qweight_fp8 = (w / weight_scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8)
            qweight_storage = qweight_fp8.to(torch.bfloat16).cpu().contiguous()
            
            new_state[key] = qweight_storage
            new_state[module_name + ".weight_scale"] = weight_scale.float().reshape(1).cpu()
            new_state[module_name + ".input_scale"] = input_scales[module_name].float().reshape(1).cpu()
            new_state[module_name + ".output_scale"] = torch.ones(1, dtype=torch.float32)
            
            quantized_count += 1
            
            if quantized_count <= 5:
                print()
                print(module_name)

                print("weight:", tuple(tensor.shape))
                print("weight_scale:", weight_scale.item())
                print("input_scale:", input_scales[module_name].item())
                
            continue
        
    new_state[key] = tensor.detach().cpu().contiguous().clone()
    
print()
print("Quantized Linear count:", quantized_count)

assert (quantized_count == len(linear_names))

print("========== COPY MODEL FILES ==========")

if DST.exists():
    shutil.rmtree(DST)
    
shutil.copytree(SRC, DST)

print("========== SAVE CHECKPOINT ==========")

save_file(new_state, str(DST / "model.safetensors"), metadata={"format": "pt"})

print("========== WRITE CONFIG ==========")

config_path = (DST / "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

config["quantization_config"] = {
    "quant_method": "fp8_static_quant",
    "zero_point": False,
    "group_size": 0,
    "bits": 8,
    "fp8_static_quant": True,
    "kv_cache_quant_layers": [],
    "modules_to_not_convert": [
        "lm_head"
    ],
    "per_tensor": True,
}
            
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    
print()
print("========== DONE ==========")
print("Output:", DST)

