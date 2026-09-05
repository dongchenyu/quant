import torch

from runtime_refact.core.api import AutoQuantForCausalLM

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct-sq"
MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct-smooth-sq"

print("========== LOAD MODEL ==========")

model = AutoQuantForCausalLM.from_quantized(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    fuse_layers=False
)

print("\n========== BASIC INFO ==========")

print("is_quantized:", model.is_quantized)
print("model_type:", model.model_type)
print("quant_method:", model.quant_config.quant_method)
print("per_tensor:", model.quant_config.per_tensor)

print("\n========== CHECK Q_PROJ ==========")

q_proj = model.model.model.layers[0].self_attn.q_proj

print(q_proj)
print("class: ", type(q_proj))

print("qweight:")
print("shape = ", q_proj.qweight.shape)
print("dtype = ", q_proj.qweight.dtype)

print("weight_scale:")
print("shape = ", q_proj.weight_scale.shape)
print("dtype = ", q_proj.weight_scale.dtype)
print("value = ", q_proj.weight_scale)

print("input_scale:")
print("shape = ", q_proj.input_scale.shape)
print("dtype = ", q_proj.input_scale.dtype)

print("\nPASS")