import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from runtime_refact.core.api import AutoQuantForCausalLM

MODEL_PATH = Path("/root/models/Qwen2.5-0.5B-Instruct-fp8-static")

print("========== LOAD ==========")

model = AutoQuantForCausalLM.from_quantized(str(MODEL_PATH), torch_dtype=torch.bfloat16,
                                            device_map="auto", fuse_layers=False)

print("\n========== INFO ==========")
print("quant_method:", model.quant_config.quant_method)
print("per_tensor:", model.quant_config.per_tensor)

q_proj = (model.model.model.layers[0].self_attn.q_proj)

print("\n========== Q_PROJ ==========")
print(q_proj)
print("class:", type(q_proj))
print("weight:", q_proj.weight.shape, q_proj.weight.dtype)
print("weight_scale:", q_proj.weight_scale.shape, q_proj.weight_scale.dtype, q_proj.weight_scale)

print("\nPASS")