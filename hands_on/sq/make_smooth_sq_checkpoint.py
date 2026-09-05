import os
import json
import shutil

import torch

from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SRC = "/root/models/Qwen2.5-0.5B-Instruct"

DST = (
    "/root/models/"
    "Qwen2.5-0.5B-Instruct-smooth-sq"
)

ACT_SCALE_PATH = (
    "/root/models/qwen2_act_scales.pt"
)

ALPHA = 0.85

print("========== LOAD MODEL ==========")
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu")

model.eval()

print("========== LOAD ACT SCALES ==========")
act_scales = torch.load(ACT_SCALE_PATH, map_location="cpu")

print("activation scale entries: ", len(act_scales))

# Smooth one: RMSNorm -> [Linear, Linear, ...]
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scale, alpha):
    if not isinstance(fcs, list):
        fcs = [fcs]
    act_scale = (act_scale.float().clamp(min=1e-5))

    # 对每一个 input channel： max_o |W[o, k]|
    # 多个 FC 时再取最大值
    
    weight_scales = []
    
    for fc in fcs:
        w_scale = (fc.weight.detach().float().abs().amax(dim=0))
        weight_scales.append(w_scale)
        
    weight_scale = (torch.stack(weight_scales, dim=0).amax(dim=0).clamp(min=1e-5))
    
    # SmoothQuant 
    # s = act^alpha / weight^(1-alpha)
    
    scales = act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)
    scales = scales.clamp(min=1e-5)
    
    scales = scales.to(dtype=ln.weight.dtype, device=ln.weight.device)
    
    # X' = X / s, RMSNorm.weight /= s 离线吸收
    
    ln.weight.div_(scales)
    
    # W' = W * s

    for fc in fcs:
        fc.weight.mul_(scales.reshape(1, -1))
        
    return scales.float()

# Apply smoothing
print("========== SMOOTH ==========")
all_smooth_scales = {}

for i, layer in enumerate(model.model.layers):
    # Attention: input_layernorm -> (q / k / v)
    q_name = (f"model.layers.{i}.self_attn.q_proj")
    
    q_scale = act_scales[q_name]
    
    s_attn = smooth_ln_fcs(layer.input_layernorm,
                           [layer.self_attn.q_proj,
                            layer.self_attn.k_proj,
                            layer.self_attn.v_proj], q_scale, ALPHA)
    
    all_smooth_scales[f"layer_{i}.attn"] = s_attn

    # MLP: post_attention_layernorm -> gate / up
    gate_name = (f"model.layers.{i}.mlp.gate_proj")
    gate_scale = act_scales[gate_name]
    
    s_mlp = smooth_ln_fcs(layer.post_attention_layernorm,
                          [layer.mlp.gate_proj, layer.mlp.up_proj], gate_scale, ALPHA)
    
    all_smooth_scales[f"layer_{i}.mlp"] = s_mlp
    
    if i < 3:
        print()
        print("layer:", i)

        print("attn smooth scale:")

        print(" min:", s_attn.min().item(), " mean:", s_attn.mean().item(), " max:", s_attn.max().item())

        print("mlp smooth scale:")

        print(" min:", s_mlp.min().item(), " mean:", s_mlp.mean().item(), " max:", s_mlp.max().item())
    
# Quantize all Linear weights
print()
print("========== QUANTIZE ==========")
linear_names = {
    name
    for name, module
    in model.named_modules()
    if (isinstance(module,torch.nn.Linear) and name != "lm_head")
}

state = model.state_dict()
new_state = {}
quantized_count = 0

for key, tensor in state.items():
    if key.endswith(".weight"):
        module_name = key[:-len(".weight")]

        if module_name in linear_names:

            w = tensor.float()
            absmax = (w.abs().max())

            scale = (absmax / 127.0)

            if scale.item() == 0:
                scale = torch.tensor(1.0, dtype=torch.float32)

            qweight = torch.round(w / scale)

            qweight = torch.clamp(qweight, -127, 127).to(torch.int8)

            new_state[module_name + ".qweight"] = (
                qweight.cpu().contiguous()
            )

            new_state[module_name + ".weight_scale"] = (
                scale.float().cpu())

            # 当前 LLMQRT SQ dynamic activation
            # forward 没使用 checkpoint
            # input_scale。
            new_state[module_name + ".input_scale"] = torch.tensor(
                1.0, dtype=torch.float32,)

            quantized_count += 1
            continue

    new_state[key] = (tensor.detach().cpu().contiguous().clone())

print("Quantized Linear count:", quantized_count)

# Copy model files
if os.path.exists(DST):
    shutil.rmtree(DST)

shutil.copytree(SRC, DST)

# Save quantized weights
dst_weight = os.path.join(DST, "model.safetensors")

save_file(new_state, dst_weight, metadata={"format": "pt"})

# ============================================================
# Add LLMQRT quantization config
# ============================================================

config_path = os.path.join(DST, "config.json")

with open(config_path, "r", encoding="utf-8",) as f:
    config = json.load(f)

config[
    "quantization_config"
] = {
    "quant_method": "sq",
    "zero_point": False,
    "group_size": 0,
    "bits": 8,
    "fp8_static_quant": False,
    "kv_cache_quant_layers": [],
    "modules_to_not_convert": [
        "lm_head"
    ],
    "per_tensor": True,
}


with open(config_path, "w", encoding="utf-8",
) as f:
    json.dump(config, f, indent=2, ensure_ascii=False,)


# Optional:
# 保存 smoothing scale 方便分析
torch.save(all_smooth_scales, os.path.join(DST, "smooth_scales.pt"))


print()
print("========== DONE ==========")

print("alpha:", ALPHA)

print("Output:", DST)