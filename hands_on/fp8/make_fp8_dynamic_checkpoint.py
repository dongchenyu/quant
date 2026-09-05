import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

SRC = Path("/root/models/Qwen2.5-0.5B-Instruct")
DST = Path("/root/models/Qwen2.5-0.5B-Instruct-fp8-dynamic")

FP8 = torch.float8_e4m3fn

print("========== LOAD MODEL STRUCTURE ==========")
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu")

linear_names = {
    name
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
    and name != "lm_head"
}

print("Linear modules:", len(linear_names))

'''
print(type(model))
print(model.__class__.__name__)
print(model.config.model_type)
print(model)
'''

print("\n========== LOAD CHECKPOINT ==========")
state = load_file(str(SRC / "model.safetensors"))

for i, k in enumerate(state.keys()):
    print(k)
    if i >= 20:
        break
 
new_state = {}
quantized_count = 0
finfo = torch.finfo(FP8)

print("\n========== FP8 WEIGHT QUANT ==========")
# model.layers.0.self_attn.q_proj.weight
for key, tensor in state.items():
    if key.endswith(".weight"):
        module_name = key[:-len(".weight")]
        if module_name in linear_names:
            w = tensor.float()
            
            # per-tensor FP8 weight scale 
            # W_fp8 = round_fp8(W / scale)
            amax = w.abs().max()
            scale = (amax.clamp(min=1e-12) / finfo.max)
            
            qweight_fp8 = (w / scale).clamp(min=finfo.min, max=finfo.max).to(FP8)
            
            # FP8DynamicLinear.weight 是 BF16 参数。
            # 因此保存 FP8 round 后的数值，再转换回 BF16, forward 时会再次 .to(FP8)。
            qweight_storage = (qweight_fp8.to(torch.bfloat16).cpu().contiguous())
            new_state[key] = qweight_storage
            
            new_state[module_name + ".weight_scale"] = scale.float().reshape(1).cpu()
            
            quantized_count += 1
            
            if quantized_count <= 5:
                print()
                print(module_name)
                
                print("original:", tuple(tensor.shape), tensor.dtype)
                print("FP8 storage:", tuple(qweight_storage.shape), qweight_storage.dtype)
                print("weight scale:", scale.item())
                
            continue
        
    new_state[key] = (tensor.detach().cpu().contiguous().clone())
    
print()
print("new_state:")
for i, k in enumerate(new_state.keys()):
    print(k)
    if i >= 20:
        break
    
print()
print("Quantized Linear count:", quantized_count)

assert quantized_count == len(linear_names)

print("\n========== COPY MODEL FILES ==========")

if DST.exists():
    shutil.rmtree(DST)
    
shutil.copytree(SRC, DST)
        
print("\n========== SAVE CHECKPOINT ==========")
save_file(new_state, str(DST / "model.safetensors"), metadata={"format": "pt"})

print("\n========== WRITE CONFIG ==========")
config_path = DST / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
    
config["quantization_config"] = {
    "quant_method": "fp8_dynamic_quant",
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

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
        
print()
print("========== DONE ==========")
print("Output:", DST)
            
'''
root@autodl-container-c25d449539-f42d3fe8:~/LLMQRT-main/hands_on/fp8# python make_fp8_dynamic_checkpoint.py
========== LOAD MODEL STRUCTURE ==========
Linear modules: 168
<class 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'>
Qwen2ForCausalLM
qwen2
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 896)
    (layers): ModuleList(
      (0-23): 24 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=896, out_features=896, bias=True)
          (k_proj): Linear(in_features=896, out_features=128, bias=True)
          (v_proj): Linear(in_features=896, out_features=128, bias=True)
          (o_proj): Linear(in_features=896, out_features=896, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
          (up_proj): Linear(in_features=896, out_features=4864, bias=False)
          (down_proj): Linear(in_features=4864, out_features=896, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((896,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=896, out_features=151936, bias=False)
)

model.embed_tokens.weight torch.Size([151936, 896]) torch.bfloat16
model.layers.0.input_layernorm.weight torch.Size([896]) torch.bfloat16
model.layers.0.mlp.down_proj.weight torch.Size([896, 4864]) torch.bfloat16
model.layers.0.mlp.gate_proj.weight torch.Size([4864, 896]) torch.bfloat16
model.layers.0.mlp.up_proj.weight torch.Size([4864, 896]) torch.bfloat16
model.layers.0.post_attention_layernorm.weight torch.Size([896]) torch.bfloat16
model.layers.0.self_attn.k_proj.bias torch.Size([128]) torch.bfloat16
model.layers.0.self_attn.k_proj.weight torch.Size([128, 896]) torch.bfloat16
model.layers.0.self_attn.o_proj.weight torch.Size([896, 896]) torch.bfloat16
model.layers.0.self_attn.q_proj.bias torch.Size([896]) torch.bfloat16
model.layers.0.self_attn.q_proj.weight torch.Size([896, 896]) torch.bfloat16
model.layers.0.self_attn.v_proj.bias torch.Size([128]) torch.bfloat16
model.layers.0.self_attn.v_proj.weight torch.Size([128, 896]) torch.bfloat16


model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.self_attn.k_proj.bias
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.v_proj.bias
model.layers.0.self_attn.v_proj.weight

new_state:
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.down_proj.weight_scale
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.gate_proj.weight_scale
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.up_proj.weight_scale
model.layers.0.post_attention_layernorm.weight
model.layers.0.self_attn.k_proj.bias
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.k_proj.weight_scale
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.o_proj.weight_scale
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.weight_scale
model.layers.0.self_attn.v_proj.bias
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.v_proj.weight_scale
model.layers.1.input_layernorm.weight

root@autodl-container-c25d449539-f42d3fe8:~/models/Qwen2.5-0.5B-Instruct-fp8-dynamic# cat config.json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  ...
  ...
  "quantization_config": {
    "quant_method": "fp8_dynamic_quant",
    "zero_point": false,
    "group_size": 0,
    "bits": 8,
    "fp8_static_quant": false,
    "kv_cache_quant_layers": [],
    "modules_to_not_convert": [
      "lm_head"
    ],
    "per_tensor": true
  }

'''
            

