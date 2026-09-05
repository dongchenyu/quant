import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "/root/models/qwen2_act_scales.pt"

# Calibration texts

texts = [
    "What is GPU quantization and why is it useful?",
    "Explain CUDA thread blocks and warps.",
    "What is the difference between CPU and GPU computing?",
    "Explain matrix multiplication in simple terms.",
    "What is machine learning?",
    "Explain transformer attention.",
    "What is a tensor core?",
    "How does INT8 quantization work?",
    "Explain floating point arithmetic.",
    "What is model inference?",
    "Explain the difference between training and inference.",
    "What is a neural network?",
    "Write a short explanation of large language models.",
    "What is memory bandwidth?",
    "Explain cache hierarchy in modern processors.",
    "What is parallel computing?",
    "Why are GPUs suitable for deep learning?",
    "Explain softmax.",
    "What is layer normalization?",
    "Explain RMSNorm.",
    "What is positional encoding?",
    "Explain rotary positional embedding.",
    "What is KV cache?",
    "What is FlashAttention?",
    "Explain post-training quantization.",
    "What is calibration in model quantization?",
    "Explain symmetric quantization.",
    "What is per-channel quantization?",
    "What is per-tensor quantization?",
    "What is SmoothQuant?",
    "Describe the CUDA programming model.",
    "Explain GEMM optimization.",
    "人工智能模型量化是什么？",
    "解释一下GPU为什么适合矩阵计算。",
    "什么是大语言模型推理？",
    "解释一下INT8量化。",
    "什么是CUDA线程块?",
    "解释Transformer中的注意力机制.",
    "什么是模型推理中的KV Cache?",
    "为什么量化可以提高模型推理性能?",
]

print("Loading BF16 model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.eval()

# name -> [hidden_size]
# 保存每一个 input channel 的最大 abs value

act_scales = {}

def make_hook(name):
    def hook(module, inputs, output):
        # [B, S, K] -> [B * S, k]
        x = inputs[0]
        x = x.detach().float().reshape(-1, x.shape[-1])
        
        # result: [K]
        current = x.abs().amax(dim=0).cpu()
        
        if name not in act_scales:
            act_scales[name] = current
        else:
            act_scales[name] = torch.maximum(act_scales[name], current)
            
    return hook
        
# register hooks
# q/k/v 共享一个输入，所以只观察 q_proj 就够。
# gate/up 共享一个输入，所以只观察 gate_proj 就够。

handles = []

for i, layer in enumerate(model.model.layers):
    q_name = (f"model.layers.{i}.self_attn.q_proj")
    gate_name = (f"model.layers.{i}.mlp.gate_proj")
    
    handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(q_name)))
    handles.append(layer.mlp.gate_proj.register_forward_hook(make_hook(gate_name)))

# Run calibration
print(f"Running calibration with {len(texts)} samples...")

device = next(model.parameters()).device

with torch.inference_mode():
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device)
                  for k, v in inputs.items()}
        model(**inputs, use_cache=False)
        
        if (i + 1) % 5 == 0:
            print(f"calibration {i + 1}/{len(texts)}")
            
for h in handles:
    h.remove()
    
# Save
torch.save(act_scales, OUTPUT_PATH)

print()
print("Collected activation scales:", len(act_scales))
print("Saved to:", OUTPUT_PATH)

# Show first few
for name in list(act_scales.keys())[:5]:
    x = act_scales[name]
    
    print()
    print(name)
    print("shape:", x.shape)

    print("min channel absmax:", x.min().item())
    print("mean channel absmax:", x.mean().item())
    print("max channel absmax:", x.max().item())
    print("max / mean:", (x.max() / x.mean()).item())


    