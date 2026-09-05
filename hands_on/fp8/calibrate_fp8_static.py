import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_PATH = Path("/root/models/Qwen2.5-0.5B-Instruct")

OUTPUT_PATH = Path("/root/models/qwen2_fp8_static_input_scales.pt")

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max

texts = [
    "What is GPU quantization and why is it useful?",
    "Explain CUDA thread blocks and warps.",
    "What is the difference between CPU and GPU computing?",
    "Explain matrix multiplication.",
    "What is machine learning?",
    "Explain transformer attention.",
    "What is a tensor core?",
    "How does INT8 quantization work?",
    "Explain floating point arithmetic.",
    "What is model inference?",
    "Explain training and inference.",
    "What is a neural network?",
    "What is a large language model?",
    "What is memory bandwidth?",
    "Explain GPU cache hierarchy.",
    "What is parallel computing?",
    "Why are GPUs useful for deep learning?",
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
    "What is FP8 quantization?",
    "Describe the CUDA programming model.",
    "Explain GEMM optimization.",
    "人工智能模型量化是什么？",
    "解释一下GPU为什么适合矩阵计算。",
    "什么是大语言模型推理？",
    "解释一下FP8量化。",
    "什么是CUDA线程块?",
    "解释Transformer中的注意力机制。",
    "什么是模型推理中的KV Cache?",
    "为什么量化可以提高模型推理性能？",
]

print("========== LOAD BF16 MODEL ==========")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.eval()

input_amax = {}

def make_hook(name):
    def hook(module, inputs, output):
        x = inputs[0].detach()
        current_amax = (x.float().abs().max().cpu())
        
        if name not in input_amax:
            input_amax[name] = current_amax
        else:
            input_amax[name] = torch.maximum(input_amax[name], current_amax)
            
    return hook 

print("========== REGISTER HOOKS ==========")

handles = []
linear_count = 0

for name, module in model.named_modules():
    if(isinstance(module, torch.nn.Linear) and name != "lm_head"):
        handles.append(module.register_forward_hook(make_hook(name)))
        linear_count += 1

print("Linear modules:", linear_count)

print("========== CALIBRATION ==========")
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
    
print("========== BUILD INPUT SCALES ==========")
input_scales = {}

for name, amax in input_amax.items():
    scale = (amax.clamp(min=1e-12) / FP8_MAX).float()
    input_scales[name] = scale.reshape(1)
    
print("Collected input scales:", len(input_scales))

assert len(input_scales) == linear_count

for name in list(input_scales.keys())[:8]:
    print(name, "amax =", input_amax[name].item(), "scale =", input_scales[name].item())
    
torch.save(input_scales, OUTPUT_PATH)


print()
print("Saved to:", OUTPUT_PATH)
print("PASS")
    







