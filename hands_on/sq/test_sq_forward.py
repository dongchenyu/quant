import torch

from transformers import AutoTokenizer
from runtime_refact.core.api import AutoQuantForCausalLM

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct-sq"
#MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct-smooth-sq"

print("========== LOAD ==========")

model = AutoQuantForCausalLM.from_quantized(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", fuse_layers=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("\n========== TOKENIZE ==========")

text = "Hello, what is GPU quantization?"

inputs = tokenizer(text, return_tensors="pt")

device = next(model.model.parameters()).device

inputs = {
    k:v.to(device)
    for k,v in inputs.items()    
}

print("input_ids shape:", inputs["input_ids"].shape)
print("device:", device)

print("\n========== FORWARD ==========")
with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)

print("\n========== OUTPUT ==========")

print("logits shape:", outputs.logits.shape)
print("logits dtype:", outputs.logits.dtype)

print("finite:", torch.isfinite(outputs.logits).all().item())
print("PASS")

