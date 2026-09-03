import torch

from transformers import AutoTokenizer
from runtime_refact.core.api import AutoQuantForCausalLM

MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct-sq"

print("========== LOAD ==========")

model = AutoQuantForCausalLM.from_quantized(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", fuse_layers=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("\n========== PROMPT ==========")

prompt = "What is GPU quantization?"

inputs = tokenizer(prompt, return_tensors="pt")

device = next(model.model.parameters()).device

inputs = {
    k: v.to(device)
    for k, v in inputs.items()    
}

print("prompt:", prompt)
print("input shape:", inputs["input_ids"].shape)

print("\n========== GENERATE ==========")

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    
print("\n========== OUTPUT ==========")

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(text)

print("\nPASS")