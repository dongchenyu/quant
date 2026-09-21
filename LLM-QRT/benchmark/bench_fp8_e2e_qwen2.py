import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from runtime_refact.core.api import AutoQuantForCausalLM

from runtime_refact.nn_models.modules.linear.linear_fp8 import print_fp8_profile

BF16_MODEL = "/root/LLMQRT-main/Qwen2.5-0.5B-Instruct"
FP8_MODEL = "/root/models/Qwen2.5-0.5B-Instruct-fp8-static"

PROMPT = "What is GPU quantization?"
MAX_NEW_TOKENS = 128

def load_bf16():
    print("\n========== LOAD BF16 ==========")
    model = AutoModelForCausalLM.from_pretrained(BF16_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL)
    
    return model, tokenizer

def load_fp8():
    print("\n========== LOAD FP8 ==========")
    model = AutoQuantForCausalLM.from_quantized(FP8_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", fuse_layers=False)
    
    tokenizer = AutoTokenizer.from_pretrained(FP8_MODEL)

    return model, tokenizer

@torch.no_grad()
def benchmark(model, tokenizer, name):
    print("\n================================")
    print(name)
    print("================================")
    
    device = next(model.parameters()).device
    
    inputs = tokenizer(PROMPT, return_tensors="pt")
    
    inputs = {
        k:v.to(device)
        for k,v in inputs.items()
    }
    
    # warmup
    model.generate(**inputs, max_new_tokens=16, do_sample=False)

    torch.cuda.synchronize()
    
    # Prefill / TTFT
    
    torch.cuda.synchronize()

    start = time.perf_counter()
    outputs = model(**inputs, use_cache=True)

    logits = outputs.logits[:, -1, :]

    first_token = torch.argmax(logits, dim=-1)

    torch.cuda.synchronize()

    ttft = (time.perf_counter()-start)*1000
    
    print(f"TTFT: {ttft:.3f} ms")
    
    past = outputs.past_key_values
    
    input_ids = first_token.unsqueeze(0)
    latencies=[]
    
    for _ in range(MAX_NEW_TOKENS):
        torch.cuda.synchronize()

        start = time.perf_counter()

        outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        next_token = torch.argmax(outputs.logits[:,-1,:], dim=-1)
        
        past = outputs.past_key_values

        input_ids = next_token.unsqueeze(0)

        torch.cuda.synchronize()

        latencies.append((time.perf_counter() - start) * 1000)
        
    avg = sum(latencies) / len(latencies)
    
    print(f"Decode latency: {avg:.3f} ms/token")
    print(f"Decode throughput: {1000/avg:.2f} tokens/s")
    
if __name__=="__main__":
    bf16_model,bf16_tok=load_bf16()
    benchmark(bf16_model, bf16_tok, "BF16")

    del bf16_model
    torch.cuda.empty_cache()

    fp8_model,fp8_tok=load_fp8()
    benchmark(fp8_model, fp8_tok, "FP8 Static + My Tensorwise")
    
    #### print_fp8_profile()