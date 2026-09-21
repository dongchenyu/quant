LOADED MY linear_fp8.py

### BF16

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
TTFT: 38.305 ms
Decode latency: 33.543 ms/token
Decode throughput: 29.81 tokens/s

### FP8 

Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
using my fp8 tensorwise kernel
TTFT: 61.699 ms
Decode latency: 60.487 ms/token
Decode throughput: 16.53 tokens/s

似乎是收益没有抵消量化的开销？