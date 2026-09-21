上个版本分析了之后发现当前性能kernel在于block数量太少以及每个block太重了，导致很多SM没有被充分的利用，所以这次优先调整了ThreadblockShape，WarpShape，把这两个所表示的每个block计算的tile调整的小一点。然后进行对比。
```
调整前:
ThreadblockShape = <128, 128, 64>
WarpShape        = <64, 64, 64>
InstructionShape = <16, 8, 32>

调整后:
ThreadblockShape=<64, 64, 64>
WarpShape=<32, 64, 64>
InstructionShape=<16, 8, 32>
```

结果如下:
## Benchmark Results

| Layer | M | N | K | Kernel | Latency(us) | TFLOPS | Speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| attn_q_o | 1 | 896 | 896 | BF16 cuBLAS | 23.491 | 0.068 | 1.000x |
| attn_q_o | 1 | 896 | 896 | FP8 Tensor | 9.820 | 0.164 | 2.392x |
| attn_q_o | 1 | 896 | 896 | FP8 Row | 15.923 | 0.101 | 1.475x |
| attn_q_o | 16 | 896 | 896 | BF16 cuBLAS | 24.209 | 1.061 | 1.000x |
| attn_q_o | 16 | 896 | 896 | FP8 Tensor | 8.479 | 3.030 | 2.855x |
| attn_q_o | 16 | 896 | 896 | FP8 Row | 16.851 | 1.525 | 1.437x |
| attn_q_o | 128 | 896 | 896 | BF16 cuBLAS | 31.272 | 6.572 | 1.000x |
| attn_q_o | 128 | 896 | 896 | FP8 Tensor | 8.579 | 23.956 | 3.645x |
| attn_q_o | 128 | 896 | 896 | FP8 Row | 17.532 | 11.723 | 1.784x |
| mlp_up_gate | 1 | 4864 | 896 | BF16 cuBLAS | 22.593 | 0.386 | 1.000x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Tensor | 9.318 | 0.935 | 2.425x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Row | 17.352 | 0.502 | 1.302x |
| mlp_up_gate | 16 | 4864 | 896 | BF16 cuBLAS | 24.515 | 5.689 | 1.000x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Tensor | 9.160 | 15.225 | 2.676x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Row | 16.963 | 8.222 | 1.445x |
| mlp_up_gate | 128 | 4864 | 896 | BF16 cuBLAS | 29.844 | 37.383 | 1.000x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Tensor | 9.421 | 118.428 | 3.168x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Row | 23.117 | 48.263 | 1.291x |
| mlp_down | 1 | 896 | 4864 | BF16 cuBLAS | 22.128 | 0.394 | 1.000x |
| mlp_down | 1 | 896 | 4864 | FP8 Tensor | 34.693 | 0.251 | 0.638x |
| mlp_down | 1 | 896 | 4864 | FP8 Row | 20.326 | 0.429 | 1.089x |
| mlp_down | 16 | 896 | 4864 | BF16 cuBLAS | 30.684 | 4.545 | 1.000x |
| mlp_down | 16 | 896 | 4864 | FP8 Tensor | 34.632 | 4.027 | 0.886x |
| mlp_down | 16 | 896 | 4864 | FP8 Row | 20.019 | 6.966 | 1.533x |
| mlp_down | 128 | 896 | 4864 | BF16 cuBLAS | 28.022 | 39.815 | 1.000x |
| mlp_down | 128 | 896 | 4864 | FP8 Tensor | 34.794 | 32.065 | 0.805x |
| mlp_down | 128 | 896 | 4864 | FP8 Row | 18.775 | 59.424 | 1.493x |

可以看到从latency中可以看到tensorwise有了十分明显的提升。

然后将ncu中的一些指标做一些对比，当前修改之后的版本叫V2，之前没修改的版本叫V1.

首先是block数量相比于V1明显翻了4倍，和我们的调整预期是对得上的，以及每个block占用的寄存器与shared memory都减少了
```
V2:
Grid Size = 28
Registers/thread = 212
Dynamic Shared Memory = 24.58 KB

V1:
Grid Size = 7
Registers/thread = 255
Dynamic Shared Memory = 49.15 KB
```

接下来找M=128的例子, 可以看到这几个M=128的例子相比于V1都提升了将近4倍，而我们所做的就是将每个block处理的tile的shape降为原来的1/4，即将block数量提升为原来的4倍。
```
attn_q_o:
V1: 32.998 us 6.228 TFLOPS
V2: 8.579 us 23.956 TFLOPS

mlp_up_gate:
V1: 37.627 us 29.651 TFLOPS
V2: 9.421 us 118.428 TFLOPS

mlp_down:
V1: 130.391 us 8.556 TFLOPS
V2: 34.794 us 32.065 TFLOPS
```

当然occupancy还是低，甚至比V1版本更低，不过关于性能调优，在大量SM的算力没有被打满之前，还是优先处理算力利用率低的问题，occupancy是warp被阻塞的时候再考虑对这方面进行优化。