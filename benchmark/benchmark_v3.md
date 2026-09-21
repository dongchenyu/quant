在V2的基础上继续改进，将shape进一步缩小到原来的1/2，由64 * 64 = 4096变成32 * 64 = 2048，同时K tile扩大一倍由64扩充到128，
```
调整前:
ThreadblockShape=<64, 64, 64>
WarpShape=<32, 64, 64>
InstructionShape=<16, 8, 32>

调整后:
ThreadblockShape=<32, 64, 128>
WarpShape=<16, 64, 64>
InstructionShape=<16, 8, 32>
```

结果如下:
## Benchmark Results

| Layer | M | N | K | Kernel | Latency(us) | TFLOPS | Speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| attn_q_o | 1 | 896 | 896 | BF16 cuBLAS | 22.636 | 0.071 | 1.000x |
| attn_q_o | 1 | 896 | 896 | FP8 Tensor | 8.237 | 0.195 | 2.748x |
| attn_q_o | 1 | 896 | 896 | FP8 Row | 15.483 | 0.104 | 1.462x |
| attn_q_o | 16 | 896 | 896 | BF16 cuBLAS | 23.121 | 1.111 | 1.000x |
| attn_q_o | 16 | 896 | 896 | FP8 Tensor | 7.956 | 3.229 | 2.906x |
| attn_q_o | 16 | 896 | 896 | FP8 Row | 15.597 | 1.647 | 1.482x |
| attn_q_o | 128 | 896 | 896 | BF16 cuBLAS | 29.280 | 7.019 | 1.000x |
| attn_q_o | 128 | 896 | 896 | FP8 Tensor | 8.602 | 23.893 | 3.404x |
| attn_q_o | 128 | 896 | 896 | FP8 Row | 15.981 | 12.861 | 1.832x |
| mlp_up_gate | 1 | 4864 | 896 | BF16 cuBLAS | 22.232 | 0.392 | 1.000x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Tensor | 7.675 | 1.136 | 2.897x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Row | 16.108 | 0.541 | 1.380x |
| mlp_up_gate | 16 | 4864 | 896 | BF16 cuBLAS | 23.008 | 6.061 | 1.000x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Tensor | 8.243 | 16.918 | 2.791x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Row | 16.522 | 8.441 | 1.393x |
| mlp_up_gate | 128 | 4864 | 896 | BF16 cuBLAS | 27.677 | 40.310 | 1.000x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Tensor | 11.776 | 94.742 | 2.350x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Row | 23.229 | 48.029 | 1.191x |
| mlp_down | 1 | 896 | 4864 | BF16 cuBLAS | 21.340 | 0.408 | 1.000x |
| mlp_down | 1 | 896 | 4864 | FP8 Tensor | 10.766 | 0.810 | 1.982x |
| mlp_down | 1 | 896 | 4864 | FP8 Row | 20.541 | 0.424 | 1.039x |
| mlp_down | 16 | 896 | 4864 | BF16 cuBLAS | 28.093 | 4.964 | 1.000x |
| mlp_down | 16 | 896 | 4864 | FP8 Tensor | 10.747 | 12.977 | 2.614x |
| mlp_down | 16 | 896 | 4864 | FP8 Row | 19.027 | 7.330 | 1.477x |
| mlp_down | 128 | 896 | 4864 | BF16 cuBLAS | 26.379 | 42.294 | 1.000x |
| mlp_down | 128 | 896 | 4864 | FP8 Tensor | 11.037 | 101.082 | 2.390x |
| mlp_down | 128 | 896 | 4864 | FP8 Row | 17.920 | 62.260 | 1.472x |

跟V2做对比，先不看NCU，可以分析出一个很明显的现象，结果不是简单变好或者变差，而是出现下面的现象:
+ mlp_down 巨幅提升
+ attn_q_o 有小幅变化
+ mlp_up_gate 反而下降

#### mlp_down
对mlp_down来说:

```
mlp_down

M=128
N=896
K=4864

V2: 34.794 us 32.065 TFLOPS
V3: 11.037 us 101.082 TFLOPS
```

V3比V2提升了3.5倍，这个shape下K很大，N中等，M很小。
V2: Threadblock<64, 64, 64>, block: (128 * 896) / (64 * 64) = 28, K times: 4864 / 64 = 76
V3: Threadblock<32, 64, 128>, block: (128 * 896) / (32 * 64) = 56, K times: 4864 / 128 = 38

所以应该是在block提升且K维度减少，两方面的作用下，V3的mlp_down相比于V2有了3.X倍的提升。

#### attn_q_o
对attn_q_o来说:

```
attn_q_o

M=128
N=896
K=896

V2: 8.579 us 23.956 TFLOPS
V3: 8.602 us 23.893 TFLOPS
```

V3与V2的表现类似，M = N = 896，
V2: Threadblock<64, 64, 64>, block: (896 * 896) / (64 * 64) = 196.
V3: Threadblock<32, 64, 128>, block: (896 * 896) / (32 * 64) = 392.
V3的block固然更多，但是相比于4090的128个SM，其实V2提供的并行度已经足够，此时并行度已经不再是瓶颈了，继续增加block数量收益可能会被其他中间计算开销给抵消掉。

#### mlp_up_gate
V3还有个现象是mlp_up_gate的性能明显下降了

```
mlp_up_gate

M=128
N=4864
K=896

V2: 9.421 us 118 TFLOPS
V3: 11.776 us 94 TFLOPS
```

这个shape和mlp_down在N和K维度上正好相反，N非常大，K反而不大。
V2: Threadblock<64, 64, 64>, block: (128 * 4864) / (64 * 64) = 152
V3: Threadblock<32, 64, 128>, block: (128 * 4864) / (32 * 64) = 304

这里能看到V2实际上block数量已经可以很好的使用GPU上的SM，V3虽然block tile更小，但是block的数量却更多，由此带来的epilogue等等中间运算也更多。

所以V3相比于V2的改进在K-heavy GEMM表现很好，但是对N-heavy GEMM未必。

接下来分析一下ncu:
```
V2:
Compute (SM) Throughput           %         4.83
DRAM Throughput                   %        12.80
Duration                         us        39.81

V3:
Compute (SM) Throughput           %        13.00
DRAM Throughput                   %        33.93
Duration                         us        15.04
```

从这几个指标可以看出V3减少k迭代的次数，然后通过减小block tile，增加更多block数量实际上是提高了GPU的算力利用率的，但是不是每个场景都适用，在K维度很大的场景下V3的表现就比较好，而K维度小，N维度大的场景则因为N已经提供了足够的并行度所以V3的切分方式可能会造成更多的中间计算从而影响性能。

