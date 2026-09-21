这里先记录第一版的benchmark结果，再加上我的一些分析，结果如下，是以qwen2为基础进行的简单分析:

## Benchmark Results

| Layer | M | N | K | Kernel | Latency (us) | TFLOPS | Speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| attn_q_o | 1 | 896 | 896 | BF16 cuBLAS | 23.188 | 0.069 | 1.000x |
| attn_q_o | 1 | 896 | 896 | FP8 Tensor | 35.190 | 0.046 | 0.659x |
| attn_q_o | 1 | 896 | 896 | FP8 Row | 24.007 | 0.067 | 0.966x |
| attn_q_o | 16 | 896 | 896 | BF16 cuBLAS | 24.986 | 1.028 | 1.000x |
| attn_q_o | 16 | 896 | 896 | FP8 Tensor | 35.025 | 0.733 | 0.713x |
| attn_q_o | 16 | 896 | 896 | FP8 Row | 24.274 | 1.058 | 1.029x |
| attn_q_o | 128 | 896 | 896 | BF16 cuBLAS | 31.713 | 6.481 | 1.000x |
| attn_q_o | 128 | 896 | 896 | FP8 Tensor | 35.779 | 5.744 | 0.886x |
| attn_q_o | 128 | 896 | 896 | FP8 Row | 27.458 | 7.485 | 1.155x |
| mlp_up_gate | 1 | 4864 | 896 | BF16 cuBLAS | 21.780 | 0.400 | 1.000x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Tensor | 36.193 | 0.241 | 0.602x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Row | 27.116 | 0.321 | 0.803x |
| mlp_up_gate | 16 | 4864 | 896 | BF16 cuBLAS | 25.212 | 5.531 | 1.000x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Tensor | 35.953 | 3.879 | 0.701x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Row | 26.892 | 5.186 | 0.938x |
| mlp_up_gate | 128 | 4864 | 896 | BF16 cuBLAS | 29.756 | 37.495 | 1.000x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Tensor | 37.627 | 29.651 | 0.791x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Row | 39.793 | 28.037 | 0.748x |
| mlp_down | 1 | 896 | 4864 | BF16 cuBLAS | 21.631 | 0.403 | 1.000x |
| mlp_down | 1 | 896 | 4864 | FP8 Tensor | 139.377 | 0.063 | 0.155x |
| mlp_down | 1 | 896 | 4864 | FP8 Row | 33.308 | 0.262 | 0.649x |
| mlp_down | 16 | 896 | 4864 | BF16 cuBLAS | 30.950 | 4.506 | 1.000x |
| mlp_down | 16 | 896 | 4864 | FP8 Tensor | 128.742 | 1.083 | 0.240x |
| mlp_down | 16 | 896 | 4864 | FP8 Row | 33.439 | 4.171 | 0.926x |
| mlp_down | 128 | 896 | 4864 | BF16 cuBLAS | 27.162 | 41.075 | 1.000x |
| mlp_down | 128 | 896 | 4864 | FP8 Tensor | 133.133 | 8.380 | 0.204x |
| mlp_down | 128 | 896 | 4864 | FP8 Row | 33.772 | 33.036 | 0.804x |

从当前来看，值得关注的是rowwise在很多场景下已经可以逼近BF16 cublas，但是tensorwise就差很多了，首先把关注的重点转移到配置信息上。

然后重点看一下我在当前这个版本所做的一些配置: 这些配置当初在选择的事情其实没有考虑太多，只是注意了InstructionShape与tensor core的MMA指令贴近。
```
Tensorwise:
ThreadblockShape = <128, 128, 64>
WarpShape        = <64, 64, 64>
InstructionShape = <16, 8, 32>
Stages           = 3
Swizzle          = Identity

Rowwise:
ThreadblockShape = <32, 64, 128>
WarpShape        = <16, 64, 64>
InstructionShape = <16, 8, 32>
Stages           = 5
Swizzle          = StreamK
```

接下来挑一个最明显的例子吧，mlp_down K=4864，因为这个数据上看着差异就很明显，尤其是tensorwise

M = 128, N = 896, K = 4864
所以对于tensorwise，用的block数量为(128 * 896) / (128 * 128) = 7, 而每个tile的K维度的迭代是4864 / 64 = 76。
对比一下同样是手动执行的rowwise，用的block数量为(128 * 896) / (32 * 64) = 56, 而每个tile的K维度的迭代是4864 / 128 = 38.

所以通常来说多用一些block，减少一些K维度的迭代对于性能是有帮助的。

但是mlp_up_gate M=128 tensorwise反而比rowwise要好一些，继续分析:

M=128, N=4864, K=896

tensorwise: 用的block数量为(128 * 4864) / (128 * 128) = 38, K维度的迭代为896 / 64 = 14
rowwise: 用的block数量为(128 * 4864) / (32 * 64) = 304, K维度的迭代为896 / 128 = 7

所以能看出这个时候tensorwise用到38个block之后其实已经有一点并行度了，而rowwise调用epilogue的问题开始影响性能了，因为rowwise用了更复杂的EVT树。

在V1的基础之上清除了一些干扰项，主要有两点:
1. tensorwise: 之前bias处理的有问题，有没有bias都新创建一个tensor，这样benchmark在没有bias测试的时候会在这里吃时间。
2. rowwise多加了同步。

改完这两点之后是这样子的

## Benchmark Results

| Layer | M | N | K | Kernel | Latency(us) | TFLOPS | Speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| attn_q_o | 1 | 896 | 896 | BF16 cuBLAS | 22.997 | 0.070 | 1.000x |
| attn_q_o | 1 | 896 | 896 | FP8 Tensor | 32.108 | 0.050 | 0.716x |
| attn_q_o | 1 | 896 | 896 | FP8 Row | 14.935 | 0.108 | 1.540x |
| attn_q_o | 16 | 896 | 896 | BF16 cuBLAS | 24.668 | 1.041 | 1.000x |
| attn_q_o | 16 | 896 | 896 | FP8 Tensor | 32.148 | 0.799 | 0.767x |
| attn_q_o | 16 | 896 | 896 | FP8 Row | 14.778 | 1.738 | 1.669x |
| attn_q_o | 128 | 896 | 896 | BF16 cuBLAS | 29.809 | 6.895 | 1.000x |
| attn_q_o | 128 | 896 | 896 | FP8 Tensor | 32.998 | 6.228 | 0.903x |
| attn_q_o | 128 | 896 | 896 | FP8 Row | 16.261 | 12.639 | 1.833x |
| mlp_up_gate | 1 | 4864 | 896 | BF16 cuBLAS | 21.504 | 0.405 | 1.000x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Tensor | 33.060 | 0.264 | 0.650x |
| mlp_up_gate | 1 | 4864 | 896 | FP8 Row | 15.857 | 0.550 | 1.356x |
| mlp_up_gate | 16 | 4864 | 896 | BF16 cuBLAS | 23.957 | 5.821 | 1.000x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Tensor | 33.198 | 4.201 | 0.722x |
| mlp_up_gate | 16 | 4864 | 896 | FP8 Row | 16.707 | 8.348 | 1.434x |
| mlp_up_gate | 128 | 4864 | 896 | BF16 cuBLAS | 27.802 | 40.130 | 1.000x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Tensor | 33.884 | 32.927 | 0.821x |
| mlp_up_gate | 128 | 4864 | 896 | FP8 Row | 23.296 | 47.891 | 1.193x |
| mlp_down | 1 | 896 | 4864 | BF16 cuBLAS | 21.796 | 0.400 | 1.000x |
| mlp_down | 1 | 896 | 4864 | FP8 Tensor | 136.259 | 0.064 | 0.160x |
| mlp_down | 1 | 896 | 4864 | FP8 Row | 18.387 | 0.474 | 1.185x |
| mlp_down | 16 | 896 | 4864 | BF16 cuBLAS | 28.640 | 4.869 | 1.000x |
| mlp_down | 16 | 896 | 4864 | FP8 Tensor | 126.525 | 1.102 | 0.226x |
| mlp_down | 16 | 896 | 4864 | FP8 Row | 18.562 | 7.513 | 1.543x |
| mlp_down | 128 | 896 | 4864 | BF16 cuBLAS | 27.557 | 40.486 | 1.000x |
| mlp_down | 128 | 896 | 4864 | FP8 Tensor | 130.391 | 8.556 | 0.211x |
| mlp_down | 128 | 896 | 4864 | FP8 Row | 17.722 | 62.955 | 1.555x |

挑下面这个例子做分析:
```
M = 128
N = 896
K = 4864

ThreadblockShape = 128×128×64

M tile = 128/128 = 1
N tile = 896/128 = 7

Block Size 128    
Grid Size 7
```
根据之前的分析，用了7个block，每个block在K维度上进行了4864 / 64 = 76次迭代.从ncu分析中可以看到是能对的上的，然后每个block里面有128个线程，即4个warp。

然后分析下Occupancy，可以看到的是Achieved Occupancy 非常低，只有2/24 = 8.33%，这里可以看到每个SM上本来最多可以放24个block，但是因为Registers和Shared Mem的限制，每个SM上只能放2个block。所以也不止block太少，还有个原因是每个block消耗资源太多了。

```
Block Limit SM                        block           24
Block Limit Registers                 block            2
Block Limit Shared Mem                block            2
Block Limit Warps                     block           12
Theoretical Active Warps per SM        warp            8
Theoretical Occupancy                     %        16.67
Achieved Occupancy                        %         8.33
Achieved Active Warps Per SM           warp         4.00
```

接下来分析下Compute / Memory，这三个数据非常之低，既不是memory bound(memory bandwidth 饱和)，也不是compute bound(Tensor Core 饱和)，而是latency bound。
```
Memory Throughput           %      8.07
DRAM Throughput             %      3.10
Compute (SM) Throughput     %      1.71
```
7 blocks + 每block消耗大量的register/shared memory

### 优化思路
所以现在的问题是ThreadblockShape = <32, 64, 128>设置的太大了，导致block数量少，而且每个block太重，消耗的寄存器过多(看上去已经逼近255的上限了)，这就导致了SM利用率不高(很多SM似乎就没参与计算)，所以首先考虑的不是优化occupancy，而是先让更多的SM参与工作，warp stall cycle过高，每周期发射指令数量过少的时候才优先考虑提升occupancy。