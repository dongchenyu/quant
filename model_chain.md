## 模型调用链
### 1. 模型结构与参数修改 
由于量化前后不论是模型的权重参数值还是模型的参数结构都是有变化的，参数结构的变化是加了类似scale层.
   
先加载一波模型参数，这一次加载的是Qwen2，这里面加载的效果更类似于从类的角度来加载，代码和效果如下:

可以看到Qwen内部的block，大体可以分为attention，mlp以及layernorm.
```
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu")

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
```

下面的load_file函数则更倾向于打印模型的内容了，比如模型的权重叫什么名字，是什么类型，shape是多少等等，代码和打印结果如下:

```
state = load_file(str(SRC / "model.safetensors"))

model.layers.0.self_attn.q_proj.bias 
model.layers.0.self_attn.q_proj.weight
```

打印出上面的state之后我们要做的是将权重提取出来，然后根据量化公式来进行计算，得到量化后的权重与scale，取出量化weight的方式是筛选出后缀为.weight的权重，然后将权重保存到new_state中，然后同时提取出weight_scale，也用类似的手段，然后save_file将weight与scale保存起来。修改前后的参数对比如下:

```
// 修改前
model.layers.0.self_attn.q_proj.bias 
model.layers.0.self_attn.q_proj.weight

// 修改后 
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.weight_scale
````

最后在config.json里面添加一下我们修改的量化配置
```
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
```

### 2. 加载模型，替换linear
上一步将原始的模型修改成了量化之后的模型，那接下来要做的事情就是加载这个模型，并且考虑到量化之后的GEMM跟pytorch自带的nn.Linear()还不太一样，所以需要做一个替换，替换的逻辑在base.py的class BaseModelForCausalLM中.

大体的思路是用linear中的class FP8DynamicLinear来替换掉nn.Linear(其实这也是一个class)，然后因为我们之前将模型的参数由(weight, bias)转换成了(weight, weight_scale, bias)，所以在BaseModelForCausalLM的init函数中也需要注册(self.weight, self.weight_scale, self.bias)，不然的话参数会对不上，这里会留下(weight, weight_scale, bias)的槽位，然后根据我们前面搞出来的那几个模型层，一波checkpoint就将权重加载了。

然后FP8DynamicLinear中的forward实际上起到了nn.Linear的作用，实际上是由pybind绑定的cuda核函数。

### 3. cuda核函数
这里的gemm是用cutlass实现的，分为per-tensor与per-channel两个部分，其中per-tensor简单一些

#### per-tensor
首先构建一个Gemm参数，这里是一个默认的线性计算，表达式为$D = alpha * A * B + beta * C$
```
using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, cutlass::layout::RowMajor, // A矩阵的数据类型，A矩阵是row-major
    ElementInputB, cutlass::layout::ColumnMajor, // B矩阵的数据类型，B矩阵是col-major
    ElementOutput, cutlass::layout::RowMajor,  // 最终的结果矩阵D的数据类型，D是Row-major
    ElementAccumulator, // Acc计算的结果数据类型，就是矩阵乘法A*B
    cutlass::arch::OpClassTensorOp,  // 使用cuda core
    cutlass::arch::Sm89, // sm架构
    ThreadblockShape,/*cutlass::gemm::GemmShape<256, 128, 64>*/ // 每个block处理的数据tile，其中M=256，N=128，这是处理的数据总数，然后K维度每次迭代处理64个元素
    WarpShape,/*cutlass::gemm::GemmShape<64, 64, 64>*/  // 每个warp处理的数据tile，其中M=64，N=64，这是处理的数据总数，然后K维度每次迭代处理64个元素， 这里可以看出有8个warp，并且在K方向只需要迭代一次。
    cutlass::gemm::GemmShape<16, 8, 32>, // 每个tensorcore指令一次处理的数据tile，这里的每条指令是一个warp中32个thread一起完成的。M=16，N=8，K方向每次迭代处理32个元素。这里能看出每条指令在K方向其实要迭代两次。
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 8,//128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementComputeEpilogue>,
    // 这个参数表示LinearCombination，即线性运算的数据类型，其中ElementOutput和ElementAccumulator上面介绍过，是output和Accumulate的数据类型，然后ElementComputeEpilogue是最后的累加计算的数据类型，即(alpha * A * B)与(beta * C)的数据类型是ElementComputeEpilogue，累加之后的结果是ElementOutput。
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;
```

然后紧接着是问题规模的定义，其中





