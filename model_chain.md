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

接下来要做的是构建Arguments，Arguments代码如下，解析一下。

```
typename Gemm::Arguments arguments{
    problem_size, // <- problem size of matrix multiplication
    input_ref,    // <- reference to matrix A on device
    weight_ref,   // <- reference to matrix B on device
    out_ref,      // <- reference to matrix C on device
    out_ref,      // <- reference to matrix D on device
    {alpha, beta}, 1};

cutlass::gemm::GemmCoord problem_size(M, N, K);

auto input_size = cutlass::MatrixCoord(M, K);
auto weight_size = cutlass::MatrixCoord(K, N);
auto output_size = cutlass::MatrixCoord(M, N);

void* out_ptr = out.data_ptr();
ElementOutput* out_data = static_cast<ElementOutput*>(out_ptr);

void* input_ptr = input.data_ptr();
ElementInputA* input_data = static_cast<ElementInputA*>(input_ptr);

void* weight_ptr = weight.data_ptr();
ElementInputB* weight_data = static_cast<ElementInputB*>(weight_ptr);

cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
    input_data, LayoutInputA::packed(input_size));
    //input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
    weight_data, LayoutInputB::packed(weight_size));
    //weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
    out_data, LayoutOutput::packed(output_size));
```
这里的参数一共可以分为三组，第一组是problem_size，顾名思义，就是问题的规模(M,N,K)，这个比较简单。

第二组比较复杂，是input_ref，output_ref这些，这里的每个ref其实包含两个部分，一个是数据的指针，一个是如何用stride来解释这样一组逻辑数据。数据指针其实比较好理解，对于输入参数的input，weight以及后续定义的out来说其实就是torch::Tensor的.data_ptr()，后面的stride其实可以这么理解，前面的input，weight的数据在内存中都是以一维形式排布的，现在就是需要我们的代码配置将它们解释成为一个逻辑上的二维矩阵，所以前面通过cutlass::MatrixCoord先确定矩阵的规模以及前面给出了矩阵是row/col-major，其实可以推导出矩阵的stride，这里可以记住就是用LayoutInputA::packed(input_size) 这种，比如说cutlass::MatrixCoord(M, K) + row-major的stride就是(K, 1),如果列优先就是(1, M).

第三组很简单了，就是alpha，beta，就是epilogue的参数，后面的1是split-K的参数。

接下来是一个既定的城西，首先Gemm::get_workspace_size(arguments)，判断下需不需要workspace，这个一般是split-K存储中间结果用的，然后是gemm_op.can_implement(arguments)，看下参数是否合法，然后是gemm_op.initialize(arguments, workspace.get())做初始化，最后执行。


#### per-channel
这个就复杂很多了，因为per-channel的scale不是一个值，而是一个张量。或者说对于每个输出位置的元素爱说，w_scale和input_scale都是不一样的，这就麻烦很多。

首先看一下这里的cutlass::epilogue::threadblock::OutputTileThreadLayout，这个是我觉得在per-channel或者说在自定义计算树里非常核心也不太好理解的一个概念:
就是每个线程分别负责处理输出tensor的哪些元素(哪些坐标)，因为前面其实已经提过，每个输出元素要乘上的scale其实不一样，所以需要搞清楚每个元素要乘的参数分别是什么，这里就是做这个的，先确定好某个元素(m,n)由哪个元素来进行处理，然后再根据坐标看它对应的scale是什么。

下面的cutlass::epilogue::threadblock::VisitorAccFetch类似，表示当前线程计算的是Acc计算结果的哪个元素。
```
using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          DtypeOutput,
          AlignmentOutput,
          NumEVTEpilogueStages>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
```

接下来这个参数跟前面的有相关性，它记录的是当前的线程处理的是元素(m,n)，那么(m,n)对应的元素应该与XScale的哪个元素做计算呢？这里取决于cute::Stride<cute::_1, cute::_0, int64_t>>，如果是(m,n)的话就是与XScale[m * 1 + n * 0]计算了，最后一个是运行时参数，是表示batch的，在后面实例化XScaleArguments的时候才加载。所以可以看到对于输出元素(m, n)而言，使用的Xscale只取决于横坐标m。
```
using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, DtypeScale,
      cute::Stride<cute::_1, cute::_0, int64_t>>;
  using XScaleArguments = typename XScale::Arguments;

  XScaleArguments x_scale_arguments{
      (DtypeScale*)x_scale.data_ptr(),
      DtypeScale(1),
      {cute::_1{}, cute::_0{}, problem_size.m()}
  };
```

接下来进入计算树的构建，ApplyXScale是计算节点，描述的是计算的种类与数据类型，像下面这个就可以确定是进行乘法运算，第一个DtypeEpilogue表示计算结束后的输出类型，第二个DtypeEpilogue表示的是计算的操作数的类型，即参与计算的数据的类型。

然后EVTApplyXScale实际上已经开始参与计算树的构建了，其中ApplyXScale表示计算节点，是这里面的根节点，然后Accum，XScale表示当前线程处理或者使用的Accum和XScale数据，是叶节点，这里可以看成一个中序遍历求值，就是当前线程对Accum，XScale做一个ApplyXScale的计算，这里是乘法，这里完成了计算节点的构建。
```
using ApplyXScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, DtypeEpilogue, DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

using EVTApplyXScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyXScale,// NodeOp即根节点
      Accum,//childOp
      XScale>;//childOp
```

然后最后这个存储节点似乎有一点点特殊, 它表示当前线程要参与存储哪个数据(坐标)，这里面其实能看出来是一个行主序，因为第二维stride为1.

然后最后这个EVTOutput是计算树的最顶部，EVTApplyBias是最终的计算结果，然后Output是存储操作，就表示将最终的计算树的结果按照output定义的方式(行主序)来进行存储。

```
using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, DtypeOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplyBias>;
```
