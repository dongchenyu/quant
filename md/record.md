### 1.python 怎么绑定C++
下面这段函数在python中调用了一个函数w8a8_int8_linear_bbf16_obf16_per_tensor,然而这个函数其实并没有定义在python中,那我们怎么找它定义在哪里,其实没有在python中定义,那它基本上就是绑定C++来定义了. 顺便找一下在哪里import的
```
from runtime.sq_fp8_kernels import w8a8_int8_linear_bbf16_obf16_per_tensor

# 确保qweight.shape=[K,N],stride=[1,K],
y = w8a8_int8_linear_bbf16_obf16_per_tensor(qx, qweight, self.bias, alpha, 1.0)
```

然后能找到setup_runtime.py中查找到绑定的setup信息:
这里能得到的信息是source里面的一系列cu,cpp文件被编译链接成了runtime.sq_fp8_kernels.so
```
setup(
    name="runtime",
    version="0.1",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='runtime.sq_fp8_kernels',
            sources=[
                'runtime_refact/csrc/awq/gemm.cu',
                'runtime_refact/csrc/awq/gemm_db.cu',
                'runtime_refact/csrc/awq/gemv.cu',
                'runtime_refact/csrc/awq/gemv_coalesced.cu',
                'runtime_refact/csrc/smoothquant/sq_gemm.cu',
                'runtime_refact/csrc/fp8/sm89_fp8_rowwise_fbgemm.cu',
                'runtime_refact/csrc/fp8/sm89_fp8_tensorwise_cutlassgemm.cu',
                'runtime_refact/csrc/bindings.cpp',
            ],
        )
    ]
)
```

在bindings.cpp中, 所以接下来要在cpp/cuda文件中寻找w8a8_int8_linear_bbf16_obf16_per_tensor函数,最终在sq_gemm.cu中找到了.

以及为什么可以在C++中直接调用Tensor,这是#include <torch/extension.h>起作用的地方,python中的Tensor qx,到了C++里面是torch::Tensor input. 这里面是通过pytorch Extension来自动包装的.

这里还有个要注意的点是tensor数据通常没有被复制到CPU再传,而是从"device = cuda:0"直接传,变成了"torch::Tensor input",后续"input.data_ptr<int8_t>()"就可以直接拿到GPU的指针.

调用链条是:

Python -> C++ function -> CUTLASS C++ API -> CUDA kernel

```
m.def("w8a8_int8_linear_bbf16_obf16_per_tensor", &w8a8_int8_linear_bbf16_obf16_per_tensor, "int8 linear per tensor");

#include <torch/torch.h>
#include <torch/extension.h>
// A[M,K]xB[K,N], A rowmajor B colmajor
torch::Tensor w8a8_int8_linear_bbf16_obf16_per_tensor(...){}
```

### 2. CUTLASS
核心类: 前四组参数分别表示ABCD的数据类型与内存排布方式,后面是架构,先看这些,后面的分型随后再看吧.
```
using Gemm = cutlass::gemm::device::Gemm<
      int8_t, 
      cutlass::layout::RowMajor, 
      
      int8_t, 
      cutlass::layout::ColumnMajor,
      
      ElementOutput, 
      cutlass::layout::RowMajor, 
      
      ElementAccumulator,
      
      cutlass::arch::OpClassTensorOp, 
      cutlass::arch::Sm80,
      
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, 
      cutlass::gemm::GemmShape<16, 8, 32>,
      
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 8,//128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
```

这里面cutlass::MatrixCoord可以理解成是一个二维坐标结构体,标记出ABC矩阵的结构,这个后面的CUTLASS可以通过自身代码计算出layout(stride),然后后面的problem_size的数据类型也是一个GEMM的坐标参数结构体GemmCoord,包括矩阵运算的规模参数M,N,K.

```
auto input_size = cutlass::MatrixCoord(M, K);
auto weight_size = cutlass::MatrixCoord(K, N);
auto output_size = cutlass::MatrixCoord(M, N);
cutlass::gemm::GemmCoord problem_size(M, N, K);
```
然后是三个TensorRef与Arguments,这些是矩阵运算的核心.
其中对于三个TensorRef来说,组成部分包括两部分,第一部分是计算数据的其实地址,比如input.data_ptr<ElementInputA>(), 第二部分是layout,暂且可以当成数据排布格式+stride.这里面可以看作是前面的MatrixCoord被packed打包之后的结果.

```
  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out_data, LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;
```
下面是矩阵的三层计算,分别是每个block tile, warp tile, mma指令tile.
其中最上面的GemmShape<256, 128, 64>表示每个block tile计算一个256 * 128的tile,其中k维度的tile是64,每次迭代计算的bk = 64, 那么在K方向的迭代次数就是K / bK.

然后是下面GemmShape<64, 64, 64>, 表示每个warp tile计算一个64 * 64的tile,其中k维度tile是64, 跟上面的block tile做一个对比就能得出,每个block中的warp数量是(256 * 128) / (64 * 64) = 8, K方向上迭代的次数是1.

最下面GemmShape<16, 8, 32>, 表示一次MMA指令计算一个16 * 8的tile, 在mma里可以简单看成这种指令 m16n8k32, K维度的tile是32,也就是说K维度上两次迭代,这里面有个值得注意的点,一次MMA指令是一个warp共同做的,里面每个线程做了什么暂且不用考虑,所以这里可以看到其实一个warp执行了(64 * 64 * 64) / (16 * 8 * 32) = 64, 等于是一个warp执行64个MMA指令.

```
// 每个block计算256*128的tile,M=256,N=128,K=64
cutlass::gemm::GemmShape<256, 128, 64>,
cutlass::gemm::GemmShape<64, 64, 64>, 
cutlass::gemm::GemmShape<16, 8, 32>,
```

### 3. 模型调用链路
def from_pretrained()函数的作用是从config.json中先拿到对应的模型类,比如llama,Qwen之类.
check_and_get_model_type的作用是获取模型的种类.
```
class AutoQuantForCausalLM:
@classmethod
    def from_pretrained():
        model_type = check_and_get_model_type(quant_path, trust_remote_code)
```
这一步所做的事情是创建模型,但是不给分配真正的权重内存,我的理解是只确定每层权重的size,但是不往里面填充,比如可能标记这里有一层weight:Linear(4096,4096),但是不真正的创建这样一个线性层.
```
with init_empty_weights():
    model = target_cls.from_config(
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
```

此时创建的模型应该是还没有量化之前的模型,量化之后模型的层和权重是会不一样的.但是我们不想完全重新开一个模型,所以就通过
```
def _load_quantized_modules(self, model, quant_config, dtype=torch.float16):
```
其核心思想是将原来的网络层权重模型替换成量化之后的权重模型, 此时我们已经完成了根据原始的,未量化之前的模型转变成了量化之后的模型.
```
// 量化之前的网络层的名称为
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.bias
// 量化之后的网络层的名称为
model.layers.0.self_attn.q_proj.qweight
model.layers.0.self_attn.q_proj.weight_scale
model.layers.0.self_attn.q_proj.bias

//_load_quantized_modules的作用是将模型结构从上面变成下面 
```
接下来是加载量化之后的模型的权重.
```
load_checkpoint_and_dispatch(
    model,
    checkpoint=model_weights_path,
    device_map="auto",
    max_memory=max_memory,
    no_split_module_classes=[self.layer_type],
    offload_folder=offload_folder,
    dtype=torch_dtype,
)
```

```
_load_quantized_modules():
输入:

HF模型

每个layer里面:

nn.Linear
        |
找到Linear
        |
根据quant_method选择:

AWQLinear
SQLinear
FP8Linear
        |
调用from_linear()
        |
创建QuantLinear空壳
        |
替换原Linear
```

register_buffer可以理解成一个模型参数,但是不是可训练参数,不更新.
```
self.register_buffer('qweight', torch.randint(-127, 127, (self out_features, elf.in_features), dtype=torch.int8, requires_grad=False, device=dev))
```