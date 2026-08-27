#include "learn_int8_gemm.h"
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include <iostream>
#include <cmath>

/*
torch::Tensor input,  // INT8
torch::Tensor weight, // INT8
torch::Tensor bias,   // BF16
float alpha,          // BF16
float beta            // BF16

float result = alpha * float(accumulator) + beta * float(C);
D = half(result);
D = half(scale_a * scale_w * Acc_INT32 + bias);
*/
torch::Tensor cutlass_int8_gemm_per_tensor(torch::Tensor input, 
        torch::Tensor weight, torch::Tensor bias, float alpha, float beta){

    auto M = input.size(0);
    auto N = weight.size(1);
    auto K = input.size(1);

    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementAccumulator = int32_t;
    using ElementComputeEpilogue = float;
    using ElementOutput = cutlass::half_t;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutInputOutput = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutInputOutput,

        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,

        cutlass::gemm::GemmShape<256, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,

        cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t,
            8,
            int32_t,
            float
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3
    >;

    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be int8");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat16, "bias must be float16");
    
    TORCH_CHECK(input.stride(0) == K && input.stride(1) == 1, "input must be RowMajor");
    TORCH_CHECK(weight.stride(0) == 1 && weight.stride(1) == K, "weight must be ColumnMajor [K,N]");

    auto problem_size = cutlass::gemm::GemmCoord(M, N, K);

    auto input_size = cutlass::MatrixCoord(M, K);
    auto weight_size = cutlass::MatrixCoord(K, N);
    auto output_size = cutlass::MatrixCoord(M, N);

    auto C = bias.view({1, N}).repeat({M, 1});
    auto D = C.clone();

    using TensorRefA = cutlass::TensorRef<ElementInputA, LayoutInputA>;
    using TensorRefB = cutlass::TensorRef<ElementInputB, LayoutInputB>;
    using TensorRefC = cutlass::TensorRef<ElementOutput, LayoutInputOutput>;
    
    TensorRefA input_ref(reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
            LayoutInputA::packed(input_size));

    TensorRefB weight_ref(reinterpret_cast<ElementInputB*>(weight.data_ptr<int8_t>()),
            LayoutInputB::packed(weight_size));

    TensorRefC C_ref(reinterpret_cast<ElementOutput*>(C.data_ptr<at::Half>()),
            LayoutInputOutput::packed(output_size));

    TensorRefC D_ref(reinterpret_cast<ElementOutput*>(D.data_ptr<at::Half>()),
            LayoutInputOutput::packed(output_size));

    typename Gemm::Arguments arguments(
        problem_size,
        input_ref,
        weight_ref,
        C_ref,
        D_ref,
        {alpha, beta}, 1
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS can_implement failed"
    );

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace_ptr = nullptr;

    torch::Tensor workspace;

    //// ????
    if(workspace_size > 0){
        workspace = torch::empty({static_cast<int64_t>(workspace_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

        workspace_ptr = workspace.data_ptr();
    }

    status = gemm_op.initialize(arguments, workspace_ptr);
    
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS initialize failed"
    );

    status = gemm_op();

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS run failed"
    );
    return D;
}

torch::Tensor cpu_int8_gemm_ref(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float alpha, float beta){
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight.size(1);

    auto A_cpu = input.cpu().contiguous();
    auto W_cpu = weight.t().cpu().contiguous();

    auto bias_cpu = bias.cpu().to(torch::kFloat32).contiguous();

    auto ref = torch::empty(
        {M, N},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
    );

    const int8_t* A_ptr = A_cpu.data_ptr<int8_t>();
    const int8_t* W_ptr = W_cpu.data_ptr<int8_t>();
    const float* bias_ptr = bias_cpu.data_ptr<float>();
    float* ref_ptr = ref.data_ptr<float>();

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                int8_t a = A_ptr[m * K + k];
                int8_t w = W_ptr[n * K + k];
                acc += static_cast<int32_t>(a) * static_cast<int32_t>(w);
            }
            ref_ptr[m * N + n] = alpha * static_cast<float>(acc) + beta * bias_ptr[n];
        }
    }
    return ref;
}

int main(){
    torch::manual_seed(0);
    torch::Device device(torch::kCUDA, 0);

    int M = 128;
    int N = 512;
    int K = 1024;

    // Aq [M,K] RowMajor contiguous
    // Wq [N,K] RowMajor
    auto Aq = torch::randint(-10, 11, {M, K},
        torch::TensorOptions().dtype(torch::kInt8).device(device));

    auto Wq = torch::randint(-10, 11, {N, K},
        torch::TensorOptions().dtype(torch::kInt8).device(device));

    // Wq trans: logical [K,N] ColumnMajor stride = (1,K)
    auto Wq_trans = Wq.t();

    auto bias = torch::randn({N},
        torch::TensorOptions().dtype(torch::kFloat16).device(device));

    float alpha = 0.1f;
    float beta = 1.0f;

    std::cout << "A shape: " << Aq.sizes() << std::endl;
    std::cout << "A stride: " << Aq.strides() << std::endl;
    std::cout << "B shape: " << Wq_trans.sizes() << std::endl;
    std::cout << "B stride: " << Wq_trans.strides() << std::endl;

    auto output = cutlass_int8_gemm_per_tensor(Aq, Wq_trans, bias, alpha, beta);
    
    cudaError_t sync_status = cudaDeviceSynchronize();
    if(sync_status != cudaSuccess){
        std::cerr << "cudaDeviceSynchronize failed: "
                  << cudaGetErrorString(sync_status)
                  << std::endl;
        return 1;
    }
    
    output = output.cpu().to(torch::kFloat32);

    // Reference
    // Aq @ Wq.T = [M,K] @ [K,N]
    auto ref_fp32 = cpu_int8_gemm_ref(Aq, Wq_trans, bias, alpha, beta);
    auto ref_fp16_as_fp32 = ref_fp32.to(torch::kFloat16).to(torch::kFloat32);

    auto diff = (output.to(torch::kFloat32) - ref_fp16_as_fp32).abs();

    float max_diff = diff.max().item().toFloat();
    float mean_diff = diff.mean().item().toFloat();

    std::cout << "max diff = " << max_diff << std::endl;
    std::cout << "mean diff = " << mean_diff << std::endl;
    std::cout << std::endl;

    if (mean_diff < 0.01f) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }


    return 0;
}

/*
nvcc \
  runtime_refact/csrc/smoothquant/learn_int8_gemm.cu \
  -o learn_int8_gemm \
  -std=c++17 \
  -O3 \
  -arch=sm_89 \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -U__CUDA_NO_HALF_OPERATORS__ \
  -U__CUDA_NO_HALF_CONVERSIONS__ \
  -U__CUDA_NO_HALF2_OPERATORS__ \
  -I runtime_refact/3rdparty/cutlass/include \
  -I runtime_refact/3rdparty/cutlass/tools/util/include \
  -I /root/miniconda3/lib/python3.12/site-packages/torch/include \
  -I /root/miniconda3/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
  -I /usr/local/cuda/include \
  -I /root/miniconda3/include/python3.12 \
  -L /root/miniconda3/lib/python3.12/site-packages/torch/lib \
  -L /usr/local/cuda/lib64 \
  -ltorch \
  -ltorch_cpu \
  -ltorch_cuda \
  -lc10 \
  -lc10_cuda \
  -lcudart
*/