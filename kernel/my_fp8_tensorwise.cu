#ifndef STANDALONE_TEST
#include "my_fp8_gemm.h"
#endif
// #include <ATen/ATen.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include <optional>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// ============================================================
// CUDA CHECK
// ============================================================

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: "                                    \
                      << cudaGetErrorString(err)                            \
                      << " at line " << __LINE__ << std::endl;              \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

using ElementInputA = cutlass::float_e4m3_t;
using ElementInputB = cutlass::float_e4m3_t;     
using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementOutput = cutlass::bfloat16_t;

using LayoutInputA = cutlass::layout::RowMajor; 
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
    
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

using LinearCombination = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    8,
    ElementAccumulator,
    ElementComputeEpilogue
>;

using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    LinearCombination,
    ThreadblockSwizzle,
    NumStages>;

void run_cutlass_fp8_gemm(
    ElementInputA* input_ptr,
    ElementInputB* weight_ptr,
    ElementOutput* bias_ptr,
    ElementOutput* output_ptr,
    int M, int N, int K, float alpha, float beta
){
    auto input_size = cutlass::MatrixCoord(M, K);
    auto weight_size = cutlass::MatrixCoord(K, N);
    auto output_size = cutlass::MatrixCoord(M, N);

    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
        input_ptr, LayoutInputA::packed(input_size)
    );

    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
        weight_ptr, LayoutInputB::packed(weight_size)
    );

    cutlass::TensorRef<ElementOutput, LayoutOutput> bias_ref(
        bias_ptr, LayoutOutput::packed(output_size)
    );

    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(
        output_ptr, LayoutOutput::packed(output_size)
    );

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments arguments{
        problem_size,
        input_ref,
        weight_ref,
        bias_ref,
        output_ref,
        {alpha, beta}, 1
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if(status != cutlass::Status::kSuccess){
        throw std::runtime_error(
            std::string("CUTLASS GEMM initialize failed ") +
            cutlassGetStatusString(status));
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if(status != cutlass::Status::kSuccess){
        throw std::runtime_error(
            std::string("CUTLASS GEMM initialize failed: ") +
            cutlassGetStatusString(status));
    }

    status = gemm_op();
    if(status != cutlass::Status::kSuccess){
        throw std::runtime_error(
            std::string("CUTLASS GEMM run failed") +
            cutlassGetStatusString(status));
    }
}

#ifndef STANDALONE_TEST
torch::Tensor my_f8f8bf16_tensorwise(
    torch::Tensor input, // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias, // BF16
    float alpha,
    float beta,
    const std::string& dtype_str) 
{
    int M = input.size(0);
    int K = weight.size(0);
    int N = weight.size(1);

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(input.device());
    
    torch::Tensor bias_matrix;
    if(bias.has_value()){
        bias_matrix = bias->view({1, N}).repeat({M, 1}).contiguous();
    } else{
        bias_matrix = torch::zeros({M, N}, options);
    }

    auto output = torch::zeros({M, N},options);
    
    ////
    //void* input_data = input.data_ptr();
    //void* weight_data = weight.data_ptr();
    //void* output_data = output.data_ptr();

    ElementInputA* input_ptr = static_cast<ElementInputA*>(input.data_ptr());
    ElementInputB* weight_ptr = static_cast<ElementInputB*>(weight.data_ptr());
    ElementOutput* bias_ptr = static_cast<ElementOutput*>(bias_matrix.data_ptr());
    ElementOutput* output_ptr = static_cast<ElementOutput*>(output.data_ptr());

    run_cutlass_fp8_gemm(input_ptr, weight_ptr, bias_ptr, output_ptr,
        M, N, K, alpha, beta);

    return output;
}
#endif

template<typename Element>
float quantize_tensorwise(
    const std::vector<float>& input,
    std::vector<Element>& output)
{
    constexpr float FP8_MAX = 448.0f;
    float amax = 0.0f;

    for(float x : input){
        amax = std::max(amax, std::abs(x));
    }

    float scale = (amax == 0.0f) ? 1.0f : amax / FP8_MAX;

    for(size_t i = 0; i < input.size(); i ++){
        float x = input[i] / scale;
        x = std::max(-FP8_MAX, std::min(FP8_MAX, x));
        output[i] = Element(x);
    }

    return scale;
}

void cpu_ref(
    const std::vector<ElementInputA>& A,
    const std::vector<ElementInputB>& B,
    const std::vector<float>& bias,
    std::vector<float>& ref,
    int M, int N, int K, float alpha, float beta
){
    for(int m = 0; m < M; m ++){
        for(int n = 0; n < N; n ++){
            float acc = 0.0f;
            for(int k = 0; k < K; k ++){
                float a = static_cast<float>(A[m * K + k]);
                float b = static_cast<float>(B[k + n * K]);
                acc += a * b;
            }
            float value = alpha * acc + beta * bias[n];
            ElementOutput rounded = ElementOutput(value);
            ref[m * N + n] = static_cast<float>(rounded);
        }
    }
}

int main(){
    int M = 128;
    int N = 128;
    int K = 256;

    std::cout << "M=" << M << ", N=" << N << ", K=" << K << '\n';

    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(N * K);
    std::vector<float> bias(N);

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(float& x: A_fp32){
        x = dist(gen);
    }

    for(int n = 0; n < N; n ++){
        for(int k = 0; k < K; k ++){
            B_fp32[k + n * K] = dist(gen);
        }
    }

    for(float& x : bias){
        x = dist(gen);
    }

    std::vector<ElementInputA> A_fp8(M * K);
    std::vector<ElementInputB> B_fp8(K * N);

    float scale_a = quantize_tensorwise(A_fp32, A_fp8);
    float scale_b = quantize_tensorwise(B_fp32, B_fp8);

    float alpha = scale_a * scale_b;
    float beta = 1.0f;

    std::cout << "scale_a = " << scale_a << ", scale_b = " << scale_b
              << ", alpha=" << alpha << std::endl;

    std::vector<ElementOutput> C_host(M * N);
    for(int m = 0; m < M; m ++){
        for(int n = 0; n < N; n ++){
            C_host[m * N + n] = ElementOutput(bias[n]);
        }
    }

    ElementInputA* d_A = nullptr;
    ElementInputB* d_B = nullptr;
    ElementOutput* d_C = nullptr;
    ElementOutput* d_D = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), 
        sizeof(ElementInputA) * A_fp8.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), 
        sizeof(ElementInputB) * B_fp8.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), 
        sizeof(ElementOutput) * C_host.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_D), 
        sizeof(ElementOutput) * M * N));

    CUDA_CHECK(cudaMemcpy(d_A, A_fp8.data(),sizeof(ElementInputA) * A_fp8.size(),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_fp8.data(),sizeof(ElementInputB) * B_fp8.size(),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C_host.data(),sizeof(ElementOutput) * C_host.size(),
        cudaMemcpyHostToDevice));
    
    run_cutlass_fp8_gemm(d_A, d_B, d_C, d_D, M, N, K, alpha, beta);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<ElementOutput> D_host(M * N);
    CUDA_CHECK(cudaMemcpy(D_host.data(), d_D, sizeof(ElementOutput) * D_host.size(),
        cudaMemcpyDeviceToHost));

    std::vector<float> ref(M * N);
    cpu_ref(A_fp8, B_fp8, bias, ref, M, N, K, alpha, beta);

    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    int max_idx = 0;

    for(int i = 0; i < M * N; i ++){
        float gpu = static_cast<float>(D_host[i]);
        float diff = std::abs(gpu - ref[i]);
        mean_diff += diff;
        if(diff > max_diff){
            max_diff = diff;
            max_idx = i;
        }
    }

    mean_diff /= static_cast<float>(M * N);

    std::cout << "max diff = " << max_diff << std::endl;
    std::cout << "mean diff = " << mean_diff << std::endl;
    std::cout << "max diff index = " << max_idx << ","
            << ", GPU=" << static_cast<float>(D_host[max_idx])
            << ", CPU=" << ref[max_idx] << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

}

/*
nvcc my_fp8_tensorwise.cu \
    -std=c++17 \
    -arch=sm_89 \
    -DSTANDALONE_TEST \
    -I/root/LLMQRT-main/runtime_refact/3rdparty/cutlass/include \
    -I/root/LLMQRT-main/runtime_refact/3rdparty/cutlass/tools/util/include \
    -o my_fp8_tensorwise
*/