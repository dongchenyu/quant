#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/util/device_memory.h>

#include <optional>
#include <torch/torch.h>
#include <torch/extension.h>


#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUTLASS_CHECK(status)                                                    \
  do {                                                                           \
    cutlass::Status s = (status);                                                 \
    if (s != cutlass::Status::kSuccess) {                                         \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(s)               \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

using DtypeA = cutlass::float_e4m3_t;
using DtypeB = cutlass::float_e4m3_t;
using DtypeScale = float;
using DtypeBias = cutlass::bfloat16_t;
using DtypeAccum = float;
using DtypeEpilogue = float;
using DtypeOutput = cutlass::bfloat16_t;

std::vector<DtypeOutput> cpu_ref(
    const std::vector<DtypeA>& A,
    const std::vector<DtypeB>& B,
    const std::vector<float>& x_scale,
    const std::vector<float>& w_scale,
    const std::vector<float>& bias,
    int M, int N, int K)
{
    std::vector<DtypeOutput> out(M * N);
    for(int m = 0; m < M; m ++){
        for(int n = 0; n < N; n ++){
            float acc = 0.0f;
            for(int k = 0; k < K; k ++){
                float a = static_cast<float>(A[m * K + k]);
                float b = static_cast<float>(B[n * K + k]);
                acc += a * b;
            }
            float y = acc * x_scale[m] * w_scale[n] + bias[n];
            out[m * N + n] = DtypeOutput(y);
        }
    }
    return out;
}    

torch::Tensor my_f8f8bf16_rowwise(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum)
{
    const int M = XQ.size(0);
    const int K = XQ.size(1);
    const int N = WQ.size(1);

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    constexpr int AlignmentInputA = 16 / sizeof(DtypeA);
    constexpr int AlignmentInputB = 16 / sizeof(DtypeB);
    constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

    using ArchTag = cutlass::arch::Sm89;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 128>;
    using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

    constexpr int NumStages = 5;
    constexpr int NumEVTEpilogueStages = 1;

    using Operator = cutlass::arch::OpMultiplyAdd;

    // EVT tree begin:
    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        DtypeOutput,
        AlignmentOutput,
        NumEVTEpilogueStages>;

    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        OutputTileThreadMap,
        DtypeScale,
        cute::Stride<cute::_1, cute::_0, int64_t> >;
    using XScaleArguments = typename XScale::Arguments;

    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap,
        DtypeScale,
        cute::Stride<cute::_0, cute::_1, int64_t> >;
    using WScaleArguments = typename WScale::Arguments;

    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap,
        DtypeBias,
        cute::Stride<cute::_0, cute::_1, int64_t> >;
    using BiasArguments = typename Bias::Arguments;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    constexpr auto SplitKFactor = 1;

    XScaleArguments x_scale_arguments{
        (DtypeScale*)x_scale.data_ptr(),
        DtypeScale(1),
        {cute::_1{}, cute::_0{}, problem_size.m()}
    };

    WScaleArguments w_scale_arguments{
        (DtypeScale*)w_scale.data_ptr(),
        DtypeScale(1),
        {cute::_0{}, cute::_1{}, problem_size.n()}
    };

    BiasArguments bias_arguments{
        bias.has_value() ? reinterpret_cast<DtypeBias*>(bias -> data_ptr()) : nullptr,
        DtypeBias(1),
        {cute::_0{}, cute::_1{}, problem_size.n()}
    };

    using ApplyXScale = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        DtypeEpilogue,
        DtypeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTApplyXScale = cutlass::epilogue::threadblock::Sm80EVT<
        ApplyXScale,
        Accum,
        XScale>;
    
    using ApplyWScale = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        DtypeEpilogue,
        DtypeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTApplyWScale = cutlass::epilogue::threadblock::Sm80EVT<
        ApplyWScale,
        EVTApplyXScale,
        WScale>;

    using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus,
        DtypeEpilogue,
        DtypeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
        ApplyBias,
        EVTApplyWScale,
        Bias>;

    using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap,
        DtypeOutput,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t> >;

    using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
        Output,
        EVTApplyBias>;

    using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        DtypeA, LayoutInputA, cutlass::ComplexTransform::kNone, AlignmentInputA,
        DtypeB, LayoutInputB, cutlass::ComplexTransform::kNone, AlignmentInputB,
        DtypeOutput, LayoutOutput, AlignmentOutput,
        DtypeAccum, DtypeEpilogue, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape,
        EVTOutput,
        ThreadblockSwizzle,
        NumStages,
        Operator,
        NumEVTEpilogueStages
    >::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

    auto out = torch::empty({problem_size.m(), problem_size.n()},
            torch::TensorOptions().dtype(torch::kBFloat16).device(XQ.device()));

    typename Output::Arguments output_arguments{
        (DtypeOutput*)out.data_ptr(),
        {problem_size.n(), cute::_1{}, problem_size.mn().product()}
    };

    typename EVTOutput::Arguments callback_arguments{
        {
            {
                {
                    {},                 // Accum
                    x_scale_arguments,  // XScale
                    {},                 // ApplyXScale
                },                      // EVTApplyXScale
                w_scale_arguments,      // WScale
                {}                      // ApplyWScale
            },                          // EVTApplyWScale
            bias_arguments,             // Bias
            {}                          // ApplyBias
        },                              // EVTApplyBias
        output_arguments                // output
    };

    typename Gemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        SplitKFactor,
        callback_arguments,
        (DtypeA*)XQ.data_ptr(),
        (DtypeB*)WQ.data_ptr(),
        nullptr,
        nullptr,
        problem_size.mk().product(),
        problem_size.nk().product(),
        0,
        0,
        problem_size.k(),
        problem_size.k(),
        0,
        0);

    Gemm gemm;

    CUTLASS_CHECK(gemm.can_implement(arguments));

    
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    //torch::Tensor workspace;
    /*if(workspace_size > 0){
        workspace = torch.empty({static_cast<int64_t>(workspace_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(XQ.device()));
    }*/

    auto workspace = XQ.new_empty(
        {static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8));

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr()));
    CUTLASS_CHECK(gemm());
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}

int main(){
    constexpr int M = 32;
    constexpr int N = 64;
    constexpr int K = 128;

    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // host
    std::vector<DtypeA> h_A(M * K);
    std::vector<DtypeA> h_B(K * N);
    std::vector<float> h_x_scale(M);
    std::vector<float> h_w_scale(N);
    std::vector<float> h_bias(N);
    std::vector<DtypeBias> h_bias_bf16(N);
    std::vector<DtypeOutput> h_out(M * N);

    std::mt19937 gen(2026);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.01f,0.10f);
    std::uniform_real_distribution<float> bias_dist(-0.5f, 0.5f);

    for(int m = 0; m < M; m ++){
        for(int k = 0; k < K; k ++){
            h_A[m * K + k] = DtypeA(data_dist(gen));
        }
    }

    for(int n = 0; n < N; n ++){
        for(int k = 0; k < K; k ++){
            h_B[n * K + k] = DtypeB(data_dist(gen));
        }
    }

    for(int m = 0; m < M; m ++){
        h_x_scale[m] = scale_dist(gen);
    }

    for(int n = 0; n < N; n ++){
        h_w_scale[n] = scale_dist(gen);
        h_bias[n] = bias_dist(gen);
        h_bias_bf16[n] = DtypeBias(h_bias[n]);
    }

    std::vector<DtypeOutput> h_ref = cpu_ref(h_A, h_B, h_x_scale, h_w_scale, h_bias, M, N, K);

    // ---------------- CUDA torch::Tensor ----------------
    auto fp8_opts = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

    torch::Tensor XQ = torch::empty({M, K}, fp8_opts);
    torch::Tensor WQ = torch::empty({K, N}, fp8_opts);

    torch::Tensor x_scale = torch::empty({M}, fp32_opts);
    torch::Tensor w_scale = torch::empty({N}, fp32_opts);
    torch::Tensor bias = torch::empty({N}, bf16_opts);

    CUDA_CHECK(cudaMemcpy(XQ.data_ptr(), h_A.data(), 
        sizeof(DtypeA) * h_A.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(WQ.data_ptr(), h_B.data(), 
        sizeof(DtypeB) * h_B.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(x_scale.data_ptr(), h_x_scale.data(), 
        sizeof(float) * h_x_scale.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(w_scale.data_ptr(), h_w_scale.data(), 
        sizeof(float) * h_w_scale.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(bias.data_ptr(), h_bias_bf16.data(),
        sizeof(DtypeBias) * h_bias_bf16.size(), cudaMemcpyHostToDevice));

    std::cout << "XQ shape/stride = ["
            << XQ.size(0) << ", " << XQ.size(1) << "] / ["
            << XQ.stride(0) << ", " << XQ.stride(1) << "]" << std::endl;

    std::cout << "WQ shape/stride = ["
            << WQ.size(0) << ", " << WQ.size(1) << "] / ["
            << WQ.stride(0) << ", " << WQ.stride(1) << "]" << std::endl;

    torch::Tensor out = my_f8f8bf16_rowwise(XQ, WQ, std::optional<torch::Tensor>(bias),
        x_scale, w_scale, false);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), out.data_ptr(), 
        sizeof(DtypeOutput) * h_out.size(), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    double mean_diff = 0.0;
    int max_idx = -1;

    for(int i = 0; i < M * N; i ++){
        float gpu = static_cast<float>(h_out[i]);
        float cpu = static_cast<float>(h_ref[i]);
        float diff = std::abs(gpu - cpu);

        mean_diff += diff;
        if(diff > max_diff){
            max_diff = diff;
            max_idx = i;
        }
    }

    mean_diff /= static_cast<double>(M * N);

    std::cout << "max diff  = " << max_diff << std::endl;
    std::cout << "mean diff = " << mean_diff << std::endl;

    if (max_idx >= 0) {
        int m = max_idx / N;
        int n = max_idx % N;
        std::cout << "max diff at (" << m << ", " << n << ")"
              << ", gpu=" << static_cast<float>(h_out[max_idx])
              << ", ref=" << static_cast<float>(h_ref[max_idx])
              << std::endl;
    }

    constexpr float atol = 1e-2f;
    bool pass = max_diff <= atol;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    return pass ? 0 : 1;
}

/*
nvcc my_fp8_rowwise.cu \
    -std=c++17 \
    -arch=sm_89 \
    -I/root/LLMQRT-main/runtime_refact/3rdparty/cutlass/include \
    -I/root/LLMQRT-main/runtime_refact/3rdparty/cutlass/tools/util/include \
    -I/root/miniconda3/lib/python3.12/site-packages/torch/include \
    -I/root/miniconda3/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
    -I/root/miniconda3/include/python3.12 \
    -L/root/miniconda3/lib/python3.12/site-packages/torch/lib \
    -ltorch \
    -ltorch_cpu \
    -lc10 \
    -ltorch_cuda \
    -lc10_cuda \
    -lcudart \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -o my_fp8_rowwise
*/