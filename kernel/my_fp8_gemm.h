#pragma once

#include <torch/types.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <optional>
#include <torch/torch.h>
#include <torch/extension.h>

#include <c10/macros/Macros.h>

torch::Tensor my_f8f8bf16_tensorwise(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    float alpha,
    float beta,
    const std::string& dtype_str);