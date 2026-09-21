#pragma once

#include <torch/types.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/extension.h>

/*
torch::Tensor input,  // INT8
torch::Tensor weight, // INT8
torch::Tensor bias,   // BF16
float alpha,          // BF16
float beta            // BF16
*/
torch::Tensor cutlass_int8_gemm_per_tensor(torch::Tensor input, 
        torch::Tensor weight, torch::Tensor bias, float alpha, float beta);

torch::Tensor cutlass_int8_gemm_per_channel(torch::Tensor input, 
        torch::Tensor weight, torch::Tensor bias, float alpha, float beta);