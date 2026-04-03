#pragma once
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace uipc::tensor_core_lab::detail
{
inline void throw_if_cuda_failed(cudaError_t status, const char* expr)
{
    if(status != cudaSuccess)
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(status));
}

inline void throw_if_cublas_failed(cublasStatus_t status, const char* expr)
{
    if(status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(expr) + ": cublas failure");
}
}  // namespace uipc::tensor_core_lab::detail

#define TCL_CUDA_CHECK(expr) ::uipc::tensor_core_lab::detail::throw_if_cuda_failed((expr), #expr)
#define TCL_CUBLAS_CHECK(expr) ::uipc::tensor_core_lab::detail::throw_if_cublas_failed((expr), #expr)
