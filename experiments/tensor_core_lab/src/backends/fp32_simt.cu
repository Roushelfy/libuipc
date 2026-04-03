#include <tcl/backend_api.h>

namespace uipc::tensor_core_lab::detail
{
cublasStatus_t gemm_strided_batched_fp32_simt(cublasHandle_t   handle,
                                              cublasOperation_t op_a,
                                              cublasOperation_t op_b,
                                              int               m,
                                              int               n,
                                              int               k,
                                              const float*      alpha,
                                              const float*      a,
                                              int               lda,
                                              long long         stride_a,
                                              const float*      b,
                                              int               ldb,
                                              long long         stride_b,
                                              const float*      beta,
                                              float*            c,
                                              int               ldc,
                                              long long         stride_c,
                                              int               batch_count)
{
    return cublasGemmStridedBatchedEx(handle,
                                      op_a,
                                      op_b,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      a,
                                      CUDA_R_32F,
                                      lda,
                                      stride_a,
                                      b,
                                      CUDA_R_32F,
                                      ldb,
                                      stride_b,
                                      beta,
                                      c,
                                      CUDA_R_32F,
                                      ldc,
                                      stride_c,
                                      batch_count,
                                      CUBLAS_COMPUTE_32F_PEDANTIC,
                                      CUBLAS_GEMM_DEFAULT);
}
}  // namespace uipc::tensor_core_lab::detail
