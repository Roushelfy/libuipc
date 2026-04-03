#include <tcl/backend_api.h>

#include <tcl/device_info.h>

namespace uipc::tensor_core_lab::detail
{
bool tc32_tf32_supported(const DeviceInfo& device_info) noexcept
{
    return supports_tf32_tensor_cores(device_info);
}

std::string tc32_tf32_unsupported_reason()
{
    return "tc32_tf32 requires SM80+";
}

namespace
{
int64_t rows_for_layout(cublasOperation_t op, int rows_if_n, int rows_if_t)
{
    return (op == CUBLAS_OP_N) ? rows_if_n : rows_if_t;
}

int64_t cols_for_layout(cublasOperation_t op, int cols_if_n, int cols_if_t)
{
    return (op == CUBLAS_OP_N) ? cols_if_n : cols_if_t;
}
}  // namespace

bool lt_matmul_strided_batched_tc32_tf32(cublasLtHandle_t    handle,
                                         const void*         workspace,
                                         size_t              workspace_bytes,
                                         cublasOperation_t   op_a,
                                         cublasOperation_t   op_b,
                                         int                 m,
                                         int                 n,
                                         int                 k,
                                         const float*        alpha,
                                         const float*        a,
                                         int                 lda,
                                         long long           stride_a,
                                         const float*        b,
                                         int                 ldb,
                                         long long           stride_b,
                                         const float*        beta,
                                         const float*        c,
                                         int                 ldc,
                                         long long           stride_c,
                                         float*              d,
                                         int                 ldd,
                                         long long           stride_d,
                                         int                 batch_count)
{
    cublasLtMatmulDesc_t        op_desc    = nullptr;
    cublasLtMatrixLayout_t      a_desc     = nullptr;
    cublasLtMatrixLayout_t      b_desc     = nullptr;
    cublasLtMatrixLayout_t      c_desc     = nullptr;
    cublasLtMatrixLayout_t      d_desc     = nullptr;
    cublasLtMatmulPreference_t  preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic_result{};
    int                         return_count = 0;
    bool                        success      = false;

    do
    {
        if(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F)
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatmulDescSetAttribute(
               op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a))
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatmulDescSetAttribute(
               op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b))
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatrixLayoutCreate(&a_desc,
                                      CUDA_R_32F,
                                      rows_for_layout(op_a, m, k),
                                      cols_for_layout(op_a, k, m),
                                      lda)
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutCreate(&b_desc,
                                      CUDA_R_32F,
                                      rows_for_layout(op_b, k, n),
                                      cols_for_layout(op_b, n, k),
                                      ldb)
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, ldc) != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, m, n, ldd) != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatrixLayoutSetAttribute(
               a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutSetAttribute(a_desc,
                                            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                            &stride_a,
                                            sizeof(stride_a))
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatrixLayoutSetAttribute(
               b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutSetAttribute(b_desc,
                                            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                            &stride_b,
                                            sizeof(stride_b))
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatrixLayoutSetAttribute(
               c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutSetAttribute(c_desc,
                                            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                            &stride_c,
                                            sizeof(stride_c))
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatrixLayoutSetAttribute(
               d_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))
           != CUBLAS_STATUS_SUCCESS)
            break;
        if(cublasLtMatrixLayoutSetAttribute(d_desc,
                                            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                            &stride_d,
                                            sizeof(stride_d))
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatmulPreferenceSetAttribute(preference,
                                                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &workspace_bytes,
                                                sizeof(workspace_bytes))
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatmulAlgoGetHeuristic(handle,
                                          op_desc,
                                          a_desc,
                                          b_desc,
                                          c_desc,
                                          d_desc,
                                          preference,
                                          1,
                                          &heuristic_result,
                                          &return_count)
           != CUBLAS_STATUS_SUCCESS)
            break;

        if(return_count == 0 || heuristic_result.state != CUBLAS_STATUS_SUCCESS)
            break;

        if(cublasLtMatmul(handle,
                          op_desc,
                          alpha,
                          a,
                          a_desc,
                          b,
                          b_desc,
                          beta,
                          c,
                          c_desc,
                          d,
                          d_desc,
                          &heuristic_result.algo,
                          const_cast<void*>(workspace),
                          workspace_bytes,
                          nullptr)
           != CUBLAS_STATUS_SUCCESS)
            break;

        success = true;
    } while(false);

    if(preference)
        cublasLtMatmulPreferenceDestroy(preference);
    if(d_desc)
        cublasLtMatrixLayoutDestroy(d_desc);
    if(c_desc)
        cublasLtMatrixLayoutDestroy(c_desc);
    if(b_desc)
        cublasLtMatrixLayoutDestroy(b_desc);
    if(a_desc)
        cublasLtMatrixLayoutDestroy(a_desc);
    if(op_desc)
        cublasLtMatmulDescDestroy(op_desc);
    return success;
}

cublasStatus_t gemm_strided_batched_tc32_tf32(cublasHandle_t   handle,
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
                                      CUBLAS_COMPUTE_32F_FAST_TF32,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
}  // namespace uipc::tensor_core_lab::detail
