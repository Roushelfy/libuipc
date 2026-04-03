#include <tcl/backend_api.h>

#include <tcl/device_info.h>

#include "../common/cuda_check.h"

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
                                              int               batch_count);

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
                                              int               batch_count);

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
                                         int                 batch_count);

bool        tc32_tf32_supported(const DeviceInfo& device_info) noexcept;
std::string tc32_tf32_unsupported_reason();
}  // namespace uipc::tensor_core_lab::detail

namespace uipc::tensor_core_lab
{
BackendContext::BackendContext(Mode mode)
    : m_mode(mode)
    , m_device_info(query_device_info())
{
    TCL_CUBLAS_CHECK(cublasCreate(&m_handle));
    TCL_CUBLAS_CHECK(cublasLtCreate(&m_lt_handle));
    m_lt_workspace.resize(4u << 20);
}

BackendContext::~BackendContext()
{
    if(m_lt_handle)
        cublasLtDestroy(m_lt_handle);
    if(m_handle)
        cublasDestroy(m_handle);
}

bool BackendContext::is_supported() const noexcept
{
    if(m_mode == Mode::Tc32Tf32)
        return detail::tc32_tf32_supported(m_device_info);
    return true;
}

std::string BackendContext::unsupported_reason() const
{
    if(is_supported())
        return {};
    if(m_mode == Mode::Tc32Tf32)
        return detail::tc32_tf32_unsupported_reason();
    return "unsupported backend mode";
}

void BackendContext::reset_trace() noexcept
{
    m_trace = {};
}

void BackendContext::set_trace(ImplPath               impl_path,
                               bool                   tensor_core_requested,
                               TensorCoreVerification tensor_core_verified) noexcept
{
    m_trace.impl_path             = impl_path;
    m_trace.tensor_core_requested = tensor_core_requested;
    m_trace.tensor_core_verified  = tensor_core_verified;
}

cublasStatus_t BackendContext::gemm_strided_batched(cublasOperation_t op_a,
                                                    cublasOperation_t op_b,
                                                    int               m,
                                                    int               n,
                                                    int               k,
                                                    const double*     alpha,
                                                    const double*     a,
                                                    int               lda,
                                                    long long         stride_a,
                                                    const double*     b,
                                                    int               ldb,
                                                    long long         stride_b,
                                                    const double*     beta,
                                                    double*           c,
                                                    int               ldc,
                                                    long long         stride_c,
                                                    int               batch_count)
{
    return cublasDgemmStridedBatched(m_handle,
                                     op_a,
                                     op_b,
                                     m,
                                     n,
                                     k,
                                     alpha,
                                     a,
                                     lda,
                                     stride_a,
                                     b,
                                     ldb,
                                     stride_b,
                                     beta,
                                     c,
                                     ldc,
                                     stride_c,
                                     batch_count);
}

cublasStatus_t BackendContext::gemm_strided_batched(cublasOperation_t op_a,
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
    if(m_mode == Mode::Tc32Tf32)
    {
        return detail::gemm_strided_batched_tc32_tf32(m_handle,
                                                      op_a,
                                                      op_b,
                                                      m,
                                                      n,
                                                      k,
                                                      alpha,
                                                      a,
                                                      lda,
                                                      stride_a,
                                                      b,
                                                      ldb,
                                                      stride_b,
                                                      beta,
                                                      c,
                                                      ldc,
                                                      stride_c,
                                                      batch_count);
    }

    return detail::gemm_strided_batched_fp32_simt(m_handle,
                                                  op_a,
                                                  op_b,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  a,
                                                  lda,
                                                  stride_a,
                                                  b,
                                                  ldb,
                                                  stride_b,
                                                  beta,
                                                  c,
                                                  ldc,
                                                  stride_c,
                                                  batch_count);
}

cublasStatus_t BackendContext::tc_blas_matmul_strided_batched(cublasOperation_t op_a,
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
                                                              const float*      c,
                                                              int               ldc,
                                                              long long         stride_c,
                                                              float*            d,
                                                              int               ldd,
                                                              long long         stride_d,
                                                              int               batch_count)
{
    if(m_mode != Mode::Tc32Tf32)
        return CUBLAS_STATUS_NOT_SUPPORTED;

    const bool ran_lt = detail::lt_matmul_strided_batched_tc32_tf32(m_lt_handle,
                                                                     m_lt_workspace.data(),
                                                                     m_lt_workspace.size(),
                                                                     op_a,
                                                                     op_b,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     alpha,
                                                                     a,
                                                                     lda,
                                                                     stride_a,
                                                                     b,
                                                                     ldb,
                                                                     stride_b,
                                                                     beta,
                                                                     c,
                                                                     ldc,
                                                                     stride_c,
                                                                     d,
                                                                     ldd,
                                                                     stride_d,
                                                                     batch_count);
    if(ran_lt)
        return CUBLAS_STATUS_SUCCESS;

    return detail::gemm_strided_batched_tc32_tf32(m_handle,
                                                  op_a,
                                                  op_b,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  a,
                                                  lda,
                                                  stride_a,
                                                  b,
                                                  ldb,
                                                  stride_b,
                                                  beta,
                                                  d,
                                                  ldd,
                                                  stride_d,
                                                  batch_count);
}
}  // namespace uipc::tensor_core_lab
