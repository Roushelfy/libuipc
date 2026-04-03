#include <tcl/runner.h>

#include "../common/cuda_check.h"

namespace uipc::tensor_core_lab
{
namespace
{
template <typename T>
__global__ void add_inplace(T* dst, const T* src, size_t count)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < count)
        dst[i] += src[i];
}

template <typename T, typename PreparedCaseT>
RunOutcome execute_baseline_impl(BackendContext& context, PreparedCaseT& data, bool measure_time)
{
    if(!data.has_secondary)
        return {RunStatus::InvalidArgument, 0.0, {}, "joint24 expects secondary term"};

    RunOutcome out;
    const int dim = data.spec.shape.physical_rows;
    const long long stride =
        static_cast<long long>(dim) * static_cast<long long>(dim);
    const T alpha = T{1};
    const T beta  = T{0};

    context.reset_trace();
    context.set_trace(ImplPath::Baseline, false, TensorCoreVerification::No);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;

    if(measure_time)
    {
        TCL_CUDA_CHECK(cudaEventCreate(&start));
        TCL_CUDA_CHECK(cudaEventCreate(&stop));
        TCL_CUDA_CHECK(cudaEventRecord(start));
    }

    TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_N,
                                                  CUBLAS_OP_N,
                                                  dim,
                                                  dim,
                                                  dim,
                                                  &alpha,
                                                  data.middle.data(),
                                                  dim,
                                                  stride,
                                                  data.left.data(),
                                                  dim,
                                                  stride,
                                                  &beta,
                                                  data.temp0.data(),
                                                  dim,
                                                  stride,
                                                  data.spec.batch_count));

    TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  dim,
                                                  dim,
                                                  dim,
                                                  &alpha,
                                                  data.left.data(),
                                                  dim,
                                                  stride,
                                                  data.temp0.data(),
                                                  dim,
                                                  stride,
                                                  &beta,
                                                  data.output.data(),
                                                  dim,
                                                  stride,
                                                  data.spec.batch_count));

    TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_N,
                                                  CUBLAS_OP_N,
                                                  dim,
                                                  dim,
                                                  dim,
                                                  &alpha,
                                                  data.aux.data(),
                                                  dim,
                                                  stride,
                                                  data.right.data(),
                                                  dim,
                                                  stride,
                                                  &beta,
                                                  data.temp0.data(),
                                                  dim,
                                                  stride,
                                                  data.spec.batch_count));

    TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  dim,
                                                  dim,
                                                  dim,
                                                  &alpha,
                                                  data.right.data(),
                                                  dim,
                                                  stride,
                                                  data.temp0.data(),
                                                  dim,
                                                  stride,
                                                  &beta,
                                                  data.temp1.data(),
                                                  dim,
                                                  stride,
                                                  data.spec.batch_count));

    const size_t count = data.output.size();
    const int    block_size = 256;
    const int    grid_size =
        static_cast<int>((count + static_cast<size_t>(block_size) - 1) / block_size);
    add_inplace<<<grid_size, block_size>>>(data.output.data(), data.temp1.data(), count);
    TCL_CUDA_CHECK(cudaGetLastError());

    if(measure_time)
    {
        TCL_CUDA_CHECK(cudaEventRecord(stop));
        TCL_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        TCL_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        out.elapsed_ms = static_cast<double>(elapsed_ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    out.status = RunStatus::Ok;
    out.trace  = context.trace();
    return out;
}

RunOutcome execute_tc_blas_impl(BackendContext& context,
                                PreparedCaseF32& data,
                                bool             measure_time)
{
    if(!data.has_secondary)
        return {RunStatus::InvalidArgument, 0.0, {}, "joint24 expects secondary term"};

    RunOutcome out;
    const int dim = data.spec.shape.physical_rows;
    const long long stride =
        static_cast<long long>(dim) * static_cast<long long>(dim);
    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    context.reset_trace();
    context.set_trace(ImplPath::TcBlas, true, TensorCoreVerification::BlockedByPermissions);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;

    if(measure_time)
    {
        TCL_CUDA_CHECK(cudaEventCreate(&start));
        TCL_CUDA_CHECK(cudaEventCreate(&stop));
        TCL_CUDA_CHECK(cudaEventRecord(start));
    }

    TCL_CUBLAS_CHECK(context.tc_blas_matmul_strided_batched(CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            dim,
                                                            dim,
                                                            dim,
                                                            &alpha,
                                                            data.middle.data(),
                                                            dim,
                                                            stride,
                                                            data.left.data(),
                                                            dim,
                                                            stride,
                                                            &beta0,
                                                            data.temp0.data(),
                                                            dim,
                                                            stride,
                                                            data.temp0.data(),
                                                            dim,
                                                            stride,
                                                            data.spec.batch_count));

    TCL_CUBLAS_CHECK(context.tc_blas_matmul_strided_batched(CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            dim,
                                                            dim,
                                                            dim,
                                                            &alpha,
                                                            data.left.data(),
                                                            dim,
                                                            stride,
                                                            data.temp0.data(),
                                                            dim,
                                                            stride,
                                                            &beta0,
                                                            data.output.data(),
                                                            dim,
                                                            stride,
                                                            data.output.data(),
                                                            dim,
                                                            stride,
                                                            data.spec.batch_count));

    TCL_CUBLAS_CHECK(context.tc_blas_matmul_strided_batched(CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            dim,
                                                            dim,
                                                            dim,
                                                            &alpha,
                                                            data.aux.data(),
                                                            dim,
                                                            stride,
                                                            data.right.data(),
                                                            dim,
                                                            stride,
                                                            &beta0,
                                                            data.temp1.data(),
                                                            dim,
                                                            stride,
                                                            data.temp1.data(),
                                                            dim,
                                                            stride,
                                                            data.spec.batch_count));

    TCL_CUBLAS_CHECK(context.tc_blas_matmul_strided_batched(CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            dim,
                                                            dim,
                                                            dim,
                                                            &alpha,
                                                            data.right.data(),
                                                            dim,
                                                            stride,
                                                            data.temp1.data(),
                                                            dim,
                                                            stride,
                                                            &beta1,
                                                            data.output.data(),
                                                            dim,
                                                            stride,
                                                            data.output.data(),
                                                            dim,
                                                            stride,
                                                            data.spec.batch_count));

    if(measure_time)
    {
        TCL_CUDA_CHECK(cudaEventRecord(stop));
        TCL_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        TCL_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        out.elapsed_ms = static_cast<double>(elapsed_ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    out.status = RunStatus::Ok;
    out.trace  = context.trace();
    return out;
}
}  // namespace

RunOutcome execute_joint24_case(BackendContext& context,
                                PreparedCaseF32& data,
                                bool             measure_time)
{
    if(context.mode() == Mode::Tc32Tf32)
        return execute_tc_blas_impl(context, data, measure_time);
    return execute_baseline_impl<float>(context, data, measure_time);
}

RunOutcome execute_joint24_case(BackendContext& context,
                                PreparedCaseF64& data,
                                bool             measure_time)
{
    return execute_baseline_impl<double>(context, data, measure_time);
}
}  // namespace uipc::tensor_core_lab
