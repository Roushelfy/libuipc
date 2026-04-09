#include <tcl/runner.h>

#include <mma.h>

#include "../common/cuda_check.h"

namespace uipc::tensor_core_lab
{
namespace
{
using namespace nvcuda;

template <typename T>
__global__ void symmetrize_square_kernel(T* matrix, int dim, int batch_count)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    T* a = matrix + static_cast<size_t>(batch) * dim * dim;
    for(int col = 0; col < dim; ++col)
    {
        for(int row = 0; row < col; ++row)
        {
            const size_t idx0 = static_cast<size_t>(col) * dim + row;
            const size_t idx1 = static_cast<size_t>(row) * dim + col;
            const T avg = T(0.5) * (a[idx0] + a[idx1]);
            a[idx0] = avg;
            a[idx1] = avg;
        }
    }
}

template <typename T, typename PreparedCaseT>
RunOutcome execute_baseline_impl(BackendContext& context, PreparedCaseT& data, bool measure_time)
{
    RunOutcome out;

    const int dim = data.spec.shape.physical_rows;
    if(dim != data.spec.shape.physical_cols)
        return {RunStatus::InvalidArgument, 0.0, {}, "fem12 expects square padded matrices"};

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

    symmetrize_square_kernel<<<data.spec.batch_count, 1>>>(
        data.output.data(), dim, data.spec.batch_count);
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
    RunOutcome out;

    const int dim = data.spec.shape.physical_rows;
    if(dim != data.spec.shape.physical_cols)
        return {RunStatus::InvalidArgument, 0.0, {}, "fem12 expects square padded matrices"};

    const long long stride =
        static_cast<long long>(dim) * static_cast<long long>(dim);
    const float alpha = 1.0f;
    const float beta0 = 0.0f;

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

    symmetrize_square_kernel<<<data.spec.batch_count, 1>>>(
        data.output.data(), dim, data.spec.batch_count);
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

__global__ void fem12_wmma_kernel(const float* left,
                                  const float* middle,
                                  float*       output,
                                  int          batch_count)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    constexpr int dim            = 16;
    constexpr int tile_k         = 8;
    constexpr int matrix_stride  = dim * dim;
    const int     tid            = threadIdx.x;

    __shared__ float s_left[matrix_stride];
    __shared__ float s_middle[matrix_stride];
    __shared__ float s_temp[matrix_stride];
    __shared__ float s_temp_tf32[matrix_stride];

    const float* left_batch   = left + static_cast<size_t>(batch) * matrix_stride;
    const float* middle_batch = middle + static_cast<size_t>(batch) * matrix_stride;

    for(int i = tid; i < matrix_stride; i += blockDim.x)
    {
        s_left[i]   = wmma::__float_to_tf32(left_batch[i]);
        s_middle[i] = wmma::__float_to_tf32(middle_batch[i]);
        s_temp[i]   = 0.0f;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, dim, dim, tile_k, float> acc0;
    wmma::fill_fragment(acc0, 0.0f);

    for(int k0 = 0; k0 < dim; k0 += tile_k)
    {
        wmma::fragment<wmma::matrix_a, dim, dim, tile_k, wmma::precision::tf32, wmma::col_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, dim, dim, tile_k, wmma::precision::tf32, wmma::col_major>
            b_frag;

        wmma::load_matrix_sync(a_frag, s_middle + static_cast<size_t>(k0) * dim, dim);
        wmma::load_matrix_sync(b_frag, s_left + k0, dim);
        wmma::mma_sync(acc0, a_frag, b_frag, acc0);
    }

    wmma::store_matrix_sync(s_temp, acc0, dim, wmma::mem_col_major);
    __syncthreads();

    for(int i = tid; i < matrix_stride; i += blockDim.x)
        s_temp_tf32[i] = wmma::__float_to_tf32(s_temp[i]);
    __syncthreads();

    wmma::fragment<wmma::accumulator, dim, dim, tile_k, float> acc1;
    wmma::fill_fragment(acc1, 0.0f);

    for(int k0 = 0; k0 < dim; k0 += tile_k)
    {
        wmma::fragment<wmma::matrix_a, dim, dim, tile_k, wmma::precision::tf32, wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, dim, dim, tile_k, wmma::precision::tf32, wmma::col_major>
            b_frag;

        wmma::load_matrix_sync(a_frag, s_left + k0, dim);
        wmma::load_matrix_sync(b_frag, s_temp_tf32 + k0, dim);
        wmma::mma_sync(acc1, a_frag, b_frag, acc1);
    }

    wmma::store_matrix_sync(output + static_cast<size_t>(batch) * matrix_stride,
                            acc1,
                            dim,
                            wmma::mem_col_major);
#else
    (void)left;
    (void)middle;
    (void)output;
    (void)batch_count;
#endif
}

RunOutcome execute_tc_wmma_impl(BackendContext& context,
                                PreparedCaseF32& data,
                                bool             measure_time)
{
    RunOutcome out;

    const int dim = data.spec.shape.physical_rows;
    if(dim != data.spec.shape.physical_cols || dim != 16)
        return {RunStatus::InvalidArgument,
                0.0,
                {},
                "fem12 tc_wmma expects 16x16 padded matrices"};

    context.reset_trace();
    context.set_trace(ImplPath::TcWmma, true, TensorCoreVerification::Yes);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;

    if(measure_time)
    {
        TCL_CUDA_CHECK(cudaEventCreate(&start));
        TCL_CUDA_CHECK(cudaEventCreate(&stop));
        TCL_CUDA_CHECK(cudaEventRecord(start));
    }

    fem12_wmma_kernel<<<data.spec.batch_count, 32>>>(
        data.left.data(), data.middle.data(), data.output.data(), data.spec.batch_count);
    TCL_CUDA_CHECK(cudaGetLastError());

    symmetrize_square_kernel<<<data.spec.batch_count, 1>>>(
        data.output.data(), dim, data.spec.batch_count);
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
}  // namespace

RunOutcome execute_fem12_case(BackendContext& context,
                              PreparedCaseF32& data,
                              bool             measure_time)
{
    if(context.mode() == Mode::Tc32Tf32)
        return execute_tc_wmma_impl(context, data, measure_time);
    return execute_baseline_impl<float>(context, data, measure_time);
}

RunOutcome execute_fem12_case(BackendContext& context,
                              PreparedCaseF64& data,
                              bool             measure_time)
{
    return execute_baseline_impl<double>(context, data, measure_time);
}

RunOutcome execute_fem12_case_blas(BackendContext& context,
                                   PreparedCaseF32& data,
                                   bool             measure_time)
{
    if(context.mode() == Mode::Tc32Tf32)
        return execute_tc_blas_impl(context, data, measure_time);
    return execute_baseline_impl<float>(context, data, measure_time);
}

RunOutcome execute_fem12_case_blas(BackendContext& context,
                                   PreparedCaseF64& data,
                                   bool             measure_time)
{
    return execute_baseline_impl<double>(context, data, measure_time);
}
}  // namespace uipc::tensor_core_lab
