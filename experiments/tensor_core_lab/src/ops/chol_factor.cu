#include "chol_ops.h"

#include <tcl/runner.h>

#include <mma.h>

#include <string>
#include <vector>

#include "../common/cuda_check.h"
#include "../common/event_timer.h"

namespace uipc::tensor_core_lab
{
namespace
{
using namespace nvcuda;

template <typename T>
__global__ void copy_buffer_kernel(T* dst, const T* src, size_t count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < count)
        dst[idx] = src[idx];
}

template <typename T>
__global__ void potrf_block_kernel(T* matrix,
                                   int physical_dim,
                                   int block_offset,
                                   int block_size,
                                   int batch_count,
                                   int* status)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    T* a = matrix + static_cast<size_t>(batch) * physical_dim * physical_dim;

    for(int j = 0; j < block_size; ++j)
    {
        T diag = a[static_cast<size_t>(block_offset + j) * physical_dim + (block_offset + j)];
        for(int k = 0; k < j; ++k)
        {
            const T l =
                a[static_cast<size_t>(block_offset + k) * physical_dim + (block_offset + j)];
            diag -= l * l;
        }

        if(diag <= T(0))
        {
            status[batch] = 1;
            return;
        }

        const T ljj = sqrt(diag);
        a[static_cast<size_t>(block_offset + j) * physical_dim + (block_offset + j)] = ljj;

        for(int i = j + 1; i < block_size; ++i)
        {
            T value =
                a[static_cast<size_t>(block_offset + j) * physical_dim + (block_offset + i)];
            for(int k = 0; k < j; ++k)
            {
                value -= a[static_cast<size_t>(block_offset + k) * physical_dim
                           + (block_offset + i)]
                         * a[static_cast<size_t>(block_offset + k) * physical_dim
                             + (block_offset + j)];
            }
            a[static_cast<size_t>(block_offset + j) * physical_dim + (block_offset + i)] =
                value / ljj;
        }
    }
}

template <typename T>
__global__ void trsm_right_lower_transpose_kernel(T* matrix,
                                                  int physical_dim,
                                                  int row_offset,
                                                  int col_offset,
                                                  int block_size,
                                                  int batch_count)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    T* a = matrix + static_cast<size_t>(batch) * physical_dim * physical_dim;

    for(int col = 0; col < block_size; ++col)
    {
        const T diag =
            a[static_cast<size_t>(col_offset + col) * physical_dim + (col_offset + col)];
        for(int row = 0; row < block_size; ++row)
        {
            T value =
                a[static_cast<size_t>(col_offset + col) * physical_dim + (row_offset + row)];
            for(int k = 0; k < col; ++k)
            {
                value -= a[static_cast<size_t>(col_offset + k) * physical_dim
                           + (row_offset + row)]
                         * a[static_cast<size_t>(col_offset + k) * physical_dim
                             + (col_offset + col)];
            }
            a[static_cast<size_t>(col_offset + col) * physical_dim + (row_offset + row)] =
                value / diag;
        }
    }
}

template <typename T>
__global__ void zero_upper_kernel(T* matrix, int physical_dim, int batch_count)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    T* a = matrix + static_cast<size_t>(batch) * physical_dim * physical_dim;
    for(int col = 0; col < physical_dim; ++col)
    {
        for(int row = 0; row < col; ++row)
            a[static_cast<size_t>(col) * physical_dim + row] = T(0);
    }
}

template <typename T>
void copy_device_buffer(T* dst, const T* src, size_t count)
{
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((count + threads - 1) / threads);
    copy_buffer_kernel<<<blocks, threads>>>(dst, src, count);
    TCL_CUDA_CHECK(cudaGetLastError());
}

std::vector<int> download_factor_status(const DeviceBuffer<int>& device_status)
{
    std::vector<int> host(device_status.size());
    device_status.copy_to_host(std::span<int>(host.data(), host.size()));
    return host;
}

void check_factor_status(const DeviceBuffer<int>& device_status)
{
    const auto host = download_factor_status(device_status);
    for(const int s : host)
    {
        if(s != 0)
            throw std::runtime_error("SPD factorization failed: matrix is not numerically SPD");
    }
}

__global__ void chol16_wmma_kernel(const float* source_matrix,
                                   float*       factor_matrix,
                                   int          batch_count,
                                   int*         status)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    constexpr int dim        = 16;
    constexpr int half       = 8;
    constexpr int tile_size  = dim * dim;
    const int     tid        = threadIdx.x;

    __shared__ float s_a[tile_size];
    __shared__ float s_l10_a[dim * half];
    __shared__ float s_l10_t[half * dim];
    __shared__ float s_update[tile_size];

    const float* src = source_matrix + static_cast<size_t>(batch) * tile_size;

    for(int i = tid; i < tile_size; i += blockDim.x)
    {
        s_a[i]      = src[i];
        s_update[i] = 0.0f;
    }
    __syncthreads();

    if(tid == 0)
    {
        for(int j = 0; j < half; ++j)
        {
            float diag = s_a[static_cast<size_t>(j) * dim + j];
            for(int k = 0; k < j; ++k)
            {
                const float l = s_a[static_cast<size_t>(k) * dim + j];
                diag -= l * l;
            }

            if(diag <= 0.0f)
            {
                status[batch] = 1;
                return;
            }

            const float ljj = sqrtf(diag);
            s_a[static_cast<size_t>(j) * dim + j] = ljj;

            for(int i = j + 1; i < half; ++i)
            {
                float value = s_a[static_cast<size_t>(j) * dim + i];
                for(int k = 0; k < j; ++k)
                {
                    value -= s_a[static_cast<size_t>(k) * dim + i]
                             * s_a[static_cast<size_t>(k) * dim + j];
                }
                s_a[static_cast<size_t>(j) * dim + i] = value / ljj;
            }
        }

        for(int col = 0; col < half; ++col)
        {
            const float diag = s_a[static_cast<size_t>(col) * dim + col];
            for(int row = 0; row < half; ++row)
            {
                float value = s_a[static_cast<size_t>(col) * dim + (half + row)];
                for(int k = 0; k < col; ++k)
                {
                    value -= s_a[static_cast<size_t>(k) * dim + (half + row)]
                             * s_a[static_cast<size_t>(k) * dim + col];
                }
                s_a[static_cast<size_t>(col) * dim + (half + row)] = value / diag;
            }
        }
    }
    __syncthreads();

    for(int i = tid; i < dim * half; i += blockDim.x)
        s_l10_a[i] = 0.0f;
    for(int i = tid; i < half * dim; i += blockDim.x)
        s_l10_t[i] = 0.0f;
    __syncthreads();

    for(int col = tid; col < half; col += blockDim.x)
    {
        for(int row = 0; row < half; ++row)
        {
            const float value = wmma::__float_to_tf32(
                s_a[static_cast<size_t>(col) * dim + (half + row)]);
            s_l10_a[static_cast<size_t>(col) * dim + (half + row)] = value;
            s_l10_t[static_cast<size_t>(col) * dim + (half + row)] = value;
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, dim, dim, half, wmma::precision::tf32, wmma::col_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, dim, dim, half, wmma::precision::tf32, wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, dim, dim, half, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::load_matrix_sync(a_frag, s_l10_a, dim);
    wmma::load_matrix_sync(b_frag, s_l10_t, dim);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    wmma::store_matrix_sync(s_update, acc, dim, wmma::mem_col_major);
    __syncthreads();

    if(tid == 0)
    {
        for(int col = 0; col < half; ++col)
        {
            for(int row = col; row < half; ++row)
            {
                s_a[static_cast<size_t>(half + col) * dim + (half + row)] -=
                    s_update[static_cast<size_t>(half + col) * dim + (half + row)];
            }
        }

        for(int j = 0; j < half; ++j)
        {
            float diag = s_a[static_cast<size_t>(half + j) * dim + (half + j)];
            for(int k = 0; k < j; ++k)
            {
                const float l =
                    s_a[static_cast<size_t>(half + k) * dim + (half + j)];
                diag -= l * l;
            }

            if(diag <= 0.0f)
            {
                status[batch] = 1;
                return;
            }

            const float ljj = sqrtf(diag);
            s_a[static_cast<size_t>(half + j) * dim + (half + j)] = ljj;

            for(int i = j + 1; i < half; ++i)
            {
                float value = s_a[static_cast<size_t>(half + j) * dim + (half + i)];
                for(int k = 0; k < j; ++k)
                {
                    value -= s_a[static_cast<size_t>(half + k) * dim + (half + i)]
                             * s_a[static_cast<size_t>(half + k) * dim + (half + j)];
                }
                s_a[static_cast<size_t>(half + j) * dim + (half + i)] = value / ljj;
            }
        }

        for(int col = 0; col < dim; ++col)
        {
            for(int row = 0; row < dim; ++row)
            {
                if(row < col)
                    s_a[static_cast<size_t>(col) * dim + row] = 0.0f;
            }
        }
    }
    __syncthreads();

    float* dst = factor_matrix + static_cast<size_t>(batch) * tile_size;
    for(int i = tid; i < tile_size; i += blockDim.x)
        dst[i] = s_a[i];
#else
    (void)source_matrix;
    (void)factor_matrix;
    (void)batch_count;
    (void)status;
#endif
}

template <typename T>
void factorize_spd_baseline_impl(BackendContext& context,
                                 int             physical_dim,
                                 int             batch_count,
                                 const T*        source_matrix,
                                 T*              factor_matrix)
{
    if(physical_dim != 16 && physical_dim != 48)
        throw std::runtime_error("factorize_spd only supports physical_dim 16 or 48");

    const size_t matrix_elements =
        static_cast<size_t>(physical_dim) * physical_dim * static_cast<size_t>(batch_count);
    copy_device_buffer(factor_matrix, source_matrix, matrix_elements);

    DeviceBuffer<int> status(static_cast<size_t>(batch_count));
    TCL_CUDA_CHECK(cudaMemset(status.data(), 0, status.size() * sizeof(int)));

    const int block = (physical_dim == 48) ? 16 : 16;
    for(int k = 0; k < physical_dim; k += block)
    {
        potrf_block_kernel<<<batch_count, 1>>>(
            factor_matrix, physical_dim, k, block, batch_count, status.data());
        TCL_CUDA_CHECK(cudaGetLastError());
        TCL_CUDA_CHECK(cudaDeviceSynchronize());
        check_factor_status(status);

        for(int row = k + block; row < physical_dim; row += block)
        {
            trsm_right_lower_transpose_kernel<<<batch_count, 1>>>(
                factor_matrix, physical_dim, row, k, block, batch_count);
            TCL_CUDA_CHECK(cudaGetLastError());
        }
        TCL_CUDA_CHECK(cudaDeviceSynchronize());

        const T alpha_neg = T(-1);
        const T beta_one  = T(1);
        for(int i = k + block; i < physical_dim; i += block)
        {
            const T* lik = factor_matrix + static_cast<size_t>(k) * physical_dim + i;
            T*       aii = factor_matrix + static_cast<size_t>(i) * physical_dim + i;

            TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_N,
                                                          CUBLAS_OP_T,
                                                          block,
                                                          block,
                                                          block,
                                                          &alpha_neg,
                                                          lik,
                                                          physical_dim,
                                                          static_cast<long long>(physical_dim)
                                                              * physical_dim,
                                                          lik,
                                                          physical_dim,
                                                          static_cast<long long>(physical_dim)
                                                              * physical_dim,
                                                          &beta_one,
                                                          aii,
                                                          physical_dim,
                                                          static_cast<long long>(physical_dim)
                                                              * physical_dim,
                                                          batch_count));

            for(int j = i + block; j < physical_dim; j += block)
            {
                const T* ljk = factor_matrix + static_cast<size_t>(k) * physical_dim + j;
                T*       aji = factor_matrix + static_cast<size_t>(i) * physical_dim + j;

                TCL_CUBLAS_CHECK(context.gemm_strided_batched(
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    block,
                    block,
                    block,
                    &alpha_neg,
                    ljk,
                    physical_dim,
                    static_cast<long long>(physical_dim) * physical_dim,
                    lik,
                    physical_dim,
                    static_cast<long long>(physical_dim) * physical_dim,
                    &beta_one,
                    aji,
                    physical_dim,
                    static_cast<long long>(physical_dim) * physical_dim,
                    batch_count));
            }
        }
    }

    zero_upper_kernel<<<batch_count, 1>>>(factor_matrix, physical_dim, batch_count);
    TCL_CUDA_CHECK(cudaGetLastError());
    TCL_CUDA_CHECK(cudaDeviceSynchronize());
}

bool factorize_spd_tc_wmma_16_impl(int          batch_count,
                                   const float* source_matrix,
                                   float*       factor_matrix)
{
    DeviceBuffer<int> status(static_cast<size_t>(batch_count));
    TCL_CUDA_CHECK(cudaMemset(status.data(), 0, status.size() * sizeof(int)));
    chol16_wmma_kernel<<<batch_count, 32>>>(
        source_matrix, factor_matrix, batch_count, status.data());
    TCL_CUDA_CHECK(cudaGetLastError());
    TCL_CUDA_CHECK(cudaDeviceSynchronize());
    const auto host = download_factor_status(status);
    for(const int s : host)
    {
        if(s != 0)
            return false;
    }
    return true;
}

}  // namespace

int spd_block_size(int physical_dim) noexcept
{
    return (physical_dim == 48) ? 16 : 16;
}

void factorize_spd(BackendContext& context,
                   int             physical_dim,
                   int             batch_count,
                   const float*    source_matrix,
                   float*          factor_matrix)
{
    if(context.mode() == Mode::Tc32Tf32 && physical_dim == 16)
    {
        if(factorize_spd_tc_wmma_16_impl(batch_count, source_matrix, factor_matrix))
        {
            context.reset_trace();
            context.set_trace(ImplPath::TcWmma, true, TensorCoreVerification::Yes);
            return;
        }
    }

    if(context.mode() == Mode::Tc32Tf32)
    {
        context.reset_trace();
        context.set_trace(ImplPath::Baseline,
                          true,
                          TensorCoreVerification::BlockedByPermissions);
        factorize_spd_baseline_impl(
            context, physical_dim, batch_count, source_matrix, factor_matrix);
        return;
    }

    context.reset_trace();
    context.set_trace(ImplPath::Baseline, false, TensorCoreVerification::No);
    factorize_spd_baseline_impl(
        context, physical_dim, batch_count, source_matrix, factor_matrix);
}

void factorize_spd(BackendContext& context,
                   int             physical_dim,
                   int             batch_count,
                   const double*   source_matrix,
                   double*         factor_matrix)
{
    context.reset_trace();
    context.set_trace(ImplPath::Baseline, false, TensorCoreVerification::No);
    factorize_spd_baseline_impl(
        context, physical_dim, batch_count, source_matrix, factor_matrix);
}

namespace
{
template <typename PreparedT>
RunOutcome execute_factorize_impl(BackendContext& context,
                                  PreparedT&      data,
                                  bool            measure_time)
{
    RunOutcome out;
    out.status = RunStatus::Ok;

    EventTimer timer;
    if(measure_time)
        timer.start();

    factorize_spd(context,
                  data.spec.shape.physical_rows,
                  data.spec.batch_count,
                  data.source_matrix.data(),
                  data.factor.data());

    if(measure_time)
        out.elapsed_ms = timer.stop_elapsed_ms();

    out.trace = context.trace();
    return out;
}
}  // namespace

RunOutcome execute_spd_factorize_case(BackendContext&    context,
                                      PreparedSpdCaseF32& data,
                                      bool               measure_time)
{
    return execute_factorize_impl(context, data, measure_time);
}

RunOutcome execute_spd_factorize_case(BackendContext&    context,
                                      PreparedSpdCaseF64& data,
                                      bool               measure_time)
{
    return execute_factorize_impl(context, data, measure_time);
}
}  // namespace uipc::tensor_core_lab
