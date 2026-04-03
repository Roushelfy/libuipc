#include "chol_ops.h"

#include <tcl/runner.h>

#include <type_traits>

#include "../common/cuda_check.h"
#include "../common/event_timer.h"

namespace uipc::tensor_core_lab
{
namespace
{
template <typename T>
__global__ void invert_lower_kernel(const T* factor,
                                    T*       inv_l,
                                    int      physical_dim,
                                    int      batch_count)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    const T* l = factor + static_cast<size_t>(batch) * physical_dim * physical_dim;
    T*       y = inv_l + static_cast<size_t>(batch) * physical_dim * physical_dim;

    for(int col = 0; col < physical_dim; ++col)
    {
        for(int row = 0; row < physical_dim; ++row)
            y[static_cast<size_t>(col) * physical_dim + row] = T(0);

        y[static_cast<size_t>(col) * physical_dim + col] =
            T(1) / l[static_cast<size_t>(col) * physical_dim + col];

        for(int row = col + 1; row < physical_dim; ++row)
        {
            T sum = T(0);
            for(int k = col; k < row; ++k)
            {
                sum += l[static_cast<size_t>(k) * physical_dim + row]
                       * y[static_cast<size_t>(col) * physical_dim + k];
            }
            y[static_cast<size_t>(col) * physical_dim + row] =
                -sum / l[static_cast<size_t>(row) * physical_dim + row];
        }
    }
}

template <typename T>
__global__ void symmetrize_square_kernel(T* matrix, int physical_dim, int batch_count)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    T* a = matrix + static_cast<size_t>(batch) * physical_dim * physical_dim;
    for(int col = 0; col < physical_dim; ++col)
    {
        for(int row = 0; row < col; ++row)
        {
            const size_t idx0 = static_cast<size_t>(col) * physical_dim + row;
            const size_t idx1 = static_cast<size_t>(row) * physical_dim + col;
            const T avg = T(0.5) * (a[idx0] + a[idx1]);
            a[idx0] = avg;
            a[idx1] = avg;
        }
    }
}

template <typename T>
void invert_lower_from_factor_impl(int      physical_dim,
                                   int      batch_count,
                                   const T* factor_matrix,
                                   T*       inv_l_matrix)
{
    invert_lower_kernel<<<batch_count, 1>>>(
        factor_matrix, inv_l_matrix, physical_dim, batch_count);
    TCL_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void inverse_from_factor_impl(BackendContext& context,
                              int             physical_dim,
                              int             batch_count,
                              const T*        factor_matrix,
                              T*              inv_l_matrix,
                              T*              inverse_matrix)
{
    invert_lower_from_factor_impl(
        physical_dim, batch_count, factor_matrix, inv_l_matrix);

    const T alpha = T(1);
    const T beta  = T(0);
    const long long stride = static_cast<long long>(physical_dim) * physical_dim;

    if constexpr(std::is_same_v<T, float>)
    {
        (void)context;
    }

    TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  physical_dim,
                                                  physical_dim,
                                                  physical_dim,
                                                  &alpha,
                                                  inv_l_matrix,
                                                  physical_dim,
                                                  stride,
                                                  inv_l_matrix,
                                                  physical_dim,
                                                  stride,
                                                  &beta,
                                                  inverse_matrix,
                                                  physical_dim,
                                                  stride,
                                                  batch_count));
    symmetrize_square_kernel<<<batch_count, 1>>>(inverse_matrix, physical_dim, batch_count);
    TCL_CUDA_CHECK(cudaGetLastError());
    TCL_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename PreparedT>
RunOutcome execute_inverse_impl(BackendContext& context,
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
    inverse_from_factor(context,
                        data.spec.shape.physical_rows,
                        data.spec.batch_count,
                        data.factor.data(),
                        data.inv_l.data(),
                        data.inverse.data());

    if(measure_time)
        out.elapsed_ms = timer.stop_elapsed_ms();
    out.trace = context.trace();
    return out;
}
}  // namespace

void inverse_from_factor(BackendContext& context,
                         int             physical_dim,
                         int             batch_count,
                         const float*    factor_matrix,
                         float*          inv_l_matrix,
                         float*          inverse_matrix)
{
    inverse_from_factor_impl(
        context, physical_dim, batch_count, factor_matrix, inv_l_matrix, inverse_matrix);
}

void inverse_from_factor(BackendContext& context,
                         int             physical_dim,
                         int             batch_count,
                         const double*   factor_matrix,
                         double*         inv_l_matrix,
                         double*         inverse_matrix)
{
    inverse_from_factor_impl(
        context, physical_dim, batch_count, factor_matrix, inv_l_matrix, inverse_matrix);
}

RunOutcome execute_spd_inverse_case(BackendContext&    context,
                                    PreparedSpdCaseF32& data,
                                    bool               measure_time)
{
    return execute_inverse_impl(context, data, measure_time);
}

RunOutcome execute_spd_inverse_case(BackendContext&    context,
                                    PreparedSpdCaseF64& data,
                                    bool               measure_time)
{
    return execute_inverse_impl(context, data, measure_time);
}
}  // namespace uipc::tensor_core_lab
