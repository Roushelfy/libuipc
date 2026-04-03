#include "chol_ops.h"

#include <tcl/runner.h>

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
__global__ void solve_factor_rhs_kernel(const T* factor,
                                        const T* rhs,
                                        T*       solution,
                                        int      physical_dim,
                                        int      batch_count)
{
    const int batch = blockIdx.x;
    if(batch >= batch_count)
        return;

    const T* l = factor + static_cast<size_t>(batch) * physical_dim * physical_dim;
    const T* b = rhs + static_cast<size_t>(batch) * physical_dim;
    T*       x = solution + static_cast<size_t>(batch) * physical_dim;

    for(int i = 0; i < physical_dim; ++i)
    {
        T sum = b[i];
        for(int k = 0; k < i; ++k)
            sum -= l[static_cast<size_t>(k) * physical_dim + i] * x[k];
        x[i] = sum / l[static_cast<size_t>(i) * physical_dim + i];
    }

    for(int i = physical_dim - 1; i >= 0; --i)
    {
        T sum = x[i];
        for(int k = i + 1; k < physical_dim; ++k)
            sum -= l[static_cast<size_t>(i) * physical_dim + k] * x[k];
        x[i] = sum / l[static_cast<size_t>(i) * physical_dim + i];
    }
}

template <typename T>
void solve_from_factor_baseline_impl(int physical_dim,
                                     int batch_count,
                                     const T* factor_matrix,
                                     const T* rhs_vector,
                                     T*       solution_vector)
{
    solve_factor_rhs_kernel<<<batch_count, 1>>>(
        factor_matrix, rhs_vector, solution_vector, physical_dim, batch_count);
    TCL_CUDA_CHECK(cudaGetLastError());
    TCL_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename PreparedT>
RunOutcome execute_solve_impl(BackendContext& context,
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
    solve_from_factor(data.spec.shape.physical_rows,
                      context,
                      data.spec.batch_count,
                      data.factor.data(),
                      data.rhs.data(),
                      data.inv_l.data(),
                      data.temp_vector.data(),
                      data.solution.data());

    if(measure_time)
        out.elapsed_ms = timer.stop_elapsed_ms();
    out.trace = context.trace();
    return out;
}
}  // namespace

void solve_from_factor(int           physical_dim,
                       BackendContext& context,
                       int           batch_count,
                       const float*  factor_matrix,
                       const float*  rhs_vector,
                       float*        inv_l_matrix,
                       float*        temp_vector,
                       float*        solution_vector)
{
    (void)context;
    (void)inv_l_matrix;
    (void)temp_vector;
    solve_from_factor_baseline_impl(
        physical_dim, batch_count, factor_matrix, rhs_vector, solution_vector);
}

void solve_from_factor(int             physical_dim,
                       BackendContext& context,
                       int             batch_count,
                       const double*   factor_matrix,
                       const double*   rhs_vector,
                       double*         inv_l_matrix,
                       double*         temp_vector,
                       double*         solution_vector)
{
    (void)context;
    (void)inv_l_matrix;
    (void)temp_vector;
    solve_from_factor_baseline_impl(
        physical_dim, batch_count, factor_matrix, rhs_vector, solution_vector);
}

RunOutcome execute_spd_solve_case(BackendContext&    context,
                                  PreparedSpdCaseF32& data,
                                  bool               measure_time)
{
    return execute_solve_impl(context, data, measure_time);
}

RunOutcome execute_spd_solve_case(BackendContext&    context,
                                  PreparedSpdCaseF64& data,
                                  bool               measure_time)
{
    return execute_solve_impl(context, data, measure_time);
}
}  // namespace uipc::tensor_core_lab
