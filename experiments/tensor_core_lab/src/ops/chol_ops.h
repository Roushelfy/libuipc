#pragma once

#include <tcl/backend_api.h>

namespace uipc::tensor_core_lab
{
int spd_block_size(int physical_dim) noexcept;

void factorize_spd(BackendContext& context,
                   int             physical_dim,
                   int             batch_count,
                   const float*    source_matrix,
                   float*          factor_matrix);

void factorize_spd(BackendContext& context,
                   int             physical_dim,
                   int             batch_count,
                   const double*   source_matrix,
                   double*         factor_matrix);

void inverse_from_factor(BackendContext& context,
                         int             physical_dim,
                         int             batch_count,
                         const float*    factor_matrix,
                         float*          inv_l_matrix,
                         float*          inverse_matrix);

void inverse_from_factor(BackendContext& context,
                         int             physical_dim,
                         int             batch_count,
                         const double*   factor_matrix,
                         double*         inv_l_matrix,
                         double*         inverse_matrix);

void solve_from_factor(int           physical_dim,
                       BackendContext& context,
                       int           batch_count,
                       const float*  factor_matrix,
                       const float*  rhs_vector,
                       float*        inv_l_matrix,
                       float*        temp_vector,
                       float*        solution_vector);

void solve_from_factor(int             physical_dim,
                       BackendContext& context,
                       int             batch_count,
                       const double*   factor_matrix,
                       const double*   rhs_vector,
                       double*         inv_l_matrix,
                       double*         temp_vector,
                       double*         solution_vector);
}  // namespace uipc::tensor_core_lab
