#pragma once
#include <span>

namespace uipc::tensor_core_lab
{
struct MatrixMetrics
{
    double rel_fro        = 0.0;
    double abs_linf       = 0.0;
    double symmetry_error = 0.0;
    int    nan_inf_count  = 0;
};

MatrixMetrics compare_square_batches(std::span<const double> reference,
                                     std::span<const double> candidate,
                                     int                     logical_dim,
                                     int                     physical_dim,
                                     int                     batch_count);

MatrixMetrics compare_matrix_batches(std::span<const double> reference,
                                     std::span<const double> candidate,
                                     int                     logical_rows,
                                     int                     logical_cols,
                                     int                     physical_rows,
                                     int                     physical_cols,
                                     int                     batch_count);

MatrixMetrics compare_vector_batches(std::span<const double> reference,
                                     std::span<const double> candidate,
                                     int                     logical_dim,
                                     int                     physical_dim,
                                     int                     batch_count);

MatrixMetrics factorization_reconstruction_metrics(std::span<const double> source,
                                                   std::span<const double> factor,
                                                   int                     logical_dim,
                                                   int                     physical_dim,
                                                   int                     batch_count);
}  // namespace uipc::tensor_core_lab
