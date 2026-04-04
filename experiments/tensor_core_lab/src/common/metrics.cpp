#include <tcl/metrics.h>

#include <algorithm>
#include <cmath>

namespace uipc::tensor_core_lab
{
namespace
{
MatrixMetrics compare_square_like(std::span<const double> reference,
                                  std::span<const double> candidate,
                                  int                     logical_rows,
                                  int                     logical_cols,
                                  int                     physical_rows,
                                  int                     physical_cols,
                                  int                     batch_count)
{
    MatrixMetrics out{};
    const size_t  expected_size =
        static_cast<size_t>(physical_rows) * static_cast<size_t>(physical_cols)
        * static_cast<size_t>(batch_count);

    if(reference.size() != expected_size || candidate.size() != expected_size)
    {
        out.nan_inf_count = -1;
        return out;
    }

    double ref_norm_sq  = 0.0;
    double diff_norm_sq = 0.0;

    for(int b = 0; b < batch_count; ++b)
    {
        const size_t batch_offset =
            static_cast<size_t>(b) * static_cast<size_t>(physical_rows) * physical_cols;

        for(int col = 0; col < logical_cols; ++col)
        {
            for(int row = 0; row < logical_rows; ++row)
            {
                const size_t idx =
                    batch_offset + static_cast<size_t>(col) * physical_rows + row;
                const double ref = reference[idx];
                const double val = candidate[idx];

                ref_norm_sq += ref * ref;
                diff_norm_sq += (val - ref) * (val - ref);
                out.abs_linf = std::max(out.abs_linf, std::abs(val - ref));

                if(!std::isfinite(val))
                    ++out.nan_inf_count;

                if(logical_rows == logical_cols)
                {
                    const size_t transpose_idx =
                        batch_offset + static_cast<size_t>(row) * physical_rows + col;
                    out.symmetry_error = std::max(out.symmetry_error,
                                                  std::abs(val - candidate[transpose_idx]));
                }
            }
        }
    }

    const double ref_norm  = std::sqrt(ref_norm_sq);
    const double diff_norm = std::sqrt(diff_norm_sq);
    out.rel_fro            = ref_norm > 0.0 ? diff_norm / ref_norm : diff_norm;
    return out;
}

MatrixMetrics compare_vector_like(std::span<const double> reference,
                                  std::span<const double> candidate,
                                  int                     logical_dim,
                                  int                     physical_dim,
                                  int                     batch_count)
{
    MatrixMetrics out{};
    const size_t  expected_size =
        static_cast<size_t>(physical_dim) * static_cast<size_t>(batch_count);

    if(reference.size() != expected_size || candidate.size() != expected_size)
    {
        out.nan_inf_count = -1;
        return out;
    }

    double ref_norm_sq  = 0.0;
    double diff_norm_sq = 0.0;

    for(int b = 0; b < batch_count; ++b)
    {
        const size_t batch_offset = static_cast<size_t>(b) * static_cast<size_t>(physical_dim);
        for(int row = 0; row < logical_dim; ++row)
        {
            const size_t idx = batch_offset + row;
            const double ref = reference[idx];
            const double val = candidate[idx];

            ref_norm_sq += ref * ref;
            diff_norm_sq += (val - ref) * (val - ref);
            out.abs_linf = std::max(out.abs_linf, std::abs(val - ref));
            if(!std::isfinite(val))
                ++out.nan_inf_count;
        }
    }

    const double ref_norm  = std::sqrt(ref_norm_sq);
    const double diff_norm = std::sqrt(diff_norm_sq);
    out.rel_fro            = ref_norm > 0.0 ? diff_norm / ref_norm : diff_norm;
    return out;
}
}  // namespace

MatrixMetrics compare_square_batches(std::span<const double> reference,
                                     std::span<const double> candidate,
                                     int                     logical_dim,
                                     int                     physical_dim,
                                     int                     batch_count)
{
    return compare_square_like(reference,
                               candidate,
                               logical_dim,
                               logical_dim,
                               physical_dim,
                               physical_dim,
                               batch_count);
}

MatrixMetrics compare_matrix_batches(std::span<const double> reference,
                                     std::span<const double> candidate,
                                     int                     logical_rows,
                                     int                     logical_cols,
                                     int                     physical_rows,
                                     int                     physical_cols,
                                     int                     batch_count)
{
    return compare_square_like(reference,
                               candidate,
                               logical_rows,
                               logical_cols,
                               physical_rows,
                               physical_cols,
                               batch_count);
}

MatrixMetrics compare_vector_batches(std::span<const double> reference,
                                     std::span<const double> candidate,
                                     int                     logical_dim,
                                     int                     physical_dim,
                                     int                     batch_count)
{
    return compare_vector_like(reference,
                               candidate,
                               logical_dim,
                               physical_dim,
                               batch_count);
}

MatrixMetrics factorization_reconstruction_metrics(std::span<const double> source,
                                                   std::span<const double> factor,
                                                   int                     logical_dim,
                                                   int                     physical_dim,
                                                   int                     batch_count)
{
    MatrixMetrics out{};

    const size_t expected_size =
        static_cast<size_t>(physical_dim) * static_cast<size_t>(physical_dim)
        * static_cast<size_t>(batch_count);

    if(source.size() != expected_size || factor.size() != expected_size)
    {
        out.nan_inf_count = -1;
        return out;
    }

    double src_norm_sq  = 0.0;
    double diff_norm_sq = 0.0;

    for(int b = 0; b < batch_count; ++b)
    {
        const size_t batch_offset =
            static_cast<size_t>(b) * static_cast<size_t>(physical_dim) * physical_dim;

        for(int col = 0; col < logical_dim; ++col)
        {
            for(int row = 0; row < logical_dim; ++row)
            {
                double recon = 0.0;
                const int kmax = std::min(row, col);
                for(int k = 0; k <= kmax; ++k)
                {
                    const size_t idx_lhs =
                        batch_offset + static_cast<size_t>(k) * physical_dim + row;
                    const size_t idx_rhs =
                        batch_offset + static_cast<size_t>(k) * physical_dim + col;
                    recon += factor[idx_lhs] * factor[idx_rhs];
                }

                const size_t idx =
                    batch_offset + static_cast<size_t>(col) * physical_dim + row;
                const double ref = source[idx];
                const double diff = recon - ref;
                src_norm_sq += ref * ref;
                diff_norm_sq += diff * diff;
                out.abs_linf = std::max(out.abs_linf, std::abs(diff));
                if(!std::isfinite(recon))
                    ++out.nan_inf_count;
            }
        }
    }

    const double src_norm  = std::sqrt(src_norm_sq);
    const double diff_norm = std::sqrt(diff_norm_sq);
    out.rel_fro            = src_norm > 0.0 ? diff_norm / src_norm : diff_norm;
    return out;
}
}  // namespace uipc::tensor_core_lab
