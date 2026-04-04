#include <tcg/gemm_case.h>

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

#include "rng.h"

namespace uipc::tensor_core_gemm
{
namespace
{
using ::uipc::tensor_core_lab::make_rng;
using ::uipc::tensor_core_lab::uniform_real;

size_t matrix_storage_size(int rows, int cols, int batch_count)
{
    return static_cast<size_t>(rows) * static_cast<size_t>(cols)
           * static_cast<size_t>(batch_count);
}

template <typename Derived>
void store_matrix_zero_tail(const Eigen::MatrixBase<Derived>& src,
                            int                               logical_rows,
                            int                               logical_cols,
                            int                               physical_rows,
                            int                               physical_cols,
                            std::vector<double>&              dst,
                            int                               batch_index)
{
    const size_t batch_offset =
        static_cast<size_t>(batch_index) * static_cast<size_t>(physical_rows)
        * static_cast<size_t>(physical_cols);

    for(int col = 0; col < physical_cols; ++col)
    {
        for(int row = 0; row < physical_rows; ++row)
        {
            const size_t idx =
                batch_offset + static_cast<size_t>(col) * static_cast<size_t>(physical_rows)
                + static_cast<size_t>(row);
            if(row < logical_rows && col < logical_cols)
                dst[idx] = src(row, col);
            else
                dst[idx] = 0.0;
        }
    }
}

Eigen::MatrixXd make_matrix(int rows, int cols, int seed, int sequence, int salt, double scale)
{
    Eigen::MatrixXd out(rows, cols);
    auto            rng = make_rng(seed, sequence, salt);
    for(int col = 0; col < cols; ++col)
    {
        for(int row = 0; row < rows; ++row)
        {
            out(row, col) = uniform_real(rng, -1.0, 1.0) * scale;
        }
    }
    return out;
}
}  // namespace

int round_up_to_multiple_of_16(int value) noexcept
{
    return ((value + 15) / 16) * 16;
}

GemmCaseData make_gemm_case(const GemmShape&   shape,
                            const std::string& shape_group,
                            GemmLayoutVariant  layout_variant,
                            int                batch_count,
                            int                seed)
{
    GemmCaseData out;
    out.spec.shape_tag       = std::to_string(shape.m) + "x" + std::to_string(shape.n) + "x"
                         + std::to_string(shape.k);
    out.spec.shape_group     = shape_group;
    out.spec.layout_variant  = layout_variant;
    out.spec.batch_count     = batch_count;
    out.spec.seed            = seed;
    out.spec.m               = shape.m;
    out.spec.n               = shape.n;
    out.spec.k               = shape.k;
    out.spec.physical_m      = (layout_variant == GemmLayoutVariant::Padded)
                                   ? round_up_to_multiple_of_16(shape.m)
                                   : shape.m;
    out.spec.physical_n      = (layout_variant == GemmLayoutVariant::Padded)
                                   ? round_up_to_multiple_of_16(shape.n)
                                   : shape.n;
    out.spec.physical_k      = (layout_variant == GemmLayoutVariant::Padded)
                                   ? round_up_to_multiple_of_16(shape.k)
                                   : shape.k;

    out.a_fp64.resize(matrix_storage_size(
        out.spec.physical_m, out.spec.physical_k, out.spec.batch_count));
    out.b_fp64.resize(matrix_storage_size(
        out.spec.physical_k, out.spec.physical_n, out.spec.batch_count));

    const double scale = 1.0 / std::sqrt(static_cast<double>(std::max(shape.k, 1)));
    for(int batch = 0; batch < batch_count; ++batch)
    {
        const Eigen::MatrixXd a =
            make_matrix(shape.m, shape.k, seed, batch, 11, scale);
        const Eigen::MatrixXd b =
            make_matrix(shape.k, shape.n, seed, batch, 29, scale);

        store_matrix_zero_tail(a,
                               shape.m,
                               shape.k,
                               out.spec.physical_m,
                               out.spec.physical_k,
                               out.a_fp64,
                               batch);
        store_matrix_zero_tail(b,
                               shape.k,
                               shape.n,
                               out.spec.physical_k,
                               out.spec.physical_n,
                               out.b_fp64,
                               batch);
    }

    return out;
}
}  // namespace uipc::tensor_core_gemm
