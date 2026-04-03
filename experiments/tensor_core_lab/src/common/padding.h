#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace uipc::tensor_core_lab
{
size_t square_storage_size(int physical_dim, int batch_count);
size_t vector_storage_size(int physical_dim, int batch_count);
void   zero_square_storage(std::vector<double>& dst, int physical_dim, int batch_count);
void   zero_vector_storage(std::vector<double>& dst, int physical_dim, int batch_count);

template <typename Derived>
void store_square_padded(const Eigen::MatrixBase<Derived>& src,
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
                batch_offset + static_cast<size_t>(col) * physical_rows + row;
            if(row < logical_rows && col < logical_cols)
                dst[idx] = src(row, col);
            else
                dst[idx] = (row == col) ? 1.0 : 0.0;
        }
    }
}

template <typename Derived>
void store_square_padded_zero_tail(const Eigen::MatrixBase<Derived>& src,
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
                batch_offset + static_cast<size_t>(col) * physical_rows + row;
            if(row < logical_rows && col < logical_cols)
                dst[idx] = src(row, col);
            else
                dst[idx] = 0.0;
        }
    }
}

template <typename Derived>
void store_vector_padded(const Eigen::MatrixBase<Derived>& src,
                         int                               logical_dim,
                         int                               physical_dim,
                         std::vector<double>&              dst,
                         int                               batch_index)
{
    const size_t batch_offset =
        static_cast<size_t>(batch_index) * static_cast<size_t>(physical_dim);
    for(int row = 0; row < physical_dim; ++row)
    {
        dst[batch_offset + row] = (row < logical_dim) ? src(row) : 0.0;
    }
}
}  // namespace uipc::tensor_core_lab
