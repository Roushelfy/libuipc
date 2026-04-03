#include "padding.h"

namespace uipc::tensor_core_lab
{
size_t square_storage_size(int physical_dim, int batch_count)
{
    return static_cast<size_t>(physical_dim) * static_cast<size_t>(physical_dim)
           * static_cast<size_t>(batch_count);
}

size_t vector_storage_size(int physical_dim, int batch_count)
{
    return static_cast<size_t>(physical_dim) * static_cast<size_t>(batch_count);
}

void zero_square_storage(std::vector<double>& dst, int physical_dim, int batch_count)
{
    dst.assign(square_storage_size(physical_dim, batch_count), 0.0);
}

void zero_vector_storage(std::vector<double>& dst, int physical_dim, int batch_count)
{
    dst.assign(vector_storage_size(physical_dim, batch_count), 0.0);
}
}  // namespace uipc::tensor_core_lab
