#pragma once

namespace uipc::tensor_core_lab
{
struct TensorShape
{
    int logical_rows   = 0;
    int logical_cols   = 0;
    int physical_rows  = 0;
    int physical_cols  = 0;
};

inline constexpr int padded_square_dim(int logical_dim) noexcept
{
    return logical_dim <= 16 ? 16 : 32;
}
}  // namespace uipc::tensor_core_lab
