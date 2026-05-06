#pragma once

namespace uipc::backend::cuda_mixed
{
enum class StructuredContactOffbandPolicy : unsigned char
{
    Drop,
    Diag,
    DiagLump,
};
}  // namespace uipc::backend::cuda_mixed
