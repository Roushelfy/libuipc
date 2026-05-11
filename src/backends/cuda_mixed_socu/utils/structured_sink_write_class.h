#pragma once

namespace uipc::backend::cuda_mixed
{
enum class StructuredSinkWriteClass : unsigned char
{
    Skipped,
    Diag,
    FirstOffdiag,
    OffBand,
};
}  // namespace uipc::backend::cuda_mixed
