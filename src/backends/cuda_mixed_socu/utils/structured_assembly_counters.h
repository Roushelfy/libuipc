#pragma once

#include <uipc/common/type_define.h>

namespace uipc::backend::cuda_mixed
{
enum class StructuredAssemblyCounterSlot : IndexT
{
    ContactDiagScalarWrite             = 0,
    ContactFirstOffdiagScalarWrite     = 1,
    ContactOffBandScalarDrop           = 2,
    ContactNearBandPair                = 3,
    ContactOffBandPair                 = 4,
    ContactOffBandDiagFallbackStencil  = 5,
    ContactOffBandLumpFallbackStencil  = 6,
    NativeChainBaseSameBlockDenseHit   = 7,
    NativeChainBaseSameBlockDenseMiss  = 8,
    NativeChainBaseScalarFallback      = 9,
};

inline constexpr SizeT kStructuredAssemblyCounterCount = 10;
}  // namespace uipc::backend::cuda_mixed
