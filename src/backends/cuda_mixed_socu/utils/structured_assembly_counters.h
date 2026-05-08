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
    NativeChainBaseAdjacentDenseHit    = 8,
    NativeChainBaseDenseMiss           = 9,
    NativeChainBaseScalarFallback      = 10,
    NativeChainBaseDiag3x3Hit          = 11,
    NativeChainBaseDiag3x3Miss         = 12,
};

inline constexpr SizeT kStructuredAssemblyCounterCount = 13;
}  // namespace uipc::backend::cuda_mixed
