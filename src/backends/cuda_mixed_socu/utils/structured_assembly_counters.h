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
    NativeChainBasePair3x3SameBlockHit = 13,
    NativeChainBasePair3x3AdjacentHit  = 14,
    NativeChainBasePair3x3Miss         = 15,
};

inline constexpr SizeT kStructuredAssemblyCounterCount = 16;
}  // namespace uipc::backend::cuda_mixed
