#pragma once

namespace uipc::backend::cuda_mixed
{
enum class MixedPrecisionLevel
{
    FP64,
    Path1,
    Path2,
    Path3,
    Path4
};

#if defined(UIPC_MIXED_LEVEL_PATH4)
inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path4;
#elif defined(UIPC_MIXED_LEVEL_PATH3)
inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path3;
#elif defined(UIPC_MIXED_LEVEL_PATH2)
inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path2;
#elif defined(UIPC_MIXED_LEVEL_PATH1)
inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path1;
#else
inline constexpr auto kBuildLevel = MixedPrecisionLevel::FP64;
#endif
}  // namespace uipc::backend::cuda_mixed

