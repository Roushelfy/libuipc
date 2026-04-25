#pragma once

#include <string>
#include <string_view>
#include <uipc/common/type_define.h>

namespace uipc::backend::cuda_mixed
{
enum class SocuApproxGateReason
{
    None,
    SocuDisabled,
    OrderingMissing,
    OrderingReportInvalid,
    UnsupportedPrecisionContract,
    UnsupportedBlockSize,
    SocuMathDxUnsupported,
    OrderingQualityTooLow,
    ContactOffBandRatioTooHigh,
    StructuredProviderMissing,
    StubNoDirection,
};

std::string_view to_string(SocuApproxGateReason reason) noexcept;

struct SocuApproxGateReport
{
    bool                 passed = false;
    SocuApproxGateReason reason = SocuApproxGateReason::SocuDisabled;
    std::string          detail;
    std::string          ordering_report_path;
    SizeT                block_size = 0;
    double               near_band_ratio = 0.0;
    double               off_band_ratio  = 0.0;
};
}  // namespace uipc::backend::cuda_mixed
