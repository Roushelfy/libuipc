#pragma once

#include <string>
#include <string_view>
#include <uipc/common/type_define.h>
#include <vector>

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
    SocuRuntimeArtifactUnavailable,
    OrderingQualityTooLow,
    ContactOffBandRatioTooHigh,
    StructuredProviderMissing,
    StubNoDirection,
    DirectionInvalid,
    SocuRuntimeError,
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

struct SocuApproxBlockLayout
{
    SizeT block       = 0;
    SizeT chain_begin = 0;
    SizeT chain_end   = 0;
    SizeT dof_offset  = 0;
    SizeT dof_count   = 0;
};

struct SocuApproxDryRunReport
{
    bool        packed = false;
    std::string mode = "structured_dry_run";
    std::string report_path;
    std::string ordering_report_path;
    std::string contact_report_path;

    SizeT block_size            = 0;
    SizeT block_count           = 0;
    SizeT chain_atom_count      = 0;
    SizeT ordering_dof_count    = 0;
    SizeT structured_slot_count = 0;
    SizeT padding_slot_count    = 0;
    SizeT active_rhs_scalar_count = 0;
    SizeT rhs_scalar_count      = 0;
    double block_utilization    = 0.0;

    SizeT diag_block_count           = 0;
    SizeT first_offdiag_block_count  = 0;
    SizeT diag_scalar_count          = 0;
    SizeT first_offdiag_scalar_count = 0;
    SizeT diag_nonzero_count         = 0;
    SizeT first_offdiag_nonzero_count = 0;
    SizeT rhs_nonzero_count          = 0;

    SizeT near_band_contact_count              = 0;
    SizeT off_band_contact_count               = 0;
    SizeT near_band_contribution_count         = 0;
    SizeT off_band_contribution_count          = 0;
    SizeT absorbed_hessian_contribution_count = 0;
    SizeT dropped_hessian_contribution_count  = 0;
    double contribution_near_band_ratio        = 0.0;
    double contribution_off_band_ratio         = 0.0;
    double weighted_near_band_ratio            = 0.0;
    double weighted_off_band_ratio             = 0.0;
    SizeT structured_diag_write_count          = 0;
    SizeT structured_first_offdiag_write_count = 0;
    SizeT structured_off_band_drop_count       = 0;
    double structured_diag_contact_abs_sum      = 0.0;
    double structured_first_offdiag_contact_abs_sum = 0.0;
    double structured_off_band_drop_abs_sum     = 0.0;
    double rhs_abs_sum                         = 0.0;

    double dry_run_pack_time_ms = 0.0;
    double socu_factor_solve_time_ms = 0.0;
    double scatter_time_ms = 0.0;

    double damping_shift = 0.0;
    double surrogate_residual = 0.0;
    double surrogate_relative_residual = 0.0;
    double descent_dot = 0.0;
    double gradient_norm = 0.0;
    double direction_norm = 0.0;

    bool        direction_available = false;
    std::string status_reason;
    std::string status_detail;

    std::vector<SocuApproxBlockLayout> blocks;
};
}  // namespace uipc::backend::cuda_mixed
