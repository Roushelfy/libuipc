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
    OrderingInvalid,
    UnsupportedPrecisionContract,
    UnsupportedBlockSize,
    SocuMathDxUnsupported,
    SocuRuntimeArtifactUnavailable,
    StructuredProviderMissing,
    StructuredCoverageInvalid,
    StructuredSubsystemUnsupported,
    DirectionInvalid,
    SocuRuntimeError,
    LineSearchRejected,
};

std::string_view to_string(SocuApproxGateReason reason) noexcept;

struct SocuApproxGateReport
{
    bool                 passed = false;
    SocuApproxGateReason reason = SocuApproxGateReason::SocuDisabled;
    std::string          detail;
    std::string          ordering_report_path;
    std::string          provider_kind;
    std::string          structured_scope = "multi_provider";
    SizeT                block_size = 0;
    double               near_band_ratio = 0.0;
    double               off_band_ratio  = 0.0;
    double               block_utilization = 0.0;
    double               off_band_drop_norm_ratio = 0.0;
    SizeT                coverage_active_dof_count = 0;
    SizeT                coverage_padding_dof_count = 0;
    bool                 complete_dof_coverage = false;
    std::string          dtype;
    std::string          resolved_backend;
    std::string          resolved_perf_backend;
    std::string          resolved_math_mode;
    std::string          resolved_graph_mode;
    std::string          mathdx_manifest_path;
    std::string          mathdx_runtime_cache_dir;
    bool                 mathdx_manifest_ok = false;
    bool                 mathdx_artifacts_ok = false;
    bool                 mathdx_prebuilt_cubin_ok = false;
    bool                 debug_validation_enabled = false;
    bool                 debug_timing_enabled = false;
};

struct SocuApproxBlockLayout
{
    SizeT block       = 0;
    SizeT chain_begin = 0;
    SizeT chain_end   = 0;
    SizeT dof_offset  = 0;
    SizeT dof_count   = 0;
};

struct SocuApproxSolveReport
{
    bool        packed = false;
    std::string report_path;
    std::string ordering_report_path;
    std::string provider_kind;
    std::string structured_scope = "multi_provider";

    SizeT block_size            = 0;
    SizeT block_count           = 0;
    SizeT chain_atom_count      = 0;
    SizeT ordering_dof_count    = 0;
    SizeT structured_slot_count = 0;
    SizeT padding_slot_count    = 0;
    SizeT active_rhs_scalar_count = 0;
    SizeT rhs_scalar_count      = 0;
    double block_utilization    = 0.0;
    double near_band_ratio = 0.0;
    double off_band_ratio = 0.0;
    double off_band_drop_norm_ratio = 0.0;
    double min_block_utilization = 0.0;
    double min_near_band_ratio = 0.0;
    double max_off_band_ratio = 1.0;
    double max_off_band_drop_norm_ratio = 1.0;
    bool complete_dof_coverage = false;
    SizeT coverage_active_dof_count = 0;
    SizeT coverage_padding_dof_count = 0;

    SizeT diag_block_count           = 0;
    SizeT first_offdiag_block_count  = 0;
    SizeT diag_scalar_count          = 0;
    SizeT first_offdiag_scalar_count = 0;
    SizeT diag_nonzero_count         = 0;
    SizeT first_offdiag_nonzero_count = 0;
    SizeT first_offdiag_nonzero_index_sum = 0;
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

    double socu_factor_solve_time_ms = 0.0;
    double scatter_time_ms = 0.0;
    std::string stream_source;
    bool plan_created_this_solve = false;
    bool debug_validation_enabled = false;
    bool debug_timing_enabled = false;
    bool report_each_solve = false;

    double damping_shift = 0.0;
    double surrogate_residual = 0.0;
    double surrogate_relative_residual = 0.0;
    double descent_dot = 0.0;
    double gradient_norm = 0.0;
    double direction_norm = 0.0;
    double direction_min_abs_threshold = 0.0;
    double direction_min_rel_threshold = 0.0;
    std::string rhs_sign_convention = "rhs_is_global_b";

    bool line_search_feedback_available = false;
    bool line_search_accepted = true;
    bool line_search_hit_max_iter = false;
    SizeT line_search_iteration_count = 0;
    double line_search_accepted_alpha = 1.0;
    SizeT line_search_reject_streak = 0;

    bool        direction_available = false;
    std::string status_reason;
    std::string status_detail;

    std::vector<SocuApproxBlockLayout> blocks;
};

void write_solve_report(const SocuApproxSolveReport& report);
}  // namespace uipc::backend::cuda_mixed
