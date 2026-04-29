#include <linear_system/socu_approx_report.h>

#include <uipc/common/exception.h>
#include <uipc/common/json.h>

#include <fmt/format.h>

#include <filesystem>
#include <fstream>

namespace uipc::backend::cuda_mixed
{
namespace
{
namespace fs = std::filesystem;

Json to_json(const SocuApproxSolveReport& report)
{
    Json blocks = Json::array();
    for(const auto& block : report.blocks)
    {
        blocks.push_back(Json{{"block", block.block},
                              {"chain_begin", block.chain_begin},
                              {"chain_end", block.chain_end},
                              {"dof_offset", block.dof_offset},
                              {"dof_count", block.dof_count}});
    }

    return Json{{"solver", "socu_approx"},
                {"packed", report.packed},
                {"ordering_report", report.ordering_report_path},
                {"provider_kind", report.provider_kind},
                {"structured_scope", report.structured_scope},
                {"block_size", report.block_size},
                {"chain_atom_count", report.chain_atom_count},
                {"ordering_dof_count", report.ordering_dof_count},
                {"structured_slot_count", report.structured_slot_count},
                {"padding_slot_count", report.padding_slot_count},
                {"block_utilization", report.block_utilization},
                {"gates",
                 {{"near_band_ratio", report.near_band_ratio},
                  {"off_band_ratio", report.off_band_ratio},
                  {"off_band_drop_norm_ratio", report.off_band_drop_norm_ratio},
                  {"min_block_utilization", report.min_block_utilization},
                  {"min_near_band_ratio", report.min_near_band_ratio},
                  {"max_off_band_ratio", report.max_off_band_ratio},
                  {"max_off_band_drop_norm_ratio",
                   report.max_off_band_drop_norm_ratio},
                  {"complete_dof_coverage", report.complete_dof_coverage},
                  {"coverage_active_dof_count",
                   report.coverage_active_dof_count},
                  {"coverage_padding_dof_count",
                   report.coverage_padding_dof_count}}},
                {"layout",
                 {{"block_count", report.block_count},
                  {"diag_block_count", report.diag_block_count},
                  {"first_offdiag_block_count", report.first_offdiag_block_count},
                  {"active_rhs_scalar_count", report.active_rhs_scalar_count},
                  {"rhs_scalar_count", report.rhs_scalar_count},
                  {"diag_scalar_count", report.diag_scalar_count},
                  {"first_offdiag_scalar_count", report.first_offdiag_scalar_count},
                  {"rhs_nonzero_count", report.rhs_nonzero_count},
                  {"diag_nonzero_count", report.diag_nonzero_count},
                  {"first_offdiag_nonzero_count",
                   report.first_offdiag_nonzero_count},
                  {"first_offdiag_nonzero_index_sum",
                   report.first_offdiag_nonzero_index_sum},
                  {"blocks", blocks}}},
                {"contact",
                 {{"near_band_contact_count", report.near_band_contact_count},
                  {"off_band_contact_count", report.off_band_contact_count},
                  {"near_band_contribution_count", report.near_band_contribution_count},
                  {"off_band_contribution_count", report.off_band_contribution_count},
                  {"absorbed_hessian_contribution_count",
                   report.absorbed_hessian_contribution_count},
                  {"dropped_hessian_contribution_count",
                   report.dropped_hessian_contribution_count},
                  {"contribution_near_band_ratio",
                   report.contribution_near_band_ratio},
                  {"contribution_off_band_ratio",
                   report.contribution_off_band_ratio},
                  {"weighted_near_band_ratio", report.weighted_near_band_ratio},
                  {"weighted_off_band_ratio", report.weighted_off_band_ratio},
                  {"structured_diag_write_count",
                   report.structured_diag_write_count},
                  {"structured_first_offdiag_write_count",
                   report.structured_first_offdiag_write_count},
                  {"structured_off_band_drop_count",
                   report.structured_off_band_drop_count},
                  {"structured_diag_contact_abs_sum",
                   report.structured_diag_contact_abs_sum},
                  {"structured_first_offdiag_contact_abs_sum",
                   report.structured_first_offdiag_contact_abs_sum},
                  {"structured_off_band_drop_abs_sum",
                   report.structured_off_band_drop_abs_sum},
                  {"rhs_abs_sum", report.rhs_abs_sum}}},
                {"timing",
                 {{"socu_factor_solve_time_ms", report.socu_factor_solve_time_ms},
                  {"scatter_time_ms", report.scatter_time_ms},
                  {"stream_source", report.stream_source},
                  {"plan_created_this_solve", report.plan_created_this_solve},
                  {"debug_timing_enabled", report.debug_timing_enabled}}},
                {"solve",
                 {{"damping_shift", report.damping_shift},
                  {"surrogate_residual", report.surrogate_residual},
                  {"surrogate_relative_residual",
                   report.surrogate_relative_residual},
                  {"descent_dot", report.descent_dot},
                  {"gradient_norm", report.gradient_norm},
                  {"direction_norm", report.direction_norm},
                  {"direction_min_abs_threshold",
                   report.direction_min_abs_threshold},
                  {"direction_min_rel_threshold",
                   report.direction_min_rel_threshold},
                  {"rhs_sign_convention", report.rhs_sign_convention},
                  {"debug_validation_enabled", report.debug_validation_enabled},
                  {"report_each_solve", report.report_each_solve}}},
                {"line_search",
                 {{"feedback_available", report.line_search_feedback_available},
                  {"accepted", report.line_search_accepted},
                  {"hit_max_iter", report.line_search_hit_max_iter},
                  {"iteration_count", report.line_search_iteration_count},
                  {"accepted_alpha", report.line_search_accepted_alpha},
                  {"reject_streak", report.line_search_reject_streak}}},
                {"runtime_reorder",
                 {{"enabled", report.runtime_reorder_enabled},
                  {"interval", report.runtime_reorder_interval},
                  {"edge_capacity", report.runtime_reorder_edge_capacity},
                  {"collecting_frame", report.runtime_reorder_collecting_frame},
                  {"last_applied_frame", report.runtime_reorder_last_applied_frame},
                  {"edge_count", report.runtime_reorder_edge_count},
                  {"unique_edge_count", report.runtime_reorder_unique_edge_count},
                  {"overflow_count", report.runtime_reorder_overflow_count},
                  {"applied", report.runtime_reorder_applied},
                  {"failure_detail", report.runtime_reorder_failure_detail}}},
                {"status",
                 {{"direction_available", report.direction_available},
                  {"reason", report.status_reason},
                  {"detail", report.status_detail}}}};
}

}  // namespace

void write_solve_report(const SocuApproxSolveReport& report)
{
    std::filesystem::path report_path{report.report_path};
    std::filesystem::create_directories(report_path.parent_path());
    std::ofstream ofs{report_path};
    if(!ofs)
        throw Exception{fmt::format("SocuApproxSolver report '{}' cannot be written",
                                    report.report_path)};
    ofs << to_json(report).dump(2);
}

std::string_view to_string(SocuApproxGateReason reason) noexcept
{
    switch(reason)
    {
    case SocuApproxGateReason::None:
        return "none";
    case SocuApproxGateReason::SocuDisabled:
        return "socu_disabled";
    case SocuApproxGateReason::OrderingInvalid:
        return "ordering_invalid";
    case SocuApproxGateReason::UnsupportedPrecisionContract:
        return "unsupported_precision_contract";
    case SocuApproxGateReason::UnsupportedBlockSize:
        return "unsupported_block_size";
    case SocuApproxGateReason::SocuMathDxUnsupported:
        return "socu_mathdx_unsupported";
    case SocuApproxGateReason::SocuRuntimeArtifactUnavailable:
        return "socu_runtime_artifact_unavailable";
    case SocuApproxGateReason::StructuredProviderMissing:
        return "structured_provider_missing";
    case SocuApproxGateReason::StructuredCoverageInvalid:
        return "structured_coverage_invalid";
    case SocuApproxGateReason::StructuredSubsystemUnsupported:
        return "structured_subsystem_not_supported";
    case SocuApproxGateReason::DirectionInvalid:
        return "direction_invalid";
    case SocuApproxGateReason::SocuRuntimeError:
        return "socu_runtime_error";
    case SocuApproxGateReason::LineSearchRejected:
        return "line_search_rejected";
    }
    return "unknown";
}

}  // namespace uipc::backend::cuda_mixed
