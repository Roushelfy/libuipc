#include <linear_system/socu_approx_solver.h>

#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/socu_approx_kernels.h>
#include <linear_system/socu_approx_ordering.h>
#include <linear_system/socu_approx_runtime.h>
#include <mixed_precision/policy.h>
#include <sim_engine.h>
#include <uipc/common/exception.h>
#include <uipc/common/json.h>
#include <uipc/common/timer.h>
#include <backends/common/backend_path_tool.h>
#include <utils/matrix_market.h>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(SocuApproxSolver);

using namespace socu_approx;

namespace
{
namespace fs = std::filesystem;

[[noreturn]] void throw_gate_failure(const SocuApproxGateReport& report)
{
    throw Exception{fmt::format("SocuApproxSolver gate failed: reason={}, detail={}",
                                to_string(report.reason),
                                report.detail)};
}

SocuApproxGateReport make_failure(SocuApproxGateReason reason,
                                  std::string          detail,
                                  std::string          ordering_report_path = {})
{
    SocuApproxGateReport report;
    report.reason               = reason;
    report.detail               = std::move(detail);
    report.ordering_report_path = std::move(ordering_report_path);
    report.dtype = socu_dtype_name<GlobalLinearSystem::SolveScalar>();
    return report;
}
}  // namespace

auto SocuApproxSolver::assembly_requirements() const -> AssemblyRequirements
{
    AssemblyRequirements requirements;
    requirements.needs_dof_extent       = true;
    requirements.needs_gradient_b       = true;
    requirements.needs_full_sparse_A    = false;
    requirements.needs_structured_chain = true;
    requirements.needs_preconditioner   = false;
    requirements.allows_structured_offdiag = m_allows_structured_offdiag;
    requirements.assembly_mode = NewtonAssemblyMode::GradientStructuredHessian;
    return requirements;
}

SocuApproxSolver::~SocuApproxSolver() = default;

void SocuApproxSolver::do_build(BuildInfo& info)
{
    auto&      config      = world().scene().config();
    auto       solver_attr = config.find<std::string>("linear_system/solver");
    const std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "socu_approx")
    {
        throw SimSystemException("SocuApproxSolver unused");
    }

#if !UIPC_WITH_SOCU_NATIVE
    m_gate_report = make_failure(
        SocuApproxGateReason::SocuDisabled,
        "socu_native is disabled or not available; configure with UIPC_WITH_SOCU_NATIVE=AUTO or ON and initialize external/socu-native-cuda");
    throw_gate_failure(m_gate_report);
#endif

    require<GlobalLinearSystem>();

    auto debug_validation_attr =
        config.find<IndexT>("linear_system/socu_approx/debug_validation");
    auto debug_timing_attr =
        config.find<IndexT>("linear_system/socu_approx/debug_timing");
    auto report_each_solve_attr =
        config.find<IndexT>("linear_system/socu_approx/report_each_solve");
    m_debug_validation =
        debug_validation_attr && debug_validation_attr->view()[0] != 0;
    m_debug_timing = debug_timing_attr && debug_timing_attr->view()[0] != 0;
    m_report_each_solve =
        report_each_solve_attr && report_each_solve_attr->view()[0] != 0;
    m_report_counters_enabled =
        m_report_each_solve || m_debug_validation || m_debug_timing;

    auto debug_dump_structured_matrix_attr =
        config.find<IndexT>("linear_system/socu_approx/debug_dump_structured_matrix");
    m_debug_dump_structured_matrix =
        debug_dump_structured_matrix_attr
        && debug_dump_structured_matrix_attr->view()[0] != 0;

    auto damping_attr =
        config.find<Float>("linear_system/socu_approx/damping_shift");
    auto descent_eta_attr =
        config.find<Float>("linear_system/socu_approx/descent_eta");
    auto max_relative_residual_attr =
        config.find<Float>("linear_system/socu_approx/max_relative_residual");
    auto p_min_abs_attr =
        config.find<Float>("linear_system/socu_approx/p_min_abs");
    auto p_min_rel_attr =
        config.find<Float>("linear_system/socu_approx/p_min_rel");
    auto rhs_zero_abs_attr =
        config.find<Float>("linear_system/socu_approx/rhs_zero_abs");
    auto reject_streak_attr =
        config.find<IndexT>("linear_system/socu_approx/max_line_search_reject_streak");

    using SolveScalar = GlobalLinearSystem::SolveScalar;
    const double p_min_abs_default =
        sizeof(SolveScalar) == sizeof(float) ? 1e-10 : 1e-14;
    const double rhs_zero_abs_default =
        sizeof(SolveScalar) == sizeof(float) ? 1e-7 : 1e-12;
    const double configured_p_min_abs =
        p_min_abs_attr ? static_cast<double>(p_min_abs_attr->view()[0])
                       : p_min_abs_default;
    const double configured_rhs_zero_abs =
        rhs_zero_abs_attr ? static_cast<double>(rhs_zero_abs_attr->view()[0])
                          : rhs_zero_abs_default;

    m_damping_shift =
        damping_attr ? static_cast<double>(damping_attr->view()[0]) : 1e-6;
    m_descent_eta =
        descent_eta_attr ? static_cast<double>(descent_eta_attr->view()[0]) : 1e-8;
    m_max_relative_residual =
        max_relative_residual_attr
            ? static_cast<double>(max_relative_residual_attr->view()[0])
            : 1e-4;
    m_direction_min_abs =
        configured_p_min_abs > 0.0 ? configured_p_min_abs : p_min_abs_default;
    m_direction_min_rel =
        p_min_rel_attr ? static_cast<double>(p_min_rel_attr->view()[0]) : 1e-12;
    m_rhs_zero_abs =
        configured_rhs_zero_abs > 0.0 ? configured_rhs_zero_abs
                                      : rhs_zero_abs_default;
    m_max_line_search_reject_streak =
        reject_streak_attr ? reject_streak_attr->view()[0] : 1;
    if(m_damping_shift < 0.0)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::DirectionInvalid,
            "linear_system/socu_approx/damping_shift must be non-negative");
        throw_gate_failure(m_gate_report);
    }

    auto structured_scope_attr =
        config.find<std::string>("linear_system/socu_approx/structured_scope");
    std::string structured_scope =
        structured_scope_attr ? structured_scope_attr->view()[0]
                              : std::string{"multi_provider"};
    if(structured_scope != "single_provider"
       && structured_scope != "multi_provider")
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingInvalid,
            fmt::format("linear_system/socu_approx/structured_scope must be "
                        "'single_provider' or 'multi_provider', got '{}'",
                        structured_scope));
        throw_gate_failure(m_gate_report);
    }
    m_allows_structured_offdiag = structured_scope == "multi_provider";

    auto ordering_source_attr =
        config.find<std::string>("linear_system/socu_approx/ordering_source");
    const std::string ordering_source =
        ordering_source_attr ? ordering_source_attr->view()[0] : std::string{"init_time"};
    if(ordering_source != "init_time")
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingInvalid,
            fmt::format("linear_system/socu_approx/ordering_source only supports "
                        "'init_time', got '{}'",
                        ordering_source));
        throw_gate_failure(m_gate_report);
    }

    auto generated_ordering_report_attr =
        config.find<std::string>("linear_system/socu_approx/generated_ordering_report");
    std::string generated_ordering_report =
        generated_ordering_report_attr ? generated_ordering_report_attr->view()[0]
                                       : std::string{};

    fs::path ordering_report_path;
    Json     report;

    auto* abd = find<ABDLinearSubsystem>();
    SizeT body_count = abd ? abd->body_count() : SizeT{0};
    if(body_count == 0)
        body_count = count_abd_bodies_from_scene(world());
    const SizeT fem_vertex_count = fem_vertex_count_from_scene(world());
    if(body_count == 0 && fem_vertex_count == 0)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::StructuredSubsystemUnsupported,
            "init-time socu_approx ordering requires at least one supported ABD body or FEM vertex");
        throw_gate_failure(m_gate_report);
    }
    if(structured_scope == "single_provider" && body_count > 0
       && fem_vertex_count > 0)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::StructuredSubsystemUnsupported,
            "multi_provider_required: init-time socu_approx ordering found both ABD and FEM providers; set linear_system/socu_approx/structured_scope='multi_provider' to enable mixed-provider ordering");
        throw_gate_failure(m_gate_report);
    }

    auto ordering_orderer_attr =
        config.find<std::string>("linear_system/socu_approx/ordering_orderer");
    auto ordering_block_size_attr =
        config.find<std::string>("linear_system/socu_approx/ordering_block_size");
    const std::string ordering_orderer =
        ordering_orderer_attr ? ordering_orderer_attr->view()[0]
                              : std::string{"rcm"};
    const std::string ordering_block_size =
        ordering_block_size_attr ? ordering_block_size_attr->view()[0]
                                 : std::string{"auto"};
    if(ordering_block_size != "auto" && ordering_block_size != "32"
       && ordering_block_size != "64")
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::UnsupportedBlockSize,
            fmt::format("linear_system/socu_approx/ordering_block_size must be "
                        "'auto', '32', or '64', got '{}'",
                        ordering_block_size));
        throw_gate_failure(m_gate_report);
    }
    if(ordering_orderer != "rcm")
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingInvalid,
            fmt::format(
                "linear_system/socu_approx/ordering_orderer only supports 'rcm', got '{}'",
                ordering_orderer));
        throw_gate_failure(m_gate_report);
    }

    try
    {
        if(body_count > 0 && fem_vertex_count > 0)
        {
            report = generate_mixed_abd_fem_init_time_ordering_report(
                world(),
                body_count,
                ordering_orderer,
                ordering_block_size);
        }
        else if(body_count > 0)
        {
            report = generate_abd_init_time_ordering_report(
                body_count,
                ordering_orderer,
                ordering_block_size);
            report["provider_kind"] = "abd_only";
        }
        else
        {
            report = generate_fem_init_time_ordering_report(
                world(),
                ordering_orderer,
                ordering_block_size);
        }
    }
    catch(const std::exception& e)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingInvalid,
            fmt::format("init-time ordering generation failed: {}", e.what()));
        throw_gate_failure(m_gate_report);
    }

    ordering_report_path =
        generated_ordering_report.empty()
            ? default_generated_ordering_report_path(workspace())
            : absolute_workspace_path(workspace(), generated_ordering_report);
    write_json_report(ordering_report_path, report);
    logger::info("Generated socu_approx init-time ordering report at {}",
                 ordering_report_path.string());

    const Json* candidate = selected_candidate_json(report);
    const Json* ordering  = ordering_json(*candidate);
    if(!ordering || !has_basic_ordering_schema(*ordering))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingInvalid,
                                     "ordering report is missing the required ordering mapping schema",
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    m_gate_report.ordering_report_path = ordering_report_path.string();
    m_gate_report.structured_scope = structured_scope;
    if(report.contains("provider_kind") && report["provider_kind"].is_string())
        m_gate_report.provider_kind = report["provider_kind"].get<std::string>();
    else
        m_gate_report.provider_kind = "init_time";
    m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
    if(m_gate_report.block_size != 32 && m_gate_report.block_size != 64)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::UnsupportedBlockSize,
            fmt::format("ordering block_size must be 32 or 64, got {}",
                        m_gate_report.block_size),
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    if constexpr(!std::is_same_v<ActivePolicy::StoreScalar, ActivePolicy::SolveScalar>)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::UnsupportedPrecisionContract,
            "socu_approx strict structured solve only accepts StoreScalar == SolveScalar",
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    std::string                       ordering_detail;
    std::vector<SocuApproxBlockLayout> block_layouts;
    SizeT                             ordering_dof_count = 0;
    if(!parse_atom_dof_count(*ordering, ordering_dof_count, ordering_detail)
       || !parse_block_layouts(*ordering, block_layouts, ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingInvalid,
                                     ordering_detail.empty()
                                         ? "ordering report has an empty block layout"
                                         : ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    std::unique_ptr<StructuredChainProvider> provider;
    SizeT                                    padding_slot_count = 0;
    if(!build_ordering_provider(*ordering,
                                m_gate_report.block_size,
                                block_layouts,
                                provider,
                                padding_slot_count,
                                ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingInvalid,
                                     ordering_detail.empty()
                                         ? "ordering report cannot build a structured provider"
                                         : ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    StructuredQualityReport build_coverage = {};
    build_coverage.block_utilization = provider->quality_report().block_utilization;
    std::vector<IndexT> build_old_to_chain;
    std::vector<IndexT> build_chain_to_old;
    const SizeT         chain_scalar_count =
        block_layouts.size() * m_gate_report.block_size;
    if(!validate_dof_coverage(provider->dof_slots(),
                              ordering_dof_count,
                              chain_scalar_count,
                              build_old_to_chain,
                              build_chain_to_old,
                              build_coverage,
                              ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::StructuredCoverageInvalid,
                                     ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        m_gate_report.block_utilization = provider->quality_report().block_utilization;
        m_gate_report.coverage_active_dof_count = build_coverage.active_dof_count;
        m_gate_report.coverage_padding_dof_count = build_coverage.padding_dof_count;
        m_gate_report.complete_dof_coverage = false;
        throw_gate_failure(m_gate_report);
    }
    m_host_old_to_chain = std::move(build_old_to_chain);
    m_host_chain_to_old = std::move(build_chain_to_old);

    if(!validate_atom_inverse_mapping(*ordering, ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingInvalid,
                                     ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    if(const Json* metrics = metrics_json(report, *candidate))
    {
        if(metrics->contains("near_band_ratio"))
            m_gate_report.near_band_ratio = metrics->at("near_band_ratio").get<double>();
        if(metrics->contains("off_band_ratio"))
            m_gate_report.off_band_ratio = metrics->at("off_band_ratio").get<double>();
        if(metrics->contains("off_band_drop_norm_ratio"))
            m_gate_report.off_band_drop_norm_ratio =
                metrics->at("off_band_drop_norm_ratio").get<double>();
    }

    m_gate_report.block_utilization =
        provider->quality_report().block_utilization;

    auto min_near_attr =
        config.find<Float>("linear_system/socu_approx/min_near_band_ratio");
    auto max_off_attr =
        config.find<Float>("linear_system/socu_approx/max_off_band_ratio");
    auto min_util_attr =
        config.find<Float>("linear_system/socu_approx/min_block_utilization");
    auto max_drop_attr =
        config.find<Float>("linear_system/socu_approx/max_off_band_drop_norm_ratio");
    const double min_near_band_ratio =
        min_near_attr ? static_cast<double>(min_near_attr->view()[0]) : 0.90;
    const double max_off_band_ratio =
        max_off_attr ? static_cast<double>(max_off_attr->view()[0]) : 0.10;
    const double min_block_utilization =
        min_util_attr ? static_cast<double>(min_util_attr->view()[0]) : 0.65;
    const double max_off_band_drop_norm_ratio =
        max_drop_attr ? static_cast<double>(max_drop_attr->view()[0]) : 0.05;
    auto solve_report_attr =
        config.find<std::string>("linear_system/socu_approx/report");
    std::string solve_report =
        solve_report_attr ? solve_report_attr->view()[0] : std::string{};
    fs::path solve_report_path =
        solve_report.empty()
            ? fs::absolute(fs::path{workspace()} / "socu_approx_report.json")
            : absolute_workspace_path(workspace(), solve_report);

    m_report                         = SocuApproxSolveReport{};
    m_report.report_path             = solve_report_path.string();
    m_report.ordering_report_path    = ordering_report_path.string();
    m_report.provider_kind           = m_gate_report.provider_kind;
    m_report.structured_scope        = structured_scope;
    m_report.block_size         = m_gate_report.block_size;
    m_report.block_count        = block_layouts.size();
    m_report.chain_atom_count   = ordering->at("chain_to_old").size();
    m_report.ordering_dof_count = ordering_dof_count;
    m_report.structured_slot_count = provider->dof_slots().size();
    m_report.padding_slot_count = padding_slot_count;
    m_report.block_utilization =
        provider->quality_report().block_utilization;
    m_report.near_band_ratio = m_gate_report.near_band_ratio;
    m_report.off_band_ratio = m_gate_report.off_band_ratio;
    m_report.off_band_drop_norm_ratio =
        m_gate_report.off_band_drop_norm_ratio;
    m_report.min_block_utilization = min_block_utilization;
    m_report.min_near_band_ratio = min_near_band_ratio;
    m_report.max_off_band_ratio = max_off_band_ratio;
    m_report.max_off_band_drop_norm_ratio =
        max_off_band_drop_norm_ratio;
    m_report.debug_validation_enabled = m_debug_validation;
    m_report.debug_timing_enabled = m_debug_timing;
    m_report.report_each_solve = m_report_each_solve;
    m_report.damping_shift = m_damping_shift;
    m_report.direction_min_abs_threshold = m_direction_min_abs;
    m_report.direction_min_rel_threshold = m_direction_min_rel;
    m_report.coverage_active_dof_count =
        build_coverage.active_dof_count;
    m_report.coverage_padding_dof_count =
        build_coverage.padding_dof_count;
    m_report.complete_dof_coverage = true;
    m_report.blocks             = std::move(block_layouts);
    m_dof_slots.assign(provider->dof_slots().begin(), provider->dof_slots().end());

    m_gate_report.passed = true;
    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail =
        fmt::format("structured band direct solve enabled for provider_kind={} scope={}",
                    m_gate_report.provider_kind,
                    structured_scope);
    m_gate_report.dtype = socu_dtype_name<GlobalLinearSystem::SolveScalar>();
    m_gate_report.coverage_active_dof_count =
        build_coverage.active_dof_count;
    m_gate_report.coverage_padding_dof_count =
        build_coverage.padding_dof_count;
    m_gate_report.complete_dof_coverage = true;

#if UIPC_WITH_SOCU_NATIVE
    const auto manifest_path = default_mathdx_manifest_path();
    if(manifest_path.empty() || !fs::is_regular_file(manifest_path))
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
            fmt::format("MathDx manifest is missing; expected '{}'",
                        manifest_path.string()),
            ordering_report_path.string());
        m_gate_report.block_size = m_report.block_size;
        throw_gate_failure(m_gate_report);
    }

    std::string manifest_detail;
    try
    {
        if(!validate_mathdx_manifest<Runtime::Scalar>(manifest_path,
                                                      m_report.block_size,
                                                      m_gate_report,
                                                      manifest_detail))
        {
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
                manifest_detail,
                ordering_report_path.string());
            m_gate_report.block_size = m_report.block_size;
            m_gate_report.mathdx_manifest_path = manifest_path.string();
            throw_gate_failure(m_gate_report);
        }
    }
    catch(const std::exception& e)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
            fmt::format("MathDx manifest preflight failed: {}", e.what()),
            ordering_report_path.string());
        m_gate_report.block_size = m_report.block_size;
        m_gate_report.mathdx_manifest_path = manifest_path.string();
        throw_gate_failure(m_gate_report);
    }

    m_gate_report.mathdx_prebuilt_cubin_ok = m_gate_report.mathdx_artifacts_ok;
    m_gate_report.debug_validation_enabled = m_debug_validation;
    m_gate_report.debug_timing_enabled = m_debug_timing;
    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        m_gate_report = make_failure(
            SocuApproxGateReason::SocuMathDxUnsupported,
            "no CUDA device is available for socu_approx",
            ordering_report_path.string());
        m_gate_report.block_size = m_report.block_size;
        throw_gate_failure(m_gate_report);
    }

    socu_native::ProblemShape shape{
        static_cast<int>(m_report.block_count),
        static_cast<int>(m_report.block_size),
        1};
    socu_native::SolverPlanOptions options;
    options.backend      = socu_native::SolverBackend::NativePerf;
    options.perf_backend = socu_native::PerfBackend::MathDx;
    options.math_mode    = socu_native::MathMode::Auto;
    options.graph_mode   = socu_native::GraphMode::Off;

    const auto capability =
        socu_native::query_solver_capability<Runtime::Scalar>(
            shape,
            socu_native::SolverOperation::FactorAndSolve,
            options);
    m_gate_report.resolved_backend =
        to_report_string(capability.resolved_backend);
    m_gate_report.resolved_perf_backend =
        to_report_string(capability.resolved_perf_backend);
    m_gate_report.resolved_math_mode =
        to_report_string(capability.resolved_math_mode);
    m_gate_report.resolved_graph_mode =
        to_report_string(capability.resolved_graph_mode);
    if(!capability.supported
       || capability.resolved_backend != socu_native::SolverBackend::NativePerf
       || capability.resolved_perf_backend != socu_native::PerfBackend::MathDx)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::SocuMathDxUnsupported,
            fmt::format("socu_native MathDx capability rejected: {}",
                        capability.reason),
            ordering_report_path.string());
        m_gate_report.block_size = m_report.block_size;
        throw_gate_failure(m_gate_report);
    }

    try
    {
        m_runtime = std::make_unique<Runtime>(shape, options);
        m_runtime->reserve(m_debug_validation, m_report_counters_enabled);
        m_runtime->upload_mappings_once(m_host_old_to_chain, m_host_chain_to_old);
        m_runtime->create_plan();
    }
    catch(const std::exception& e)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::SocuRuntimeError,
            fmt::format("socu_native runtime initialization failed: {}", e.what()),
            ordering_report_path.string());
        m_gate_report.block_size = m_report.block_size;
        throw_gate_failure(m_gate_report);
    }
#else
    m_gate_report = make_failure(
        SocuApproxGateReason::SocuDisabled,
        "socu_native is disabled or not available");
    throw_gate_failure(m_gate_report);
#endif

    logger::info("SocuApproxSolver strict structured solve enabled: block_size={}, blocks={}, report='{}'",
                 m_report.block_size,
                 m_report.block_count,
                 m_report.report_path);
}

void SocuApproxSolver::prepare_structured_chain(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
#if !UIPC_WITH_SOCU_NATIVE
    throw Exception{"SocuApproxSolver structured solve reached without socu_native support"};
#else
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver structured assembly requires initialized runtime.");

    m_report.direction_available = false;
    m_report.status_reason.clear();
    m_report.status_detail.clear();

    if(m_max_line_search_reject_streak > 0
       && static_cast<IndexT>(m_line_search_reject_streak)
              >= m_max_line_search_reject_streak)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::LineSearchRejected,
            fmt::format("previous socu_approx direction hit line-search max_iter "
                        "{} consecutive time(s); threshold={}",
                        m_line_search_reject_streak,
                        m_max_line_search_reject_streak),
            m_gate_report.ordering_report_path);
        m_gate_report.block_size = m_report.block_size;
        m_report.status_reason =
            std::string{to_string(SocuApproxGateReason::LineSearchRejected)};
        m_report.status_detail = m_gate_report.detail;
        write_solve_report(m_report);
        throw_gate_failure(m_gate_report);
    }

    const SizeT chain_scalar_count =
        m_runtime->shape.horizon * m_runtime->shape.n;
    if(m_host_old_to_chain.size() != static_cast<SizeT>(info.b().size())
       || m_host_chain_to_old.size() != chain_scalar_count)
    {
        const std::string coverage_detail =
            fmt::format("structured coverage size mismatch: old_to_chain={}, "
                        "global_rhs={}, chain_to_old={}, chain_scalar_count={}",
                        m_host_old_to_chain.size(),
                        info.b().size(),
                        m_host_chain_to_old.size(),
                        chain_scalar_count);
        m_gate_report = make_failure(SocuApproxGateReason::StructuredCoverageInvalid,
                                     coverage_detail,
                                     m_gate_report.ordering_report_path);
        m_gate_report.block_size = m_report.block_size;
        m_gate_report.block_utilization = m_report.block_utilization;
        m_gate_report.complete_dof_coverage = false;
        m_report.status_reason =
            std::string{to_string(SocuApproxGateReason::StructuredCoverageInvalid)};
        m_report.status_detail = coverage_detail;
        write_solve_report(m_report);
        throw_gate_failure(m_gate_report);
    }

    m_gate_report.complete_dof_coverage = true;
    m_report.complete_dof_coverage = true;

    m_report.damping_shift = m_damping_shift;

    const cudaStream_t stream = system().stream();
    if(m_report_counters_enabled && m_runtime->report_counters.size() == 5)
        muda::BufferLaunch(stream).fill<IndexT>(m_runtime->report_counters.view(), 0);

    initialize_structured_workspace<GlobalLinearSystem::StoreScalar, Runtime::Scalar>(
        stream,
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        info.b(),
        m_runtime->device_diag.view(),
        m_runtime->device_off_diag.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_rhs_original.view(),
        m_runtime->device_chain_to_old.view(),
        m_report.damping_shift);

    info.set_workspace(
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        span<const StructuredDofSlot>{m_dof_slots},
        m_runtime->device_diag.view(),
        m_runtime->device_off_diag.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_old_to_chain.view(),
        m_runtime->device_chain_to_old.view(),
        stream);
    if(m_report_counters_enabled && m_runtime->report_counters.size() == 5)
        info.set_contact_counters(m_runtime->report_counters.view());

    m_report.packed = true;
    m_report.active_rhs_scalar_count = info.b().size();
    m_report.rhs_scalar_count = m_runtime->layout.rhs_element_count;
    m_report.diag_scalar_count = m_runtime->layout.diag_element_count;
    m_report.first_offdiag_scalar_count =
        m_runtime->layout.off_diag_element_count;
    m_report.diag_block_count = m_runtime->layout.diag_block_count;
    m_report.first_offdiag_block_count =
        m_runtime->layout.off_diag_block_count;
    m_report.stream_source = "mixed_backend_current_stream";
#endif
}

void SocuApproxSolver::finalize_structured_chain(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    if(info.report_counters_enabled())
    {
        std::array<IndexT, 5> contact_counts{};
        info.contact_counters().copy_to(contact_counts.data());
        info.record_contact_diag_writes(static_cast<SizeT>(contact_counts[0]));
        info.record_contact_first_offdiag_writes(static_cast<SizeT>(contact_counts[1]));
        info.record_contact_band_stats(
            static_cast<SizeT>(contact_counts[3]),
            static_cast<SizeT>(contact_counts[4]),
            static_cast<SizeT>(contact_counts[0] + contact_counts[1]),
            static_cast<SizeT>(contact_counts[2]));
    }

    m_report.structured_diag_write_count =
        info.diag_write_count() + info.contact_diag_write_count();
    m_report.structured_first_offdiag_write_count =
        info.first_offdiag_write_count()
        + info.contact_first_offdiag_write_count();
    m_report.near_band_contact_count = info.near_band_contact_count();
    m_report.off_band_contact_count = info.off_band_contact_count();
    m_report.near_band_contribution_count =
        info.near_band_contribution_count();
    m_report.off_band_contribution_count =
        info.off_band_contribution_count();
    m_report.absorbed_hessian_contribution_count =
        info.near_band_contribution_count();
    m_report.dropped_hessian_contribution_count =
        info.off_band_contribution_count();
    m_report.structured_off_band_drop_count =
        info.off_band_contribution_count();

    const SizeT total_contact_count =
        info.near_band_contact_count() + info.off_band_contact_count();
    if(total_contact_count > 0)
    {
        m_report.contribution_near_band_ratio =
            static_cast<double>(info.near_band_contact_count())
            / static_cast<double>(total_contact_count);
        m_report.contribution_off_band_ratio =
            static_cast<double>(info.off_band_contact_count())
            / static_cast<double>(total_contact_count);
    }

    const SizeT total_contribution_count =
        info.near_band_contribution_count() + info.off_band_contribution_count();
    if(total_contribution_count > 0)
    {
        m_report.weighted_near_band_ratio =
            static_cast<double>(info.near_band_contribution_count())
            / static_cast<double>(total_contribution_count);
        m_report.weighted_off_band_ratio =
            static_cast<double>(info.off_band_contribution_count())
            / static_cast<double>(total_contribution_count);
    }

    if(info.off_band_contribution_count() > 0)
    {
        m_gate_report.near_band_ratio =
            m_report.weighted_near_band_ratio;
        m_gate_report.off_band_ratio =
            m_report.weighted_off_band_ratio;
        m_gate_report.detail =
            fmt::format("structured assembly recorded {} off-band contribution(s); "
                        "socu_approx continues with the in-band structured matrix "
                        "(diagonal and first-offdiagonal blocks)",
                        info.off_band_contribution_count());
        m_report.status_detail = m_gate_report.detail;
    }
#if UIPC_WITH_SOCU_NATIVE
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver structured assembly requires initialized runtime.");
    if(m_debug_validation)
        m_runtime->snapshot_matrix(info.stream());
    if(m_debug_dump_structured_matrix) [[unlikely]]
    {
        cudaStreamSynchronize(info.stream());
        dump_structured_matrix(info);
    }
#endif
}

void SocuApproxSolver::dump_structured_matrix(
    const GlobalLinearSystem::StructuredAssemblyInfo& /*info*/)
{
#if UIPC_WITH_SOCU_NATIVE
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver dump_structured_matrix requires initialized runtime.");

    using Scalar = Runtime::Scalar;
    const auto& shape  = m_runtime->shape;
    const auto& layout = m_runtime->layout;
    const SizeT n      = static_cast<SizeT>(shape.n);
    const SizeT H      = static_cast<SizeT>(shape.horizon);

    // Download diag, off_diag, and chain_to_old to host
    std::vector<Scalar> host_diag(layout.diag_element_count, Scalar{0});
    std::vector<Scalar> host_off_diag(layout.off_diag_element_count, Scalar{0});
    const auto& host_chain_to_old = m_host_chain_to_old;

    if(!host_diag.empty())
        cudaMemcpy(host_diag.data(),
                   m_runtime->device_diag.data(),
                   layout.diag_element_count * sizeof(Scalar),
                   cudaMemcpyDeviceToHost);
    if(!host_off_diag.empty())
        cudaMemcpy(host_off_diag.data(),
                   m_runtime->device_off_diag.data(),
                   layout.off_diag_element_count * sizeof(Scalar),
                   cudaMemcpyDeviceToHost);

    // Build COO triplets in global DoF order
    // Off-diag layout: element at off_diag[b*n*n + r*n + c] represents
    //   the assembled value for chain pair (b*n+c, (b+1)*n+r).
    // For symmetric Hessians assembled with both (i,j) and (j,i),
    //   the value accumulates A[chain_i, chain_j] + A[chain_j, chain_i] = 2*A[...].
    // This is noted in the exported file header for reference.
    struct Entry { int row; int col; double val; };
    std::vector<Entry> triplets;
    triplets.reserve(layout.diag_element_count + 2 * layout.off_diag_element_count);

    auto chain_to_global = [&](SizeT chain) -> int {
        if(chain >= host_chain_to_old.size()) return -1;
        return host_chain_to_old[chain];
    };

    // Diagonal blocks: diag[b*n*n + li*n + lj] = A[b*n+li, b*n+lj]
    for(SizeT b = 0; b < H; ++b)
    {
        for(SizeT li = 0; li < n; ++li)
        {
            const int row = chain_to_global(b * n + li);
            if(row < 0)
                continue;
            for(SizeT lj = 0; lj < n; ++lj)
            {
                const int col = chain_to_global(b * n + lj);
                if(col < 0)
                    continue;
                const double val = static_cast<double>(host_diag[b * n * n + li * n + lj]);
                if(val != 0.0)
                    triplets.push_back({row, col, val});
            }
        }
    }

    // Off-diagonal blocks: off_diag[b*n*n + r*n + c] = A[b*n+c, (b+1)*n+r]
    // (accumulated, may be 2x for symmetric assemblies)
    const SizeT off_blocks = H > 0 ? H - 1 : 0;
    for(SizeT b = 0; b < off_blocks; ++b)
    {
        for(SizeT r = 0; r < n; ++r)
        {
            const int col = chain_to_global((b + 1) * n + r);
            if(col < 0)
                continue;
            for(SizeT c = 0; c < n; ++c)
            {
                const int row = chain_to_global(b * n + c);
                if(row < 0)
                    continue;
                const double val = static_cast<double>(host_off_diag[b * n * n + r * n + c]);
                if(val != 0.0)
                {
                    triplets.push_back({row, col, val});
                    triplets.push_back({col, row, val});
                }
            }
        }
    }

    // Compute global dimension from the mapping
    int global_dim = 0;
    for(const auto& old : host_chain_to_old)
    {
        if(old >= 0 && old + 1 > global_dim)
            global_dim = old + 1;
    }

    // Write Matrix Market file
    auto path_tool = BackendPathTool(workspace());
    auto output_folder = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    const auto path = fmt::format("{}A_structured.{}.{}.mtx",
                                  output_folder.string(),
                                  engine().frame(),
                                  engine().newton_iter());
    FILE* fp = std::fopen(path.c_str(), "w");
    if(!fp)
    {
        logger::warn("SocuApproxSolver: failed to open {} for structured matrix dump", path);
        return;
    }
    fmt::fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fmt::fprintf(fp, "%% SocuApprox structured band matrix (diagonal + first off-diagonal)\n");
    fmt::fprintf(fp, "%% horizon=%zu, block_size=%zu\n", H, n);
    fmt::fprintf(fp, "%% off_diag values may be 2x actual if assembler writes both (i,j) and (j,i)\n");
    fmt::fprintf(fp, "%d %d %zu\n", global_dim, global_dim, triplets.size());
    for(const auto& e : triplets)
        fmt::fprintf(fp, "%d %d %.17g\n", e.row + 1, e.col + 1, e.val);
    std::fclose(fp);
    logger::info("SocuApproxSolver: dumped structured matrix ({} entries) to {}", triplets.size(), path);
#endif
}

void SocuApproxSolver::notify_line_search_result(
    const GlobalLinearSystem::LineSearchFeedback& feedback)
{
    m_last_line_search_feedback = feedback;
    m_has_line_search_feedback  = true;
    if(feedback.hit_max_iter || !feedback.accepted)
        ++m_line_search_reject_streak;
    else
        m_line_search_reject_streak = 0;

    m_report.line_search_feedback_available = true;
    m_report.line_search_accepted = feedback.accepted;
    m_report.line_search_hit_max_iter = feedback.hit_max_iter;
    m_report.line_search_iteration_count = feedback.iteration_count;
    m_report.line_search_accepted_alpha = feedback.accepted_alpha;
    m_report.line_search_reject_streak = m_line_search_reject_streak;

    if((m_report_each_solve || m_debug_validation || m_debug_timing)
       && !m_report.report_path.empty())
        write_solve_report(m_report);
}

void SocuApproxSolver::validate_direction_light(cudaStream_t stream)
{
#if UIPC_WITH_SOCU_NATIVE
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");

    validate_structured_direction_light<Runtime::Scalar>(
        stream,
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        m_runtime->device_rhs_original.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_chain_to_old.view(),
        m_runtime->validation_sums.view());
    m_runtime->download_validation_sums(stream);
    const double* validation_sums = m_runtime->host_validation_sums;

    m_report.gradient_norm = std::sqrt(validation_sums[0]);
    m_report.direction_norm = std::sqrt(validation_sums[1]);
    m_report.descent_dot = -validation_sums[2];
    m_report.rhs_sign_convention = "rhs_is_global_b";

    m_report.direction_min_abs_threshold = m_direction_min_abs;
    m_report.direction_min_rel_threshold = m_direction_min_rel;

    const double p_threshold =
        std::max(m_direction_min_abs, m_direction_min_rel * m_report.gradient_norm);
    const bool rhs_finite =
        validation_sums[3] == 0.0
        && std::isfinite(m_report.gradient_norm);
    const bool zero_rhs = rhs_finite
        && m_report.gradient_norm <= m_rhs_zero_abs;
    if(zero_rhs)
    {
        const auto rhs_bytes =
            m_runtime->device_rhs.size() * sizeof(Runtime::Scalar);
        if(rhs_bytes)
            SOCU_NATIVE_CHECK_CUDA(
                cudaMemsetAsync(m_runtime->device_rhs.data(), 0, rhs_bytes, stream));
        m_report.direction_norm = 0.0;
        m_report.descent_dot = 0.0;
        return;
    }

    const bool finite =
        validation_sums[4] == 0.0
        && std::isfinite(m_report.descent_dot)
        && std::isfinite(m_report.gradient_norm)
        && std::isfinite(m_report.direction_norm);
    const bool nonzero =
        m_report.gradient_norm > 0.0
        && m_report.direction_norm > p_threshold;
    const bool descent =
        m_report.descent_dot
        < -m_descent_eta * m_report.gradient_norm
               * m_report.direction_norm;
    if(!finite || !nonzero || !descent)
    {
        m_report.direction_available = false;
        m_report.status_reason =
            std::string{to_string(SocuApproxGateReason::DirectionInvalid)};
        m_report.status_detail =
            fmt::format("light direction validation failed: finite={}, nonzero={}, "
                        "descent={}, nonfinite_count={}, g_dot_p={}, g_norm={}, "
                        "p_norm={}, p_threshold={}",
                        finite,
                        nonzero,
                        descent,
                        validation_sums[4],
                        m_report.descent_dot,
                        m_report.gradient_norm,
                        m_report.direction_norm,
                        p_threshold);
        write_solve_report(m_report);
        throw Exception{fmt::format("SocuApproxSolver direction invalid: {}",
                                    m_report.status_detail)};
    }
#else
    (void)stream;
#endif
}

void SocuApproxSolver::debug_validate_direction(cudaStream_t stream)
{
#if UIPC_WITH_SOCU_NATIVE
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");

    validate_structured_direction<Runtime::Scalar>(
        stream,
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        m_runtime->device_diag_original.view(),
        m_runtime->device_off_diag_original.view(),
        m_runtime->device_rhs_original.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_chain_to_old.view(),
        m_runtime->validation_sums.view());
    m_runtime->download_validation_sums(stream);
    const double* validation_sums = m_runtime->host_validation_sums;

    m_report.gradient_norm = std::sqrt(validation_sums[0]);
    m_report.direction_norm = std::sqrt(validation_sums[1]);
    m_report.descent_dot = -validation_sums[2];
    m_report.surrogate_residual = std::sqrt(validation_sums[3]);
    m_report.surrogate_relative_residual =
        m_report.surrogate_residual
        / std::max(1.0, m_report.gradient_norm);

    m_report.direction_min_abs_threshold = m_direction_min_abs;
    m_report.direction_min_rel_threshold = m_direction_min_rel;
    m_report.rhs_sign_convention = "rhs_is_global_b";

    const double p_threshold =
        std::max(m_direction_min_abs, m_direction_min_rel * m_report.gradient_norm);
    const bool finite =
        std::isfinite(m_report.surrogate_residual)
        && std::isfinite(m_report.surrogate_relative_residual)
        && std::isfinite(m_report.descent_dot)
        && std::isfinite(m_report.gradient_norm)
        && std::isfinite(m_report.direction_norm);
    const bool nonzero =
        m_report.gradient_norm > 0.0
        && m_report.direction_norm > p_threshold;
    const bool descent =
        m_report.descent_dot
        < -m_descent_eta * m_report.gradient_norm
               * m_report.direction_norm;
    const bool residual_ok =
        m_report.surrogate_relative_residual <= m_max_relative_residual;
    const bool zero_rhs_converged =
        finite
        && m_report.gradient_norm <= m_rhs_zero_abs
        && m_report.direction_norm <= p_threshold
        && residual_ok;

    if(!zero_rhs_converged && (!finite || !nonzero || !descent || !residual_ok))
    {
        m_report.direction_available = false;
        m_report.status_reason =
            std::string{to_string(SocuApproxGateReason::DirectionInvalid)};
        m_report.status_detail =
            fmt::format("direction validation failed: finite={}, nonzero={}, descent={}, "
                        "residual_ok={}, g_dot_p={}, g_norm={}, p_norm={}, "
                        "p_threshold={}, rel_residual={}, max_relative_residual={}",
                        finite,
                        nonzero,
                        descent,
                        residual_ok,
                        m_report.descent_dot,
                        m_report.gradient_norm,
                        m_report.direction_norm,
                        p_threshold,
                        m_report.surrogate_relative_residual,
                        m_max_relative_residual);
        write_solve_report(m_report);
        throw Exception{fmt::format("SocuApproxSolver direction invalid: {}",
                                    m_report.status_detail)};
    }
#else
    (void)stream;
#endif
}

void SocuApproxSolver::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
#if !UIPC_WITH_SOCU_NATIVE
    throw Exception{"SocuApproxSolver solve mode reached without socu_native support"};
#else
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");

    const cudaStream_t stream = system().stream();
    cudaEvent_t factor_start = nullptr;
    cudaEvent_t factor_done  = nullptr;
    cudaEvent_t scatter_start = nullptr;
    cudaEvent_t scatter_done = nullptr;
    auto destroy_timing_events = [&]
    {
        if(!m_debug_timing)
            return;
        if(factor_start)
            cudaEventDestroy(factor_start);
        if(factor_done)
            cudaEventDestroy(factor_done);
        if(scatter_start)
            cudaEventDestroy(scatter_start);
        if(scatter_done)
            cudaEventDestroy(scatter_done);
        factor_start = nullptr;
        factor_done = nullptr;
        scatter_start = nullptr;
        scatter_done = nullptr;
    };
    if(m_debug_timing)
    {
        SOCU_NATIVE_CHECK_CUDA(cudaEventCreate(&factor_start));
        SOCU_NATIVE_CHECK_CUDA(cudaEventCreate(&factor_done));
        SOCU_NATIVE_CHECK_CUDA(cudaEventCreate(&scatter_start));
        SOCU_NATIVE_CHECK_CUDA(cudaEventCreate(&scatter_done));
        SOCU_NATIVE_CHECK_CUDA(cudaEventRecord(factor_start, stream));
    }

    try
    {
        m_report.plan_created_this_solve =
            m_runtime->factor_and_solve(stream);
        if(m_debug_timing)
            SOCU_NATIVE_CHECK_CUDA(cudaEventRecord(factor_done, stream));
    }
    catch(const std::exception& e)
    {
        destroy_timing_events();
        m_report.direction_available = false;
        m_report.status_reason =
            std::string{to_string(SocuApproxGateReason::SocuRuntimeError)};
        m_report.status_detail =
            fmt::format("socu_native factor_and_solve failed: {}", e.what());
        write_solve_report(m_report);
        throw Exception{fmt::format("SocuApproxSolver solve failed: {}",
                                    m_report.status_detail)};
    }

    try
    {
        validate_direction_light(stream);
        if(m_debug_validation)
            debug_validate_direction(stream);

        if(m_debug_timing)
            SOCU_NATIVE_CHECK_CUDA(cudaEventRecord(scatter_start, stream));
        scatter_structured_solution<Runtime::Scalar>(
            stream,
            m_runtime->device_rhs.view(),
            m_runtime->device_old_to_chain.view(),
            info.x());
        if(m_debug_timing)
        {
            SOCU_NATIVE_CHECK_CUDA(cudaEventRecord(scatter_done, stream));
            SOCU_NATIVE_CHECK_CUDA(cudaEventSynchronize(scatter_done));
            float factor_ms = 0.0f;
            float scatter_ms = 0.0f;
            SOCU_NATIVE_CHECK_CUDA(cudaEventElapsedTime(&factor_ms,
                                                        factor_start,
                                                        factor_done));
            SOCU_NATIVE_CHECK_CUDA(cudaEventElapsedTime(&scatter_ms,
                                                        scatter_start,
                                                        scatter_done));
            m_report.socu_factor_solve_time_ms = factor_ms;
            m_report.scatter_time_ms = scatter_ms;
        }
        destroy_timing_events();
    }
    catch(...)
    {
        destroy_timing_events();
        throw;
    }

    m_report.direction_available = true;
    m_report.status_reason = std::string{to_string(SocuApproxGateReason::None)};
    if(m_report.status_detail.empty())
    {
        m_report.status_detail =
            fmt::format("structured band direction solved and scattered: provider={}, scope={}",
                        m_report.provider_kind,
                        m_report.structured_scope);
    }
    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail = m_report.status_detail;
    if(m_report_each_solve || m_debug_validation || m_debug_timing)
        write_solve_report(m_report);

    info.iter_count(1);
    logger::info("SocuApproxSolver strict structured solve launched on mixed backend "
                 "stream: provider={}, scope={}, debug_validation={}, report='{}'",
                 m_report.provider_kind,
                 m_report.structured_scope,
                 m_debug_validation,
                 m_report.report_path);
#endif
}

}  // namespace uipc::backend::cuda_mixed
