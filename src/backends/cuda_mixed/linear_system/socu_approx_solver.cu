#include <linear_system/socu_approx_solver.h>

#include <linear_system/global_linear_system.h>
#include <mixed_precision/policy.h>
#include <sim_engine.h>
#include <uipc/common/exception.h>
#include <uipc/common/json.h>

#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <utility>

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(SocuApproxSolver);

namespace
{
namespace fs = std::filesystem;

const Json* selected_candidate_json(const Json& report)
{
    if(report.contains("selected") && report["selected"].is_object())
        return &report["selected"];
    return &report;
}

const Json* ordering_json(const Json& candidate)
{
    if(candidate.contains("ordering") && candidate["ordering"].is_object())
        return &candidate["ordering"];
    if(candidate.contains("block_size"))
        return &candidate;
    return nullptr;
}

const Json* metrics_json(const Json& report, const Json& candidate)
{
    if(candidate.contains("metrics") && candidate["metrics"].is_object())
        return &candidate["metrics"];
    if(report.contains("metrics") && report["metrics"].is_object())
        return &report["metrics"];
    return nullptr;
}

bool required_array(const Json& json, std::string_view key, std::size_t expected_size)
{
    auto it = json.find(key);
    return it != json.end() && it->is_array() && it->size() == expected_size;
}

bool has_basic_ordering_schema(const Json& ordering)
{
    if(!ordering.contains("block_size")
       || !(ordering["block_size"].is_number_unsigned()
            || ordering["block_size"].is_number_integer()))
        return false;
    if(!ordering.contains("chain_to_old") || !ordering["chain_to_old"].is_array())
        return false;

    const auto atom_count = ordering["chain_to_old"].size();
    return required_array(ordering, "old_to_chain", atom_count)
           && required_array(ordering, "atom_to_block", atom_count)
           && required_array(ordering, "atom_block_offset", atom_count)
           && required_array(ordering, "atom_dof_count", atom_count)
           && ordering.contains("block_to_atom_range")
           && ordering["block_to_atom_range"].is_array()
           && !ordering["block_to_atom_range"].empty();
}

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
    return report;
}
}  // namespace

std::string_view to_string(SocuApproxGateReason reason) noexcept
{
    switch(reason)
    {
    case SocuApproxGateReason::None:
        return "none";
    case SocuApproxGateReason::SocuDisabled:
        return "socu_disabled";
    case SocuApproxGateReason::OrderingMissing:
        return "ordering_missing";
    case SocuApproxGateReason::OrderingReportInvalid:
        return "ordering_report_invalid";
    case SocuApproxGateReason::UnsupportedPrecisionContract:
        return "unsupported_precision_contract";
    case SocuApproxGateReason::UnsupportedBlockSize:
        return "unsupported_block_size";
    case SocuApproxGateReason::SocuMathDxUnsupported:
        return "socu_mathdx_unsupported";
    case SocuApproxGateReason::OrderingQualityTooLow:
        return "ordering_quality_too_low";
    case SocuApproxGateReason::ContactOffBandRatioTooHigh:
        return "contact_off_band_ratio_too_high";
    case SocuApproxGateReason::StructuredProviderMissing:
        return "structured_provider_missing";
    case SocuApproxGateReason::StubNoDirection:
        return "socu_approx_stub_no_direction";
    }
    return "unknown";
}

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

    require<GlobalLinearSystem>();

    auto ordering_report_attr =
        config.find<std::string>("linear_system/socu_approx/ordering_report");
    std::string ordering_report =
        ordering_report_attr ? ordering_report_attr->view()[0] : std::string{};

    if(ordering_report.empty())
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingMissing,
            "linear_system/socu_approx/ordering_report is empty");
        throw_gate_failure(m_gate_report);
    }

    fs::path ordering_report_path{ordering_report};
    if(ordering_report_path.is_relative())
        ordering_report_path = fs::absolute(fs::path{workspace()} / ordering_report_path);

    std::ifstream ifs{ordering_report_path};
    if(!ifs)
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingMissing,
                                     fmt::format("ordering report '{}' cannot be opened",
                                                 ordering_report_path.string()),
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    Json report = Json::parse(ifs, nullptr, false);
    if(report.is_discarded() || !report.is_object())
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     fmt::format("ordering report '{}' is not valid JSON",
                                                 ordering_report_path.string()),
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    const Json* candidate = selected_candidate_json(report);
    const Json* ordering  = ordering_json(*candidate);
    if(!ordering || !has_basic_ordering_schema(*ordering))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     "ordering report is missing the required ordering mapping schema",
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    m_gate_report.ordering_report_path = ordering_report_path.string();
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
            "M4 socu_approx skeleton only accepts StoreScalar == SolveScalar",
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
    }

    auto min_near_attr =
        config.find<Float>("linear_system/socu_approx/min_near_band_ratio");
    auto max_off_attr =
        config.find<Float>("linear_system/socu_approx/max_off_band_ratio");
    const double min_near_band_ratio =
        min_near_attr ? static_cast<double>(min_near_attr->view()[0]) : 0.0;
    const double max_off_band_ratio =
        max_off_attr ? static_cast<double>(max_off_attr->view()[0]) : 1.0;
    if(m_gate_report.near_band_ratio < min_near_band_ratio
       || m_gate_report.off_band_ratio > max_off_band_ratio)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingQualityTooLow,
            fmt::format("ordering quality rejected: near_band_ratio={}, off_band_ratio={}, "
                        "min_near_band_ratio={}, max_off_band_ratio={}",
                        m_gate_report.near_band_ratio,
                        m_gate_report.off_band_ratio,
                        min_near_band_ratio,
                        max_off_band_ratio),
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    const double near_band_ratio = m_gate_report.near_band_ratio;
    const double off_band_ratio  = m_gate_report.off_band_ratio;
    m_gate_report = make_failure(
        SocuApproxGateReason::StructuredProviderMissing,
        "M4 only registers the socu_approx gate; structured chain assembly is not implemented",
        ordering_report_path.string());
    m_gate_report.block_size      = ordering->at("block_size").get<SizeT>();
    m_gate_report.near_band_ratio = near_band_ratio;
    m_gate_report.off_band_ratio  = off_band_ratio;
    throw_gate_failure(m_gate_report);
}

void SocuApproxSolver::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    m_gate_report = make_failure(
        SocuApproxGateReason::StubNoDirection,
        "M4 socu_approx skeleton must not produce a linear solve direction");
    throw_gate_failure(m_gate_report);
}
}  // namespace uipc::backend::cuda_mixed
