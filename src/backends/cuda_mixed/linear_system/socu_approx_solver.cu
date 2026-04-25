#include <linear_system/socu_approx_solver.h>

#include <linear_system/global_linear_system.h>
#include <linear_system/structured_chain_provider.h>
#include <mixed_precision/policy.h>
#include <sim_engine.h>
#include <uipc/common/exception.h>
#include <uipc/common/json.h>
#include <uipc/common/timer.h>

#include <fmt/format.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

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

fs::path absolute_workspace_path(std::string_view workspace, const std::string& path)
{
    fs::path p{path};
    if(p.is_relative())
        p = fs::absolute(fs::path{workspace} / p);
    return p;
}

bool read_size_t_field(const Json& json, const char* key, SizeT& value)
{
    auto it = json.find(key);
    if(it == json.end()
       || !(it->is_number_unsigned() || it->is_number_integer()))
        return false;
    value = it->get<SizeT>();
    return true;
}

bool parse_atom_dof_count(const Json& ordering, SizeT& dof_count, std::string& detail)
{
    dof_count = 0;
    auto it = ordering.find("atom_dof_count");
    if(it == ordering.end() || !it->is_array())
    {
        detail = "ordering atom_dof_count must be an array";
        return false;
    }

    for(const Json& value : *it)
    {
        if(!(value.is_number_unsigned() || value.is_number_integer()))
        {
            detail = "ordering atom_dof_count contains a non-integer value";
            return false;
        }
        dof_count += value.get<SizeT>();
    }
    return true;
}

bool parse_size_t_array(const Json&             ordering,
                        const char*            key,
                        std::vector<SizeT>&    values,
                        std::string&           detail)
{
    values.clear();
    auto it = ordering.find(key);
    if(it == ordering.end() || !it->is_array())
    {
        detail = fmt::format("ordering {} must be an array", key);
        return false;
    }

    values.reserve(it->size());
    for(const Json& value : *it)
    {
        if(!(value.is_number_unsigned() || value.is_number_integer()))
        {
            detail = fmt::format("ordering {} contains a non-integer value", key);
            return false;
        }
        values.push_back(value.get<SizeT>());
    }
    return true;
}

bool parse_block_layouts(const Json&                         ordering,
                         std::vector<SocuApproxBlockLayout>& blocks,
                         std::string&                       detail)
{
    blocks.clear();
    const auto& block_ranges = ordering.at("block_to_atom_range");
    SizeT       dof_offset   = 0;

    for(const Json& entry : block_ranges)
    {
        if(!entry.is_object())
        {
            detail = "ordering block_to_atom_range entries must be objects";
            return false;
        }

        SocuApproxBlockLayout block;
        if(!read_size_t_field(entry, "block", block.block)
           || !read_size_t_field(entry, "chain_begin", block.chain_begin)
           || !read_size_t_field(entry, "chain_end", block.chain_end)
           || !read_size_t_field(entry, "dof_count", block.dof_count))
        {
            detail =
                "ordering block_to_atom_range entries must contain block, chain_begin, chain_end, and dof_count";
            return false;
        }

        if(block.chain_end < block.chain_begin)
        {
            detail = "ordering block_to_atom_range has chain_end < chain_begin";
            return false;
        }

        block.dof_offset = dof_offset;
        dof_offset += block.dof_count;
        blocks.push_back(block);
    }

    return !blocks.empty();
}

class OrderingStructuredChainProvider final : public StructuredChainProvider
{
  public:
    OrderingStructuredChainProvider(SizeT                       block_size,
                                    std::vector<StructuredDofSlot> slots,
                                    double                      block_utilization)
        : m_slots(std::move(slots))
    {
        m_shape.horizon                  = m_slots.empty() ? 0 : (m_slots.back().block + 1);
        m_shape.block_size               = block_size;
        m_shape.nrhs                     = 1;
        m_shape.symmetric_positive_definite = true;
        m_quality.block_utilization      = block_utilization;
    }

    bool is_available() const override { return !m_slots.empty(); }
    StructuredChainShape shape() const override { return m_shape; }
    span<const StructuredDofSlot> dof_slots() const override { return m_slots; }
    StructuredQualityReport quality_report() const override { return m_quality; }
    void assemble_chain(StructuredAssemblySink&) override {}

  private:
    StructuredChainShape          m_shape;
    StructuredQualityReport       m_quality;
    std::vector<StructuredDofSlot> m_slots;
};

bool build_ordering_provider(const Json&                         ordering,
                             SizeT                               block_size,
                             const std::vector<SocuApproxBlockLayout>& blocks,
                             std::unique_ptr<StructuredChainProvider>& provider,
                             SizeT&                              padding_slot_count,
                             std::string&                        detail)
{
    std::vector<SizeT> chain_to_old;
    std::vector<SizeT> atom_to_block;
    std::vector<SizeT> atom_block_offset;
    std::vector<SizeT> atom_dof_count;
    if(!parse_size_t_array(ordering, "chain_to_old", chain_to_old, detail)
       || !parse_size_t_array(ordering, "atom_to_block", atom_to_block, detail)
       || !parse_size_t_array(ordering, "atom_block_offset", atom_block_offset, detail)
       || !parse_size_t_array(ordering, "atom_dof_count", atom_dof_count, detail))
    {
        return false;
    }

    if(atom_to_block.size() != chain_to_old.size()
       || atom_block_offset.size() != chain_to_old.size()
       || atom_dof_count.size() != chain_to_old.size())
    {
        detail = "ordering mapping arrays have inconsistent atom counts";
        return false;
    }

    std::vector<SizeT> old_dof_offsets(atom_dof_count.size(), 0);
    SizeT              old_dof_offset = 0;
    for(SizeT old = 0; old < atom_dof_count.size(); ++old)
    {
        old_dof_offsets[old] = old_dof_offset;
        old_dof_offset += atom_dof_count[old];
    }

    std::vector<SizeT> valid_lanes_per_block(blocks.size(), 0);
    for(const auto& block : blocks)
    {
        if(block.block >= valid_lanes_per_block.size() || block.dof_count > block_size)
        {
            detail = "ordering block layout is inconsistent with block_size";
            return false;
        }
        valid_lanes_per_block[block.block] = block.dof_count;
    }

    std::vector<StructuredDofSlot> slots;
    slots.reserve(blocks.size() * block_size);
    for(SizeT chain = 0; chain < chain_to_old.size(); ++chain)
    {
        const SizeT old = chain_to_old[chain];
        if(old >= atom_dof_count.size())
        {
            detail = "ordering chain_to_old contains an out-of-range atom";
            return false;
        }

        const SizeT block = atom_to_block[old];
        const SizeT lane_begin = atom_block_offset[old];
        const SizeT dofs = atom_dof_count[old];
        if(block >= blocks.size() || lane_begin + dofs > block_size)
        {
            detail = "ordering atom block/lane mapping exceeds block_size";
            return false;
        }

        for(SizeT local_dof = 0; local_dof < dofs; ++local_dof)
        {
            const SizeT lane = lane_begin + local_dof;
            slots.push_back(StructuredDofSlot{
                .old_dof = static_cast<IndexT>(old_dof_offsets[old] + local_dof),
                .chain_dof = static_cast<IndexT>(block * block_size + lane),
                .block = block,
                .lane = lane,
                .is_padding = false,
                .scatter_write = true});
        }
    }

    padding_slot_count = 0;
    for(SizeT block = 0; block < valid_lanes_per_block.size(); ++block)
    {
        for(SizeT lane = valid_lanes_per_block[block]; lane < block_size; ++lane)
        {
            slots.push_back(StructuredDofSlot{.old_dof = -1,
                                             .chain_dof = static_cast<IndexT>(
                                                 block * block_size + lane),
                                             .block = block,
                                             .lane = lane,
                                             .is_padding = true,
                                             .scatter_write = false});
            ++padding_slot_count;
        }
    }

    const double padded_lanes =
        static_cast<double>(valid_lanes_per_block.size() * block_size);
    const double utilization =
        padded_lanes > 0.0
            ? static_cast<double>(old_dof_offset) / padded_lanes
            : 0.0;
    provider = std::make_unique<OrderingStructuredChainProvider>(
        block_size,
        std::move(slots),
        utilization);
    return provider->is_available();
}

SizeT diag_scalar_count(const std::vector<SocuApproxBlockLayout>& blocks)
{
    SizeT count = 0;
    for(const auto& block : blocks)
        count += block.dof_count * block.dof_count;
    return count;
}

SizeT first_offdiag_scalar_count(const std::vector<SocuApproxBlockLayout>& blocks)
{
    SizeT count = 0;
    for(SizeT i = 1; i < blocks.size(); ++i)
        count += blocks[i - 1].dof_count * blocks[i].dof_count;
    return count;
}

SizeT optional_size_t(const Json& json, const char* key)
{
    SizeT value = 0;
    if(read_size_t_field(json, key, value))
        return value;
    return 0;
}

double optional_double(const Json& json, const char* key)
{
    auto it = json.find(key);
    if(it == json.end() || !it->is_number())
        return 0.0;
    return it->get<double>();
}

void load_contact_report(SocuApproxDryRunReport& dry_run)
{
    if(dry_run.contact_report_path.empty())
        return;

    std::ifstream ifs{dry_run.contact_report_path};
    if(!ifs)
        throw Exception{fmt::format("SocuApproxSolver contact report '{}' cannot be opened",
                                    dry_run.contact_report_path)};

    Json report = Json::parse(ifs, nullptr, false);
    if(report.is_discarded() || !report.is_object())
        throw Exception{fmt::format("SocuApproxSolver contact report '{}' is not valid JSON",
                                    dry_run.contact_report_path)};

    const Json& contact =
        report.contains("contact") && report["contact"].is_object() ? report["contact"] : report;
    dry_run.near_band_contact_count =
        optional_size_t(contact, "near_band_contact_count");
    dry_run.off_band_contact_count =
        optional_size_t(contact, "off_band_contact_count");
    dry_run.near_band_contribution_count =
        optional_size_t(contact, "near_band_contribution_count");
    dry_run.off_band_contribution_count =
        optional_size_t(contact, "off_band_contribution_count");
    dry_run.absorbed_hessian_contribution_count =
        optional_size_t(contact, "estimated_absorbed_hessian_contribution_count");
    dry_run.dropped_hessian_contribution_count =
        optional_size_t(contact, "estimated_dropped_contribution_count");
    dry_run.contribution_near_band_ratio =
        optional_double(contact, "contribution_near_band_ratio");
    dry_run.contribution_off_band_ratio =
        optional_double(contact, "contribution_off_band_ratio");
    dry_run.weighted_near_band_ratio =
        optional_double(contact, "weighted_near_band_ratio");
    dry_run.weighted_off_band_ratio =
        optional_double(contact, "weighted_off_band_ratio");
}

Json to_json(const SocuApproxDryRunReport& report)
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
                {"milestone", 5},
                {"mode", "structured_dry_run"},
                {"packed", report.packed},
                {"ordering_report", report.ordering_report_path},
                {"contact_report", report.contact_report_path},
                {"block_size", report.block_size},
                {"chain_atom_count", report.chain_atom_count},
                {"ordering_dof_count", report.ordering_dof_count},
                {"structured_slot_count", report.structured_slot_count},
                {"padding_slot_count", report.padding_slot_count},
                {"block_utilization", report.block_utilization},
                {"layout",
                 {{"block_count", report.block_count},
                  {"diag_block_count", report.diag_block_count},
                  {"first_offdiag_block_count", report.first_offdiag_block_count},
                  {"rhs_scalar_count", report.rhs_scalar_count},
                  {"diag_scalar_count", report.diag_scalar_count},
                  {"first_offdiag_scalar_count", report.first_offdiag_scalar_count},
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
                  {"weighted_off_band_ratio", report.weighted_off_band_ratio}}},
                {"timing", {{"dry_run_pack_time_ms", report.dry_run_pack_time_ms}}},
                {"status",
                 {{"direction_available", false},
                  {"reason", std::string{to_string(SocuApproxGateReason::StubNoDirection)}}}}};
}

void write_dry_run_report(const SocuApproxDryRunReport& report)
{
    fs::path report_path{report.report_path};
    fs::create_directories(report_path.parent_path());
    std::ofstream ofs{report_path};
    if(!ofs)
        throw Exception{fmt::format("SocuApproxSolver dry-run report '{}' cannot be written",
                                    report.report_path)};
    ofs << to_json(report).dump(2);
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

    std::string                       ordering_detail;
    std::vector<SocuApproxBlockLayout> block_layouts;
    SizeT                             ordering_dof_count = 0;
    if(!parse_atom_dof_count(*ordering, ordering_dof_count, ordering_detail)
       || !parse_block_layouts(*ordering, block_layouts, ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
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
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     ordering_detail.empty()
                                         ? "ordering report cannot build a structured provider"
                                         : ordering_detail,
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

    auto dry_run_report_attr =
        config.find<std::string>("linear_system/socu_approx/dry_run_report");
    std::string dry_run_report =
        dry_run_report_attr ? dry_run_report_attr->view()[0] : std::string{};
    fs::path dry_run_report_path =
        dry_run_report.empty()
            ? fs::absolute(fs::path{workspace()} / "socu_approx_dry_run_report.json")
            : absolute_workspace_path(workspace(), dry_run_report);

    auto contact_report_attr =
        config.find<std::string>("linear_system/socu_approx/contact_report");
    std::string contact_report =
        contact_report_attr ? contact_report_attr->view()[0] : std::string{};

    m_dry_run_report                         = SocuApproxDryRunReport{};
    m_dry_run_report.report_path             = dry_run_report_path.string();
    m_dry_run_report.ordering_report_path    = ordering_report_path.string();
    m_dry_run_report.contact_report_path =
        contact_report.empty() ? std::string{}
                               : absolute_workspace_path(workspace(), contact_report).string();
    m_dry_run_report.block_size         = m_gate_report.block_size;
    m_dry_run_report.block_count        = block_layouts.size();
    m_dry_run_report.chain_atom_count   = ordering->at("chain_to_old").size();
    m_dry_run_report.ordering_dof_count = ordering_dof_count;
    m_dry_run_report.structured_slot_count = provider->dof_slots().size();
    m_dry_run_report.padding_slot_count = padding_slot_count;
    m_dry_run_report.block_utilization =
        provider->quality_report().block_utilization;
    m_dry_run_report.blocks             = std::move(block_layouts);

    m_gate_report.passed = true;
    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail =
        "M5 structured dry-run gate passed; solve direction is intentionally unavailable";

    logger::info("SocuApproxSolver M5 dry-run enabled: block_size={}, blocks={}, report='{}'",
                 m_dry_run_report.block_size,
                 m_dry_run_report.block_count,
                 m_dry_run_report.report_path);
}

void SocuApproxSolver::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    {
        Timer timer{"SocuApprox Dry Run Pack"};
        auto  start = std::chrono::steady_clock::now();

        m_dry_run_report.packed                    = true;
        m_dry_run_report.rhs_scalar_count          = info.b().size();
        m_dry_run_report.diag_block_count          = m_dry_run_report.blocks.size();
        m_dry_run_report.first_offdiag_block_count =
            m_dry_run_report.blocks.empty() ? 0 : m_dry_run_report.blocks.size() - 1;
        m_dry_run_report.diag_scalar_count =
            diag_scalar_count(m_dry_run_report.blocks);
        m_dry_run_report.first_offdiag_scalar_count =
            first_offdiag_scalar_count(m_dry_run_report.blocks);

        load_contact_report(m_dry_run_report);

        auto stop = std::chrono::steady_clock::now();
        m_dry_run_report.dry_run_pack_time_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
    }

    write_dry_run_report(m_dry_run_report);

    m_gate_report.reason = SocuApproxGateReason::StubNoDirection;
    m_gate_report.detail =
        "M5 structured dry-run pack completed, but no socu solve direction is available yet";
    info.x().buffer_view().fill(static_cast<GlobalLinearSystem::SolveScalar>(0));
    info.iter_count(0);

    logger::warn("SocuApproxSolver M5 dry-run completed without a solve direction; "
                 "returning a zero direction. report='{}'",
                 m_dry_run_report.report_path);
}
}  // namespace uipc::backend::cuda_mixed
