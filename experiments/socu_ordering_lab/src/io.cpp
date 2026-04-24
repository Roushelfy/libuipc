#include <sol/io.h>

#include <fmt/format.h>

#include <fstream>
#include <stdexcept>
#include <string>

namespace sol
{
namespace
{
nlohmann::json block_ranges_json(const std::vector<BlockRange>& ranges)
{
    nlohmann::json out = nlohmann::json::array();
    for(const auto& range : ranges)
    {
        out.push_back({{"block", range.block},
                       {"chain_begin", range.chain_begin},
                       {"chain_end", range.chain_end},
                       {"dof_count", range.dof_count}});
    }
    return out;
}

std::string csv_escape(const std::string& value)
{
    if(value.find_first_of(",\"\n") == std::string::npos)
        return value;

    std::string out = "\"";
    for(const char c : value)
    {
        if(c == '"')
            out += "\"\"";
        else
            out += c;
    }
    out += '"';
    return out;
}
} // namespace

nlohmann::json to_json(const AtomGraph& graph)
{
    nlohmann::json atoms = nlohmann::json::array();
    for(const auto& atom : graph.atoms)
    {
        atoms.push_back({{"id", atom.id},
                         {"source_id", atom.source_id},
                         {"dof_count", atom.dof_count},
                         {"source_kind", atom.source_kind}});
    }

    nlohmann::json edges = nlohmann::json::array();
    for(const auto& edge : graph.edges)
    {
        edges.push_back({{"a", edge.a},
                         {"b", edge.b},
                         {"weight", edge.weight},
                         {"kind", edge.kind}});
    }

    return {{"name", graph.name},
            {"atom_count", graph.atoms.size()},
            {"edge_count", graph.edges.size()},
            {"atoms", atoms},
            {"edges", edges}};
}

nlohmann::json to_json(const OrderingResult& ordering)
{
    return {{"orderer", ordering.orderer},
            {"block_size", ordering.block_size},
            {"chain_to_old", ordering.chain_to_old},
            {"old_to_chain", ordering.old_to_chain},
            {"atom_to_block", ordering.atom_to_block},
            {"atom_block_offset", ordering.atom_block_offset},
            {"atom_dof_count", ordering.atom_dof_count},
            {"block_to_atom_range", block_ranges_json(ordering.block_to_atom_range)}};
}

nlohmann::json to_json(const OrderingMetrics& metrics)
{
    return {{"total_atoms", metrics.total_atoms},
            {"total_dofs", metrics.total_dofs},
            {"block_size", metrics.block_size},
            {"block_count", metrics.block_count},
            {"avg_block_utilization", metrics.avg_block_utilization},
            {"min_block_utilization", metrics.min_block_utilization},
            {"near_band_edge_count", metrics.near_band_edge_count},
            {"off_band_edge_count", metrics.off_band_edge_count},
            {"near_band_edge_weight", metrics.near_band_edge_weight},
            {"off_band_edge_weight", metrics.off_band_edge_weight},
            {"near_band_ratio", metrics.near_band_ratio},
            {"off_band_ratio", metrics.off_band_ratio},
            {"weighted_near_band_ratio", metrics.weighted_near_band_ratio},
            {"weighted_off_band_ratio", metrics.weighted_off_band_ratio},
            {"max_block_distance", metrics.max_block_distance},
            {"avg_block_distance", metrics.avg_block_distance},
            {"edge_chain_span_sum", metrics.edge_chain_span_sum},
            {"ordering_time_ms", metrics.ordering_time_ms},
            {"valid_permutation", metrics.valid_permutation}};
}

nlohmann::json to_json(const OrderingCandidate& candidate)
{
    nlohmann::json out = {{"orderer", candidate.orderer},
                          {"block_size", candidate.block_size},
                          {"ok", candidate.ok},
                          {"score", candidate.score},
                          {"fallback_reason", candidate.fallback_reason}};
    if(candidate.ok)
    {
        out["ordering"] = to_json(candidate.ordering);
        out["metrics"]  = to_json(candidate.metrics);
    }
    return out;
}

nlohmann::json to_json(const OrderingRun& run)
{
    nlohmann::json candidates = nlohmann::json::array();
    for(std::size_t i = 0; i < run.candidates.size(); ++i)
    {
        auto row = to_json(run.candidates[i]);
        row["selected"] = i == run.selected_candidate;
        candidates.push_back(std::move(row));
    }

    nlohmann::json selected;
    if(run.selected_candidate < run.candidates.size())
        selected = to_json(run.candidates[run.selected_candidate]);

    return {{"graph_name", run.graph_name},
            {"selected_candidate", run.selected_candidate},
            {"selected", selected},
            {"candidates", candidates}};
}

OrderingResult ordering_from_json(const nlohmann::json& json)
{
    const nlohmann::json* source = &json;
    if(json.contains("selected"))
        source = &json.at("selected");
    if(source->contains("ordering"))
        source = &source->at("ordering");

    OrderingResult ordering;
    ordering.orderer           = source->value("orderer", "unknown");
    ordering.block_size        = source->at("block_size").get<std::size_t>();
    ordering.chain_to_old      = source->at("chain_to_old").get<std::vector<std::size_t>>();
    ordering.old_to_chain      = source->at("old_to_chain").get<std::vector<std::size_t>>();
    ordering.atom_to_block     = source->at("atom_to_block").get<std::vector<std::size_t>>();
    ordering.atom_block_offset = source->at("atom_block_offset").get<std::vector<std::size_t>>();
    ordering.atom_dof_count =
        source->contains("atom_dof_count")
            ? source->at("atom_dof_count").get<std::vector<std::size_t>>()
            : std::vector<std::size_t>(ordering.chain_to_old.size(), 3);

    ordering.block_to_atom_range.clear();
    if(source->contains("block_to_atom_range"))
    {
        for(const auto& row : source->at("block_to_atom_range"))
        {
            ordering.block_to_atom_range.push_back(BlockRange{row.at("block").get<std::size_t>(),
                                                              row.at("chain_begin").get<std::size_t>(),
                                                              row.at("chain_end").get<std::size_t>(),
                                                              row.at("dof_count").get<std::size_t>()});
        }
    }
    return ordering;
}

void write_json_report(const std::filesystem::path& path, const nlohmann::json& json)
{
    if(path.empty() || path == "-")
    {
        fmt::print("{}\n", json.dump(2));
        return;
    }

    std::ofstream out(path);
    if(!out)
        throw std::runtime_error(fmt::format("failed to open JSON report '{}'", path.string()));
    out << json.dump(2) << '\n';
}

void write_summary_csv(const std::filesystem::path& path, const OrderingRun& run)
{
    if(path.empty())
        return;

    std::ofstream out(path);
    if(!out)
        throw std::runtime_error(fmt::format("failed to open summary CSV '{}'", path.string()));

    out << "selected,orderer,block_size,ok,score,total_atoms,total_dofs,block_count,"
           "avg_block_utilization,min_block_utilization,near_band_edge_count,"
           "off_band_edge_count,near_band_ratio,off_band_ratio,"
           "weighted_near_band_ratio,weighted_off_band_ratio,max_block_distance,"
           "avg_block_distance,edge_chain_span_sum,ordering_time_ms,fallback_reason\n";

    for(std::size_t i = 0; i < run.candidates.size(); ++i)
    {
        const auto& c = run.candidates[i];
        const auto& m = c.metrics;
        out << (i == run.selected_candidate ? 1 : 0) << ','
            << c.orderer << ','
            << c.block_size << ','
            << (c.ok ? 1 : 0) << ','
            << c.score << ','
            << m.total_atoms << ','
            << m.total_dofs << ','
            << m.block_count << ','
            << m.avg_block_utilization << ','
            << m.min_block_utilization << ','
            << m.near_band_edge_count << ','
            << m.off_band_edge_count << ','
            << m.near_band_ratio << ','
            << m.off_band_ratio << ','
            << m.weighted_near_band_ratio << ','
            << m.weighted_off_band_ratio << ','
            << m.max_block_distance << ','
            << m.avg_block_distance << ','
            << m.edge_chain_span_sum << ','
            << m.ordering_time_ms << ','
            << csv_escape(c.fallback_reason) << '\n';
    }
}

void write_mapping_csv(const std::filesystem::path& path, const OrderingCandidate& candidate)
{
    if(path.empty())
        return;
    if(!candidate.ok)
        throw std::invalid_argument("cannot write mapping CSV for a failed candidate");

    std::ofstream out(path);
    if(!out)
        throw std::runtime_error(fmt::format("failed to open mapping CSV '{}'", path.string()));

    out << "old_atom,chain_atom,block,block_offset,dof_count\n";
    for(std::size_t old = 0; old < candidate.ordering.old_to_chain.size(); ++old)
    {
        const std::size_t chain = candidate.ordering.old_to_chain[old];
        out << old << ','
            << chain << ','
            << candidate.ordering.atom_to_block[old] << ','
            << candidate.ordering.atom_block_offset[old] << ','
            << candidate.ordering.atom_dof_count[old] << '\n';
    }
}
} // namespace sol
