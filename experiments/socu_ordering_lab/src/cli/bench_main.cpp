#include <sol/graph.h>
#include <sol/io.h>
#include <sol/ordering.h>
#include <sol/presets.h>
#include <sol/reorder.h>

#include <fmt/format.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
struct Args
{
    std::string command;
    std::unordered_map<std::string, std::string> values;
};

Args parse_args(int argc, char** argv)
{
    if(argc < 2)
        throw std::invalid_argument("usage: socu_ordering_bench <order|reorder> [options]");

    Args args;
    args.command = argv[1];
    for(int i = 2; i < argc; ++i)
    {
        std::string key = argv[i];
        if(!key.starts_with("--"))
            throw std::invalid_argument(fmt::format("unexpected positional argument '{}'", key));
        if(i + 1 >= argc)
            throw std::invalid_argument(fmt::format("missing value for '{}'", key));
        args.values[key.substr(2)] = argv[++i];
    }
    return args;
}

std::string get(const Args& args, std::string_view key, std::string fallback)
{
    auto it = args.values.find(std::string(key));
    return it == args.values.end() ? fallback : it->second;
}

std::size_t get_size(const Args& args, std::string_view key, std::size_t fallback)
{
    const auto value = get(args, key, "");
    return value.empty() ? fallback : static_cast<std::size_t>(std::stoull(value));
}

sol::AtomGraph load_graph(const Args& args)
{
    const auto input = get(args, "input", "");
    sol::AtomGraph graph = input.empty()
                               ? sol::make_preset(get(args, "preset", "rod"))
                               : sol::read_graph_json(input);

    const auto extra_edges = get(args, "extra-edges", "");
    if(!extra_edges.empty())
        sol::read_extra_edges_csv(graph, extra_edges);
    return graph;
}

sol::OrderingResult load_ordering(const Args& args)
{
    const auto path = get(args, "ordering", "");
    if(path.empty())
        throw std::invalid_argument("reorder requires --ordering <order-report.json>");

    std::ifstream in(path);
    if(!in)
        throw std::runtime_error(fmt::format("failed to open ordering JSON '{}'", path));

    nlohmann::json json;
    in >> json;
    return sol::ordering_from_json(json);
}

void write_reorder_summary_csv(const std::string& path, const sol::ReorderResult& result)
{
    if(path.empty())
        return;

    std::ofstream out(path);
    if(!out)
        throw std::runtime_error(fmt::format("failed to open reorder summary CSV '{}'", path));

    out << "mode,preset,vertices,edges,triangles,tetrahedra,counts_preserved,"
           "topology_indices_valid,original_vertex_id_complete,chain_metadata_valid,"
           "chain_index_stride_before,chain_index_stride_after,mirror_near_band_edges,"
           "mirror_off_band_edges,physical_near_band_edges,physical_off_band_edges,"
           "max_edge_length_error,max_triangle_area_error,max_tet_volume_error\n";
    out << result.mode << ','
        << result.preset << ','
        << result.after_counts.vertices << ','
        << result.after_counts.edges << ','
        << result.after_counts.triangles << ','
        << result.after_counts.tetrahedra << ','
        << (result.counts_preserved ? 1 : 0) << ','
        << (result.topology_indices_valid ? 1 : 0) << ','
        << (result.original_vertex_id_complete ? 1 : 0) << ','
        << (result.chain_metadata_valid ? 1 : 0) << ','
        << result.chain_index_stride_before << ','
        << result.chain_index_stride_after << ','
        << result.mirror_classification.near_band_edges << ','
        << result.mirror_classification.off_band_edges << ','
        << result.physical_classification.near_band_edges << ','
        << result.physical_classification.off_band_edges << ','
        << result.invariants.max_edge_length_error << ','
        << result.invariants.max_triangle_area_error << ','
        << result.invariants.max_tet_volume_error << '\n';
}

nlohmann::json timing_repeats_json(const std::vector<sol::OrderingRun>& runs,
                                   std::size_t warmup)
{
    nlohmann::json rows = nlohmann::json::array();
    if(runs.empty())
        return {{"warmup", warmup}, {"repeat", 0}, {"candidates", rows}};

    const auto& first = runs.front();
    for(std::size_t candidate = 0; candidate < first.candidates.size(); ++candidate)
    {
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0;
        double sum_ms = 0.0;
        std::size_t ok_count = 0;

        for(const auto& run : runs)
        {
            if(candidate >= run.candidates.size())
                continue;
            const auto& current = run.candidates[candidate];
            if(!current.ok)
                continue;
            const double time = current.metrics.ordering_time_ms;
            min_ms = std::min(min_ms, time);
            max_ms = std::max(max_ms, time);
            sum_ms += time;
            ++ok_count;
        }

        const auto& c = first.candidates[candidate];
        rows.push_back({{"orderer", c.orderer},
                        {"block_size", c.block_size},
                        {"ok_count", ok_count},
                        {"avg_ordering_time_ms", ok_count == 0 ? 0.0 : sum_ms / ok_count},
                        {"min_ordering_time_ms", ok_count == 0 ? 0.0 : min_ms},
                        {"max_ordering_time_ms", ok_count == 0 ? 0.0 : max_ms}});
    }
    return {{"warmup", warmup}, {"repeat", runs.size()}, {"candidates", rows}};
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args args = parse_args(argc, argv);
        if(args.command == "order")
        {
            const auto graph = load_graph(args);
            const auto orderer = get(args, "orderer", "auto");
            const auto block_size = get(args, "block-size", "auto");
            const auto warmup = get_size(args, "warmup", 0);
            const auto repeat = std::max<std::size_t>(1, get_size(args, "repeat", 1));

            for(std::size_t i = 0; i < warmup; ++i)
                (void)sol::run_ordering(graph, orderer, block_size);

            std::vector<sol::OrderingRun> runs;
            runs.reserve(repeat);
            for(std::size_t i = 0; i < repeat; ++i)
                runs.push_back(sol::run_ordering(graph, orderer, block_size));

            const auto& run = runs.back();

            auto report = sol::to_json(run);
            report["graph"] = sol::to_json(graph);
            if(warmup != 0 || repeat != 1)
                report["timing_repeats"] = timing_repeats_json(runs, warmup);

            sol::write_json_report(get(args, "report", get(args, "json", "-")), report);
            sol::write_summary_csv(get(args, "summary-csv", ""), run);
            sol::write_mapping_csv(get(args, "mapping-csv", ""), run.candidates.at(run.selected_candidate));
            return 0;
        }

        if(args.command == "reorder")
        {
            const auto preset = get(args, "preset", "rod");
            const auto geometry = sol::make_geometry_preset(preset);
            const auto ordering = load_ordering(args);
            const auto result = sol::reorder_geometry(geometry, ordering, get(args, "mode", "mirror"), preset);

            sol::write_json_report(get(args, "report", get(args, "json", "-")), sol::to_json(result));
            write_reorder_summary_csv(get(args, "summary-csv", ""), result);

            if(const auto mapping_csv = get(args, "mapping-csv", ""); !mapping_csv.empty())
            {
                sol::OrderingCandidate candidate;
                candidate.ok = true;
                candidate.ordering = ordering;
                sol::write_mapping_csv(mapping_csv, candidate);
            }
            return 0;
        }

        throw std::invalid_argument("command must be 'order' or 'reorder'");
    }
    catch(const std::exception& e)
    {
        fmt::print(stderr, "socu_ordering_bench: {}\n", e.what());
        return 2;
    }
}
