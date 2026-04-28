#include <linear_system/socu_rcm_ordering.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace uipc::backend::cuda_mixed::socu_approx::rcm
{
namespace
{
using Clock = std::chrono::steady_clock;

SizeT abs_diff(SizeT a, SizeT b)
{
    return a > b ? a - b : b - a;
}

void validate_graph(const AtomGraph& graph)
{
    for(SizeT i = 0; i < graph.atoms.size(); ++i)
    {
        if(graph.atoms[i].id != i)
            throw std::invalid_argument("atom ids must be dense and match their vector index");
        if(graph.atoms[i].dof_count == 0)
            throw std::invalid_argument("atom dof_count must be positive");
    }

    std::set<std::pair<SizeT, SizeT>> seen;
    for(const auto& edge : graph.edges)
    {
        if(edge.a >= graph.atoms.size() || edge.b >= graph.atoms.size())
            throw std::invalid_argument("edge endpoint is out of range");
        if(edge.a == edge.b)
            throw std::invalid_argument("self edges are not allowed");
        if(edge.a > edge.b)
            throw std::invalid_argument("edges must be stored in canonical order");
        if(edge.weight <= 0.0)
            throw std::invalid_argument("edge weight must be positive");
        if(!seen.insert({edge.a, edge.b}).second)
            throw std::invalid_argument("duplicate edge found");
    }
}

std::vector<std::vector<SizeT>> adjacency(const AtomGraph& graph)
{
    validate_graph(graph);
    std::vector<std::vector<SizeT>> out(graph.atoms.size());
    for(const auto& edge : graph.edges)
    {
        out[edge.a].push_back(edge.b);
        out[edge.b].push_back(edge.a);
    }
    for(auto& row : out)
        std::sort(row.begin(), row.end());
    return out;
}

std::vector<SizeT> rcm_order_from_adjacency(const std::vector<std::vector<SizeT>>& adj)
{
    const SizeT n = adj.size();
    std::vector<SizeT> order;
    std::vector<char>  seen(n, 0);
    order.reserve(n);

    auto degree_less = [&adj](SizeT a, SizeT b)
    {
        const auto da = adj[a].size();
        const auto db = adj[b].size();
        return da == db ? a < b : da < db;
    };

    while(order.size() < n)
    {
        SizeT start = 0;
        while(start < n && seen[start])
            ++start;
        if(start == n)
            break;
        for(SizeT i = start + 1; i < n; ++i)
        {
            if(!seen[i] && degree_less(i, start))
                start = i;
        }

        std::queue<SizeT> queue;
        queue.push(start);
        seen[start] = 1;
        while(!queue.empty())
        {
            const SizeT u = queue.front();
            queue.pop();
            order.push_back(u);

            auto neighbors = adj[u];
            std::sort(neighbors.begin(), neighbors.end(), degree_less);
            for(const SizeT v : neighbors)
            {
                if(v < n && !seen[v])
                {
                    seen[v] = 1;
                    queue.push(v);
                }
            }
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

std::vector<SizeT> resolve_block_sizes(std::string_view value)
{
    if(value == "32")
        return {32};
    if(value == "64")
        return {64};
    if(value == "auto")
        return {32, 64};
    throw std::invalid_argument("ordering_block_size must be 32, 64, or auto");
}

void validate_permutation(const OrderingResult& ordering, SizeT atom_count)
{
    if(ordering.block_size != 32 && ordering.block_size != 64)
        throw std::invalid_argument("ordering block_size must be 32 or 64");
    if(ordering.chain_to_old.size() != atom_count
       || ordering.old_to_chain.size() != atom_count
       || ordering.atom_to_block.size() != atom_count
       || ordering.atom_block_offset.size() != atom_count
       || ordering.atom_dof_count.size() != atom_count)
    {
        throw std::invalid_argument("ordering mapping sizes must match atom count");
    }

    std::vector<char> seen(atom_count, 0);
    for(SizeT chain = 0; chain < atom_count; ++chain)
    {
        const SizeT old = ordering.chain_to_old[chain];
        if(old >= atom_count)
            throw std::invalid_argument("chain_to_old contains an out-of-range atom");
        if(seen[old])
            throw std::invalid_argument("chain_to_old contains a duplicate atom");
        seen[old] = 1;
        if(ordering.old_to_chain[old] != chain)
            throw std::invalid_argument("old_to_chain is not the inverse of chain_to_old");
    }
    for(SizeT old = 0; old < atom_count; ++old)
    {
        const SizeT chain = ordering.old_to_chain[old];
        if(chain >= atom_count || ordering.chain_to_old[chain] != old)
            throw std::invalid_argument("chain_to_old is not the inverse of old_to_chain");
    }
}

OrderingResult finalize_ordering(const AtomGraph&        graph,
                                 SizeT                   block_size,
                                 std::vector<SizeT>      chain_to_old)
{
    OrderingResult result;
    result.orderer      = "rcm";
    result.block_size   = block_size;
    result.chain_to_old = std::move(chain_to_old);
    result.old_to_chain.assign(result.chain_to_old.size(), 0);

    std::vector<char> seen(result.chain_to_old.size(), 0);
    for(SizeT chain = 0; chain < result.chain_to_old.size(); ++chain)
    {
        const SizeT old = result.chain_to_old[chain];
        if(old >= result.chain_to_old.size())
            throw std::runtime_error("ordering contains an out-of-range atom id");
        if(seen[old])
            throw std::runtime_error("ordering contains a duplicate atom id");
        seen[old] = 1;
        result.old_to_chain[old] = chain;
    }

    result.atom_to_block.assign(graph.atoms.size(), 0);
    result.atom_block_offset.assign(graph.atoms.size(), 0);
    result.atom_dof_count.assign(graph.atoms.size(), 0);

    SizeT chain_begin = 0;
    SizeT block_dofs  = 0;
    SizeT block       = 0;
    for(SizeT chain = 0; chain < result.chain_to_old.size(); ++chain)
    {
        const SizeT old  = result.chain_to_old[chain];
        const SizeT dofs = graph.atoms[old].dof_count;
        if(dofs > block_size)
            throw std::runtime_error("atom dof_count exceeds block_size");
        if(block_dofs != 0 && block_dofs + dofs > block_size)
        {
            result.block_to_atom_range.push_back(
                BlockRange{block, chain_begin, chain, block_dofs});
            ++block;
            chain_begin = chain;
            block_dofs  = 0;
        }

        result.atom_to_block[old]     = block;
        result.atom_block_offset[old] = block_dofs;
        result.atom_dof_count[old]    = dofs;
        block_dofs += dofs;
    }

    if(!result.chain_to_old.empty())
        result.block_to_atom_range.push_back(
            BlockRange{block, chain_begin, result.chain_to_old.size(), block_dofs});

    validate_permutation(result, graph.atoms.size());
    return result;
}

OrderingMetrics score_ordering(const AtomGraph&      graph,
                               const OrderingResult& ordering,
                               double                ordering_time_ms)
{
    OrderingMetrics metrics;
    metrics.total_atoms       = graph.atoms.size();
    metrics.block_size        = ordering.block_size;
    metrics.block_count       = ordering.block_to_atom_range.size();
    metrics.ordering_time_ms  = ordering_time_ms;
    metrics.valid_permutation = true;

    for(const auto& atom : graph.atoms)
        metrics.total_dofs += atom.dof_count;

    if(!ordering.block_to_atom_range.empty())
    {
        double utilization_sum = 0.0;
        metrics.min_block_utilization = std::numeric_limits<double>::max();
        for(const auto& block : ordering.block_to_atom_range)
        {
            const double utilization =
                static_cast<double>(block.dof_count) / ordering.block_size;
            utilization_sum += utilization;
            metrics.min_block_utilization =
                std::min(metrics.min_block_utilization, utilization);
        }
        metrics.avg_block_utilization =
            utilization_sum / static_cast<double>(ordering.block_to_atom_range.size());
    }

    double block_distance_weight_sum = 0.0;
    for(const auto& edge : graph.edges)
    {
        const SizeT block_a = ordering.atom_to_block[edge.a];
        const SizeT block_b = ordering.atom_to_block[edge.b];
        const SizeT dist    = abs_diff(block_a, block_b);
        metrics.max_block_distance = std::max(metrics.max_block_distance, dist);
        block_distance_weight_sum += static_cast<double>(dist) * edge.weight;
        metrics.edge_chain_span_sum +=
            abs_diff(ordering.old_to_chain[edge.a], ordering.old_to_chain[edge.b]);

        if(dist <= 1)
        {
            ++metrics.near_band_edge_count;
            metrics.near_band_edge_weight += edge.weight;
        }
        else
        {
            ++metrics.off_band_edge_count;
            metrics.off_band_edge_weight += edge.weight;
        }
    }

    const SizeT  edge_count =
        metrics.near_band_edge_count + metrics.off_band_edge_count;
    const double edge_weight =
        metrics.near_band_edge_weight + metrics.off_band_edge_weight;
    if(edge_count != 0)
    {
        metrics.near_band_ratio =
            static_cast<double>(metrics.near_band_edge_count) / edge_count;
        metrics.off_band_ratio =
            static_cast<double>(metrics.off_band_edge_count) / edge_count;
    }
    if(edge_weight > 0.0)
    {
        metrics.weighted_near_band_ratio =
            metrics.near_band_edge_weight / edge_weight;
        metrics.weighted_off_band_ratio =
            metrics.off_band_edge_weight / edge_weight;
        metrics.avg_block_distance = block_distance_weight_sum / edge_weight;
    }
    return metrics;
}

double candidate_score(const OrderingMetrics& metrics)
{
    const double normalized_max_distance =
        metrics.block_count > 1
            ? static_cast<double>(metrics.max_block_distance)
                  / static_cast<double>(metrics.block_count - 1)
            : 0.0;
    const double normalized_time =
        metrics.ordering_time_ms / (metrics.ordering_time_ms + 1.0);
    return 4.0 * metrics.weighted_near_band_ratio
           - 4.0 * metrics.weighted_off_band_ratio
           - 0.5 * normalized_max_distance
           + 1.0 * metrics.avg_block_utilization
           - 0.1 * normalized_time;
}

Json block_ranges_json(const std::vector<BlockRange>& ranges)
{
    Json out = Json::array();
    for(const auto& range : ranges)
    {
        out.push_back({{"block", range.block},
                       {"chain_begin", range.chain_begin},
                       {"chain_end", range.chain_end},
                       {"dof_count", range.dof_count}});
    }
    return out;
}

Json to_json(const OrderingResult& ordering)
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

Json to_json(const OrderingMetrics& metrics)
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

Json to_json(const OrderingCandidate& candidate)
{
    Json out = {{"orderer", candidate.orderer},
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
}  // namespace

void add_atom(AtomGraph&      graph,
              SizeT           dof_count,
              std::string_view source_kind,
              SizeT           source_id)
{
    const SizeT id = graph.atoms.size();
    if(source_id == static_cast<SizeT>(-1))
        source_id = id;
    graph.atoms.push_back(Atom{id, source_id, dof_count, std::string(source_kind)});
}

void add_edge(AtomGraph&      graph,
              SizeT           a,
              SizeT           b,
              double          weight,
              std::string_view kind)
{
    if(a == b)
        return;
    if(a > b)
        std::swap(a, b);
    if(weight <= 0.0)
        throw std::invalid_argument("edge weight must be positive");

    const AtomEdgeKey key{a, b};
    const auto        existing = graph.edge_to_index.find(key);
    if(existing != graph.edge_to_index.end())
    {
        auto& edge = graph.edges[existing->second];
        edge.weight += weight;
        if(edge.kind != kind)
            edge.kind += "+" + std::string(kind);
        return;
    }
    graph.edge_to_index.emplace(key, graph.edges.size());
    graph.edges.push_back(AtomEdge{a, b, weight, std::string(kind)});
}

OrderingRun run_ordering(const AtomGraph& graph,
                         std::string_view orderer,
                         std::string_view block_size)
{
    if(orderer != "rcm")
        throw std::invalid_argument(
            "linear_system/socu_approx/ordering_orderer only supports 'rcm'");

    validate_graph(graph);
    OrderingRun run;
    run.graph_name = graph.name;

    const auto start = Clock::now();
    const auto permutation = rcm_order_from_adjacency(adjacency(graph));
    const auto end = Clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    for(const SizeT block : resolve_block_sizes(block_size))
    {
        OrderingCandidate candidate;
        candidate.orderer    = "rcm";
        candidate.block_size = block;
        try
        {
            candidate.ordering = finalize_ordering(graph, block, permutation);
            candidate.metrics =
                score_ordering(graph, candidate.ordering, elapsed_ms);
            candidate.score = candidate_score(candidate.metrics);
            candidate.ok    = true;
        }
        catch(const std::exception& e)
        {
            candidate.ok              = false;
            candidate.fallback_reason = e.what();
        }
        run.candidates.push_back(std::move(candidate));
    }

    auto best = run.candidates.end();
    for(auto it = run.candidates.begin(); it != run.candidates.end(); ++it)
    {
        if(!it->ok)
            continue;
        if(best == run.candidates.end() || it->score > best->score)
            best = it;
    }
    if(best == run.candidates.end())
        throw std::runtime_error("all RCM ordering candidates failed");
    run.selected_candidate =
        static_cast<SizeT>(std::distance(run.candidates.begin(), best));
    return run;
}

Json to_json(const AtomGraph& graph)
{
    Json atoms = Json::array();
    for(const auto& atom : graph.atoms)
    {
        atoms.push_back({{"id", atom.id},
                         {"source_id", atom.source_id},
                         {"dof_count", atom.dof_count},
                         {"source_kind", atom.source_kind}});
    }

    Json edges = Json::array();
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

Json to_json(const OrderingRun& run)
{
    Json candidates = Json::array();
    for(SizeT i = 0; i < run.candidates.size(); ++i)
    {
        auto row = to_json(run.candidates[i]);
        row["selected"] = i == run.selected_candidate;
        candidates.push_back(std::move(row));
    }

    Json selected;
    if(run.selected_candidate < run.candidates.size())
        selected = to_json(run.candidates[run.selected_candidate]);

    return {{"graph_name", run.graph_name},
            {"selected_candidate", run.selected_candidate},
            {"selected", selected},
            {"candidates", candidates}};
}
}  // namespace uipc::backend::cuda_mixed::socu_approx::rcm
