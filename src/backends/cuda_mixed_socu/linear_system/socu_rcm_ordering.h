#pragma once

#include <uipc/common/json.h>
#include <uipc/common/type_define.h>

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uipc::backend::cuda_mixed::socu_approx::rcm
{
struct Atom
{
    SizeT       id{};
    SizeT       source_id{};
    SizeT       dof_count{3};
    std::string source_kind{"synthetic"};
};

struct AtomEdge
{
    SizeT       a{};
    SizeT       b{};
    double      weight{1.0};
    std::string kind{"topology"};
};

struct AtomEdgeKey
{
    SizeT a{};
    SizeT b{};

    bool operator==(const AtomEdgeKey& other) const noexcept
    {
        return a == other.a && b == other.b;
    }
};

struct AtomEdgeKeyHash
{
    std::size_t operator()(const AtomEdgeKey& key) const noexcept
    {
        const auto ha = std::hash<SizeT>{}(key.a);
        const auto hb = std::hash<SizeT>{}(key.b);
        return ha ^ (hb + 0x9e3779b97f4a7c15ull + (ha << 6) + (ha >> 2));
    }
};

struct AtomGraph
{
    std::string                                            name;
    std::vector<Atom>                                      atoms;
    std::vector<AtomEdge>                                  edges;
    std::unordered_map<AtomEdgeKey, SizeT, AtomEdgeKeyHash> edge_to_index;
};

struct BlockRange
{
    SizeT block{};
    SizeT chain_begin{};
    SizeT chain_end{};
    SizeT dof_count{};
};

struct OrderingResult
{
    std::string        orderer;
    SizeT              block_size{};
    std::vector<SizeT> chain_to_old;
    std::vector<SizeT> old_to_chain;
    std::vector<SizeT> atom_to_block;
    std::vector<SizeT> atom_block_offset;
    std::vector<SizeT> atom_dof_count;
    std::vector<BlockRange> block_to_atom_range;
};

struct OrderingMetrics
{
    SizeT  total_atoms{};
    SizeT  total_dofs{};
    SizeT  block_size{};
    SizeT  block_count{};
    SizeT  near_band_edge_count{};
    SizeT  off_band_edge_count{};
    double near_band_edge_weight{};
    double off_band_edge_weight{};
    double near_band_ratio{};
    double off_band_ratio{};
    double weighted_near_band_ratio{};
    double weighted_off_band_ratio{};
    SizeT  max_block_distance{};
    double avg_block_distance{};
    SizeT  edge_chain_span_sum{};
    double avg_block_utilization{};
    double min_block_utilization{};
    double ordering_time_ms{};
    bool   valid_permutation{};
};

struct OrderingCandidate
{
    std::string     orderer;
    SizeT           block_size{};
    bool            ok{};
    std::string     fallback_reason;
    double          score{};
    OrderingResult  ordering;
    OrderingMetrics metrics;
};

struct OrderingRun
{
    std::string                    graph_name;
    SizeT                          selected_candidate{};
    std::vector<OrderingCandidate> candidates;
};

void add_atom(AtomGraph&      graph,
              SizeT           dof_count = 3,
              std::string_view source_kind = "synthetic",
              SizeT           source_id = static_cast<SizeT>(-1));

void add_edge(AtomGraph&      graph,
              SizeT           a,
              SizeT           b,
              double          weight = 1.0,
              std::string_view kind = "topology");

OrderingRun run_ordering(const AtomGraph& graph,
                         std::string_view orderer,
                         std::string_view block_size);

Json to_json(const AtomGraph& graph);
Json to_json(const OrderingRun& run);
}  // namespace uipc::backend::cuda_mixed::socu_approx::rcm
