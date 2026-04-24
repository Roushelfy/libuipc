#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sol
{
struct Atom
{
    std::size_t id{};
    std::size_t source_id{};
    std::size_t dof_count{3};
    std::string source_kind{"synthetic"};
};

struct AtomEdge
{
    std::size_t a{};
    std::size_t b{};
    double      weight{1.0};
    std::string kind{"topology"};
};

struct AtomGraph
{
    std::string           name;
    std::vector<Atom>     atoms;
    std::vector<AtomEdge> edges;
};

struct BlockRange
{
    std::size_t block{};
    std::size_t chain_begin{};
    std::size_t chain_end{};
    std::size_t dof_count{};
};

struct OrderingResult
{
    std::string              orderer;
    std::size_t              block_size{};
    std::vector<std::size_t> chain_to_old;
    std::vector<std::size_t> old_to_chain;
    std::vector<std::size_t> atom_to_block;
    std::vector<std::size_t> atom_block_offset;
    std::vector<std::size_t> atom_dof_count;
    std::vector<BlockRange>  block_to_atom_range;
};

struct OrderingMetrics
{
    std::size_t total_atoms{};
    std::size_t total_dofs{};
    std::size_t block_size{};
    std::size_t block_count{};
    std::size_t near_band_edge_count{};
    std::size_t off_band_edge_count{};
    double      near_band_edge_weight{};
    double      off_band_edge_weight{};
    double      near_band_ratio{};
    double      off_band_ratio{};
    double      weighted_near_band_ratio{};
    double      weighted_off_band_ratio{};
    std::size_t max_block_distance{};
    double      avg_block_distance{};
    std::size_t edge_chain_span_sum{};
    double      avg_block_utilization{};
    double      min_block_utilization{};
    double      ordering_time_ms{};
    bool        valid_permutation{};
};

struct OrderingCandidate
{
    std::string     orderer;
    std::size_t     block_size{};
    bool            ok{};
    std::string     fallback_reason;
    double          score{};
    OrderingResult  ordering;
    OrderingMetrics metrics;
};

struct OrderingRun
{
    std::string                    graph_name;
    std::size_t                    selected_candidate{};
    std::vector<OrderingCandidate> candidates;
};
} // namespace sol
