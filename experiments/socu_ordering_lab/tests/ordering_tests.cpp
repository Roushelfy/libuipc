#include <sol/graph.h>
#include <sol/ordering.h>
#include <sol/presets.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <numeric>
#include <string>

namespace
{
const sol::OrderingCandidate& selected(const sol::OrderingRun& run)
{
    REQUIRE(run.selected_candidate < run.candidates.size());
    return run.candidates[run.selected_candidate];
}

const sol::OrderingCandidate& require_candidate(const sol::OrderingRun& run,
                                                std::string_view orderer,
                                                std::size_t block_size)
{
    const auto it = std::find_if(run.candidates.begin(),
                                 run.candidates.end(),
                                 [&](const sol::OrderingCandidate& candidate)
                                 {
                                     return candidate.orderer == orderer
                                            && candidate.block_size == block_size;
                                 });
    REQUIRE(it != run.candidates.end());
    REQUIRE(it->ok);
    return *it;
}

sol::AtomGraph make_shuffled_rod()
{
    auto graph = sol::make_rod(128);
    std::vector<std::size_t> new_to_old(graph.atoms.size());
    std::iota(new_to_old.begin(), new_to_old.end(), 0);
    for(std::size_t i = 0; i < new_to_old.size(); ++i)
        std::swap(new_to_old[i], new_to_old[(i * 37 + 11) % new_to_old.size()]);
    return sol::relabel_graph(graph, new_to_old);
}

sol::AtomGraph make_empty_graph(std::size_t atom_count, std::string name)
{
    sol::AtomGraph graph;
    graph.name = std::move(name);
    for(std::size_t i = 0; i < atom_count; ++i)
        sol::add_atom(graph);
    return graph;
}

std::size_t chain_distance(const sol::OrderingResult& ordering,
                           std::size_t a,
                           std::size_t b)
{
    const auto ca = ordering.old_to_chain.at(a);
    const auto cb = ordering.old_to_chain.at(b);
    return ca > cb ? ca - cb : cb - ca;
}
} // namespace

TEST_CASE("permutation validation rejects malformed mappings", "[socu][ordering]")
{
    const auto graph = sol::make_rod(4);
    auto valid = sol::make_ordering_candidate(graph, "original", 32);
    REQUIRE(valid.ok);
    REQUIRE_NOTHROW(sol::validate_permutation(valid.ordering, graph.atoms.size()));

    auto duplicate = valid.ordering;
    duplicate.chain_to_old[1] = duplicate.chain_to_old[0];
    REQUIRE_THROWS(sol::validate_permutation(duplicate, graph.atoms.size()));

    auto missing_inverse = valid.ordering;
    missing_inverse.old_to_chain[0] = 3;
    REQUIRE_THROWS(sol::validate_permutation(missing_inverse, graph.atoms.size()));
}

TEST_CASE("weighted RCM matches RCM when all edge weights are uniform",
          "[socu][ordering][weighted_rcm]")
{
    auto graph = make_empty_graph(14, "uniform_weight_ladder");
    for(std::size_t i = 0; i + 1 < graph.atoms.size(); ++i)
        sol::add_edge(graph, i, i + 1, 1.0, "path");
    for(std::size_t i = 0; i + 2 < graph.atoms.size(); ++i)
        sol::add_edge(graph, i, i + 2, 1.0, "skip");

    const auto rcm = sol::make_ordering_candidate(graph, "rcm", 32);
    const auto weighted = sol::make_ordering_candidate(graph, "weighted_rcm", 32);

    REQUIRE(rcm.ok);
    REQUIRE(weighted.ok);
    CHECK(weighted.ordering.chain_to_old == rcm.ordering.chain_to_old);
    CHECK(weighted.metrics.valid_permutation);
}

TEST_CASE("weighted RCM uses weighted degree to break component root ties",
          "[socu][ordering][weighted_rcm]")
{
    auto graph = make_empty_graph(4, "weighted_root_tie");
    sol::add_edge(graph, 0, 1, 1.0, "weak_component");
    sol::add_edge(graph, 2, 3, 10.0, "strong_component");

    const auto rcm = sol::make_ordering_candidate(graph, "rcm", 32);
    const auto weighted = sol::make_ordering_candidate(graph, "weighted_rcm", 32);

    REQUIRE(rcm.ok);
    REQUIRE(weighted.ok);
    CHECK(rcm.ordering.chain_to_old == std::vector<std::size_t>{3, 2, 1, 0});
    CHECK(weighted.ordering.chain_to_old == std::vector<std::size_t>{1, 0, 3, 2});
    CHECK(weighted.ordering.chain_to_old != rcm.ordering.chain_to_old);
}

TEST_CASE("weighted RCM uses edge weight to break same-degree neighbor ties",
          "[socu][ordering][weighted_rcm]")
{
    auto graph = make_empty_graph(6, "weighted_neighbor_tie");
    sol::add_edge(graph, 0, 1, 100.0, "root_pin");
    sol::add_edge(graph, 1, 2, 1.0, "weak_branch");
    sol::add_edge(graph, 1, 3, 10.0, "strong_branch");
    sol::add_edge(graph, 2, 4, 1.0, "weak_tail");
    sol::add_edge(graph, 3, 5, 1.0, "strong_tail");

    const auto rcm = sol::make_ordering_candidate(graph, "rcm", 32);
    const auto weighted = sol::make_ordering_candidate(graph, "weighted_rcm", 32);

    REQUIRE(rcm.ok);
    REQUIRE(weighted.ok);
    CHECK(chain_distance(weighted.ordering, 1, 3)
          < chain_distance(weighted.ordering, 1, 2));
    CHECK(chain_distance(rcm.ordering, 1, 2)
          < chain_distance(rcm.ordering, 1, 3));
}

TEST_CASE("weighted RCM sees accumulated duplicate edge weights",
          "[socu][ordering][weighted_rcm]")
{
    auto graph = make_empty_graph(4, "weighted_accumulated_edges");
    sol::add_edge(graph, 0, 1, 1.0, "weak_component");
    sol::add_edge(graph, 2, 3, 1.0, "strong_component");
    sol::add_edge(graph, 2, 3, 9.0, "contact_hessian");

    const auto weighted = sol::make_ordering_candidate(graph, "weighted_rcm", 32);

    REQUIRE(weighted.ok);
    CHECK(weighted.ordering.chain_to_old == std::vector<std::size_t>{1, 0, 3, 2});
    REQUIRE(graph.edges.size() == 2);
    CHECK(graph.edges[1].weight == Catch::Approx(10.0));
}

TEST_CASE("weighted RCM keeps strong same-degree branch closer in larger block layout",
          "[socu][ordering][weighted_rcm]")
{
    auto graph = make_empty_graph(45, "weighted_large_branch");
    for(std::size_t i = 0; i + 1 < 15; ++i)
        sol::add_edge(graph, i, i + 1, 1.0, "base_path");
    for(std::size_t i = 15; i + 1 < 30; ++i)
        sol::add_edge(graph, i, i + 1, 1.0, "base_path");
    for(std::size_t i = 30; i + 1 < 45; ++i)
        sol::add_edge(graph, i, i + 1, 1.0, "base_path");

    sol::add_edge(graph, 7, 22, 50.0, "strong_contact");
    sol::add_edge(graph, 7, 37, 2.0, "weak_contact");

    const auto rcm = sol::make_ordering_candidate(graph, "rcm", 32);
    const auto weighted = sol::make_ordering_candidate(graph, "weighted_rcm", 32);

    REQUIRE(rcm.ok);
    REQUIRE(weighted.ok);
    CHECK(weighted.metrics.valid_permutation);
    CHECK(chain_distance(weighted.ordering, 7, 22)
          <= chain_distance(weighted.ordering, 7, 37));
    CHECK(weighted.metrics.weighted_off_band_ratio
          <= rcm.metrics.weighted_off_band_ratio);
}

TEST_CASE("weighted RCM produces valid permutations for varied synthetic graphs",
          "[socu][ordering][weighted_rcm]")
{
    std::vector<sol::AtomGraph> graphs;

    {
        auto graph = make_empty_graph(1, "singleton");
        graphs.push_back(std::move(graph));
    }
    {
        auto graph = make_empty_graph(8, "disconnected_pairs");
        for(std::size_t i = 0; i + 1 < graph.atoms.size(); i += 2)
            sol::add_edge(graph, i, i + 1, static_cast<double>(i + 1), "pair");
        graphs.push_back(std::move(graph));
    }
    {
        auto graph = make_empty_graph(12, "star");
        for(std::size_t i = 1; i < graph.atoms.size(); ++i)
            sol::add_edge(graph, 0, i, static_cast<double>(i), "spoke");
        graphs.push_back(std::move(graph));
    }
    {
        auto graph = make_empty_graph(18, "two_ladders");
        for(std::size_t i = 0; i + 1 < 9; ++i)
        {
            sol::add_edge(graph, i, i + 1, 1.0, "rail");
            sol::add_edge(graph, i + 9, i + 10, 1.0, "rail");
            sol::add_edge(graph, i, i + 9, static_cast<double>(i + 1), "rung");
        }
        graphs.push_back(std::move(graph));
    }
    {
        auto graph = make_empty_graph(10, "clique_tail");
        for(std::size_t i = 0; i < 5; ++i)
            for(std::size_t j = i + 1; j < 5; ++j)
                sol::add_edge(graph, i, j, static_cast<double>(i + j + 1), "clique");
        for(std::size_t i = 5; i + 1 < 10; ++i)
            sol::add_edge(graph, i, i + 1, 1.0, "tail");
        sol::add_edge(graph, 4, 5, 25.0, "bridge");
        graphs.push_back(std::move(graph));
    }

    for(const auto& graph : graphs)
    {
        CAPTURE(graph.name);
        for(const std::size_t block_size : {32, 64})
        {
            const auto candidate =
                sol::make_ordering_candidate(graph, "weighted_rcm", block_size);
            REQUIRE(candidate.ok);
            CHECK(candidate.metrics.valid_permutation);
            CHECK_NOTHROW(sol::validate_permutation(candidate.ordering,
                                                    graph.atoms.size()));
            CHECK(candidate.metrics.weighted_near_band_ratio >= 0.0);
            CHECK(candidate.metrics.weighted_near_band_ratio <= 1.0);
            CHECK(candidate.metrics.weighted_off_band_ratio >= 0.0);
            CHECK(candidate.metrics.weighted_off_band_ratio <= 1.0);
        }
    }
}

TEST_CASE("rod keeps bandwidth-zero baselines and reports METIS ND separately", "[socu][ordering]")
{
    const auto graph = sol::make_rod(96);
    const auto run = sol::run_ordering(graph, "auto", "32");

    for(const auto* orderer : {"original", "rcm", "metis_kway_rcm"})
    {
        const auto& candidate = require_candidate(run, orderer, 32);
        REQUIRE(candidate.metrics.off_band_edge_count == 0);
        REQUIRE(candidate.metrics.max_block_distance <= 1);
        REQUIRE(candidate.metrics.valid_permutation);
    }

    const auto& nvidia = require_candidate(run, "nvidia_symrcm", 32);
    REQUIRE(nvidia.metrics.valid_permutation);
    REQUIRE(nvidia.metrics.off_band_edge_count == 0);
    REQUIRE(nvidia.metrics.max_block_distance <= 1);

    const auto& metis_nd = require_candidate(run, "metis_nd", 32);
    REQUIRE(metis_nd.metrics.valid_permutation);
    REQUIRE(metis_nd.fallback_reason.empty());

    REQUIRE(selected(run).metrics.off_band_edge_count == 0);
    REQUIRE(selected(run).metrics.max_block_distance <= 1);
}

TEST_CASE("auto ordering improves or matches shuffled rod original", "[socu][ordering]")
{
    const auto graph = make_shuffled_rod();
    const auto original = sol::make_ordering_candidate(graph, "original", 32);
    const auto run = sol::run_ordering(graph, "auto", "32");

    REQUIRE(original.ok);
    REQUIRE(selected(run).ok);
    REQUIRE(selected(run).metrics.weighted_off_band_ratio
            <= original.metrics.weighted_off_band_ratio);
    REQUIRE(selected(run).metrics.max_block_distance
            <= original.metrics.max_block_distance);
}

TEST_CASE("auto ordering improves or matches shuffled cloth and tet quality", "[socu][ordering]")
{
    for(const auto* preset : {"shuffled_cloth_grid", "shuffled_tet_block"})
    {
        const auto graph = sol::make_preset(preset);
        const auto original = sol::make_ordering_candidate(graph, "original", 32);
        const auto run = sol::run_ordering(graph, "auto", "32");

        REQUIRE(original.ok);
        REQUIRE(selected(run).ok);
        REQUIRE(selected(run).metrics.weighted_off_band_ratio
                <= original.metrics.weighted_off_band_ratio);
    }
}

TEST_CASE("auto block size reports both 32 and 64 candidates", "[socu][ordering]")
{
    const auto graph = sol::make_tet_block();
    const auto run = sol::run_ordering(graph, "auto", "auto");

    REQUIRE(run.candidates.size() == 10);
    REQUIRE_NOTHROW(require_candidate(run, "original", 32));
    REQUIRE_NOTHROW(require_candidate(run, "nvidia_symrcm", 32));
    REQUIRE_NOTHROW(require_candidate(run, "metis_nd", 32));
    REQUIRE_NOTHROW(require_candidate(run, "original", 64));
    REQUIRE_NOTHROW(require_candidate(run, "nvidia_symrcm", 64));
    REQUIRE_NOTHROW(require_candidate(run, "metis_kway_rcm", 64));
}

TEST_CASE("auto stable keeps runtime candidate set small", "[socu][ordering]")
{
    const auto graph = sol::make_tet_block();
    const auto run = sol::run_ordering(graph, "auto_stable", "auto");

    REQUIRE(run.candidates.size() == 6);
    REQUIRE_NOTHROW(require_candidate(run, "original", 32));
    REQUIRE_NOTHROW(require_candidate(run, "rcm", 32));
    REQUIRE_NOTHROW(require_candidate(run, "metis_kway_rcm", 32));
    REQUIRE_NOTHROW(require_candidate(run, "original", 64));
    REQUIRE_NOTHROW(require_candidate(run, "rcm", 64));
    REQUIRE_NOTHROW(require_candidate(run, "metis_kway_rcm", 64));

    const auto has_orderer = [&](std::string_view orderer)
    {
        return std::any_of(run.candidates.begin(),
                           run.candidates.end(),
                           [&](const sol::OrderingCandidate& candidate)
                           { return candidate.orderer == orderer; });
    };
    REQUIRE_FALSE(has_orderer("nvidia_symrcm"));
    REQUIRE_FALSE(has_orderer("metis_nd"));

    const auto exhaustive = sol::run_ordering(graph, "auto_exhaustive", "32");
    REQUIRE_NOTHROW(require_candidate(exhaustive, "nvidia_symrcm", 32));
    REQUIRE_NOTHROW(require_candidate(exhaustive, "metis_nd", 32));
}

TEST_CASE("NVIDIA symrcm candidate reports valid permutation when available", "[socu][ordering][nvidia]")
{
    const auto graph = sol::make_cloth_grid();
    const auto candidate = sol::make_ordering_candidate(graph, "nvidia_symrcm", 32);

#if SOL_HAS_CUSOLVER_RCM
    REQUIRE(candidate.ok);
    REQUIRE(candidate.fallback_reason.empty());
    REQUIRE_NOTHROW(sol::validate_permutation(candidate.ordering, graph.atoms.size()));
    REQUIRE(candidate.metrics.valid_permutation);
    REQUIRE(candidate.metrics.weighted_off_band_ratio <= 1.0);
#else
    REQUIRE_FALSE(candidate.ok);
    REQUIRE(candidate.fallback_reason.find("unavailable") != std::string::npos);
#endif
}

TEST_CASE("METIS NodeND contract produces inverse perm/iperm semantics", "[socu][ordering][metis]")
{
    const auto graph = sol::make_rod(32);
    const auto candidate = sol::make_ordering_candidate(graph, "metis_nd", 32);

    REQUIRE(candidate.ok);
    REQUIRE(candidate.fallback_reason.empty());
    REQUIRE_NOTHROW(sol::validate_permutation(candidate.ordering, graph.atoms.size()));
    for(std::size_t chain = 0; chain < candidate.ordering.chain_to_old.size(); ++chain)
    {
        const std::size_t old = candidate.ordering.chain_to_old[chain];
        REQUIRE(candidate.ordering.old_to_chain[old] == chain);
    }
}
