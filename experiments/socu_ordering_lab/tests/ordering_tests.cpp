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
