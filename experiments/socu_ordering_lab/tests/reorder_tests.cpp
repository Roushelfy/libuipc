#include <sol/ordering.h>
#include <sol/presets.h>
#include <sol/reorder.h>

#include <catch2/catch_all.hpp>

#include <uipc/common/type_define.h>

#include <array>
#include <string_view>

namespace
{
constexpr std::string_view original_vertex_id_name = "socu/original_vertex_id";
constexpr std::string_view chain_id_name           = "socu/chain_id";

sol::OrderingResult ordering_for(std::string_view preset,
                                 std::string_view orderer = "rcm",
                                 std::size_t block_size = 32)
{
    const auto graph = sol::make_preset(preset);
    const auto candidate = sol::make_ordering_candidate(graph, orderer, block_size);
    REQUIRE(candidate.ok);
    REQUIRE_NOTHROW(sol::validate_permutation(candidate.ordering, graph.atoms.size()));
    return candidate.ordering;
}

void require_preserved_geometry(const sol::ReorderResult& result)
{
    REQUIRE(result.counts_preserved);
    REQUIRE(result.topology_indices_valid);
    REQUIRE(result.original_vertex_id_complete);
    REQUIRE(result.chain_metadata_valid);
    REQUIRE(result.invariants.max_edge_length_error <= 1e-12);
    REQUIRE(result.invariants.max_triangle_area_error <= 1e-12);
    REQUIRE(result.invariants.max_tet_volume_error <= 1e-12);
}
} // namespace

TEST_CASE("physical reorder passes chain_to_old as AttributeCollection New2Old mapping",
          "[socu][reorder]")
{
    const auto preset = std::string_view{"shuffled_cloth_grid"};
    const auto geometry = sol::make_geometry_preset(preset);
    const auto ordering = ordering_for(preset, "rcm", 32);

    auto result = sol::reorder_geometry(geometry, ordering, "physical", preset);
    require_preserved_geometry(result);

    const auto original = result.geometry.vertices().find<uipc::IndexT>(original_vertex_id_name);
    const auto chain    = result.geometry.vertices().find<uipc::IndexT>(chain_id_name);
    REQUIRE(original);
    REQUIRE(chain);

    const auto original_view = original->view();
    const auto chain_view    = chain->view();
    REQUIRE(original_view.size() == ordering.chain_to_old.size());
    REQUIRE(chain_view.size() == ordering.chain_to_old.size());

    for(std::size_t new_index = 0; new_index < ordering.chain_to_old.size(); ++new_index)
    {
        REQUIRE(original_view[new_index] == static_cast<uipc::IndexT>(ordering.chain_to_old[new_index]));
        REQUIRE(chain_view[new_index] == static_cast<uipc::IndexT>(new_index));
    }
}

TEST_CASE("physical reorder preserves counts topology and geometric invariants",
          "[socu][reorder]")
{
    for(const auto preset : std::array{std::string_view{"rod"},
                                       std::string_view{"cloth_grid"},
                                       std::string_view{"tet_block"}})
    {
        const auto geometry = sol::make_geometry_preset(preset);
        const auto ordering = ordering_for(preset, "rcm", 64);
        const auto result = sol::reorder_geometry(geometry, ordering, "physical", preset);

        CAPTURE(preset);
        require_preserved_geometry(result);
        REQUIRE(result.before_counts.vertices == result.after_counts.vertices);
        REQUIRE(result.before_counts.edges == result.after_counts.edges);
        REQUIRE(result.before_counts.triangles == result.after_counts.triangles);
        REQUIRE(result.before_counts.tetrahedra == result.after_counts.tetrahedra);
    }
}

TEST_CASE("mirror and physical reorder classify the same contact blocks",
          "[socu][reorder]")
{
    const auto preset = std::string_view{"shuffled_tet_block"};
    const auto geometry = sol::make_geometry_preset(preset);
    const auto ordering = ordering_for(preset, "rcm", 32);

    const auto mirror = sol::reorder_geometry(geometry, ordering, "mirror", preset);
    const auto physical = sol::reorder_geometry(geometry, ordering, "physical", preset);

    REQUIRE(mirror.mirror_classification.near_band_edges
            == physical.physical_classification.near_band_edges);
    REQUIRE(mirror.mirror_classification.off_band_edges
            == physical.physical_classification.off_band_edges);
    require_preserved_geometry(physical);
}

TEST_CASE("physical reorder collapses chain traversal to contiguous vertex memory",
          "[socu][reorder]")
{
    const auto preset = std::string_view{"shuffled_tet_block"};
    const auto geometry = sol::make_geometry_preset(preset);
    const auto ordering = ordering_for(preset, "rcm", 32);

    const auto mirror = sol::reorder_geometry(geometry, ordering, "mirror", preset);
    const auto physical = sol::reorder_geometry(geometry, ordering, "physical", preset);

    REQUIRE(mirror.chain_index_stride_after == mirror.chain_index_stride_before);
    REQUIRE(physical.chain_index_stride_after == ordering.chain_to_old.size() - 1);
    REQUIRE(physical.chain_index_stride_after < physical.chain_index_stride_before);
}

TEST_CASE("reorder rejects an ordering with the wrong vertex count", "[socu][reorder]")
{
    const auto geometry = sol::make_geometry_preset("rod");
    const auto ordering = ordering_for("cloth_grid", "rcm", 32);
    REQUIRE_THROWS_AS(sol::reorder_geometry(geometry, ordering, "physical", "rod"),
                      std::invalid_argument);
}
