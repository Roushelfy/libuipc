#include <sol/graph.h>
#include <sol/presets.h>

#include <catch2/catch_all.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <numeric>

TEST_CASE("graph presets are valid and canonical", "[socu][graph]")
{
    for(const auto* preset : {"rod", "cloth_grid", "tet_block", "shuffled_cloth_grid", "shuffled_tet_block"})
    {
        const auto graph = sol::make_preset(preset);
        REQUIRE_NOTHROW(sol::validate_graph(graph));
        REQUIRE_FALSE(graph.atoms.empty());
        REQUIRE_FALSE(graph.edges.empty());
        for(const auto& edge : graph.edges)
            REQUIRE(edge.a < edge.b);
    }
}

TEST_CASE("graph relabel requires a complete permutation", "[socu][graph]")
{
    const auto graph = sol::make_rod(8);
    std::vector<std::size_t> valid(graph.atoms.size());
    std::iota(valid.begin(), valid.end(), 0);
    std::reverse(valid.begin(), valid.end());

    const auto relabeled = sol::relabel_graph(graph, valid);
    REQUIRE(relabeled.atoms.size() == graph.atoms.size());
    REQUIRE(relabeled.edges.size() == graph.edges.size());

    auto duplicate = valid;
    duplicate[0] = duplicate[1];
    REQUIRE_THROWS(sol::relabel_graph(graph, duplicate));
}

TEST_CASE("graph JSON and extra edge CSV input are accepted", "[socu][graph]")
{
    nlohmann::json json = {
        {"name", "tiny"},
        {"atoms", {{{"dof_count", 3}}, {{"dof_count", 6}}, {{"dof_count", 3}}}},
        {"edges", {{{"a", 0}, {"b", 1}, {"weight", 2.0}}, {1, 2, 3.0, "hint"}}}};

    auto graph = sol::graph_from_json(json);
    REQUIRE(graph.name == "tiny");
    REQUIRE(graph.atoms.size() == 3);
    REQUIRE(graph.edges.size() == 2);
    REQUIRE(graph.atoms[1].dof_count == 6);

    sol::add_edge(graph, 0, 2, 4.0, "contact_hint");
    REQUIRE(graph.edges.size() == 3);
    REQUIRE_NOTHROW(sol::validate_graph(graph));
}
