#include <sol/contact.h>
#include <sol/ordering.h>
#include <sol/presets.h>

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <fstream>
#include <string_view>

namespace
{
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
} // namespace

TEST_CASE("near-band contact scenario is fully absorbed", "[socu][contact]")
{
    const auto ordering = ordering_for("cloth_grid", "rcm", 32);
    const auto primitives = sol::make_contact_scenario(ordering, "near_band", 24);
    const auto report = sol::classify_contacts(ordering, primitives, "near_band");

    REQUIRE(report.active_contact_count == 24);
    REQUIRE(report.near_band_contact_count == report.active_contact_count);
    REQUIRE(report.off_band_contact_count == 0);
    REQUIRE(report.near_band_contribution_count == report.estimated_absorbed_hessian_contribution_count);
    REQUIRE(report.off_band_contribution_count == 0);
    REQUIRE(report.weighted_off_band_dropped_norm == 0.0);
    REQUIRE(report.contact_classify_time_ms >= 0.0);
    REQUIRE(report.frame_boundary_reorder_time_ms == 0.0);
    REQUIRE(report.newton_iteration_reorder_count == 0);
    REQUIRE(report.permutation_fixed_within_frame);
}

TEST_CASE("mixed contact scenario reports primitive and contribution levels separately",
          "[socu][contact]")
{
    const auto ordering = ordering_for("shuffled_tet_block", "rcm", 32);
    const auto primitives = sol::make_contact_scenario(ordering, "mixed", 32);
    const auto report = sol::classify_contacts(ordering, primitives, "mixed", 0.125);

    REQUIRE(report.active_contact_count == 32);
    REQUIRE(report.near_band_contact_count > 0);
    REQUIRE(report.off_band_contact_count > 0);
    REQUIRE(report.mixed_contact_count > 0);
    REQUIRE(report.near_band_contribution_count > 0);
    REQUIRE(report.off_band_contribution_count > 0);
    REQUIRE(report.contribution_near_band_ratio > 0.0);
    REQUIRE(report.contribution_off_band_ratio > 0.0);
    REQUIRE(report.frame_boundary_reorder_time_ms == Catch::Approx(0.125));
    REQUIRE(report.newton_iteration_reorder_count == 0);
}

TEST_CASE("adversarial contact scenario drives off-band ratio high", "[socu][contact]")
{
    const auto ordering = ordering_for("shuffled_tet_block", "rcm", 32);
    const auto primitives = sol::make_contact_scenario(ordering, "adversarial", 24);
    const auto report = sol::classify_contacts(ordering, primitives, "adversarial");

    REQUIRE(report.active_contact_count == 24);
    REQUIRE(report.off_band_ratio >= 0.9);
    REQUIRE(report.contribution_off_band_ratio > report.contribution_near_band_ratio);
    REQUIRE(report.estimated_dropped_contribution_count == report.off_band_contribution_count);
}

TEST_CASE("weighted norm exposes a small number of stiff off-band contacts",
          "[socu][contact]")
{
    const auto ordering = ordering_for("rod", "original", 32);
    std::vector<sol::ContactPrimitive> primitives;

    for(std::size_t i = 0; i < 30; ++i)
    {
        primitives.push_back(sol::ContactPrimitive{"PP", {i, i + 1}, 1.0, "near_low_stiffness"});
    }
    primitives.push_back(
        sol::ContactPrimitive{"PP", {0, ordering.chain_to_old.size() - 1}, 1000.0, "off_high_stiffness"});

    const auto report = sol::classify_contacts(ordering, primitives, "weighted_probe");

    REQUIRE(report.off_band_contact_count == 1);
    REQUIRE(report.off_band_ratio < 0.05);
    REQUIRE(report.weighted_off_band_dropped_norm > report.weighted_near_band_contribution_norm);
    REQUIRE(report.weighted_off_band_ratio > 0.9);
}

TEST_CASE("contact CSV input supports recorded primitive files", "[socu][contact]")
{
    const auto path = std::filesystem::temp_directory_path() / "socu_contact_primitives_test.csv";
    {
        std::ofstream out(path);
        out << "kind,stiffness,atom0,atom1,atom2,atom3\n";
        out << "PP,1.0,0,1\n";
        out << "PE,2.0,0,80,81\n";
    }

    const auto primitives = sol::read_contact_primitives_csv(path);
    REQUIRE(primitives.size() == 2);
    REQUIRE(primitives[0].kind == "PP");
    REQUIRE(primitives[1].kind == "PE");

    const auto ordering = ordering_for("rod", "original", 32);
    const auto report = sol::classify_contacts(ordering, primitives, "from_file");
    REQUIRE(report.active_contact_count == 2);
    REQUIRE(report.near_band_contribution_count > 0);
    REQUIRE(report.off_band_contribution_count > 0);

    std::filesystem::remove(path);
}
