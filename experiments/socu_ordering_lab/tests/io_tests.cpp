#include <sol/io.h>
#include <sol/ordering.h>
#include <sol/presets.h>

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

TEST_CASE("JSON report exposes selected candidate, mappings, and metrics", "[socu][io]")
{
    const auto graph = sol::make_rod(16);
    const auto run = sol::run_ordering(graph, "auto", "32");
    const auto json = sol::to_json(run);

    REQUIRE(json.contains("selected"));
    REQUIRE(json.contains("candidates"));
    REQUIRE(json.at("selected").at("ok").get<bool>());
    REQUIRE(json.at("selected").at("ordering").contains("chain_to_old"));
    REQUIRE(json.at("selected").at("ordering").contains("atom_to_block"));
    REQUIRE(json.at("selected").at("metrics").contains("weighted_near_band_ratio"));
}

TEST_CASE("summary and mapping CSV files have stable headers and row counts", "[socu][io]")
{
    const auto graph = sol::make_rod(12);
    const auto run = sol::run_ordering(graph, "auto", "32");

    const auto dir = std::filesystem::temp_directory_path();
    const auto summary_path = dir / "socu_ordering_summary_test.csv";
    const auto mapping_path = dir / "socu_ordering_mapping_test.csv";

    sol::write_summary_csv(summary_path, run);
    sol::write_mapping_csv(mapping_path, run.candidates.at(run.selected_candidate));

    std::ifstream summary(summary_path);
    REQUIRE(summary.good());
    std::string header;
    std::getline(summary, header);
    REQUIRE(header.starts_with("selected,orderer,block_size,ok,score"));

    std::size_t summary_rows = 0;
    std::string line;
    while(std::getline(summary, line))
        ++summary_rows;
    REQUIRE(summary_rows == run.candidates.size());

    std::ifstream mapping(mapping_path);
    REQUIRE(mapping.good());
    std::getline(mapping, header);
    REQUIRE(header == "old_atom,chain_atom,block,block_offset,dof_count");

    std::size_t mapping_rows = 0;
    while(std::getline(mapping, line))
        ++mapping_rows;
    REQUIRE(mapping_rows == graph.atoms.size());

    std::filesystem::remove(summary_path);
    std::filesystem::remove(mapping_path);
}
