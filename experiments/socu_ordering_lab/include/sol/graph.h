#pragma once

#include <sol/types.h>

#include <nlohmann/json_fwd.hpp>

#include <filesystem>
#include <string_view>
#include <vector>

namespace sol
{
void add_atom(AtomGraph& graph,
              std::size_t dof_count = 3,
              std::string_view source_kind = "synthetic",
              std::size_t source_id = static_cast<std::size_t>(-1));

void add_edge(AtomGraph& graph,
              std::size_t a,
              std::size_t b,
              double weight = 1.0,
              std::string_view kind = "topology");

void validate_graph(const AtomGraph& graph);
AtomGraph relabel_graph(const AtomGraph& graph, const std::vector<std::size_t>& new_to_old);
std::vector<std::vector<std::size_t>> adjacency(const AtomGraph& graph);

AtomGraph graph_from_json(const nlohmann::json& json);
AtomGraph read_graph_json(const std::filesystem::path& path);
void      read_extra_edges_csv(AtomGraph& graph, const std::filesystem::path& path);
} // namespace sol
