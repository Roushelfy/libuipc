#pragma once

#include <sol/types.h>

#include <nlohmann/json_fwd.hpp>
#include <uipc/geometry/simplicial_complex.h>

#include <string>
#include <string_view>

namespace sol
{
struct GeometryCounts
{
    std::size_t vertices{};
    std::size_t edges{};
    std::size_t triangles{};
    std::size_t tetrahedra{};
};

struct GeometryInvariantReport
{
    double max_edge_length_error{};
    double max_triangle_area_error{};
    double max_tet_volume_error{};
};

struct BlockClassification
{
    std::size_t near_band_edges{};
    std::size_t off_band_edges{};
};

struct ReorderResult
{
    std::string mode;
    std::string preset;
    uipc::geometry::SimplicialComplex geometry;
    OrderingResult ordering;
    GeometryCounts before_counts;
    GeometryCounts after_counts;
    GeometryInvariantReport invariants;
    BlockClassification mirror_classification;
    BlockClassification physical_classification;
    std::size_t chain_index_stride_before{};
    std::size_t chain_index_stride_after{};
    bool topology_indices_valid{};
    bool counts_preserved{};
    bool original_vertex_id_complete{};
    bool chain_metadata_valid{};
};

uipc::geometry::SimplicialComplex make_geometry_preset(std::string_view name);
GeometryCounts geometry_counts(const uipc::geometry::SimplicialComplex& geometry);
bool topology_indices_valid(const uipc::geometry::SimplicialComplex& geometry);

ReorderResult reorder_geometry(const uipc::geometry::SimplicialComplex& geometry,
                               const OrderingResult& ordering,
                               std::string_view mode,
                               std::string_view preset_name = "custom");

nlohmann::json to_json(const GeometryCounts& counts);
nlohmann::json to_json(const GeometryInvariantReport& invariants);
nlohmann::json to_json(const BlockClassification& classification);
nlohmann::json to_json(const ReorderResult& result);
} // namespace sol
