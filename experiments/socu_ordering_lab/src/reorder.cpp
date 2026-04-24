#include <sol/reorder.h>

#include <sol/ordering.h>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/attribute_collection.h>
#include <uipc/geometry/attribute_friend.h>
#include <uipc/geometry/attribute_slot.h>
#include <uipc/geometry/utils/factory.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace sol
{
struct ReorderAccessTag
{
};
} // namespace sol

namespace uipc::geometry
{
template <>
class AttributeFriend<sol::ReorderAccessTag>
{
  public:
    static AttributeCollection& collection(SimplicialComplex::VertexAttributes attributes)
    {
        return attributes.m_attributes;
    }
};
} // namespace uipc::geometry

namespace sol
{
namespace
{
using uipc::Float;
using uipc::IndexT;
using uipc::S;
using uipc::SizeT;
using uipc::Vector2i;
using uipc::Vector3;
using uipc::Vector3i;
using uipc::Vector4i;
using uipc::geometry::AttributeCollection;
using uipc::geometry::AttributeSlot;
using uipc::geometry::SimplicialComplex;
using uipc::geometry::view;

constexpr std::string_view original_vertex_id_name = "socu/original_vertex_id";
constexpr std::string_view chain_id_name           = "socu/chain_id";
constexpr std::string_view block_id_name           = "socu/block_id";
constexpr std::string_view block_local_id_name     = "socu/block_local_id";

std::size_t abs_diff(std::size_t a, std::size_t b)
{
    return a > b ? a - b : b - a;
}

std::vector<std::size_t> deterministic_shuffle(std::size_t n, std::uint32_t seed)
{
    std::vector<std::size_t> values(n);
    std::iota(values.begin(), values.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(values.begin(), values.end(), rng);
    return values;
}

template <typename Topo>
void remap_topology(uipc::geometry::AttributeSlot<Topo>& topo_slot,
                    const std::vector<std::size_t>& old_to_chain)
{
    auto topo = view(topo_slot);
    for(auto& simplex : topo)
    {
        for(IndexT i = 0; i < simplex.size(); ++i)
        {
            const auto old = simplex[i];
            if(old < 0 || static_cast<std::size_t>(old) >= old_to_chain.size())
                throw std::runtime_error("topology contains an out-of-range vertex index");
            simplex[i] = static_cast<IndexT>(old_to_chain[static_cast<std::size_t>(old)]);
        }
    }
}

template <typename Topo>
void remap_topology_if_present(S<const AttributeSlot<Topo>> topo_slot,
                               const std::vector<std::size_t>& old_to_chain) = delete;

template <typename Topo>
void remap_topology_if_present(S<AttributeSlot<Topo>> topo_slot,
                               const std::vector<std::size_t>& old_to_chain)
{
    if(topo_slot)
        remap_topology(*topo_slot, old_to_chain);
}

void validate_ordering_for_geometry(const OrderingResult& ordering, const SimplicialComplex& geometry)
{
    validate_permutation(ordering, geometry.vertices().size());
    if(ordering.chain_to_old.size() != geometry.vertices().size())
        throw std::invalid_argument("ordering atom count must match geometry vertex count");
}

std::vector<SizeT> chain_to_old_sizet(const OrderingResult& ordering)
{
    std::vector<SizeT> out(ordering.chain_to_old.begin(), ordering.chain_to_old.end());
    return out;
}

void create_reorder_metadata(SimplicialComplex& geometry, const OrderingResult& ordering)
{
    auto vertices = geometry.vertices();
    auto original = vertices.create<IndexT>(original_vertex_id_name, -1);
    auto chain    = vertices.create<IndexT>(chain_id_name, -1);
    auto block    = vertices.create<IndexT>(block_id_name, -1);
    auto local    = vertices.create<IndexT>(block_local_id_name, -1);

    auto original_view = view(*original);
    auto chain_view    = view(*chain);
    auto block_view    = view(*block);
    auto local_view    = view(*local);

    for(std::size_t old = 0; old < ordering.old_to_chain.size(); ++old)
    {
        original_view[old] = static_cast<IndexT>(old);
        chain_view[old]    = static_cast<IndexT>(ordering.old_to_chain[old]);
        block_view[old]    = static_cast<IndexT>(ordering.atom_to_block[old]);
        local_view[old]    = static_cast<IndexT>(ordering.atom_block_offset[old]);
    }
}

bool original_vertex_id_complete(const SimplicialComplex& geometry)
{
    auto slot = geometry.vertices().find<IndexT>(original_vertex_id_name);
    if(!slot)
        return false;

    std::vector<char> seen(geometry.vertices().size(), 0);
    for(const IndexT value : slot->view())
    {
        if(value < 0 || static_cast<std::size_t>(value) >= seen.size())
            return false;
        if(seen[static_cast<std::size_t>(value)])
            return false;
        seen[static_cast<std::size_t>(value)] = 1;
    }
    return std::all_of(seen.begin(), seen.end(), [](char value) { return value != 0; });
}

bool chain_metadata_valid(const SimplicialComplex& geometry, const OrderingResult& ordering)
{
    auto original = geometry.vertices().find<IndexT>(original_vertex_id_name);
    auto chain    = geometry.vertices().find<IndexT>(chain_id_name);
    auto block    = geometry.vertices().find<IndexT>(block_id_name);
    auto local    = geometry.vertices().find<IndexT>(block_local_id_name);
    if(!original || !chain || !block || !local)
        return false;

    const auto original_view = original->view();
    const auto chain_view    = chain->view();
    const auto block_view    = block->view();
    const auto local_view    = local->view();

    for(std::size_t new_index = 0; new_index < original_view.size(); ++new_index)
    {
        const auto old = original_view[new_index];
        if(old < 0 || static_cast<std::size_t>(old) >= ordering.old_to_chain.size())
            return false;
        if(chain_view[new_index] != static_cast<IndexT>(new_index))
            return false;
        if(block_view[new_index] != static_cast<IndexT>(ordering.atom_to_block[static_cast<std::size_t>(old)]))
            return false;
        if(local_view[new_index]
           != static_cast<IndexT>(ordering.atom_block_offset[static_cast<std::size_t>(old)]))
            return false;
    }
    return true;
}

double triangle_area(const Vector3& a, const Vector3& b, const Vector3& c)
{
    return 0.5 * ((b - a).cross(c - a)).norm();
}

double tet_volume(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d)
{
    return std::abs((b - a).dot((c - a).cross(d - a))) / 6.0;
}

GeometryInvariantReport compare_invariants(const SimplicialComplex& before,
                                           const SimplicialComplex& after)
{
    GeometryInvariantReport report;
    const auto before_pos = before.positions().view();
    const auto after_pos  = after.positions().view();

    const auto before_edges_slot = before.edges().find<Vector2i>(uipc::builtin::topo);
    const auto after_edges_slot  = after.edges().find<Vector2i>(uipc::builtin::topo);
    if(before_edges_slot && after_edges_slot)
    {
        const auto before_edges = before_edges_slot->view();
        const auto after_edges  = after_edges_slot->view();
        for(std::size_t i = 0; i < before_edges.size(); ++i)
        {
            const auto& e0 = before_edges[i];
            const auto& e1 = after_edges[i];
            const double before_length = (before_pos[e0[0]] - before_pos[e0[1]]).norm();
            const double after_length  = (after_pos[e1[0]] - after_pos[e1[1]]).norm();
            report.max_edge_length_error =
                std::max(report.max_edge_length_error, std::abs(before_length - after_length));
        }
    }

    const auto before_tris_slot = before.triangles().find<Vector3i>(uipc::builtin::topo);
    const auto after_tris_slot  = after.triangles().find<Vector3i>(uipc::builtin::topo);
    if(before_tris_slot && after_tris_slot)
    {
        const auto before_tris = before_tris_slot->view();
        const auto after_tris  = after_tris_slot->view();
        for(std::size_t i = 0; i < before_tris.size(); ++i)
        {
            const auto& t0 = before_tris[i];
            const auto& t1 = after_tris[i];
            const double before_area = triangle_area(before_pos[t0[0]], before_pos[t0[1]], before_pos[t0[2]]);
            const double after_area  = triangle_area(after_pos[t1[0]], after_pos[t1[1]], after_pos[t1[2]]);
            report.max_triangle_area_error =
                std::max(report.max_triangle_area_error, std::abs(before_area - after_area));
        }
    }

    const auto before_tets_slot = before.tetrahedra().find<Vector4i>(uipc::builtin::topo);
    const auto after_tets_slot  = after.tetrahedra().find<Vector4i>(uipc::builtin::topo);
    if(before_tets_slot && after_tets_slot)
    {
        const auto before_tets = before_tets_slot->view();
        const auto after_tets  = after_tets_slot->view();
        for(std::size_t i = 0; i < before_tets.size(); ++i)
        {
            const auto& t0 = before_tets[i];
            const auto& t1 = after_tets[i];
            const double before_volume =
                tet_volume(before_pos[t0[0]], before_pos[t0[1]], before_pos[t0[2]], before_pos[t0[3]]);
            const double after_volume =
                tet_volume(after_pos[t1[0]], after_pos[t1[1]], after_pos[t1[2]], after_pos[t1[3]]);
            report.max_tet_volume_error =
                std::max(report.max_tet_volume_error, std::abs(before_volume - after_volume));
        }
    }
    return report;
}

BlockClassification classify_edges(const SimplicialComplex& geometry,
                                   const OrderingResult& ordering,
                                   bool geometry_is_physical)
{
    BlockClassification out;
    const auto edge_slot = geometry.edges().find<Vector2i>(uipc::builtin::topo);
    if(!edge_slot)
        return out;

    for(const auto& edge : edge_slot->view())
    {
        const auto a = static_cast<std::size_t>(edge[0]);
        const auto b = static_cast<std::size_t>(edge[1]);
        const std::size_t old_a = geometry_is_physical ? ordering.chain_to_old[a] : a;
        const std::size_t old_b = geometry_is_physical ? ordering.chain_to_old[b] : b;
        const auto dist = abs_diff(ordering.atom_to_block[old_a], ordering.atom_to_block[old_b]);
        if(dist <= 1)
            ++out.near_band_edges;
        else
            ++out.off_band_edges;
    }
    return out;
}

std::size_t chain_index_stride_before(const OrderingResult& ordering)
{
    std::size_t stride = 0;
    for(std::size_t i = 1; i < ordering.chain_to_old.size(); ++i)
        stride += abs_diff(ordering.chain_to_old[i - 1], ordering.chain_to_old[i]);
    return stride;
}

std::size_t chain_index_stride_after(const OrderingResult& ordering, std::string_view mode)
{
    if(mode == "physical")
        return ordering.chain_to_old.empty() ? 0 : ordering.chain_to_old.size() - 1;
    return chain_index_stride_before(ordering);
}

std::vector<std::size_t> inverse_permutation(const std::vector<std::size_t>& new_to_old)
{
    std::vector<std::size_t> old_to_new(new_to_old.size());
    for(std::size_t new_id = 0; new_id < new_to_old.size(); ++new_id)
        old_to_new[new_to_old[new_id]] = new_id;
    return old_to_new;
}

template <typename Topo>
void remap_topology_with_old_to_new(uipc::geometry::AttributeSlot<Topo>& topo_slot,
                                    const std::vector<std::size_t>& old_to_new)
{
    auto topo = view(topo_slot);
    for(auto& simplex : topo)
        for(IndexT i = 0; i < simplex.size(); ++i)
            simplex[i] = static_cast<IndexT>(old_to_new[static_cast<std::size_t>(simplex[i])]);
}

template <typename Topo>
void remap_topology_with_old_to_new_if_present(S<AttributeSlot<Topo>> topo_slot,
                                               const std::vector<std::size_t>& old_to_new)
{
    if(topo_slot)
        remap_topology_with_old_to_new(*topo_slot, old_to_new);
}

SimplicialComplex relabel_geometry(SimplicialComplex geometry, const std::vector<std::size_t>& new_to_old)
{
    const auto old_to_new = inverse_permutation(new_to_old);
    auto order = std::vector<SizeT>(new_to_old.begin(), new_to_old.end());
    uipc::geometry::AttributeFriend<sol::ReorderAccessTag>::collection(geometry.vertices())
        .reorder(uipc::span<const SizeT>{order.data(), order.size()});

    remap_topology_with_old_to_new_if_present(geometry.edges().find<Vector2i>(uipc::builtin::topo), old_to_new);
    remap_topology_with_old_to_new_if_present(geometry.triangles().find<Vector3i>(uipc::builtin::topo), old_to_new);
    remap_topology_with_old_to_new_if_present(geometry.tetrahedra().find<Vector4i>(uipc::builtin::topo), old_to_new);
    return geometry;
}

SimplicialComplex make_rod_geometry()
{
    std::vector<Vector3> positions;
    std::vector<Vector2i> edges;
    positions.reserve(96);
    for(IndexT i = 0; i < 96; ++i)
        positions.push_back(Vector3{static_cast<Float>(i), 0.0, 0.0});
    for(IndexT i = 1; i < 96; ++i)
        edges.push_back(Vector2i{i - 1, i});
    return uipc::geometry::linemesh(positions, edges);
}

SimplicialComplex make_cloth_geometry()
{
    constexpr IndexT width  = 16;
    constexpr IndexT height = 12;
    std::vector<Vector3> positions;
    std::vector<Vector3i> triangles;
    positions.reserve(static_cast<std::size_t>(width * height));
    auto id = [](IndexT x, IndexT y) { return y * width + x; };
    for(IndexT y = 0; y < height; ++y)
        for(IndexT x = 0; x < width; ++x)
            positions.push_back(Vector3{static_cast<Float>(x), static_cast<Float>(y), 0.0});
    for(IndexT y = 0; y + 1 < height; ++y)
    {
        for(IndexT x = 0; x + 1 < width; ++x)
        {
            triangles.push_back(Vector3i{id(x, y), id(x + 1, y), id(x + 1, y + 1)});
            triangles.push_back(Vector3i{id(x, y), id(x + 1, y + 1), id(x, y + 1)});
        }
    }
    return uipc::geometry::trimesh(positions, triangles);
}

SimplicialComplex make_tet_geometry()
{
    constexpr IndexT nx = 5;
    constexpr IndexT ny = 4;
    constexpr IndexT nz = 3;
    constexpr IndexT sx = nx + 1;
    constexpr IndexT sy = ny + 1;
    auto id = [](IndexT x, IndexT y, IndexT z) { return z * sx * sy + y * sx + x; };

    std::vector<Vector3> positions;
    std::vector<Vector4i> tets;
    positions.reserve(static_cast<std::size_t>(sx * sy * (nz + 1)));
    for(IndexT z = 0; z <= nz; ++z)
        for(IndexT y = 0; y <= ny; ++y)
            for(IndexT x = 0; x <= nx; ++x)
                positions.push_back(Vector3{static_cast<Float>(x), static_cast<Float>(y), static_cast<Float>(z)});

    for(IndexT z = 0; z < nz; ++z)
    {
        for(IndexT y = 0; y < ny; ++y)
        {
            for(IndexT x = 0; x < nx; ++x)
            {
                const IndexT v000 = id(x, y, z);
                const IndexT v100 = id(x + 1, y, z);
                const IndexT v010 = id(x, y + 1, z);
                const IndexT v110 = id(x + 1, y + 1, z);
                const IndexT v001 = id(x, y, z + 1);
                const IndexT v101 = id(x + 1, y, z + 1);
                const IndexT v011 = id(x, y + 1, z + 1);
                const IndexT v111 = id(x + 1, y + 1, z + 1);
                tets.push_back(Vector4i{v000, v100, v010, v001});
                tets.push_back(Vector4i{v100, v110, v010, v111});
                tets.push_back(Vector4i{v100, v010, v001, v111});
                tets.push_back(Vector4i{v100, v001, v101, v111});
                tets.push_back(Vector4i{v010, v001, v011, v111});
            }
        }
    }
    return uipc::geometry::tetmesh(positions, tets);
}
} // namespace

uipc::geometry::SimplicialComplex make_geometry_preset(std::string_view name)
{
    if(name == "rod")
        return make_rod_geometry();
    if(name == "cloth_grid")
        return make_cloth_geometry();
    if(name == "tet_block")
        return make_tet_geometry();
    if(name == "shuffled_cloth_grid")
        return relabel_geometry(make_cloth_geometry(), deterministic_shuffle(16 * 12, 74017));
    if(name == "shuffled_tet_block")
    {
        auto geometry = make_tet_geometry();
        const auto vertex_count = geometry.vertices().size();
        return relabel_geometry(std::move(geometry), deterministic_shuffle(vertex_count, 74019));
    }
    throw std::invalid_argument(
        "unsupported geometry preset; expected rod, cloth_grid, tet_block, shuffled_cloth_grid, or shuffled_tet_block");
}

GeometryCounts geometry_counts(const uipc::geometry::SimplicialComplex& geometry)
{
    return GeometryCounts{geometry.vertices().size(),
                          geometry.edges().size(),
                          geometry.triangles().size(),
                          geometry.tetrahedra().size()};
}

bool topology_indices_valid(const uipc::geometry::SimplicialComplex& geometry)
{
    const auto count = static_cast<IndexT>(geometry.vertices().size());
    auto valid_index = [count](IndexT index) { return index >= 0 && index < count; };

    if(const auto edges = geometry.edges().find<Vector2i>(uipc::builtin::topo))
        for(const auto& edge : edges->view())
            if(!valid_index(edge[0]) || !valid_index(edge[1]))
                return false;
    if(const auto tris = geometry.triangles().find<Vector3i>(uipc::builtin::topo))
        for(const auto& tri : tris->view())
            if(!valid_index(tri[0]) || !valid_index(tri[1]) || !valid_index(tri[2]))
                return false;
    if(const auto tets = geometry.tetrahedra().find<Vector4i>(uipc::builtin::topo))
        for(const auto& tet : tets->view())
            if(!valid_index(tet[0]) || !valid_index(tet[1]) || !valid_index(tet[2]) || !valid_index(tet[3]))
                return false;
    return true;
}

ReorderResult reorder_geometry(const uipc::geometry::SimplicialComplex& geometry,
                               const OrderingResult& ordering,
                               std::string_view mode,
                               std::string_view preset_name)
{
    validate_ordering_for_geometry(ordering, geometry);
    if(mode != "mirror" && mode != "physical")
        throw std::invalid_argument("reorder mode must be mirror or physical");

    ReorderResult result{std::string(mode), std::string(preset_name), geometry, ordering};
    result.before_counts = geometry_counts(geometry);
    result.mirror_classification = classify_edges(geometry, ordering, false);
    result.chain_index_stride_before = chain_index_stride_before(ordering);
    result.chain_index_stride_after  = chain_index_stride_after(ordering, mode);

    if(mode == "physical")
    {
        create_reorder_metadata(result.geometry, ordering);
        const auto order = chain_to_old_sizet(ordering);
        uipc::geometry::AttributeFriend<sol::ReorderAccessTag>::collection(result.geometry.vertices())
            .reorder(uipc::span<const SizeT>{order.data(), order.size()});

        remap_topology_if_present(result.geometry.edges().find<Vector2i>(uipc::builtin::topo), ordering.old_to_chain);
        remap_topology_if_present(result.geometry.triangles().find<Vector3i>(uipc::builtin::topo), ordering.old_to_chain);
        remap_topology_if_present(result.geometry.tetrahedra().find<Vector4i>(uipc::builtin::topo),
                                  ordering.old_to_chain);

        result.physical_classification = classify_edges(result.geometry, ordering, true);
        result.original_vertex_id_complete = original_vertex_id_complete(result.geometry);
        result.chain_metadata_valid = chain_metadata_valid(result.geometry, ordering);
    }
    else
    {
        result.physical_classification = result.mirror_classification;
        result.original_vertex_id_complete = true;
        result.chain_metadata_valid = true;
    }

    result.after_counts = geometry_counts(result.geometry);
    result.topology_indices_valid = topology_indices_valid(result.geometry);
    result.counts_preserved = result.before_counts.vertices == result.after_counts.vertices
                              && result.before_counts.edges == result.after_counts.edges
                              && result.before_counts.triangles == result.after_counts.triangles
                              && result.before_counts.tetrahedra == result.after_counts.tetrahedra;
    result.invariants = compare_invariants(geometry, result.geometry);
    return result;
}

nlohmann::json to_json(const GeometryCounts& counts)
{
    return {{"vertices", counts.vertices},
            {"edges", counts.edges},
            {"triangles", counts.triangles},
            {"tetrahedra", counts.tetrahedra}};
}

nlohmann::json to_json(const GeometryInvariantReport& invariants)
{
    return {{"max_edge_length_error", invariants.max_edge_length_error},
            {"max_triangle_area_error", invariants.max_triangle_area_error},
            {"max_tet_volume_error", invariants.max_tet_volume_error}};
}

nlohmann::json to_json(const BlockClassification& classification)
{
    return {{"near_band_edges", classification.near_band_edges},
            {"off_band_edges", classification.off_band_edges}};
}

nlohmann::json to_json(const ReorderResult& result)
{
    return {{"mode", result.mode},
            {"preset", result.preset},
            {"ordering",
             {{"orderer", result.ordering.orderer},
              {"block_size", result.ordering.block_size},
              {"atom_count", result.ordering.chain_to_old.size()},
              {"block_count", result.ordering.block_to_atom_range.size()}}},
            {"before_counts", to_json(result.before_counts)},
            {"after_counts", to_json(result.after_counts)},
            {"counts_preserved", result.counts_preserved},
            {"topology_indices_valid", result.topology_indices_valid},
            {"original_vertex_id_complete", result.original_vertex_id_complete},
            {"chain_metadata_valid", result.chain_metadata_valid},
            {"chain_index_stride_before", result.chain_index_stride_before},
            {"chain_index_stride_after", result.chain_index_stride_after},
            {"mirror_classification", to_json(result.mirror_classification)},
            {"physical_classification", to_json(result.physical_classification)},
            {"invariants", to_json(result.invariants)}};
}
} // namespace sol
