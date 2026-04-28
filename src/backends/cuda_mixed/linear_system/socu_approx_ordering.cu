#include <linear_system/socu_approx_ordering.h>
#include <linear_system/socu_rcm_ordering.h>

#include <finite_element/finite_element_method.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_collection.h>
#include <uipc/builtin/geometry_type.h>
#include <uipc/common/exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>

namespace uipc::backend::cuda_mixed::socu_approx
{
namespace fs = std::filesystem;
namespace ordering = rcm;

const Json* selected_candidate_json(const Json& report)
{
    if(report.contains("selected") && report["selected"].is_object())
        return &report["selected"];
    return &report;
}

const Json* ordering_json(const Json& candidate)
{
    if(candidate.contains("ordering") && candidate["ordering"].is_object())
        return &candidate["ordering"];
    if(candidate.contains("block_size"))
        return &candidate;
    return nullptr;
}

const Json* metrics_json(const Json& report, const Json& candidate)
{
    if(candidate.contains("metrics") && candidate["metrics"].is_object())
        return &candidate["metrics"];
    if(report.contains("metrics") && report["metrics"].is_object())
        return &report["metrics"];
    return nullptr;
}

bool required_array(const Json& json, std::string_view key, std::size_t expected_size)
{
    auto it = json.find(key);
    return it != json.end() && it->is_array() && it->size() == expected_size;
}

bool has_basic_ordering_schema(const Json& ordering)
{
    if(!ordering.contains("block_size")
       || !(ordering["block_size"].is_number_unsigned()
            || ordering["block_size"].is_number_integer()))
        return false;
    if(!ordering.contains("chain_to_old") || !ordering["chain_to_old"].is_array())
        return false;

    const auto atom_count = ordering["chain_to_old"].size();
    return required_array(ordering, "old_to_chain", atom_count)
           && required_array(ordering, "atom_to_block", atom_count)
           && required_array(ordering, "atom_block_offset", atom_count)
           && required_array(ordering, "atom_dof_count", atom_count)
           && ordering.contains("block_to_atom_range")
           && ordering["block_to_atom_range"].is_array()
           && !ordering["block_to_atom_range"].empty();
}

fs::path absolute_workspace_path(std::string_view workspace, const std::string& path)
{
    fs::path p{path};
    if(p.is_relative())
        p = fs::absolute(fs::path{workspace} / p);
    return p;
}

fs::path default_generated_ordering_report_path(std::string_view workspace)
{
    return fs::absolute(fs::path{workspace} / "socu_approx" / "ordering.json");
}

ordering::AtomGraph make_abd_body_local_atom_graph(SizeT body_count)
{
    ordering::AtomGraph graph;
    graph.name = "cuda_mixed_abd_init_time";
    for(SizeT body = 0; body < body_count; ++body)
    {
        for(SizeT local_atom = 0; local_atom < 4; ++local_atom)
            ordering::add_atom(graph, 3, "abd_body_local", body * 4 + local_atom);
    }

    for(SizeT body = 0; body < body_count; ++body)
    {
        const SizeT base = body * 4;
        for(SizeT i = 0; i < 4; ++i)
        {
            for(SizeT j = i + 1; j < 4; ++j)
                ordering::add_edge(graph, base + i, base + j, 4.0, "abd_body");
        }
    }

    for(SizeT body = 1; body < body_count; ++body)
        ordering::add_edge(
            graph, (body - 1) * 4, body * 4, 1.0, "abd_body_sequence");

    return graph;
}

Json generate_abd_init_time_ordering_report(SizeT            body_count,
                                            std::string_view orderer,
                                            std::string_view block_size)
{
    auto graph  = make_abd_body_local_atom_graph(body_count);
    auto run    = ordering::run_ordering(graph, orderer, block_size);
    Json report = ordering::to_json(run);
    report["graph"] = ordering::to_json(graph);
    report["generated_by"] = "cuda_mixed_socu_init_time";
    report["ordering_source"] = "init_time";
    return report;
}

struct FemOrderingGeometry
{
    SizeT geo_slot_index = 0;
    SizeT dim            = 0;
    U64   uid            = 0;
    SizeT vertex_offset  = 0;
    SizeT vertex_count   = 0;
};

std::vector<FemOrderingGeometry> collect_fem_ordering_geometries(WorldVisitor& world)
{
    std::vector<FemOrderingGeometry> geos;
    for(auto&& [slot_index, geo_slot] : enumerate(world.scene().geometries()))
    {
        auto& geo = geo_slot->geometry();
        if(geo.type() != builtin::SimplicialComplex)
            continue;

        auto uid_attr = geo.meta().find<U64>(builtin::constitution_uid);
        if(!uid_attr)
            continue;

        const U64 uid = uid_attr->view()[0];
        const auto& uid_info =
            builtin::ConstitutionUIDCollection::instance().find(uid);
        if(uid_info.type != builtin::FiniteElement)
            continue;

        auto* sc = geo.as<geometry::SimplicialComplex>();
        if(!sc)
            continue;

        geos.push_back(FemOrderingGeometry{static_cast<SizeT>(slot_index),
                                           static_cast<SizeT>(sc->dim()),
                                           uid,
                                           0,
                                           sc->vertices().size()});
    }

    std::sort(geos.begin(),
              geos.end(),
              [](const FemOrderingGeometry& a, const FemOrderingGeometry& b)
              {
                  if(a.dim != b.dim)
                      return a.dim < b.dim;
                  if(a.uid != b.uid)
                      return a.uid < b.uid;
                  return a.geo_slot_index < b.geo_slot_index;
              });

    SizeT vertex_offset = 0;
    for(auto& geo : geos)
    {
        geo.vertex_offset = vertex_offset;
        vertex_offset += geo.vertex_count;
    }
    return geos;
}

SizeT fem_vertex_count_from_scene(WorldVisitor& world)
{
    SizeT count = 0;
    for(const auto& geo : collect_fem_ordering_geometries(world))
        count += geo.vertex_count;
    return count;
}

template <typename Vec>
void add_fem_simplex_edges(ordering::AtomGraph& graph,
                           const Vec&          simplex,
                           SizeT          vertex_offset,
                           int            count,
                           double         weight,
                           std::string_view label)
{
    for(int i = 0; i < count; ++i)
    {
        for(int j = i + 1; j < count; ++j)
        {
            ordering::add_edge(graph,
                               vertex_offset + static_cast<SizeT>(simplex(i)),
                               vertex_offset + static_cast<SizeT>(simplex(j)),
                               weight,
                               label);
        }
    }
}

ordering::AtomGraph make_fem_vertex_atom_graph(WorldVisitor& world)
{
    ordering::AtomGraph graph;
    graph.name = "cuda_mixed_fem_init_time";
    auto geos = collect_fem_ordering_geometries(world);
    for(const auto& ordered_geo : geos)
    {
        for(SizeT local_vertex = 0; local_vertex < ordered_geo.vertex_count;
            ++local_vertex)
        {
            ordering::add_atom(graph,
                               3,
                               "fem_vertex",
                               ordered_geo.vertex_offset + local_vertex);
        }
    }

    auto geo_slots = world.scene().geometries();
    for(const auto& ordered_geo : geos)
    {
        auto& geo = geo_slots[ordered_geo.geo_slot_index]->geometry();
        auto* sc  = geo.as<geometry::SimplicialComplex>();
        if(!sc)
            continue;

        switch(sc->dim())
        {
            case 1: {
                for(const auto& edge : sc->edges().topo().view())
                    add_fem_simplex_edges(
                        graph, edge, ordered_geo.vertex_offset, 2, 1.0, "fem_edge");
            }
            break;
            case 2: {
                for(const auto& tri : sc->triangles().topo().view())
                    add_fem_simplex_edges(
                        graph, tri, ordered_geo.vertex_offset, 3, 2.0, "fem_triangle");
            }
            break;
            case 3: {
                for(const auto& tet : sc->tetrahedra().topo().view())
                    add_fem_simplex_edges(
                        graph, tet, ordered_geo.vertex_offset, 4, 4.0, "fem_tet");
            }
            break;
            default:
                break;
        }
    }
    return graph;
}

Json generate_fem_init_time_ordering_report(WorldVisitor&     world,
                                            std::string_view orderer,
                                            std::string_view block_size)
{
    auto graph  = make_fem_vertex_atom_graph(world);
    auto run    = ordering::run_ordering(graph, orderer, block_size);
    Json report = ordering::to_json(run);
    report["graph"] = ordering::to_json(graph);
    report["generated_by"] = "cuda_mixed_socu_init_time";
    report["ordering_source"] = "init_time";
    report["provider_kind"] = "fem_only";
    return report;
}

Json generate_mixed_abd_fem_init_time_ordering_report(WorldVisitor&     world,
                                                      SizeT            body_count,
                                                      std::string_view orderer,
                                                      std::string_view block_size)
{
    auto graph = make_abd_body_local_atom_graph(body_count);
    graph.name = "cuda_mixed_abd_fem_init_time";
    const SizeT fem_atom_base = graph.atoms.size();
    auto fem_graph = make_fem_vertex_atom_graph(world);
    for(SizeT atom = 0; atom < fem_graph.atoms.size(); ++atom)
        ordering::add_atom(graph, 3, "fem_vertex", fem_atom_base + atom);
    for(const auto& edge : fem_graph.edges)
        ordering::add_edge(graph,
                           fem_atom_base + edge.a,
                           fem_atom_base + edge.b,
                           edge.weight,
                           edge.kind);
    if(body_count > 0 && !fem_graph.atoms.empty())
        ordering::add_edge(graph, 0, fem_atom_base, 0.25, "abd_fem_sequence");

    auto run    = ordering::run_ordering(graph, orderer, block_size);
    Json report = ordering::to_json(run);
    report["graph"] = ordering::to_json(graph);
    report["generated_by"] = "cuda_mixed_socu_init_time";
    report["ordering_source"] = "init_time";
    report["provider_kind"] = "multi_provider";
    return report;
}

void write_json_report(const fs::path& path, const Json& report)
{
    fs::create_directories(path.parent_path());
    std::ofstream ofs{path};
    if(!ofs)
        throw Exception{fmt::format("failed to open '{}' for writing", path.string())};
    ofs << report.dump(2);
}

SizeT count_abd_bodies_from_scene(WorldVisitor& world)
{
    SizeT body_count = 0;
    for(auto&& geo_slot : world.scene().geometries())
    {
        auto& geo = geo_slot->geometry();
        if(geo.type() != builtin::SimplicialComplex)
            continue;

        auto uid_attr = geo.meta().find<U64>(builtin::constitution_uid);
        if(!uid_attr)
            continue;

        const auto& uid_info =
            builtin::ConstitutionUIDCollection::instance().find(uid_attr->view()[0]);
        if(uid_info.type == builtin::AffineBody)
            body_count += geo.instances().size();
    }
    return body_count;
}

bool read_size_t_field(const Json& json, const char* key, SizeT& value)
{
    auto it = json.find(key);
    if(it == json.end()
       || !(it->is_number_unsigned() || it->is_number_integer()))
        return false;
    value = it->get<SizeT>();
    return true;
}

bool parse_atom_dof_count(const Json& ordering, SizeT& dof_count, std::string& detail)
{
    dof_count = 0;
    auto it = ordering.find("atom_dof_count");
    if(it == ordering.end() || !it->is_array())
    {
        detail = "ordering atom_dof_count must be an array";
        return false;
    }

    for(const Json& value : *it)
    {
        if(!(value.is_number_unsigned() || value.is_number_integer()))
        {
            detail = "ordering atom_dof_count contains a non-integer value";
            return false;
        }
        dof_count += value.get<SizeT>();
    }
    return true;
}

bool parse_size_t_array(const Json&             ordering,
                        const char*            key,
                        std::vector<SizeT>&    values,
                        std::string&           detail)
{
    values.clear();
    auto it = ordering.find(key);
    if(it == ordering.end() || !it->is_array())
    {
        detail = fmt::format("ordering {} must be an array", key);
        return false;
    }

    values.reserve(it->size());
    for(const Json& value : *it)
    {
        if(!(value.is_number_unsigned() || value.is_number_integer()))
        {
            detail = fmt::format("ordering {} contains a non-integer value", key);
            return false;
        }
        values.push_back(value.get<SizeT>());
    }
    return true;
}

bool parse_block_layouts(const Json&                         ordering,
                         std::vector<SocuApproxBlockLayout>& blocks,
                         std::string&                       detail)
{
    blocks.clear();
    const auto& block_ranges = ordering.at("block_to_atom_range");
    SizeT       dof_offset   = 0;

    for(const Json& entry : block_ranges)
    {
        if(!entry.is_object())
        {
            detail = "ordering block_to_atom_range entries must be objects";
            return false;
        }

        SocuApproxBlockLayout block;
        if(!read_size_t_field(entry, "block", block.block)
           || !read_size_t_field(entry, "chain_begin", block.chain_begin)
           || !read_size_t_field(entry, "chain_end", block.chain_end)
           || !read_size_t_field(entry, "dof_count", block.dof_count))
        {
            detail =
                "ordering block_to_atom_range entries must contain block, chain_begin, chain_end, and dof_count";
            return false;
        }

        if(block.chain_end < block.chain_begin)
        {
            detail = "ordering block_to_atom_range has chain_end < chain_begin";
            return false;
        }

        block.dof_offset = dof_offset;
        dof_offset += block.dof_count;
        blocks.push_back(block);
    }

    return !blocks.empty();
}

bool validate_atom_inverse_mapping(const Json& ordering, std::string& detail)
{
    std::vector<SizeT> chain_to_old;
    std::vector<SizeT> old_to_chain;
    if(!parse_size_t_array(ordering, "chain_to_old", chain_to_old, detail)
       || !parse_size_t_array(ordering, "old_to_chain", old_to_chain, detail))
    {
        return false;
    }

    if(chain_to_old.size() != old_to_chain.size())
    {
        detail = "ordering old_to_chain size must match chain_to_old size";
        return false;
    }

    for(SizeT old = 0; old < old_to_chain.size(); ++old)
    {
        if(old_to_chain[old] >= chain_to_old.size())
        {
            detail = fmt::format(
                "ordering old_to_chain[{}]={} is out of chain range [0,{})",
                old,
                old_to_chain[old],
                chain_to_old.size());
            return false;
        }
    }

    for(SizeT chain = 0; chain < chain_to_old.size(); ++chain)
    {
        const SizeT old = chain_to_old[chain];
        if(old >= old_to_chain.size())
        {
            detail = fmt::format(
                "ordering chain_to_old[{}]={} is out of old atom range [0,{})",
                chain,
                old,
                old_to_chain.size());
            return false;
        }
        if(old_to_chain[old] != chain)
        {
            detail = fmt::format(
                "ordering old_to_chain is not the inverse of chain_to_old at chain {}: "
                "chain_to_old[{}]={}, old_to_chain[{}]={}",
                chain,
                chain,
                old,
                old,
                old_to_chain[old]);
            return false;
        }
    }

    return true;
}

class OrderingStructuredChainProvider final : public StructuredChainProvider
{
  public:
    OrderingStructuredChainProvider(SizeT                       block_size,
                                    std::vector<StructuredDofSlot> slots,
                                    double                      block_utilization)
        : m_slots(std::move(slots))
    {
        m_shape.horizon                  = m_slots.empty() ? 0 : (m_slots.back().block + 1);
        m_shape.block_size               = block_size;
        m_shape.nrhs                     = 1;
        m_shape.symmetric_positive_definite = true;
        m_quality.block_utilization      = block_utilization;
        for(const auto& slot : m_slots)
        {
            if(slot.is_padding)
                ++m_quality.padding_dof_count;
            else
                ++m_quality.active_dof_count;
        }
    }

    bool is_available() const override { return !m_slots.empty(); }
    StructuredChainShape shape() const override { return m_shape; }
    span<const StructuredDofSlot> dof_slots() const override { return m_slots; }
    StructuredQualityReport quality_report() const override { return m_quality; }
    void assemble_chain() override {}

  private:
    StructuredChainShape          m_shape;
    StructuredQualityReport       m_quality;
    std::vector<StructuredDofSlot> m_slots;
};

bool build_ordering_provider(const Json&                         ordering,
                             SizeT                               block_size,
                             const std::vector<SocuApproxBlockLayout>& blocks,
                             std::unique_ptr<StructuredChainProvider>& provider,
                             SizeT&                              padding_slot_count,
                             std::string&                        detail)
{
    std::vector<SizeT> chain_to_old;
    std::vector<SizeT> atom_to_block;
    std::vector<SizeT> atom_block_offset;
    std::vector<SizeT> atom_dof_count;
    if(!parse_size_t_array(ordering, "chain_to_old", chain_to_old, detail)
       || !parse_size_t_array(ordering, "atom_to_block", atom_to_block, detail)
       || !parse_size_t_array(ordering, "atom_block_offset", atom_block_offset, detail)
       || !parse_size_t_array(ordering, "atom_dof_count", atom_dof_count, detail))
    {
        return false;
    }

    if(atom_to_block.size() != chain_to_old.size()
       || atom_block_offset.size() != chain_to_old.size()
       || atom_dof_count.size() != chain_to_old.size())
    {
        detail = "ordering mapping arrays have inconsistent atom counts";
        return false;
    }

    std::vector<SizeT> old_dof_offsets(atom_dof_count.size(), 0);
    SizeT              old_dof_offset = 0;
    for(SizeT old = 0; old < atom_dof_count.size(); ++old)
    {
        old_dof_offsets[old] = old_dof_offset;
        old_dof_offset += atom_dof_count[old];
    }

    std::vector<SizeT> valid_lanes_per_block(blocks.size(), 0);
    for(const auto& block : blocks)
    {
        if(block.block >= valid_lanes_per_block.size() || block.dof_count > block_size)
        {
            detail = "ordering block layout is inconsistent with block_size";
            return false;
        }
        valid_lanes_per_block[block.block] = block.dof_count;
    }

    std::vector<StructuredDofSlot> slots;
    slots.reserve(blocks.size() * block_size);
    for(SizeT chain = 0; chain < chain_to_old.size(); ++chain)
    {
        const SizeT old = chain_to_old[chain];
        if(old >= atom_dof_count.size())
        {
            detail = "ordering chain_to_old contains an out-of-range atom";
            return false;
        }

        const SizeT block = atom_to_block[old];
        const SizeT lane_begin = atom_block_offset[old];
        const SizeT dofs = atom_dof_count[old];
        if(block >= blocks.size() || lane_begin + dofs > block_size)
        {
            detail = "ordering atom block/lane mapping exceeds block_size";
            return false;
        }

        for(SizeT local_dof = 0; local_dof < dofs; ++local_dof)
        {
            const SizeT lane = lane_begin + local_dof;
            slots.push_back(StructuredDofSlot{
                .old_dof = static_cast<IndexT>(old_dof_offsets[old] + local_dof),
                .chain_dof = static_cast<IndexT>(block * block_size + lane),
                .block = block,
                .lane = lane,
                .is_padding = false,
                .scatter_write = true});
        }
    }

    padding_slot_count = 0;
    for(SizeT block = 0; block < valid_lanes_per_block.size(); ++block)
    {
        for(SizeT lane = valid_lanes_per_block[block]; lane < block_size; ++lane)
        {
            slots.push_back(StructuredDofSlot{.old_dof = -1,
                                             .chain_dof = static_cast<IndexT>(
                                                 block * block_size + lane),
                                             .block = block,
                                             .lane = lane,
                                             .is_padding = true,
                                             .scatter_write = false});
            ++padding_slot_count;
        }
    }

    const double padded_lanes =
        static_cast<double>(valid_lanes_per_block.size() * block_size);
    const double utilization =
        padded_lanes > 0.0
            ? static_cast<double>(old_dof_offset) / padded_lanes
            : 0.0;
    provider = std::make_unique<OrderingStructuredChainProvider>(
        block_size,
        std::move(slots),
        utilization);
    return provider->is_available();
}

bool validate_dof_coverage(span<const StructuredDofSlot> slots,
                           SizeT                         dof_count,
                           SizeT                         chain_scalar_count,
                           std::vector<IndexT>&          old_to_chain,
                           std::vector<IndexT>&          chain_to_old,
                           StructuredQualityReport&      quality,
                           std::string&                  detail)
{
    constexpr IndexT Missing = -2;
    old_to_chain.assign(dof_count, Missing);
    chain_to_old.assign(chain_scalar_count, Missing);
    quality.active_dof_count          = 0;
    quality.padding_dof_count         = 0;
    quality.duplicate_old_dof_count   = 0;
    quality.duplicate_chain_dof_count = 0;
    quality.missing_old_dof_count     = 0;
    quality.missing_chain_dof_count   = 0;
    quality.complete_dof_coverage     = false;

    for(const StructuredDofSlot& slot : slots)
    {
        if(slot.chain_dof < 0
           || static_cast<SizeT>(slot.chain_dof) >= chain_scalar_count)
        {
            detail = fmt::format("structured slot chain_dof {} is out of range [0,{})",
                                 slot.chain_dof,
                                 chain_scalar_count);
            return false;
        }

        const SizeT chain = static_cast<SizeT>(slot.chain_dof);
        if(chain_to_old[chain] != Missing)
            ++quality.duplicate_chain_dof_count;

        if(slot.is_padding)
        {
            ++quality.padding_dof_count;
            chain_to_old[chain] = -1;
            continue;
        }

        if(slot.old_dof < 0 || static_cast<SizeT>(slot.old_dof) >= dof_count)
        {
            detail = fmt::format("structured slot old_dof {} is out of range [0,{})",
                                 slot.old_dof,
                                 dof_count);
            return false;
        }

        ++quality.active_dof_count;
        const SizeT old = static_cast<SizeT>(slot.old_dof);
        if(old_to_chain[old] != Missing)
            ++quality.duplicate_old_dof_count;
        old_to_chain[old] = slot.chain_dof;
        chain_to_old[chain] = slot.old_dof;
    }

    for(IndexT value : old_to_chain)
    {
        if(value == Missing)
            ++quality.missing_old_dof_count;
    }
    for(IndexT value : chain_to_old)
    {
        if(value == Missing)
            ++quality.missing_chain_dof_count;
    }

    quality.complete_dof_coverage =
        quality.active_dof_count == dof_count
        && quality.duplicate_old_dof_count == 0
        && quality.duplicate_chain_dof_count == 0
        && quality.missing_old_dof_count == 0
        && quality.missing_chain_dof_count == 0;

    if(!quality.complete_dof_coverage)
    {
        detail = fmt::format("structured coverage invalid: active_dofs={}, expected_dofs={}, "
                             "padding_dofs={}, duplicate_old={}, duplicate_chain={}, "
                             "missing_old={}, missing_chain={}",
                             quality.active_dof_count,
                             dof_count,
                             quality.padding_dof_count,
                             quality.duplicate_old_dof_count,
                             quality.duplicate_chain_dof_count,
                             quality.missing_old_dof_count,
                             quality.missing_chain_dof_count);
        return false;
    }

    return true;
}

}  // namespace uipc::backend::cuda_mixed::socu_approx
