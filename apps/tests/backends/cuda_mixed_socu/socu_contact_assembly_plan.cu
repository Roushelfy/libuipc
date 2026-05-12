#include <app/app.h>
#include <linear_system/socu_contact_assembly_plan.h>

#include <cuda_runtime.h>
#include <muda/buffer/device_buffer.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;
using uipc::IndexT;
using uipc::SizeT;
using uipc::Vector2i;
using uipc::Vector3i;
using uipc::Vector4i;
using uipc::span;

static_assert(std::is_trivially_copyable_v<SocuAssemblyDofLane>);
static_assert(std::is_trivially_copyable_v<SocuAssemblySideRecord>);
static_assert(std::is_trivially_copyable_v<SocuContactSourceHeader>);
static_assert(std::is_trivially_copyable_v<SocuContactSourceToProgram>);
static_assert(std::is_trivially_copyable_v<SocuContactProgramHeader>);
static_assert(std::is_trivially_copyable_v<SocuContactMicroTask>);
static_assert(sizeof(SocuAssemblyDofLane) <= 16);
static_assert(sizeof(SocuAssemblySideRecord) <= 64);
static_assert(sizeof(SocuContactSourceHeader) <= 32);
static_assert(sizeof(SocuContactProgramHeader) <= 64);
static_assert(sizeof(SocuContactMicroTask) <= 32);

bool has_cuda_device()
{
    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

SocuNativeVertexDescriptor make_vertex(SocuNativeDescriptorKind kind,
                                       bool fixed,
                                       IndexT old_dof,
                                       IndexT dof_count,
                                       SizeT block,
                                       SizeT lane,
                                       IndexT abd_body = -1,
                                       IndexT abd_j_index = -1)
{
    SocuNativeVertexDescriptor out;
    out.kind = kind;
    out.fixed = fixed;
    out.old_dof = old_dof;
    out.dof_count = dof_count;
    out.block = block;
    out.lane = lane;
    out.abd_body = abd_body;
    out.abd_j_index = abd_j_index;
    out.epoch = 7;
    out.active = true;
    return out;
}

std::vector<SocuNativeVertexDescriptor> fixture_vertices()
{
    std::vector<SocuNativeVertexDescriptor> vertices(8);
    vertices[0] = make_vertex(SocuNativeDescriptorKind::Fem, false, 0, 3, 0, 0);
    vertices[1] = make_vertex(SocuNativeDescriptorKind::Fem, false, 3, 3, 0, 3);
    vertices[2] = make_vertex(SocuNativeDescriptorKind::Fem, false, 16, 3, 1, 0);
    vertices[3] = make_vertex(SocuNativeDescriptorKind::Fem, true, 19, 3, 1, 3);
    vertices[4] = {};
    vertices[5] = make_vertex(SocuNativeDescriptorKind::Abd, false, 32, 12, 2, 0, 0, 5);
    vertices[6] = make_vertex(SocuNativeDescriptorKind::Abd, true, 44, 12, 3, 0, 1, 6);
    return vertices;
}

SocuContactAssemblyPlanM2BuildInput make_input(
    const muda::DeviceBuffer<SocuNativeVertexDescriptor>& vertices,
    const muda::DeviceBuffer<Vector4i>& pts,
    const muda::DeviceBuffer<Vector2i>& phs,
    StructuredContactOffbandPolicy policy)
{
    SocuVertexSidePlanKey side_key;
    side_key.ordering_epoch = 1;
    side_key.native_descriptor_epoch = 7;
    side_key.fixed_mapping_epoch = 3;
    side_key.vertex_projection_epoch = 5;
    side_key.horizon = 4;
    side_key.block_size = 16;

    SocuContactProgramPlanKey program_key;
    program_key.side_key = side_key;
    program_key.contact_topology_epoch = 11;
    program_key.contact_layout_hash = 13;
    program_key.contact_content_hash = 17;
    program_key.offband_policy = policy;

    SocuContactAssemblyPlanM2BuildInput input;
    input.side_key = side_key;
    input.program_key = program_key;
    input.vertex_descriptors = vertices.view();
    input.pt_contacts = pts.view();
    input.ph_contacts = phs.view();
    input.pt_source = SocuContactM2SourceInput{0, 10, SocuContactModelKind::SimplexNormal};
    input.ph_source =
        SocuContactM2SourceInput{1, 11, SocuContactModelKind::VertexHalfPlaneNormal};
    input.offband_policy = policy;
    return input;
}

SocuContactAssemblyPlan build_plan(
    const std::vector<Vector4i>& pts_host,
    const std::vector<Vector2i>& phs_host,
    StructuredContactOffbandPolicy policy,
    SocuContactAssemblyPlanM2Workspace& workspace)
{
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{fixture_vertices()};
    muda::DeviceBuffer<Vector4i> pts{pts_host};
    muda::DeviceBuffer<Vector2i> phs{phs_host};

    SocuContactAssemblyPlan plan;
    build_socu_contact_assembly_plan_m2_active_set_temporary(
        plan,
        workspace,
        make_input(vertices, pts, phs, policy));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    return plan;
}

const SocuAssemblySideRecord& side_for(
    const std::vector<SocuAssemblySideRecord>& sides,
    IndexT                                     global_vertex)
{
    const auto it = std::find_if(sides.begin(),
                                 sides.end(),
                                 [global_vertex](const SocuAssemblySideRecord& side)
                                 {
                                     return side.global_vertex == global_vertex;
                                 });
    REQUIRE(it != sides.end());
    return *it;
}

SocuAssemblySideKind cpu_side_kind(SocuNativeDescriptorKind kind) noexcept
{
    switch(kind)
    {
        case SocuNativeDescriptorKind::Fem:
            return SocuAssemblySideKind::Fem;
        case SocuNativeDescriptorKind::Abd:
            return SocuAssemblySideKind::Abd;
        case SocuNativeDescriptorKind::None:
        default:
            return SocuAssemblySideKind::None;
    }
}

SocuAssemblySideRecord cpu_make_side(
    const std::vector<SocuNativeVertexDescriptor>& vertices,
    IndexT                                         vertex)
{
    SocuAssemblySideRecord side;
    side.global_vertex = vertex;
    if(vertex < 0 || static_cast<SizeT>(vertex) >= vertices.size())
        return side;

    const auto& descriptor = vertices[static_cast<SizeT>(vertex)];
    side.kind = cpu_side_kind(descriptor.kind);
    side.fixed = descriptor.fixed;
    side.writable = descriptor.writable();
    side.old_dof = descriptor.old_dof;
    side.dof_count = descriptor.dof_count;
    side.abd_body = descriptor.abd_body;
    side.abd_jacobian_index = descriptor.abd_j_index;
    side.block = static_cast<std::uint32_t>(descriptor.block);
    side.lane = static_cast<std::uint16_t>(descriptor.lane);
    side.lane_count = descriptor.mapped()
                          ? static_cast<std::uint16_t>(descriptor.dof_count)
                          : std::uint16_t{0};
    return side;
}

bool cpu_find_side_id(const std::vector<IndexT>& sorted_vertices,
                      IndexT                     vertex,
                      SocuAssemblySideId&        side_id)
{
    const auto it = std::lower_bound(sorted_vertices.begin(),
                                     sorted_vertices.end(),
                                     vertex);
    if(it == sorted_vertices.end() || *it != vertex)
    {
        side_id = SocuInvalidAssemblySideId;
        return false;
    }
    side_id =
        static_cast<SocuAssemblySideId>(std::distance(sorted_vertices.begin(), it));
    return true;
}

bool cpu_side_pair_in_band(const SocuAssemblySideRecord& row,
                           const SocuAssemblySideRecord& col,
                           SocuAssemblyBand&             band,
                           std::uint32_t&                block_or_left_block,
                           std::uint8_t&                 flags) noexcept
{
    flags = 0;
    if(row.block == col.block)
    {
        band = SocuAssemblyBand::Diag;
        block_or_left_block = row.block;
        if(row.global_vertex != col.global_vertex)
            flags |= static_cast<std::uint8_t>(
                SocuContactTaskFlag::MirrorDiagBlock);
        return true;
    }

    const SizeT row_block = row.block;
    const SizeT col_block = col.block;
    const SizeT distance =
        row_block > col_block ? row_block - col_block : col_block - row_block;
    if(distance != 1)
        return false;

    band = SocuAssemblyBand::FirstOffdiag;
    block_or_left_block =
        static_cast<std::uint32_t>(row_block < col_block ? row_block : col_block);
    if(row_block < col_block)
        flags |= static_cast<std::uint8_t>(
            SocuContactTaskFlag::TransposedFirstOffdiag);
    return true;
}

SocuAssemblyWriteKind cpu_exact_write_kind(const SocuAssemblySideRecord& row,
                                           const SocuAssemblySideRecord& col,
                                           std::uint8_t& flags) noexcept
{
    if(row.kind == SocuAssemblySideKind::Fem
       && col.kind == SocuAssemblySideKind::Fem)
        return SocuAssemblyWriteKind::ExactFemFem;
    if(row.kind == SocuAssemblySideKind::Abd
       && col.kind == SocuAssemblySideKind::Fem)
        return SocuAssemblyWriteKind::ExactAbdFem;
    if(row.kind == SocuAssemblySideKind::Fem
       && col.kind == SocuAssemblySideKind::Abd)
        return SocuAssemblyWriteKind::ExactFemAbd;
    if(row.kind == SocuAssemblySideKind::Abd
       && col.kind == SocuAssemblySideKind::Abd)
    {
        if(row.abd_body == col.abd_body)
        {
            flags |= static_cast<std::uint8_t>(
                SocuContactTaskFlag::SameAbdBody);
            return SocuAssemblyWriteKind::ExactAbdAbdSameBody;
        }
        return SocuAssemblyWriteKind::ExactAbdAbdCrossBody;
    }
    return SocuAssemblyWriteKind::Skipped;
}

SocuAssemblyWriteKind cpu_diag_block_write_kind(
    const SocuAssemblySideRecord& side) noexcept
{
    return side.kind == SocuAssemblySideKind::Abd
               ? SocuAssemblyWriteKind::DiagBlockAbd
               : SocuAssemblyWriteKind::DiagBlockFem;
}

SocuAssemblyWriteKind cpu_lump_write_kind(
    const SocuAssemblySideRecord& side) noexcept
{
    return side.kind == SocuAssemblySideKind::Abd
               ? SocuAssemblyWriteKind::LumpScalarAbd
               : SocuAssemblyWriteKind::LumpScalarFem;
}

struct CpuOracleTask
{
    SocuAssemblySideId row_side = SocuInvalidAssemblySideId;
    SocuAssemblySideId col_side = SocuInvalidAssemblySideId;
    std::uint8_t local_row = 0;
    std::uint8_t local_col = 0;
    SocuAssemblyBand band = SocuAssemblyBand::Diag;
    SocuAssemblyWriteKind write_kind = SocuAssemblyWriteKind::Skipped;
    std::uint32_t block_or_left_block = 0;
    std::uint8_t flags = 0;
};

struct CpuOracleProgram
{
    SocuContactProgramKind program_kind = SocuContactProgramKind::Skipped;
    SocuContactProgramMapStatus map_status =
        SocuContactProgramMapStatus::Missing;
    std::vector<CpuOracleTask> tasks;
};

void cpu_append_diag_tasks_for_stencil(
    const std::array<SocuAssemblySideId, 4>&      side_ids,
    const std::array<SocuAssemblySideRecord, 4>&  sides,
    int                                           stencil_size,
    bool                                          lump,
    std::vector<CpuOracleTask>&                   tasks)
{
    for(int local = 0; local < stencil_size; ++local)
    {
        if(side_ids[static_cast<SizeT>(local)] == SocuInvalidAssemblySideId)
            continue;
        const auto& side = sides[static_cast<SizeT>(local)];
        if(!side.writable)
            continue;

        CpuOracleTask task;
        task.row_side = side_ids[static_cast<SizeT>(local)];
        task.col_side = side_ids[static_cast<SizeT>(local)];
        task.local_row = static_cast<std::uint8_t>(local);
        task.local_col = static_cast<std::uint8_t>(local);
        task.band = SocuAssemblyBand::Diag;
        task.write_kind =
            lump ? cpu_lump_write_kind(side) : cpu_diag_block_write_kind(side);
        task.block_or_left_block = side.block;
        tasks.push_back(task);
    }
}

CpuOracleProgram cpu_oracle_simplex_program(
    const std::vector<SocuNativeVertexDescriptor>& vertices,
    const std::vector<IndexT>&                     sorted_vertices,
    const std::array<IndexT, 4>&                   stencil,
    int                                           stencil_size,
    StructuredContactOffbandPolicy                policy)
{
    std::array<SocuAssemblySideId, 4> side_ids = {SocuInvalidAssemblySideId,
                                                  SocuInvalidAssemblySideId,
                                                  SocuInvalidAssemblySideId,
                                                  SocuInvalidAssemblySideId};
    std::array<SocuAssemblySideRecord, 4> sides;
    for(int local = 0; local < stencil_size; ++local)
    {
        cpu_find_side_id(sorted_vertices,
                         stencil[static_cast<SizeT>(local)],
                         side_ids[static_cast<SizeT>(local)]);
        sides[static_cast<SizeT>(local)] =
            cpu_make_side(vertices, stencil[static_cast<SizeT>(local)]);
    }

    CpuOracleProgram oracle;
    oracle.map_status = SocuContactProgramMapStatus::Valid;
    bool has_offband = false;
    for(int row = 0; row < stencil_size; ++row)
    {
        for(int col = row; col < stencil_size; ++col)
        {
            if(side_ids[static_cast<SizeT>(row)] == SocuInvalidAssemblySideId
               || side_ids[static_cast<SizeT>(col)] == SocuInvalidAssemblySideId)
                continue;
            const auto& row_side = sides[static_cast<SizeT>(row)];
            const auto& col_side = sides[static_cast<SizeT>(col)];
            if(!row_side.writable || !col_side.writable)
                continue;

            SocuAssemblyBand band = SocuAssemblyBand::Diag;
            std::uint32_t block_or_left_block = 0;
            std::uint8_t flags = 0;
            if(!cpu_side_pair_in_band(row_side,
                                      col_side,
                                      band,
                                      block_or_left_block,
                                      flags))
            {
                has_offband = true;
                continue;
            }

            const auto write_kind =
                cpu_exact_write_kind(row_side, col_side, flags);
            if(write_kind == SocuAssemblyWriteKind::Skipped)
                continue;

            CpuOracleTask task;
            task.row_side = side_ids[static_cast<SizeT>(row)];
            task.col_side = side_ids[static_cast<SizeT>(col)];
            task.local_row = static_cast<std::uint8_t>(row);
            task.local_col = static_cast<std::uint8_t>(col);
            task.band = band;
            task.write_kind = write_kind;
            task.block_or_left_block = block_or_left_block;
            task.flags = flags;
            oracle.tasks.push_back(task);
        }
    }

    if(has_offband)
    {
        if(policy == StructuredContactOffbandPolicy::Drop)
        {
            oracle.program_kind = SocuContactProgramKind::Drop;
            oracle.map_status = SocuContactProgramMapStatus::Dropped;
            oracle.tasks.clear();
        }
        else if(policy == StructuredContactOffbandPolicy::Diag)
        {
            oracle.program_kind = SocuContactProgramKind::Diag;
            oracle.tasks.clear();
            cpu_append_diag_tasks_for_stencil(
                side_ids,
                sides,
                stencil_size,
                false,
                oracle.tasks);
        }
        else
        {
            oracle.program_kind = SocuContactProgramKind::DiagLump;
            oracle.tasks.clear();
            cpu_append_diag_tasks_for_stencil(
                side_ids,
                sides,
                stencil_size,
                true,
                oracle.tasks);
        }
    }
    else if(!oracle.tasks.empty())
    {
        oracle.program_kind = SocuContactProgramKind::Exact;
    }
    else
    {
        oracle.program_kind = SocuContactProgramKind::Skipped;
        oracle.map_status = SocuContactProgramMapStatus::Skipped;
    }
    return oracle;
}

CpuOracleProgram cpu_oracle_ph_program(
    const std::vector<SocuNativeVertexDescriptor>& vertices,
    const std::vector<IndexT>&                     sorted_vertices,
    IndexT                                        vertex)
{
    SocuAssemblySideId side_id = SocuInvalidAssemblySideId;
    cpu_find_side_id(sorted_vertices, vertex, side_id);
    const auto side = cpu_make_side(vertices, vertex);

    CpuOracleProgram oracle;
    if(side_id != SocuInvalidAssemblySideId && side.writable)
    {
        std::uint8_t flags = 0;
        CpuOracleTask task;
        task.row_side = side_id;
        task.col_side = side_id;
        task.band = SocuAssemblyBand::Diag;
        task.write_kind = cpu_exact_write_kind(side, side, flags);
        task.block_or_left_block = side.block;
        task.flags = flags;
        oracle.program_kind = SocuContactProgramKind::Exact;
        oracle.map_status = SocuContactProgramMapStatus::Valid;
        oracle.tasks.push_back(task);
    }
    else
    {
        oracle.program_kind = SocuContactProgramKind::Skipped;
        oracle.map_status = SocuContactProgramMapStatus::Skipped;
    }
    return oracle;
}

void require_program_matches_oracle(
    const SocuContactProgramHeader&       program,
    const SocuContactSourceToProgram&     map,
    const std::vector<SocuContactMicroTask>& tasks,
    const CpuOracleProgram&               oracle)
{
    CHECK(program.program_kind == oracle.program_kind);
    CHECK(map.status == oracle.map_status);
    if(oracle.map_status == SocuContactProgramMapStatus::Valid)
        CHECK(map.program_id != SocuInvalidContactProgramId);
    else
        CHECK(map.program_id == SocuInvalidContactProgramId);

    CHECK(program.task_count == oracle.tasks.size());
    REQUIRE(static_cast<SizeT>(program.first_task) + program.task_count
            <= tasks.size());
    for(SizeT i = 0; i < oracle.tasks.size(); ++i)
    {
        const auto& task = tasks[static_cast<SizeT>(program.first_task) + i];
        const auto& expected = oracle.tasks[i];
        CHECK(task.row_side == expected.row_side);
        CHECK(task.col_side == expected.col_side);
        CHECK(task.local_row_vertex == expected.local_row);
        CHECK(task.local_col_vertex == expected.local_col);
        CHECK(task.band == expected.band);
        CHECK(task.write_kind == expected.write_kind);
        CHECK(task.block_or_left_block == expected.block_or_left_block);
        CHECK(task.flags == expected.flags);
    }
}

SizeT total_program_task_count(
    const std::vector<SocuContactProgramHeader>& programs) noexcept
{
    SizeT total = 0;
    for(const auto& program : programs)
        total += program.task_count;
    return total;
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream ifs{path};
    REQUIRE(ifs.good());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}
}  // namespace

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_side_table_active_set",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_plan({Vector4i{0, 1, 2, 0}, Vector4i{3, 4, 5, 0}},
                           {Vector2i{2, 90}, Vector2i{5, 91}},
                           StructuredContactOffbandPolicy::Drop,
                           workspace);

    std::vector<IndexT> sorted_vertices;
    std::vector<SocuAssemblySideRecord> sides;
    std::vector<SocuAssemblyDofLane> lanes;
    plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);
    plan.side_plan.sides.copy_to(sides);
    plan.side_plan.lanes.copy_to(lanes);

    CHECK(sorted_vertices == std::vector<IndexT>{0, 1, 2, 3, 4, 5});
    CHECK(sides.size() == 6);
    CHECK(plan.side_plan.coverage.mode
          == SocuVertexSideCoverageMode::ActiveSetTemporary);
    CHECK(plan.side_plan.coverage.covered_vertex_count == 6);
    CHECK(plan.side_plan.coverage.complete_for_current_contacts);
    CHECK(lanes.size() == 24);

    const auto& fem0 = side_for(sides, 0);
    CHECK(fem0.kind == SocuAssemblySideKind::Fem);
    CHECK(fem0.writable);
    CHECK(fem0.lane_count == 3);
    CHECK(lanes[fem0.first_lane + 2].block == 0);
    CHECK(lanes[fem0.first_lane + 2].lane == 2);

    const auto& fixed = side_for(sides, 3);
    CHECK(fixed.kind == SocuAssemblySideKind::Fem);
    CHECK(fixed.fixed);
    CHECK(!fixed.writable);
    CHECK(fixed.lane_count == 3);

    const auto& unmapped = side_for(sides, 4);
    CHECK(unmapped.kind == SocuAssemblySideKind::None);
    CHECK(!unmapped.writable);
    CHECK(unmapped.lane_count == 0);

    const auto& abd = side_for(sides, 5);
    CHECK(abd.kind == SocuAssemblySideKind::Abd);
    CHECK(abd.writable);
    CHECK(abd.abd_body == 0);
    CHECK(abd.lane_count == 12);
    CHECK(lanes[abd.first_lane + 11].block == 2);
    CHECK(lanes[abd.first_lane + 11].lane == 11);
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_program_map_pt_ph",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_plan({Vector4i{0, 1, 2, 0}},
                           {Vector2i{2, 90}},
                           StructuredContactOffbandPolicy::Drop,
                           workspace);

    std::vector<SocuContactSourceHeader> sources;
    std::vector<SocuContactProgramHeader> programs;
    std::vector<SocuContactSourceToProgram> maps;
    std::vector<SocuContactMicroTask> tasks;
    std::vector<IndexT> sorted_vertices;
    plan.program_plan.sources.copy_to(sources);
    plan.program_plan.programs.copy_to(programs);
    plan.program_plan.source_to_program.copy_to(maps);
    plan.program_plan.tasks.copy_to(tasks);
    plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);

    REQUIRE(sources.size() == 2);
    CHECK(sources[0].source_id == 0);
    CHECK(sources[0].family == SocuContactFamily::PT);
    CHECK(sources[0].contact_count == 1);
    CHECK(sources[0].first_program == 0);
    CHECK(sources[0].first_source_to_program == 0);
    CHECK(sources[1].source_id == 1);
    CHECK(sources[1].family == SocuContactFamily::PH);
    CHECK(sources[1].contact_count == 1);
    CHECK(sources[1].first_program == 1);
    CHECK(sources[1].first_source_to_program == 1);

    REQUIRE(maps.size() == 2);
    CHECK(maps[0].program_id == 0);
    CHECK(maps[0].status == SocuContactProgramMapStatus::Valid);
    CHECK(maps[1].program_id == 1);
    CHECK(maps[1].status == SocuContactProgramMapStatus::Valid);

    REQUIRE(programs.size() == 2);
    CHECK(programs[0].source_id == 0);
    CHECK(programs[0].local_contact_id == 0);
    CHECK(programs[0].program_kind == SocuContactProgramKind::Exact);
    CHECK(programs[0].task_count > 0);

    CHECK(programs[1].source_id == 1);
    CHECK(programs[1].family == SocuContactFamily::PH);
    CHECK(programs[1].program_kind == SocuContactProgramKind::Exact);
    CHECK(programs[1].task_count == 1);
    CHECK(programs[1].side_ids[0] != SocuInvalidAssemblySideId);
    CHECK(programs[1].side_ids[1] == SocuInvalidAssemblySideId);
    CHECK(std::find(sorted_vertices.begin(), sorted_vertices.end(), 90)
          == sorted_vertices.end());

    REQUIRE(!tasks.empty());
    const auto ph_task = tasks[programs[1].first_task];
    CHECK(ph_task.row_side == programs[1].side_ids[0]);
    CHECK(ph_task.col_side == programs[1].side_ids[0]);
    CHECK(ph_task.band == SocuAssemblyBand::Diag);
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_simplex_families_and_friction_sources",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{fixture_vertices()};
    muda::DeviceBuffer<Vector4i> pts{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 0}}};
    muda::DeviceBuffer<Vector4i> ees{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 3}}};
    muda::DeviceBuffer<Vector3i> pes{
        std::vector<Vector3i>{Vector3i{0, 1, 2}}};
    muda::DeviceBuffer<Vector2i> pps{
        std::vector<Vector2i>{Vector2i{1, 2}}};
    muda::DeviceBuffer<Vector2i> phs{std::vector<Vector2i>{}};
    muda::DeviceBuffer<Vector4i> friction_pts{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 0}}};
    muda::DeviceBuffer<Vector4i> friction_ees{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 3}}};
    muda::DeviceBuffer<Vector3i> friction_pes{
        std::vector<Vector3i>{Vector3i{0, 1, 2}}};
    muda::DeviceBuffer<Vector2i> friction_pps{
        std::vector<Vector2i>{Vector2i{1, 2}}};
    muda::DeviceBuffer<Vector2i> friction_phs{
        std::vector<Vector2i>{Vector2i{2, 99}}};

    auto input = make_input(vertices, pts, phs, StructuredContactOffbandPolicy::Drop);
    input.ee_contacts = ees.view();
    input.pe_contacts = pes.view();
    input.pp_contacts = pps.view();
    input.friction_pt_contacts = friction_pts.view();
    input.friction_ee_contacts = friction_ees.view();
    input.friction_pe_contacts = friction_pes.view();
    input.friction_pp_contacts = friction_pps.view();
    input.friction_ph_contacts = friction_phs.view();
    input.ee_source = SocuContactM2SourceInput{
        1,
        10,
        SocuContactModelKind::SimplexNormal};
    input.pe_source = SocuContactM2SourceInput{
        2,
        10,
        SocuContactModelKind::SimplexNormal};
    input.pp_source = SocuContactM2SourceInput{
        3,
        10,
        SocuContactModelKind::SimplexNormal};
    input.ph_source = {};
    input.friction_pt_source = SocuContactM2SourceInput{
        4,
        20,
        SocuContactModelKind::SimplexFrictional};
    input.friction_ee_source = SocuContactM2SourceInput{
        5,
        20,
        SocuContactModelKind::SimplexFrictional};
    input.friction_pe_source = SocuContactM2SourceInput{
        6,
        20,
        SocuContactModelKind::SimplexFrictional};
    input.friction_pp_source = SocuContactM2SourceInput{
        7,
        20,
        SocuContactModelKind::SimplexFrictional};
    input.friction_ph_source = SocuContactM2SourceInput{
        8,
        21,
        SocuContactModelKind::VertexHalfPlaneFrictional};

    SocuContactAssemblyPlanM2Workspace workspace;
    SocuContactAssemblyPlan plan;
    build_socu_contact_assembly_plan_m2_active_set_temporary(
        plan,
        workspace,
        input);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<SocuContactSourceHeader> sources;
    std::vector<SocuContactProgramHeader> programs;
    std::vector<SocuContactSourceToProgram> maps;
    std::vector<IndexT> sorted_vertices;
    plan.program_plan.sources.copy_to(sources);
    plan.program_plan.programs.copy_to(programs);
    plan.program_plan.source_to_program.copy_to(maps);
    plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);

    REQUIRE(sources.size() == 9);
    REQUIRE(programs.size() == 9);
    REQUIRE(maps.size() == 9);
    CHECK(sorted_vertices == std::vector<IndexT>{0, 1, 2, 3});
    CHECK(std::find(sorted_vertices.begin(), sorted_vertices.end(), 99)
          == sorted_vertices.end());

    CHECK(sources[0].source_id == 0);
    CHECK(sources[0].family == SocuContactFamily::PT);
    CHECK(sources[0].model == SocuContactModelKind::SimplexNormal);
    CHECK(sources[1].family == SocuContactFamily::EE);
    CHECK(sources[1].stencil_size == 4);
    CHECK(sources[2].family == SocuContactFamily::PE);
    CHECK(sources[2].stencil_size == 3);
    CHECK(sources[3].family == SocuContactFamily::PP);
    CHECK(sources[3].stencil_size == 2);
    CHECK(sources[4].source_id == 4);
    CHECK(sources[4].family == SocuContactFamily::PT);
    CHECK(sources[4].model == SocuContactModelKind::SimplexFrictional);
    CHECK(sources[5].family == SocuContactFamily::EE);
    CHECK(sources[5].model == SocuContactModelKind::SimplexFrictional);
    CHECK(sources[6].family == SocuContactFamily::PE);
    CHECK(sources[6].model == SocuContactModelKind::SimplexFrictional);
    CHECK(sources[7].family == SocuContactFamily::PP);
    CHECK(sources[7].model == SocuContactModelKind::SimplexFrictional);
    CHECK(sources[8].family == SocuContactFamily::PH);
    CHECK(sources[8].model == SocuContactModelKind::VertexHalfPlaneFrictional);

    for(SizeT i = 0; i < programs.size(); ++i)
    {
        CHECK(maps[i].program_id == i);
        CHECK(maps[i].status == SocuContactProgramMapStatus::Valid);
        CHECK(programs[i].source_id == i);
        CHECK(programs[i].local_contact_id == 0);
        CHECK(programs[i].task_count > 0);
    }
    CHECK(programs[0].family == SocuContactFamily::PT);
    CHECK(programs[1].family == SocuContactFamily::EE);
    CHECK(programs[2].family == SocuContactFamily::PE);
    CHECK(programs[3].family == SocuContactFamily::PP);
    CHECK(programs[4].family == SocuContactFamily::PT);
    CHECK(programs[4].model == SocuContactModelKind::SimplexFrictional);
    CHECK(programs[4].source_id != programs[0].source_id);
    CHECK(programs[5].family == SocuContactFamily::EE);
    CHECK(programs[6].family == SocuContactFamily::PE);
    CHECK(programs[7].family == SocuContactFamily::PP);
    CHECK(programs[8].family == SocuContactFamily::PH);
    CHECK(programs[8].model == SocuContactModelKind::VertexHalfPlaneFrictional);
    CHECK(programs[8].side_ids[1] == SocuInvalidAssemblySideId);
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_source_span_multiple_reporters",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{fixture_vertices()};
    muda::DeviceBuffer<Vector4i> reporter_a_pts{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 0}}};
    muda::DeviceBuffer<Vector4i> reporter_b_pts{
        std::vector<Vector4i>{Vector4i{1, 2, 0, 1}}};

    std::vector<SocuContactM2SourceInput> sources(2);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PT;
    sources[0].stencil_size = 4;
    sources[0].stencil4 = reporter_a_pts.view();
    sources[1].source_id = 1;
    sources[1].reporter_id = 11;
    sources[1].model = SocuContactModelKind::SimplexNormal;
    sources[1].family = SocuContactFamily::PT;
    sources[1].stencil_size = 4;
    sources[1].stencil4 = reporter_b_pts.view();

    muda::DeviceBuffer<Vector4i> empty_pts{std::vector<Vector4i>{}};
    muda::DeviceBuffer<Vector2i> empty_phs{std::vector<Vector2i>{}};
    auto input = make_input(vertices,
                            empty_pts,
                            empty_phs,
                            StructuredContactOffbandPolicy::Drop);
    input.vertex_descriptors = vertices.view();
    input.sources = span<const SocuContactM2SourceInput>{sources};

    SocuContactAssemblyPlanM2Workspace workspace;
    SocuContactAssemblyPlan plan;
    build_socu_contact_assembly_plan_m2_active_set_temporary(
        plan,
        workspace,
        input);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<SocuContactSourceHeader> headers;
    std::vector<SocuContactProgramHeader> programs;
    std::vector<SocuContactSourceToProgram> maps;
    plan.program_plan.sources.copy_to(headers);
    plan.program_plan.programs.copy_to(programs);
    plan.program_plan.source_to_program.copy_to(maps);

    REQUIRE(headers.size() == 2);
    REQUIRE(programs.size() == 2);
    REQUIRE(maps.size() == 2);
    CHECK(headers[0].source_id == 0);
    CHECK(headers[0].reporter_id == 10);
    CHECK(headers[1].source_id == 1);
    CHECK(headers[1].reporter_id == 11);
    CHECK(programs[0].source_id == 0);
    CHECK(programs[0].local_contact_id == 0);
    CHECK(programs[1].source_id == 1);
    CHECK(programs[1].local_contact_id == 0);
    CHECK(maps[0].program_id == 0);
    CHECK(maps[1].program_id == 1);
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_dense_source_validation",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{fixture_vertices()};
    muda::DeviceBuffer<Vector4i> pts{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 0}}};
    muda::DeviceBuffer<Vector4i> ees{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 3}}};
    muda::DeviceBuffer<Vector2i> phs{std::vector<Vector2i>{}};

    auto require_invalid_argument_containing =
        [](SocuContactAssemblyPlanM2BuildInput input, const char* token)
    {
        SocuContactAssemblyPlanM2Workspace workspace;
        SocuContactAssemblyPlan plan;
        try
        {
            build_socu_contact_assembly_plan_m2_active_set_temporary(
                plan,
                workspace,
                input);
            FAIL("expected invalid_argument");
        }
        catch(const std::invalid_argument& e)
        {
            CHECK(std::string{e.what()}.find(token) != std::string::npos);
        }
    };

    {
        auto input = make_input(vertices, pts, phs, StructuredContactOffbandPolicy::Drop);
        input.pt_source.source_id = 1;
        input.ph_source = {};
        require_invalid_argument_containing(input, "out_of_range");
    }

    {
        auto input = make_input(vertices, pts, phs, StructuredContactOffbandPolicy::Drop);
        input.ee_contacts = ees.view();
        input.ee_source = SocuContactM2SourceInput{
            0,
            10,
            SocuContactModelKind::SimplexNormal};
        input.ph_source = {};
        require_invalid_argument_containing(input, "duplicate");
    }

    {
        auto input = make_input(vertices, pts, phs, StructuredContactOffbandPolicy::Drop);
        input.pt_source = {};
        input.ph_source = {};
        SocuContactAssemblyPlanM2Workspace workspace;
        SocuContactAssemblyPlan plan;
        CHECK_THROWS_AS(
            build_socu_contact_assembly_plan_m2_active_set_temporary(
                plan,
                workspace,
                input),
            std::invalid_argument);
    }
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_offband_policy",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_plan({Vector4i{0, 5, 2, 0}},
                               {},
                               StructuredContactOffbandPolicy::Drop,
                               workspace);
        std::vector<SocuContactProgramHeader> programs;
        std::vector<SocuContactSourceToProgram> maps;
        std::vector<SocuContactMicroTask> tasks;
        plan.program_plan.programs.copy_to(programs);
        plan.program_plan.source_to_program.copy_to(maps);
        plan.program_plan.tasks.copy_to(tasks);
        REQUIRE(programs.size() == 1);
        CHECK(programs[0].program_kind == SocuContactProgramKind::Drop);
        CHECK(programs[0].task_count == 0);
        REQUIRE(!maps.empty());
        CHECK(maps[0].status == SocuContactProgramMapStatus::Dropped);
        CHECK(maps[0].program_id == SocuInvalidContactProgramId);
        CHECK(tasks.empty());
    }

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_plan({Vector4i{0, 5, 2, 0}},
                               {},
                               StructuredContactOffbandPolicy::Diag,
                               workspace);
        std::vector<SocuContactProgramHeader> programs;
        std::vector<SocuContactMicroTask> tasks;
        plan.program_plan.programs.copy_to(programs);
        plan.program_plan.tasks.copy_to(tasks);
        REQUIRE(programs.size() == 1);
        CHECK(programs[0].program_kind == SocuContactProgramKind::Diag);
        CHECK(programs[0].task_count > 0);
        REQUIRE(!tasks.empty());
        CHECK(tasks[programs[0].first_task].write_kind
              == SocuAssemblyWriteKind::DiagBlockFem);
    }

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_plan({Vector4i{0, 5, 2, 0}},
                               {},
                               StructuredContactOffbandPolicy::DiagLump,
                               workspace);
        std::vector<SocuContactProgramHeader> programs;
        std::vector<SocuContactMicroTask> tasks;
        plan.program_plan.programs.copy_to(programs);
        plan.program_plan.tasks.copy_to(tasks);
        REQUIRE(programs.size() == 1);
        CHECK(programs[0].program_kind == SocuContactProgramKind::DiagLump);
        CHECK(programs[0].task_count > 0);
        REQUIRE(!tasks.empty());
        CHECK(tasks[programs[0].first_task].write_kind
              == SocuAssemblyWriteKind::LumpScalarFem);
    }
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_buckets_and_stats",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_plan({Vector4i{0, 1, 2, 0},
                            Vector4i{0, 5, 2, 0},
                            Vector4i{3, 4, 3, 4}},
                           {},
                           StructuredContactOffbandPolicy::Drop,
                           workspace);

    std::vector<SocuContactProgramHeader> programs;
    std::vector<SocuContactSourceToProgram> maps;
    std::vector<SocuContactProgramBucket> buckets;
    plan.program_plan.programs.copy_to(programs);
    plan.program_plan.source_to_program.copy_to(maps);
    plan.program_plan.buckets.copy_to(buckets);

    REQUIRE(programs.size() == 3);
    REQUIRE(maps.size() == 3);
    CHECK(programs[0].program_kind == SocuContactProgramKind::Exact);
    CHECK(programs[1].program_kind == SocuContactProgramKind::Drop);
    CHECK(programs[2].program_kind == SocuContactProgramKind::Skipped);
    CHECK(maps[0].status == SocuContactProgramMapStatus::Valid);
    CHECK(maps[1].status == SocuContactProgramMapStatus::Dropped);
    CHECK(maps[2].status == SocuContactProgramMapStatus::Skipped);
    CHECK(maps[0].program_id == 0);
    CHECK(maps[1].program_id == SocuInvalidContactProgramId);
    CHECK(maps[2].program_id == SocuInvalidContactProgramId);

    REQUIRE(buckets.size() == 3);
    CHECK(buckets[0].program_kind == SocuContactProgramKind::Exact);
    CHECK(buckets[0].execution_strategy
          == SocuContactExecutionStrategy::DirectScatter);
    CHECK(buckets[0].first_program == 0);
    CHECK(buckets[0].program_count == 1);
    CHECK(buckets[1].program_kind == SocuContactProgramKind::Drop);
    CHECK(buckets[1].execution_strategy == SocuContactExecutionStrategy::DetectOnly);
    CHECK(buckets[1].first_program == 1);
    CHECK(buckets[1].program_count == 1);
    CHECK(buckets[2].program_kind == SocuContactProgramKind::Skipped);
    CHECK(buckets[2].execution_strategy == SocuContactExecutionStrategy::DetectOnly);
    CHECK(buckets[2].first_program == 2);
    CHECK(buckets[2].program_count == 1);

    const auto& stats = plan.program_plan.last_stats;
    CHECK(stats.source_id_validation_status
          == SocuContactSourceIdValidationStatus::ValidDense);
    CHECK(stats.source_count == 2);
    CHECK(stats.program_count == 3);
    CHECK(stats.source_to_program_count == 3);
    CHECK(stats.valid_program_map_count == 1);
    CHECK(stats.missing_program_map_count == 0);
    CHECK(stats.invalid_program_map_count == 0);
    CHECK(stats.dropped_program_map_count == 1);
    CHECK(stats.skipped_program_map_count == 1);
    CHECK(stats.mixed_rejected_program_map_count == 0);
    CHECK(stats.bucket_count == 3);
    CHECK(stats.exact_program_count == 1);
    CHECK(stats.drop_program_count == 1);
    CHECK(stats.skipped_program_count == 1);
    CHECK(stats.diag_program_count == 0);
    CHECK(stats.diag_lump_program_count == 0);
    CHECK(stats.task_count == programs[0].task_count);
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_global_side_coverage",
          "[cuda_mixed_socu][contract][socu_approx][m2][m2b]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    const auto vertices_host = fixture_vertices();
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{vertices_host};
    muda::DeviceBuffer<Vector4i> pts_a{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 0}}};
    muda::DeviceBuffer<Vector4i> pts_b{
        std::vector<Vector4i>{Vector4i{0, 5, 2, 0}}};
    muda::DeviceBuffer<Vector4i> empty_pts{std::vector<Vector4i>{}};
    muda::DeviceBuffer<Vector2i> empty_phs{std::vector<Vector2i>{}};

    SocuContactAssemblyPlanM2Workspace workspace;
    SocuContactAssemblyPlan plan;

    auto input_a = make_input(vertices,
                              pts_a,
                              empty_phs,
                              StructuredContactOffbandPolicy::Drop);
    input_a.ph_source = {};
    input_a.side_coverage_mode = SocuVertexSideCoverageMode::Global;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_a);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> sorted_a;
    std::vector<SocuAssemblySideRecord> sides_a;
    std::vector<SocuAssemblyDofLane> lanes_a;
    plan.side_plan.sorted_side_vertices.copy_to(sorted_a);
    plan.side_plan.sides.copy_to(sides_a);
    plan.side_plan.lanes.copy_to(lanes_a);

    CHECK(sorted_a == std::vector<IndexT>{0, 1, 2, 3, 4, 5, 6, 7});
    CHECK(sides_a.size() == vertices_host.size());
    CHECK(lanes_a.size() == 36);
    CHECK(plan.side_plan.coverage.mode == SocuVertexSideCoverageMode::Global);
    CHECK(plan.side_plan.last_stats.coverage_mode
          == SocuVertexSideCoverageMode::Global);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 0);
    CHECK(plan.side_plan.last_stats.side_count == vertices_host.size());
    CHECK(side_for(sides_a, 5).kind == SocuAssemblySideKind::Abd);
    CHECK(side_for(sides_a, 5).writable);

    auto input_b = make_input(vertices,
                              pts_b,
                              empty_phs,
                              StructuredContactOffbandPolicy::Drop);
    input_b.ph_source = {};
    input_b.side_coverage_mode = SocuVertexSideCoverageMode::Global;
    input_b.program_key.contact_topology_epoch++;
    input_b.program_key.contact_content_hash++;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_b);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> sorted_b;
    std::vector<SocuContactProgramHeader> programs_b;
    std::vector<SocuContactSourceToProgram> maps_b;
    plan.side_plan.sorted_side_vertices.copy_to(sorted_b);
    plan.program_plan.programs.copy_to(programs_b);
    plan.program_plan.source_to_program.copy_to(maps_b);

    CHECK(sorted_b == sorted_a);
    CHECK(plan.side_plan.coverage.mode == SocuVertexSideCoverageMode::Global);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 1);
    CHECK(plan.side_plan.last_stats.side_coverage_refresh_count == 0);
    CHECK(plan.side_plan.last_stats.active_side_set_changed_count == 0);
    REQUIRE(programs_b.size() == 1);
    REQUIRE(maps_b.size() == 1);
    CHECK(programs_b[0].program_kind == SocuContactProgramKind::Drop);
    CHECK(programs_b[0].side_ids[1] == 5);
    CHECK(maps_b[0].status == SocuContactProgramMapStatus::Dropped);

    auto input_empty = make_input(vertices,
                                  empty_pts,
                                  empty_phs,
                                  StructuredContactOffbandPolicy::Drop);
    input_empty.pt_source = {};
    input_empty.ph_source = {};
    input_empty.side_coverage_mode = SocuVertexSideCoverageMode::Global;
    input_empty.program_key.contact_topology_epoch += 2;
    input_empty.program_key.contact_content_hash += 2;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_empty);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> sorted_empty;
    std::vector<SocuContactProgramHeader> programs_empty;
    plan.side_plan.sorted_side_vertices.copy_to(sorted_empty);
    plan.program_plan.programs.copy_to(programs_empty);
    CHECK(sorted_empty == sorted_a);
    CHECK(programs_empty.empty());
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 1);
    CHECK(plan.side_plan.last_stats.side_count == vertices_host.size());
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_demand_filled_side_coverage",
          "[cuda_mixed_socu][contract][socu_approx][m2][m2b]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    const auto vertices_host = fixture_vertices();
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{vertices_host};
    muda::DeviceBuffer<Vector4i> pts_a{
        std::vector<Vector4i>{Vector4i{0, 1, 2, 0}}};
    muda::DeviceBuffer<Vector4i> pts_b{
        std::vector<Vector4i>{Vector4i{0, 5, 2, 0}}};
    muda::DeviceBuffer<Vector4i> pts_c{
        std::vector<Vector4i>{Vector4i{5, 2, 0, 0}}};
    muda::DeviceBuffer<Vector4i> empty_pts{std::vector<Vector4i>{}};
    muda::DeviceBuffer<Vector2i> empty_phs{std::vector<Vector2i>{}};

    SocuContactAssemblyPlanM2Workspace workspace;
    SocuContactAssemblyPlan plan;

    auto input_a = make_input(vertices,
                              pts_a,
                              empty_phs,
                              StructuredContactOffbandPolicy::Drop);
    input_a.ph_source = {};
    input_a.side_coverage_mode = SocuVertexSideCoverageMode::DemandFilled;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_a);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> vertices_a;
    std::vector<SocuAssemblySideId> lookup_a;
    plan.side_plan.sorted_side_vertices.copy_to(vertices_a);
    plan.side_plan.vertex_to_side_id.copy_to(lookup_a);
    CHECK(vertices_a == std::vector<IndexT>{0, 1, 2});
    REQUIRE(lookup_a.size() == vertices_host.size());
    CHECK(lookup_a[0] == 0);
    CHECK(lookup_a[1] == 1);
    CHECK(lookup_a[2] == 2);
    CHECK(lookup_a[5] == SocuInvalidAssemblySideId);
    CHECK(plan.side_plan.coverage.mode
          == SocuVertexSideCoverageMode::DemandFilled);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 0);
    CHECK(plan.side_plan.last_stats.side_coverage_fill_count == 0);
    CHECK(plan.side_plan.last_stats.side_count == 3);
    CHECK(plan.side_plan.last_stats.active_side_vertex_count == 3);

    auto input_b = make_input(vertices,
                              pts_b,
                              empty_phs,
                              StructuredContactOffbandPolicy::Drop);
    input_b.ph_source = {};
    input_b.side_coverage_mode = SocuVertexSideCoverageMode::DemandFilled;
    input_b.program_key.contact_topology_epoch++;
    input_b.program_key.contact_content_hash++;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_b);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> vertices_b;
    std::vector<SocuAssemblySideId> lookup_b;
    std::vector<SocuContactProgramHeader> programs_b;
    std::vector<SocuContactSourceToProgram> maps_b;
    plan.side_plan.sorted_side_vertices.copy_to(vertices_b);
    plan.side_plan.vertex_to_side_id.copy_to(lookup_b);
    plan.program_plan.programs.copy_to(programs_b);
    plan.program_plan.source_to_program.copy_to(maps_b);
    CHECK(vertices_b == std::vector<IndexT>{0, 1, 2, 5});
    CHECK(lookup_b[0] == 0);
    CHECK(lookup_b[1] == 1);
    CHECK(lookup_b[2] == 2);
    CHECK(lookup_b[5] == 3);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 0);
    CHECK(plan.side_plan.last_stats.side_coverage_fill_count == 1);
    CHECK(plan.side_plan.last_stats.side_coverage_refresh_count == 0);
    CHECK(plan.side_plan.last_stats.active_side_vertex_count == 3);
    CHECK(plan.side_plan.last_stats.side_count == 4);
    REQUIRE(programs_b.size() == 1);
    REQUIRE(maps_b.size() == 1);
    CHECK(programs_b[0].program_kind == SocuContactProgramKind::Drop);
    CHECK(programs_b[0].side_ids[0] == 0);
    CHECK(programs_b[0].side_ids[1] == 3);
    CHECK(programs_b[0].side_ids[2] == 2);
    CHECK(maps_b[0].status == SocuContactProgramMapStatus::Dropped);

    auto input_c = make_input(vertices,
                              pts_c,
                              empty_phs,
                              StructuredContactOffbandPolicy::Drop);
    input_c.ph_source = {};
    input_c.side_coverage_mode = SocuVertexSideCoverageMode::DemandFilled;
    input_c.program_key.contact_topology_epoch += 2;
    input_c.program_key.contact_content_hash += 2;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_c);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> vertices_c;
    std::vector<SocuContactProgramHeader> programs_c;
    plan.side_plan.sorted_side_vertices.copy_to(vertices_c);
    plan.program_plan.programs.copy_to(programs_c);
    CHECK(vertices_c == vertices_b);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 1);
    CHECK(plan.side_plan.last_stats.side_coverage_fill_count == 0);
    CHECK(plan.side_plan.last_stats.side_count == 4);
    REQUIRE(programs_c.size() == 1);
    CHECK(programs_c[0].side_ids[0] == 3);
    CHECK(programs_c[0].side_ids[1] == 2);
    CHECK(programs_c[0].side_ids[2] == 0);

    auto input_empty = make_input(vertices,
                                  empty_pts,
                                  empty_phs,
                                  StructuredContactOffbandPolicy::Drop);
    input_empty.pt_source = {};
    input_empty.ph_source = {};
    input_empty.side_coverage_mode = SocuVertexSideCoverageMode::DemandFilled;
    input_empty.program_key.contact_topology_epoch += 3;
    input_empty.program_key.contact_content_hash += 3;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_empty);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> vertices_empty;
    plan.side_plan.sorted_side_vertices.copy_to(vertices_empty);
    CHECK(vertices_empty == vertices_b);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 1);
    CHECK(plan.side_plan.last_stats.side_coverage_fill_count == 0);
    CHECK(plan.side_plan.last_stats.active_side_vertex_count == 0);

    auto input_reset = make_input(vertices,
                                  pts_c,
                                  empty_phs,
                                  StructuredContactOffbandPolicy::Drop);
    input_reset.ph_source = {};
    input_reset.side_coverage_mode = SocuVertexSideCoverageMode::DemandFilled;
    input_reset.side_key.native_descriptor_epoch++;
    input_reset.program_key.side_key = input_reset.side_key;
    input_reset.program_key.contact_topology_epoch += 4;
    input_reset.program_key.contact_content_hash += 4;
    build_socu_contact_assembly_plan_m2(plan, workspace, input_reset);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> vertices_reset;
    std::vector<SocuAssemblySideId> lookup_reset;
    plan.side_plan.sorted_side_vertices.copy_to(vertices_reset);
    plan.side_plan.vertex_to_side_id.copy_to(lookup_reset);
    CHECK(vertices_reset == std::vector<IndexT>{0, 2, 5});
    CHECK(lookup_reset[0] == 0);
    CHECK(lookup_reset[2] == 1);
    CHECK(lookup_reset[5] == 2);
    CHECK(lookup_reset[1] == SocuInvalidAssemblySideId);
    CHECK(plan.side_plan.last_stats.side_coverage_hit_count == 0);
    CHECK(plan.side_plan.last_stats.side_coverage_fill_count == 0);
    CHECK(plan.side_plan.last_stats.side_count == 3);
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_symbolic_cpu_oracle",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact assembly plan tests");

    const auto vertices_host = fixture_vertices();
    auto require_single_program =
        [&](SocuContactAssemblyPlan& plan, const CpuOracleProgram& oracle)
    {
        std::vector<SocuContactProgramHeader> programs;
        std::vector<SocuContactSourceToProgram> maps;
        std::vector<SocuContactMicroTask> tasks;
        plan.program_plan.programs.copy_to(programs);
        plan.program_plan.source_to_program.copy_to(maps);
        plan.program_plan.tasks.copy_to(tasks);

        REQUIRE(programs.size() == 1);
        REQUIRE(maps.size() == 1);
        CHECK(total_program_task_count(programs) == tasks.size());
        require_program_matches_oracle(programs[0], maps[0], tasks, oracle);
    };

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_plan({Vector4i{0, 1, 2, 0}},
                               {},
                               StructuredContactOffbandPolicy::Drop,
                               workspace);
        std::vector<IndexT> sorted_vertices;
        plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);
        const auto oracle = cpu_oracle_simplex_program(
            vertices_host,
            sorted_vertices,
            std::array<IndexT, 4>{0, 1, 2, 0},
            4,
            StructuredContactOffbandPolicy::Drop);
        require_single_program(plan, oracle);
    }

    for(const auto policy : {StructuredContactOffbandPolicy::Drop,
                            StructuredContactOffbandPolicy::Diag,
                            StructuredContactOffbandPolicy::DiagLump})
    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_plan({Vector4i{0, 5, 2, 0}}, {}, policy, workspace);
        std::vector<IndexT> sorted_vertices;
        plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);
        const auto oracle = cpu_oracle_simplex_program(
            vertices_host,
            sorted_vertices,
            std::array<IndexT, 4>{0, 5, 2, 0},
            4,
            policy);
        require_single_program(plan, oracle);
    }

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_plan({Vector4i{3, 4, 3, 4}},
                               {},
                               StructuredContactOffbandPolicy::Drop,
                               workspace);
        std::vector<IndexT> sorted_vertices;
        plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);
        const auto oracle = cpu_oracle_simplex_program(
            vertices_host,
            sorted_vertices,
            std::array<IndexT, 4>{3, 4, 3, 4},
            4,
            StructuredContactOffbandPolicy::Drop);
        require_single_program(plan, oracle);
    }

    {
        muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{vertices_host};
        muda::DeviceBuffer<Vector4i> empty_pts{std::vector<Vector4i>{}};
        muda::DeviceBuffer<Vector2i> empty_phs{std::vector<Vector2i>{}};
        muda::DeviceBuffer<Vector2i> pps{
            std::vector<Vector2i>{Vector2i{5, 2}}};

        auto input = make_input(vertices,
                                empty_pts,
                                empty_phs,
                                StructuredContactOffbandPolicy::Drop);
        input.pt_source = {};
        input.ph_source = {};
        input.pp_contacts = pps.view();
        input.pp_source = SocuContactM2SourceInput{
            0,
            12,
            SocuContactModelKind::SimplexNormal};

        SocuContactAssemblyPlanM2Workspace workspace;
        SocuContactAssemblyPlan plan;
        build_socu_contact_assembly_plan_m2_active_set_temporary(
            plan,
            workspace,
            input);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        std::vector<IndexT> sorted_vertices;
        plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);
        const auto oracle = cpu_oracle_simplex_program(
            vertices_host,
            sorted_vertices,
            std::array<IndexT, 4>{5, 2, -1, -1},
            2,
            StructuredContactOffbandPolicy::Drop);
        require_single_program(plan, oracle);
    }

    {
        muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{vertices_host};
        muda::DeviceBuffer<Vector4i> empty_pts{std::vector<Vector4i>{}};
        muda::DeviceBuffer<Vector2i> phs{
            std::vector<Vector2i>{Vector2i{2, 90}}};

        auto input = make_input(vertices,
                                empty_pts,
                                phs,
                                StructuredContactOffbandPolicy::Drop);
        input.pt_source = {};
        input.ph_source = SocuContactM2SourceInput{
            0,
            13,
            SocuContactModelKind::VertexHalfPlaneNormal};

        SocuContactAssemblyPlanM2Workspace workspace;
        SocuContactAssemblyPlan plan;
        build_socu_contact_assembly_plan_m2_active_set_temporary(
            plan,
            workspace,
            input);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        std::vector<IndexT> sorted_vertices;
        plan.side_plan.sorted_side_vertices.copy_to(sorted_vertices);
        CHECK(std::find(sorted_vertices.begin(), sorted_vertices.end(), 90)
              == sorted_vertices.end());
        const auto oracle =
            cpu_oracle_ph_program(vertices_host, sorted_vertices, 2);
        require_single_program(plan, oracle);
    }
}

TEST_CASE("cuda_mixed_socu_contact_assembly_plan_source_scan",
          "[cuda_mixed_socu][contract][socu_approx][m2][m2b]")
{
    const auto root = std::filesystem::path{UIPC_PROJECT_DIR};
    const auto builder_path =
        root / "src/backends/cuda_mixed_socu/linear_system/"
               "socu_contact_assembly_plan.cu";
    const auto builder = read_text_file(builder_path);

    for(const char* token : {"structured_contact_assembly_sink.h",
                             "socu_native_contact_target",
                             "SocuNativeContactStencilTarget",
                             "old_to_chain",
                             "classify_dof_pair"})
    {
        CHECK(builder.find(token) == std::string::npos);
    }

    for(const char* token : {"pt_contacts.copy_to",
                             "ee_contacts.copy_to",
                             "pe_contacts.copy_to",
                             "pp_contacts.copy_to",
                             "ph_contacts.copy_to",
                             "friction_pt_contacts.copy_to",
                             "friction_ee_contacts.copy_to",
                             "friction_pe_contacts.copy_to",
                             "friction_pp_contacts.copy_to",
                             "friction_ph_contacts.copy_to",
                             "stencil4.copy_to",
                             "stencil3.copy_to",
                             "stencil2.copy_to"})
    {
        CHECK(builder.find(token) == std::string::npos);
    }

    const auto solver =
        read_text_file(root / "src/backends/cuda_mixed_socu/linear_system/"
                              "socu_approx_solver.cu");
    CHECK(solver.find("info.build_socu_contact_assembly_plan_m2(")
          != std::string::npos);
    CHECK(solver.find("native_contact_side_coverage_mode") != std::string::npos);
    CHECK(solver.find("\"demand_filled\"") != std::string::npos);
    CHECK(solver.find("apply_native_contact_plan_stats")
          != std::string::npos);

    const auto dytopo =
        read_text_file(root / "src/backends/cuda_mixed_socu/"
                              "dytopo_effect_system/"
                              "global_dytopo_effect_manager.cu");
    CHECK(dytopo.find("input.sources = span<const SocuContactM2SourceInput>")
          != std::string::npos);
    CHECK(dytopo.find("socu_native_contact_plan_unsupported_reporter")
          != std::string::npos);

    const auto defaults =
        read_text_file(root / "src/core/core/scene_default_config.cpp");
    CHECK(defaults.find("native_contact_side_coverage_mode") != std::string::npos);
    CHECK(defaults.find("std::string{\"global\"}") != std::string::npos);
}
