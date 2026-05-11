#include <app/app.h>
#include <linear_system/socu_contact_assembly_plan.h>

#include <cuda_runtime.h>
#include <muda/buffer/device_buffer.h>

#include <algorithm>
#include <stdexcept>
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

    {
        auto input = make_input(vertices, pts, phs, StructuredContactOffbandPolicy::Drop);
        input.pt_source.source_id = 1;
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

    {
        auto input = make_input(vertices, pts, phs, StructuredContactOffbandPolicy::Drop);
        input.ee_contacts = ees.view();
        input.ee_source = SocuContactM2SourceInput{
            0,
            10,
            SocuContactModelKind::SimplexNormal};
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
