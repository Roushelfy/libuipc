#include <app/app.h>
#include <linear_system/socu_contact_assembly_plan.h>

#include <cuda_runtime.h>
#include <muda/buffer/device_buffer.h>

#include <algorithm>
#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;
using uipc::IndexT;
using uipc::SizeT;
using uipc::Vector2i;

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

SocuNativeVertexDescriptor make_vertex(IndexT old_dof,
                                       SizeT block,
                                       SizeT lane)
{
    SocuNativeVertexDescriptor out;
    out.kind = SocuNativeDescriptorKind::Fem;
    out.fixed = false;
    out.old_dof = old_dof;
    out.dof_count = 3;
    out.block = block;
    out.lane = lane;
    out.epoch = 47;
    out.active = true;
    return out;
}

std::vector<SocuNativeVertexDescriptor> hot_block_vertices()
{
    std::vector<SocuNativeVertexDescriptor> vertices(4);
    vertices[0] = make_vertex(0, 0, 0);
    vertices[1] = make_vertex(3, 0, 3);
    vertices[2] = make_vertex(16, 1, 0);
    vertices[3] = make_vertex(19, 1, 3);
    return vertices;
}

SocuContactAssemblyPlan build_hot_block_plan(
    const std::vector<Vector2i>& contacts,
    SizeT threshold,
    SocuContactAssemblyPlanM2Workspace& workspace)
{
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{hot_block_vertices()};
    muda::DeviceBuffer<Vector2i> pp_contacts{contacts};

    SocuVertexSidePlanKey side_key;
    side_key.ordering_epoch = 3;
    side_key.native_descriptor_epoch = 47;
    side_key.horizon = 4;
    side_key.block_size = 16;

    SocuContactProgramPlanKey program_key;
    program_key.side_key = side_key;
    program_key.contact_topology_epoch = 5;
    program_key.contact_layout_hash = 7;
    program_key.contact_content_hash = 11;
    program_key.offband_policy = StructuredContactOffbandPolicy::Drop;

    SocuContactAssemblyPlanM2BuildInput input;
    input.side_key = side_key;
    input.program_key = program_key;
    input.vertex_descriptors = vertices.view();
    input.pp_contacts = pp_contacts.view();
    input.pp_source =
        SocuContactM2SourceInput{0, 10, SocuContactModelKind::SimplexNormal};
    input.offband_policy = StructuredContactOffbandPolicy::Drop;
    input.side_coverage_mode = SocuVertexSideCoverageMode::ActiveSetTemporary;
    input.build_hot_block_plan = true;
    input.hot_block_threshold = threshold;

    SocuContactAssemblyPlan plan;
    build_socu_contact_assembly_plan_m2_active_set_temporary(
        plan,
        workspace,
        input);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    return plan;
}

const SocuHotBlockRange* find_range(
    const std::vector<SocuHotBlockRange>& ranges,
    SocuAssemblyBand band,
    std::uint32_t block)
{
    const auto it = std::find_if(
        ranges.begin(),
        ranges.end(),
        [band, block](const SocuHotBlockRange& range)
        {
            return range.band == band && range.block_or_left_block == block;
        });
    return it == ranges.end() ? nullptr : &*it;
}

void require_range_refs_match_tasks(
    const SocuHotBlockRange& range,
    const std::vector<SocuHotBlockRef>& refs,
    const std::vector<SocuContactMicroTask>& tasks)
{
    REQUIRE(static_cast<SizeT>(range.first_ref) + range.ref_count <= refs.size());
    for(SizeT i = 0; i < range.ref_count; ++i)
    {
        const auto task_id =
            refs[static_cast<SizeT>(range.first_ref) + i].task_id;
        REQUIRE(static_cast<SizeT>(task_id) < tasks.size());
        const auto& task = tasks[static_cast<SizeT>(task_id)];
        CHECK(task.band == range.band);
        CHECK(task.block_or_left_block == range.block_or_left_block);
    }
}
}  // namespace

TEST_CASE("cuda_mixed_socu_contact_hot_blocks_detects_repeated_diag_blocks",
          "[cuda_mixed_socu][contract][socu_approx][m6][socu_contact_hot_blocks]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact hot-block tests");

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_hot_block_plan({Vector2i{0, 1},
                                      Vector2i{0, 1},
                                      Vector2i{0, 1},
                                      Vector2i{0, 1}},
                                     4,
                                     workspace);

    std::vector<SocuHotBlockRange> ranges;
    std::vector<SocuHotBlockRef> refs;
    std::vector<SocuContactMicroTask> tasks;
    plan.program_plan.hot_blocks.ranges.copy_to(ranges);
    plan.program_plan.hot_blocks.refs.copy_to(refs);
    plan.program_plan.tasks.copy_to(tasks);

    REQUIRE(ranges.size() == 1);
    const auto* diag = find_range(ranges, SocuAssemblyBand::Diag, 0);
    REQUIRE(diag != nullptr);
    CHECK(diag->ref_count == 12);
    require_range_refs_match_tasks(*diag, refs, tasks);

    CHECK(plan.program_plan.hot_blocks.detect_only);
    CHECK(plan.program_plan.hot_blocks.threshold == 4);
    CHECK(plan.program_plan.hot_blocks.eligible_task_count == 12);
    CHECK(plan.program_plan.last_stats.hot_diag_block_count == 1);
    CHECK(plan.program_plan.last_stats.hot_offdiag_block_count == 0);
}

TEST_CASE("cuda_mixed_socu_contact_hot_blocks_detects_first_offdiag_blocks",
          "[cuda_mixed_socu][contract][socu_approx][m6][socu_contact_hot_blocks]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact hot-block tests");

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_hot_block_plan({Vector2i{0, 2},
                                      Vector2i{0, 2},
                                      Vector2i{0, 2},
                                      Vector2i{0, 2}},
                                     4,
                                     workspace);

    std::vector<SocuHotBlockRange> ranges;
    std::vector<SocuHotBlockRef> refs;
    std::vector<SocuContactMicroTask> tasks;
    plan.program_plan.hot_blocks.ranges.copy_to(ranges);
    plan.program_plan.hot_blocks.refs.copy_to(refs);
    plan.program_plan.tasks.copy_to(tasks);

    REQUIRE(ranges.size() == 3);
    const auto* diag0 = find_range(ranges, SocuAssemblyBand::Diag, 0);
    const auto* diag1 = find_range(ranges, SocuAssemblyBand::Diag, 1);
    const auto* offdiag = find_range(ranges, SocuAssemblyBand::FirstOffdiag, 0);
    REQUIRE(diag0 != nullptr);
    REQUIRE(diag1 != nullptr);
    REQUIRE(offdiag != nullptr);
    CHECK(diag0->ref_count == 4);
    CHECK(diag1->ref_count == 4);
    CHECK(offdiag->ref_count == 4);
    require_range_refs_match_tasks(*diag0, refs, tasks);
    require_range_refs_match_tasks(*diag1, refs, tasks);
    require_range_refs_match_tasks(*offdiag, refs, tasks);

    CHECK(plan.program_plan.hot_blocks.eligible_task_count == 12);
    CHECK(plan.program_plan.last_stats.hot_diag_block_count == 2);
    CHECK(plan.program_plan.last_stats.hot_offdiag_block_count == 1);
}

TEST_CASE("cuda_mixed_socu_contact_hot_blocks_threshold_extremes",
          "[cuda_mixed_socu][contract][socu_approx][m6][socu_contact_hot_blocks]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact hot-block tests");

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_hot_block_plan({Vector2i{0, 2}, Vector2i{0, 2}},
                                         0,
                                         workspace);
        std::vector<SocuHotBlockRange> ranges;
        plan.program_plan.hot_blocks.ranges.copy_to(ranges);
        CHECK(ranges.size() == 3);
        CHECK(plan.program_plan.hot_blocks.eligible_task_count == 6);
        CHECK(plan.program_plan.last_stats.hot_diag_block_count == 2);
        CHECK(plan.program_plan.last_stats.hot_offdiag_block_count == 1);
    }

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_hot_block_plan({Vector2i{0, 2}, Vector2i{0, 2}},
                                         64,
                                         workspace);
        std::vector<SocuHotBlockRange> ranges;
        plan.program_plan.hot_blocks.ranges.copy_to(ranges);
        CHECK(ranges.empty());
        CHECK(plan.program_plan.hot_blocks.eligible_task_count == 6);
        CHECK(plan.program_plan.last_stats.hot_diag_block_count == 0);
        CHECK(plan.program_plan.last_stats.hot_offdiag_block_count == 0);
    }
}
