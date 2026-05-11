#include <linear_system/socu_contact_assembly_plan.h>

#include <muda/buffer/buffer_launch.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_scan.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uipc::backend::cuda_mixed
{
namespace
{
constexpr IndexT InvalidVertex = std::numeric_limits<IndexT>::max();

cudaStream_t launch_stream(cudaStream_t stream) noexcept
{
    return stream == cudaStreamLegacy ? nullptr : stream;
}

MUDA_GENERIC SocuAssemblySideKind side_kind_from_native(
    SocuNativeDescriptorKind kind) noexcept
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

MUDA_DEVICE SocuAssemblySideId find_side_id(
    muda::CBufferView<IndexT> sorted_side_vertices,
    IndexT                    vertex) noexcept
{
    SizeT first = 0;
    SizeT count = sorted_side_vertices.size();
    while(count > 0)
    {
        const SizeT step = count / 2;
        const SizeT it   = first + step;
        if(sorted_side_vertices.data()[it] < vertex)
        {
            first = it + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    if(first >= sorted_side_vertices.size()
       || sorted_side_vertices.data()[first] != vertex)
        return SocuInvalidAssemblySideId;
    return static_cast<SocuAssemblySideId>(first);
}

MUDA_DEVICE bool side_id_valid(SocuAssemblySideId side_id,
                               muda::CBufferView<SocuAssemblySideRecord> sides) noexcept
{
    return side_id != SocuInvalidAssemblySideId
           && static_cast<SizeT>(side_id) < sides.size();
}

MUDA_DEVICE bool side_pair_in_band(const SocuAssemblySideRecord& row,
                                   const SocuAssemblySideRecord& col,
                                   SocuAssemblyBand&             band,
                                   std::uint32_t&                block_or_left_block,
                                   std::uint8_t&                 flags) noexcept
{
    flags = 0;
    if(row.block == col.block)
    {
        band = SocuAssemblyBand::Diag;
        block_or_left_block = static_cast<std::uint32_t>(row.block);
        if(row.global_vertex != col.global_vertex)
            flags |= static_cast<std::uint8_t>(SocuContactTaskFlag::MirrorDiagBlock);
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

MUDA_DEVICE SocuAssemblyWriteKind exact_write_kind(
    const SocuAssemblySideRecord& row,
    const SocuAssemblySideRecord& col,
    std::uint8_t&                 flags) noexcept
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
            flags |= static_cast<std::uint8_t>(SocuContactTaskFlag::SameAbdBody);
            return SocuAssemblyWriteKind::ExactAbdAbdSameBody;
        }
        return SocuAssemblyWriteKind::ExactAbdAbdCrossBody;
    }
    return SocuAssemblyWriteKind::Skipped;
}

MUDA_DEVICE SocuAssemblyWriteKind diag_block_write_kind(
    const SocuAssemblySideRecord& side) noexcept
{
    return side.kind == SocuAssemblySideKind::Abd
               ? SocuAssemblyWriteKind::DiagBlockAbd
               : SocuAssemblyWriteKind::DiagBlockFem;
}

MUDA_DEVICE SocuAssemblyWriteKind lump_write_kind(
    const SocuAssemblySideRecord& side) noexcept
{
    return side.kind == SocuAssemblySideKind::Abd
               ? SocuAssemblyWriteKind::LumpScalarAbd
               : SocuAssemblyWriteKind::LumpScalarFem;
}

MUDA_DEVICE SocuContactMicroTask make_task(SocuAssemblySideId row_side,
                                           SocuAssemblySideId col_side,
                                           std::uint8_t       local_row,
                                           std::uint8_t       local_col,
                                           SocuAssemblyBand   band,
                                           SocuAssemblyWriteKind write_kind,
                                           std::uint32_t      block_or_left_block,
                                           std::uint8_t       flags) noexcept
{
    SocuContactMicroTask task;
    task.row_side = row_side;
    task.col_side = col_side;
    task.local_row_vertex = local_row;
    task.local_col_vertex = local_col;
    task.band = band;
    task.write_kind = write_kind;
    task.block_or_left_block = block_or_left_block;
    task.flags = flags;
    return task;
}

__global__ void collect_active_vertices_kernel(muda::CBufferView<Vector4i> pts,
                                               muda::CBufferView<Vector2i> phs,
                                               muda::BufferView<IndexT> refs)
{
    const SizeT pt_refs = pts.size() * 4;
    const SizeT ref_count = pt_refs + phs.size();
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= ref_count)
        return;

    IndexT vertex = InvalidVertex;
    if(i < pt_refs)
    {
        const SizeT contact = i / 4;
        const SizeT local = i % 4;
        vertex = pts.data()[contact](static_cast<Eigen::Index>(local));
    }
    else
    {
        vertex = phs.data()[i - pt_refs](0);
    }
    refs.data()[i] = vertex >= 0 ? vertex : InvalidVertex;
}

__global__ void mark_unique_vertices_kernel(muda::CBufferView<IndexT> sorted,
                                            muda::BufferView<int> flags)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= sorted.size())
        return;
    const IndexT vertex = sorted.data()[i];
    flags.data()[i] =
        vertex != InvalidVertex && (i == 0 || sorted.data()[i - 1] != vertex)
            ? 1
            : 0;
}

__global__ void compact_unique_vertices_kernel(muda::CBufferView<IndexT> sorted,
                                               muda::CBufferView<int> flags,
                                               muda::CBufferView<int> offsets,
                                               muda::BufferView<IndexT> unique_vertices,
                                               muda::BufferView<int> total)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= sorted.size())
        return;
    if(flags.data()[i])
        unique_vertices.data()[static_cast<SizeT>(offsets.data()[i])] =
            sorted.data()[i];
    if(i + 1 == sorted.size())
        total.data()[0] = offsets.data()[i] + flags.data()[i];
}

__global__ void compute_side_lane_counts_kernel(
    muda::CBufferView<IndexT> sorted_vertices,
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
    muda::BufferView<int> lane_counts)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= sorted_vertices.size())
        return;

    const IndexT vertex = sorted_vertices.data()[i];
    int count = 0;
    if(vertex >= 0 && static_cast<SizeT>(vertex) < vertex_descriptors.size())
    {
        const auto descriptor = vertex_descriptors.data()[vertex];
        if(descriptor.mapped())
            count = static_cast<int>(descriptor.dof_count);
    }
    lane_counts.data()[i] = count;
}

__global__ void write_last_scan_total_kernel(muda::CBufferView<int> counts,
                                             muda::CBufferView<int> offsets,
                                             muda::BufferView<int> total)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if(counts.size() == 0)
    {
        total.data()[0] = 0;
        return;
    }
    const SizeT last = counts.size() - 1;
    total.data()[0] = offsets.data()[last] + counts.data()[last];
}

__global__ void materialize_sides_kernel(
    muda::CBufferView<IndexT> sorted_vertices,
    muda::CBufferView<int> lane_offsets,
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
    muda::BufferView<SocuAssemblySideRecord> sides,
    muda::BufferView<SocuAssemblyDofLane> lanes)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= sorted_vertices.size())
        return;

    const IndexT vertex = sorted_vertices.data()[i];
    SocuAssemblySideRecord side;
    side.global_vertex = vertex;
    if(vertex >= 0 && static_cast<SizeT>(vertex) < vertex_descriptors.size())
    {
        const auto descriptor = vertex_descriptors.data()[vertex];
        side.kind = side_kind_from_native(descriptor.kind);
        side.fixed = descriptor.fixed;
        side.writable = descriptor.writable();
        side.old_dof = descriptor.old_dof;
        side.dof_count = descriptor.dof_count;
        side.abd_body = descriptor.abd_body;
        side.abd_jacobian_index = descriptor.abd_j_index;
        side.block = static_cast<std::uint32_t>(descriptor.block);
        side.lane = static_cast<std::uint16_t>(descriptor.lane);
        side.first_lane = static_cast<std::uint32_t>(lane_offsets.data()[i]);
        side.lane_count = descriptor.mapped()
                              ? static_cast<std::uint16_t>(descriptor.dof_count)
                              : std::uint16_t{0};

        for(IndexT lane = 0; lane < descriptor.dof_count; ++lane)
        {
            const SizeT out = static_cast<SizeT>(side.first_lane + lane);
            if(out >= lanes.size())
                continue;
            SocuAssemblyDofLane dof_lane;
            dof_lane.block = static_cast<std::uint32_t>(descriptor.block);
            dof_lane.lane = static_cast<std::uint16_t>(descriptor.lane
                                                       + static_cast<SizeT>(lane));
            dof_lane.component = static_cast<std::uint8_t>(lane);
            dof_lane.weight = Float{1};
            lanes.data()[out] = dof_lane;
        }
    }
    sides.data()[i] = side;
}

MUDA_DEVICE int append_diag_tasks_for_stencil(
    const SocuAssemblySideId* side_ids,
    int                       stencil_size,
    muda::CBufferView<SocuAssemblySideRecord> sides,
    bool                      lump,
    SocuContactMicroTask* local_tasks) noexcept
{
    int task_count = 0;
    for(int local = 0; local < stencil_size; ++local)
    {
        const auto side_id = side_ids[local];
        if(!side_id_valid(side_id, sides))
            continue;
        const auto side = sides.data()[side_id];
        if(!side.writable)
            continue;
        local_tasks[task_count++] = make_task(side_id,
                                              side_id,
                                              static_cast<std::uint8_t>(local),
                                              static_cast<std::uint8_t>(local),
                                              SocuAssemblyBand::Diag,
                                              lump ? lump_write_kind(side)
                                                   : diag_block_write_kind(side),
                                              static_cast<std::uint32_t>(side.block),
                                              0);
    }
    return task_count;
}

__global__ void emit_pt_programs_kernel(
    muda::CBufferView<Vector4i> pts,
    muda::CBufferView<IndexT> sorted_side_vertices,
    muda::CBufferView<SocuAssemblySideRecord> sides,
    StructuredContactOffbandPolicy offband_policy,
    std::uint32_t first_program,
    std::uint32_t first_source_to_program,
    SocuContactSourceId source_id,
    SocuContactModelKind model,
    muda::BufferView<SocuContactProgramHeader> programs,
    muda::BufferView<SocuContactMicroTask> tasks,
    muda::BufferView<SocuContactSourceToProgram> source_to_program,
    muda::BufferView<int> task_cursor)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= pts.size())
        return;

    SocuContactProgramHeader program;
    program.source_id = source_id;
    program.local_contact_id = static_cast<IndexT>(i);
    program.model = model;
    program.family = SocuContactFamily::PT;
    program.stencil_size = 4;

    const auto stencil = pts.data()[i];
    SocuAssemblySideId side_ids[4] = {SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId};
    for(int local = 0; local < 4; ++local)
    {
        side_ids[local] =
            find_side_id(sorted_side_vertices, stencil(static_cast<Eigen::Index>(local)));
        program.side_ids[local] = side_ids[local];
    }

    SocuContactMicroTask local_tasks[10];
    int                  exact_task_count = 0;
    bool                 has_offband = false;
    for(int row = 0; row < 4; ++row)
    {
        for(int col = row; col < 4; ++col)
        {
            if(!side_id_valid(side_ids[row], sides)
               || !side_id_valid(side_ids[col], sides))
                continue;
            const auto row_side = sides.data()[side_ids[row]];
            const auto col_side = sides.data()[side_ids[col]];
            if(!row_side.writable || !col_side.writable)
                continue;

            SocuAssemblyBand band = SocuAssemblyBand::Diag;
            std::uint32_t    block_or_left_block = 0;
            std::uint8_t     flags = 0;
            if(!side_pair_in_band(row_side,
                                  col_side,
                                  band,
                                  block_or_left_block,
                                  flags))
            {
                has_offband = true;
                continue;
            }

            const auto kind = exact_write_kind(row_side, col_side, flags);
            if(kind == SocuAssemblyWriteKind::Skipped)
                continue;
            local_tasks[exact_task_count++] =
                make_task(side_ids[row],
                          side_ids[col],
                          static_cast<std::uint8_t>(row),
                          static_cast<std::uint8_t>(col),
                          band,
                          kind,
                          block_or_left_block,
                          flags);
        }
    }

    SocuContactProgramMapStatus status = SocuContactProgramMapStatus::Valid;
    int emit_task_count = exact_task_count;
    if(has_offband)
    {
        if(offband_policy == StructuredContactOffbandPolicy::Drop)
        {
            program.program_kind = SocuContactProgramKind::Drop;
            status = SocuContactProgramMapStatus::Dropped;
            emit_task_count = 0;
        }
        else if(offband_policy == StructuredContactOffbandPolicy::Diag)
        {
            program.program_kind = SocuContactProgramKind::Diag;
            emit_task_count = append_diag_tasks_for_stencil(
                side_ids,
                4,
                sides,
                false,
                local_tasks);
        }
        else
        {
            program.program_kind = SocuContactProgramKind::DiagLump;
            emit_task_count = append_diag_tasks_for_stencil(
                side_ids,
                4,
                sides,
                true,
                local_tasks);
        }
    }
    else if(exact_task_count > 0)
    {
        program.program_kind = SocuContactProgramKind::Exact;
    }
    else
    {
        program.program_kind = SocuContactProgramKind::Skipped;
        status = SocuContactProgramMapStatus::Skipped;
    }

    if(emit_task_count > 0)
    {
        const int first_task = atomicAdd(task_cursor.data(), emit_task_count);
        program.first_task = static_cast<std::uint32_t>(first_task);
        program.task_count = static_cast<std::uint16_t>(emit_task_count);
        for(int task = 0; task < emit_task_count; ++task)
            tasks.data()[first_task + task] = local_tasks[task];
    }

    const SizeT program_id = first_program + i;
    programs.data()[program_id] = program;
    source_to_program.data()[first_source_to_program + i] =
        SocuContactSourceToProgram{static_cast<SocuContactProgramId>(program_id),
                                   status,
                                   0};
}

__global__ void emit_ph_programs_kernel(
    muda::CBufferView<Vector2i> phs,
    muda::CBufferView<IndexT> sorted_side_vertices,
    muda::CBufferView<SocuAssemblySideRecord> sides,
    std::uint32_t first_program,
    std::uint32_t first_source_to_program,
    SocuContactSourceId source_id,
    SocuContactModelKind model,
    muda::BufferView<SocuContactProgramHeader> programs,
    muda::BufferView<SocuContactMicroTask> tasks,
    muda::BufferView<SocuContactSourceToProgram> source_to_program,
    muda::BufferView<int> task_cursor)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= phs.size())
        return;

    SocuContactProgramHeader program;
    program.source_id = source_id;
    program.local_contact_id = static_cast<IndexT>(i);
    program.model = model;
    program.family = SocuContactFamily::PH;
    program.stencil_size = 2;

    const auto vertex = phs.data()[i](0);
    const auto side_id = find_side_id(sorted_side_vertices, vertex);
    program.side_ids[0] = side_id;

    SocuContactProgramMapStatus status = SocuContactProgramMapStatus::Skipped;
    if(side_id_valid(side_id, sides) && sides.data()[side_id].writable)
    {
        const auto side = sides.data()[side_id];
        const int first_task = atomicAdd(task_cursor.data(), 1);
        std::uint8_t flags = 0;
        program.first_task = static_cast<std::uint32_t>(first_task);
        program.task_count = 1;
        program.program_kind = SocuContactProgramKind::Exact;
        tasks.data()[first_task] = make_task(side_id,
                                             side_id,
                                             0,
                                             0,
                                             SocuAssemblyBand::Diag,
                                             exact_write_kind(side, side, flags),
                                             static_cast<std::uint32_t>(side.block),
                                             flags);
        status = SocuContactProgramMapStatus::Valid;
    }
    else
    {
        program.program_kind = SocuContactProgramKind::Skipped;
    }

    const SizeT program_id = first_program + i;
    programs.data()[program_id] = program;
    source_to_program.data()[first_source_to_program + i] =
        SocuContactSourceToProgram{static_cast<SocuContactProgramId>(program_id),
                                   status,
                                   0};
}

void launch_1d(SizeT count, auto&& launcher)
{
    if(count == 0)
        return;
    constexpr int block_dim = 256;
    const auto grid_dim =
        static_cast<unsigned int>((count + block_dim - 1) / block_dim);
    launcher(grid_dim, block_dim);
}

int copy_first_int(muda::DeviceBuffer<int>& buffer)
{
    int out = 0;
    buffer.view(0, 1).copy_to(&out);
    return out;
}

void resize_zero_side_plan(SocuVertexSidePlan& plan)
{
    plan.sorted_side_vertices.resize(0);
    plan.sides.resize(0);
    plan.lanes.resize(0);
    plan.coverage.mode = SocuVertexSideCoverageMode::ActiveSetTemporary;
    plan.coverage.covered_vertex_count = 0;
    plan.coverage.complete_for_current_contacts = true;
    plan.last_stats.active_side_vertex_count = 0;
    plan.last_stats.side_count = 0;
    plan.last_stats.lane_count = 0;
}

SocuContactSourceHeader make_source_header(SocuContactSourceId source_id,
                                           std::uint32_t reporter_id,
                                           SocuContactModelKind model,
                                           SocuContactFamily family,
                                           std::uint16_t stencil_size,
                                           SizeT contact_count,
                                           SizeT first_program,
                                           SizeT first_map)
{
    SocuContactSourceHeader source;
    source.source_id = source_id;
    source.reporter_id = reporter_id;
    source.model = model;
    source.family = family;
    source.stencil_size = stencil_size;
    source.contact_count = static_cast<std::uint32_t>(contact_count);
    source.first_program = static_cast<std::uint32_t>(first_program);
    source.program_count = static_cast<std::uint32_t>(contact_count);
    source.first_source_to_program = static_cast<std::uint32_t>(first_map);
    return source;
}
}  // namespace

SocuContactAssemblyPlanView socu_contact_assembly_plan_view(
    const SocuContactAssemblyPlan& plan) noexcept
{
    return SocuContactAssemblyPlanView{plan.side_plan.key,
                                       plan.program_plan.key,
                                       plan.program_plan.sources.view(),
                                       plan.side_plan.sides.view(),
                                       plan.side_plan.lanes.view(),
                                       plan.side_plan.sorted_side_vertices.view(),
                                       plan.program_plan.programs.view(),
                                       plan.program_plan.tasks.view(),
                                       plan.program_plan.buckets.view(),
                                       plan.program_plan.source_to_program.view()};
}

void build_socu_contact_assembly_plan_m2_active_set_temporary(
    SocuContactAssemblyPlan&                   plan,
    SocuContactAssemblyPlanM2Workspace&        workspace,
    const SocuContactAssemblyPlanM2BuildInput& input)
{
    if(input.pt_source.source_id != 0 || input.ph_source.source_id != 1)
    {
        throw std::invalid_argument{
            "M2 active_set_temporary builder currently requires dense PT source 0 and PH source 1"};
    }

    plan.side_plan.key = input.side_key;
    plan.program_plan.key = input.program_key;

    const SizeT pt_count = input.pt_contacts.size();
    const SizeT ph_count = input.ph_contacts.size();
    const SizeT ref_count = pt_count * 4 + ph_count;

    workspace.scalar_total.resize(1);
    workspace.task_cursor.resize(1);

    if(ref_count == 0)
    {
        resize_zero_side_plan(plan.side_plan);
    }
    else
    {
        workspace.vertex_refs.resize(ref_count);
        workspace.sorted_vertex_refs.resize(ref_count);
        workspace.unique_flags.resize(ref_count);
        workspace.unique_offsets.resize(ref_count);
        plan.side_plan.sorted_side_vertices.resize(ref_count);

        launch_1d(ref_count,
                  [&](unsigned int grid, int block)
                  {
                      collect_active_vertices_kernel<<<grid,
                                                       block,
                                                       0,
                                                       launch_stream(input.stream)>>>(
                          input.pt_contacts,
                          input.ph_contacts,
                          workspace.vertex_refs.view());
                  });

        muda::DeviceRadixSort().SortKeys(workspace.vertex_refs.data(),
                                         workspace.sorted_vertex_refs.data(),
                                         static_cast<int>(ref_count));

        launch_1d(ref_count,
                  [&](unsigned int grid, int block)
                  {
                      mark_unique_vertices_kernel<<<grid,
                                                    block,
                                                    0,
                                                    launch_stream(input.stream)>>>(
                          workspace.sorted_vertex_refs.view(),
                          workspace.unique_flags.view());
                  });
        muda::DeviceScan().ExclusiveSum(workspace.unique_flags.data(),
                                        workspace.unique_offsets.data(),
                                        static_cast<int>(ref_count));
        muda::BufferLaunch(input.stream).fill<int>(workspace.scalar_total.view(), 0);
        launch_1d(ref_count,
                  [&](unsigned int grid, int block)
                  {
                      compact_unique_vertices_kernel<<<grid,
                                                       block,
                                                       0,
                                                       launch_stream(input.stream)>>>(
                          workspace.sorted_vertex_refs.view(),
                          workspace.unique_flags.view(),
                          workspace.unique_offsets.view(),
                          plan.side_plan.sorted_side_vertices.view(),
                          workspace.scalar_total.view());
                  });

        const int side_count = copy_first_int(workspace.scalar_total);
        plan.side_plan.sorted_side_vertices.resize(static_cast<SizeT>(side_count));
        plan.side_plan.sides.resize(static_cast<SizeT>(side_count));
        workspace.scalar_counts.resize(static_cast<SizeT>(side_count));
        workspace.scalar_offsets.resize(static_cast<SizeT>(side_count));

        if(side_count == 0)
        {
            plan.side_plan.lanes.resize(0);
        }
        else
        {
            launch_1d(static_cast<SizeT>(side_count),
                      [&](unsigned int grid, int block)
                      {
                          compute_side_lane_counts_kernel<<<grid,
                                                            block,
                                                            0,
                                                            launch_stream(input.stream)>>>(
                              plan.side_plan.sorted_side_vertices.view(),
                              input.vertex_descriptors,
                              workspace.scalar_counts.view());
                      });
            muda::DeviceScan().ExclusiveSum(workspace.scalar_counts.data(),
                                            workspace.scalar_offsets.data(),
                                            side_count);
            write_last_scan_total_kernel<<<1, 1, 0, launch_stream(input.stream)>>>(
                workspace.scalar_counts.view(),
                workspace.scalar_offsets.view(),
                workspace.scalar_total.view());
            const int lane_count = copy_first_int(workspace.scalar_total);
            plan.side_plan.lanes.resize(static_cast<SizeT>(lane_count));
            if(lane_count > 0)
                muda::BufferLaunch(input.stream)
                    .fill<SocuAssemblyDofLane>(plan.side_plan.lanes.view(), {});

            launch_1d(static_cast<SizeT>(side_count),
                      [&](unsigned int grid, int block)
                      {
                          materialize_sides_kernel<<<grid,
                                                     block,
                                                     0,
                                                     launch_stream(input.stream)>>>(
                              plan.side_plan.sorted_side_vertices.view(),
                              workspace.scalar_offsets.view(),
                              input.vertex_descriptors,
                              plan.side_plan.sides.view(),
                              plan.side_plan.lanes.view());
                      });
        }

        plan.side_plan.coverage.mode =
            SocuVertexSideCoverageMode::ActiveSetTemporary;
        plan.side_plan.coverage.covered_vertex_count =
            static_cast<SizeT>(side_count);
        plan.side_plan.coverage.complete_for_current_contacts = true;
        plan.side_plan.last_stats.active_side_vertex_count =
            static_cast<SizeT>(side_count);
        plan.side_plan.last_stats.side_count = static_cast<SizeT>(side_count);
        plan.side_plan.last_stats.lane_count = plan.side_plan.lanes.size();
    }

    const SizeT total_program_count = pt_count + ph_count;
    const SizeT max_task_count = pt_count * 10 + ph_count;
    plan.program_plan.sources =
        std::vector<SocuContactSourceHeader>{
            make_source_header(input.pt_source.source_id,
                               input.pt_source.reporter_id,
                               input.pt_source.model,
                               SocuContactFamily::PT,
                               4,
                               pt_count,
                               0,
                               0),
            make_source_header(input.ph_source.source_id,
                               input.ph_source.reporter_id,
                               input.ph_source.model,
                               SocuContactFamily::PH,
                               2,
                               ph_count,
                               pt_count,
                               pt_count)};
    plan.program_plan.programs.resize(total_program_count);
    plan.program_plan.source_to_program.resize(total_program_count);
    plan.program_plan.tasks.resize(max_task_count);
    plan.program_plan.buckets.resize(0);
    if(total_program_count > 0)
    {
        muda::BufferLaunch(input.stream)
            .fill<SocuContactProgramHeader>(plan.program_plan.programs.view(), {});
        muda::BufferLaunch(input.stream)
            .fill<SocuContactSourceToProgram>(
                plan.program_plan.source_to_program.view(),
                {});
    }
    if(max_task_count > 0)
        muda::BufferLaunch(input.stream)
            .fill<SocuContactMicroTask>(plan.program_plan.tasks.view(), {});
    muda::BufferLaunch(input.stream).fill<int>(workspace.task_cursor.view(), 0);

    launch_1d(pt_count,
              [&](unsigned int grid, int block)
              {
                  emit_pt_programs_kernel<<<grid,
                                            block,
                                            0,
                                            launch_stream(input.stream)>>>(
                      input.pt_contacts,
                      plan.side_plan.sorted_side_vertices.view(),
                      plan.side_plan.sides.view(),
                      input.offband_policy,
                      0,
                      0,
                      input.pt_source.source_id,
                      input.pt_source.model,
                      plan.program_plan.programs.view(),
                      plan.program_plan.tasks.view(),
                      plan.program_plan.source_to_program.view(),
                      workspace.task_cursor.view());
              });
    launch_1d(ph_count,
              [&](unsigned int grid, int block)
              {
                  emit_ph_programs_kernel<<<grid,
                                            block,
                                            0,
                                            launch_stream(input.stream)>>>(
                      input.ph_contacts,
                      plan.side_plan.sorted_side_vertices.view(),
                      plan.side_plan.sides.view(),
                      static_cast<std::uint32_t>(pt_count),
                      static_cast<std::uint32_t>(pt_count),
                      input.ph_source.source_id,
                      input.ph_source.model,
                      plan.program_plan.programs.view(),
                      plan.program_plan.tasks.view(),
                      plan.program_plan.source_to_program.view(),
                      workspace.task_cursor.view());
              });

    const int task_count = copy_first_int(workspace.task_cursor);
    plan.program_plan.tasks.resize(static_cast<SizeT>(task_count));
    plan.program_plan.last_stats.program_count = total_program_count;
    plan.program_plan.last_stats.task_count = static_cast<SizeT>(task_count);
    plan.program_plan.last_stats.bucket_count = 0;
}
}  // namespace uipc::backend::cuda_mixed
