#include <linear_system/socu_contact_assembly_plan.h>

#include <muda/buffer/buffer_launch.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_scan.h>

#include <algorithm>
#include <stdexcept>
#include <string>
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

template <typename VectorT, int StencilSize>
__global__ void collect_stencil_vertices_kernel(muda::CBufferView<VectorT> contacts,
                                                SizeT output_offset,
                                                muda::BufferView<IndexT> refs)
{
    const SizeT ref_count = contacts.size() * StencilSize;
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= ref_count)
        return;

    const SizeT contact = i / StencilSize;
    const SizeT local = i % StencilSize;
    const IndexT vertex =
        contacts.data()[contact](static_cast<Eigen::Index>(local));
    refs.data()[output_offset + i] = vertex >= 0 ? vertex : InvalidVertex;
}

__global__ void collect_ph_active_vertices_kernel(muda::CBufferView<Vector2i> phs,
                                                  SizeT output_offset,
                                                  muda::BufferView<IndexT> refs)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= phs.size())
        return;

    const IndexT vertex = phs.data()[i](0);
    refs.data()[output_offset + i] = vertex >= 0 ? vertex : InvalidVertex;
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

template <typename VectorT, int StencilSize>
__global__ void emit_simplex_programs_kernel(
    muda::CBufferView<VectorT> contacts,
    muda::CBufferView<IndexT> sorted_side_vertices,
    muda::CBufferView<SocuAssemblySideRecord> sides,
    StructuredContactOffbandPolicy offband_policy,
    std::uint32_t first_program,
    std::uint32_t first_source_to_program,
    SocuContactSourceId source_id,
    SocuContactModelKind model,
    SocuContactFamily family,
    muda::BufferView<SocuContactProgramHeader> programs,
    muda::BufferView<SocuContactMicroTask> tasks,
    muda::BufferView<SocuContactSourceToProgram> source_to_program,
    muda::BufferView<int> task_cursor)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= contacts.size())
        return;

    SocuContactProgramHeader program;
    program.source_id = source_id;
    program.local_contact_id = static_cast<IndexT>(i);
    program.model = model;
    program.family = family;
    program.stencil_size = StencilSize;

    const auto stencil = contacts.data()[i];
    SocuAssemblySideId side_ids[4] = {SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId};
    for(int local = 0; local < StencilSize; ++local)
    {
        side_ids[local] =
            find_side_id(sorted_side_vertices, stencil(static_cast<Eigen::Index>(local)));
        program.side_ids[local] = side_ids[local];
    }

    SocuContactMicroTask local_tasks[10];
    int                  exact_task_count = 0;
    bool                 has_offband = false;
    for(int row = 0; row < StencilSize; ++row)
    {
        for(int col = row; col < StencilSize; ++col)
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
                StencilSize,
                sides,
                false,
                local_tasks);
        }
        else
        {
            program.program_kind = SocuContactProgramKind::DiagLump;
            emit_task_count = append_diag_tasks_for_stencil(
                side_ids,
                StencilSize,
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

enum class M2SourceSlot : std::uint8_t
{
    PT,
    EE,
    PE,
    PP,
    PH,
    FrictionPT,
    FrictionEE,
    FrictionPE,
    FrictionPP,
    FrictionPH,
};

struct M2SourceSpec
{
    SocuContactM2SourceInput source;
    SocuContactFamily        family = SocuContactFamily::PT;
    std::uint16_t            stencil_size = 0;
    SizeT                    contact_count = 0;
    M2SourceSlot             slot = M2SourceSlot::PT;
    SizeT                    first_program = 0;
    SizeT                    first_map = 0;
};

bool source_valid(const SocuContactM2SourceInput& source) noexcept
{
    return source.source_id != SocuInvalidContactSourceId;
}

void require_source_for_contacts(const SocuContactM2SourceInput& source,
                                 SizeT contact_count,
                                 const char* name)
{
    if(contact_count != 0 && !source_valid(source))
        throw std::invalid_argument{std::string{name}
                                    + " contacts require a valid dense source id"};
}

void push_source_if_valid(std::vector<M2SourceSpec>& specs,
                          const SocuContactM2SourceInput& source,
                          SocuContactFamily family,
                          std::uint16_t stencil_size,
                          SizeT contact_count,
                          M2SourceSlot slot)
{
    if(!source_valid(source))
        return;
    specs.push_back(M2SourceSpec{source,
                                 family,
                                 stencil_size,
                                 contact_count,
                                 slot,
                                 0,
                                 0});
}

void sort_and_validate_dense_sources(std::vector<M2SourceSpec>& specs)
{
    std::sort(specs.begin(),
              specs.end(),
              [](const M2SourceSpec& lhs, const M2SourceSpec& rhs)
              {
                  return lhs.source.source_id < rhs.source.source_id;
              });

    for(SizeT i = 0; i < static_cast<SizeT>(specs.size()); ++i)
    {
        if(specs[i].source.source_id != static_cast<SocuContactSourceId>(i))
        {
            throw std::invalid_argument{
                "M2 active_set_temporary builder requires dense source ids: "
                "source_id == sources[source_id].source_id"};
        }
    }
}

SizeT max_tasks_per_contact(const M2SourceSpec& spec) noexcept
{
    if(spec.family == SocuContactFamily::PH)
        return 1;
    const SizeT stencil_size = spec.stencil_size;
    return stencil_size * (stencil_size + 1) / 2;
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
    plan.side_plan.key = input.side_key;
    plan.program_plan.key = input.program_key;

    const SizeT pt_count = input.pt_contacts.size();
    const SizeT ee_count = input.ee_contacts.size();
    const SizeT pe_count = input.pe_contacts.size();
    const SizeT pp_count = input.pp_contacts.size();
    const SizeT ph_count = input.ph_contacts.size();

    const SizeT friction_pt_count = input.friction_pt_contacts.size();
    const SizeT friction_ee_count = input.friction_ee_contacts.size();
    const SizeT friction_pe_count = input.friction_pe_contacts.size();
    const SizeT friction_pp_count = input.friction_pp_contacts.size();
    const SizeT friction_ph_count = input.friction_ph_contacts.size();

    require_source_for_contacts(input.pt_source, pt_count, "PT");
    require_source_for_contacts(input.ee_source, ee_count, "EE");
    require_source_for_contacts(input.pe_source, pe_count, "PE");
    require_source_for_contacts(input.pp_source, pp_count, "PP");
    require_source_for_contacts(input.ph_source, ph_count, "PH");
    require_source_for_contacts(
        input.friction_pt_source,
        friction_pt_count,
        "friction PT");
    require_source_for_contacts(
        input.friction_ee_source,
        friction_ee_count,
        "friction EE");
    require_source_for_contacts(
        input.friction_pe_source,
        friction_pe_count,
        "friction PE");
    require_source_for_contacts(
        input.friction_pp_source,
        friction_pp_count,
        "friction PP");
    require_source_for_contacts(
        input.friction_ph_source,
        friction_ph_count,
        "friction PH");

    std::vector<M2SourceSpec> source_specs;
    source_specs.reserve(10);
    push_source_if_valid(source_specs,
                         input.pt_source,
                         SocuContactFamily::PT,
                         4,
                         pt_count,
                         M2SourceSlot::PT);
    push_source_if_valid(source_specs,
                         input.ee_source,
                         SocuContactFamily::EE,
                         4,
                         ee_count,
                         M2SourceSlot::EE);
    push_source_if_valid(source_specs,
                         input.pe_source,
                         SocuContactFamily::PE,
                         3,
                         pe_count,
                         M2SourceSlot::PE);
    push_source_if_valid(source_specs,
                         input.pp_source,
                         SocuContactFamily::PP,
                         2,
                         pp_count,
                         M2SourceSlot::PP);
    push_source_if_valid(source_specs,
                         input.ph_source,
                         SocuContactFamily::PH,
                         2,
                         ph_count,
                         M2SourceSlot::PH);
    push_source_if_valid(source_specs,
                         input.friction_pt_source,
                         SocuContactFamily::PT,
                         4,
                         friction_pt_count,
                         M2SourceSlot::FrictionPT);
    push_source_if_valid(source_specs,
                         input.friction_ee_source,
                         SocuContactFamily::EE,
                         4,
                         friction_ee_count,
                         M2SourceSlot::FrictionEE);
    push_source_if_valid(source_specs,
                         input.friction_pe_source,
                         SocuContactFamily::PE,
                         3,
                         friction_pe_count,
                         M2SourceSlot::FrictionPE);
    push_source_if_valid(source_specs,
                         input.friction_pp_source,
                         SocuContactFamily::PP,
                         2,
                         friction_pp_count,
                         M2SourceSlot::FrictionPP);
    push_source_if_valid(source_specs,
                         input.friction_ph_source,
                         SocuContactFamily::PH,
                         2,
                         friction_ph_count,
                         M2SourceSlot::FrictionPH);
    sort_and_validate_dense_sources(source_specs);

    SizeT ref_count = 0;
    for(const auto& spec : source_specs)
    {
        if(spec.family == SocuContactFamily::PH)
            ref_count += spec.contact_count;
        else
            ref_count += spec.contact_count * spec.stencil_size;
    }

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

        SizeT ref_offset = 0;
        auto collect_vec4 = [&](muda::CBufferView<Vector4i> contacts)
        {
            const SizeT count = contacts.size() * 4;
            const SizeT base = ref_offset;
            launch_1d(count,
                      [&](unsigned int grid, int block)
                      {
                          collect_stencil_vertices_kernel<Vector4i, 4>
                              <<<grid, block, 0, launch_stream(input.stream)>>>(
                                  contacts,
                                  base,
                                  workspace.vertex_refs.view());
                      });
            ref_offset += count;
        };
        auto collect_vec3 = [&](muda::CBufferView<Vector3i> contacts)
        {
            const SizeT count = contacts.size() * 3;
            const SizeT base = ref_offset;
            launch_1d(count,
                      [&](unsigned int grid, int block)
                      {
                          collect_stencil_vertices_kernel<Vector3i, 3>
                              <<<grid, block, 0, launch_stream(input.stream)>>>(
                                  contacts,
                                  base,
                                  workspace.vertex_refs.view());
                      });
            ref_offset += count;
        };
        auto collect_vec2 = [&](muda::CBufferView<Vector2i> contacts)
        {
            const SizeT count = contacts.size() * 2;
            const SizeT base = ref_offset;
            launch_1d(count,
                      [&](unsigned int grid, int block)
                      {
                          collect_stencil_vertices_kernel<Vector2i, 2>
                              <<<grid, block, 0, launch_stream(input.stream)>>>(
                                  contacts,
                                  base,
                                  workspace.vertex_refs.view());
                      });
            ref_offset += count;
        };
        auto collect_ph = [&](muda::CBufferView<Vector2i> contacts)
        {
            const SizeT count = contacts.size();
            const SizeT base = ref_offset;
            launch_1d(count,
                      [&](unsigned int grid, int block)
                      {
                          collect_ph_active_vertices_kernel<<<grid,
                                                              block,
                                                              0,
                                                              launch_stream(input.stream)>>>(
                              contacts,
                              base,
                              workspace.vertex_refs.view());
                      });
            ref_offset += count;
        };

        for(const auto& spec : source_specs)
        {
            switch(spec.slot)
            {
                case M2SourceSlot::PT:
                    collect_vec4(input.pt_contacts);
                    break;
                case M2SourceSlot::EE:
                    collect_vec4(input.ee_contacts);
                    break;
                case M2SourceSlot::PE:
                    collect_vec3(input.pe_contacts);
                    break;
                case M2SourceSlot::PP:
                    collect_vec2(input.pp_contacts);
                    break;
                case M2SourceSlot::PH:
                    collect_ph(input.ph_contacts);
                    break;
                case M2SourceSlot::FrictionPT:
                    collect_vec4(input.friction_pt_contacts);
                    break;
                case M2SourceSlot::FrictionEE:
                    collect_vec4(input.friction_ee_contacts);
                    break;
                case M2SourceSlot::FrictionPE:
                    collect_vec3(input.friction_pe_contacts);
                    break;
                case M2SourceSlot::FrictionPP:
                    collect_vec2(input.friction_pp_contacts);
                    break;
                case M2SourceSlot::FrictionPH:
                    collect_ph(input.friction_ph_contacts);
                    break;
            }
        }

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

    SizeT total_program_count = 0;
    SizeT max_task_count = 0;
    std::vector<SocuContactSourceHeader> source_headers;
    source_headers.reserve(source_specs.size());
    for(auto& spec : source_specs)
    {
        spec.first_program = total_program_count;
        spec.first_map = total_program_count;
        source_headers.push_back(make_source_header(spec.source.source_id,
                                                    spec.source.reporter_id,
                                                    spec.source.model,
                                                    spec.family,
                                                    spec.stencil_size,
                                                    spec.contact_count,
                                                    spec.first_program,
                                                    spec.first_map));
        total_program_count += spec.contact_count;
        max_task_count += spec.contact_count * max_tasks_per_contact(spec);
    }
    plan.program_plan.sources = std::move(source_headers);
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

    auto launch_simplex4 = [&](muda::CBufferView<Vector4i> contacts,
                               const M2SourceSpec& spec)
    {
        launch_1d(spec.contact_count,
                  [&](unsigned int grid, int block)
                  {
                      emit_simplex_programs_kernel<Vector4i, 4>
                          <<<grid, block, 0, launch_stream(input.stream)>>>(
                              contacts,
                              plan.side_plan.sorted_side_vertices.view(),
                              plan.side_plan.sides.view(),
                              input.offband_policy,
                              static_cast<std::uint32_t>(spec.first_program),
                              static_cast<std::uint32_t>(spec.first_map),
                              spec.source.source_id,
                              spec.source.model,
                              spec.family,
                              plan.program_plan.programs.view(),
                              plan.program_plan.tasks.view(),
                              plan.program_plan.source_to_program.view(),
                              workspace.task_cursor.view());
                  });
    };
    auto launch_simplex3 = [&](muda::CBufferView<Vector3i> contacts,
                               const M2SourceSpec& spec)
    {
        launch_1d(spec.contact_count,
                  [&](unsigned int grid, int block)
                  {
                      emit_simplex_programs_kernel<Vector3i, 3>
                          <<<grid, block, 0, launch_stream(input.stream)>>>(
                              contacts,
                              plan.side_plan.sorted_side_vertices.view(),
                              plan.side_plan.sides.view(),
                              input.offband_policy,
                              static_cast<std::uint32_t>(spec.first_program),
                              static_cast<std::uint32_t>(spec.first_map),
                              spec.source.source_id,
                              spec.source.model,
                              spec.family,
                              plan.program_plan.programs.view(),
                              plan.program_plan.tasks.view(),
                              plan.program_plan.source_to_program.view(),
                              workspace.task_cursor.view());
                  });
    };
    auto launch_simplex2 = [&](muda::CBufferView<Vector2i> contacts,
                               const M2SourceSpec& spec)
    {
        launch_1d(spec.contact_count,
                  [&](unsigned int grid, int block)
                  {
                      emit_simplex_programs_kernel<Vector2i, 2>
                          <<<grid, block, 0, launch_stream(input.stream)>>>(
                              contacts,
                              plan.side_plan.sorted_side_vertices.view(),
                              plan.side_plan.sides.view(),
                              input.offband_policy,
                              static_cast<std::uint32_t>(spec.first_program),
                              static_cast<std::uint32_t>(spec.first_map),
                              spec.source.source_id,
                              spec.source.model,
                              spec.family,
                              plan.program_plan.programs.view(),
                              plan.program_plan.tasks.view(),
                              plan.program_plan.source_to_program.view(),
                              workspace.task_cursor.view());
                  });
    };
    auto launch_ph = [&](muda::CBufferView<Vector2i> contacts,
                         const M2SourceSpec& spec)
    {
        launch_1d(spec.contact_count,
                  [&](unsigned int grid, int block)
                  {
                      emit_ph_programs_kernel<<<grid,
                                                block,
                                                0,
                                                launch_stream(input.stream)>>>(
                          contacts,
                          plan.side_plan.sorted_side_vertices.view(),
                          plan.side_plan.sides.view(),
                          static_cast<std::uint32_t>(spec.first_program),
                          static_cast<std::uint32_t>(spec.first_map),
                          spec.source.source_id,
                          spec.source.model,
                          plan.program_plan.programs.view(),
                          plan.program_plan.tasks.view(),
                          plan.program_plan.source_to_program.view(),
                          workspace.task_cursor.view());
                  });
    };

    for(const auto& spec : source_specs)
    {
        switch(spec.slot)
        {
            case M2SourceSlot::PT:
                launch_simplex4(input.pt_contacts, spec);
                break;
            case M2SourceSlot::EE:
                launch_simplex4(input.ee_contacts, spec);
                break;
            case M2SourceSlot::PE:
                launch_simplex3(input.pe_contacts, spec);
                break;
            case M2SourceSlot::PP:
                launch_simplex2(input.pp_contacts, spec);
                break;
            case M2SourceSlot::PH:
                launch_ph(input.ph_contacts, spec);
                break;
            case M2SourceSlot::FrictionPT:
                launch_simplex4(input.friction_pt_contacts, spec);
                break;
            case M2SourceSlot::FrictionEE:
                launch_simplex4(input.friction_ee_contacts, spec);
                break;
            case M2SourceSlot::FrictionPE:
                launch_simplex3(input.friction_pe_contacts, spec);
                break;
            case M2SourceSlot::FrictionPP:
                launch_simplex2(input.friction_pp_contacts, spec);
                break;
            case M2SourceSlot::FrictionPH:
                launch_ph(input.friction_ph_contacts, spec);
                break;
        }
    }

    const int task_count = copy_first_int(workspace.task_cursor);
    plan.program_plan.tasks.resize(static_cast<SizeT>(task_count));
    plan.program_plan.last_stats.program_count = total_program_count;
    plan.program_plan.last_stats.task_count = static_cast<SizeT>(task_count);
    plan.program_plan.last_stats.bucket_count = 0;
}
}  // namespace uipc::backend::cuda_mixed
