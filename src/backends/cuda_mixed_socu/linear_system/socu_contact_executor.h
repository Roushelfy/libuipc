#pragma once

#include <linear_system/socu_contact_program_writer.h>

#include <cuda_runtime_api.h>
#include <muda/ext/linear_system/triplet_matrix_view.h>

#include <limits>

namespace uipc::backend::cuda_mixed
{
enum class SocuContactExecutorCounterSlot : IndexT
{
    BucketVisit = 0,
    EmptyBucket,
    ProgramVisit,
    ExactProgram,
    DiagProgram,
    DiagLumpProgram,
    DropProgram,
    SkippedProgram,
    MixedRejectedProgram,
    TaskWrite,
    UnsupportedProgram,
    DirectHotTaskSkipped,
    HotBlockRangeVisit,
    HotReduceTaskVisit,
    HotMicroblockCacheWrite,
    Count,
};

template <typename StoreT>
struct SocuDeterministicContactHessian
{
    static constexpr SizeT MaxStencilSize = 4;
    static constexpr SizeT MaxDofCount    = MaxStencilSize * 3;

    StoreT values[MaxDofCount * MaxDofCount] = {};
    std::uint16_t stencil_size = 0;

    MUDA_GENERIC StoreT operator()(IndexT row, IndexT col) const noexcept
    {
        if(row < 0 || col < 0)
            return StoreT{0};
        const auto r = static_cast<SizeT>(row);
        const auto c = static_cast<SizeT>(col);
        if(r >= MaxDofCount || c >= MaxDofCount)
            return StoreT{0};
        return values[r * MaxDofCount + c];
    }
};

template <typename StoreT>
struct SocuDeterministicContactEvaluator
{
    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> operator()(
        const SocuContactProgramHeader& program) const noexcept
    {
        SocuDeterministicContactHessian<StoreT> H;
        H.stencil_size = program.stencil_size;
        const SizeT dof_count =
            static_cast<SizeT>(program.stencil_size) * SizeT{3};
        const auto source_component =
            static_cast<StoreT>(static_cast<IndexT>(program.source_id) * 1000);
        const auto contact_component =
            static_cast<StoreT>(program.local_contact_id * 100);
        const auto model_component =
            static_cast<StoreT>(static_cast<IndexT>(program.model) * 17);
        const auto family_component =
            static_cast<StoreT>(static_cast<IndexT>(program.family) * 7);
        const auto base = source_component + contact_component + model_component
                          + family_component + StoreT{1};

        for(SizeT row = 0; row < dof_count
                          && row < SocuDeterministicContactHessian<
                                       StoreT>::MaxDofCount;
            ++row)
        {
            for(SizeT col = 0; col < dof_count
                              && col < SocuDeterministicContactHessian<
                                           StoreT>::MaxDofCount;
                ++col)
            {
                StoreT value = base + static_cast<StoreT>(row * 10 + col);
                if(((row + col) & SizeT{1}) != 0)
                    value = -value;
                H.values[row * SocuDeterministicContactHessian<StoreT>::MaxDofCount
                         + col] = value;
            }
        }
        return H;
    }
};

template <typename StoreT>
struct SocuSimplexContactEvaluatorSourceView
{
    muda::CTripletMatrixView<StoreT, 3> pt_hessians;
    muda::CTripletMatrixView<StoreT, 3> ee_hessians;
    muda::CTripletMatrixView<StoreT, 3> pe_hessians;
    muda::CTripletMatrixView<StoreT, 3> pp_hessians;

    MUDA_GENERIC muda::CTripletMatrixView<StoreT, 3> hessians_for(
        SocuContactFamily family) const noexcept
    {
        switch(family)
        {
            case SocuContactFamily::PT:
                return pt_hessians;
            case SocuContactFamily::EE:
                return ee_hessians;
            case SocuContactFamily::PE:
                return pe_hessians;
            case SocuContactFamily::PP:
                return pp_hessians;
            case SocuContactFamily::PH:
            default:
                return {};
        }
    }
};

template <typename StoreT>
struct SocuVertexHalfPlaneContactEvaluatorSourceView
{
    muda::CTripletMatrixView<StoreT, 3> ph_hessians;

    MUDA_GENERIC muda::CTripletMatrixView<StoreT, 3> hessians_for(
        SocuContactFamily family) const noexcept
    {
        return family == SocuContactFamily::PH ? ph_hessians
                                               : muda::CTripletMatrixView<StoreT, 3>{};
    }
};

template <typename StoreT>
struct SocuContactEvaluatorSourceTable
{
    SocuSimplexContactEvaluatorSourceView<StoreT> simplex_normal;
    SocuSimplexContactEvaluatorSourceView<StoreT> simplex_frictional;
    SocuVertexHalfPlaneContactEvaluatorSourceView<StoreT> vertex_half_plane_normal;
    SocuVertexHalfPlaneContactEvaluatorSourceView<StoreT> vertex_half_plane_frictional;

    MUDA_GENERIC muda::CTripletMatrixView<StoreT, 3> hessians_for(
        SocuContactModelKind model,
        SocuContactFamily family) const noexcept
    {
        switch(model)
        {
            case SocuContactModelKind::SimplexNormal:
                return simplex_normal.hessians_for(family);
            case SocuContactModelKind::SimplexFrictional:
                return simplex_frictional.hessians_for(family);
            case SocuContactModelKind::VertexHalfPlaneNormal:
                return vertex_half_plane_normal.hessians_for(family);
            case SocuContactModelKind::VertexHalfPlaneFrictional:
                return vertex_half_plane_frictional.hessians_for(family);
            default:
                return {};
        }
    }
};

MUDA_GENERIC inline SizeT socu_contact_half_hessian_size(
    SocuContactFamily family) noexcept
{
    switch(family)
    {
        case SocuContactFamily::PT:
        case SocuContactFamily::EE:
            return 10;
        case SocuContactFamily::PE:
            return 6;
        case SocuContactFamily::PP:
            return 3;
        case SocuContactFamily::PH:
            return 1;
    }
    return 0;
}

template <typename StoreT>
struct SocuContactTripletEvaluator
{
    SocuContactAssemblyPlanView plan;
    SocuContactEvaluatorSourceTable<StoreT> sources;

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> operator()(
        const SocuContactProgramHeader& program) const noexcept
    {
        SocuDeterministicContactHessian<StoreT> H;
        H.stencil_size = program.stencil_size;

        const auto hessians = sources.hessians_for(program.model, program.family);
        const SizeT half_size = socu_contact_half_hessian_size(program.family);
        if(hessians.triplet_count() == 0 || half_size == 0
           || program.local_contact_id < 0)
            return H;

        const SizeT first_triplet =
            static_cast<SizeT>(program.local_contact_id) * half_size;
        if(first_triplet >= static_cast<SizeT>(hessians.triplet_count()))
            return H;

        const auto viewer = hessians.cviewer();
        for(SizeT half_index = 0; half_index < half_size; ++half_index)
        {
            const SizeT triplet_index = first_triplet + half_index;
            if(triplet_index >= static_cast<SizeT>(hessians.triplet_count()))
                break;

            const auto triplet =
                viewer(static_cast<int>(triplet_index));
            const IndexT local_row = local_vertex_for_global(program,
                                                             triplet.row_index);
            const IndexT local_col = local_vertex_for_global(program,
                                                             triplet.col_index);
            if(local_row < 0 || local_col < 0)
                continue;

            write_block(H,
                        static_cast<SizeT>(local_row),
                        static_cast<SizeT>(local_col),
                        triplet.value);
            if(local_row != local_col)
                write_transposed_block(H,
                                       static_cast<SizeT>(local_col),
                                       static_cast<SizeT>(local_row),
                                       triplet.value);
        }
        return H;
    }

    MUDA_DEVICE IndexT local_vertex_for_global(
        const SocuContactProgramHeader& program,
        IndexT global_vertex) const noexcept
    {
        for(std::uint16_t local = 0; local < program.stencil_size; ++local)
        {
            const auto side_id = program.side_ids[local];
            if(side_id == SocuInvalidAssemblySideId
               || static_cast<SizeT>(side_id) >= plan.sides.size())
                continue;
            if(plan.sides.data()[static_cast<SizeT>(side_id)].global_vertex
               == global_vertex)
                return static_cast<IndexT>(local);
        }
        return -1;
    }

    template <typename Block3>
    MUDA_DEVICE static void write_block(SocuDeterministicContactHessian<StoreT>& H,
                                        SizeT local_row,
                                        SizeT local_col,
                                        const Block3& block) noexcept
    {
        if(local_row >= H.stencil_size || local_col >= H.stencil_size)
            return;
        for(SizeT row = 0; row < 3; ++row)
        {
            for(SizeT col = 0; col < 3; ++col)
            {
                const SizeT h_row = local_row * 3 + row;
                const SizeT h_col = local_col * 3 + col;
                H.values[h_row
                         * SocuDeterministicContactHessian<StoreT>::MaxDofCount
                         + h_col] = static_cast<StoreT>(block(row, col));
            }
        }
    }

    template <typename Block3>
    MUDA_DEVICE static void write_transposed_block(
        SocuDeterministicContactHessian<StoreT>& H,
        SizeT local_row,
        SizeT local_col,
        const Block3& block) noexcept
    {
        if(local_row >= H.stencil_size || local_col >= H.stencil_size)
            return;
        for(SizeT row = 0; row < 3; ++row)
        {
            for(SizeT col = 0; col < 3; ++col)
            {
                const SizeT h_row = local_row * 3 + row;
                const SizeT h_col = local_col * 3 + col;
                H.values[h_row
                         * SocuDeterministicContactHessian<StoreT>::MaxDofCount
                         + h_col] = static_cast<StoreT>(block(col, row));
            }
        }
    }
};

template <typename StoreT, typename SolveT, typename EvaluatorT>
struct SocuContactBucketExecutor
{
    SocuContactAssemblyPlanView plan;
    SocuNativeMatrixView<SolveT> matrix;
    EvaluatorT evaluator;
    muda::BufferView<IndexT> counters;

    MUDA_GENERIC bool valid() const noexcept
    {
        return plan.valid() && matrix.valid();
    }

    MUDA_DEVICE void record(SocuContactExecutorCounterSlot slot,
                            IndexT amount = 1) const noexcept
    {
        const auto index = static_cast<IndexT>(slot);
        if(counters.data() != nullptr && index >= 0
           && static_cast<SizeT>(index) < counters.size())
            muda::atomic_add(counters.data(static_cast<SizeT>(index)), amount);
    }

    MUDA_GENERIC static bool owner_reduce_strategy(
        SocuContactExecutionStrategy strategy) noexcept
    {
        return strategy == SocuContactExecutionStrategy::Recompute
               || strategy == SocuContactExecutionStrategy::CachedMicroblock;
    }

    MUDA_GENERIC static bool has_flag(const SocuContactMicroTask& task,
                                      SocuContactTaskFlag flag) noexcept
    {
        return (task.flags & static_cast<std::uint8_t>(flag)) != 0;
    }

    MUDA_DEVICE bool skip_direct_task(
        const SocuContactMicroTask& task) const noexcept
    {
        return owner_reduce_strategy(plan.hot_block_strategy)
               && has_flag(task, SocuContactTaskFlag::HotReduceSelected);
    }

    MUDA_DEVICE void execute_program(SizeT program_id) const noexcept
    {
        if(!valid() || program_id >= plan.programs.size())
            return;

        const auto program = plan.programs.data()[program_id];
        record(SocuContactExecutorCounterSlot::ProgramVisit);

        switch(program.program_kind)
        {
            case SocuContactProgramKind::Exact:
                record(SocuContactExecutorCounterSlot::ExactProgram);
                break;
            case SocuContactProgramKind::Diag:
                record(SocuContactExecutorCounterSlot::DiagProgram);
                break;
            case SocuContactProgramKind::DiagLump:
                record(SocuContactExecutorCounterSlot::DiagLumpProgram);
                break;
            case SocuContactProgramKind::Drop:
                record(SocuContactExecutorCounterSlot::DropProgram);
                return;
            case SocuContactProgramKind::Skipped:
                record(SocuContactExecutorCounterSlot::SkippedProgram);
                return;
            case SocuContactProgramKind::MixedRejectedDebugOnly:
                record(SocuContactExecutorCounterSlot::MixedRejectedProgram);
                return;
            default:
                record(SocuContactExecutorCounterSlot::UnsupportedProgram);
                return;
        }

        if(program.task_count == 0)
            return;

        const auto H = evaluator(program);
        const SocuContactProgramWriter<StoreT, SolveT> writer{plan, matrix, {}};
        for(std::uint16_t task_index = 0; task_index < program.task_count;
            ++task_index)
        {
            const SizeT task_id = static_cast<SizeT>(program.first_task) + task_index;
            if(task_id >= plan.tasks.size())
                continue;
            const auto task = plan.tasks.data()[task_id];
            if(skip_direct_task(task))
            {
                record(SocuContactExecutorCounterSlot::DirectHotTaskSkipped);
                continue;
            }
            writer.write_task(program, task, H);
            record(SocuContactExecutorCounterSlot::TaskWrite);
        }
    }

    MUDA_DEVICE void execute_bucket(SizeT bucket_id,
                                    SizeT program_offset_stride,
                                    SizeT program_offset_begin) const noexcept
    {
        if(!valid() || bucket_id >= plan.buckets.size())
            return;

        const auto bucket = plan.buckets.data()[bucket_id];
        if(program_offset_begin == 0)
            record(SocuContactExecutorCounterSlot::BucketVisit);
        if(bucket.program_count == 0)
        {
            if(program_offset_begin == 0)
                record(SocuContactExecutorCounterSlot::EmptyBucket);
            return;
        }

        for(SizeT local_program = program_offset_begin;
            local_program < static_cast<SizeT>(bucket.program_count);
            local_program += program_offset_stride)
        {
            execute_program(static_cast<SizeT>(bucket.first_program) + local_program);
        }
    }

    MUDA_DEVICE SolveT hot_task_cell_value(const SocuContactMicroTask& task,
                                           SizeT storage_row,
                                           SizeT storage_col) const noexcept
    {
        if(task.program_id == SocuInvalidContactProgramId
           || static_cast<SizeT>(task.program_id) >= plan.programs.size())
            return SolveT{0};
        const auto program =
            plan.programs.data()[static_cast<SizeT>(task.program_id)];
        const auto H = evaluator(program);
        const SocuContactProgramWriter<StoreT, SolveT> writer{plan, matrix, {}};
        return static_cast<SolveT>(
            writer.task_storage_cell_contribution(program,
                                                  task,
                                                  H,
                                                  storage_row,
                                                  storage_col));
    }

    MUDA_DEVICE SolveT hot_ref_cell_value(const SocuHotBlockRef& ref,
                                          const SocuHotBlockRange& range,
                                          SizeT storage_row,
                                          SizeT storage_col) const noexcept
    {
        if(static_cast<SizeT>(ref.task_id) >= plan.tasks.size())
            return SolveT{0};
        const auto task = plan.tasks.data()[static_cast<SizeT>(ref.task_id)];
        if(!has_flag(task, SocuContactTaskFlag::HotReduceSelected)
           || task.band != range.band
           || task.block_or_left_block != range.block_or_left_block)
            return SolveT{0};
        return hot_task_cell_value(task, storage_row, storage_col);
    }

    MUDA_DEVICE void add_hot_reduced_cell(const SocuHotBlockRange& range,
                                          SizeT storage_row,
                                          SizeT storage_col,
                                          SolveT value) const noexcept
    {
        if(value == SolveT{0})
            return;

        if(range.band == SocuAssemblyBand::Diag)
        {
            if(!matrix.valid_block_entry(range.block_or_left_block,
                                         storage_row,
                                         storage_col))
                return;
            const SizeT index =
                matrix.diag_index(range.block_or_left_block,
                                  storage_row,
                                  storage_col);
            if(index < matrix.D.size())
                *matrix.D.data(index) += value;
            return;
        }

        if(range.block_or_left_block >= matrix.first_offdiag_block_count
           || storage_row >= matrix.block_size
           || storage_col >= matrix.block_size)
            return;
        const SizeT index =
            matrix.first_offdiag_index(range.block_or_left_block,
                                       storage_row,
                                       storage_col);
        if(index < matrix.E.size())
            *matrix.E.data(index) += value;
    }

    MUDA_DEVICE void execute_hot_range_recompute(SizeT range_id) const noexcept
    {
        if(!valid() || range_id >= plan.hot_block_ranges.size())
            return;
        const auto range = plan.hot_block_ranges.data()[range_id];
        if(range.ref_count == 0)
            return;

        if(threadIdx.x == 0)
            record(SocuContactExecutorCounterSlot::HotBlockRangeVisit);
        const SizeT cell_count = matrix.block_size * matrix.block_size;
        for(SizeT cell = static_cast<SizeT>(threadIdx.x);
            cell < cell_count;
            cell += static_cast<SizeT>(blockDim.x))
        {
            const SizeT storage_row = cell / matrix.block_size;
            const SizeT storage_col = cell % matrix.block_size;
            SolveT sum = SolveT{0};
            for(SizeT ref_offset = 0;
                ref_offset < static_cast<SizeT>(range.ref_count);
                ++ref_offset)
            {
                const SizeT ref_index =
                    static_cast<SizeT>(range.first_ref) + ref_offset;
                if(ref_index >= plan.hot_block_refs.size())
                    continue;
                sum += hot_ref_cell_value(plan.hot_block_refs.data()[ref_index],
                                          range,
                                          storage_row,
                                          storage_col);
            }
            add_hot_reduced_cell(range, storage_row, storage_col, sum);
        }
    }

    MUDA_DEVICE void fill_hot_microblock_cache(
        SizeT range_id,
        muda::BufferView<SolveT> microblocks) const noexcept
    {
        if(!valid() || range_id >= plan.hot_block_ranges.size())
            return;
        const auto range = plan.hot_block_ranges.data()[range_id];
        if(range.ref_count == 0)
            return;

        const SizeT cell_count = matrix.block_size * matrix.block_size;
        for(SizeT cell = static_cast<SizeT>(threadIdx.x);
            cell < cell_count;
            cell += static_cast<SizeT>(blockDim.x))
        {
            const SizeT storage_row = cell / matrix.block_size;
            const SizeT storage_col = cell % matrix.block_size;
            for(SizeT ref_offset = 0;
                ref_offset < static_cast<SizeT>(range.ref_count);
                ++ref_offset)
            {
                const SizeT ref_index =
                    static_cast<SizeT>(range.first_ref) + ref_offset;
                const SizeT cache_index = ref_index * cell_count + cell;
                if(ref_index >= plan.hot_block_refs.size()
                   || cache_index >= microblocks.size())
                    continue;
                microblocks.data()[cache_index] =
                    hot_ref_cell_value(plan.hot_block_refs.data()[ref_index],
                                       range,
                                       storage_row,
                                       storage_col);
                record(SocuContactExecutorCounterSlot::HotMicroblockCacheWrite);
            }
        }
    }

    MUDA_DEVICE void execute_hot_range_cached(
        SizeT range_id,
        muda::CBufferView<SolveT> microblocks) const noexcept
    {
        if(!valid() || range_id >= plan.hot_block_ranges.size())
            return;
        const auto range = plan.hot_block_ranges.data()[range_id];
        if(range.ref_count == 0)
            return;

        if(threadIdx.x == 0)
            record(SocuContactExecutorCounterSlot::HotBlockRangeVisit);
        const SizeT cell_count = matrix.block_size * matrix.block_size;
        for(SizeT cell = static_cast<SizeT>(threadIdx.x);
            cell < cell_count;
            cell += static_cast<SizeT>(blockDim.x))
        {
            const SizeT storage_row = cell / matrix.block_size;
            const SizeT storage_col = cell % matrix.block_size;
            SolveT sum = SolveT{0};
            for(SizeT ref_offset = 0;
                ref_offset < static_cast<SizeT>(range.ref_count);
                ++ref_offset)
            {
                const SizeT ref_index =
                    static_cast<SizeT>(range.first_ref) + ref_offset;
                const SizeT cache_index = ref_index * cell_count + cell;
                if(cache_index >= microblocks.size())
                    continue;
                sum += microblocks.data()[cache_index];
            }
            add_hot_reduced_cell(range, storage_row, storage_col, sum);
        }
    }
};

template <typename StoreT, typename SolveT, typename EvaluatorT>
__global__ void socu_contact_execute_buckets_kernel(
    SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT> executor,
    SizeT first_bucket,
    SizeT bucket_count)
{
    const SizeT local_bucket = static_cast<SizeT>(blockIdx.x);
    if(local_bucket >= bucket_count)
        return;
    executor.execute_bucket(first_bucket + local_bucket,
                            static_cast<SizeT>(blockDim.x),
                            static_cast<SizeT>(threadIdx.x));
}

template <typename StoreT, typename SolveT, typename EvaluatorT>
__global__ void socu_contact_hot_reduce_recompute_kernel(
    SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT> executor,
    SizeT first_range,
    SizeT range_count)
{
    const SizeT local_range = static_cast<SizeT>(blockIdx.x);
    if(local_range >= range_count)
        return;
    executor.execute_hot_range_recompute(first_range + local_range);
}

template <typename StoreT, typename SolveT, typename EvaluatorT>
__global__ void socu_contact_hot_microblock_fill_kernel(
    SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT> executor,
    muda::BufferView<SolveT> microblocks,
    SizeT first_range,
    SizeT range_count)
{
    const SizeT local_range = static_cast<SizeT>(blockIdx.x);
    if(local_range >= range_count)
        return;
    executor.fill_hot_microblock_cache(first_range + local_range, microblocks);
}

template <typename StoreT, typename SolveT, typename EvaluatorT>
__global__ void socu_contact_hot_microblock_reduce_kernel(
    SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT> executor,
    muda::CBufferView<SolveT> microblocks,
    SizeT first_range,
    SizeT range_count)
{
    const SizeT local_range = static_cast<SizeT>(blockIdx.x);
    if(local_range >= range_count)
        return;
    executor.execute_hot_range_cached(first_range + local_range, microblocks);
}

inline cudaStream_t socu_contact_executor_stream(cudaStream_t stream) noexcept
{
    return stream == cudaStreamLegacy ? nullptr : stream;
}

template <typename StoreT, typename SolveT, typename EvaluatorT>
void launch_socu_contact_executor_direct_scatter(
    SocuContactAssemblyPlanView plan,
    SocuNativeMatrixView<SolveT> matrix,
    EvaluatorT evaluator,
    muda::BufferView<IndexT> counters = {},
    cudaStream_t stream = cudaStreamLegacy,
    SizeT first_bucket = 0,
    SizeT bucket_count = std::numeric_limits<SizeT>::max())
{
    if(plan.buckets.size() == 0 || first_bucket >= plan.buckets.size())
        return;

    const SizeT available = plan.buckets.size() - first_bucket;
    const SizeT launch_count =
        bucket_count == std::numeric_limits<SizeT>::max()
            ? available
            : (bucket_count < available ? bucket_count : available);
    if(launch_count == 0)
        return;

    constexpr unsigned int block_dim = 128;
    SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT> executor{
        plan,
        matrix,
        evaluator,
        counters};
    socu_contact_execute_buckets_kernel<StoreT, SolveT, EvaluatorT>
        <<<static_cast<unsigned int>(launch_count),
           block_dim,
           0,
           socu_contact_executor_stream(stream)>>>(executor,
                                                   first_bucket,
                                                   launch_count);
}

template <typename StoreT, typename SolveT, typename EvaluatorT>
void launch_socu_contact_executor_hot_reduce(
    SocuContactAssemblyPlanView plan,
    SocuNativeMatrixView<SolveT> matrix,
    EvaluatorT evaluator,
    muda::BufferView<IndexT> counters = {},
    cudaStream_t stream = cudaStreamLegacy,
    SizeT first_range = 0,
    SizeT range_count = std::numeric_limits<SizeT>::max())
{
    if(!SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT>::owner_reduce_strategy(
           plan.hot_block_strategy)
       || plan.hot_block_ranges.size() == 0
       || first_range >= plan.hot_block_ranges.size())
        return;

    const SizeT available = plan.hot_block_ranges.size() - first_range;
    const SizeT launch_count =
        range_count == std::numeric_limits<SizeT>::max()
            ? available
            : (range_count < available ? range_count : available);
    if(launch_count == 0)
        return;

    constexpr unsigned int block_dim = 256;
    SocuContactBucketExecutor<StoreT, SolveT, EvaluatorT> executor{
        plan,
        matrix,
        evaluator,
        counters};
    const auto cuda_stream = socu_contact_executor_stream(stream);
    if(plan.hot_block_strategy == SocuContactExecutionStrategy::Recompute)
    {
        socu_contact_hot_reduce_recompute_kernel<StoreT, SolveT, EvaluatorT>
            <<<static_cast<unsigned int>(launch_count),
               block_dim,
               0,
               cuda_stream>>>(executor, first_range, launch_count);
        return;
    }

    if(plan.hot_block_strategy == SocuContactExecutionStrategy::CachedMicroblock)
    {
        const SizeT cell_count = matrix.block_size * matrix.block_size;
        const SizeT cache_count = plan.hot_block_refs.size() * cell_count;
        if(cache_count == 0)
            return;
        muda::DeviceBuffer<SolveT> microblocks;
        microblocks.resize(cache_count);
        socu_contact_hot_microblock_fill_kernel<StoreT, SolveT, EvaluatorT>
            <<<static_cast<unsigned int>(launch_count),
               block_dim,
               0,
               cuda_stream>>>(executor,
                              microblocks.view(),
                              first_range,
                              launch_count);
        socu_contact_hot_microblock_reduce_kernel<StoreT, SolveT, EvaluatorT>
            <<<static_cast<unsigned int>(launch_count),
               block_dim,
               0,
               cuda_stream>>>(executor,
                              microblocks.view().as_const(),
                              first_range,
                              launch_count);
        cudaStreamSynchronize(cuda_stream);
    }
}

template <typename StoreT, typename SolveT, typename EvaluatorT>
void launch_socu_contact_executor(
    SocuContactAssemblyPlanView plan,
    SocuNativeMatrixView<SolveT> matrix,
    EvaluatorT evaluator,
    muda::BufferView<IndexT> counters = {},
    cudaStream_t stream = cudaStreamLegacy,
    SizeT first_bucket = 0,
    SizeT bucket_count = std::numeric_limits<SizeT>::max())
{
    launch_socu_contact_executor_direct_scatter<StoreT, SolveT>(
        plan,
        matrix,
        evaluator,
        counters,
        stream,
        first_bucket,
        bucket_count);
    launch_socu_contact_executor_hot_reduce<StoreT, SolveT>(
        plan,
        matrix,
        evaluator,
        counters,
        stream);
}
}  // namespace uipc::backend::cuda_mixed
