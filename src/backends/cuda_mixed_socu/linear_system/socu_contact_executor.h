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
            writer.write_task(program, plan.tasks.data()[task_id], H);
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

inline cudaStream_t socu_contact_executor_stream(cudaStream_t stream) noexcept
{
    return stream == cudaStreamLegacy ? nullptr : stream;
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
}  // namespace uipc::backend::cuda_mixed
