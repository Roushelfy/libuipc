#pragma once

#include <linear_system/socu_contact_assembly_plan.h>
#include <linear_system/socu_native_matrix_builder.h>
#include <muda/atomic.h>

namespace uipc::backend::cuda_mixed
{
enum class SocuContactProgramWriterCounterSlot : IndexT
{
    ContactWrite = 0,
    ProgramMissing,
    ProgramSkipped,
    ProgramDropped,
    ProgramMixedRejected,
    ExactTaskWrite,
    UnsupportedTask,
    Count,
};

template <typename StoreT, typename SolveT>
struct SocuContactProgramWriter
{
    SocuContactAssemblyPlanView plan;
    SocuNativeMatrixView<SolveT> matrix;
    muda::BufferView<IndexT> counters;

    MUDA_GENERIC bool valid() const noexcept
    {
        return plan.valid() && matrix.valid();
    }

    MUDA_DEVICE void record(SocuContactProgramWriterCounterSlot slot,
                            IndexT amount = 1) const noexcept
    {
        const auto index = static_cast<IndexT>(slot);
        if(counters.data() != nullptr && index >= 0
           && static_cast<SizeT>(index) < counters.size())
            muda::atomic_add(counters.data(static_cast<SizeT>(index)), amount);
    }

    MUDA_GENERIC muda::CBufferView<SocuAssemblyDofLane> lanes_for(
        SocuAssemblySideId side_id) const noexcept
    {
        if(side_id == SocuInvalidAssemblySideId
           || static_cast<SizeT>(side_id) >= plan.sides.size())
            return {};

        const auto& side = plan.sides.data()[static_cast<SizeT>(side_id)];
        const SizeT first = static_cast<SizeT>(side.first_lane);
        const SizeT count = static_cast<SizeT>(side.lane_count);
        if(first + count > plan.lanes.size())
            return {};
        return plan.lanes.subview(first, count);
    }

    template <typename HMat>
    MUDA_DEVICE void write_contact(SocuContactSourceId source_id,
                                   SizeT local_contact_id,
                                   const HMat& H) const noexcept
    {
        if(!valid())
            return;

        const auto mapping = plan.program_for(source_id, local_contact_id);
        if(mapping.status != SocuContactProgramMapStatus::Valid)
        {
            record_map_status(mapping.status);
            return;
        }

        if(mapping.program_id == SocuInvalidContactProgramId
           || static_cast<SizeT>(mapping.program_id) >= plan.programs.size())
        {
            record(SocuContactProgramWriterCounterSlot::ProgramMissing);
            return;
        }

        const auto program =
            plan.programs.data()[static_cast<SizeT>(mapping.program_id)];
        if(program.program_kind == SocuContactProgramKind::Skipped
           || program.task_count == 0)
        {
            record(SocuContactProgramWriterCounterSlot::ProgramSkipped);
            return;
        }
        if(program.program_kind == SocuContactProgramKind::Drop)
        {
            record(SocuContactProgramWriterCounterSlot::ProgramDropped);
            return;
        }
        if(program.program_kind == SocuContactProgramKind::MixedRejectedDebugOnly)
        {
            record(SocuContactProgramWriterCounterSlot::ProgramMixedRejected);
            return;
        }

        record(SocuContactProgramWriterCounterSlot::ContactWrite);
        for(std::uint16_t i = 0; i < program.task_count; ++i)
        {
            const SizeT task_id = static_cast<SizeT>(program.first_task) + i;
            if(task_id >= plan.tasks.size())
                continue;
            write_task(program, plan.tasks.data()[task_id], H);
        }
    }

    template <typename HMat>
    MUDA_DEVICE void write_task(const SocuContactProgramHeader& program,
                                const SocuContactMicroTask& task,
                                const HMat& H) const noexcept
    {
        (void)program;
        if(!exact_task(task.write_kind))
        {
            record(SocuContactProgramWriterCounterSlot::UnsupportedTask);
            return;
        }

        const auto row_lanes = lanes_for(task.row_side);
        const auto col_lanes = lanes_for(task.col_side);
        if(row_lanes.data() == nullptr || col_lanes.data() == nullptr)
        {
            record(SocuContactProgramWriterCounterSlot::ProgramSkipped);
            return;
        }

        if(task.band == SocuAssemblyBand::Diag)
        {
            write_diag_task(task, row_lanes, col_lanes, H);
            record(SocuContactProgramWriterCounterSlot::ExactTaskWrite);
            return;
        }
        if(task.band == SocuAssemblyBand::FirstOffdiag)
        {
            write_first_offdiag_task(task, row_lanes, col_lanes, H);
            record(SocuContactProgramWriterCounterSlot::ExactTaskWrite);
            return;
        }

        record(SocuContactProgramWriterCounterSlot::UnsupportedTask);
    }

    MUDA_GENERIC static bool exact_task(SocuAssemblyWriteKind kind) noexcept
    {
        return kind == SocuAssemblyWriteKind::ExactFemFem
               || kind == SocuAssemblyWriteKind::ExactAbdFem
               || kind == SocuAssemblyWriteKind::ExactFemAbd;
    }

    MUDA_DEVICE void record_map_status(
        SocuContactProgramMapStatus status) const noexcept
    {
        switch(status)
        {
            case SocuContactProgramMapStatus::Skipped:
                record(SocuContactProgramWriterCounterSlot::ProgramSkipped);
                break;
            case SocuContactProgramMapStatus::Dropped:
                record(SocuContactProgramWriterCounterSlot::ProgramDropped);
                break;
            case SocuContactProgramMapStatus::MixedRejected:
                record(SocuContactProgramWriterCounterSlot::ProgramMixedRejected);
                break;
            case SocuContactProgramMapStatus::Missing:
            default:
                record(SocuContactProgramWriterCounterSlot::ProgramMissing);
                break;
        }
    }

    MUDA_GENERIC static bool has_flag(const SocuContactMicroTask& task,
                                      SocuContactTaskFlag flag) noexcept
    {
        return (task.flags & static_cast<std::uint8_t>(flag)) != 0;
    }

    template <typename HMat>
    MUDA_DEVICE StoreT projected_value(const SocuContactMicroTask& task,
                                       const SocuAssemblyDofLane& row_lane,
                                       const SocuAssemblyDofLane& col_lane,
                                       const HMat& H) const noexcept
    {
        if(row_lane.component >= 3 || col_lane.component >= 3)
            return StoreT{0};

        const IndexT h_row =
            static_cast<IndexT>(task.local_row_vertex) * 3
            + static_cast<IndexT>(row_lane.component);
        const IndexT h_col =
            static_cast<IndexT>(task.local_col_vertex) * 3
            + static_cast<IndexT>(col_lane.component);
        return static_cast<StoreT>(static_cast<StoreT>(row_lane.weight)
                                   * static_cast<StoreT>(H(h_row, h_col))
                                   * static_cast<StoreT>(col_lane.weight));
    }

    template <typename HMat>
    MUDA_DEVICE void write_diag_task(
        const SocuContactMicroTask& task,
        muda::CBufferView<SocuAssemblyDofLane> row_lanes,
        muda::CBufferView<SocuAssemblyDofLane> col_lanes,
        const HMat& H) const noexcept
    {
        const bool mirror =
            has_flag(task, SocuContactTaskFlag::MirrorDiagBlock);
        for(SizeT row = 0; row < row_lanes.size(); ++row)
        {
            const auto row_lane = row_lanes.data()[row];
            for(SizeT col = 0; col < col_lanes.size(); ++col)
            {
                const auto col_lane = col_lanes.data()[col];
                const auto value = projected_value(task, row_lane, col_lane, H);
                matrix.add_diag_scalar(task.block_or_left_block,
                                       row_lane.lane,
                                       col_lane.lane,
                                       static_cast<SolveT>(value));
                if(mirror
                   && (row_lane.block != col_lane.block
                       || row_lane.lane != col_lane.lane))
                {
                    matrix.add_diag_scalar(task.block_or_left_block,
                                           col_lane.lane,
                                           row_lane.lane,
                                           static_cast<SolveT>(value));
                }
            }
        }
    }

    template <typename HMat>
    MUDA_DEVICE void write_first_offdiag_task(
        const SocuContactMicroTask& task,
        muda::CBufferView<SocuAssemblyDofLane> row_lanes,
        muda::CBufferView<SocuAssemblyDofLane> col_lanes,
        const HMat& H) const noexcept
    {
        const bool transposed =
            has_flag(task, SocuContactTaskFlag::TransposedFirstOffdiag);
        for(SizeT row = 0; row < row_lanes.size(); ++row)
        {
            const auto row_lane = row_lanes.data()[row];
            for(SizeT col = 0; col < col_lanes.size(); ++col)
            {
                const auto col_lane = col_lanes.data()[col];
                const auto value = projected_value(task, row_lane, col_lane, H);
                const auto storage_row =
                    transposed ? col_lane.lane : row_lane.lane;
                const auto storage_col =
                    transposed ? row_lane.lane : col_lane.lane;
                matrix.add_first_offdiag_scalar(task.block_or_left_block,
                                                storage_row,
                                                storage_col,
                                                static_cast<SolveT>(value));
            }
        }
    }
};
}  // namespace uipc::backend::cuda_mixed
