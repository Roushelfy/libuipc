#pragma once

#include <affine_body/abd_jacobi_matrix.h>
#include <linear_system/socu_native_contact_targets.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT, typename SolveT>
struct SocuNativeContactExactWriter
{
    StructuredDeviceMatrixSink<StoreT, SolveT>   matrix;
    muda::CBufferView<ABDJacobi>                 abd_vertex_to_J;
    muda::BufferView<IndexT>                     counters;

    MUDA_GENERIC bool valid() const noexcept
    {
        return matrix.use_native_matrix && matrix.native_matrix.D.data() != nullptr
               && matrix.native_matrix.block_count != 0
               && matrix.native_matrix.block_size != 0;
    }

    static MUDA_DEVICE ActivePolicy::AluScalar
    direct_weight(const SocuNativeContactDirectSide& side, IndexT local) noexcept
    {
        return static_cast<ActivePolicy::AluScalar>(side.weight(local));
    }

    static MUDA_DEVICE SizeT
    direct_block(const SocuNativeContactDirectSide& side, IndexT local) noexcept
    {
        return side.block(local);
    }

    template <typename H3>
    static MUDA_DEVICE StoreT direct_projected_value(
        const SocuNativeContactStencilTarget& target,
        IndexT                                local_row,
        IndexT                                local_col,
        const H3&                             H) noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        const auto& row = target.row_direct;
        const auto& col = target.col_direct;
        const Alu row_w = direct_weight(row, local_row);
        const Alu col_w = direct_weight(col, local_col);
        const IndexT row_comp = row.component(local_row);
        const IndexT col_comp = col.component(local_col);
        return static_cast<StoreT>(
            row_w * static_cast<Alu>(H(row_comp, col_comp)) * col_w);
    }

    MUDA_DEVICE void record_counter(StructuredSinkWriteClass cls) const noexcept
    {
        StructuredAssemblyCounterSlot slot =
            StructuredAssemblyCounterSlot::ContactOffBandScalarDrop;
        switch(cls)
        {
            case StructuredSinkWriteClass::Diag:
                slot = StructuredAssemblyCounterSlot::ContactDiagScalarWrite;
                break;
            case StructuredSinkWriteClass::FirstOffdiag:
                slot =
                    StructuredAssemblyCounterSlot::ContactFirstOffdiagScalarWrite;
                break;
            case StructuredSinkWriteClass::OffBand:
                slot = StructuredAssemblyCounterSlot::ContactOffBandScalarDrop;
                break;
            case StructuredSinkWriteClass::Skipped:
            default:
                return;
        }

        const IndexT index = static_cast<IndexT>(slot);
        if(counters.data() != nullptr && index >= 0
           && static_cast<SizeT>(index) < counters.size())
            muda::atomic_add(counters.data(static_cast<SizeT>(index)), IndexT{1});
    }

    MUDA_DEVICE void record_counter_slot(
        StructuredAssemblyCounterSlot slot) const noexcept
    {
        const IndexT index = static_cast<IndexT>(slot);
        if(counters.data() != nullptr && index >= 0
           && static_cast<SizeT>(index) < counters.size())
            muda::atomic_add(counters.data(static_cast<SizeT>(index)), IndexT{1});
    }

    MUDA_DEVICE StructuredSinkWriteClass add_direct_primary(
        const SocuNativeContactStencilTarget& target,
        IndexT                                local_row,
        IndexT                                local_col,
        StoreT                                value) const noexcept
    {
        if(!valid() || !target.direct_lanes_valid
           || !target.row_direct.valid_index(local_row)
           || !target.col_direct.valid_index(local_col))
            return StructuredSinkWriteClass::Skipped;

        const auto& native = matrix.native_matrix;
        const auto  v      = static_cast<SolveT>(value);
        const SizeT row_block = target.row_direct.block(local_row);
        const SizeT col_block = target.col_direct.block(local_col);
        const SizeT row_lane  = target.row_direct.lane(local_row);
        const SizeT col_lane  = target.col_direct.lane(local_col);

        if(row_block == col_block)
        {
            if(!native.valid_block_entry(row_block, row_lane, col_lane))
                return StructuredSinkWriteClass::Skipped;
            const SizeT index =
                native.diag_index(row_block, row_lane, col_lane);
            if(index >= native.D.size())
                return StructuredSinkWriteClass::Skipped;
            muda::atomic_add(native.D.data(index), v);
            return StructuredSinkWriteClass::Diag;
        }

        const SizeT min_block = row_block < col_block ? row_block : col_block;
        const SizeT max_block = row_block > col_block ? row_block : col_block;
        if(max_block != min_block + 1 || native.E.data() == nullptr)
            return StructuredSinkWriteClass::OffBand;

        const bool  row_is_left = row_block < col_block;
        const SizeT write_row   = row_is_left ? col_lane : row_lane;
        const SizeT write_col   = row_is_left ? row_lane : col_lane;
        if(min_block >= native.first_offdiag_block_count
           || write_row >= native.block_size || write_col >= native.block_size)
            return StructuredSinkWriteClass::Skipped;
        const SizeT index =
            native.first_offdiag_index(min_block, write_row, write_col);
        if(index >= native.E.size())
            return StructuredSinkWriteClass::Skipped;
        muda::atomic_add(native.E.data(index), v);
        return StructuredSinkWriteClass::FirstOffdiag;
    }

    MUDA_DEVICE bool add_direct_diag_primary(
        const SocuNativeContactDirectSide& side,
        IndexT                             local,
        StoreT                             value) const noexcept
    {
        if(!valid() || !side.valid_index(local))
            return false;

        const auto& native = matrix.native_matrix;
        const SizeT block  = side.block(local);
        const SizeT lane   = side.lane(local);
        if(!native.valid_block_entry(block, lane, lane))
            return false;
        const SizeT index = native.diag_index(block, lane, lane);
        if(index >= native.D.size())
            return false;
        muda::atomic_add(native.D.data(index), static_cast<SolveT>(value));
        record_counter(StructuredSinkWriteClass::Diag);
        return true;
    }

    MUDA_DEVICE StructuredSinkWriteClass add_target_scalar_with_old(
        const SocuNativeContactStencilTarget& target,
        IndexT                                local_row,
        IndexT                                local_col,
        StoreT                                value,
        bool                                  mirror_diag_block,
        IndexT                                old_row,
        IndexT                                old_col) const noexcept
    {
        if(!target.direct_lanes_valid)
            return StructuredSinkWriteClass::Skipped;

        const auto cls = add_direct_primary(target, local_row, local_col, value);
        record_counter(cls);
        matrix.add_hessian_scalar_compare(old_row, old_col, value);
        if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
           && old_row != old_col)
        {
            const SizeT block    = target.row_direct.block(local_row);
            const SizeT row_lane = target.row_direct.lane(local_row);
            const SizeT col_lane = target.col_direct.lane(local_col);
            const auto& native = matrix.native_matrix;
            if(native.valid_block_entry(block, col_lane, row_lane))
            {
                const SizeT index = native.diag_index(block, col_lane, row_lane);
                if(index < native.D.size())
                {
                    muda::atomic_add(native.D.data(index),
                                     static_cast<SolveT>(value));
                    record_counter(StructuredSinkWriteClass::Diag);
                }
            }
            matrix.add_hessian_scalar_compare(old_col, old_row, value);
        }
        return cls;
    }

    MUDA_DEVICE StructuredSinkWriteClass add_target_scalar(
        const SocuNativeContactStencilTarget& target,
        IndexT                                local_row,
        IndexT                                local_col,
        StoreT                                value,
        bool                                  mirror_diag_block) const noexcept
    {
        return add_target_scalar_with_old(target,
                                          local_row,
                                          local_col,
                                          value,
                                          mirror_diag_block,
                                          target.row_old_dof + local_row,
                                          target.col_old_dof + local_col);
    }

    template <typename H3>
    MUDA_DEVICE void add_fem_fem(const SocuNativeContactStencilTarget& target,
                                 const H3& H) const noexcept
    {
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                add_target_scalar(target,
                                  r,
                                  c,
                                  static_cast<StoreT>(H(r, c)),
                                  target.mirror_diag_block);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void add_abd_fem(const SocuNativeContactStencilTarget& target,
                                 const H3& H) const noexcept
    {
#pragma unroll 1
        for(IndexT r = 0; r < target.row_direct.count; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < target.col_direct.count; ++c)
            {
                add_target_scalar(target,
                                  r,
                                  c,
                                  direct_projected_value(target, r, c, H),
                                  target.mirror_diag_block);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void add_fem_abd(const SocuNativeContactStencilTarget& target,
                                 const H3& H) const noexcept
    {
#pragma unroll
        for(IndexT r = 0; r < target.row_direct.count; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < target.col_direct.count; ++c)
            {
                add_target_scalar(target,
                                  r,
                                  c,
                                  direct_projected_value(target, r, c, H),
                                  target.mirror_diag_block);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void add_abd_abd(const SocuNativeContactStencilTarget& target,
                                 const H3& H) const noexcept
    {
        if(target.row_abd_body == target.col_abd_body)
        {
#pragma unroll
            for(IndexT row_block = 0; row_block < 4; ++row_block)
            {
#pragma unroll
                for(IndexT col_block = row_block; col_block < 4; ++col_block)
                {
#pragma unroll
                    for(IndexT row = 0; row < 3; ++row)
                    {
#pragma unroll
                        for(IndexT col = 0; col < 3; ++col)
                        {
                            const IndexT local_i = row_block * 3 + row;
                            const IndexT local_j = col_block * 3 + col;
                            auto value = static_cast<ActivePolicy::AluScalar>(
                                direct_projected_value(target, local_i, local_j, H));
                            if(target.row_global_vertex != target.col_global_vertex)
                            {
                                value += static_cast<ActivePolicy::AluScalar>(
                                    direct_projected_value(target,
                                                           local_j,
                                                           local_i,
                                                           H));
                            }
                            add_target_scalar(target,
                                              local_i,
                                              local_j,
                                              static_cast<StoreT>(value),
                                              row_block != col_block);
                        }
                    }
                }
            }
            return;
        }

        if(target.row_abd_body < target.col_abd_body)
        {
#pragma unroll 1
            for(IndexT r = 0; r < target.row_direct.count; ++r)
            {
#pragma unroll 1
                for(IndexT c = 0; c < target.col_direct.count; ++c)
                {
                    add_target_scalar(target,
                                      r,
                                      c,
                                      direct_projected_value(target, r, c, H),
                                      target.mirror_diag_block);
                }
            }
            return;
        }

#pragma unroll 1
        for(IndexT r = 0; r < target.col_direct.count; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < target.row_direct.count; ++c)
            {
                add_target_scalar_with_old(target,
                                           c,
                                           r,
                                           direct_projected_value(target, c, r, H),
                                           target.mirror_diag_block,
                                           target.col_old_dof + r,
                                           target.row_old_dof + c);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE bool write_direct_scalar_diag(
        const SocuNativeContactStencilTarget& target,
        const H3&                             H) const noexcept
    {
        if(!target.direct_lanes_valid || target.row_old_dof < 0)
            return false;

        bool wrote = false;
#pragma unroll 1
        for(IndexT local = 0; local < target.row_direct.count; ++local)
        {
            const StoreT value =
                direct_projected_value(target, local, local, H);
            wrote |= add_direct_diag_primary(target.row_direct, local, value);
            matrix.add_hessian_scalar_compare(target.row_old_dof + local,
                                              target.row_old_dof + local,
                                              value);
        }
        return wrote;
    }

    MUDA_DEVICE bool write_direct_lumped_diag(
        const SocuNativeContactStencilTarget& target,
        const ActivePolicy::AluScalar (&lump)[3]) const noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        if(!target.direct_lanes_valid || target.row_old_dof < 0)
            return false;

        bool wrote = false;
#pragma unroll 1
        for(IndexT local = 0; local < target.row_direct.count; ++local)
        {
            const IndexT comp = target.row_direct.component(local);
            const Alu    w    = direct_weight(target.row_direct, local);
            const StoreT value =
                static_cast<StoreT>(w * lump[comp] * w);
            wrote |= add_direct_diag_primary(target.row_direct, local, value);
            matrix.add_hessian_scalar_compare(target.row_old_dof + local,
                                              target.row_old_dof + local,
                                              value);
        }
        return wrote;
    }

    template <int StencilSize>
    static MUDA_DEVICE SizeT diagonal_target_offset(IndexT local_vertex) noexcept
    {
        SizeT offset = 0;
#pragma unroll
        for(IndexT row = 0; row < StencilSize; ++row)
        {
            if(row == local_vertex)
                return offset;
            offset += static_cast<SizeT>(StencilSize - row);
        }
        return offset;
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_diag_fallback_stencil(
        muda::CBufferView<SocuNativeContactStencilTarget> targets,
        SizeT                                             base,
        const HMat&                                       H) const noexcept
    {
        record_counter_slot(
            StructuredAssemblyCounterSlot::ContactOffBandDiagFallbackStencil);
#pragma unroll
        for(IndexT local = 0; local < StencilSize; ++local)
        {
            const auto target =
                targets.data()[base + diagonal_target_offset<StencilSize>(local)];
            write_direct_scalar_diag(
                target,
                H.template block<3, 3>(local * 3, local * 3));
        }
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_lump_fallback_stencil(
        muda::CBufferView<SocuNativeContactStencilTarget> targets,
        SizeT                                             base,
        const HMat&                                       H) const noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        record_counter_slot(
            StructuredAssemblyCounterSlot::ContactOffBandLumpFallbackStencil);
#pragma unroll
        for(IndexT local = 0; local < StencilSize; ++local)
        {
            Alu lump[3] = {Alu{0}, Alu{0}, Alu{0}};
#pragma unroll
            for(IndexT r = 0; r < 3; ++r)
            {
#pragma unroll 1
                for(IndexT j = 0; j < StencilSize; ++j)
                {
#pragma unroll
                    for(IndexT c = 0; c < 3; ++c)
                    {
                        const Alu value =
                            static_cast<Alu>(H(local * 3 + r, j * 3 + c));
                        lump[r] += value < Alu{0} ? -value : value;
                    }
                }
            }

            const auto target =
                targets.data()[base + diagonal_target_offset<StencilSize>(local)];
            write_direct_lumped_diag(target, lump);
        }
    }

    template <typename H3>
    MUDA_DEVICE bool write_half_block(
        const SocuNativeContactStencilTarget& target,
        const H3& H) const noexcept
    {
        if(!valid() || !target.exact_in_band()
           || !target.direct_projection_valid)
            return false;
        if(target.row_old_dof < 0 || target.col_old_dof < 0)
            return false;

        if(target.row_kind == SocuNativeDescriptorKind::Fem
           && target.col_kind == SocuNativeDescriptorKind::Fem)
        {
            add_fem_fem(target, H);
            return true;
        }

        if(target.row_kind == SocuNativeDescriptorKind::Abd
           && target.col_kind == SocuNativeDescriptorKind::Fem)
        {
            add_abd_fem(target, H);
            return true;
        }

        if(target.row_kind == SocuNativeDescriptorKind::Fem
           && target.col_kind == SocuNativeDescriptorKind::Abd)
        {
            add_fem_abd(target, H);
            return true;
        }

        if(target.row_kind == SocuNativeDescriptorKind::Abd
           && target.col_kind == SocuNativeDescriptorKind::Abd)
        {
            add_abd_abd(target, H);
            return true;
        }

        return false;
    }
};
}  // namespace uipc::backend::cuda_mixed
