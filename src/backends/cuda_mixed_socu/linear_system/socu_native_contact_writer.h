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

    static MUDA_DEVICE IndexT abd_component(IndexT dof) noexcept
    {
        return dof < 3 ? dof : (dof - 3) / 3;
    }

    static MUDA_DEVICE ActivePolicy::AluScalar
    abd_weight(const ABDJacobi& J, IndexT dof) noexcept
    {
        if(dof < 3)
            return ActivePolicy::AluScalar{1};
        return static_cast<ActivePolicy::AluScalar>(J.x_bar()((dof - 3) % 3));
    }

    MUDA_DEVICE ABDJacobi jacobian(IndexT index) const noexcept
    {
        if(index < 0 || static_cast<SizeT>(index) >= abd_vertex_to_J.size()
           || abd_vertex_to_J.data() == nullptr)
            return ABDJacobi{};
        return abd_vertex_to_J.data()[static_cast<SizeT>(index)];
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

    MUDA_DEVICE StructuredSinkWriteClass add_scalar(IndexT old_row,
                                                    IndexT old_col,
                                                    StoreT value,
                                                    bool   mirror_diag_block) const noexcept
    {
        if(!valid())
            return StructuredSinkWriteClass::Skipped;

        const auto cls = matrix.add_hessian_scalar_status_native(
            matrix.native_matrix,
            old_row,
            old_col,
            value);
        record_counter(cls);
        matrix.add_hessian_scalar_compare(old_row, old_col, value);
        if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
           && old_row != old_col)
        {
            const auto mirror_cls = matrix.add_hessian_scalar_status_native(
                matrix.native_matrix,
                old_col,
                old_row,
                value);
            record_counter(mirror_cls);
            matrix.add_hessian_scalar_compare(old_col, old_row, value);
        }
        return cls;
    }

    MUDA_DEVICE StructuredSinkWriteClass add_direct_primary(
        const SocuNativeContactStencilTarget& target,
        IndexT                                local_row,
        IndexT                                local_col,
        StoreT                                value) const noexcept
    {
        if(!valid() || !target.direct_lanes_valid || local_row < 0
           || local_col < 0
           || local_row >= SocuNativeContactMaxDofsPerVertex
           || local_col >= SocuNativeContactMaxDofsPerVertex)
            return StructuredSinkWriteClass::Skipped;

        const auto& native = matrix.native_matrix;
        const auto  v      = static_cast<SolveT>(value);
        if(target.half_block_class == SocuNativeBandClass::Diag)
        {
            const SizeT row_lane =
                target.row_direct_lanes[static_cast<SizeT>(local_row)];
            const SizeT col_lane =
                target.col_direct_lanes[static_cast<SizeT>(local_col)];
            if(!native.valid_block_entry(target.block_or_left_block,
                                         row_lane,
                                         col_lane))
                return StructuredSinkWriteClass::Skipped;
            const SizeT index =
                native.diag_index(target.block_or_left_block, row_lane, col_lane);
            if(index >= native.D.size())
                return StructuredSinkWriteClass::Skipped;
            muda::atomic_add(native.D.data(index), v);
            return StructuredSinkWriteClass::Diag;
        }

        if(target.half_block_class == SocuNativeBandClass::FirstOffdiag)
        {
            const SizeT row_lane =
                target.transposed_first_offdiag
                    ? target.col_direct_lanes[static_cast<SizeT>(local_col)]
                    : target.row_direct_lanes[static_cast<SizeT>(local_row)];
            const SizeT col_lane =
                target.transposed_first_offdiag
                    ? target.row_direct_lanes[static_cast<SizeT>(local_row)]
                    : target.col_direct_lanes[static_cast<SizeT>(local_col)];
            if(target.block_or_left_block >= native.first_offdiag_block_count
               || row_lane >= native.block_size || col_lane >= native.block_size
               || native.E.data() == nullptr)
                return StructuredSinkWriteClass::Skipped;
            const SizeT index = native.first_offdiag_index(
                target.block_or_left_block,
                row_lane,
                col_lane);
            if(index >= native.E.size())
                return StructuredSinkWriteClass::Skipped;
            muda::atomic_add(native.E.data(index), v);
            return StructuredSinkWriteClass::FirstOffdiag;
        }

        return StructuredSinkWriteClass::OffBand;
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
            return add_scalar(old_row, old_col, value, mirror_diag_block);

        const auto cls = add_direct_primary(target, local_row, local_col, value);
        record_counter(cls);
        matrix.add_hessian_scalar_compare(old_row, old_col, value);
        if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
           && old_row != old_col)
        {
            const SizeT row_lane =
                target.row_direct_lanes[static_cast<SizeT>(local_row)];
            const SizeT col_lane =
                target.col_direct_lanes[static_cast<SizeT>(local_col)];
            const auto& native = matrix.native_matrix;
            if(native.valid_block_entry(target.block_or_left_block,
                                        col_lane,
                                        row_lane))
            {
                const SizeT index = native.diag_index(
                    target.block_or_left_block,
                    col_lane,
                    row_lane);
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
        using Alu      = ActivePolicy::AluScalar;
        const auto row_J = jacobian(target.row_jacobian_index);
#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
            const IndexT comp = abd_component(r);
            const Alu    wr   = abd_weight(row_J, r);
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                const StoreT value =
                    static_cast<StoreT>(wr * static_cast<Alu>(H(comp, c)));
                add_target_scalar(target, r, c, value, target.mirror_diag_block);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void add_fem_abd(const SocuNativeContactStencilTarget& target,
                                 const H3& H) const noexcept
    {
        using Alu      = ActivePolicy::AluScalar;
        const auto col_J = jacobian(target.col_jacobian_index);
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                const IndexT comp = abd_component(c);
                const Alu    wc   = abd_weight(col_J, c);
                const StoreT value =
                    static_cast<StoreT>(static_cast<Alu>(H(r, comp)) * wc);
                add_target_scalar(target, r, c, value, target.mirror_diag_block);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void add_abd_abd(const SocuNativeContactStencilTarget& target,
                                 const H3& H) const noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        const auto row_J = jacobian(target.row_jacobian_index);
        const auto col_J = jacobian(target.col_jacobian_index);

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
                            const IndexT comp_i  = abd_component(local_i);
                            const IndexT comp_j  = abd_component(local_j);
                            const Alu    row_w_i = abd_weight(row_J, local_i);
                            const Alu    row_w_j = abd_weight(row_J, local_j);
                            const Alu    col_w_i = abd_weight(col_J, local_i);
                            const Alu    col_w_j = abd_weight(col_J, local_j);
                            Alu value = row_w_i * static_cast<Alu>(H(comp_i, comp_j))
                                        * col_w_j;
                            if(target.row_global_vertex != target.col_global_vertex)
                            {
                                value += col_w_i
                                         * static_cast<Alu>(H(comp_j, comp_i))
                                         * row_w_j;
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
            for(IndexT r = 0; r < 12; ++r)
            {
                const IndexT comp_r = abd_component(r);
                const Alu    wr     = abd_weight(row_J, r);
#pragma unroll 1
                for(IndexT c = 0; c < 12; ++c)
                {
                    const IndexT comp_c = abd_component(c);
                    const Alu    wc     = abd_weight(col_J, c);
                    const Alu value =
                        wr * static_cast<Alu>(H(comp_r, comp_c)) * wc;
                    add_target_scalar(target,
                                      r,
                                      c,
                                      static_cast<StoreT>(value),
                                      target.mirror_diag_block);
                }
            }
            return;
        }

#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
            const IndexT comp_r = abd_component(r);
            const Alu    wr     = abd_weight(col_J, r);
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                const IndexT comp_c = abd_component(c);
                const Alu    wc     = abd_weight(row_J, c);
                const Alu value =
                    wr * static_cast<Alu>(H(comp_c, comp_r)) * wc;
                add_target_scalar_with_old(target,
                                           c,
                                           r,
                                           static_cast<StoreT>(value),
                                           target.mirror_diag_block,
                                           target.col_old_dof + r,
                                           target.row_old_dof + c);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE bool write_half_block(
        const SocuNativeContactStencilTarget& target,
        const H3& H) const noexcept
    {
        if(!valid() || !target.exact_in_band())
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
