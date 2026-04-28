#pragma once

#include <affine_body/abd_jacobi_matrix.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>
#include <muda/atomic.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT, typename SolveT>
struct StructuredContactAssemblySink
{
    StructuredDeviceAssemblySink<StoreT, SolveT> sink;

    IndexT abd_vertex_offset = -1;
    IndexT abd_vertex_count  = 0;
    IndexT abd_old_dof_offset = -1;
    muda::CBufferView<IndexT>    abd_vertex_to_body;
    muda::CBufferView<ABDJacobi> abd_vertex_to_J;
    muda::CBufferView<IndexT>    abd_body_is_fixed;

    IndexT fem_vertex_offset = -1;
    IndexT fem_vertex_count  = 0;
    IndexT fem_old_dof_offset = -1;
    muda::CBufferView<IndexT> fem_vertex_is_fixed;

    // [diag scalar writes, first-offdiag scalar writes, off-band scalar drops,
    //  near contact pairs, off-band contact pairs]
    muda::BufferView<IndexT> counters;

    struct VertexMap
    {
        enum Kind : IndexT
        {
            None = 0,
            Abd  = 1,
            Fem  = 2,
        };

        Kind      kind = None;
        IndexT    old_dof = -1;
        IndexT    body = -1;
        IndexT    local_vertex = -1;
        ABDJacobi J;
        bool      fixed = false;
    };

    MUDA_GENERIC bool valid() const noexcept { return sink.valid(); }

    MUDA_DEVICE VertexMap map_vertex(IndexT global_vertex) const noexcept
    {
        VertexMap mapped;
        if(abd_vertex_offset >= 0 && global_vertex >= abd_vertex_offset
           && global_vertex < abd_vertex_offset + abd_vertex_count)
        {
            const IndexT local = global_vertex - abd_vertex_offset;
            const IndexT body  = abd_vertex_to_body.data()[local];
            mapped.kind        = VertexMap::Abd;
            mapped.local_vertex = local;
            mapped.body        = body;
            mapped.old_dof     = abd_old_dof_offset + body * 12;
            mapped.J           = abd_vertex_to_J.data()[local];
            mapped.fixed       = abd_body_is_fixed.data()[body] != 0;
            return mapped;
        }

        if(fem_vertex_offset >= 0 && global_vertex >= fem_vertex_offset
           && global_vertex < fem_vertex_offset + fem_vertex_count)
        {
            const IndexT local = global_vertex - fem_vertex_offset;
            mapped.kind        = VertexMap::Fem;
            mapped.local_vertex = local;
            mapped.old_dof     = fem_old_dof_offset + local * 3;
            mapped.fixed       = fem_vertex_is_fixed.data()[local] != 0;
            return mapped;
        }

        return mapped;
    }

    MUDA_DEVICE void add_counter(StructuredSinkWriteClass cls) const noexcept
    {
        if(counters.data() == nullptr)
            return;
        switch(cls)
        {
            case StructuredSinkWriteClass::Diag:
                muda::atomic_add(counters.data(0), IndexT{1});
                break;
            case StructuredSinkWriteClass::FirstOffdiag:
                muda::atomic_add(counters.data(1), IndexT{1});
                break;
            case StructuredSinkWriteClass::OffBand:
                muda::atomic_add(counters.data(2), IndexT{1});
                break;
            case StructuredSinkWriteClass::Skipped:
            default:
                break;
        }
    }

    MUDA_DEVICE void add_pair_counter(bool saw_near, bool saw_off_band) const noexcept
    {
        if(counters.data() == nullptr)
            return;
        if(saw_off_band)
            muda::atomic_add(counters.data(4), IndexT{1});
        else if(saw_near)
            muda::atomic_add(counters.data(3), IndexT{1});
    }

    MUDA_DEVICE StructuredSinkWriteClass add_scalar_counted(IndexT old_row,
                                                            IndexT old_col,
                                                            StoreT value) const noexcept
    {
        const auto cls = sink.add_hessian_scalar_status(old_row, old_col, value);
        add_counter(cls);
        return cls;
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

    template <typename H3>
    MUDA_DEVICE void add_fem_fem(IndexT old_row,
                                 IndexT old_col,
                                 const H3& H,
                                 bool mirror_diag_block = false) const noexcept
    {
        bool saw_near     = false;
        bool saw_off_band = false;
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                const IndexT row = old_row + r;
                const IndexT col = old_col + c;
                const StoreT value = static_cast<StoreT>(H(r, c));
                const auto cls = add_scalar_counted(row, col, value);
                if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
                   && row != col)
                    add_scalar_counted(col, row, value);
                saw_near |= cls == StructuredSinkWriteClass::Diag
                            || cls == StructuredSinkWriteClass::FirstOffdiag;
                saw_off_band |= cls == StructuredSinkWriteClass::OffBand;
            }
        }
        add_pair_counter(saw_near, saw_off_band);
    }

    template <typename H3>
    MUDA_DEVICE void add_abd_fem(const VertexMap& lhs,
                                 const VertexMap& rhs,
                                 const H3&        H,
                                 bool mirror_diag_block = false) const noexcept
    {
        using Alu       = ActivePolicy::AluScalar;
        bool saw_near     = false;
        bool saw_off_band = false;
#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
            const IndexT comp = abd_component(r);
            const Alu    wr   = abd_weight(lhs.J, r);
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                const StoreT value = static_cast<StoreT>(wr * static_cast<Alu>(H(comp, c)));
                const IndexT row = lhs.old_dof + r;
                const IndexT col = rhs.old_dof + c;
                const auto   cls = add_scalar_counted(row, col, value);
                if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
                   && row != col)
                    add_scalar_counted(col, row, value);
                saw_near |= cls == StructuredSinkWriteClass::Diag
                            || cls == StructuredSinkWriteClass::FirstOffdiag;
                saw_off_band |= cls == StructuredSinkWriteClass::OffBand;
            }
        }
        add_pair_counter(saw_near, saw_off_band);
    }

    template <typename H3>
    MUDA_DEVICE void add_fem_abd(const VertexMap& lhs,
                                 const VertexMap& rhs,
                                 const H3&        H,
                                 bool mirror_diag_block = false) const noexcept
    {
        using Alu       = ActivePolicy::AluScalar;
        bool saw_near     = false;
        bool saw_off_band = false;
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                const IndexT comp = abd_component(c);
                const Alu    wc   = abd_weight(rhs.J, c);
                const StoreT value =
                    static_cast<StoreT>(static_cast<Alu>(H(r, comp)) * wc);
                const IndexT row = lhs.old_dof + r;
                const IndexT col = rhs.old_dof + c;
                const auto cls = add_scalar_counted(row, col, value);
                if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
                   && row != col)
                    add_scalar_counted(col, row, value);
                saw_near |= cls == StructuredSinkWriteClass::Diag
                            || cls == StructuredSinkWriteClass::FirstOffdiag;
                saw_off_band |= cls == StructuredSinkWriteClass::OffBand;
            }
        }
        add_pair_counter(saw_near, saw_off_band);
    }

    template <typename H3>
    MUDA_DEVICE void add_abd_abd(const VertexMap& lhs,
                                 const VertexMap& rhs,
                                 IndexT           global_i,
                                 IndexT           global_j,
                                 const H3&        H,
                                 bool mirror_diag_block = false) const noexcept
    {
        using Alu       = ActivePolicy::AluScalar;
        if(lhs.body == rhs.body)
        {
            auto H12 = ABDJacobi::JT_H_J_t<Alu>(
                lhs.J.T(),
                H.template cast<Alu>(),
                rhs.J);
            if(global_i != global_j)
            {
                H12 += ABDJacobi::JT_H_J_t<Alu>(
                    rhs.J.T(),
                    H.template cast<Alu>().transpose(),
                    lhs.J);
            }

            bool saw_near     = false;
            bool saw_off_band = false;
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
                            const IndexT old_i = lhs.old_dof + local_i;
                            const IndexT old_j = lhs.old_dof + local_j;
                            const StoreT value_store =
                                static_cast<StoreT>(H12(local_i, local_j));
                            const auto cls = add_scalar_counted(
                                old_i,
                                old_j,
                                value_store);
                            if(row_block != col_block
                               && cls == StructuredSinkWriteClass::Diag)
                            {
                                add_scalar_counted(old_j, old_i, value_store);
                            }
                            saw_near |= cls == StructuredSinkWriteClass::Diag
                                        || cls == StructuredSinkWriteClass::FirstOffdiag;
                            saw_off_band |= cls == StructuredSinkWriteClass::OffBand;
                        }
                    }
                }
            }
            add_pair_counter(saw_near, saw_off_band);
            return;
        }

        bool saw_near     = false;
        bool saw_off_band = false;
#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
            const IndexT comp_r = abd_component(r);
            const Alu    wr     = abd_weight(lhs.J, r);
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                const IndexT comp_c = abd_component(c);
                const Alu    wc     = abd_weight(rhs.J, c);
                Alu value = wr * static_cast<Alu>(H(comp_r, comp_c)) * wc;

                const IndexT row = lhs.old_dof + r;
                const IndexT col = rhs.old_dof + c;
                const StoreT value_store = static_cast<StoreT>(value);
                const auto cls = add_scalar_counted(row, col, value_store);
                if(mirror_diag_block && cls == StructuredSinkWriteClass::Diag
                   && row != col)
                    add_scalar_counted(col, row, value_store);
                saw_near |= cls == StructuredSinkWriteClass::Diag
                            || cls == StructuredSinkWriteClass::FirstOffdiag;
                saw_off_band |= cls == StructuredSinkWriteClass::OffBand;
            }
        }
        add_pair_counter(saw_near, saw_off_band);
    }

    template <typename H3>
    MUDA_DEVICE void write_hessian(IndexT global_vertex, const H3& H) const noexcept
    {
        write_hessian_block(global_vertex, global_vertex, H);
    }

    MUDA_DEVICE bool upper_lr(IndexT left_value,
                              IndexT right_value,
                              IndexT left_slot,
                              IndexT right_slot,
                              IndexT& L,
                              IndexT& R) const noexcept
    {
        if(left_value < right_value)
        {
            L = left_slot;
            R = right_slot;
            return false;
        }

        L = right_slot;
        R = left_slot;
        return left_slot != right_slot;
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_hessian_half(
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const HMat& H) const noexcept
    {
#pragma unroll
        for(IndexT row_block = 0; row_block < StencilSize; ++row_block)
        {
#pragma unroll
            for(IndexT col_block = row_block; col_block < StencilSize; ++col_block)
            {
                IndexT L = row_block;
                IndexT R = col_block;
                const bool swapped = upper_lr(indices(row_block),
                                              indices(col_block),
                                              row_block,
                                              col_block,
                                              L,
                                              R);
                write_hessian_block(
                    indices(L),
                    indices(R),
                    H.template block<3, 3>(L * 3, R * 3),
                    indices(L) != indices(R) || swapped);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void write_hessian_block(IndexT global_i,
                                         IndexT global_j,
                                         const H3& H3x3,
                                         bool mirror_diag_block = false) const noexcept
    {
        if(!valid())
            return;

        const auto lhs = map_vertex(global_i);
        const auto rhs = map_vertex(global_j);
        if(lhs.kind == VertexMap::None || rhs.kind == VertexMap::None)
            return;
        if(lhs.fixed || rhs.fixed)
            return;

        if(lhs.kind == VertexMap::Fem && rhs.kind == VertexMap::Fem)
        {
            add_fem_fem(lhs.old_dof, rhs.old_dof, H3x3, mirror_diag_block);
            return;
        }

        if(lhs.kind == VertexMap::Abd && rhs.kind == VertexMap::Fem)
        {
            add_abd_fem(lhs, rhs, H3x3, mirror_diag_block);
            return;
        }

        if(lhs.kind == VertexMap::Fem && rhs.kind == VertexMap::Abd)
        {
            add_fem_abd(lhs, rhs, H3x3, mirror_diag_block);
            return;
        }

        add_abd_abd(lhs, rhs, global_i, global_j, H3x3, mirror_diag_block);
    }
};
}  // namespace uipc::backend::cuda_mixed
