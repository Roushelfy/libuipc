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
                                 const H3& H) const noexcept
    {
        bool saw_near     = false;
        bool saw_off_band = false;
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                const auto cls = add_scalar_counted(
                    old_row + r, old_col + c, static_cast<StoreT>(H(r, c)));
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
                                 const H3&        H) const noexcept
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
                const auto   cls =
                    add_scalar_counted(lhs.old_dof + r, rhs.old_dof + c, value);
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
                                 const H3&        H) const noexcept
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
                const auto cls =
                    add_scalar_counted(lhs.old_dof + r, rhs.old_dof + c, value);
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
                                 const H3&        H) const noexcept
    {
        using Alu       = ActivePolicy::AluScalar;
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
                if(lhs.body == rhs.body && global_i != global_j)
                {
                    const Alu wr_sym = abd_weight(rhs.J, r);
                    const Alu wc_sym = abd_weight(lhs.J, c);
                    value += wr_sym * static_cast<Alu>(H(comp_c, comp_r)) * wc_sym;
                }

                const auto cls = add_scalar_counted(
                    lhs.old_dof + r, rhs.old_dof + c, static_cast<StoreT>(value));
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
                write_hessian_block(
                    indices(row_block),
                    indices(col_block),
                    H.template block<3, 3>(row_block * 3, col_block * 3));
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void write_hessian_block(IndexT global_i,
                                         IndexT global_j,
                                         const H3& H3x3) const noexcept
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
            add_fem_fem(lhs.old_dof, rhs.old_dof, H3x3);
            return;
        }

        if(lhs.kind == VertexMap::Abd && rhs.kind == VertexMap::Fem)
        {
            add_abd_fem(lhs, rhs, H3x3);
            return;
        }

        if(lhs.kind == VertexMap::Fem && rhs.kind == VertexMap::Abd)
        {
            add_fem_abd(lhs, rhs, H3x3);
            return;
        }

        add_abd_abd(lhs, rhs, global_i, global_j, H3x3);
    }
};
}  // namespace uipc::backend::cuda_mixed
