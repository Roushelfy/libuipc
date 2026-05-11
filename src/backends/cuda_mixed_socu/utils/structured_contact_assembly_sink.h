#pragma once

#include <affine_body/abd_jacobi_matrix.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>
#include <utils/structured_assembly_counters.h>
#include <utils/structured_contact_hessian_cache.h>
#include <utils/structured_contact_offband_policy.h>
#include <cuda_runtime_api.h>
#include <muda/atomic.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT, typename SolveT>
struct StructuredContactAssemblySink
{
    StructuredDeviceAssemblySink<StoreT, SolveT> sink;

    IndexT abd_vertex_offset = -1;
    IndexT abd_vertex_count  = 0;
    IndexT abd_body_count    = 0;
    IndexT abd_old_dof_offset = -1;
    muda::CBufferView<IndexT>    abd_vertex_to_body;
    muda::CBufferView<ABDJacobi> abd_vertex_to_J;
    muda::CBufferView<IndexT>    abd_body_is_fixed;

    IndexT fem_vertex_offset = -1;
    IndexT fem_vertex_count  = 0;
    IndexT fem_old_dof_offset = -1;
    muda::CBufferView<IndexT> fem_vertex_is_fixed;

    // Shared StructuredAssemblyCounterSlot buffer. Contact currently owns the
    // first seven slots; native chain/base instrumentation starts after them.
    muda::BufferView<IndexT> counters;
    StructuredContactHessianCache<StoreT> hessian_cache;
    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;

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

    MUDA_GENERIC bool topology_probe_only() const noexcept
    {
        return sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only
               && sink.runtime_ordering.topology_only;
    }

    MUDA_GENERIC bool approximate_weight_probe_only() const noexcept
    {
        return sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only
               && sink.runtime_ordering.approximate_weight;
    }

    MUDA_DEVICE VertexMap map_vertex(IndexT global_vertex) const noexcept
    {
        VertexMap mapped;
        if(abd_vertex_offset >= 0 && global_vertex >= abd_vertex_offset
           && global_vertex < abd_vertex_offset + abd_vertex_count)
        {
            const IndexT local = global_vertex - abd_vertex_offset;
            if(abd_vertex_to_body.data() == nullptr || abd_vertex_to_J.data() == nullptr
               || local < 0 || local >= abd_vertex_to_body.size()
               || local >= abd_vertex_to_J.size())
                return mapped;
            const IndexT body  = abd_vertex_to_body.data()[local];
            if(abd_body_is_fixed.data() == nullptr || body < 0 || body >= abd_body_count
               || body >= abd_body_is_fixed.size())
                return mapped;
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
            if(fem_vertex_is_fixed.data() == nullptr || local < 0
               || local >= fem_vertex_is_fixed.size())
                return mapped;
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
        switch(cls)
        {
            case StructuredSinkWriteClass::Diag:
                add_counter_slot(
                    StructuredAssemblyCounterSlot::ContactDiagScalarWrite);
                break;
            case StructuredSinkWriteClass::FirstOffdiag:
                add_counter_slot(StructuredAssemblyCounterSlot::
                                     ContactFirstOffdiagScalarWrite);
                break;
            case StructuredSinkWriteClass::OffBand:
                add_counter_slot(
                    StructuredAssemblyCounterSlot::ContactOffBandScalarDrop);
                break;
            case StructuredSinkWriteClass::Skipped:
            default:
                break;
        }
    }

    MUDA_DEVICE void add_counter_slot(StructuredAssemblyCounterSlot slot) const noexcept
    {
        const IndexT index = static_cast<IndexT>(slot);
        if(counters.data() == nullptr || index < 0
           || static_cast<SizeT>(index) >= counters.size())
            return;
        muda::atomic_add(counters.data(static_cast<SizeT>(index)), IndexT{1});
    }

    MUDA_DEVICE void add_pair_counter(bool saw_near, bool saw_off_band) const noexcept
    {
        if(saw_off_band)
            add_counter_slot(StructuredAssemblyCounterSlot::ContactOffBandPair);
        else if(saw_near)
            add_counter_slot(StructuredAssemblyCounterSlot::ContactNearBandPair);
    }

    MUDA_DEVICE void add_diag_fallback_counter() const noexcept
    {
        add_counter_slot(
            StructuredAssemblyCounterSlot::ContactOffBandDiagFallbackStencil);
    }

    MUDA_DEVICE void add_lump_fallback_counter() const noexcept
    {
        add_counter_slot(
            StructuredAssemblyCounterSlot::ContactOffBandLumpFallbackStencil);
    }

    MUDA_GENERIC bool matrix_offband_policy_active() const noexcept
    {
        return offband_policy != StructuredContactOffbandPolicy::Drop
               && !(sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only);
    }

    MUDA_DEVICE StructuredSinkWriteClass add_scalar_counted(IndexT old_row,
                                                            IndexT old_col,
                                                            StoreT value) const noexcept
    {
        const auto cls = sink.add_hessian_scalar_status(old_row, old_col, value);
        add_counter(cls);
        return cls;
    }

    MUDA_DEVICE void record_topology_pair(const VertexMap& lhs,
                                          const VertexMap& rhs) const noexcept
    {
        if(!valid() || !sink.runtime_ordering.valid())
            return;
        if(lhs.kind == VertexMap::None || rhs.kind == VertexMap::None)
            return;
        if(lhs.fixed || rhs.fixed)
            return;

        const IndexT lhs_atom_count = lhs.kind == VertexMap::Abd ? 4 : 1;
        const IndexT rhs_atom_count = rhs.kind == VertexMap::Abd ? 4 : 1;
#pragma unroll 1  // suppress unroll: register pressure with up to 4x4 atom loops
        for(IndexT lhs_atom = 0; lhs_atom < 4; ++lhs_atom)
        {
            if(lhs_atom >= lhs_atom_count)
                continue;
#pragma unroll 1  // same reason
            for(IndexT rhs_atom = 0; rhs_atom < 4; ++rhs_atom)
            {
                if(rhs_atom >= rhs_atom_count)
                    continue;
                sink.record_runtime_ordering_edge(lhs.old_dof + lhs_atom * 3,
                                                  rhs.old_dof + rhs_atom * 3,
                                                  StoreT{1});
            }
        }
    }

    MUDA_DEVICE void record_weighted_pair(const VertexMap& lhs,
                                          const VertexMap& rhs,
                                          StoreT           weight) const noexcept
    {
        if(!valid() || !sink.runtime_ordering.valid())
            return;
        if(lhs.kind == VertexMap::None || rhs.kind == VertexMap::None)
            return;
        if(lhs.fixed || rhs.fixed)
            return;
        if(static_cast<double>(weight) == 0.0)
            return;

        const IndexT lhs_atom_count = lhs.kind == VertexMap::Abd ? 4 : 1;
        const IndexT rhs_atom_count = rhs.kind == VertexMap::Abd ? 4 : 1;
#pragma unroll 1  // suppress unroll: register pressure with up to 4x4 atom loops
        for(IndexT lhs_atom = 0; lhs_atom < 4; ++lhs_atom)
        {
            if(lhs_atom >= lhs_atom_count)
                continue;
#pragma unroll 1  // same reason
            for(IndexT rhs_atom = 0; rhs_atom < 4; ++rhs_atom)
            {
                if(rhs_atom >= rhs_atom_count)
                    continue;
                sink.record_runtime_ordering_edge(lhs.old_dof + lhs_atom * 3,
                                                  rhs.old_dof + rhs_atom * 3,
                                                  weight);
            }
        }
    }

    MUDA_DEVICE void record_topology_pair(IndexT global_i,
                                          IndexT global_j) const noexcept
    {
        const auto lhs = map_vertex(global_i);
        const auto rhs = map_vertex(global_j);
        record_topology_pair(lhs, rhs);
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
    static MUDA_DEVICE StoreT h3_abs_sum(const H3& H) noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        Alu sum   = 0;
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                const Alu value = static_cast<Alu>(H(r, c));
                sum += value < Alu{0} ? -value : value;
            }
        }
        return static_cast<StoreT>(sum);
    }

    MUDA_DEVICE bool scalar_pair_offband(IndexT old_i, IndexT old_j) const noexcept
    {
        return sink.matrix.classify_dof_pair(old_i, old_j)
               == StructuredSinkWriteClass::OffBand;
    }

    MUDA_DEVICE bool fem_fem_has_offband(IndexT old_row,
                                         IndexT old_col) const noexcept
    {
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                if(scalar_pair_offband(old_row + r, old_col + c))
                    return true;
            }
        }
        return false;
    }

    MUDA_DEVICE bool abd_fem_has_offband(const VertexMap& lhs,
                                         const VertexMap& rhs) const noexcept
    {
#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
            {
                if(scalar_pair_offband(lhs.old_dof + r, rhs.old_dof + c))
                    return true;
            }
        }
        return false;
    }

    MUDA_DEVICE bool fem_abd_has_offband(const VertexMap& lhs,
                                         const VertexMap& rhs) const noexcept
    {
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                if(scalar_pair_offband(lhs.old_dof + r, rhs.old_dof + c))
                    return true;
            }
        }
        return false;
    }

    MUDA_DEVICE bool abd_same_body_half_has_offband(IndexT old_dof) const noexcept
    {
#pragma unroll 1
        for(IndexT row_block = 0; row_block < 4; ++row_block)
        {
#pragma unroll 1
            for(IndexT col_block = row_block; col_block < 4; ++col_block)
            {
#pragma unroll
                for(IndexT row = 0; row < 3; ++row)
                {
#pragma unroll
                    for(IndexT col = 0; col < 3; ++col)
                    {
                        const IndexT old_i = old_dof + row_block * 3 + row;
                        const IndexT old_j = old_dof + col_block * 3 + col;
                        if(scalar_pair_offband(old_i, old_j))
                            return true;
                    }
                }
            }
        }
        return false;
    }

    MUDA_DEVICE bool abd_abd_projected_has_offband(IndexT old_row,
                                                   IndexT old_col) const noexcept
    {
#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                if(scalar_pair_offband(old_row + r, old_col + c))
                    return true;
            }
        }
        return false;
    }

    MUDA_DEVICE bool single_vertex_hessian_has_offband(
        const VertexMap& v) const noexcept
    {
        if(v.kind == VertexMap::Fem)
            return fem_fem_has_offband(v.old_dof, v.old_dof);
        if(v.kind == VertexMap::Abd)
            return abd_same_body_half_has_offband(v.old_dof);
        return false;
    }

    MUDA_DEVICE bool hessian_block_has_offband(IndexT global_i,
                                               IndexT global_j) const noexcept
    {
        const auto lhs = map_vertex(global_i);
        const auto rhs = map_vertex(global_j);
        if(lhs.kind == VertexMap::None || rhs.kind == VertexMap::None)
            return false;
        if(lhs.fixed || rhs.fixed)
            return false;

        if(lhs.kind == VertexMap::Fem && rhs.kind == VertexMap::Fem)
            return fem_fem_has_offband(lhs.old_dof, rhs.old_dof);
        if(lhs.kind == VertexMap::Abd && rhs.kind == VertexMap::Fem)
            return abd_fem_has_offband(lhs, rhs);
        if(lhs.kind == VertexMap::Fem && rhs.kind == VertexMap::Abd)
            return fem_abd_has_offband(lhs, rhs);
        if(lhs.body == rhs.body)
            return abd_same_body_half_has_offband(lhs.old_dof);
        return abd_abd_projected_has_offband(lhs.old_dof, rhs.old_dof);
    }

    template <int StencilSize>
    MUDA_DEVICE bool stencil_has_offband(
        const Eigen::Vector<IndexT, StencilSize>& indices) const noexcept
    {
#pragma unroll 1
        for(IndexT row_block = 0; row_block < StencilSize; ++row_block)
        {
#pragma unroll 1
            for(IndexT col_block = row_block; col_block < StencilSize; ++col_block)
            {
                IndexT L = row_block;
                IndexT R = col_block;
                if(indices(row_block) > indices(col_block))
                {
                    L = col_block;
                    R = row_block;
                }
                if(hessian_block_has_offband(indices(L), indices(R)))
                    return true;
            }
        }
        return false;
    }

    template <typename H3>
    MUDA_DEVICE void write_vertex_exact_scalar_diag(const VertexMap& v,
                                                    const H3&        H) const noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        if(v.kind == VertexMap::Fem)
        {
#pragma unroll
            for(IndexT r = 0; r < 3; ++r)
            {
                add_scalar_counted(v.old_dof + r,
                                   v.old_dof + r,
                                   static_cast<StoreT>(H(r, r)));
            }
            return;
        }

        if(v.kind != VertexMap::Abd)
            return;

#pragma unroll 1
        for(IndexT q = 0; q < 12; ++q)
        {
            const IndexT comp = abd_component(q);
            const Alu    w    = abd_weight(v.J, q);
            const StoreT value = static_cast<StoreT>(
                w * static_cast<Alu>(H(comp, comp)) * w);
            add_scalar_counted(v.old_dof + q, v.old_dof + q, value);
        }
    }

    template <typename H3>
    MUDA_DEVICE void write_vertex_exact_diag_block(IndexT global_vertex,
                                                   const H3& H) const noexcept
    {
        const auto v = map_vertex(global_vertex);
        if(v.kind == VertexMap::None || v.fixed)
            return;

        if(!single_vertex_hessian_has_offband(v))
        {
            if(v.kind == VertexMap::Fem)
                add_fem_fem(v.old_dof, v.old_dof, H);
            else
                add_abd_diag_hessian(v, H);
            return;
        }

        write_vertex_exact_scalar_diag(v, H);
    }

    MUDA_DEVICE void write_vertex_lumped_scalar_diag(
        IndexT                    global_vertex,
        ActivePolicy::AluScalar   lump_x,
        ActivePolicy::AluScalar   lump_y,
        ActivePolicy::AluScalar   lump_z) const noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        const auto v = map_vertex(global_vertex);
        if(v.kind == VertexMap::None || v.fixed)
            return;

        if(v.kind == VertexMap::Fem)
        {
            add_scalar_counted(v.old_dof + 0, v.old_dof + 0, static_cast<StoreT>(lump_x));
            add_scalar_counted(v.old_dof + 1, v.old_dof + 1, static_cast<StoreT>(lump_y));
            add_scalar_counted(v.old_dof + 2, v.old_dof + 2, static_cast<StoreT>(lump_z));
            return;
        }

        const Alu lumps[3] = {lump_x, lump_y, lump_z};
#pragma unroll 1
        for(IndexT q = 0; q < 12; ++q)
        {
            const IndexT comp = abd_component(q);
            const Alu    w    = abd_weight(v.J, q);
            const StoreT value =
                static_cast<StoreT>(w * lumps[comp] * w);
            add_scalar_counted(v.old_dof + q, v.old_dof + q, value);
        }
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
    MUDA_DEVICE Eigen::Matrix<ActivePolicy::AluScalar, 12, 12>
    project_abd_abd_half(const VertexMap& lhs,
                         const VertexMap& rhs,
                         IndexT           global_i,
                         IndexT           global_j,
                         const H3&        H) const noexcept
    {
        using Alu = ActivePolicy::AluScalar;
        const Eigen::Matrix<Alu, 3, 12> J_i_mat = lhs.J.template to_mat_t<Alu>();
        const Eigen::Matrix<Alu, 3, 12> J_j_mat = rhs.J.template to_mat_t<Alu>();
        const Eigen::Matrix<Alu, 3, 3>  H3x3_alu = H.template cast<Alu>();

        if(lhs.body < rhs.body)
            return (J_i_mat.transpose() * H3x3_alu * J_j_mat).eval();
        if(lhs.body > rhs.body)
            return (J_j_mat.transpose() * H3x3_alu.transpose() * J_i_mat).eval();

        if(global_i != global_j)
        {
            return (J_i_mat.transpose() * H3x3_alu * J_j_mat).eval()
                   + (J_j_mat.transpose() * H3x3_alu.transpose() * J_i_mat)
                         .eval();
        }
        return (J_i_mat.transpose() * H3x3_alu * J_j_mat).eval();
    }

    template <typename H12>
    MUDA_DEVICE void add_abd_same_body_half_projected(IndexT    old_dof,
                                                      const H12& projected) const noexcept
    {
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
                        const IndexT old_i   = old_dof + local_i;
                        const IndexT old_j   = old_dof + local_j;
                        const StoreT value_store =
                            static_cast<StoreT>(projected(local_i, local_j));
                        const auto cls =
                            add_scalar_counted(old_i, old_j, value_store);
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
    }

    template <typename H12>
    MUDA_DEVICE void add_abd_abd_projected(IndexT    old_row,
                                           IndexT    old_col,
                                           const H12& projected,
                                           bool mirror_diag_block = false) const noexcept
    {
        bool saw_near     = false;
        bool saw_off_band = false;
#pragma unroll 1
        for(IndexT r = 0; r < 12; ++r)
        {
#pragma unroll 1
            for(IndexT c = 0; c < 12; ++c)
            {
                const StoreT value_store = static_cast<StoreT>(projected(r, c));
                const IndexT row         = old_row + r;
                const IndexT col         = old_col + c;
                const auto   cls = add_scalar_counted(row, col, value_store);
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
    MUDA_DEVICE void add_abd_diag_hessian(const VertexMap& v, const H3& H) const noexcept
    {
        using Alu       = ActivePolicy::AluScalar;
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
                    const IndexT local_i = row_block * 3 + row;
                    const IndexT comp_i  = abd_component(local_i);
                    const Alu    wi      = abd_weight(v.J, local_i);
#pragma unroll
                    for(IndexT col = 0; col < 3; ++col)
                    {
                        const IndexT local_j = col_block * 3 + col;
                        const IndexT comp_j  = abd_component(local_j);
                        const Alu    wj      = abd_weight(v.J, local_j);
                        const IndexT old_i   = v.old_dof + local_i;
                        const IndexT old_j   = v.old_dof + local_j;
                        const StoreT value_store = static_cast<StoreT>(
                            wi * static_cast<Alu>(H(comp_i, comp_j)) * wj);
                        const auto cls =
                            add_scalar_counted(old_i, old_j, value_store);
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
    }

    template <typename H3>
    MUDA_DEVICE void add_abd_abd_half(const VertexMap& lhs,
                                      const VertexMap& rhs,
                                      IndexT           global_i,
                                      IndexT           global_j,
                                      const H3&        H,
                                      bool mirror_diag_block = false) const noexcept
    {
        const auto H12 = project_abd_abd_half(lhs, rhs, global_i, global_j, H);
        if(lhs.body == rhs.body)
        {
            add_abd_same_body_half_projected(lhs.old_dof, H12);
            return;
        }

        if(lhs.body < rhs.body)
            add_abd_abd_projected(lhs.old_dof, rhs.old_dof, H12, mirror_diag_block);
        else
            add_abd_abd_projected(rhs.old_dof, lhs.old_dof, H12, mirror_diag_block);
    }

    template <typename H3>
    MUDA_DEVICE void write_hessian(IndexT global_vertex, const H3& H) const noexcept
    {
        if(!valid())
            return;

        const auto v = map_vertex(global_vertex);
        if(v.kind == VertexMap::None || v.fixed)
            return;

        if(sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only
           && !sink.runtime_ordering.topology_only)
        {
            append_hessian_cache(global_vertex, global_vertex, H, false);
        }

        if(matrix_offband_policy_active() && single_vertex_hessian_has_offband(v))
        {
            if(offband_policy == StructuredContactOffbandPolicy::Diag)
            {
                add_diag_fallback_counter();
                write_vertex_exact_scalar_diag(v, H);
                return;
            }

            if(offband_policy == StructuredContactOffbandPolicy::DiagLump)
            {
                using Alu = ActivePolicy::AluScalar;
                Alu lump[3] = {Alu{0}, Alu{0}, Alu{0}};
#pragma unroll
                for(IndexT r = 0; r < 3; ++r)
                {
#pragma unroll
                    for(IndexT c = 0; c < 3; ++c)
                    {
                        const Alu value = static_cast<Alu>(H(r, c));
                        lump[r] += value < Alu{0} ? -value : value;
                    }
                }
                add_lump_fallback_counter();
                write_vertex_lumped_scalar_diag(global_vertex,
                                                lump[0],
                                                lump[1],
                                                lump[2]);
                return;
            }
        }

        if(v.kind == VertexMap::Fem)
        {
            add_fem_fem(v.old_dof, v.old_dof, H);
            return;
        }

        add_abd_diag_hessian(v, H);
    }

    MUDA_GENERIC bool upper_lr(IndexT left_value,
                              IndexT right_value,
                              IndexT left_slot,
                              IndexT right_slot,
                              IndexT& L,
                              IndexT& R) const noexcept
    {
        if(left_value <= right_value)
        {
            L = left_slot;
            R = right_slot;
            return false;
        }

        L = right_slot;
        R = left_slot;
        return true;
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_hessian_half_offband_fallback(
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const HMat& H) const noexcept
    {
        if(offband_policy == StructuredContactOffbandPolicy::Diag)
        {
            add_diag_fallback_counter();
#pragma unroll 1
            for(IndexT k = 0; k < StencilSize; ++k)
            {
                write_vertex_exact_diag_block(
                    indices(k),
                    H.template block<3, 3>(k * 3, k * 3));
            }
            return;
        }

        if(offband_policy != StructuredContactOffbandPolicy::DiagLump)
            return;

        using Alu = ActivePolicy::AluScalar;
        add_lump_fallback_counter();
#pragma unroll 1
        for(IndexT k = 0; k < StencilSize; ++k)
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
                            static_cast<Alu>(H(k * 3 + r, j * 3 + c));
                        lump[r] += value < Alu{0} ? -value : value;
                    }
                }
            }
            write_vertex_lumped_scalar_diag(indices(k),
                                            lump[0],
                                            lump[1],
                                            lump[2]);
        }
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_hessian_half(
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const HMat& H) const noexcept
    {
        if(matrix_offband_policy_active() && stencil_has_offband(indices))
        {
            write_hessian_half_offband_fallback(indices, H);
            return;
        }

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
                write_contact_half_block(
                    indices(L),
                    indices(R),
                    H.template block<3, 3>(L * 3, R * 3),
                    indices(L) != indices(R) || swapped);
            }
        }
    }

    template <int StencilSize>
    MUDA_DEVICE void write_topology_half(
        const Eigen::Vector<IndexT, StencilSize>& indices) const noexcept
    {
#pragma unroll
        for(IndexT row_block = 0; row_block < StencilSize; ++row_block)
        {
#pragma unroll
            for(IndexT col_block = row_block; col_block < StencilSize; ++col_block)
            {
                IndexT L = row_block;
                IndexT R = col_block;
                upper_lr(indices(row_block),
                         indices(col_block),
                         row_block,
                         col_block,
                         L,
                         R);
                record_topology_pair(indices(L), indices(R));
            }
        }
    }

    MUDA_DEVICE void write_weighted_hessian(IndexT global_vertex,
                                            StoreT weight) const noexcept
    {
        const auto v = map_vertex(global_vertex);
        record_weighted_pair(v, v, weight);
    }

    template <int StencilSize>
    MUDA_DEVICE void write_weighted_half(
        const Eigen::Vector<IndexT, StencilSize>& indices,
        StoreT                                    weight) const noexcept
    {
#pragma unroll
        for(IndexT row_block = 0; row_block < StencilSize; ++row_block)
        {
#pragma unroll
            for(IndexT col_block = row_block; col_block < StencilSize; ++col_block)
            {
                IndexT L = row_block;
                IndexT R = col_block;
                upper_lr(indices(row_block),
                         indices(col_block),
                         row_block,
                         col_block,
                         L,
                         R);
                const auto lhs = map_vertex(indices(L));
                const auto rhs = map_vertex(indices(R));
                record_weighted_pair(lhs, rhs, weight);
            }
        }
    }

    template <typename H3>
    MUDA_DEVICE void write_contact_half_block(IndexT global_i,
                                              IndexT global_j,
                                              const H3& H3x3,
                                              bool mirror_diag_block = false) const noexcept
    {
        write_hessian_block(global_i, global_j, H3x3, mirror_diag_block);
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

        if(sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only)
        {
            if(sink.runtime_ordering.topology_only)
            {
                record_topology_pair(lhs, rhs);
            }
            else
            {
                record_weighted_pair(lhs, rhs, h3_abs_sum(H3x3));
                append_hessian_cache(global_i, global_j, H3x3, mirror_diag_block);
            }
            return;
        }

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

        add_abd_abd_half(lhs, rhs, global_i, global_j, H3x3, mirror_diag_block);
    }

    template <typename H3>
    MUDA_DEVICE void append_hessian_cache(IndexT global_i,
                                          IndexT global_j,
                                          const H3& H3x3,
                                          bool mirror_diag_block) const noexcept
    {
        if(!hessian_cache.collect_valid())
            return;

        const IndexT slot = muda::atomic_add(hessian_cache.cursor.data(0), IndexT{1});
        if(slot < 0 || static_cast<SizeT>(slot) >= hessian_cache.records.size())
        {
            muda::atomic_add(hessian_cache.cursor.data(1), IndexT{1});
            return;
        }

        auto& record = hessian_cache.records.data()[slot];
        record.global_i = global_i;
        record.global_j = global_j;
        record.mirror_diag_block = mirror_diag_block ? IndexT{1} : IndexT{0};
#pragma unroll
        for(IndexT r = 0; r < 3; ++r)
        {
#pragma unroll
            for(IndexT c = 0; c < 3; ++c)
                record.H[r * 3 + c] = static_cast<StoreT>(H3x3(r, c));
        }
    }

};

void replay_structured_contact_hessian_cache(
    cudaStream_t stream,
    StructuredContactAssemblySink<ActivePolicy::StoreScalar, ActivePolicy::SolveScalar>
        sink);
}  // namespace uipc::backend::cuda_mixed
