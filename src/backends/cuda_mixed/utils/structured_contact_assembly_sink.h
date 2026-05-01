#pragma once

#include <affine_body/abd_jacobi_matrix.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>
#include <muda/atomic.h>

namespace uipc::backend::cuda_mixed
{
struct StructuredContactVertexSlot
{
    enum Kind : IndexT
    {
        None = 0,
        Abd  = 1,
        Fem  = 2,
    };

    Kind   kind         = None;
    IndexT old_dof      = -1;
    IndexT body         = -1;
    IndexT local_vertex = -1;
    IndexT fixed        = 0;
};

struct StructuredContactHalfBlockPlan
{
    StructuredContactVertexSlot lhs;
    StructuredContactVertexSlot rhs;
    IndexT                      global_i = -1;
    IndexT                      global_j = -1;
    IndexT                      h_l      = 0;
    IndexT                      h_r      = 0;
    IndexT                      mirror_diag_block = 0;
    IndexT                      valid = 0;
};

template <typename StoreT, typename SolveT>
struct StructuredContactAssemblySink
{
    StructuredDeviceAssemblySink<StoreT, SolveT> sink;

    muda::CBufferView<StructuredContactVertexSlot> vertex_slots;

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

    MUDA_GENERIC bool topology_probe_only() const noexcept
    {
        return sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only
               && sink.runtime_ordering.topology_only;
    }

    MUDA_DEVICE VertexMap slot_to_vertex_map(
        const StructuredContactVertexSlot& slot) const noexcept
    {
        VertexMap mapped;
        if(slot.kind == StructuredContactVertexSlot::None)
            return mapped;
        if(slot.fixed)
            mapped.fixed = true;

        mapped.old_dof      = slot.old_dof;
        mapped.body         = slot.body;
        mapped.local_vertex = slot.local_vertex;
        mapped.fixed        = slot.fixed != 0;
        if(slot.kind == StructuredContactVertexSlot::Fem)
        {
            mapped.kind = VertexMap::Fem;
            return mapped;
        }

        mapped.kind = VertexMap::Abd;
        if(abd_vertex_to_J.data() == nullptr || slot.local_vertex < 0
           || slot.local_vertex >= abd_vertex_to_J.size())
        {
            mapped.kind = VertexMap::None;
            return mapped;
        }
        mapped.J = abd_vertex_to_J.data()[slot.local_vertex];
        return mapped;
    }

    static MUDA_DEVICE StructuredContactVertexSlot vertex_slot_from_map(
        const VertexMap& mapped) noexcept
    {
        StructuredContactVertexSlot slot;
        if(mapped.kind == VertexMap::None)
            return slot;
        slot.kind = mapped.kind == VertexMap::Abd
                        ? StructuredContactVertexSlot::Abd
                        : StructuredContactVertexSlot::Fem;
        slot.old_dof      = mapped.old_dof;
        slot.body         = mapped.body;
        slot.local_vertex = mapped.local_vertex;
        slot.fixed        = mapped.fixed ? 1 : 0;
        return slot;
    }

    MUDA_DEVICE VertexMap map_vertex_slow(IndexT global_vertex) const noexcept
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

    MUDA_DEVICE VertexMap map_vertex(IndexT global_vertex) const noexcept
    {
        if(vertex_slots.data() != nullptr && global_vertex >= 0
           && global_vertex < vertex_slots.size())
        {
            const auto slot = vertex_slots[global_vertex];
            if(slot.kind != StructuredContactVertexSlot::None)
                return slot_to_vertex_map(slot);
        }
        return map_vertex_slow(global_vertex);
    }

    MUDA_DEVICE StructuredContactVertexSlot vertex_slot(IndexT global_vertex) const noexcept
    {
        if(vertex_slots.data() != nullptr && global_vertex >= 0
           && global_vertex < vertex_slots.size())
        {
            const auto slot = vertex_slots[global_vertex];
            if(slot.kind != StructuredContactVertexSlot::None)
                return slot;
        }
        return vertex_slot_from_map(map_vertex_slow(global_vertex));
    }

    MUDA_DEVICE void add_counter(StructuredSinkWriteClass cls) const noexcept
    {
        if(counters.data() == nullptr || counters.size() < 3)
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
        if(counters.data() == nullptr || counters.size() < 5)
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
#pragma unroll 1
        for(IndexT lhs_atom = 0; lhs_atom < 4; ++lhs_atom)
        {
            if(lhs_atom >= lhs_atom_count)
                continue;
#pragma unroll 1
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
#pragma unroll 1
        for(IndexT lhs_atom = 0; lhs_atom < 4; ++lhs_atom)
        {
            if(lhs_atom >= lhs_atom_count)
                continue;
#pragma unroll 1
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

        if(v.kind == VertexMap::Fem)
        {
            add_fem_fem(v.old_dof, v.old_dof, H);
            return;
        }

        add_abd_diag_hessian(v, H);
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

    MUDA_DEVICE StructuredContactHalfBlockPlan make_half_block_plan(
        IndexT global_i,
        IndexT global_j,
        IndexT h_l,
        IndexT h_r,
        bool   mirror_diag_block) const noexcept
    {
        StructuredContactHalfBlockPlan plan;
        plan.global_i          = global_i;
        plan.global_j          = global_j;
        plan.h_l               = h_l;
        plan.h_r               = h_r;
        plan.mirror_diag_block = mirror_diag_block ? 1 : 0;

        const auto lhs = vertex_slot(global_i);
        const auto rhs = vertex_slot(global_j);
        if(lhs.kind == StructuredContactVertexSlot::None
           || rhs.kind == StructuredContactVertexSlot::None)
            return plan;
        if(lhs.fixed || rhs.fixed)
            return plan;

        plan.lhs   = lhs;
        plan.rhs   = rhs;
        plan.valid = 1;
        return plan;
    }

    template <int StencilSize, typename PlanView>
    MUDA_DEVICE void build_hessian_half_plan(
        const Eigen::Vector<IndexT, StencilSize>& indices,
        PlanView                                  plans,
        IndexT                                    plan_offset) const noexcept
    {
        IndexT pair = 0;
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
                plans(plan_offset + pair) = make_half_block_plan(
                    indices(L),
                    indices(R),
                    L,
                    R,
                    indices(L) != indices(R) || swapped);
                ++pair;
            }
        }
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

    template <typename H3>
    MUDA_DEVICE void write_contact_half_block(IndexT global_i,
                                              IndexT global_j,
                                              const H3& H3x3,
                                              bool mirror_diag_block = false) const noexcept
    {
        write_hessian_block(global_i, global_j, H3x3, mirror_diag_block);
    }

    template <typename H3>
    MUDA_DEVICE void write_contact_plan_block(
        const StructuredContactHalfBlockPlan& plan,
        const H3&                             H3x3) const noexcept
    {
        if(!plan.valid || !valid())
            return;

        const auto lhs = slot_to_vertex_map(plan.lhs);
        const auto rhs = slot_to_vertex_map(plan.rhs);
        if(lhs.kind == VertexMap::None || rhs.kind == VertexMap::None)
            return;
        if(lhs.fixed || rhs.fixed)
            return;

        if(sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only)
        {
            if(sink.runtime_ordering.topology_only)
                record_topology_pair(lhs, rhs);
            else
                record_weighted_pair(lhs, rhs, h3_abs_sum(H3x3));
            return;
        }

        const bool mirror_diag_block = plan.mirror_diag_block != 0;
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

        add_abd_abd_half(lhs, rhs, plan.global_i, plan.global_j, H3x3, mirror_diag_block);
    }

    template <int StencilSize, typename PlanView, typename HMat>
    MUDA_DEVICE void write_hessian_half_with_plan(PlanView    plans,
                                                  IndexT      plan_offset,
                                                  const HMat& H) const noexcept
    {
        constexpr IndexT PairCount = StencilSize * (StencilSize + 1) / 2;
#pragma unroll
        for(IndexT pair = 0; pair < PairCount; ++pair)
        {
            const auto plan = plans(plan_offset + pair);
            write_contact_plan_block(
                plan,
                H.template block<3, 3>(plan.h_l * 3, plan.h_r * 3));
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

        if(sink.runtime_ordering.valid() && sink.runtime_ordering.graph_only)
        {
            if(sink.runtime_ordering.topology_only)
                record_topology_pair(lhs, rhs);
            else
                record_weighted_pair(lhs, rhs, h3_abs_sum(H3x3));
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
};
}  // namespace uipc::backend::cuda_mixed
