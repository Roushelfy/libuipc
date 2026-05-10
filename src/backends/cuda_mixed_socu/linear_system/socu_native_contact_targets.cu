#include <linear_system/socu_native_contact_targets.h>

#include <muda/buffer/buffer_view.h>

namespace uipc::backend::cuda_mixed
{
namespace
{
MUDA_DEVICE SocuNativeScalarTarget classify_old_to_chain_pair(
    muda::CBufferView<IndexT> old_to_chain,
    SizeT                     horizon,
    SizeT                     block_size,
    IndexT                    old_i,
    IndexT                    old_j) noexcept
{
    SocuNativeScalarTarget target;
    if(block_size == 0 || old_i < 0 || old_j < 0)
        return target;
    if(static_cast<SizeT>(old_i) >= old_to_chain.size()
       || static_cast<SizeT>(old_j) >= old_to_chain.size())
        return target;

    const IndexT chain_i = old_to_chain.data()[static_cast<SizeT>(old_i)];
    const IndexT chain_j = old_to_chain.data()[static_cast<SizeT>(old_j)];
    if(chain_i < 0 || chain_j < 0)
        return target;

    const SizeT ci = static_cast<SizeT>(chain_i);
    const SizeT cj = static_cast<SizeT>(chain_j);
    const SizeT bi = ci / block_size;
    const SizeT bj = cj / block_size;
    const SizeT li = ci % block_size;
    const SizeT lj = cj % block_size;
    if(bi >= horizon || bj >= horizon)
        return target;

    if(bi == bj)
    {
        target.cls        = SocuNativeBandClass::Diag;
        target.left_block = bi;
        target.row_lane   = li;
        target.col_lane   = lj;
        return target;
    }

    const SizeT min_block = bi < bj ? bi : bj;
    const SizeT max_block = bi > bj ? bi : bj;
    if(max_block != min_block + 1)
    {
        target.cls = SocuNativeBandClass::OffBand;
        return target;
    }

    const bool ij_is_forward = bi < bj;
    target.cls        = SocuNativeBandClass::FirstOffdiag;
    target.left_block = min_block;
    target.row_lane   = ij_is_forward ? lj : li;
    target.col_lane   = ij_is_forward ? li : lj;
    target.transposed_first_offdiag = ij_is_forward;
    return target;
}

MUDA_DEVICE void accumulate_class(SocuNativeHalfBlockClassification& out,
                                  SocuNativeBandClass cls) noexcept
{
    switch(cls)
    {
        case SocuNativeBandClass::Diag:
            ++out.diag_scalar_count;
            break;
        case SocuNativeBandClass::FirstOffdiag:
            ++out.first_offdiag_scalar_count;
            break;
        case SocuNativeBandClass::OffBand:
            ++out.offband_scalar_count;
            break;
        case SocuNativeBandClass::Skipped:
        default:
            ++out.skipped_scalar_count;
            break;
    }
}

MUDA_DEVICE SocuNativeBandClass
dominant_class(const SocuNativeHalfBlockClassification& in) noexcept
{
    if(in.offband_scalar_count != 0)
        return SocuNativeBandClass::OffBand;
    if(in.first_offdiag_scalar_count != 0)
        return SocuNativeBandClass::FirstOffdiag;
    if(in.diag_scalar_count != 0)
        return SocuNativeBandClass::Diag;
    return SocuNativeBandClass::Skipped;
}

MUDA_DEVICE SocuNativeHalfBlockClassification classify_half_block(
    const SocuNativeVertexDescriptor& row,
    const SocuNativeVertexDescriptor& col,
    muda::CBufferView<IndexT>         old_to_chain,
    SizeT                             horizon,
    SizeT                             block_size) noexcept
{
    SocuNativeHalfBlockClassification out;
    if(!row.mapped() || !col.mapped() || row.fixed || col.fixed)
    {
        out.cls = SocuNativeBandClass::Skipped;
        return out;
    }

    for(IndexT r = 0; r < row.dof_count; ++r)
    {
        for(IndexT c = 0; c < col.dof_count; ++c)
        {
            const auto target = classify_old_to_chain_pair(old_to_chain,
                                                           horizon,
                                                           block_size,
                                                           row.old_dof + r,
                                                           col.old_dof + c);
            accumulate_class(out, target.cls);
        }
    }
    out.cls = dominant_class(out);
    return out;
}

MUDA_DEVICE bool load_old_to_chain_block_lane(muda::CBufferView<IndexT> old_to_chain,
                                              SizeT                     horizon,
                                              SizeT                     block_size,
                                              IndexT                    old_dof,
                                              SizeT&                    block,
                                              SizeT&                    lane) noexcept
{
    if(block_size == 0 || old_dof < 0
       || static_cast<SizeT>(old_dof) >= old_to_chain.size())
        return false;

    const IndexT chain = old_to_chain.data()[static_cast<SizeT>(old_dof)];
    if(chain < 0)
        return false;

    const SizeT chain_dof = static_cast<SizeT>(chain);
    block = chain_dof / block_size;
    lane  = chain_dof % block_size;
    return block < horizon;
}

MUDA_DEVICE bool fill_direct_side(SocuNativeContactDirectSide& side,
                                  const SocuNativeVertexDescriptor& vertex,
                                  muda::CBufferView<IndexT> old_to_chain,
                                  muda::CBufferView<ABDJacobi> abd_vertex_to_J,
                                  SizeT horizon,
                                  SizeT block_size) noexcept
{
    if(vertex.old_dof < 0 || vertex.dof_count <= 0
       || vertex.dof_count > SocuNativeContactMaxDofsPerVertex)
        return false;

    SocuNativeContactDirectSide out;
    out.count = vertex.dof_count;

    ABDJacobi J;
    if(vertex.kind == SocuNativeDescriptorKind::Abd)
    {
        if(vertex.abd_j_index < 0
           || static_cast<SizeT>(vertex.abd_j_index) >= abd_vertex_to_J.size()
           || abd_vertex_to_J.data() == nullptr)
            return false;
        J = abd_vertex_to_J.data()[static_cast<SizeT>(vertex.abd_j_index)];
    }

    for(IndexT local = 0; local < vertex.dof_count; ++local)
    {
        SizeT block = 0;
        SizeT lane  = 0;
        if(!load_old_to_chain_block_lane(old_to_chain,
                                         horizon,
                                         block_size,
                                         vertex.old_dof + local,
                                         block,
                                         lane)
           || block > SocuNativeContactMaxPackedBlock
           || lane > SocuNativeContactMaxPackedLane)
            return false;

        out.blocks[static_cast<SizeT>(local)] =
            static_cast<std::uint32_t>(block);
        out.lanes[static_cast<SizeT>(local)] =
            static_cast<std::uint16_t>(lane);
        if(vertex.kind == SocuNativeDescriptorKind::Fem)
        {
            if(local >= 3)
                return false;
            out.components[static_cast<SizeT>(local)] =
                static_cast<std::uint8_t>(local);
            out.weights[static_cast<SizeT>(local)] = Float{1};
        }
        else if(vertex.kind == SocuNativeDescriptorKind::Abd)
        {
            out.components[static_cast<SizeT>(local)] =
                static_cast<std::uint8_t>(
                    socu_native_contact_abd_component(local));
            out.weights[static_cast<SizeT>(local)] =
                socu_native_contact_abd_weight(J, local);
        }
        else
        {
            return false;
        }
    }

    side = out;
    return true;
}

MUDA_DEVICE bool fill_direct_data(SocuNativeContactStencilTarget& target,
                                  const SocuNativeVertexDescriptor& row,
                                  const SocuNativeVertexDescriptor& col,
                                  muda::CBufferView<IndexT> old_to_chain,
                                  muda::CBufferView<ABDJacobi> abd_vertex_to_J,
                                  SizeT horizon,
                                  SizeT block_size) noexcept
{
    if(!target.scalar_classification.fully_in_band()
       || target.half_block_class == SocuNativeBandClass::Skipped)
        return false;

    SocuNativeContactDirectSide row_side;
    SocuNativeContactDirectSide col_side;
    if(!fill_direct_side(row_side,
                         row,
                         old_to_chain,
                         abd_vertex_to_J,
                         horizon,
                         block_size)
       || !fill_direct_side(col_side,
                            col,
                            old_to_chain,
                            abd_vertex_to_J,
                            horizon,
                            block_size))
        return false;

    target.row_direct = row_side;
    target.col_direct = col_side;
    target.direct_lanes_valid = true;
    target.direct_projection_valid = true;
    return true;
}

MUDA_DEVICE bool fill_diagonal_direct_data(
    SocuNativeContactStencilTarget& target,
    const SocuNativeVertexDescriptor& vertex,
    muda::CBufferView<IndexT> old_to_chain,
    muda::CBufferView<ABDJacobi> abd_vertex_to_J,
    SizeT horizon,
    SizeT block_size) noexcept
{
    SocuNativeContactDirectSide side;
    if(!fill_direct_side(side,
                         vertex,
                         old_to_chain,
                         abd_vertex_to_J,
                         horizon,
                         block_size))
        return false;

    target.row_direct = side;
    target.col_direct = side;
    target.direct_lanes_valid = true;
    target.direct_projection_valid = true;
    return true;
}

MUDA_DEVICE void accumulate_stencil_class(SocuNativeStencilClassification& out,
                                          SocuNativeBandClass cls) noexcept
{
    switch(cls)
    {
        case SocuNativeBandClass::Diag:
            ++out.diag_half_block_count;
            break;
        case SocuNativeBandClass::FirstOffdiag:
            ++out.first_offdiag_half_block_count;
            break;
        case SocuNativeBandClass::OffBand:
            ++out.offband_half_block_count;
            break;
        case SocuNativeBandClass::Skipped:
        default:
            ++out.skipped_half_block_count;
            break;
    }
}

MUDA_DEVICE SocuNativeContactWriteMode make_write_mode(
    const SocuNativeStencilClassification& classification,
    StructuredContactOffbandPolicy         fallback_policy) noexcept
{
    if(classification.offband_half_block_count == 0)
        return SocuNativeContactWriteMode::ExactInBand;
    if(fallback_policy == StructuredContactOffbandPolicy::Diag)
        return SocuNativeContactWriteMode::DiagFallback;
    if(fallback_policy == StructuredContactOffbandPolicy::DiagLump)
        return SocuNativeContactWriteMode::DiagLumpFallback;
    return SocuNativeContactWriteMode::DropOffBand;
}

MUDA_DEVICE bool upper_lr(IndexT left_value,
                          IndexT right_value,
                          IndexT left_slot,
                          IndexT right_slot,
                          IndexT& L,
                          IndexT& R) noexcept
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

template <int StencilSize, typename Stencil>
MUDA_DEVICE SocuNativeStencilClassification classify_stencil(
    const SocuNativeVertexDescriptor (&vertices)[StencilSize],
    const Stencil&                     indices,
    muda::CBufferView<IndexT>          old_to_chain,
    SizeT                              horizon,
    SizeT                              block_size) noexcept
{
    SocuNativeStencilClassification out;
    for(IndexT row = 0; row < StencilSize; ++row)
    {
        for(IndexT col = row; col < StencilSize; ++col)
        {
            IndexT L = row;
            IndexT R = col;
            upper_lr(indices(row), indices(col), row, col, L, R);
            accumulate_stencil_class(out,
                                     classify_half_block(vertices[L],
                                                         vertices[R],
                                                         old_to_chain,
                                                         horizon,
                                                         block_size)
                                         .cls);
        }
    }
    return out;
}

template <int StencilSize>
MUDA_DEVICE SocuNativeContactStencilTarget make_half_block_target(
    IndexT                         contact_id,
    IndexT                         local_row_vertex,
    IndexT                         local_col_vertex,
    IndexT                         row_global_vertex,
    IndexT                         col_global_vertex,
    bool                           mirror_diag_block,
    const SocuNativeVertexDescriptor& row,
    const SocuNativeVertexDescriptor& col,
    muda::CBufferView<IndexT>      old_to_chain,
    muda::CBufferView<ABDJacobi>   abd_vertex_to_J,
    SizeT                          horizon,
    SizeT                          block_size,
    StructuredContactOffbandPolicy fallback_policy,
    SocuNativeContactWriteMode     stencil_write_mode) noexcept
{
    SocuNativeContactStencilTarget target;
    target.contact_id       = contact_id;
    target.row_global_vertex = row_global_vertex;
    target.col_global_vertex = col_global_vertex;
    target.local_row_vertex = local_row_vertex;
    target.local_col_vertex = local_col_vertex;
    target.mirror_diag_block = mirror_diag_block;
    target.row_old_dof      = row.old_dof;
    target.row_dof_count    = row.dof_count;
    target.col_old_dof      = col.old_dof;
    target.col_dof_count    = col.dof_count;
    target.row_kind         = row.kind;
    target.col_kind         = col.kind;
    target.row_abd_body     = row.abd_body;
    target.col_abd_body     = col.abd_body;
    target.row_jacobian_index = row.abd_j_index;
    target.col_jacobian_index = col.abd_j_index;
    target.fallback_policy    = fallback_policy;

    target.scalar_classification =
        classify_half_block(row, col, old_to_chain, horizon, block_size);
    target.half_block_class = target.scalar_classification.cls;

    if(stencil_write_mode == SocuNativeContactWriteMode::DiagFallback
       || stencil_write_mode == SocuNativeContactWriteMode::DiagLumpFallback)
    {
        target.write_mode = stencil_write_mode;
        if(local_row_vertex == local_col_vertex)
            fill_diagonal_direct_data(target,
                                      row,
                                      old_to_chain,
                                      abd_vertex_to_J,
                                      horizon,
                                      block_size);
        return target;
    }

    if(target.half_block_class == SocuNativeBandClass::Skipped)
    {
        target.write_mode = SocuNativeContactWriteMode::Skipped;
        return target;
    }

    if(target.half_block_class == SocuNativeBandClass::OffBand)
    {
        target.write_mode = SocuNativeContactWriteMode::DropOffBand;
        return target;
    }

    target.write_mode = SocuNativeContactWriteMode::ExactInBand;
    for(IndexT r = 0; r < row.dof_count; ++r)
    {
        for(IndexT c = 0; c < col.dof_count; ++c)
        {
            const auto scalar = classify_old_to_chain_pair(old_to_chain,
                                                           horizon,
                                                           block_size,
                                                           row.old_dof + r,
                                                           col.old_dof + c);
            if(scalar.cls != target.half_block_class)
                continue;
            target.block_or_left_block = scalar.left_block;
            target.row_lane            = scalar.row_lane;
            target.col_lane            = scalar.col_lane;
            target.transposed_first_offdiag =
                scalar.transposed_first_offdiag;
            fill_direct_data(target,
                             row,
                             col,
                             old_to_chain,
                             abd_vertex_to_J,
                             horizon,
                             block_size);
            return target;
        }
    }
    return target;
}

template <int StencilSize, typename StencilView>
void rebuild_stencil_targets(cudaStream_t stream,
                             muda::BufferView<SocuNativeContactStencilTarget> targets,
                             StencilView stencils,
                             muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
                             muda::CBufferView<IndexT> old_to_chain,
                             muda::CBufferView<ABDJacobi> abd_vertex_to_J,
                             SizeT horizon,
                             SizeT block_size,
                             StructuredContactOffbandPolicy fallback_policy)
{
    constexpr SizeT HalfBlockCount = StencilSize * (StencilSize + 1) / 2;
    if(stencils.size() == 0 || targets.size() == 0)
        return;

    using namespace muda;
    ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(stencils.size(),
               [targets,
                stencils,
                vertex_descriptors,
                old_to_chain,
                abd_vertex_to_J,
                horizon,
                block_size,
                fallback_policy] __device__(int i) mutable
               {
                   const SizeT base =
                       static_cast<SizeT>(i) * HalfBlockCount;
                   if(base + HalfBlockCount > targets.size())
                       return;

                   const auto stencil = stencils.data()[static_cast<SizeT>(i)];
                   SocuNativeVertexDescriptor vertices[StencilSize];
                   for(IndexT local = 0; local < StencilSize; ++local)
                   {
                       const IndexT global_vertex = stencil(local);
                       if(global_vertex >= 0
                          && static_cast<SizeT>(global_vertex)
                                 < vertex_descriptors.size())
                           vertices[local] =
                               vertex_descriptors.data()[static_cast<SizeT>(global_vertex)];
                   }

                   const auto classification =
                       classify_stencil<StencilSize>(vertices,
                                                     stencil,
                                                     old_to_chain,
                                                     horizon,
                                                     block_size);
                   const auto write_mode =
                       make_write_mode(classification, fallback_policy);

                   SizeT target_index = base;
                   for(IndexT row = 0; row < StencilSize; ++row)
                   {
                       for(IndexT col = row; col < StencilSize; ++col)
                       {
                           IndexT L = row;
                           IndexT R = col;
                           const bool swapped = upper_lr(stencil(row),
                                                         stencil(col),
                                                         row,
                                                         col,
                                                         L,
                                                         R);
                           targets.data()[target_index++] =
                               make_half_block_target<StencilSize>(i,
                                                                   L,
                                                                   R,
                                                                   stencil(L),
                                                                   stencil(R),
                                                                   stencil(L)
                                                                           != stencil(R)
                                                                       || swapped,
                                                                   vertices[L],
                                                                   vertices[R],
                                                                   old_to_chain,
                                                                   abd_vertex_to_J,
                                                                   horizon,
                                                                   block_size,
                                                                   fallback_policy,
                                                                   write_mode);
                       }
                   }
               });
}
}  // namespace

void rebuild_socu_native_simplex_contact_targets(
    cudaStream_t stream,
    muda::BufferView<SocuNativeContactStencilTarget> pt_targets,
    muda::BufferView<SocuNativeContactStencilTarget> ee_targets,
    muda::BufferView<SocuNativeContactStencilTarget> pe_targets,
    muda::BufferView<SocuNativeContactStencilTarget> pp_targets,
    muda::CBufferView<Vector4i> pts,
    muda::CBufferView<Vector4i> ees,
    muda::CBufferView<Vector3i> pes,
    muda::CBufferView<Vector2i> pps,
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
    muda::CBufferView<IndexT> old_to_chain,
    muda::CBufferView<ABDJacobi> abd_vertex_to_J,
    SizeT horizon,
    SizeT block_size,
    StructuredContactOffbandPolicy fallback_policy)
{
    rebuild_stencil_targets<4>(stream,
                               pt_targets,
                               pts,
                               vertex_descriptors,
                               old_to_chain,
                               abd_vertex_to_J,
                               horizon,
                               block_size,
                               fallback_policy);
    rebuild_stencil_targets<4>(stream,
                               ee_targets,
                               ees,
                               vertex_descriptors,
                               old_to_chain,
                               abd_vertex_to_J,
                               horizon,
                               block_size,
                               fallback_policy);
    rebuild_stencil_targets<3>(stream,
                               pe_targets,
                               pes,
                               vertex_descriptors,
                               old_to_chain,
                               abd_vertex_to_J,
                               horizon,
                               block_size,
                               fallback_policy);
    rebuild_stencil_targets<2>(stream,
                               pp_targets,
                               pps,
                               vertex_descriptors,
                               old_to_chain,
                               abd_vertex_to_J,
                               horizon,
                               block_size,
                               fallback_policy);
}

void rebuild_socu_native_vertex_half_plane_contact_targets(
    cudaStream_t stream,
    muda::BufferView<SocuNativeContactStencilTarget> ph_targets,
    muda::CBufferView<Vector2i> phs,
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
    muda::CBufferView<IndexT> old_to_chain,
    muda::CBufferView<ABDJacobi> abd_vertex_to_J,
    SizeT horizon,
    SizeT block_size,
    StructuredContactOffbandPolicy fallback_policy)
{
    rebuild_stencil_targets<1>(stream,
                               ph_targets,
                               phs,
                               vertex_descriptors,
                               old_to_chain,
                               abd_vertex_to_J,
                               horizon,
                               block_size,
                               fallback_policy);
}
}  // namespace uipc::backend::cuda_mixed
