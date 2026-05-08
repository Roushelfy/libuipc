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
    const SocuNativeVertexDescriptor& row,
    const SocuNativeVertexDescriptor& col,
    muda::CBufferView<IndexT>      old_to_chain,
    SizeT                          horizon,
    SizeT                          block_size,
    StructuredContactOffbandPolicy fallback_policy,
    SocuNativeContactWriteMode     stencil_write_mode) noexcept
{
    SocuNativeContactStencilTarget target;
    target.contact_id       = contact_id;
    target.local_row_vertex = local_row_vertex;
    target.local_col_vertex = local_col_vertex;
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
                           upper_lr(stencil(row), stencil(col), row, col, L, R);
                           targets.data()[target_index++] =
                               make_half_block_target<StencilSize>(i,
                                                                   L,
                                                                   R,
                                                                   vertices[L],
                                                                   vertices[R],
                                                                   old_to_chain,
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
    SizeT horizon,
    SizeT block_size,
    StructuredContactOffbandPolicy fallback_policy)
{
    rebuild_stencil_targets<4>(stream,
                               pt_targets,
                               pts,
                               vertex_descriptors,
                               old_to_chain,
                               horizon,
                               block_size,
                               fallback_policy);
    rebuild_stencil_targets<4>(stream,
                               ee_targets,
                               ees,
                               vertex_descriptors,
                               old_to_chain,
                               horizon,
                               block_size,
                               fallback_policy);
    rebuild_stencil_targets<3>(stream,
                               pe_targets,
                               pes,
                               vertex_descriptors,
                               old_to_chain,
                               horizon,
                               block_size,
                               fallback_policy);
    rebuild_stencil_targets<2>(stream,
                               pp_targets,
                               pps,
                               vertex_descriptors,
                               old_to_chain,
                               horizon,
                               block_size,
                               fallback_policy);
}
}  // namespace uipc::backend::cuda_mixed
