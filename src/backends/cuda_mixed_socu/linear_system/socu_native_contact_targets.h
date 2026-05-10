#pragma once

#include <linear_system/socu_native_descriptors.h>
#include <utils/structured_contact_offband_policy.h>

namespace uipc::backend::cuda_mixed
{
inline constexpr IndexT SocuNativeContactMaxDofsPerVertex = 12;

enum class SocuNativeContactWriteMode : IndexT
{
    Skipped          = 0,
    ExactInBand      = 1,
    DropOffBand      = 2,
    DiagFallback     = 3,
    DiagLumpFallback = 4,
};

struct SocuNativeContactStencilPolicy
{
    StructuredContactOffbandPolicy fallback_policy =
        StructuredContactOffbandPolicy::Drop;
    SocuNativeContactWriteMode     write_mode =
        SocuNativeContactWriteMode::ExactInBand;
    SocuNativeStencilClassification classification;

    MUDA_GENERIC bool whole_stencil_fallback() const noexcept
    {
        return write_mode == SocuNativeContactWriteMode::DiagFallback
               || write_mode == SocuNativeContactWriteMode::DiagLumpFallback;
    }
};

struct SocuNativeContactStencilTarget
{
    IndexT contact_id       = -1;
    IndexT row_global_vertex = -1;
    IndexT col_global_vertex = -1;
    IndexT local_row_vertex = -1;
    IndexT local_col_vertex = -1;

    SocuNativeContactWriteMode write_mode =
        SocuNativeContactWriteMode::Skipped;
    SocuNativeBandClass half_block_class = SocuNativeBandClass::Skipped;
    SocuNativeHalfBlockClassification scalar_classification;

    SizeT block_or_left_block = 0;
    SizeT row_lane            = 0;
    SizeT col_lane            = 0;
    bool  transposed_first_offdiag = false;
    bool  mirror_diag_block        = false;
    bool  direct_lanes_valid       = false;
    SizeT row_direct_lanes[SocuNativeContactMaxDofsPerVertex] = {};
    SizeT col_direct_lanes[SocuNativeContactMaxDofsPerVertex] = {};

    IndexT row_old_dof   = -1;
    IndexT row_dof_count = 0;
    IndexT col_old_dof   = -1;
    IndexT col_dof_count = 0;

    SocuNativeDescriptorKind row_kind = SocuNativeDescriptorKind::None;
    SocuNativeDescriptorKind col_kind = SocuNativeDescriptorKind::None;
    IndexT row_abd_body               = -1;
    IndexT col_abd_body               = -1;
    IndexT row_jacobian_index         = -1;
    IndexT col_jacobian_index         = -1;

    StructuredContactOffbandPolicy fallback_policy =
        StructuredContactOffbandPolicy::Drop;

    MUDA_GENERIC bool exact_in_band() const noexcept
    {
        return write_mode == SocuNativeContactWriteMode::ExactInBand;
    }

    MUDA_GENERIC bool whole_stencil_fallback() const noexcept
    {
        return write_mode == SocuNativeContactWriteMode::DiagFallback
               || write_mode == SocuNativeContactWriteMode::DiagLumpFallback;
    }
};

inline SocuNativeScalarTarget
socu_native_classify_dof_descriptor_pair(
    span<const SocuNativeDofDescriptor> dofs,
    SizeT                               horizon,
    SizeT                               block_size,
    IndexT                              old_i,
    IndexT                              old_j) noexcept
{
    SocuNativeScalarTarget target;
    if(block_size == 0 || old_i < 0 || old_j < 0)
        return target;
    if(static_cast<SizeT>(old_i) >= dofs.size()
       || static_cast<SizeT>(old_j) >= dofs.size())
        return target;

    const auto& di = dofs[static_cast<SizeT>(old_i)];
    const auto& dj = dofs[static_cast<SizeT>(old_j)];
    if(!di.active || !dj.active || di.old_dof != old_i || dj.old_dof != old_j)
        return target;
    if(di.block >= horizon || dj.block >= horizon || di.lane >= block_size
       || dj.lane >= block_size)
        return target;

    if(di.block == dj.block)
    {
        target.cls        = SocuNativeBandClass::Diag;
        target.left_block = di.block;
        target.row_lane   = di.lane;
        target.col_lane   = dj.lane;
        return target;
    }

    const SizeT min_block = di.block < dj.block ? di.block : dj.block;
    const SizeT max_block = di.block > dj.block ? di.block : dj.block;
    if(max_block != min_block + 1)
    {
        target.cls = SocuNativeBandClass::OffBand;
        return target;
    }

    const bool ij_is_forward = di.block < dj.block;
    target.cls        = SocuNativeBandClass::FirstOffdiag;
    target.left_block = min_block;
    target.row_lane   = ij_is_forward ? dj.lane : di.lane;
    target.col_lane   = ij_is_forward ? di.lane : dj.lane;
    target.transposed_first_offdiag = ij_is_forward;
    return target;
}

inline bool socu_native_contact_fill_direct_lanes(
    SocuNativeContactStencilTarget&      target,
    const SocuNativeVertexDescriptor&    row,
    const SocuNativeVertexDescriptor&    col,
    span<const SocuNativeDofDescriptor>  dofs,
    SizeT                                horizon,
    SizeT                                block_size) noexcept
{
    if(target.half_block_class != SocuNativeBandClass::Diag
       && target.half_block_class != SocuNativeBandClass::FirstOffdiag)
        return false;
    if(row.old_dof < 0 || col.old_dof < 0 || row.dof_count <= 0
       || col.dof_count <= 0
       || row.dof_count > SocuNativeContactMaxDofsPerVertex
       || col.dof_count > SocuNativeContactMaxDofsPerVertex)
        return false;
    if(static_cast<SizeT>(row.old_dof + row.dof_count) > dofs.size()
       || static_cast<SizeT>(col.old_dof + col.dof_count) > dofs.size())
        return false;

    for(IndexT r = 0; r < row.dof_count; ++r)
    {
        const auto& dof = dofs[static_cast<SizeT>(row.old_dof + r)];
        if(!dof.active || dof.old_dof != row.old_dof + r
           || dof.block >= horizon || dof.lane >= block_size)
            return false;
        target.row_direct_lanes[r] = dof.lane;
    }

    for(IndexT c = 0; c < col.dof_count; ++c)
    {
        const auto& dof = dofs[static_cast<SizeT>(col.old_dof + c)];
        if(!dof.active || dof.old_dof != col.old_dof + c
           || dof.block >= horizon || dof.lane >= block_size)
            return false;
        target.col_direct_lanes[c] = dof.lane;
    }

    for(IndexT r = 0; r < row.dof_count; ++r)
    {
        for(IndexT c = 0; c < col.dof_count; ++c)
        {
            const auto scalar = socu_native_classify_dof_descriptor_pair(
                dofs,
                horizon,
                block_size,
                row.old_dof + r,
                col.old_dof + c);
            if(scalar.cls != target.half_block_class
               || scalar.left_block != target.block_or_left_block)
                return false;
        }
    }

    target.direct_lanes_valid = true;
    return true;
}

inline SocuNativeHalfBlockClassification
socu_native_contact_classify_half_block(
    const SocuNativeVertexDescriptor& row,
    const SocuNativeVertexDescriptor& col,
    span<const SocuNativeDofDescriptor> dofs,
    SizeT                               horizon,
    SizeT                               block_size) noexcept
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
            const auto target = socu_native_classify_dof_descriptor_pair(
                dofs,
                horizon,
                block_size,
                row.old_dof + r,
                col.old_dof + c);
            socu_native_accumulate_class(out, target.cls);
        }
    }
    out.cls = socu_native_dominant_class(out);
    return out;
}

inline SocuNativeStencilClassification
socu_native_contact_classify_stencil_half(
    span<const SocuNativeVertexDescriptor> vertices,
    span<const SocuNativeDofDescriptor>    dofs,
    SizeT                                  horizon,
    SizeT                                  block_size) noexcept
{
    SocuNativeStencilClassification out;
    for(SizeT row = 0; row < vertices.size(); ++row)
    {
        for(SizeT col = row; col < vertices.size(); ++col)
        {
            const auto cls = socu_native_contact_classify_half_block(
                                 vertices[row],
                                 vertices[col],
                                 dofs,
                                 horizon,
                                 block_size)
                                 .cls;
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
    }
    return out;
}

inline SocuNativeContactStencilPolicy
socu_native_contact_make_stencil_policy(
    const SocuNativeStencilClassification& classification,
    StructuredContactOffbandPolicy         fallback_policy) noexcept
{
    SocuNativeContactStencilPolicy policy;
    policy.fallback_policy = fallback_policy;
    policy.classification  = classification;

    if(!classification.fully_in_band())
    {
        if(fallback_policy == StructuredContactOffbandPolicy::Diag)
            policy.write_mode = SocuNativeContactWriteMode::DiagFallback;
        else if(fallback_policy == StructuredContactOffbandPolicy::DiagLump)
            policy.write_mode = SocuNativeContactWriteMode::DiagLumpFallback;
        else
            policy.write_mode = SocuNativeContactWriteMode::DropOffBand;
        return policy;
    }

    policy.write_mode = SocuNativeContactWriteMode::ExactInBand;
    return policy;
}

inline SocuNativeContactStencilTarget
socu_native_contact_make_half_block_target(
    IndexT                               contact_id,
    IndexT                               local_row_vertex,
    IndexT                               local_col_vertex,
    IndexT                               row_global_vertex,
    IndexT                               col_global_vertex,
    const SocuNativeVertexDescriptor&    row,
    const SocuNativeVertexDescriptor&    col,
    span<const SocuNativeDofDescriptor>  dofs,
    SizeT                                horizon,
    SizeT                                block_size,
    StructuredContactOffbandPolicy       fallback_policy,
    SocuNativeContactWriteMode           stencil_write_mode =
        SocuNativeContactWriteMode::ExactInBand) noexcept
{
    SocuNativeContactStencilTarget target;
    target.contact_id       = contact_id;
    target.row_global_vertex = row_global_vertex;
    target.col_global_vertex = col_global_vertex;
    target.local_row_vertex = local_row_vertex;
    target.local_col_vertex = local_col_vertex;
    target.mirror_diag_block =
        row_global_vertex >= 0 && col_global_vertex >= 0
            ? row_global_vertex != col_global_vertex
            : local_row_vertex != local_col_vertex;
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
        socu_native_contact_classify_half_block(row, col, dofs, horizon, block_size);
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
        if(fallback_policy == StructuredContactOffbandPolicy::Diag)
            target.write_mode = SocuNativeContactWriteMode::DiagFallback;
        else if(fallback_policy == StructuredContactOffbandPolicy::DiagLump)
            target.write_mode = SocuNativeContactWriteMode::DiagLumpFallback;
        else
            target.write_mode = SocuNativeContactWriteMode::DropOffBand;
        return target;
    }

    target.write_mode = SocuNativeContactWriteMode::ExactInBand;
    for(IndexT r = 0; r < row.dof_count; ++r)
    {
        for(IndexT c = 0; c < col.dof_count; ++c)
        {
            const auto scalar = socu_native_classify_dof_descriptor_pair(
                dofs,
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
            socu_native_contact_fill_direct_lanes(
                target,
                row,
                col,
                dofs,
                horizon,
                block_size);
            return target;
        }
    }
    socu_native_contact_fill_direct_lanes(
        target,
        row,
        col,
        dofs,
        horizon,
        block_size);
    return target;
}

inline SocuNativeContactStencilTarget
socu_native_contact_make_half_block_target(
    IndexT                               contact_id,
    IndexT                               local_row_vertex,
    IndexT                               local_col_vertex,
    const SocuNativeVertexDescriptor&    row,
    const SocuNativeVertexDescriptor&    col,
    span<const SocuNativeDofDescriptor>  dofs,
    SizeT                                horizon,
    SizeT                                block_size,
    StructuredContactOffbandPolicy       fallback_policy,
    SocuNativeContactWriteMode           stencil_write_mode =
        SocuNativeContactWriteMode::ExactInBand) noexcept
{
    return socu_native_contact_make_half_block_target(contact_id,
                                                      local_row_vertex,
                                                      local_col_vertex,
                                                      -1,
                                                      -1,
                                                      row,
                                                      col,
                                                      dofs,
                                                      horizon,
                                                      block_size,
                                                      fallback_policy,
                                                      stencil_write_mode);
}

void rebuild_socu_native_simplex_contact_targets(
    cudaStream_t                              stream,
    muda::BufferView<SocuNativeContactStencilTarget> pt_targets,
    muda::BufferView<SocuNativeContactStencilTarget> ee_targets,
    muda::BufferView<SocuNativeContactStencilTarget> pe_targets,
    muda::BufferView<SocuNativeContactStencilTarget> pp_targets,
    muda::CBufferView<Vector4i>               pts,
    muda::CBufferView<Vector4i>               ees,
    muda::CBufferView<Vector3i>               pes,
    muda::CBufferView<Vector2i>               pps,
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
    muda::CBufferView<IndexT>                 old_to_chain,
    SizeT                                     horizon,
    SizeT                                     block_size,
    StructuredContactOffbandPolicy            fallback_policy);

void rebuild_socu_native_vertex_half_plane_contact_targets(
    cudaStream_t                              stream,
    muda::BufferView<SocuNativeContactStencilTarget> ph_targets,
    muda::CBufferView<Vector2i>               phs,
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
    muda::CBufferView<IndexT>                 old_to_chain,
    SizeT                                     horizon,
    SizeT                                     block_size,
    StructuredContactOffbandPolicy            fallback_policy);
}  // namespace uipc::backend::cuda_mixed
