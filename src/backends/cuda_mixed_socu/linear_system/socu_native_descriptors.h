#pragma once

#include <uipc/common/span.h>
#include <uipc/common/type_define.h>

#include <cuda_runtime_api.h>
#include <muda/buffer/buffer_view.h>

#include <stdexcept>
#include <vector>

namespace uipc::backend::cuda_mixed
{
enum class SocuNativeDescriptorKind : IndexT
{
    None = 0,
    Fem  = 1,
    Abd  = 2,
};

enum class SocuNativeBandClass : IndexT
{
    Skipped      = 0,
    Diag         = 1,
    FirstOffdiag = 2,
    OffBand      = 3,
};

struct SocuNativeDofDescriptor
{
    IndexT old_dof   = -1;
    IndexT chain_dof = -1;
    IndexT atom      = -1;
    SizeT  block     = 0;
    SizeT  lane      = 0;
    IndexT epoch     = 0;
    bool   active    = false;
};

struct SocuNativeVertexDescriptor
{
    SocuNativeDescriptorKind kind = SocuNativeDescriptorKind::None;
    bool                     fixed = false;
    IndexT                   old_dof = -1;
    IndexT                   dof_count = 0;
    SizeT                    block = 0;
    SizeT                    lane  = 0;
    IndexT                   abd_body = -1;
    IndexT                   abd_j_index = -1;
    IndexT                   epoch = 0;
    bool                     active = false;

    MUDA_GENERIC bool mapped() const noexcept
    {
        return kind != SocuNativeDescriptorKind::None && old_dof >= 0
               && dof_count > 0;
    }

    MUDA_GENERIC bool writable() const noexcept
    {
        return mapped() && active && !fixed;
    }
};

struct SocuNativeDescriptorTable
{
    IndexT epoch      = 0;
    SizeT  horizon    = 0;
    SizeT  block_size = 0;
    std::vector<SocuNativeDofDescriptor>    dofs;
    std::vector<SocuNativeVertexDescriptor> vertices;

    bool valid_for(IndexT expected_epoch,
                   SizeT  expected_horizon = 0,
                   SizeT  expected_block_size = 0) const noexcept
    {
        return epoch == expected_epoch
               && (expected_horizon == 0 || horizon == expected_horizon)
               && (expected_block_size == 0 || block_size == expected_block_size);
    }
};

struct SocuNativeVertexDescriptorBuildInput
{
    SizeT  global_vertex_count = 0;
    SizeT  horizon = 0;
    SizeT  block_size = 0;
    IndexT epoch = 0;
    span<const SocuNativeDofDescriptor> dofs;

    IndexT fem_vertex_offset = -1;
    IndexT fem_vertex_count = 0;
    IndexT fem_old_dof_offset = -1;
    span<const IndexT> fem_vertex_is_fixed;

    IndexT abd_vertex_offset = -1;
    IndexT abd_vertex_count = 0;
    IndexT abd_old_dof_offset = -1;
    IndexT abd_body_count = 0;
    span<const IndexT> abd_vertex_to_body;
    span<const IndexT> abd_body_is_fixed;
};

struct SocuNativeScalarTarget
{
    SocuNativeBandClass cls = SocuNativeBandClass::Skipped;
    SizeT               left_block = 0;
    SizeT               row_lane = 0;
    SizeT               col_lane = 0;
    bool                transposed_first_offdiag = false;
};

struct SocuNativeHalfBlockClassification
{
    SocuNativeBandClass cls = SocuNativeBandClass::Skipped;
    SizeT               diag_scalar_count = 0;
    SizeT               first_offdiag_scalar_count = 0;
    SizeT               offband_scalar_count = 0;
    SizeT               skipped_scalar_count = 0;

    // "In band" means no scalar would be dropped as off-band. Fixed or
    // otherwise skipped scalars can still be present.
    bool fully_in_band() const noexcept
    {
        return offband_scalar_count == 0;
    }

    // Use this when the caller needs every scalar to be writable.
    bool fully_writable_in_band() const noexcept
    {
        return fully_in_band() && skipped_scalar_count == 0
               && cls != SocuNativeBandClass::Skipped;
    }
};

struct SocuNativeStencilClassification
{
    SizeT diag_half_block_count = 0;
    SizeT first_offdiag_half_block_count = 0;
    SizeT offband_half_block_count = 0;
    SizeT skipped_half_block_count = 0;

    // "In band" means no half-block would be dropped as off-band. Fixed or
    // otherwise skipped half-blocks can still be present.
    bool fully_in_band() const noexcept
    {
        return offband_half_block_count == 0;
    }

    // Use this when the caller needs every half-block to be writable.
    bool fully_writable_in_band() const noexcept
    {
        return fully_in_band() && skipped_half_block_count == 0;
    }
};

inline std::vector<SocuNativeDofDescriptor>
build_socu_native_dof_descriptors(span<const IndexT> old_to_chain,
                                  span<const IndexT> old_dof_to_atom,
                                  SizeT              horizon,
                                  SizeT              block_size,
                                  IndexT             epoch)
{
    if(block_size == 0)
        throw std::invalid_argument("SOCU native descriptors require block_size > 0");

    std::vector<SocuNativeDofDescriptor> descriptors(old_to_chain.size());
    for(SizeT old = 0; old < old_to_chain.size(); ++old)
    {
        auto& descriptor = descriptors[old];
        descriptor.old_dof = static_cast<IndexT>(old);
        descriptor.epoch   = epoch;
        if(old < old_dof_to_atom.size())
            descriptor.atom = old_dof_to_atom[old];

        const IndexT chain = old_to_chain[old];
        if(chain < 0)
            continue;

        descriptor.chain_dof = chain;
        descriptor.block = static_cast<SizeT>(chain) / block_size;
        descriptor.lane  = static_cast<SizeT>(chain) % block_size;
        descriptor.active = descriptor.block < horizon;
    }
    return descriptors;
}

inline bool socu_native_dof_range_active(span<const SocuNativeDofDescriptor> dofs,
                                         IndexT old_dof,
                                         IndexT dof_count) noexcept
{
    if(old_dof < 0 || dof_count <= 0)
        return false;
    if(static_cast<SizeT>(old_dof + dof_count) > dofs.size())
        return false;

    const auto& first = dofs[static_cast<SizeT>(old_dof)];
    if(!first.active || first.chain_dof < 0)
        return false;
    for(IndexT i = 1; i < dof_count; ++i)
    {
        const auto& dof = dofs[static_cast<SizeT>(old_dof + i)];
        if(!dof.active)
            return false;
        if(dof.chain_dof != first.chain_dof + i)
            return false;
        if(dof.block != first.block
           || dof.lane != first.lane + static_cast<SizeT>(i))
            return false;
    }
    return true;
}

inline SocuNativeVertexDescriptor make_socu_native_vertex_descriptor(
    SocuNativeDescriptorKind              kind,
    bool                                  fixed,
    IndexT                                old_dof,
    IndexT                                dof_count,
    IndexT                                abd_body,
    IndexT                                abd_j_index,
    IndexT                                epoch,
    span<const SocuNativeDofDescriptor>   dofs)
{
    SocuNativeVertexDescriptor descriptor;
    descriptor.kind        = kind;
    descriptor.fixed       = fixed;
    descriptor.old_dof     = old_dof;
    descriptor.dof_count   = dof_count;
    descriptor.abd_body    = abd_body;
    descriptor.abd_j_index = abd_j_index;
    descriptor.epoch       = epoch;
    descriptor.active      = socu_native_dof_range_active(dofs, old_dof, dof_count);

    if(old_dof >= 0 && static_cast<SizeT>(old_dof) < dofs.size())
    {
        const auto& dof = dofs[static_cast<SizeT>(old_dof)];
        descriptor.block = dof.block;
        descriptor.lane  = dof.lane;
    }
    return descriptor;
}

inline std::vector<SocuNativeVertexDescriptor>
build_socu_native_vertex_descriptors(const SocuNativeVertexDescriptorBuildInput& input)
{
    std::vector<SocuNativeVertexDescriptor> descriptors(input.global_vertex_count);

    auto require_vertex_range = [&](IndexT offset, IndexT count, const char* name)
    {
        if(count <= 0)
            return;
        if(offset < 0 || static_cast<SizeT>(offset + count) > descriptors.size())
            throw std::out_of_range(name);
    };

    require_vertex_range(input.fem_vertex_offset,
                         input.fem_vertex_count,
                         "SOCU native FEM vertex descriptor range is out of bounds");
    for(IndexT local = 0; local < input.fem_vertex_count; ++local)
    {
        const SizeT global =
            static_cast<SizeT>(input.fem_vertex_offset + local);
        const bool fixed =
            local >= 0 && static_cast<SizeT>(local) < input.fem_vertex_is_fixed.size()
                ? input.fem_vertex_is_fixed[static_cast<SizeT>(local)] != 0
                : false;
        descriptors[global] = make_socu_native_vertex_descriptor(
            SocuNativeDescriptorKind::Fem,
            fixed,
            input.fem_old_dof_offset + local * 3,
            3,
            -1,
            -1,
            input.epoch,
            input.dofs);
    }

    require_vertex_range(input.abd_vertex_offset,
                         input.abd_vertex_count,
                         "SOCU native ABD vertex descriptor range is out of bounds");
    for(IndexT local = 0; local < input.abd_vertex_count; ++local)
    {
        if(static_cast<SizeT>(local) >= input.abd_vertex_to_body.size())
            continue;
        const IndexT body = input.abd_vertex_to_body[static_cast<SizeT>(local)];
        if(body < 0 || body >= input.abd_body_count)
            continue;
        const bool fixed =
            static_cast<SizeT>(body) < input.abd_body_is_fixed.size()
                ? input.abd_body_is_fixed[static_cast<SizeT>(body)] != 0
                : false;
        const SizeT global =
            static_cast<SizeT>(input.abd_vertex_offset + local);
        descriptors[global] = make_socu_native_vertex_descriptor(
            SocuNativeDescriptorKind::Abd,
            fixed,
            input.abd_old_dof_offset + body * 12,
            12,
            body,
            local,
            input.epoch,
            input.dofs);
    }

    return descriptors;
}

inline SocuNativeScalarTarget
socu_native_classify_old_dof_pair(span<const IndexT> old_to_chain,
                                  SizeT              horizon,
                                  SizeT              block_size,
                                  IndexT             old_i,
                                  IndexT             old_j) noexcept
{
    SocuNativeScalarTarget target;
    if(block_size == 0 || old_i < 0 || old_j < 0)
        return target;
    if(static_cast<SizeT>(old_i) >= old_to_chain.size()
       || static_cast<SizeT>(old_j) >= old_to_chain.size())
        return target;

    const IndexT chain_i = old_to_chain[static_cast<SizeT>(old_i)];
    const IndexT chain_j = old_to_chain[static_cast<SizeT>(old_j)];
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

    const SizeT distance = bi > bj ? bi - bj : bj - bi;
    if(distance != 1)
    {
        target.cls = SocuNativeBandClass::OffBand;
        return target;
    }

    const bool ij_is_forward = bi < bj;
    target.cls = SocuNativeBandClass::FirstOffdiag;
    target.left_block = ij_is_forward ? bi : bj;
    target.row_lane = ij_is_forward ? lj : li;
    target.col_lane = ij_is_forward ? li : lj;
    target.transposed_first_offdiag = ij_is_forward;
    return target;
}

inline void socu_native_accumulate_class(SocuNativeHalfBlockClassification& out,
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

inline SocuNativeBandClass
socu_native_dominant_class(const SocuNativeHalfBlockClassification& in) noexcept
{
    if(in.offband_scalar_count != 0)
        return SocuNativeBandClass::OffBand;
    if(in.first_offdiag_scalar_count != 0)
        return SocuNativeBandClass::FirstOffdiag;
    if(in.diag_scalar_count != 0)
        return SocuNativeBandClass::Diag;
    return SocuNativeBandClass::Skipped;
}

inline SocuNativeHalfBlockClassification
socu_native_classify_half_block(const SocuNativeVertexDescriptor& lhs,
                                const SocuNativeVertexDescriptor& rhs,
                                span<const IndexT>                old_to_chain,
                                SizeT                             horizon,
                                SizeT                             block_size) noexcept
{
    SocuNativeHalfBlockClassification out;
    if(!lhs.writable() || !rhs.writable())
    {
        out.cls = SocuNativeBandClass::Skipped;
        return out;
    }

    for(IndexT r = 0; r < lhs.dof_count; ++r)
    {
        for(IndexT c = 0; c < rhs.dof_count; ++c)
        {
            const auto target = socu_native_classify_old_dof_pair(
                old_to_chain,
                horizon,
                block_size,
                lhs.old_dof + r,
                rhs.old_dof + c);
            socu_native_accumulate_class(out, target.cls);
        }
    }
    out.cls = socu_native_dominant_class(out);
    return out;
}

inline SocuNativeStencilClassification
socu_native_classify_stencil_half(span<const SocuNativeVertexDescriptor> vertices,
                                  span<const IndexT> old_to_chain,
                                  SizeT              horizon,
                                  SizeT              block_size) noexcept
{
    SocuNativeStencilClassification out;
    for(SizeT row = 0; row < vertices.size(); ++row)
    {
        for(SizeT col = row; col < vertices.size(); ++col)
        {
            const auto cls = socu_native_classify_half_block(vertices[row],
                                                            vertices[col],
                                                            old_to_chain,
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

void rebuild_socu_native_vertex_descriptors(
    cudaStream_t                           stream,
    muda::BufferView<SocuNativeVertexDescriptor> descriptors,
    muda::CBufferView<IndexT>              old_to_chain,
    SizeT                                  horizon,
    SizeT                                  block_size,
    IndexT                                 epoch,
    IndexT                                 fem_vertex_offset,
    IndexT                                 fem_vertex_count,
    IndexT                                 fem_old_dof_offset,
    muda::CBufferView<IndexT>              fem_vertex_is_fixed,
    IndexT                                 abd_vertex_offset,
    IndexT                                 abd_vertex_count,
    IndexT                                 abd_old_dof_offset,
    IndexT                                 abd_body_count,
    muda::CBufferView<IndexT>              abd_vertex_to_body,
    muda::CBufferView<IndexT>              abd_body_is_fixed);
}  // namespace uipc::backend::cuda_mixed
