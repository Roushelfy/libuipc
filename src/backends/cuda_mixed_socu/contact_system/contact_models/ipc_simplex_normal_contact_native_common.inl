#pragma once

#include <linear_system/socu_native_contact_assembly_sink.h>
#include <mixed_precision/policy.h>
#include <utils/structured_contact_assembly_sink.h>

namespace uipc::backend::cuda_mixed::ipc_simplex_normal_native_detail
{
template <int StencilSize>
MUDA_DEVICE bool contact_targets_are_exact_or_skipped(
    muda::CBufferView<SocuNativeContactStencilTarget> targets,
    SizeT                                             base) noexcept
{
    constexpr SizeT HalfBlockCount = StencilSize * (StencilSize + 1) / 2;
    if(targets.data() == nullptr || base + HalfBlockCount > targets.size())
        return false;

#pragma unroll
    for(IndexT target_offset = 0; target_offset < HalfBlockCount; ++target_offset)
    {
        const auto& target =
            targets.data()[base + static_cast<SizeT>(target_offset)];
        if(target.write_mode == SocuNativeContactWriteMode::ExactInBand
           || target.write_mode == SocuNativeContactWriteMode::Skipped)
            continue;
        return false;
    }

    return true;
}

template <int StencilSize>
MUDA_DEVICE bool contact_targets_are_whole_stencil_fallback(
    muda::CBufferView<SocuNativeContactStencilTarget> targets,
    SizeT                                             base,
    SocuNativeContactWriteMode&                       write_mode) noexcept
{
    constexpr SizeT HalfBlockCount = StencilSize * (StencilSize + 1) / 2;
    if(targets.data() == nullptr || base + HalfBlockCount > targets.size())
        return false;

    write_mode = targets.data()[base].write_mode;
    if(write_mode != SocuNativeContactWriteMode::DiagFallback
       && write_mode != SocuNativeContactWriteMode::DiagLumpFallback)
        return false;

#pragma unroll
    for(IndexT target_offset = 1; target_offset < HalfBlockCount; ++target_offset)
    {
        const auto mode =
            targets.data()[base + static_cast<SizeT>(target_offset)].write_mode;
        if(mode != write_mode)
            return false;
    }
    return true;
}

template <int StencilSize, typename Stencil, typename HMat>
MUDA_DEVICE void write_exact_targets(
    SocuNativeContactAssemblySink<ActivePolicy::StoreScalar,
                                  ActivePolicy::SolveScalar> native_sink,
    muda::CBufferView<SocuNativeContactStencilTarget> targets,
    IndexT                                           contact_id,
    const Stencil&                                   indices,
    const HMat&                                      H) noexcept
{
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT HalfBlockCount = StencilSize * (StencilSize + 1) / 2;
    const SizeT     base = static_cast<SizeT>(contact_id) * HalfBlockCount;

    SocuNativeContactExactWriter<Store, Solve> writer = native_sink.writer();

    if(!contact_targets_are_exact_or_skipped<StencilSize>(targets, base))
    {
        SocuNativeContactWriteMode fallback_mode =
            SocuNativeContactWriteMode::Skipped;
        if(!contact_targets_are_whole_stencil_fallback<StencilSize>(
               targets,
               base,
               fallback_mode))
            return;

        if(fallback_mode == SocuNativeContactWriteMode::DiagFallback)
            writer.template write_diag_fallback_stencil<StencilSize>(
                targets,
                base,
                H);
        else if(fallback_mode == SocuNativeContactWriteMode::DiagLumpFallback)
            writer.template write_lump_fallback_stencil<StencilSize>(
                targets,
                base,
                H);
        return;
    }

#pragma unroll
    for(IndexT target_offset = 0; target_offset < HalfBlockCount; ++target_offset)
    {
        const auto target =
            targets.data()[base + static_cast<SizeT>(target_offset)];
        if(!target.exact_in_band())
            continue;

        const auto H3 = H.template block<3, 3>(target.local_row_vertex * 3,
                                               target.local_col_vertex * 3);
        writer.write_half_block(target, H3);
    }
}
}  // namespace uipc::backend::cuda_mixed::ipc_simplex_normal_native_detail
