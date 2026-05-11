#pragma once

#include <affine_body/abd_jacobi_matrix.h>
#include <linear_system/socu_native_contact_writer.h>
#include <utils/structured_contact_offband_policy.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT, typename SolveT>
struct SocuNativeContactAssemblySink
{
    SocuNativeMatrixView<SolveT>               native_matrix;
    muda::CBufferView<SocuNativeDofDescriptor> native_dof_descriptors;
    muda::CBufferView<IndexT>                  old_to_chain;
    SizeT                                      horizon = 0;
    SizeT                                      block_size = 0;
    bool                                       use_native_matrix = false;
    muda::BufferView<SolveT>                   compare_diag;
    muda::BufferView<SolveT>                   compare_first_offdiag;
    SocuNativeMatrixView<SolveT>               compare_native_matrix;
    bool                                       compare_enabled = false;
    bool                                       compare_uses_native_matrix = false;

    muda::CBufferView<ABDJacobi> abd_vertex_to_J;
    muda::BufferView<IndexT>     counters;
    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;

    MUDA_GENERIC bool native_enabled() const noexcept
    {
        return use_native_matrix && native_matrix.D.data() != nullptr
               && native_matrix.block_count != 0 && native_matrix.block_size != 0;
    }

    MUDA_GENERIC SocuNativeContactExactWriter<StoreT, SolveT> writer() const noexcept
    {
        return SocuNativeContactExactWriter<StoreT, SolveT>{
            native_matrix,
            native_dof_descriptors,
            old_to_chain,
            horizon,
            block_size,
            use_native_matrix,
            compare_diag,
            compare_first_offdiag,
            compare_native_matrix,
            compare_enabled,
            compare_uses_native_matrix,
            abd_vertex_to_J,
            counters};
    }
};
}  // namespace uipc::backend::cuda_mixed
