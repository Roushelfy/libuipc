#pragma once

#include <utils/matrix_assembler.h>
#include <linear_system/structured_chain_provider.h>
#include <muda/atomic.h>
#include <muda/buffer/buffer_view.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT, int BlockDim>
struct TripletAssemblySink
{
    muda::DoubletVectorViewer<StoreT, BlockDim> gradients;
    muda::TripletMatrixViewer<StoreT, BlockDim> hessians;
    bool                                        gradient_only = false;

    TripletAssemblySink(muda::DoubletVectorView<StoreT, BlockDim> gradients_view,
                        muda::TripletMatrixView<StoreT, BlockDim> hessians_view,
                        bool gradient_only) noexcept
        : gradients(gradients_view.viewer())
        , hessians(hessians_view.viewer())
        , gradient_only(gradient_only)
    {
    }

    template <int StencilSize, typename GVec>
    MUDA_DEVICE __forceinline__ void write_gradient(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const GVec&                               G) const
    {
        DoubletVectorAssembler dva{gradients};
        dva.template segment<StencilSize>(event_offset).write(indices, G);
    }

    template <int StencilSize, typename GVec>
    MUDA_DEVICE __forceinline__ void write_gradient(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const Eigen::Vector<int8_t, StencilSize>& ignore,
        const GVec&                               G) const
    {
        DoubletVectorAssembler dva{gradients};
        dva.template segment<StencilSize>(event_offset).write(indices, ignore, G);
    }

    template <typename GVec>
    MUDA_DEVICE __forceinline__ void write_gradient(IndexT     event_offset,
                                                    IndexT     index,
                                                    const GVec& G) const
    {
        DoubletVectorAssembler dva{gradients};
        dva(event_offset).write(index, G);
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE __forceinline__ void write_hessian_half(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const HMat&                               H) const
    {
        if(gradient_only)
            return;

        TripletMatrixAssembler tma{hessians};
        tma.template half_block<StencilSize>(event_offset).write(indices, H);
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE __forceinline__ void write_hessian_half(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const Eigen::Vector<int8_t, StencilSize>& ignore,
        const HMat&                               H) const
    {
        if(gradient_only)
            return;

        TripletMatrixAssembler tma{hessians};
        tma.template half_block<StencilSize>(event_offset).write(indices, ignore, H);
    }

    template <typename HMat>
    MUDA_DEVICE __forceinline__ void write_hessian(IndexT     event_offset,
                                                   IndexT     index,
                                                   const HMat& H) const
    {
        if(gradient_only)
            return;

        TripletMatrixAssembler tma{hessians};
        tma(event_offset).write(index, H);
    }
};

template <typename StoreT, typename SolveT>
struct StructuredDeviceAssemblySink
{
    muda::BufferView<SolveT>  diag;
    muda::BufferView<SolveT>  first_offdiag;
    muda::CBufferView<IndexT> old_to_chain;
    SizeT                     horizon    = 0;
    SizeT                     block_size = 0;

    MUDA_GENERIC bool valid() const noexcept
    {
        return diag.data() != nullptr && old_to_chain.data() != nullptr
               && block_size != 0;
    }

    MUDA_DEVICE void add_hessian_scalar(IndexT old_i,
                                        IndexT old_j,
                                        StoreT value) const noexcept
    {
        if(old_i < 0 || old_j < 0)
            return;
        if(static_cast<SizeT>(old_i) >= old_to_chain.size()
           || static_cast<SizeT>(old_j) >= old_to_chain.size())
            return;

        const IndexT chain_i = old_to_chain[static_cast<SizeT>(old_i)];
        const IndexT chain_j = old_to_chain[static_cast<SizeT>(old_j)];
        if(chain_i < 0 || chain_j < 0)
            return;

        const SizeT ci = static_cast<SizeT>(chain_i);
        const SizeT cj = static_cast<SizeT>(chain_j);
        const SizeT bi = ci / block_size;
        const SizeT bj = cj / block_size;
        const SizeT li = ci % block_size;
        const SizeT lj = cj % block_size;
        if(bi >= horizon || bj >= horizon)
            return;

        const SolveT v = static_cast<SolveT>(value);
        if(bi == bj)
        {
            const SizeT index = (bi * block_size + li) * block_size + lj;
            muda::atomic_add(diag.data(index), v);
            return;
        }

        const SizeT distance = bi > bj ? bi - bj : bj - bi;
        if(distance != 1 || first_offdiag.data() == nullptr)
            return;

        const bool  ij_is_forward = bi < bj;
        const SizeT left_block    = ij_is_forward ? bi : bj;
        const SizeT row           = ij_is_forward ? li : lj;
        const SizeT col           = ij_is_forward ? lj : li;
        const SizeT index = (left_block * block_size + row) * block_size + col;
        if(index < first_offdiag.size())
            muda::atomic_add(first_offdiag.data(index), v);
    }

    template <typename HMat>
    MUDA_DEVICE void add_dense_block(IndexT old_dof_begin, const HMat& H) const noexcept
    {
        for(IndexT row = 0; row < H.rows(); ++row)
        {
            for(IndexT col = 0; col < H.cols(); ++col)
            {
                add_hessian_scalar(old_dof_begin + row,
                                   old_dof_begin + col,
                                   static_cast<StoreT>(H(row, col)));
            }
        }
    }
};
}  // namespace uipc::backend::cuda_mixed
