#pragma once

#include <utils/matrix_assembler.h>
#include <linear_system/structured_chain_provider.h>
#include <muda/atomic.h>
#include <muda/buffer/buffer_view.h>

namespace uipc::backend::cuda_mixed
{
enum class StructuredSinkWriteClass : unsigned char
{
    Skipped,
    Diag,
    FirstOffdiag,
    OffBand,
};

struct RuntimeOrderingEdge
{
    IndexT atom_a = -1;
    IndexT atom_b = -1;
    double abs_weight = 0.0;
};

struct RuntimeOrderingCollector
{
    muda::BufferView<RuntimeOrderingEdge> edges;
    muda::BufferView<IndexT>              cursor;
    muda::CBufferView<IndexT>             old_dof_to_atom;
    bool                                  enabled = false;

    MUDA_GENERIC bool valid() const noexcept
    {
        return enabled && edges.data() != nullptr && cursor.data() != nullptr
               && cursor.size() >= 2 && old_dof_to_atom.data() != nullptr;
    }
};

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
        const Eigen::Vector<IndexT, StencilSize>& row_indices,
        const Eigen::Vector<IndexT, StencilSize>& col_indices,
        const HMat&                               H) const
    {
        if(gradient_only)
            return;

        TripletMatrixAssembler tma{hessians};
        tma.template half_block<StencilSize>(event_offset)
            .write(row_indices, col_indices, H);
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
    muda::BufferView<IndexT>  counters;
    RuntimeOrderingCollector  runtime_ordering;

    MUDA_GENERIC bool valid() const noexcept
    {
        return diag.data() != nullptr && old_to_chain.data() != nullptr
               && block_size != 0;
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    add_hessian_scalar_status(IndexT old_i, IndexT old_j, StoreT value) const noexcept
    {
        record_runtime_ordering_edge(old_i, old_j, value);

        if(old_i < 0 || old_j < 0)
            return StructuredSinkWriteClass::Skipped;
        if(static_cast<SizeT>(old_i) >= old_to_chain.size()
           || static_cast<SizeT>(old_j) >= old_to_chain.size())
            return StructuredSinkWriteClass::Skipped;

        const IndexT chain_i = old_to_chain[static_cast<SizeT>(old_i)];
        const IndexT chain_j = old_to_chain[static_cast<SizeT>(old_j)];
        if(chain_i < 0 || chain_j < 0)
            return StructuredSinkWriteClass::Skipped;

        const SizeT ci = static_cast<SizeT>(chain_i);
        const SizeT cj = static_cast<SizeT>(chain_j);
        const SizeT bi = ci / block_size;
        const SizeT bj = cj / block_size;
        const SizeT li = ci % block_size;
        const SizeT lj = cj % block_size;
        if(bi >= horizon || bj >= horizon)
            return StructuredSinkWriteClass::Skipped;

        const SolveT v = static_cast<SolveT>(value);
        if(bi == bj)
        {
            const SizeT index = (bi * block_size + li) * block_size + lj;
            muda::atomic_add(diag.data(index), v);
            return StructuredSinkWriteClass::Diag;
        }

        const SizeT distance = bi > bj ? bi - bj : bj - bi;
        if(distance != 1 || first_offdiag.data() == nullptr)
            return StructuredSinkWriteClass::OffBand;

        const bool  ij_is_forward = bi < bj;
        const SizeT left_block    = ij_is_forward ? bi : bj;
        const SizeT row           = ij_is_forward ? lj : li;
        const SizeT col           = ij_is_forward ? li : lj;
        const SizeT index = (left_block * block_size + row) * block_size + col;
        if(index < first_offdiag.size())
        {
            muda::atomic_add(first_offdiag.data(index), v);
            return StructuredSinkWriteClass::FirstOffdiag;
        }
        return StructuredSinkWriteClass::Skipped;
    }

    MUDA_DEVICE __forceinline__ void record_runtime_ordering_edge(
        IndexT old_i,
        IndexT old_j,
        StoreT value) const noexcept
    {
        if(!runtime_ordering.valid() || old_i < 0 || old_j < 0)
            return;
        if(static_cast<SizeT>(old_i) >= runtime_ordering.old_dof_to_atom.size()
           || static_cast<SizeT>(old_j) >= runtime_ordering.old_dof_to_atom.size())
            return;

        IndexT atom_i = runtime_ordering.old_dof_to_atom[static_cast<SizeT>(old_i)];
        IndexT atom_j = runtime_ordering.old_dof_to_atom[static_cast<SizeT>(old_j)];
        if(atom_i < 0 || atom_j < 0 || atom_i == atom_j)
            return;
        if(atom_i > atom_j)
        {
            const IndexT tmp = atom_i;
            atom_i = atom_j;
            atom_j = tmp;
        }

        const IndexT slot = muda::atomic_add(runtime_ordering.cursor.data(0), IndexT{1});
        if(static_cast<SizeT>(slot) >= runtime_ordering.edges.size())
        {
            muda::atomic_add(runtime_ordering.cursor.data(1), IndexT{1});
            return;
        }

        const auto v = static_cast<double>(value);
        runtime_ordering.edges.data(slot)->atom_a = atom_i;
        runtime_ordering.edges.data(slot)->atom_b = atom_j;
        runtime_ordering.edges.data(slot)->abs_weight = v < 0.0 ? -v : v;
    }

    MUDA_DEVICE __forceinline__ void add_hessian_scalar(IndexT old_i,
                                                        IndexT old_j,
                                                        StoreT value) const noexcept
    {
        (void)add_hessian_scalar_status(old_i, old_j, value);
    }

    MUDA_DEVICE __forceinline__ void record_off_band_drop() const noexcept
    {
        if(counters.data() != nullptr && counters.size() > 2)
            muda::atomic_add(counters.data(2), IndexT{1});
    }

    template <typename HMat>
    MUDA_DEVICE __forceinline__ void add_dense_block(IndexT old_dof_begin,
                                                     const HMat& H) const noexcept
    {
        for(IndexT row = 0; row < H.rows(); ++row)
        {
            for(IndexT col = row; col < H.cols(); ++col)
            {
                const IndexT old_i = old_dof_begin + row;
                const IndexT old_j = old_dof_begin + col;
                const auto cls = add_hessian_scalar_status(
                    old_i,
                    old_j,
                    static_cast<StoreT>(H(row, col)));
                if(cls == StructuredSinkWriteClass::Diag && old_i != old_j)
                    add_hessian_scalar(old_j, old_i, static_cast<StoreT>(H(row, col)));
            }
        }
    }

    template <int Rows, int Cols, typename HMat>
    MUDA_DEVICE __forceinline__ void add_dense_block_fixed(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
#pragma unroll
        for(IndexT row = 0; row < Rows; ++row)
        {
#pragma unroll
            for(IndexT col = row; col < Cols; ++col)
            {
                const IndexT old_i = old_dof_begin + row;
                const IndexT old_j = old_dof_begin + col;
                const auto cls = add_hessian_scalar_status(
                    old_i,
                    old_j,
                    static_cast<StoreT>(H(row, col)));
                if(cls == StructuredSinkWriteClass::Diag && old_i != old_j)
                    add_hessian_scalar(old_j, old_i, static_cast<StoreT>(H(row, col)));
            }
        }
    }

    template <int SubBlockDim, int SubBlockCount, typename HMat>
    MUDA_DEVICE __forceinline__ void add_dense_block_upper_subblocks_fixed(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
#pragma unroll
        for(IndexT row_block = 0; row_block < SubBlockCount; ++row_block)
        {
#pragma unroll
            for(IndexT col_block = row_block; col_block < SubBlockCount; ++col_block)
            {
#pragma unroll
                for(IndexT row = 0; row < SubBlockDim; ++row)
                {
#pragma unroll
                    for(IndexT col = 0; col < SubBlockDim; ++col)
                    {
                        const IndexT local_i = row_block * SubBlockDim + row;
                        const IndexT local_j = col_block * SubBlockDim + col;
                        const IndexT old_i   = old_dof_begin + local_i;
                        const IndexT old_j   = old_dof_begin + local_j;
                        const auto   value   = static_cast<StoreT>(H(local_i, local_j));
                        const auto cls =
                            add_hessian_scalar_status(old_i, old_j, value);
                        if(row_block != col_block
                           && cls == StructuredSinkWriteClass::Diag)
                        {
                            add_hessian_scalar(old_j, old_i, value);
                        }
                    }
                }
            }
        }
    }

    template <int Rows, int Cols, typename HMat>
    MUDA_DEVICE __forceinline__ void add_dense_block_between_fixed(
        IndexT old_row_dof_begin,
        IndexT old_col_dof_begin,
        const HMat& H) const noexcept
    {
#pragma unroll
        for(IndexT row = 0; row < Rows; ++row)
        {
#pragma unroll
            for(IndexT col = 0; col < Cols; ++col)
            {
                add_hessian_scalar(old_row_dof_begin + row,
                                   old_col_dof_begin + col,
                                   static_cast<StoreT>(H(row, col)));
            }
        }
    }
};

template <typename StoreT, typename SolveT, int BlockDim>
struct LocalAssemblySink
{
    muda::DoubletVectorViewer<StoreT, BlockDim> gradients;
    muda::TripletMatrixViewer<StoreT, BlockDim> hessians;
    StructuredDeviceAssemblySink<StoreT, SolveT> structured;
    muda::CBufferView<IndexT> fixed_blocks;
    IndexT old_dof_offset = 0;
    bool   gradient_only  = false;
    bool   write_gradients = true;
    bool   identity_fixed_diagonal = false;

    LocalAssemblySink(muda::DoubletVectorView<StoreT, BlockDim> gradients_view,
                      muda::TripletMatrixView<StoreT, BlockDim> hessians_view,
                      bool gradient_only) noexcept
        : gradients(gradients_view.viewer())
        , hessians(hessians_view.viewer())
        , gradient_only(gradient_only)
    {
    }

    LocalAssemblySink(muda::DoubletVectorView<StoreT, BlockDim> gradients_view,
                      muda::TripletMatrixView<StoreT, BlockDim> hessians_view,
                      bool gradient_only,
                      StructuredDeviceAssemblySink<StoreT, SolveT> structured,
                      IndexT old_dof_offset,
                      muda::CBufferView<IndexT> fixed_blocks = {},
                      bool identity_fixed_diagonal = false,
                      bool write_gradients = true) noexcept
        : gradients(gradients_view.viewer())
        , hessians(hessians_view.viewer())
        , structured(structured)
        , fixed_blocks(fixed_blocks)
        , old_dof_offset(old_dof_offset)
        , gradient_only(gradient_only)
        , write_gradients(write_gradients)
        , identity_fixed_diagonal(identity_fixed_diagonal)
    {
    }

    MUDA_GENERIC bool structured_enabled() const noexcept
    {
        return structured.valid();
    }

    MUDA_DEVICE __forceinline__ bool fixed(IndexT block) const noexcept
    {
        return fixed_blocks.data() != nullptr && block >= 0
               && static_cast<SizeT>(block) < fixed_blocks.size()
               && fixed_blocks[static_cast<SizeT>(block)] != 0;
    }

    template <int StencilSize, typename GVec>
    MUDA_DEVICE __forceinline__ void write_gradient(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const GVec&                               G) const
    {
        if(!write_gradients)
            return;

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
        if(!write_gradients)
            return;

        DoubletVectorAssembler dva{gradients};
        dva.template segment<StencilSize>(event_offset).write(indices, ignore, G);
    }

    template <typename GVec>
    MUDA_DEVICE __forceinline__ void write_gradient(IndexT     event_offset,
                                                    IndexT     index,
                                                    const GVec& G) const
    {
        if(!write_gradients)
            return;

        DoubletVectorAssembler dva{gradients};
        dva(event_offset).write(index, G);
    }

    MUDA_DEVICE __forceinline__ void record_write_class(
        StructuredSinkWriteClass cls) const noexcept
    {
        if(cls == StructuredSinkWriteClass::OffBand)
            structured.record_off_band_drop();
    }

    template <typename HBlock>
    MUDA_DEVICE __forceinline__ void add_structured_block(
        IndexT        block_i,
        IndexT        block_j,
        const HBlock& H,
        bool          mirror_when_same_structured_block = false) const noexcept
    {
        const bool i_fixed = fixed(block_i);
        const bool j_fixed = fixed(block_j);
        if(i_fixed || j_fixed)
        {
            if(!(identity_fixed_diagonal && block_i == block_j))
                return;

#pragma unroll
            for(IndexT d = 0; d < BlockDim; ++d)
            {
                const auto cls = structured.add_hessian_scalar_status(
                    old_dof_offset + block_i * BlockDim + d,
                    old_dof_offset + block_j * BlockDim + d,
                    StoreT{1});
                record_write_class(cls);
            }
            return;
        }

#pragma unroll
        for(IndexT row = 0; row < BlockDim; ++row)
        {
#pragma unroll
            for(IndexT col = 0; col < BlockDim; ++col)
            {
                const IndexT old_i = old_dof_offset + block_i * BlockDim + row;
                const IndexT old_j = old_dof_offset + block_j * BlockDim + col;
                const auto cls = structured.add_hessian_scalar_status(
                    old_i,
                    old_j,
                    static_cast<StoreT>(H(row, col)));
                record_write_class(cls);

                if(mirror_when_same_structured_block
                   && cls == StructuredSinkWriteClass::Diag && old_i != old_j)
                {
                    record_write_class(structured.add_hessian_scalar_status(
                        old_j,
                        old_i,
                        static_cast<StoreT>(H(row, col))));
                }
            }
        }
    }

    MUDA_DEVICE __forceinline__ bool upper_lr(IndexT left_value,
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

    template <int StencilSize, typename HMat>
    MUDA_DEVICE __forceinline__ void write_hessian_half(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& indices,
        const HMat&                               H) const
    {
        if(gradient_only)
            return;

        if(structured_enabled())
        {
            (void)event_offset;
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
                    add_structured_block(
                        indices(L),
                        indices(R),
                        H.template block<BlockDim, BlockDim>(L * BlockDim,
                                                             R * BlockDim),
                        indices(L) != indices(R) || swapped);
                }
            }
            return;
        }

        TripletMatrixAssembler tma{hessians};
        tma.template half_block<StencilSize>(event_offset).write(indices, H);
    }

    template <int StencilSize, typename HMat>
    MUDA_DEVICE __forceinline__ void write_hessian_half(
        IndexT event_offset,
        const Eigen::Vector<IndexT, StencilSize>& row_indices,
        const Eigen::Vector<IndexT, StencilSize>& col_indices,
        const HMat&                               H) const
    {
        if(gradient_only)
            return;

        if(structured_enabled())
        {
            (void)event_offset;
#pragma unroll
            for(IndexT row_block = 0; row_block < StencilSize; ++row_block)
            {
#pragma unroll
                for(IndexT col_block = row_block; col_block < StencilSize; ++col_block)
                {
                    IndexT L = row_block;
                    IndexT R = col_block;
                    const bool swapped = upper_lr(row_indices(row_block),
                                                  col_indices(col_block),
                                                  row_block,
                                                  col_block,
                                                  L,
                                                  R);
                    add_structured_block(
                        row_indices(L),
                        col_indices(R),
                        H.template block<BlockDim, BlockDim>(L * BlockDim,
                                                             R * BlockDim),
                        row_indices(L) != col_indices(R) || swapped);
                }
            }
            return;
        }

        TripletMatrixAssembler tma{hessians};
        tma.template half_block<StencilSize>(event_offset)
            .write(row_indices, col_indices, H);
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

        if(structured_enabled())
        {
            (void)event_offset;
#pragma unroll
            for(IndexT row_block = 0; row_block < StencilSize; ++row_block)
            {
                if(ignore(row_block))
                    continue;
#pragma unroll
                for(IndexT col_block = row_block; col_block < StencilSize; ++col_block)
                {
                    if(ignore(col_block))
                        continue;
                    IndexT L = row_block;
                    IndexT R = col_block;
                    const bool swapped = upper_lr(indices(row_block),
                                                  indices(col_block),
                                                  row_block,
                                                  col_block,
                                                  L,
                                                  R);
                    add_structured_block(
                        indices(L),
                        indices(R),
                        H.template block<BlockDim, BlockDim>(L * BlockDim,
                                                             R * BlockDim),
                        indices(L) != indices(R) || swapped);
                }
            }
            return;
        }

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

        if(structured_enabled())
        {
            (void)event_offset;
            add_structured_block(index, index, H);
            return;
        }

        TripletMatrixAssembler tma{hessians};
        tma(event_offset).write(index, H);
    }
};
}  // namespace uipc::backend::cuda_mixed
