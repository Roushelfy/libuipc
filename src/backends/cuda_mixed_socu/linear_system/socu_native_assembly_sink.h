#pragma once

#include <linear_system/socu_native_matrix_builder.h>
#include <utils/assembly_sink.h>
#include <utils/structured_assembly_counters.h>

namespace uipc::backend::cuda_mixed
{
enum class StructuredNativeDenseBlockTarget : unsigned char
{
    Miss,
    SameBlock,
    AdjacentBlock,
};

template <typename StoreT, typename SolveT>
struct SocuNativeStructuredDeviceMatrixSink
{
    muda::BufferView<SolveT>  diag;
    muda::BufferView<SolveT>  first_offdiag;
    muda::CBufferView<IndexT> old_to_chain;
    SizeT                     horizon    = 0;
    SizeT                     block_size = 0;
    muda::BufferView<IndexT>  counters;
    SocuNativeMatrixView<SolveT> native_matrix;
    muda::CBufferView<SocuNativeDofDescriptor> native_dof_descriptors;
    bool use_native_matrix = false;
    muda::BufferView<SolveT>  compare_diag;
    muda::BufferView<SolveT>  compare_first_offdiag;
    SocuNativeMatrixView<SolveT> compare_native_matrix;
    bool compare_enabled = false;
    bool compare_uses_native_matrix = false;

    MUDA_GENERIC bool valid() const noexcept
    {
        return diag.data() != nullptr && old_to_chain.data() != nullptr
               && block_size != 0;
    }

    MUDA_GENERIC bool native_enabled() const noexcept
    {
        return use_native_matrix && native_dof_descriptors.data() != nullptr
               && native_matrix.D.data() != nullptr && native_matrix.block_count != 0
               && native_matrix.block_size != 0;
    }

    MUDA_GENERIC bool compare_native_enabled() const noexcept
    {
        return compare_enabled && compare_uses_native_matrix
               && native_dof_descriptors.data() != nullptr
               && compare_native_matrix.D.data() != nullptr
               && compare_native_matrix.block_count != 0
               && compare_native_matrix.block_size != 0;
    }

    MUDA_GENERIC bool compare_legacy_enabled() const noexcept
    {
        return compare_enabled && !compare_uses_native_matrix
               && compare_diag.data() != nullptr;
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    classify_dof_pair(IndexT old_i, IndexT old_j) const noexcept
    {
        if(native_enabled())
            return classify_dof_pair_native(old_i, old_j);
        return classify_dof_pair_legacy(old_i, old_j);
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    classify_dof_pair_legacy(IndexT old_i, IndexT old_j) const noexcept
    {
        if(old_i < 0 || old_j < 0)
            return StructuredSinkWriteClass::Skipped;
        if(static_cast<SizeT>(old_i) >= old_to_chain.size()
           || static_cast<SizeT>(old_j) >= old_to_chain.size())
            return StructuredSinkWriteClass::Skipped;

        const IndexT chain_i = old_to_chain[static_cast<SizeT>(old_i)];
        const IndexT chain_j = old_to_chain[static_cast<SizeT>(old_j)];
        if(chain_i < 0 || chain_j < 0)
            return StructuredSinkWriteClass::Skipped;

        const SizeT bi = static_cast<SizeT>(chain_i) / block_size;
        const SizeT bj = static_cast<SizeT>(chain_j) / block_size;
        if(bi >= horizon || bj >= horizon)
            return StructuredSinkWriteClass::Skipped;
        if(bi == bj)
            return StructuredSinkWriteClass::Diag;

        const SizeT distance = bi > bj ? bi - bj : bj - bi;
        return distance == 1 && first_offdiag.data() != nullptr
                   ? StructuredSinkWriteClass::FirstOffdiag
                   : StructuredSinkWriteClass::OffBand;
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    classify_dof_pair_native(IndexT old_i, IndexT old_j) const noexcept
    {
        if(old_i < 0 || old_j < 0)
            return StructuredSinkWriteClass::Skipped;
        if(static_cast<SizeT>(old_i) >= native_dof_descriptors.size()
           || static_cast<SizeT>(old_j) >= native_dof_descriptors.size())
            return StructuredSinkWriteClass::Skipped;

        const auto dof_i = native_dof_descriptors[static_cast<SizeT>(old_i)];
        const auto dof_j = native_dof_descriptors[static_cast<SizeT>(old_j)];
        if(!native_matrix.valid_dof_descriptor(dof_i)
           || !native_matrix.valid_dof_descriptor(dof_j))
            return StructuredSinkWriteClass::Skipped;

        if(dof_i.block == dof_j.block)
            return StructuredSinkWriteClass::Diag;

        const SizeT distance = dof_i.block > dof_j.block
                                   ? dof_i.block - dof_j.block
                                   : dof_j.block - dof_i.block;
        return distance == 1 && native_matrix.E.data() != nullptr
                   ? StructuredSinkWriteClass::FirstOffdiag
                   : StructuredSinkWriteClass::OffBand;
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    add_hessian_scalar_status_legacy(muda::BufferView<SolveT> target_diag,
                                     muda::BufferView<SolveT> target_first_offdiag,
                                     IndexT old_i,
                                     IndexT old_j,
                                     StoreT value) const noexcept
    {
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
            if(index >= target_diag.size())
                return StructuredSinkWriteClass::Skipped;
            muda::atomic_add(target_diag.data(index), v);
            return StructuredSinkWriteClass::Diag;
        }

        const SizeT distance = bi > bj ? bi - bj : bj - bi;
        if(distance != 1 || target_first_offdiag.data() == nullptr)
            return StructuredSinkWriteClass::OffBand;

        const bool  ij_is_forward = bi < bj;
        const SizeT left_block    = ij_is_forward ? bi : bj;
        const SizeT row           = ij_is_forward ? lj : li;
        const SizeT col           = ij_is_forward ? li : lj;
        const SizeT index = (left_block * block_size + row) * block_size + col;
        if(index < target_first_offdiag.size())
        {
            muda::atomic_add(target_first_offdiag.data(index), v);
            return StructuredSinkWriteClass::FirstOffdiag;
        }
        return StructuredSinkWriteClass::Skipped;
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    add_hessian_scalar_status_native(SocuNativeMatrixView<SolveT> target,
                                     IndexT old_i,
                                     IndexT old_j,
                                     StoreT value) const noexcept
    {
        if(old_i < 0 || old_j < 0)
            return StructuredSinkWriteClass::Skipped;
        if(static_cast<SizeT>(old_i) >= native_dof_descriptors.size()
           || static_cast<SizeT>(old_j) >= native_dof_descriptors.size())
            return StructuredSinkWriteClass::Skipped;

        const auto dof_i = native_dof_descriptors[static_cast<SizeT>(old_i)];
        const auto dof_j = native_dof_descriptors[static_cast<SizeT>(old_j)];
        if(!target.valid_dof_descriptor(dof_i)
           || !target.valid_dof_descriptor(dof_j))
            return StructuredSinkWriteClass::Skipped;

        const SolveT v = static_cast<SolveT>(value);
        if(dof_i.block == dof_j.block)
        {
            const SizeT index =
                target.diag_index(dof_i.block, dof_i.lane, dof_j.lane);
            if(index >= target.D.size())
                return StructuredSinkWriteClass::Skipped;
            muda::atomic_add(target.D.data(index), v);
            return StructuredSinkWriteClass::Diag;
        }

        const SizeT distance = dof_i.block > dof_j.block
                                   ? dof_i.block - dof_j.block
                                   : dof_j.block - dof_i.block;
        if(distance != 1 || target.E.data() == nullptr)
            return StructuredSinkWriteClass::OffBand;

        const bool ij_is_forward = dof_i.block < dof_j.block;
        const SizeT left_block = ij_is_forward ? dof_i.block : dof_j.block;
        const SizeT row = ij_is_forward ? dof_j.lane : dof_i.lane;
        const SizeT col = ij_is_forward ? dof_i.lane : dof_j.lane;
        if(left_block >= target.first_offdiag_block_count
           || row >= target.block_size || col >= target.block_size)
            return StructuredSinkWriteClass::Skipped;
        const SizeT index = target.first_offdiag_index(left_block, row, col);
        if(index >= target.E.size())
            return StructuredSinkWriteClass::Skipped;
        muda::atomic_add(target.E.data(index), v);
        return StructuredSinkWriteClass::FirstOffdiag;
    }

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    add_hessian_scalar_status(IndexT old_i, IndexT old_j, StoreT value) const noexcept
    {
        StructuredSinkWriteClass primary_class = StructuredSinkWriteClass::Skipped;
        if(native_enabled())
        {
            primary_class =
                add_hessian_scalar_status_native(native_matrix, old_i, old_j, value);
        }
        else
        {
            primary_class = add_hessian_scalar_status_legacy(
                diag,
                first_offdiag,
                old_i,
                old_j,
                value);
        }

        add_hessian_scalar_compare(old_i, old_j, value);
        return primary_class;
    }

    MUDA_DEVICE __forceinline__ void add_hessian_scalar_compare(IndexT old_i,
                                                                IndexT old_j,
                                                                StoreT value) const noexcept
    {
        if(compare_native_enabled())
        {
            (void)add_hessian_scalar_status_native(
                compare_native_matrix,
                old_i,
                old_j,
                value);
        }
        else if(compare_legacy_enabled())
        {
            (void)add_hessian_scalar_status_legacy(compare_diag,
                                                   compare_first_offdiag,
                                                   old_i,
                                                   old_j,
                                                   value);
        }
    }

    MUDA_DEVICE __forceinline__ void add_hessian_scalar(IndexT old_i,
                                                        IndexT old_j,
                                                        StoreT value) const noexcept
    {
        (void)add_hessian_scalar_status(old_i, old_j, value);
    }

    MUDA_DEVICE __forceinline__ void record_off_band_drop() const noexcept
    {
        record_counter(StructuredAssemblyCounterSlot::ContactOffBandScalarDrop);
    }

    MUDA_DEVICE __forceinline__ void record_counter(
        StructuredAssemblyCounterSlot slot,
        IndexT amount = 1) const noexcept
    {
        const IndexT index = static_cast<IndexT>(slot);
        if(counters.data() != nullptr && index >= 0
           && static_cast<SizeT>(index) < counters.size())
            muda::atomic_add(counters.data(static_cast<SizeT>(index)), amount);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_same_block_dense_hit() const noexcept
    {
        record_counter(
            StructuredAssemblyCounterSlot::NativeChainBaseSameBlockDenseHit);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_adjacent_dense_hit() const noexcept
    {
        record_counter(
            StructuredAssemblyCounterSlot::NativeChainBaseAdjacentDenseHit);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_dense_miss() const noexcept
    {
        record_counter(StructuredAssemblyCounterSlot::NativeChainBaseDenseMiss);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_scalar_fallback() const noexcept
    {
        record_counter(
            StructuredAssemblyCounterSlot::NativeChainBaseScalarFallback);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_diag3x3_hit() const noexcept
    {
        record_counter(StructuredAssemblyCounterSlot::NativeChainBaseDiag3x3Hit);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_diag3x3_miss() const noexcept
    {
        record_counter(StructuredAssemblyCounterSlot::NativeChainBaseDiag3x3Miss);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_pair3x3_same_block_hit() const noexcept
    {
        record_counter(
            StructuredAssemblyCounterSlot::NativeChainBasePair3x3SameBlockHit);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_pair3x3_adjacent_hit() const noexcept
    {
        record_counter(
            StructuredAssemblyCounterSlot::NativeChainBasePair3x3AdjacentHit);
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_pair3x3_miss() const noexcept
    {
        record_counter(StructuredAssemblyCounterSlot::NativeChainBasePair3x3Miss);
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

    template <int Rows, int Cols, typename HMat>
    MUDA_DEVICE __forceinline__ bool try_add_native_same_block_dense_block_fixed(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
        static_assert(Rows > 0);
        static_assert(Cols > 0);
        static_assert(Rows == Cols);

        if(!native_enabled() || old_dof_begin < 0)
            return false;
        const SizeT old_begin = static_cast<SizeT>(old_dof_begin);
        if(old_begin + static_cast<SizeT>(Rows) > native_dof_descriptors.size())
            return false;

        const auto first = native_dof_descriptors[old_begin];
        if(!native_matrix.valid_dof_descriptor(first)
           || first.old_dof != old_dof_begin)
            return false;

        const SizeT block = first.block;
        SizeT       lanes[Rows];
        lanes[0] = first.lane;

#pragma unroll
        for(IndexT local = 1; local < Rows; ++local)
        {
            const auto dof =
                native_dof_descriptors[old_begin + static_cast<SizeT>(local)];
            if(!native_matrix.valid_dof_descriptor(dof)
               || dof.old_dof != old_dof_begin + local
               || dof.block != block)
                return false;
            lanes[local] = dof.lane;
        }

        if(native_matrix.diag_index(block,
                                    native_matrix.block_size - 1,
                                    native_matrix.block_size - 1)
           >= native_matrix.D.size())
            return false;

#pragma unroll
        for(IndexT row = 0; row < Rows; ++row)
        {
#pragma unroll
            for(IndexT col = 0; col < Cols; ++col)
            {
                const auto value = static_cast<StoreT>(H(row, col));
                muda::atomic_add(native_matrix.D.data(native_matrix.diag_index(
                                     block,
                                     lanes[row],
                                     lanes[col])),
                                 static_cast<SolveT>(value));
                add_hessian_scalar_compare(old_dof_begin + row,
                                           old_dof_begin + col,
                                           value);
            }
        }

        return true;
    }

    template <typename HMat>
    MUDA_DEVICE __forceinline__ StructuredNativeDenseBlockTarget
    try_add_native_pair3x3_block_target(
        IndexT old_row_dof_begin,
        IndexT old_col_dof_begin,
        const HMat& H,
        bool mirror_when_same_native_block) const noexcept
    {
        if(!native_enabled() || old_row_dof_begin < 0 || old_col_dof_begin < 0)
            return StructuredNativeDenseBlockTarget::Miss;
        const SizeT old_row_begin = static_cast<SizeT>(old_row_dof_begin);
        const SizeT old_col_begin = static_cast<SizeT>(old_col_dof_begin);
        if(old_row_begin + SizeT{3} > native_dof_descriptors.size()
           || old_col_begin + SizeT{3} > native_dof_descriptors.size())
            return StructuredNativeDenseBlockTarget::Miss;

        SizeT row_lanes[3];
        SizeT col_lanes[3];

        const auto first_row = native_dof_descriptors[old_row_begin];
        const auto first_col = native_dof_descriptors[old_col_begin];
        if(!native_matrix.valid_dof_descriptor(first_row)
           || !native_matrix.valid_dof_descriptor(first_col)
           || first_row.old_dof != old_row_dof_begin
           || first_col.old_dof != old_col_dof_begin)
            return StructuredNativeDenseBlockTarget::Miss;

        const SizeT row_block = first_row.block;
        const SizeT col_block = first_col.block;
        row_lanes[0] = first_row.lane;
        col_lanes[0] = first_col.lane;

#pragma unroll
        for(IndexT local = 1; local < 3; ++local)
        {
            const auto row_dof =
                native_dof_descriptors[old_row_begin + static_cast<SizeT>(local)];
            const auto col_dof =
                native_dof_descriptors[old_col_begin + static_cast<SizeT>(local)];
            if(!native_matrix.valid_dof_descriptor(row_dof)
               || !native_matrix.valid_dof_descriptor(col_dof)
               || row_dof.old_dof != old_row_dof_begin + local
               || col_dof.old_dof != old_col_dof_begin + local
               || row_dof.block != row_block || col_dof.block != col_block)
                return StructuredNativeDenseBlockTarget::Miss;
            row_lanes[local] = row_dof.lane;
            col_lanes[local] = col_dof.lane;
        }

        const bool same_block = row_block == col_block;
        const SizeT min_block = row_block < col_block ? row_block : col_block;
        const SizeT max_block = row_block > col_block ? row_block : col_block;
        const bool adjacent_block = max_block == min_block + 1;
        if(!same_block && !adjacent_block)
            return StructuredNativeDenseBlockTarget::Miss;

        if(native_matrix.diag_index(max_block,
                                    native_matrix.block_size - 1,
                                    native_matrix.block_size - 1)
           >= native_matrix.D.size())
            return StructuredNativeDenseBlockTarget::Miss;

        if(adjacent_block)
        {
            if(native_matrix.E.data() == nullptr
               || min_block >= native_matrix.first_offdiag_block_count)
                return StructuredNativeDenseBlockTarget::Miss;
            if(native_matrix.first_offdiag_index(min_block,
                                                 native_matrix.block_size - 1,
                                                 native_matrix.block_size - 1)
               >= native_matrix.E.size())
                return StructuredNativeDenseBlockTarget::Miss;
        }

#pragma unroll
        for(IndexT row = 0; row < 3; ++row)
        {
#pragma unroll
            for(IndexT col = 0; col < 3; ++col)
            {
                const auto value = static_cast<StoreT>(H(row, col));
                const auto v     = static_cast<SolveT>(value);
                const SizeT row_lane = row_lanes[row];
                const SizeT col_lane = col_lanes[col];

                if(same_block)
                {
                    muda::atomic_add(
                        native_matrix.D.data(native_matrix.diag_index(row_block,
                                                                      row_lane,
                                                                      col_lane)),
                        v);
                    add_hessian_scalar_compare(old_row_dof_begin + row,
                                               old_col_dof_begin + col,
                                               value);

                    if(mirror_when_same_native_block
                       && old_row_dof_begin + row != old_col_dof_begin + col)
                    {
                        muda::atomic_add(
                            native_matrix.D.data(native_matrix.diag_index(row_block,
                                                                          col_lane,
                                                                          row_lane)),
                            v);
                        add_hessian_scalar_compare(old_col_dof_begin + col,
                                                   old_row_dof_begin + row,
                                                   value);
                    }
                }
                else
                {
                    const bool row_col_is_forward = row_block < col_block;
                    const SizeT offdiag_row =
                        row_col_is_forward ? col_lane : row_lane;
                    const SizeT offdiag_col =
                        row_col_is_forward ? row_lane : col_lane;
                    muda::atomic_add(
                        native_matrix.E.data(native_matrix.first_offdiag_index(
                            min_block,
                            offdiag_row,
                            offdiag_col)),
                        v);
                    add_hessian_scalar_compare(old_row_dof_begin + row,
                                               old_col_dof_begin + col,
                                               value);
                }
            }
        }

        return same_block ? StructuredNativeDenseBlockTarget::SameBlock
                          : StructuredNativeDenseBlockTarget::AdjacentBlock;
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

    template <int SubBlockDim, int SubBlockCount, typename HMat>
    MUDA_DEVICE __forceinline__ StructuredNativeDenseBlockTarget
    try_add_native_dense_block_upper_subblocks_fixed_target(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
        constexpr IndexT Rows = SubBlockDim * SubBlockCount;
        static_assert(SubBlockDim > 0);
        static_assert(SubBlockCount > 0);

        if(!native_enabled() || old_dof_begin < 0)
            return StructuredNativeDenseBlockTarget::Miss;
        const SizeT old_begin = static_cast<SizeT>(old_dof_begin);
        if(old_begin + static_cast<SizeT>(Rows) > native_dof_descriptors.size())
            return StructuredNativeDenseBlockTarget::Miss;

        const auto first = native_dof_descriptors[old_begin];
        if(!native_matrix.valid_dof_descriptor(first)
           || first.old_dof != old_dof_begin)
            return StructuredNativeDenseBlockTarget::Miss;

        SizeT blocks[Rows];
        SizeT lanes[Rows];
        SizeT min_block = first.block;
        SizeT max_block = first.block;
        blocks[0] = first.block;
        lanes[0]  = first.lane;

#pragma unroll
        for(IndexT local = 1; local < Rows; ++local)
        {
            const auto dof =
                native_dof_descriptors[old_begin + static_cast<SizeT>(local)];
            if(!native_matrix.valid_dof_descriptor(dof)
               || dof.old_dof != old_dof_begin + local
               || dof.lane >= native_matrix.block_size)
                return StructuredNativeDenseBlockTarget::Miss;
            blocks[local] = dof.block;
            lanes[local]  = dof.lane;
            min_block = dof.block < min_block ? dof.block : min_block;
            max_block = dof.block > max_block ? dof.block : max_block;
        }

        const bool same_block     = min_block == max_block;
        const bool adjacent_block = max_block == min_block + 1;
        if(!same_block && !adjacent_block)
            return StructuredNativeDenseBlockTarget::Miss;

        if(native_matrix.diag_index(max_block,
                                    native_matrix.block_size - 1,
                                    native_matrix.block_size - 1)
           >= native_matrix.D.size())
            return StructuredNativeDenseBlockTarget::Miss;

        if(adjacent_block)
        {
            if(native_matrix.E.data() == nullptr
               || min_block >= native_matrix.first_offdiag_block_count)
                return StructuredNativeDenseBlockTarget::Miss;
            if(native_matrix.first_offdiag_index(min_block,
                                                 native_matrix.block_size - 1,
                                                 native_matrix.block_size - 1)
               >= native_matrix.E.size())
                return StructuredNativeDenseBlockTarget::Miss;
        }

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
                        const SizeT  block_i = blocks[local_i];
                        const SizeT  block_j = blocks[local_j];
                        const SizeT row_lane = lanes[local_i];
                        const SizeT col_lane = lanes[local_j];
                        const auto value = static_cast<StoreT>(H(local_i, local_j));
                        const auto v     = static_cast<SolveT>(value);

                        if(block_i == block_j)
                        {
                            muda::atomic_add(
                                native_matrix.D.data(native_matrix.diag_index(
                                    block_i,
                                    row_lane,
                                    col_lane)),
                                v);
                        }
                        else
                        {
                            const bool ij_is_forward = block_i < block_j;
                            const SizeT row_offdiag =
                                ij_is_forward ? col_lane : row_lane;
                            const SizeT col_offdiag =
                                ij_is_forward ? row_lane : col_lane;
                            muda::atomic_add(
                                native_matrix.E.data(
                                    native_matrix.first_offdiag_index(min_block,
                                                                      row_offdiag,
                                                                      col_offdiag)),
                                v);
                        }
                        add_hessian_scalar_compare(old_dof_begin + local_i,
                                                   old_dof_begin + local_j,
                                                   value);

                        if(row_block != col_block && block_i == block_j)
                        {
                            muda::atomic_add(
                                native_matrix.D.data(native_matrix.diag_index(
                                    block_i,
                                    col_lane,
                                    row_lane)),
                                v);
                            add_hessian_scalar_compare(old_dof_begin + local_j,
                                                       old_dof_begin + local_i,
                                                       value);
                        }
                    }
                }
            }
        }
        return same_block ? StructuredNativeDenseBlockTarget::SameBlock
                          : StructuredNativeDenseBlockTarget::AdjacentBlock;
    }

    template <int SubBlockDim, int SubBlockCount, typename HMat>
    MUDA_DEVICE __forceinline__ bool try_add_native_dense_block_upper_subblocks_fixed(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
        return try_add_native_dense_block_upper_subblocks_fixed_target<
                   SubBlockDim,
                   SubBlockCount>(old_dof_begin, H)
               != StructuredNativeDenseBlockTarget::Miss;
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

template <typename StoreT, typename SolveT>
struct SocuNativeStructuredDeviceAssemblySink
{
    SocuNativeStructuredDeviceMatrixSink<StoreT, SolveT> matrix;
    RuntimeOrderingCollector                  runtime_ordering;

    SocuNativeStructuredDeviceAssemblySink() = default;

    SocuNativeStructuredDeviceAssemblySink(muda::BufferView<SolveT>  diag,
                                 muda::BufferView<SolveT>  first_offdiag,
                                 muda::CBufferView<IndexT> old_to_chain,
                                 SizeT                     horizon,
                                 SizeT                     block_size,
                                 muda::BufferView<IndexT>  counters,
                                 RuntimeOrderingCollector  collector = {}) noexcept
        : matrix{diag, first_offdiag, old_to_chain, horizon, block_size, counters}
        , runtime_ordering(collector)
    {
    }

    MUDA_GENERIC operator StructuredDeviceAssemblySink<StoreT, SolveT>() const noexcept
    {
        return StructuredDeviceAssemblySink<StoreT, SolveT>{
            matrix.diag,
            matrix.first_offdiag,
            matrix.old_to_chain,
            matrix.horizon,
            matrix.block_size,
            matrix.counters,
            runtime_ordering};
    }

    MUDA_GENERIC bool valid() const noexcept { return matrix.valid(); }

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

    MUDA_DEVICE __forceinline__ StructuredSinkWriteClass
    add_hessian_scalar_status(IndexT old_i, IndexT old_j, StoreT value) const noexcept
    {
        if(!runtime_ordering.topology_only)
            record_runtime_ordering_edge(old_i, old_j, value);
        if(runtime_ordering.graph_only)
            return StructuredSinkWriteClass::Skipped;
        return matrix.add_hessian_scalar_status(old_i, old_j, value);
    }

    MUDA_DEVICE __forceinline__ void add_hessian_scalar(IndexT old_i,
                                                        IndexT old_j,
                                                        StoreT value) const noexcept
    {
        (void)add_hessian_scalar_status(old_i, old_j, value);
    }

    MUDA_DEVICE __forceinline__ void record_off_band_drop() const noexcept
    {
        matrix.record_off_band_drop();
    }

    MUDA_DEVICE __forceinline__ void
    record_native_chain_base_scalar_fallback() const noexcept
    {
        matrix.record_native_chain_base_scalar_fallback();
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

    template <int SubBlockDim, int SubBlockCount, typename HMat>
    MUDA_DEVICE __forceinline__ bool try_add_native_dense_block_upper_subblocks_fixed(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
        if(runtime_ordering.enabled)
            return false;
        const auto target =
            matrix.template try_add_native_dense_block_upper_subblocks_fixed_target<
                SubBlockDim,
                SubBlockCount>(old_dof_begin, H);
        if(target == StructuredNativeDenseBlockTarget::SameBlock)
            matrix.record_native_chain_base_same_block_dense_hit();
        else if(target == StructuredNativeDenseBlockTarget::AdjacentBlock)
            matrix.record_native_chain_base_adjacent_dense_hit();
        else
            matrix.record_native_chain_base_dense_miss();
        return target != StructuredNativeDenseBlockTarget::Miss;
    }

    template <int Rows, int Cols, typename HMat>
    MUDA_DEVICE __forceinline__ bool try_add_native_same_block_dense_block_fixed(
        IndexT old_dof_begin,
        const HMat& H) const noexcept
    {
        if(runtime_ordering.enabled)
            return false;
        const bool used =
            matrix.template try_add_native_same_block_dense_block_fixed<Rows, Cols>(
                old_dof_begin,
                H);
        if(used)
            matrix.record_native_chain_base_diag3x3_hit();
        else
            matrix.record_native_chain_base_diag3x3_miss();
        return used;
    }

    template <typename HMat>
    MUDA_DEVICE __forceinline__ StructuredNativeDenseBlockTarget
    try_add_native_pair3x3_block(
        IndexT old_row_dof_begin,
        IndexT old_col_dof_begin,
        const HMat& H,
        bool mirror_when_same_native_block) const noexcept
    {
        if(runtime_ordering.enabled)
            return StructuredNativeDenseBlockTarget::Miss;
        const auto target = matrix.try_add_native_pair3x3_block_target(
            old_row_dof_begin,
            old_col_dof_begin,
            H,
            mirror_when_same_native_block);
        if(target == StructuredNativeDenseBlockTarget::SameBlock)
            matrix.record_native_chain_base_pair3x3_same_block_hit();
        else if(target == StructuredNativeDenseBlockTarget::AdjacentBlock)
            matrix.record_native_chain_base_pair3x3_adjacent_hit();
        else
            matrix.record_native_chain_base_pair3x3_miss();
        return target;
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

}  // namespace uipc::backend::cuda_mixed
