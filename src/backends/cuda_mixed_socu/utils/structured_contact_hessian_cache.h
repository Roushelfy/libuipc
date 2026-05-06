#pragma once

#include <uipc/common/type_define.h>
#include <muda/buffer/buffer_view.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT>
struct StructuredContactHessianRecord
{
    IndexT global_i = -1;
    IndexT global_j = -1;
    IndexT mirror_diag_block = 0;
    StoreT H[9] = {};
};

template <typename StoreT>
struct StructuredContactHessianCache
{
    muda::BufferView<StructuredContactHessianRecord<StoreT>> records;
    muda::BufferView<IndexT> cursor;
    SizeT replay_count = 0;
    bool  collect = false;
    bool  replay = false;

    MUDA_GENERIC bool collect_valid() const noexcept
    {
        return collect && records.data() != nullptr && cursor.data() != nullptr
               && cursor.size() >= 2;
    }

    MUDA_GENERIC bool replay_valid() const noexcept
    {
        return replay && records.data() != nullptr && replay_count != 0;
    }
};
}  // namespace uipc::backend::cuda_mixed
