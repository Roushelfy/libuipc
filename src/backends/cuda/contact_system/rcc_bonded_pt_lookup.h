#pragma once

#include <muda/buffer/buffer_view.h>
#include <muda/viewer/dense.h>
#include <type_define.h>

#include <contact_system/contact_models/codim_ipc_simplex_rcc_adhesive_function.h>

namespace uipc::backend::cuda
{
UIPC_GENERIC inline U64 rcc_bonded_pt_key(IndexT p,
                                          IndexT t0,
                                          IndexT t1,
                                          IndexT t2) noexcept
{
    return sym::codim_ipc_rcc_adhesive::PT_pair_key(p, t0, t1, t2);
}

UIPC_GENERIC inline U64 rcc_bonded_pt_key(const Vector4i& pt) noexcept
{
    return rcc_bonded_pt_key(pt[0], pt[1], pt[2], pt[3]);
}

UIPC_GENERIC inline IndexT
rcc_bonded_pt_lower_bound(muda::CBufferView<U64> sorted_keys, U64 key) noexcept
{
    IndexT lo = 0;
    IndexT hi = static_cast<IndexT>(sorted_keys.size());

    while(lo < hi)
    {
        const IndexT mid = (lo + hi) >> 1;
        if(sorted_keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

UIPC_GENERIC inline bool
rcc_bonded_pt_is_locked(muda::CBufferView<U64> sorted_keys, U64 key) noexcept
{
    const IndexT index = rcc_bonded_pt_lower_bound(sorted_keys, key);
    return index < static_cast<IndexT>(sorted_keys.size()) && sorted_keys[index] == key;
}

UIPC_GENERIC inline bool
rcc_bonded_pt_is_locked(muda::CBufferView<U64> sorted_keys, const Vector4i& pt) noexcept
{
    return rcc_bonded_pt_is_locked(sorted_keys, rcc_bonded_pt_key(pt));
}

// Decide whether a PT broadphase candidate, already resolved to its global
// point id `V` and triangle global ids `F`, belongs to a bonded (locked) pair.
// Used by every simplex filter backend's PT broadphase predicate so the
// membership semantics cannot drift between backends.
UIPC_GENERIC inline bool rcc_bonded_pt_candidate_is_locked(muda::CBufferView<U64> sorted_keys,
                                                           IndexT V,
                                                           const Vector3i& F) noexcept
{
    return rcc_bonded_pt_is_locked(sorted_keys, Vector4i{V, F(0), F(1), F(2)});
}
}  // namespace uipc::backend::cuda
