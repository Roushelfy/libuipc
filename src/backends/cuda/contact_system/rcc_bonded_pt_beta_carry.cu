#include <contact_system/rcc_bonded_pt_beta_carry.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <muda/cub/device/device_select.h>
#include <muda/launch/parallel_for.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <uipc/common/log.h>

namespace uipc::backend::cuda
{
void RCCBondedPTBetaCarryScratch::merge_released_beta(
    muda::DeviceBuffer<U64>& prev_keys,
    muda::DeviceBuffer<Float>& prev_beta,
    muda::CBufferView<U64> released_keys,
    muda::CBufferView<Float> released_beta)
{
    UIPC_ASSERT(released_keys.size() == released_beta.size(),
                "RCC bonded PT beta carry received unzipped release buffers.");
    UIPC_ASSERT(prev_keys.size() == prev_beta.size(),
                "RCC bonded PT beta carry received unzipped previous buffers.");

    using namespace muda;

    const SizeT release_n = released_keys.size();
    if(release_n == 0)
        return;

    const SizeT prev_n = prev_keys.size();
    m_release_entries.resize(release_n);
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(release_n,
               [keys = released_keys.viewer().name("released_keys"),
                beta = released_beta.viewer().name("released_beta"),
                entries = m_release_entries.view().viewer().name("release_entries")] __device__(int i) mutable
               {
                   entries(i) = RCCBondedPTBetaCarryEntry{keys(i), beta(i)};
               });

    m_filtered_release_entries.resize(release_n);
    if(release_n > 0)
    {
        CBufferView<U64> prev_keys_view = prev_keys;
        DeviceSelect().If(
            m_release_entries.data(),
            m_filtered_release_entries.data(),
            m_filtered_count.data(),
            release_n,
            [prev_keys_view] CUB_RUNTIME_FUNCTION(
                const RCCBondedPTBetaCarryEntry& entry)
            {
                return !rcc_bonded_pt_is_locked(prev_keys_view, entry.key);
            });
    }
    else
    {
        m_filtered_count = 0;
    }

    IndexT filtered_count = m_filtered_count;
    if(filtered_count == 0)
        return;
    m_filtered_release_entries.resize(filtered_count);

    const SizeT total = prev_n + static_cast<SizeT>(filtered_count);
    m_merge_keys.resize(total);
    m_merge_beta.resize(total);
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(total,
               [prev_n,
                prev_keys = prev_keys.view().viewer().name("prev_keys"),
                prev_beta = prev_beta.view().viewer().name("prev_beta"),
                released =
                    m_filtered_release_entries.view().viewer().name("filtered_release"),
                merge_keys = m_merge_keys.view().viewer().name("merge_keys"),
                merge_beta = m_merge_beta.view().viewer().name("merge_beta")] __device__(int i) mutable
               {
                   if(static_cast<SizeT>(i) < prev_n)
                   {
                       merge_keys(i) = prev_keys(i);
                       merge_beta(i) = prev_beta(i);
                   }
                   else
                   {
                       const auto entry = released(i - static_cast<IndexT>(prev_n));
                       merge_keys(i) = entry.key;
                       merge_beta(i) = entry.beta;
                   }
               });

    thrust::sort_by_key(thrust::device,
                        m_merge_keys.data(),
                        m_merge_keys.data() + total,
                        m_merge_beta.data());
    auto unique_end = thrust::unique_by_key(thrust::device,
                                            m_merge_keys.data(),
                                            m_merge_keys.data() + total,
                                            m_merge_beta.data());
    const SizeT unique_count =
        static_cast<SizeT>(unique_end.first - m_merge_keys.data());
    m_merge_keys.resize(unique_count);
    m_merge_beta.resize(unique_count);

    prev_keys.resize(unique_count);
    prev_beta.resize(unique_count);
    if(unique_count > 0)
    {
        prev_keys.view().copy_from(m_merge_keys.view());
        prev_beta.view().copy_from(m_merge_beta.view());
    }
}
}  // namespace uipc::backend::cuda
