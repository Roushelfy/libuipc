#include <contact_system/rcc_bonded_pt_system.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <muda/cub/device/device_select.h>
#include <muda/launch/parallel_for.h>
#include <sim_engine.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(RCCBondedPTSystem);

void RCCBondedPTSystem::Impl::clear()
{
    m_bridge.clear();
    m_counters = {};
    m_candidate_entries.resize(0);
    m_new_locked_entries.resize(0);
    m_new_locked_keys.resize(0);
    m_prev_entries.resize(0);
    m_carry_prev_entries.resize(0);
    m_merged_entries.resize(0);
    m_merged_keys.resize(0);
    clear_filter_keys();
}

void RCCBondedPTSystem::Impl::upload(const core::RCCBondedPTState& state)
{
    m_bridge.upload(state);
    m_counters = state.counters();
}

void RCCBondedPTSystem::Impl::lock_from_rcc_pt_snapshot(
    muda::CBufferView<Vector4i> pairs,
    muda::CBufferView<Float> beta,
    Float beta_lock_threshold)
{
    if(!m_enabled)
        return;

    UIPC_ASSERT(pairs.size() == 0 || pairs.size() == beta.size(),
                "RCC bonded PT producer received unzipped PT/beta buffers.");

    using namespace muda;

    const SizeT n = pairs.size();
    m_counters.candidate_count += n;

    const auto prev_keys  = m_bridge.locked_keys();
    const auto prev_topos = m_bridge.locked_topos();
    const auto prev_beta  = m_bridge.locked_beta();
    const auto prev_age   = m_bridge.locked_age();
    const auto prev_flags = m_bridge.release_flags();
    const SizeT prev_n    = m_bridge.size();

    m_candidate_entries.resize(n);
    if(n > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pairs = pairs.viewer().name("pairs"),
                    beta  = beta.viewer().name("beta"),
                    prev_keys_view = prev_keys,
                    prev_keys = prev_keys.viewer().name("prev_keys"),
                    prev_age = prev_age.viewer().name("prev_age"),
                    entries = m_candidate_entries.view().viewer().name("candidate_entries")] __device__(int i) mutable
                   {
                       const Vector4i topo = pairs(i);
                       const U64      key  = rcc_bonded_pt_key(topo);
                       IndexT         age  = 1;
                       const IndexT   prev = rcc_bonded_pt_lower_bound(prev_keys_view, key);
                       if(prev < static_cast<IndexT>(prev_keys_view.size())
                          && prev_keys(prev) == key)
                           age = prev_age(prev) + 1;

                       RCCBondedPTDeviceEntry entry;
                       entry.key           = key;
                       entry.topo          = topo;
                       entry.beta          = beta(i);
                       entry.age           = age;
                       entry.release_flags = core::RCCBondedPTReleaseNone;
                       entries(i)          = entry;
                   });
    }

    m_new_locked_entries.resize(n);
    if(n > 0)
    {
        DeviceSelect().If(
            m_candidate_entries.data(),
            m_new_locked_entries.data(),
            m_new_locked_count.data(),
            n,
            [beta_lock_threshold] CUB_RUNTIME_FUNCTION(
                const RCCBondedPTDeviceEntry& entry)
            {
                return entry.beta >= beta_lock_threshold
                       && entry.release_flags == core::RCCBondedPTReleaseNone;
            });
    }
    else
    {
        m_new_locked_count = 0;
    }

    IndexT new_locked_count = m_new_locked_count;
    m_new_locked_entries.resize(new_locked_count);
    m_new_locked_keys.resize(new_locked_count);

    if(new_locked_count > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(new_locked_count,
                   [entries = m_new_locked_entries.view().viewer().name("new_entries"),
                    keys = m_new_locked_keys.view().viewer().name("new_keys")] __device__(int i) mutable
                   {
                       keys(i) = entries(i).key;
                   });

        thrust::sort_by_key(thrust::device,
                            m_new_locked_keys.data(),
                            m_new_locked_keys.data() + new_locked_count,
                            m_new_locked_entries.data());
        auto unique_end = thrust::unique_by_key(thrust::device,
                                                m_new_locked_keys.data(),
                                                m_new_locked_keys.data()
                                                    + new_locked_count,
                                                m_new_locked_entries.data());
        const IndexT unique_new_locked_count =
            static_cast<IndexT>(unique_end.first - m_new_locked_keys.data());
        m_counters.duplicate_suppressed_count +=
            static_cast<SizeT>(new_locked_count - unique_new_locked_count);
        new_locked_count = unique_new_locked_count;
        m_new_locked_entries.resize(new_locked_count);
        m_new_locked_keys.resize(new_locked_count);
    }

    m_prev_entries.resize(prev_n);
    if(prev_n > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(prev_n,
                   [keys = prev_keys.viewer().name("prev_keys"),
                    topos = prev_topos.viewer().name("prev_topos"),
                    beta = prev_beta.viewer().name("prev_beta"),
                    age = prev_age.viewer().name("prev_age"),
                    flags = prev_flags.viewer().name("prev_flags"),
                    entries = m_prev_entries.view().viewer().name("prev_entries")] __device__(int i) mutable
                   {
                       RCCBondedPTDeviceEntry entry;
                       entry.key           = keys(i);
                       entry.topo          = topos(i);
                       entry.beta          = beta(i);
                       entry.age           = age(i) + 1;
                       entry.release_flags = flags(i);
                       entries(i)          = entry;
                   });
    }

    m_carry_prev_entries.resize(prev_n);
    if(prev_n > 0)
    {
        muda::CBufferView<U64> new_locked_keys = m_new_locked_keys;
        DeviceSelect().If(
            m_prev_entries.data(),
            m_carry_prev_entries.data(),
            m_carry_prev_count.data(),
            prev_n,
            [new_locked_keys] CUB_RUNTIME_FUNCTION(
                const RCCBondedPTDeviceEntry& entry)
            {
                return entry.release_flags == core::RCCBondedPTReleaseNone
                       && !rcc_bonded_pt_is_locked(new_locked_keys, entry.key);
            });
    }
    else
    {
        m_carry_prev_count = 0;
    }

    const IndexT carry_prev_count = m_carry_prev_count;
    m_carry_prev_entries.resize(carry_prev_count);

    const IndexT merged_count = carry_prev_count + new_locked_count;
    m_merged_entries.resize(merged_count);
    m_merged_keys.resize(merged_count);
    if(merged_count > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(merged_count,
                   [carry_count = carry_prev_count,
                    carry = m_carry_prev_entries.view().viewer().name("carry_prev"),
                    fresh = m_new_locked_entries.view().viewer().name("new_locked"),
                    merged = m_merged_entries.view().viewer().name("merged_entries"),
                    keys = m_merged_keys.view().viewer().name("merged_keys")] __device__(int i) mutable
                   {
                       const RCCBondedPTDeviceEntry entry =
                           i < carry_count ? carry(i) : fresh(i - carry_count);
                       merged(i) = entry;
                       keys(i)   = entry.key;
                   });

        thrust::sort_by_key(thrust::device,
                            m_merged_keys.data(),
                            m_merged_keys.data() + merged_count,
                            m_merged_entries.data());
    }

    m_counters.locked_count = static_cast<SizeT>(merged_count);
    m_bridge.replace_from_sorted_device_entries(m_merged_entries.view(), m_counters);
    feed_filter_keys();
}

core::RCCBondedPTState RCCBondedPTSystem::Impl::download() const
{
    auto state = m_bridge.download();
    state.set_counters(m_counters);
    return state;
}

SizeT RCCBondedPTSystem::Impl::size() const noexcept
{
    return m_bridge.size();
}

bool RCCBondedPTSystem::Impl::empty() const noexcept
{
    return m_bridge.empty();
}

void RCCBondedPTSystem::Impl::set_enabled(bool enabled) noexcept
{
    m_enabled = enabled;
}

bool RCCBondedPTSystem::Impl::enabled() const noexcept
{
    return m_enabled;
}

void RCCBondedPTSystem::Impl::bind_filter(SimplexTrajectoryFilter* filter) noexcept
{
    simplex_trajectory_filter = filter;
}

void RCCBondedPTSystem::Impl::feed_filter_keys() const noexcept
{
    if(!m_enabled || !simplex_trajectory_filter)
        return;
    simplex_trajectory_filter->set_rcc_bonded_pt_locked_keys(m_bridge.locked_keys());
}

void RCCBondedPTSystem::Impl::clear_filter_keys() const noexcept
{
    if(simplex_trajectory_filter)
        simplex_trajectory_filter->clear_rcc_bonded_pt_locked_keys();
}

void RCCBondedPTSystem::Impl::sync_filter_skipped_count() noexcept
{
    if(!m_enabled || !simplex_trajectory_filter)
        return;
    m_counters.filter_skipped_count +=
        simplex_trajectory_filter->rcc_bonded_pt_filter_skipped_count();
    m_counters.locked_count = size();
}

void RCCBondedPTSystem::Impl::feed_filter_keys(
    SimplexTrajectoryFilter::Impl& filter) const noexcept
{
    if(!m_enabled)
        return;
    filter.set_rcc_bonded_pt_locked_keys(m_bridge.locked_keys());
}

void RCCBondedPTSystem::Impl::sync_filter_skipped_count(
    const SimplexTrajectoryFilter::Impl& filter) noexcept
{
    if(!m_enabled)
        return;
    m_counters.filter_skipped_count += filter.rcc_bonded_pt_filter_skipped_count();
    m_counters.locked_count = size();
}

const core::RCCBondedPTCounters& RCCBondedPTSystem::Impl::counters() const noexcept
{
    return m_counters;
}

RCCBondedPTStateBridge& RCCBondedPTSystem::Impl::bridge() noexcept
{
    return m_bridge;
}

const RCCBondedPTStateBridge& RCCBondedPTSystem::Impl::bridge() const noexcept
{
    return m_bridge;
}

void RCCBondedPTSystem::do_build()
{
    auto& config = world().scene().config();
    auto  enabled_attr = config.find<IndexT>("rcc_bonded_pt_enabled");
    m_impl.set_enabled(enabled_attr && enabled_attr->view()[0] != 0);
    m_impl.global_trajectory_filter = find<GlobalTrajectoryFilter>();

    on_init_scene(
        [this]
        {
            if(!m_impl.enabled() || !m_impl.global_trajectory_filter)
                return;
            m_impl.bind_filter(
                m_impl.global_trajectory_filter->find<SimplexTrajectoryFilter>().view());
            m_impl.feed_filter_keys();
        });
    on_rebuild_scene(
        [this]
        {
            if(!m_impl.enabled())
                return;
            m_impl.feed_filter_keys();
        });
}

void RCCBondedPTSystem::clear()
{
    m_impl.clear();
}

void RCCBondedPTSystem::upload(const core::RCCBondedPTState& state)
{
    m_impl.upload(state);
    m_impl.feed_filter_keys();
}

void RCCBondedPTSystem::lock_from_rcc_pt_snapshot(
    muda::CBufferView<Vector4i> pairs,
    muda::CBufferView<Float> beta,
    Float beta_lock_threshold)
{
    m_impl.lock_from_rcc_pt_snapshot(pairs, beta, beta_lock_threshold);
}

core::RCCBondedPTState RCCBondedPTSystem::download() const
{
    return m_impl.download();
}

SizeT RCCBondedPTSystem::size() const noexcept
{
    return m_impl.size();
}

bool RCCBondedPTSystem::empty() const noexcept
{
    return m_impl.empty();
}

bool RCCBondedPTSystem::enabled() const noexcept
{
    return m_impl.enabled();
}

const core::RCCBondedPTCounters& RCCBondedPTSystem::counters() const noexcept
{
    return m_impl.counters();
}

void RCCBondedPTSystem::feed_filter_keys() const noexcept
{
    m_impl.feed_filter_keys();
}

void RCCBondedPTSystem::clear_filter_keys() const noexcept
{
    m_impl.clear_filter_keys();
}

void RCCBondedPTSystem::sync_filter_skipped_count() noexcept
{
    m_impl.sync_filter_skipped_count();
}
}  // namespace uipc::backend::cuda
