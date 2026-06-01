#include <contact_system/rcc_bonded_pt_system.h>
#include <sim_engine.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(RCCBondedPTSystem);

void RCCBondedPTSystem::Impl::clear()
{
    m_bridge.clear();
    m_counters = {};
    clear_filter_keys();
}

void RCCBondedPTSystem::Impl::upload(const core::RCCBondedPTState& state)
{
    m_bridge.upload(state);
    m_counters = state.counters();
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
