#pragma once

#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/rcc_bonded_pt_state_bridge.h>
#include <sim_system.h>

namespace uipc::backend::cuda
{
class RCCBondedPTSystem final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        void clear();
        void upload(const core::RCCBondedPTState& state);
        core::RCCBondedPTState download() const;

        SizeT size() const noexcept;
        bool  empty() const noexcept;

        void set_enabled(bool enabled) noexcept;
        bool enabled() const noexcept;

        void bind_filter(SimplexTrajectoryFilter* filter) noexcept;
        void feed_filter_keys() const noexcept;
        void clear_filter_keys() const noexcept;
        void sync_filter_skipped_count() noexcept;

        void feed_filter_keys(SimplexTrajectoryFilter::Impl& filter) const noexcept;
        void sync_filter_skipped_count(const SimplexTrajectoryFilter::Impl& filter) noexcept;

        const core::RCCBondedPTCounters& counters() const noexcept;
        RCCBondedPTStateBridge&          bridge() noexcept;
        const RCCBondedPTStateBridge&    bridge() const noexcept;

        SimSystemSlot<GlobalTrajectoryFilter>  global_trajectory_filter;
        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;

      private:
        RCCBondedPTStateBridge     m_bridge;
        core::RCCBondedPTCounters  m_counters;
        bool                       m_enabled = false;
    };

    void clear();
    void upload(const core::RCCBondedPTState& state);
    core::RCCBondedPTState download() const;

    SizeT size() const noexcept;
    bool  empty() const noexcept;
    bool  enabled() const noexcept;
    const core::RCCBondedPTCounters& counters() const noexcept;

    void feed_filter_keys() const noexcept;
    void clear_filter_keys() const noexcept;
    void sync_filter_skipped_count() noexcept;

  protected:
    virtual void do_build() override;

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::cuda
