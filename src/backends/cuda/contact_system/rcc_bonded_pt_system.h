#pragma once

#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/rcc_bonded_pt_state_bridge.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
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
        void lock_from_rcc_pt_snapshot(muda::CBufferView<Vector4i> pairs,
                                       muda::CBufferView<Float> beta,
                                       Float beta_lock_threshold);
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
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_candidate_entries;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_new_locked_entries;
        muda::DeviceBuffer<U64>                    m_new_locked_keys;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_prev_entries;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_carry_prev_entries;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_merged_entries;
        muda::DeviceBuffer<U64>                    m_merged_keys;
        muda::DeviceVar<IndexT>                    m_new_locked_count;
        muda::DeviceVar<IndexT>                    m_carry_prev_count;
        bool                       m_enabled = false;
    };

    void clear();
    void upload(const core::RCCBondedPTState& state);
    void lock_from_rcc_pt_snapshot(muda::CBufferView<Vector4i> pairs,
                                   muda::CBufferView<Float> beta,
                                   Float beta_lock_threshold);
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
