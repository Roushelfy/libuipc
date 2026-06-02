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
                                       muda::CBufferView<Vector3> positions,
                                       Float beta_lock_threshold);
        core::RCCBondedPTState download() const;

        SizeT size() const noexcept;
        bool  empty() const noexcept;

        void set_enabled(bool enabled) noexcept;
        bool enabled() const noexcept;
        void set_rest_shape_config(Float min_separate_distance,
                                   Float det_dm_min) noexcept;

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
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_valid_new_locked_entries;
        muda::DeviceBuffer<U64>                    m_new_locked_keys;
        muda::DeviceBuffer<Matrix3x3>              m_new_locked_dm_inv;
        muda::DeviceBuffer<Float>                  m_new_locked_rest_volume;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_prev_entries;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_carry_prev_entries;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_merged_entries;
        muda::DeviceBuffer<U64>                    m_merged_keys;
        muda::DeviceVar<IndexT>                    m_new_locked_count;
        muda::DeviceVar<IndexT>                    m_valid_new_locked_count;
        muda::DeviceVar<IndexT>                    m_carry_prev_count;
        Float                      m_min_separate_distance = 1e-6;
        Float                      m_det_dm_min = 1e-12;
        SizeT                      m_last_synced_filter_generation = 0;
        bool                       m_enabled = false;
    };

    void clear();
    void upload(const core::RCCBondedPTState& state);
    void lock_from_rcc_pt_snapshot(muda::CBufferView<Vector4i> pairs,
                                   muda::CBufferView<Float> beta,
                                   muda::CBufferView<Vector3> positions,
                                   Float beta_lock_threshold);
    core::RCCBondedPTState download() const;

    SizeT size() const noexcept;
    bool  empty() const noexcept;
    bool  enabled() const noexcept;
    const core::RCCBondedPTCounters& counters() const noexcept;
    muda::CBufferView<Vector4i> locked_topos() const noexcept;
    muda::CBufferView<Matrix3x3> locked_dm_inv() const noexcept;
    muda::CBufferView<Float> locked_rest_volume() const noexcept;

    void feed_filter_keys() const noexcept;
    void clear_filter_keys() const noexcept;
    void sync_filter_skipped_count() noexcept;

  protected:
    virtual void do_build() override;

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::cuda
