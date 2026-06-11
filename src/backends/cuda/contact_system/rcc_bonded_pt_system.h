#pragma once

#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/rcc_adhesive_coeff.h>
#include <contact_system/rcc_bonded_pt_state_bridge.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <sim_system.h>

namespace uipc::backend::cuda
{
struct RCCBondedPTReleaseContext
{
    muda::CBufferView<IndexT>  sticky_sign;
    muda::CBufferView<Vector3> vertex_normal;
    bool                       sticky_side_enabled = false;

    muda::CBufferView<IndexT> contact_element_ids;
    muda::CBufferView<IndexT> subscene_element_ids;
    muda::CBuffer2DView<IndexT> contact_mask_tabular;
    muda::CBuffer2DView<IndexT> subscene_mask_tabular;
    muda::CBuffer2DView<RCCAdhesiveCoeff> adhesive_tabular;
    bool policy_enabled = false;
};

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
        void lock_from_rcc_pt_snapshot(muda::CBufferView<Vector4i> pairs,
                                       muda::CBufferView<Float> beta,
                                       muda::CBufferView<Vector3> positions,
                                       Float beta_lock_threshold,
                                       const RCCBondedPTReleaseContext& release_context);
        core::RCCBondedPTState download() const;

        SizeT size() const noexcept;
        bool  empty() const noexcept;

        void set_enabled(bool enabled) noexcept;
        bool enabled() const noexcept;
        void set_skip_ccd(bool enabled) noexcept;
        void set_rest_shape_config(Float min_separate_distance,
                                   Float det_dm_min) noexcept;
        void set_release_config(Float strain_threshold,
                                Float gap_threshold,
                                Float slip_threshold) noexcept;
        // Force/energy release: fires when the bond's restoring force exceeds
        // the threshold. Needs kappa and dt to scale the F-space gradient.
        void set_release_force_config(Float force_threshold,
                                      Float kappa,
                                      Float dt) noexcept;

        void bind_filter(SimplexTrajectoryFilter* filter) noexcept;
        void feed_filter_keys() const noexcept;
        void clear_filter_keys() const noexcept;
        void sync_filter_skipped_count() noexcept;

        void feed_filter_keys(SimplexTrajectoryFilter::Impl& filter) const noexcept;
        void sync_filter_skipped_count(const SimplexTrajectoryFilter::Impl& filter) noexcept;

        const core::RCCBondedPTCounters& counters() const noexcept;
        // Distance-lock mode (Phase 7): the lock producer reports its
        // eligibility-kernel rejections here (the producer owns the gate;
        // this system owns the counters).
        void add_lock_rejection_counts(SizeT distance_rejected,
                                       SizeT policy_rejected) noexcept;
        muda::CBufferView<U64>      released_keys() const noexcept;
        muda::CBufferView<Vector4i> released_topos() const noexcept;
        muda::CBufferView<Float>    released_beta() const noexcept;
        muda::CBufferView<IndexT>   released_age() const noexcept;
        muda::CBufferView<U32>      released_flags() const noexcept;
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
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_released_entries;
        muda::DeviceBuffer<U64>                    m_released_keys;
        muda::DeviceBuffer<Vector4i>               m_released_topos;
        muda::DeviceBuffer<Float>                  m_released_beta;
        muda::DeviceBuffer<IndexT>                 m_released_age;
        muda::DeviceBuffer<U32>                    m_released_flags;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_carry_prev_entries;
        muda::DeviceBuffer<RCCBondedPTDeviceEntry> m_merged_entries;
        muda::DeviceBuffer<U64>                    m_merged_keys;
        muda::DeviceVar<IndexT>                    m_new_locked_count;
        muda::DeviceVar<IndexT>                    m_valid_new_locked_count;
        muda::DeviceVar<IndexT>                    m_released_count;
        muda::DeviceVar<IndexT>                    m_carry_prev_count;
        Float                      m_min_separate_distance = 1e-6;
        Float                      m_det_dm_min = 1e-12;
        Float                      m_release_strain_threshold = 1e30;
        Float                      m_release_gap_threshold = 1e30;
        Float                      m_release_slip_threshold = 1e30;
        Float                      m_release_force_threshold = 1e30;
        Float                      m_kappa = 1e8;
        Float                      m_dt = 0.01;
        SizeT                      m_last_synced_filter_generation = 0;
        bool                       m_enabled = false;
        bool                       m_skip_ccd = false;
    };

    void clear();
    void upload(const core::RCCBondedPTState& state);
    void lock_from_rcc_pt_snapshot(muda::CBufferView<Vector4i> pairs,
                                   muda::CBufferView<Float> beta,
                                   muda::CBufferView<Vector3> positions,
                                   Float beta_lock_threshold);
    void lock_from_rcc_pt_snapshot(muda::CBufferView<Vector4i> pairs,
                                   muda::CBufferView<Float> beta,
                                   muda::CBufferView<Vector3> positions,
                                   Float beta_lock_threshold,
                                   const RCCBondedPTReleaseContext& release_context);
    core::RCCBondedPTState download() const;

    SizeT size() const noexcept;
    bool  empty() const noexcept;
    bool  enabled() const noexcept;
    const core::RCCBondedPTCounters& counters() const noexcept;
    void add_lock_rejection_counts(SizeT distance_rejected,
                                   SizeT policy_rejected) noexcept;
    muda::CBufferView<Vector4i> locked_topos() const noexcept;
    muda::CBufferView<Matrix3x3> locked_dm_inv() const noexcept;
    muda::CBufferView<Float> locked_rest_volume() const noexcept;
    muda::CBufferView<U64> locked_keys() const noexcept;
    muda::CBufferView<Float> locked_beta() const noexcept;
    muda::CBufferView<U64> released_keys() const noexcept;
    muda::CBufferView<Float> released_beta() const noexcept;

    void feed_filter_keys() const noexcept;
    void clear_filter_keys() const noexcept;
    void sync_filter_skipped_count() noexcept;

  protected:
    virtual void do_build() override;

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::cuda
