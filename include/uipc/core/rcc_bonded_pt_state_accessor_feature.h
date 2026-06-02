#pragma once
#include <uipc/core/feature.h>
#include <uipc/core/rcc_bonded_pt_state.h>

namespace uipc::core
{
class UIPC_CORE_API RCCBondedPTStateAccessorFeatureOverrider
{
  public:
    RCCBondedPTStateAccessorFeatureOverrider() = default;
    virtual ~RCCBondedPTStateAccessorFeatureOverrider() = default;

    virtual SizeT get_locked_pair_count() = 0;
    virtual RCCBondedPTCounters get_counters() = 0;
    virtual RCCBondedPTState do_dump_state() = 0;

    // World-space positions of the four vertices of each locked virtual tet,
    // flattened as [p, t0, t1, t2] per lock (length 4 * locked_pair_count).
    // Default empty so non-backend overriders (e.g. test mocks) need not
    // implement it.
    virtual vector<Vector3> do_dump_locked_tet_world_positions() { return {}; }
};

class UIPC_CORE_API RCCBondedPTStateAccessorFeature final : public Feature
{
  public:
    constexpr static std::string_view FeatureName =
        "core/rcc_bonded_pt_state_accessor";

    explicit RCCBondedPTStateAccessorFeature(
        S<RCCBondedPTStateAccessorFeatureOverrider> overrider);

    SizeT locked_pair_count() const;
    RCCBondedPTCounters counters() const;
    RCCBondedPTState dump_state() const;

    // World-space positions of each locked virtual tet's four vertices,
    // flattened as [p, t0, t1, t2] per lock. For visualization.
    vector<Vector3> dump_locked_tet_world_positions() const;

  private:
    virtual std::string_view get_name() const override;
    S<RCCBondedPTStateAccessorFeatureOverrider> m_impl;
};
}  // namespace uipc::core
