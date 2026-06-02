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

  private:
    virtual std::string_view get_name() const override;
    S<RCCBondedPTStateAccessorFeatureOverrider> m_impl;
};
}  // namespace uipc::core
