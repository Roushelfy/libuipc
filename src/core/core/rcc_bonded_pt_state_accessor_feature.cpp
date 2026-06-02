#include <uipc/core/rcc_bonded_pt_state_accessor_feature.h>
#include <uipc/common/log.h>

namespace uipc::core
{
RCCBondedPTStateAccessorFeature::RCCBondedPTStateAccessorFeature(
    S<RCCBondedPTStateAccessorFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
    UIPC_ASSERT(m_impl,
                "RCCBondedPTStateAccessorFeatureOverrider must not be null.");
}

SizeT RCCBondedPTStateAccessorFeature::locked_pair_count() const
{
    return m_impl->get_locked_pair_count();
}

RCCBondedPTCounters RCCBondedPTStateAccessorFeature::counters() const
{
    return m_impl->get_counters();
}

RCCBondedPTState RCCBondedPTStateAccessorFeature::dump_state() const
{
    return m_impl->do_dump_state();
}

std::string_view RCCBondedPTStateAccessorFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core
