#include <uipc/core/rcc_adhesion_state_accessor_feature.h>
#include <uipc/common/log.h>

namespace uipc::core
{
RCCAdhesionStateAccessorFeature::RCCAdhesionStateAccessorFeature(
    S<RCCAdhesionStateAccessorFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
    UIPC_ASSERT(m_impl,
                "RCCAdhesionStateAccessorFeatureOverrider must not be null.");
}

SizeT RCCAdhesionStateAccessorFeature::pt_pair_count() const
{
    return m_impl->get_pt_pair_count();
}

void RCCAdhesionStateAccessorFeature::dump_pt_state(vector<U64>&   out_keys,
                                                     vector<Float>& out_betas) const
{
    m_impl->do_dump_pt_state(out_keys, out_betas);
}

void RCCAdhesionStateAccessorFeature::load_pt_state(span<const U64>   keys,
                                                     span<const Float> betas) const
{
    UIPC_ASSERT(keys.size() == betas.size(),
                "RCCAdhesionStateAccessorFeature::load_pt_state: "
                "keys.size() ({}) must equal betas.size() ({})",
                keys.size(),
                betas.size());
    m_impl->do_load_pt_state(keys, betas);
}

std::string_view RCCAdhesionStateAccessorFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core
