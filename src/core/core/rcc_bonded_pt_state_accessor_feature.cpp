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

vector<Vector3> RCCBondedPTStateAccessorFeature::dump_locked_tet_world_positions() const
{
    return m_impl->do_dump_locked_tet_world_positions();
}

void RCCBondedPTStateAccessorFeature::dump_locked_pairs(vector<Vector4i>& out_topos,
                                                        vector<Float>& out_betas) const
{
    m_impl->do_dump_locked_pairs(out_topos, out_betas);
}

void RCCBondedPTStateAccessorFeature::seed_locks(span<const Vector4i> topos,
                                                 span<const Float>    betas,
                                                 Float beta_lock_threshold) const
{
    UIPC_ASSERT(topos.size() == betas.size(),
                "RCCBondedPTStateAccessorFeature::seed_locks: topos.size() ({}) "
                "must equal betas.size() ({}).",
                topos.size(),
                betas.size());
    m_impl->do_seed_locks(topos, betas, beta_lock_threshold);
}

std::string_view RCCBondedPTStateAccessorFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core
