#include <contact_system/rcc_bonded_pt_system.h>
#include <sim_engine.h>
#include <uipc/core/rcc_bonded_pt_state_accessor_feature.h>

namespace uipc::backend::cuda
{
class RCCBondedPTStateAccessorOverriderImpl final
    : public core::RCCBondedPTStateAccessorFeatureOverrider
{
  public:
    explicit RCCBondedPTStateAccessorOverriderImpl(RCCBondedPTSystem& owner)
        : m_owner{owner}
    {
    }

    SizeT get_locked_pair_count() override
    {
        m_owner.sync_filter_skipped_count();
        return m_owner.size();
    }

    core::RCCBondedPTCounters get_counters() override
    {
        m_owner.sync_filter_skipped_count();
        return m_owner.counters();
    }

    core::RCCBondedPTState do_dump_state() override
    {
        m_owner.sync_filter_skipped_count();
        return m_owner.download();
    }

  private:
    RCCBondedPTSystem& m_owner;
};

class RCCBondedPTStateAccessor final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    virtual void do_build() override
    {
        auto& owner = require<RCCBondedPTSystem>();
        auto overrider =
            std::make_shared<RCCBondedPTStateAccessorOverriderImpl>(owner);
        auto feature =
            std::make_shared<core::RCCBondedPTStateAccessorFeature>(overrider);
        features().insert(feature);
    }
};

REGISTER_SIM_SYSTEM(RCCBondedPTStateAccessor);
}  // namespace uipc::backend::cuda
