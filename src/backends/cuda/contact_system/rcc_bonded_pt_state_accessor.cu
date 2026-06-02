#include <contact_system/rcc_bonded_pt_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <sim_engine.h>
#include <uipc/core/rcc_bonded_pt_state_accessor_feature.h>

namespace uipc::backend::cuda
{
class RCCBondedPTStateAccessorOverriderImpl final
    : public core::RCCBondedPTStateAccessorFeatureOverrider
{
  public:
    RCCBondedPTStateAccessorOverriderImpl(RCCBondedPTSystem& owner,
                                          GlobalVertexManager& gvm)
        : m_owner{owner}
        , m_gvm{gvm}
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

    uipc::vector<Vector3> do_dump_locked_tet_world_positions() override
    {
        m_owner.sync_filter_skipped_count();
        auto topos     = m_owner.locked_topos();
        auto positions = m_gvm.positions();

        const SizeT m = topos.size();
        uipc::vector<Vector3> out;
        if(m == 0)
            return out;

        uipc::vector<Vector4i> h_topos(m);
        topos.copy_to(h_topos.data());

        uipc::vector<Vector3> h_pos(positions.size());
        if(!h_pos.empty())
            positions.copy_to(h_pos.data());

        const IndexT n = static_cast<IndexT>(h_pos.size());
        out.reserve(m * 4);
        for(SizeT i = 0; i < m; ++i)
        {
            const Vector4i tet = h_topos[i];
            for(int j = 0; j < 4; ++j)
            {
                const IndexT vid = tet[j];
                out.push_back((vid >= 0 && vid < n) ? h_pos[vid]
                                                    : Vector3::Zero());
            }
        }
        return out;
    }

  private:
    RCCBondedPTSystem&   m_owner;
    GlobalVertexManager& m_gvm;
};

class RCCBondedPTStateAccessor final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    virtual void do_build() override
    {
        auto& owner = require<RCCBondedPTSystem>();
        auto& gvm   = require<GlobalVertexManager>();
        auto overrider =
            std::make_shared<RCCBondedPTStateAccessorOverriderImpl>(owner, gvm);
        auto feature =
            std::make_shared<core::RCCBondedPTStateAccessorFeature>(overrider);
        features().insert(feature);
    }
};

REGISTER_SIM_SYSTEM(RCCBondedPTStateAccessor);
}  // namespace uipc::backend::cuda
