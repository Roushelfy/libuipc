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

    void do_dump_locked_pairs(uipc::vector<Vector4i>& out_topos,
                              uipc::vector<Float>&    out_betas) override
    {
        m_owner.sync_filter_skipped_count();
        auto        topos = m_owner.locked_topos();
        auto        betas = m_owner.locked_beta();
        const SizeT m     = topos.size();
        out_topos.resize(m);
        out_betas.resize(m);
        if(m > 0)
        {
            topos.copy_to(out_topos.data());
            betas.copy_to(out_betas.data());
        }
    }

    void do_seed_locks(span<const Vector4i> topos,
                       span<const Float>    betas,
                       Float                beta_lock_threshold) override
    {
        const SizeT n = topos.size();
        if(n == 0)
            return;
        // Upload the saved (topo, beta) to device and re-lock against the
        // current (loaded) geometry. lock_from_rcc_pt_snapshot rebuilds each
        // rest shape from m_gvm.positions(), derives keys from the topos, and
        // feeds the locked keys to the trajectory filter — so the first step's
        // filter already compacts these out of friction_VTs (no re-form
        // transient). The no-release-context overload is used: a freshly seeded
        // (empty) bridge has no carried locks for the release policy to act on.
        muda::DeviceBuffer<Vector4i> d_topos(n);
        muda::DeviceBuffer<Float>    d_betas(n);
        d_topos.view().copy_from(topos.data());
        d_betas.view().copy_from(betas.data());
        m_owner.lock_from_rcc_pt_snapshot(d_topos.view(),
                                          d_betas.view(),
                                          m_gvm.positions(),
                                          beta_lock_threshold);
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
