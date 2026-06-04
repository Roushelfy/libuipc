#pragma once
#include <uipc/core/feature.h>
#include <uipc/core/rcc_bonded_pt_state.h>
#include <uipc/common/span.h>

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

    // Per-lock topology (Vector4i [p, t0, t1, t2]) and beta, for persisting the
    // bonded lock state to an asset. Both index-aligned, length locked_pair_count.
    // Default empty so non-backend overriders need not implement it.
    virtual void do_dump_locked_pairs(vector<Vector4i>& out_topos,
                                      vector<Float>&    out_betas)
    {
        out_topos.clear();
        out_betas.clear();
    }

    // Seed the bonded bridge with locks from saved topologies + betas, using the
    // current (loaded) geometry to (re)build each rest shape. Call after
    // world.init and before the first world.advance so the first step's
    // trajectory filter already sees the locked keys. Default no-op for mocks.
    virtual void do_seed_locks(span<const Vector4i> topos,
                               span<const Float>    betas,
                               Float                beta_lock_threshold)
    {
    }
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

    // Persist / restore the bonded lock state across an asset round-trip.
    // `dump_locked_pairs` pulls each lock's topology (Vector4i [p,t0,t1,t2]) and
    // beta; `seed_locks` re-locks those pairs against the current geometry
    // (rest shape recomputed from the loaded positions). Call seed_locks after
    // world.init and before the first world.advance.
    void dump_locked_pairs(vector<Vector4i>& out_topos,
                           vector<Float>&    out_betas) const;
    void seed_locks(span<const Vector4i> topos,
                    span<const Float>    betas,
                    Float                beta_lock_threshold) const;

  private:
    virtual std::string_view get_name() const override;
    S<RCCBondedPTStateAccessorFeatureOverrider> m_impl;
};
}  // namespace uipc::core
