#pragma once
#include <uipc/core/feature.h>
#include <uipc/common/span.h>
#include <uipc/common/vector.h>
#include <uipc/common/type_define.h>

namespace uipc::core
{
/**
 * @brief Backend-side hook for the RCCAdhesionStateAccessorFeature.
 *
 * The CUDA backend implements this and talks to the live
 * `IPCSimplexRCCAdhesiveContact` reporter.
 */
class UIPC_CORE_API RCCAdhesionStateAccessorFeatureOverrider
{
  public:
    RCCAdhesionStateAccessorFeatureOverrider()          = default;
    virtual ~RCCAdhesionStateAccessorFeatureOverrider() = default;

    // Reports the size of the reporter's `m_prev_keys_PT` buffer (= the
    // number of PT contact pairs whose β was snapshotted at the end of the
    // previous step).
    virtual SizeT get_pt_pair_count() const = 0;

    // Device → host. Overload populates `out_keys` and `out_betas` from
    // the reporter's `m_prev_keys_PT` / `m_prev_beta_PT` buffers. The
    // returned arrays are aligned by index and sorted by key.
    virtual void do_dump_pt_state(vector<U64>&   out_keys,
                                  vector<Float>& out_betas) const = 0;

    // Host → device. Uploads `keys` and `betas` into the reporter's
    // prev-state buffers (resize+copy+sort) and arms the
    // `m_has_loaded_prev_state` flag so the next step's Phase B takes
    // the match_or_init branch instead of init_all_new.
    virtual void do_load_pt_state(span<const U64>   keys,
                                  span<const Float> betas) = 0;
};

/**
 * @brief Save / restore RCC adhesion per-pair β across processes.
 *
 * The CUDA backend persists β across `world.advance()` steps via sorted
 * u64 hash keys (see `ipc_simplex_rcc_adhesive_contact.cu:39-79`), but
 * that persistence is in-memory only. This feature exposes the same
 * (keys, β) snapshot to the frontend so applications can save it
 * alongside an asset (e.g. into a .npz) and restore it in a follow-up
 * process — letting wound-state β survive a wind → save → reload cycle.
 *
 * Usage:
 *   auto acc = world.features().find<RCCAdhesionStateAccessorFeature>();
 *   // After some world.advance() steps:
 *   vector<U64>   keys; vector<Float> betas;
 *   acc->dump_pt_state(keys, betas);
 *   // → write (keys, betas) to disk.
 *
 *   // In a fresh process, after world.init(scene) and BEFORE the first
 *   // world.advance():
 *   acc->load_pt_state(keys_loaded, betas_loaded);
 */
class UIPC_CORE_API RCCAdhesionStateAccessorFeature final : public Feature
{
  public:
    constexpr static std::string_view FeatureName =
        "core/rcc_adhesion_state_accessor";

    explicit RCCAdhesionStateAccessorFeature(
        S<RCCAdhesionStateAccessorFeatureOverrider> overrider);

    /**
     * @brief Number of PT contact pairs whose β is currently held in the
     * reporter's prev-state snapshot.
     */
    SizeT pt_pair_count() const;

    /**
     * @brief Pull the prev-state (keys, β) snapshot from the device into
     * host vectors. The two arrays are index-aligned and sorted by key.
     */
    void dump_pt_state(vector<U64>&   out_keys,
                       vector<Float>& out_betas) const;

    /**
     * @brief Push (keys, β) into the reporter as the prev-state snapshot.
     * Must be called after `world.init(scene)` and before the first
     * `world.advance()`. Sizes of `keys` and `betas` must match.
     */
    void load_pt_state(span<const U64>   keys,
                       span<const Float> betas) const;

  private:
    virtual std::string_view                       get_name() const override;
    S<RCCAdhesionStateAccessorFeatureOverrider>    m_impl;
};
}  // namespace uipc::core
