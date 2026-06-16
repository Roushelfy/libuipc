#pragma once
#include <uipc/common/dllexport.h>
#include <uipc/common/type_define.h>
#include <uipc/common/json.h>
#include <uipc/core/contact_element.h>
#include <uipc/core/contact_tabular.h>

namespace uipc::geometry
{
class SimplicialComplex;
}  // namespace uipc::geometry

namespace uipc::constitution
{
// RCC Adhesion frontend handle (augmented IPC for sticky interactions;
// see Fang et al. 2024). `RCCAdhesive` is a contact-tabular augmentor: it
// registers per-contact-pair adhesion parameters on
// `core::ContactTabular::contact_models()` so the CUDA backend evaluates an
// additive normal+tangential adhesion energy on top of IPC barrier + friction.
//
// Parameters per (L,R) contact-element pair:
//   Cn               : normal adhesion stiffness
//   Ct               : tangential adhesion stiffness
//   W                : maximum adhesion energy (normalized)
//   eta              : viscosity parameter
//   bonding_rate     : r
//   p0               : compression value for saturation
//   initial_beta     : initial adhesion intensity for newly created pairs
//   adhesion_enabled : 0 => no adhesion for this (L,R) pair
//
// UID = 1000, type = "ContactModel". If apply_to is never called, the backend
// reporters are no-ops (full backward compat).
class UIPC_CONSTITUTION_API RCCAdhesive
{
  public:
    explicit RCCAdhesive(const Json& config = default_config()) noexcept;

    /// Ensure the 8 adhesion attributes exist on `tabular.contact_models()`.
    /// Idempotent: if they already exist, no-op.
    void apply_to(core::ContactTabular& tabular) const;

    /// Set adhesion parameters for an existing (L,R) contact-element pair.
    /// Requires that the pair has been inserted via `tabular.insert(L, R, ...)` first.
    /// Calls `apply_to` internally to guarantee the attributes exist.
    void set(core::ContactTabular&      tabular,
             const core::ContactElement& L,
             const core::ContactElement& R,
             Float                       Cn,
             Float                       Ct,
             Float                       W,
             Float                       eta,
             Float                       bonding_rate,
             Float                       p0,
             Float                       initial_beta,
             bool                        enabled = true) const;

    /// Set adhesion parameters for the default (un-specified) (L,R) pair (index 0).
    /// Calls `apply_to` internally to guarantee the attributes exist.
    void default_model(core::ContactTabular& tabular,
                       Float                 Cn,
                       Float                 Ct,
                       Float                 W,
                       Float                 eta,
                       Float                 bonding_rate,
                       Float                 p0,
                       Float                 initial_beta,
                       bool                  enabled = true) const;

    /// Set PER-PAIR bonded-PT parameters for an existing (L,R) contact pair, so
    /// different surface pairs can lock / separate under different conditions.
    /// Any value < 0 means "inherit the global rcc_bonded_pt_* scene config".
    ///   lock_threshold : beta >= this -> the pair locks into a bonded ABD tet
    ///   release_strain/gap/slip/force : a locked bond on this pair releases
    ///       when the corresponding measure exceeds the threshold (1e30 = never)
    /// (bonded kappa is currently global — the ABD-tet energy stiffness.)
    /// Requires the pair to have been inserted via `tabular.insert(L,R,...)`.
    /// `distance_lock` selects this pair's adhesion MODE: < 0 inherits the
    /// global rcc_bonded_pt_distance_lock flag, 0 = soft RCC adhesion, > 0 =
    /// distance-lock (no soft energy; lock by the distance band). With a
    /// per-pair band ratio (< 0 = inherit the global ratio). A scene may thus
    /// mix soft-adhesion pairs and distance-lock pairs.
    void set_bonded(core::ContactTabular&       tabular,
                    const core::ContactElement& L,
                    const core::ContactElement& R,
                    Float                       lock_threshold,
                    Float                       release_strain,
                    Float                       release_gap,
                    Float                       release_slip,
                    Float                       release_force,
                    Float                       distance_lock       = -1.0,
                    Float                       distance_lock_ratio = -1.0) const;

    /// Per-pair bonded params for the default (un-specified) pair (index 0).
    void default_bonded(core::ContactTabular& tabular,
                        Float                 lock_threshold,
                        Float                 release_strain,
                        Float                 release_gap,
                        Float                 release_slip,
                        Float                 release_force,
                        Float                 distance_lock       = -1.0,
                        Float                 distance_lock_ratio = -1.0) const;

    /// Mark a shell geometry's sticky face for v3 single-sided adhesion.
    ///
    /// `sign = +1`  → the +n̂ face (the face that the triangle winding makes
    ///                outward) is the sticky face.
    /// `sign = -1`  → the -n̂ face is sticky.
    /// `sign = 0`   → double-sided (identical to v2 behaviour; this is the
    ///                default if `set_sticky_side` is never called).
    ///
    /// Writes/overwrites the per-vertex `rcc_sticky_sign` <IndexT> attribute
    /// on the geometry, broadcast-filling every vertex with the same sign.
    /// The CUDA backend reads this attribute once at scene init and gates
    /// adhesion contributions accordingly.
    static void set_sticky_side(geometry::SimplicialComplex& geo, IndexT sign);

    U64         get_uid() const noexcept;
    static Json default_config() noexcept;

  private:
    Json m_config;
};
}  // namespace uipc::constitution
