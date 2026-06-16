#pragma once
#include <type_define.h>

namespace uipc::backend::cuda
{
// Per-(contact_element_i, contact_element_j) RCC adhesion parameters.
// Filled on init/rebuild by IPCSimplexRCCAdhesiveContact from
// world().scene().contact_tabular().contact_models() — see XBow's
// RCCAdhesionEnergy3D for the canonical math.
//
// All scalar fields use XBow-style adaptive scaling at energy time:
//   normal energy ~ Cn/(2 dHat) * beta^2 * D
//   tangent energy ~ Ct/(2 dHat) * beta^2 * |u|^2
//
// `enabled = 0` ⇒ this pair contributes no adhesion (reporter early-out).
struct RCCAdhesiveCoeff
{
    Float  Cn           = 0.0;
    Float  Ct           = 0.0;
    Float  W            = 0.0;
    Float  eta          = 1.0;
    Float  bonding_rate = 0.0;
    Float  p0           = 0.0;
    Float  initial_beta = 0.0;
    IndexT enabled      = 0;

    // Per-pair BONDED-PT parameters. The lock filter and the release-flag
    // kernel look these up per contact-element pair (via contact_element_ids)
    // so different surface pairs can lock/separate under different conditions.
    // Filled by _rebuild_adhesive_tabular: a per-pair value from the
    // contact-model attribute, or the GLOBAL rcc_bonded_pt_* scene-config value
    // when the per-pair attribute is unset (sentinel < 0). bonded_kappa is NOT
    // per-pair yet (the ABD-tet energy uses the global stiffness).
    Float bonded_lock_threshold = 1.0;     // beta >= this -> lock (this pair)
    Float bonded_release_strain = 1e30;    // release if ABD strain exceeds
    Float bonded_release_gap    = 1e30;    // release if normal gap exceeds
    Float bonded_release_slip   = 1e30;    // release if tangential slip exceeds
    Float bonded_release_force  = 1e30;    // release if restoring force exceeds

    // Per-pair adhesion MODE (Phase 8). Resolved at rebuild against the global
    // rcc_bonded_pt_distance_lock flag (per-pair attr sentinel < 0 = inherit):
    //   0 = soft RCC adhesion (beta law + Cn/Ct energy, optional bond when
    //       beta >= bonded_lock_threshold);
    //   1 = distance-lock (no soft energy; lock purely by the end-of-step
    //       distance band d < (xi + distance_lock_ratio*d_hat)^2).
    // Lets one scene mix soft-adhesion pairs and distance-lock pairs.
    IndexT distance_lock       = 0;
    Float  distance_lock_ratio = 0.5;      // band coefficient c for mode 1
};
}  // namespace uipc::backend::cuda
