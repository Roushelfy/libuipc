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
};
}  // namespace uipc::backend::cuda
