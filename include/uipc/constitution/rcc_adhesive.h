#pragma once
#include <uipc/common/dllexport.h>
#include <uipc/common/type_define.h>
#include <uipc/common/json.h>
#include <uipc/core/contact_element.h>
#include <uipc/core/contact_tabular.h>

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

    U64         get_uid() const noexcept;
    static Json default_config() noexcept;

  private:
    Json m_config;
};
}  // namespace uipc::constitution
