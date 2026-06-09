#include <pyuipc/constitution/rcc_adhesive.h>
#include <uipc/constitution/rcc_adhesive.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/contact_element.h>
#include <uipc/geometry/simplicial_complex.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
using namespace uipc::core;

PyRCCAdhesive::PyRCCAdhesive(py::module& m)
{
    auto class_RCCAdhesive = py::class_<RCCAdhesive>(
        m,
        "RCCAdhesive",
        R"(RCC Adhesion contact-tabular augmentor.

Adds per-(L,R) adhesion attributes (Cn, Ct, W, eta, bonding_rate, p0, initial_beta,
adhesion_enabled) on top of an existing contact tabular. The CUDA backend
discovers these attributes and runs an additive adhesion energy on top of IPC
barrier + friction. See docs/specification/contact_models/rcc_adhesion.md.)");

    class_RCCAdhesive.def(py::init<const Json&>(),
                          py::arg("config") = RCCAdhesive::default_config());

    class_RCCAdhesive.def_static("default_config", &RCCAdhesive::default_config);

    class_RCCAdhesive.def(
        "apply_to",
        [](const RCCAdhesive& self, ContactTabular& tabular)
        { self.apply_to(tabular); },
        py::arg("tabular"),
        R"(Register the 8 RCC adhesion attributes on the contact tabular. Idempotent.)");

    class_RCCAdhesive.def(
        "set",
        [](const RCCAdhesive&    self,
           ContactTabular&       tabular,
           const ContactElement& L,
           const ContactElement& R,
           Float                 Cn,
           Float                 Ct,
           Float                 W,
           Float                 eta,
           Float                 bonding_rate,
           Float                 p0,
           Float                 initial_beta,
           bool                  enabled)
        {
            self.set(tabular, L, R, Cn, Ct, W, eta, bonding_rate, p0, initial_beta, enabled);
        },
        py::arg("tabular"),
        py::arg("L"),
        py::arg("R"),
        py::arg("Cn"),
        py::arg("Ct"),
        py::arg("W"),
        py::arg("eta"),
        py::arg("bonding_rate"),
        py::arg("p0"),
        py::arg("initial_beta"),
        py::arg("enabled") = true,
        R"(Set RCC adhesion parameters for an existing (L, R) contact pair.)");

    class_RCCAdhesive.def(
        "default_model",
        [](const RCCAdhesive& self,
           ContactTabular&    tabular,
           Float              Cn,
           Float              Ct,
           Float              W,
           Float              eta,
           Float              bonding_rate,
           Float              p0,
           Float              initial_beta,
           bool               enabled)
        {
            self.default_model(tabular, Cn, Ct, W, eta, bonding_rate, p0, initial_beta, enabled);
        },
        py::arg("tabular"),
        py::arg("Cn"),
        py::arg("Ct"),
        py::arg("W"),
        py::arg("eta"),
        py::arg("bonding_rate"),
        py::arg("p0"),
        py::arg("initial_beta"),
        py::arg("enabled") = true,
        R"(Set default RCC adhesion parameters for unspecified (L, R) pairs (row 0).)");

    class_RCCAdhesive.def(
        "set_bonded",
        [](const RCCAdhesive&    self,
           ContactTabular&       tabular,
           const ContactElement& L,
           const ContactElement& R,
           Float                 lock_threshold,
           Float                 release_strain,
           Float                 release_gap,
           Float                 release_slip,
           Float                 release_force)
        {
            self.set_bonded(tabular, L, R, lock_threshold, release_strain,
                            release_gap, release_slip, release_force);
        },
        py::arg("tabular"),
        py::arg("L"),
        py::arg("R"),
        py::arg("lock_threshold"),
        py::arg("release_strain") = 1e30,
        py::arg("release_gap")    = 1e30,
        py::arg("release_slip")   = 1e30,
        py::arg("release_force")  = 1e30,
        R"(Set PER-PAIR bonded-PT params for an existing (L, R) contact pair
(lock threshold + release strain/gap/slip/force). Any value < 0 inherits the
global rcc_bonded_pt_* scene config. bonded kappa is currently global.)");

    class_RCCAdhesive.def(
        "default_bonded",
        [](const RCCAdhesive& self,
           ContactTabular&    tabular,
           Float              lock_threshold,
           Float              release_strain,
           Float              release_gap,
           Float              release_slip,
           Float              release_force)
        {
            self.default_bonded(tabular, lock_threshold, release_strain,
                                release_gap, release_slip, release_force);
        },
        py::arg("tabular"),
        py::arg("lock_threshold"),
        py::arg("release_strain") = 1e30,
        py::arg("release_gap")    = 1e30,
        py::arg("release_slip")   = 1e30,
        py::arg("release_force")  = 1e30,
        R"(Set default per-pair bonded-PT params for unspecified (L, R) pairs (row 0).)");

    class_RCCAdhesive.def_static(
        "set_sticky_side",
        &RCCAdhesive::set_sticky_side,
        py::arg("geo"),
        py::arg("sign"),
        R"(Mark a shell geometry's sticky face for single-sided (oriented) adhesion.

sign = +1  → the +n̂ face (the face that the triangle winding makes outward)
              is the sticky face.
sign = -1  → the -n̂ face is sticky.
sign =  0  → double-sided (default behaviour when set_sticky_side is never
              called).

Writes (or overwrites) the per-vertex `rcc_sticky_sign` <IndexT> attribute
on the geometry, broadcast-filling every vertex with the same sign.)");

    class_RCCAdhesive.def("get_uid", &RCCAdhesive::get_uid);
}
}  // namespace pyuipc::constitution
