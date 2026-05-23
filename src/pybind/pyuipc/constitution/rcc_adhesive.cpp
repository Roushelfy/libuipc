#include <pyuipc/constitution/rcc_adhesive.h>
#include <uipc/constitution/rcc_adhesive.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/contact_element.h>

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

    class_RCCAdhesive.def("get_uid", &RCCAdhesive::get_uid);
}
}  // namespace pyuipc::constitution
