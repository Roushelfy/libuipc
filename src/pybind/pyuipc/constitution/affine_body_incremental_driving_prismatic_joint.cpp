#include <pyuipc/constitution/affine_body_incremental_driving_prismatic_joint.h>
#include <uipc/constitution/affine_body_incremental_driving_prismatic_joint.h>
#include <uipc/constitution/constitution.h>
#include <pyuipc/common/json.h>
namespace pyuipc::constitution
{
using namespace uipc::constitution;

PyAffineBodyIncrementalDrivingPrismaticJoint::PyAffineBodyIncrementalDrivingPrismaticJoint(py::module& m)
{
    auto class_AffineBodyIncrementalDrivingPrismaticJoint =
        py::class_<AffineBodyIncrementalDrivingPrismaticJoint, IConstitution>(
            m,
            "AffineBodyIncrementalDrivingPrismaticJoint",
            R"(AffineBodyIncrementalDrivingPrismaticJoint per-edge implicit-PD driving constraint for prismatic (sliding) joints between affine bodies.)");

    class_AffineBodyIncrementalDrivingPrismaticJoint.def(
        py::init<const Json&>(),
        py::arg("config") = AffineBodyIncrementalDrivingPrismaticJoint::default_config(),
        R"(Create an AffineBodyIncrementalDrivingPrismaticJoint.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_AffineBodyIncrementalDrivingPrismaticJoint.def_static("default_config",
                                                &AffineBodyIncrementalDrivingPrismaticJoint::default_config,
                                                R"(Get the default AffineBodyIncrementalDrivingPrismaticJoint configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_AffineBodyIncrementalDrivingPrismaticJoint.def(
        "apply_to",
        [](AffineBodyIncrementalDrivingPrismaticJoint& self, geometry::SimplicialComplex& edges)
        { self.apply_to(edges); },
        py::arg("sc"),
        py::doc(R"(Apply the implicit-PD prismatic joint constraint to a base prismatic joint mesh.
The mesh must already carry the base AffineBodyPrismaticJoint constitution (UID 20).
Each edge becomes a per-edge implicit-PD driving constraint; the pd/strength,
pd/aim_increment and pd/is_constrained attributes are created and updated externally each step.)"));
}
}  // namespace pyuipc::constitution
