#include <pyuipc/constitution/affine_body_incremental_driving_revolute_joint.h>
#include <uipc/constitution/affine_body_incremental_driving_revolute_joint.h>
#include <uipc/constitution/constitution.h>
#include <pyuipc/common/json.h>
namespace pyuipc::constitution
{
using namespace uipc::constitution;

PyAffineBodyIncrementalDrivingRevoluteJoint::PyAffineBodyIncrementalDrivingRevoluteJoint(py::module& m)
{
    auto class_AffineBodyIncrementalDrivingRevoluteJoint =
        py::class_<AffineBodyIncrementalDrivingRevoluteJoint, IConstitution>(
            m,
            "AffineBodyIncrementalDrivingRevoluteJoint",
            R"(AffineBodyIncrementalDrivingRevoluteJoint per-edge implicit-PD driving constraint for revolute (hinge) joints between affine bodies.)");

    class_AffineBodyIncrementalDrivingRevoluteJoint.def(
        py::init<const Json&>(),
        py::arg("config") = AffineBodyIncrementalDrivingRevoluteJoint::default_config(),
        R"(Create an AffineBodyIncrementalDrivingRevoluteJoint.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_AffineBodyIncrementalDrivingRevoluteJoint.def_static("default_config",
                                               &AffineBodyIncrementalDrivingRevoluteJoint::default_config,
                                               R"(Get the default AffineBodyIncrementalDrivingRevoluteJoint configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_AffineBodyIncrementalDrivingRevoluteJoint.def(
        "apply_to",
        [](AffineBodyIncrementalDrivingRevoluteJoint& self, geometry::SimplicialComplex& edges)
        { self.apply_to(edges); },
        py::arg("sc"),
        py::doc(R"(Apply the implicit-PD revolute joint constraint to a base revolute joint mesh.
The mesh must already carry the base AffineBodyRevoluteJoint constitution (UID 18).
Each edge becomes a per-edge implicit-PD driving constraint; the pd/strength,
pd/aim_increment and pd/is_constrained attributes are created and updated externally each step.)"));
}
}  // namespace pyuipc::constitution
