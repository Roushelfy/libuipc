#include <pyuipc/core/rcc_adhesive_diagnoser_feature.h>
#include <uipc/core/rcc_adhesive_diagnoser_feature.h>

namespace pyuipc::core
{
using namespace uipc::core;

PyRCCAdhesiveDiagnoserFeature::PyRCCAdhesiveDiagnoserFeature(py::module& m)
{
    auto class_RCCAdhesiveDiagnoserFeature =
        py::class_<RCCAdhesiveDiagnoserFeature, IFeature, S<RCCAdhesiveDiagnoserFeature>>(
            m, "RCCAdhesiveDiagnoserFeature");

    class_RCCAdhesiveDiagnoserFeature.def(
        "compute_pt_adhesion",
        [](RCCAdhesiveDiagnoserFeature&  self,
           geometry::Geometry&           R,
           geometry::SimplicialComplex&  points,
           geometry::SimplicialComplex&  triangles,
           geometry::SimplicialComplex&  prev_points,
           geometry::SimplicialComplex&  prev_triangles)
        { self.compute_pt_adhesion(R, points, triangles, prev_points, prev_triangles); },
        py::arg("R"),
        py::arg("points"),
        py::arg("triangles"),
        py::arg("prev_points"),
        py::arg("prev_triangles"));

    class_RCCAdhesiveDiagnoserFeature.attr("FeatureName") =
        RCCAdhesiveDiagnoserFeature::FeatureName;
}
}  // namespace pyuipc::core
