#include <uipc/core/rcc_adhesive_diagnoser_feature.h>

namespace uipc::core
{
RCCAdhesiveDiagnoserFeature::RCCAdhesiveDiagnoserFeature(
    S<RCCAdhesiveDiagnoserFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
}

void RCCAdhesiveDiagnoserFeature::compute_pt_adhesion(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& triangles,
    const geometry::SimplicialComplex& prev_points,
    const geometry::SimplicialComplex& prev_triangles)
{
    m_impl->do_compute_pt_adhesion(R, points, triangles, prev_points, prev_triangles);
}

std::string_view RCCAdhesiveDiagnoserFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core
