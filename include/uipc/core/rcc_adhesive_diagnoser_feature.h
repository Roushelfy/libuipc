#pragma once
#include <uipc/core/feature.h>
#include <uipc/geometry/geometry.h>
#include <uipc/geometry/simplicial_complex.h>

namespace uipc::core
{
class UIPC_CORE_API RCCAdhesiveDiagnoserFeatureOverrider
{
  public:
    virtual ~RCCAdhesiveDiagnoserFeatureOverrider() = default;

    // Compute PT normal and tangential adhesion energy, gradient, and Hessian
    // for every (point, triangle) pair.
    //
    // RCC parameters are read from instance attributes on R:
    //   "rcc/Cn"    Float  normal adhesion stiffness    (default 1.0)
    //   "rcc/Ct"    Float  tangential adhesion stiffness (default 1.0)
    //   "rcc/beta"  Float  bonding state ∈ [0,1]         (default 1.0)
    //   "rcc/d_hat" Float  activation distance            (default 0.1)
    //   "rcc/dt"    Float  time step                      (default 0.01)
    //
    // Results written to R:
    //   "normal/energy"  Float        per pair
    //   "normal/grad"    Vector12     per pair
    //   "normal/hess"    Matrix12x12  per pair
    //   "tan/energy"     Float        per pair
    //   "tan/grad"       Vector12     per pair
    //   "tan/hess"       Matrix12x12  per pair
    virtual void do_compute_pt_adhesion(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points,
        const geometry::SimplicialComplex& triangles,
        const geometry::SimplicialComplex& prev_points,
        const geometry::SimplicialComplex& prev_triangles) = 0;
};

class UIPC_CORE_API RCCAdhesiveDiagnoserFeature final : public Feature
{
  public:
    constexpr static std::string_view FeatureName = "core/rcc_adhesive_diagnoser";

    RCCAdhesiveDiagnoserFeature(S<RCCAdhesiveDiagnoserFeatureOverrider> overrider);

    void compute_pt_adhesion(geometry::Geometry&                R,
                             const geometry::SimplicialComplex& points,
                             const geometry::SimplicialComplex& triangles,
                             const geometry::SimplicialComplex& prev_points,
                             const geometry::SimplicialComplex& prev_triangles);

  private:
    virtual std::string_view                      get_name() const override;
    S<RCCAdhesiveDiagnoserFeatureOverrider> m_impl;
};
}  // namespace uipc::core
