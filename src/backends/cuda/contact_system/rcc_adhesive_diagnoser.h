#pragma once
#include <sim_system.h>
#include <uipc/core/rcc_adhesive_diagnoser_feature.h>

namespace uipc::backend::cuda
{
class RCCAdhesiveDiagnoser;

class RCCAdhesiveDiagnoserFeatureOverrider final
    : public core::RCCAdhesiveDiagnoserFeatureOverrider
{
  public:
    RCCAdhesiveDiagnoserFeatureOverrider(RCCAdhesiveDiagnoser* diagnoser);

  private:
    void do_compute_pt_adhesion(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points,
        const geometry::SimplicialComplex& triangles,
        const geometry::SimplicialComplex& prev_points,
        const geometry::SimplicialComplex& prev_triangles) override;

    SimSystemSlot<RCCAdhesiveDiagnoser> m_diagnoser;
};

class RCCAdhesiveDiagnoser final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    void compute_pt_adhesion(geometry::Geometry&                R,
                             const geometry::SimplicialComplex& points,
                             const geometry::SimplicialComplex& triangles,
                             const geometry::SimplicialComplex& prev_points,
                             const geometry::SimplicialComplex& prev_triangles);

  private:
    friend class RCCAdhesiveDiagnoserFeatureOverrider;
    void do_build() override;
};
}  // namespace uipc::backend::cuda
