#pragma once
#include <uipc/constitution/constraint.h>
#include <uipc/geometry/simplicial_complex.h>

namespace uipc::constitution
{
/**
 * @brief AffineBodyIncrementalDrivingPrismaticJoint (constitution UID 34).
 *
 * Per-edge implicit-PD driving constraint on a base AffineBodyPrismaticJoint
 * (UID 20). Drives the joint using an INCREMENTAL coordinate delta-theta with a
 * per-edge scalar stiffness. The coupler overwrites `pd/strength`,
 * `pd/aim_increment` and `pd/is_constrained` each step.
 */
class UIPC_CONSTITUTION_API AffineBodyIncrementalDrivingPrismaticJoint final : public Constraint
{
    using Base = Constraint;

  public:
    AffineBodyIncrementalDrivingPrismaticJoint(const Json& config = default_config());

    ~AffineBodyIncrementalDrivingPrismaticJoint() override;

    /**
     * @brief Apply the implicit-PD prismatic joint constraint to the edges of
     * a base prismatic joint mesh.
     *
     * Requires the simplicial complex to already carry the base
     * AffineBodyPrismaticJoint constitution (UID 20).
     *
     * @param sc The simplicial complex (linemesh) whose edges are the joints.
     */
    void apply_to(geometry::SimplicialComplex& sc);

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept override;
    Json        m_config;
};
}  // namespace uipc::constitution
