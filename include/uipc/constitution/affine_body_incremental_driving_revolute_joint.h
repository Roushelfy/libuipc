#pragma once
#include <uipc/constitution/constraint.h>
#include <uipc/geometry/simplicial_complex.h>

namespace uipc::constitution
{
/**
 * @brief AffineBodyIncrementalDrivingRevoluteJoint (constitution UID 33).
 *
 * Per-edge implicit-PD driving constraint on a base AffineBodyRevoluteJoint
 * (UID 18). Drives the joint using an INCREMENTAL angle delta-theta with a
 * per-edge scalar stiffness. The coupler overwrites `pd/strength`,
 * `pd/aim_increment` and `pd/is_constrained` each step.
 */
class UIPC_CONSTITUTION_API AffineBodyIncrementalDrivingRevoluteJoint final : public Constraint
{
    using Base = Constraint;

  public:
    AffineBodyIncrementalDrivingRevoluteJoint(const Json& config = default_config());

    ~AffineBodyIncrementalDrivingRevoluteJoint() override;

    /**
     * @brief Apply the implicit-PD revolute joint constraint to the edges of
     * a base revolute joint mesh.
     *
     * Requires the simplicial complex to already carry the base
     * AffineBodyRevoluteJoint constitution (UID 18).
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
