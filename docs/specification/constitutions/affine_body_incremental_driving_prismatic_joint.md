# Affine Body Incremental Driving Prismatic Joint

References:

[A unified newton barrier method for multibody dynamics](https://dl.acm.org/doi/pdf/10.1145/3528223.3530076)

## #34 AffineBodyIncrementalDrivingPrismaticJoint

The **Affine Body Incremental Driving Prismatic Joint** is a constraint constitution that drives a [Prismatic Joint](./affine_body_prismatic_joint.md) (UID=20) by a target **distance increment** per time step. It must be applied to a geometry that already has an `AffineBodyPrismaticJoint` constitution.

It is the prismatic counterpart of the [Affine Body Incremental Driving Revolute Joint](./affine_body_incremental_driving_revolute_joint.md) (UID=33), penalizing the **incremental** signed axis displacement measured against the *previous-step* configuration — the same variational joint DOF used by the [External Articulation Constraint](./external_articulation_constraint.md) (UID=23) for prismatic joints. The prismatic coordinate is a linear axis projection with no $\operatorname{atan2}$, so unlike the revolute case it has no branch-cut problem; this constitution exists for symmetry with the revolute one and to provide the same single-term energy with direct joint-stiffness units (compare the [Affine Body Driving Prismatic Joint](./affine_body_driving_prismatic_joint.md) (UID=21), whose stiffness is mass-scaled and whose energy has two penalty terms).

## Energy

We assume 2 affine body indices $i$ and $j$, each with their own state vector $\mathbf{q}_i$ and $\mathbf{q}_j$ as defined in the [Affine Body](./affine_body.md) constitution.

The signed relative displacement along the joint axis is the symmetric projection maintained by the base [Prismatic Joint](./affine_body_prismatic_joint.md):

$$
d(\mathbf{q}) = \frac{(\mathbf{c}_j - \mathbf{c}_i)\cdot\hat{\mathbf{t}}_i - (\mathbf{c}_i - \mathbf{c}_j)\cdot\hat{\mathbf{t}}_j}{2},
$$

where $\mathbf{c}_k$ and $\hat{\mathbf{t}}_k$ are body $k$'s current anchor position and tangent (sliding) direction, obtained from the stored rest-space frame via the affine map.

The **incremental** displacement between the current state $\mathbf{q}$ and the previous-step state $\mathbf{q}^t$ is

$$
\delta d(\mathbf{q}, \mathbf{q}^t) = d(\mathbf{q}) - d(\mathbf{q}^t).
$$

The previous-step state $\mathbf{q}^t$ is the solver's own previous-step DOF; no `ref_dof_prev` attribute is required.

The energy function is a quadratic penalty on the increment error:

$$
E = \frac{s}{2} \left(\delta d - \tilde{\delta d}\right)^2,
$$

where:

- $s$ is the per-edge `pd/strength`, the joint stiffness **directly** (unit: force per length). There is no body-mass scaling $K = \gamma(m_i + m_j)$, and the energy is a single term in the scalar $\delta d$ — no factor-of-2 relative to the two-term penalty of the absolute driving joint.
- $\tilde{\delta d}$ is the per-edge `pd/aim_increment`, the target distance increment for the current time step.

The gradient and the Hessian are

$$
\mathbf{g} = s\left(\delta d - \tilde{\delta d}\right) \frac{\partial \delta d}{\partial \mathbf{q}}, \quad
\mathbf{H} = s\, \frac{\partial \delta d}{\partial \mathbf{q}} \frac{\partial \delta d}{\partial \mathbf{q}}^{\top} + s\left(\delta d - \tilde{\delta d}\right) \frac{\partial^2 \delta d}{\partial \mathbf{q}^2},
$$

where the first (Gauss-Newton) Hessian term is positive semi-definite for any $s \ge 0$ and the second-order term, which can be indefinite, is projected to be positive semi-definite before assembly — the same treatment used by the [External Articulation Constraint](./external_articulation_constraint.md) for the same $\partial^2\delta d/\partial\mathbf{q}^2$ kernel. At the converged state the joint carries the axial force $s(\tilde{\delta d} - \delta d)$, realizing the same implicit PD control law as the revolute counterpart (see its gain mapping for $k_p$, $k_v$).

When `pd/is_constrained = 0`, the energy, gradient, and Hessian are all zero and the driving effect is disabled.

## Requirement

This constitution must be applied to a geometry that already has an [AffineBodyPrismaticJoint](./affine_body_prismatic_joint.md) (UID=20) constitution; it reuses that joint's linking data (`l_geo_id` / `r_geo_id` / `l_inst_id` / `r_inst_id`) and per-body joint frame by edge index.

## Attributes

On the joint geometry (1D simplicial complex), on **edges** (one edge per joint). The edge inherits all linking and state fields of the base [Affine Body Prismatic Joint](./affine_body_prismatic_joint.md): `l_geo_id`, `r_geo_id`, `l_inst_id`, `r_inst_id`, `strength_ratio`, `distance`, `init_distance`, and optional `l_position0`, `l_position1`, `r_position0`, `r_position1` when created via Local `create_geometry`.

Driving-specific attributes on **edges**:

- `pd/strength` (`Float`): $s$, the per-edge joint stiffness ($= k_p + k_v/\Delta t$ for implicit PD). Default `0.0`.
- `pd/aim_increment` (`Float`): $\tilde{\delta d}$, the target distance increment for the current time step. Default `0.0`.
- `pd/is_constrained` (`IndexT`): enables (`1`) or disables (`0`) the driving effect. Default `0`.

`apply_to` resets all three attributes to their defaults; the animator (controller) is expected to overwrite them every time step.

## Note on Tracking Fidelity

The driver-side caveat in the revolute counterpart's [Note on Tracking Fidelity](./affine_body_incremental_driving_revolute_joint.md#note-on-tracking-fidelity) applies here as well: $\tilde{\delta d}$ must be the target increment measured in the same symmetric axis-projection convention this constitution uses; a driver that reconstructs the previous displacement differently (e.g. from the body transforms assuming rigidity) mis-references the target under affine non-rigidity and produces a tracking lag. Since the prismatic coordinate has no branch cut, a driver may also reference $\tilde{\delta d}$ to a fixed pose to close the lag exactly, without the wrap-around penalty the revolute joint would incur.
