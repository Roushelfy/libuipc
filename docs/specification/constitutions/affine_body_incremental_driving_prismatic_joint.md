# Affine Body Incremental Driving Prismatic Joint

## #34 AffineBodyIncrementalDrivingPrismaticJoint

The prismatic counterpart of
[AffineBodyIncrementalDrivingRevoluteJoint](./affine_body_incremental_driving_revolute_joint.md)
(UID=33). It drives a base [Prismatic Joint](./affine_body_prismatic_joint.md) (UID=20) toward a
target **distance increment** `pd/aim_increment` using the **incremental** signed axis
displacement $\delta\theta(\mathbf{q},\mathbf{q}^t)$ measured against the previous-step state
(the same quantity used by the [External Articulation Constraint](./external_articulation_constraint.md)
prismatic path — a linear axis projection, no `atan2`, no branch cut).

## Energy

$$
E = \tfrac{1}{2}\, s \,\big(\delta\theta - \tilde{\delta\theta}\big)^2,
\qquad
\mathbf{g} = s\,(\delta\theta - \tilde{\delta\theta})\,\frac{\partial\delta\theta}{\partial\mathbf{q}},
\qquad
\mathbf{H} = s\,\frac{\partial\delta\theta}{\partial\mathbf{q}}\frac{\partial\delta\theta}{\partial\mathbf{q}}^{\!\top}
$$

with `pd/strength` $=s$ the joint stiffness directly and `pd/aim_increment` $=\tilde{\delta\theta}$.
Gauss-Newton Hessian only (PSD, no `make_spd`). $\delta\theta$ is a single scalar, so **no
factor-of-2** relative to the absolute prismatic driving joint. `pd/is_constrained = 0` disables it.

## Requirement

Must be applied to a geometry that already has an
[AffineBodyPrismaticJoint](./affine_body_prismatic_joint.md) (UID=20); it reuses that joint's
`body_ids` / `rest_cs` (c̄) / `rest_ts` (t̄) by index (co-located in `affine_body_prismatic_joint.cu`).
The per-body basis is reconstructed as `[c̄, t̄]` from the base joint's `rest_cs`/`rest_ts` storage.

## Attributes

On **edges**: `pd/strength`, `pd/aim_increment`, `pd/is_constrained` (see the revolute doc).

Prismatic is linear, so it never had a branch-cut problem; this constitution exists for symmetry
with the revolute one and to provide the same clean per-edge `½·strength·(δθ−aim_increment)²` energy
with direct joint-stiffness units.
