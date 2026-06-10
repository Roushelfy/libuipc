# Affine Body Incremental Driving Revolute Joint

References:

[A unified newton barrier method for multibody dynamics](https://dl.acm.org/doi/pdf/10.1145/3528223.3530076)

## #33 AffineBodyIncrementalDrivingRevoluteJoint

The **Affine Body Incremental Driving Revolute Joint** is a constraint constitution that
drives a [Revolute Joint](./affine_body_revolute_joint.md) (UID=18) toward a target
**angle increment** `aim_increment` (δθ̃) per step. It must be applied to a geometry that
already has an `AffineBodyRevoluteJoint` constitution.

It is the **incremental-angle** counterpart of the
[Affine Body Driving Revolute Joint](./affine_body_driving_revolute_joint.md) (UID=19).
Where the driving joint penalizes the **absolute** angle `θ` against the rest frame — and so
hits the `atan2` ±π branch cut when the joint operates far from rest under fast motion (the
energy becomes discontinuous and the Newton/line-search fails) — this constitution penalizes
the **incremental** angle measured against the *previous-step* configuration, which stays near
0 every step and therefore never crosses the branch cut. It is the same incremental angle used
by the [External Articulation Constraint](./external_articulation_constraint.md) (UID=23/24).

## Energy

We assume 2 affine body indices $i$ and $j$ with state vectors $\mathbf{q}_i$, $\mathbf{q}_j$.
Let $\delta\theta(\mathbf{q}, \mathbf{q}^t)$ be the **incremental** relative angle between the
current state $\mathbf{q}$ and the previous-step state $\mathbf{q}^t$ about the joint axis,
computed exactly as in the External Articulation Constraint:

$$
\delta\theta = \operatorname{atan2}\!\big(\sin\theta\cos\theta^t - \cos\theta\sin\theta^t,\;
\cos\theta\cos\theta^t + \sin\theta\sin\theta^t\big),
$$

where $\sin\theta,\cos\theta$ are the symmetric half-sums of the per-body $(\hat{\mathbf n},\hat{\mathbf b})$
basis dot products (see the base [Revolute Joint](./affine_body_revolute_joint.md)) and
$\theta^t$ is evaluated from $\mathbf{q}^t$. The previous-step state $\mathbf{q}^t$ is the
solver's own `q_prev` (no `ref_dof_prev` attribute is required).

The energy is a quadratic penalty on the increment error:

$$
E = \tfrac{1}{2}\, s \,\big(\delta\theta - \tilde{\delta\theta}\big)^2,
$$

where $s$ is the per-edge `pd/strength` and $\tilde{\delta\theta}$ is the per-edge
`pd/aim_increment`. **`pd/strength` is the joint stiffness directly** (the energy is a single
term in the scalar $\delta\theta$ — there is no body-mass scaling and, unlike the driving
joint, no factor-of-2 for the prismatic counterpart).

**Gradient** (generalized force IPC applies):

$$
\mathbf{g} = s\,(\delta\theta - \tilde{\delta\theta})\;\frac{\partial \delta\theta}{\partial \mathbf{q}},
$$

so at the converged state the joint torque is $s\,(\tilde{\delta\theta} - \delta\theta)$ — a PD
law when the controller sets $s = k_p + k_v/\mathrm{dt}$ and
$\tilde{\delta\theta} = (k_p q_{\text{des}} + (k_v/\mathrm{dt})(q_n + v_{\text{des}}\mathrm{dt}))/s - q_n$.

**Hessian — Gauss-Newton only:**

$$
\mathbf{H} = s\;\frac{\partial \delta\theta}{\partial \mathbf{q}}\frac{\partial \delta\theta}{\partial \mathbf{q}}^{\!\top}.
$$

The second-order term $s(\delta\theta-\tilde{\delta\theta})\,\partial^2\delta\theta/\partial\mathbf{q}^2$
is **dropped**, and no `make_spd` is needed: the outer product is PSD for any $s \ge 0$. This
keeps the Newton system unconditionally PSD without the indefinite-curvature projection the
driving joint requires.

When `pd/is_constrained = 0`, the energy, gradient, and Hessian are all zero.

## Requirement

Must be applied to a geometry that already has an
[AffineBodyRevoluteJoint](./affine_body_revolute_joint.md) (UID=18) constitution; it reuses that
joint's `body_ids` / `l_basis` / `r_basis` by index (the CUDA constraint is co-located in
`affine_body_revolute_joint.cu`, same as the driving joint).

## Attributes

Driving-specific attributes on **edges**:

- `pd/strength`: $s$, the per-edge joint stiffness (= $k_p + k_v/\mathrm{dt}$ for implicit PD).
- `pd/aim_increment`: $\tilde{\delta\theta}$, the target angle increment for this step.
- `pd/is_constrained`: enables (`1`) or disables (`0`) the driving effect.

## Note on tracking fidelity (driver responsibility)

The constitution's angle is faithful: `δθ(x)` and `∂δθ/∂x` are **bit-identical** to the
absolute-angle driving joint's `θ`/`∂θ/∂x` (verified by direct evaluation of the symbolic
kernels, rigid and non-rigid, at any angle). So the angle parametrization introduces **no**
fidelity loss by itself.

A tracking lag can nonetheless appear, and it is a property of **how the driver computes the
target** `aim_increment` (δθ̃), not of this constitution. δθ̃ must be the increment to the target
measured in the *same* angle convention this constitution uses (the symmetric basis-dot `atan2`
above). If the driver instead forms δθ̃ from a *different* angle reconstruction of the previous
state — e.g. a quaternion-extracted joint angle from the body transforms — the two reconstructions
agree only while the bodies are rigid; under the affine scale/shear that develops in a stiff solve
they diverge, the target is mis-referenced, and the joint tracks with a gain/motion-dependent lag
(empirically up to ~3× the 1-DOF RMS at high stiffness).

This is a genuine tension, not a bug. Making δθ̃ self-consistent by referencing it to a **fixed**
pose (build/rest) closes the lag exactly **but reintroduces the ±π `atan2` branch cut** of the
absolute angle once a joint's excursion from that pose exceeds π — defeating the whole purpose of
the incremental form. A recent reference in the *same* convention cancels back to the absolute
angle (same branch cut). So for a **non-stateful** driver the choices are: branch-cut-robust with a
modest lag (use a recent, possibly-different reconstruction for δθ̃), or faithful but branch-cut-
fragile (fixed reference). Closing the lag *and* staying robust requires **stateful continuous-angle
unwrapping** (tracking the winding number so the absolute basis-dot angle never wraps). Equilibrium
and holding are unaffected in all cases.

The prismatic counterpart is
[AffineBodyIncrementalDrivingPrismaticJoint](./affine_body_incremental_driving_prismatic_joint.md)
(UID=34).
