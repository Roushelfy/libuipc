# Affine Body Incremental Driving Revolute Joint

References:

[A unified newton barrier method for multibody dynamics](https://dl.acm.org/doi/pdf/10.1145/3528223.3530076)

## #33 AffineBodyIncrementalDrivingRevoluteJoint

The **Affine Body Incremental Driving Revolute Joint** is a constraint constitution that drives a [Revolute Joint](./affine_body_revolute_joint.md) (UID=18) by a target **angle increment** per time step. It must be applied to a geometry that already has an `AffineBodyRevoluteJoint` constitution.

It is the incremental-angle counterpart of the [Affine Body Driving Revolute Joint](./affine_body_driving_revolute_joint.md) (UID=19). The driving joint penalizes the **absolute** angle $\theta$ against the rest frame, so when the joint operates far from rest its energy hits the $\pm\pi$ branch cut of $\operatorname{atan2}$ and becomes discontinuous. This constitution instead penalizes the **incremental** angle $\delta\theta$ measured against the *previous-step* configuration; since $\delta\theta$ stays near $0$ every step, the energy never crosses the branch cut. The incremental angle is the same quantity used by the [External Articulation Constraint](./external_articulation_constraint.md) (UID=23).

The constitution implements an implicit PD (proportional-derivative) control law: an animator writes the per-edge stiffness and target increment each step, and the solver realizes the corresponding joint torque at the converged state (see the gradient interpretation below).

The prismatic counterpart is the [Affine Body Incremental Driving Prismatic Joint](./affine_body_incremental_driving_prismatic_joint.md) (UID=34).

## Energy

We assume 2 affine body indices $i$ and $j$, each with their own state vector $\mathbf{q}_i$ and $\mathbf{q}_j$ as defined in the [Affine Body](./affine_body.md) constitution.

The relative rotation angle between the two bodies about the joint axis is extracted from the per-body $(\hat{\mathbf{n}}_k, \hat{\mathbf{b}}_k)$ basis maintained by the base [Revolute Joint](./affine_body_revolute_joint.md):

$$
\cos\theta = \frac{\hat{\mathbf{n}}_i \cdot \hat{\mathbf{n}}_j + \hat{\mathbf{b}}_i \cdot \hat{\mathbf{b}}_j}{2}, \quad
\sin\theta = \frac{\hat{\mathbf{b}}_i \cdot \hat{\mathbf{n}}_j - \hat{\mathbf{n}}_i \cdot \hat{\mathbf{b}}_j}{2},
$$

where $\hat{\mathbf{n}}_k$, $\hat{\mathbf{b}}_k$ are the normal and binormal directions in body $k$'s current frame, obtained from the stored rest-space basis via the affine map.

The **incremental** angle between the current state $\mathbf{q}$ and the previous-step state $\mathbf{q}^t$ is then

$$
\delta\theta(\mathbf{q}, \mathbf{q}^t) = \operatorname{atan2}\left(\sin\theta\cos\theta^t - \cos\theta\sin\theta^t,\; \cos\theta\cos\theta^t + \sin\theta\sin\theta^t\right),
$$

where $\sin\theta^t$, $\cos\theta^t$ are evaluated from $\mathbf{q}^t$ by the same formulas. This is exactly the variational joint DOF of the [External Articulation Constraint](./external_articulation_constraint.md). The previous-step state $\mathbf{q}^t$ is the solver's own previous-step DOF; no `ref_dof_prev` attribute is required.

The energy function is a quadratic penalty on the increment error:

$$
E = \frac{s}{2} \left(\delta\theta - \tilde{\delta\theta}\right)^2,
$$

where:

- $s$ is the per-edge `pd/strength`, the joint stiffness **directly** (unit: torque per radian). Unlike the driving joint, there is no body-mass scaling $K = \gamma(m_i + m_j)$, and the energy is a single term in the scalar $\delta\theta$.
- $\tilde{\delta\theta}$ is the per-edge `pd/aim_increment`, the target angle increment for the current time step.

The gradient (the generalized force applied to the affine body DOFs) is

$$
\mathbf{g} = s\left(\delta\theta - \tilde{\delta\theta}\right) \frac{\partial \delta\theta}{\partial \mathbf{q}},
$$

so at the converged state the joint carries the torque $s(\tilde{\delta\theta} - \delta\theta)$. This realizes an implicit PD controller with gains $k_p$, $k_v$ when the animator sets

$$
s = k_p + \frac{k_v}{\Delta t}, \quad
\tilde{\delta\theta} = \frac{k_p\, q_{\text{des}} + (k_v/\Delta t)\left(q_n + v_{\text{des}}\, \Delta t\right)}{s} - q_n,
$$

where $q_{\text{des}}$, $v_{\text{des}}$ are the desired joint position and velocity, $q_n$ is the joint position at the beginning of the step, and $\Delta t$ is the time step.

The Hessian is

$$
\mathbf{H} = s\, \frac{\partial \delta\theta}{\partial \mathbf{q}} \frac{\partial \delta\theta}{\partial \mathbf{q}}^{\top} + s\left(\delta\theta - \tilde{\delta\theta}\right) \frac{\partial^2 \delta\theta}{\partial \mathbf{q}^2}.
$$

The first (Gauss-Newton) term is positive semi-definite for any $s \ge 0$. The second-order term can be indefinite, so it is projected to be positive semi-definite before assembly — the same treatment used by the [Affine Body Revolute Joint Limit](./affine_body_revolute_joint_limit.md) and the [External Articulation Constraint](./external_articulation_constraint.md) for the same $\partial^2\delta\theta/\partial\mathbf{q}^2$ kernel.

When `pd/is_constrained = 0`, the energy, gradient, and Hessian are all zero and the driving effect is disabled.

## Requirement

This constitution must be applied to a geometry that already has an [AffineBodyRevoluteJoint](./affine_body_revolute_joint.md) (UID=18) constitution; it reuses that joint's linking data (`l_geo_id` / `r_geo_id` / `l_inst_id` / `r_inst_id`) and per-body joint basis by edge index.

## Attributes

On the joint geometry (1D simplicial complex), on **edges** (one edge per joint). The edge inherits all linking and state fields of the base [Affine Body Revolute Joint](./affine_body_revolute_joint.md): `l_geo_id`, `r_geo_id`, `l_inst_id`, `r_inst_id`, `strength_ratio`, `angle`, `init_angle`, and optional `l_position0`, `l_position1`, `r_position0`, `r_position1` when created via Local `create_geometry`.

Driving-specific attributes on **edges**:

- `pd/strength` (`Float`): $s$, the per-edge joint stiffness ($= k_p + k_v/\Delta t$ for implicit PD). Default `0.0`.
- `pd/aim_increment` (`Float`): $\tilde{\delta\theta}$, the target angle increment for the current time step. Default `0.0`.
- `pd/is_constrained` (`IndexT`): enables (`1`) or disables (`0`) the driving effect. Default `0`.

`apply_to` resets all three attributes to their defaults; the animator (controller) is expected to overwrite them every time step.

## Note on Tracking Fidelity

The constitution's angle measurement is faithful: $\delta\theta(\mathbf{q}, \mathbf{q}^t)$ and $\partial\delta\theta/\partial\mathbf{q}$ are identical to the absolute-angle driving joint's $\theta$ and $\partial\theta/\partial\mathbf{q}$ up to the constant offset $\theta^t$, for rigid and non-rigid states at any angle. The incremental parametrization therefore introduces no fidelity loss by itself.

A tracking lag can nonetheless appear, and it is a property of how the **driver** computes the target $\tilde{\delta\theta}$, not of this constitution. $\tilde{\delta\theta}$ must be the increment to the target measured in the *same* angle convention this constitution uses (the symmetric basis-dot $\operatorname{atan2}$ above). If the driver instead forms $\tilde{\delta\theta}$ from a *different* angle reconstruction of the previous state — e.g. a joint angle extracted from the body transforms via quaternions — the two reconstructions agree only while the bodies are rigid. Under the affine scale/shear that develops in a stiff solve they diverge, the target is mis-referenced, and the joint tracks with a gain- and motion-dependent lag (observed up to $\sim 3\times$ the single-DOF RMS error at high stiffness).

This is a genuine tension, not a defect. Making $\tilde{\delta\theta}$ self-consistent by referencing it to a **fixed** pose (build or rest) closes the lag exactly, but reintroduces the $\pm\pi$ branch cut of the absolute angle once a joint's excursion from that pose exceeds $\pi$ — defeating the purpose of the incremental form. Referencing a recent state in the *same* convention cancels back to the absolute angle, with the same branch cut. A **non-stateful** driver therefore chooses between branch-cut robustness with a modest lag (recent, possibly mismatched reconstruction) and exact tracking with branch-cut fragility (fixed reference). Closing the lag *while* staying branch-cut-robust requires **stateful continuous-angle unwrapping**, i.e. tracking the winding number so the absolute basis-dot angle never wraps. Equilibrium and holding behavior are unaffected in all cases.
