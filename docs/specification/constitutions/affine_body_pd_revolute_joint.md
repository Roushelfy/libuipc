# Affine Body PD Revolute Joint

> **Status: design / deferred — NOT yet implemented.** The implicit-PD behavior this spec
> describes currently ships by *folding* it onto the existing
> [Affine Body Driving Revolute Joint](./affine_body_driving_revolute_joint.md) (UID=19):
> since the PD energy is exactly its `½K(θ−θ̃)²` with `K=γ(m_i+m_j)`, a controller sets
> `γ=(kp+kv/dt)/(m_i+m_j)`, `aim_angle=θ̃` each step. This dedicated constitution is a future
> option, worthwhile only for a built-in `force_range` torque clamp or a `kp/kv`-direct API
> (so the controller need not know the body masses). UID 31 is reserved for it.

References:

[A unified newton barrier method for multibody dynamics](https://dl.acm.org/doi/pdf/10.1145/3528223.3530076)

## #31 AffineBodyPDRevoluteJoint

The **Affine Body PD Revolute Joint** is a constraint constitution that actuates a [Revolute Joint](./affine_body_revolute_joint.md) (UID=18) with a **proportional–derivative (PD) controller evaluated implicitly** inside the Newton solve. It must be applied to a geometry that already has an `AffineBodyRevoluteJoint` constitution.

Unlike the [Affine Body Driving Revolute Joint](./affine_body_driving_revolute_joint.md) (UID=19), which is a pure stiffness penalty whose gain is scaled by the body masses, the PD joint uses **physical gains** `kp` (N·m/rad) and `kv` (N·m·s/rad) directly and adds a **velocity term**. Its gradient at the converged state is exactly the PD torque

$$
\tau = k_p\,(q_{\text{des}} - \theta) + k_v\,(v_{\text{des}} - \dot\theta),
$$

evaluated at the **new** state $(\theta_{n+1}, \dot\theta_{n+1})$. Because the controller enters as a convex quadratic in the unknown body states, it is **unconditionally stable** for any $k_p, k_v \ge 0$ — there is no explicit $k_p\,\mathrm{d}t^2/I < 4$ or $k_v\,\mathrm{d}t/I < 2$ step-size bound. This is the implicit counterpart of a position/velocity actuator and matches a fully-implicit PD integrator (e.g. MuJoCo/Genesis `integrator=implicit`).

The PD joint supports two operating modes:

- **Active mode** (`is_passive = 0`): the joint is driven toward the targets `pd/aim_angle` ($q_{\text{des}}$) and `pd/aim_velocity` ($v_{\text{des}}$).
- **Passive mode** (`is_passive = 1`): the joint resists external forces by treating the current angle as the position target with zero velocity target, effectively a critically-tunable hold.

The constraint can be toggled on and off at runtime via the `pd/is_constrained` flag.

## Energy

We assume 2 affine body indices $i$ and $j$, each with their own state vector $\mathbf{q}_i$ and $\mathbf{q}_j$ as defined in the [Affine Body](./affine_body.md) constitution. The current relative rotation angle $\theta(\mathbf{q})$ about the joint axis is extracted exactly as for the base [Revolute Joint](./affine_body_revolute_joint.md) / [Driving Revolute Joint](./affine_body_driving_revolute_joint.md) (the $\operatorname{atan2}$ of the per-body $(\hat{\mathbf n}_k, \hat{\mathbf b}_k)$ basis), and $\theta_{\text{init}}$ is the base joint's `init_angle`.

Let $\mathrm{d}t$ be the time step (provided by the integrator), and let $\theta_n$ be the reported `angle` at the **start** of the step (the base [Revolute Joint](./affine_body_revolute_joint.md)'s `angle`, held fixed during the solve). Backward-Euler relates the joint velocity to the unknown angle by $\dot\theta = (\theta - \theta_n)/\mathrm{d}t$.

The energy is the sum of a position-stiffness penalty and a Rayleigh dissipation (damping) penalty, both written in the raw-angle space (subtract $\theta_{\text{init}}$ to convert the user-facing targets):

$$
E = \frac{k_p}{2}\,\bigl(\theta - \tilde\theta_{p}\bigr)^2
  + \frac{k_v}{2\,\mathrm{d}t}\,\bigl(\theta - \tilde\theta_{d}\bigr)^2,
$$

$$
\tilde\theta_{p} =
\begin{cases} q_{\text{des}} - \theta_{\text{init}}, & \text{is\_passive}=0 \\ \theta_n - \theta_{\text{init}}, & \text{is\_passive}=1 \end{cases}
\qquad
\tilde\theta_{d} = (\theta_n - \theta_{\text{init}}) + v_{\text{des}}\,\mathrm{d}t .
$$

The gradient is the PD torque evaluated at the current (implicit) angle:

$$
-\frac{\partial E}{\partial \theta}
= k_p\,(\tilde\theta_p - \theta) + \frac{k_v}{\mathrm{d}t}\,(\tilde\theta_d - \theta)
= k_p\,(q_{\text{des}} - \theta) + k_v\,(v_{\text{des}} - \dot\theta),
$$

using $\dot\theta = (\theta - \theta_n)/\mathrm{d}t$. The two quadratics combine into a single penalty $\tfrac{K}{2}(\theta - \tilde\theta)^2$ with

$$
K = k_p + \frac{k_v}{\mathrm{d}t}, \qquad
\tilde\theta = \frac{k_p\,\tilde\theta_p + (k_v/\mathrm{d}t)\,\tilde\theta_d}{K},
$$

so the **energy, gradient and Hessian reuse the [Driving Revolute Joint](./affine_body_driving_revolute_joint.md) kernels** with $K$ and $\tilde\theta$ in place of $\gamma(m_i+m_j)$ and the driving target. The Hessian contribution is $K\,(\partial\theta/\partial\mathbf q)(\partial\theta/\partial\mathbf q)^\top + K(\theta-\tilde\theta)\,\partial^2\theta/\partial\mathbf q^2$; since $K>0$ the penalty is convex in $\theta$, giving an SPD contribution near the operating point.

As with the driving joint, $\theta - \tilde\theta$ is a raw difference of $\operatorname{atan2}$ values in $(-\pi,\pi]$ and is **not** wrapped; the penalty is meaningful while $|\theta - \tilde\theta| < \pi$ (the controller is expected to deliver a target close to the current angle). When `pd/is_constrained = 0`, the energy is zero and the actuation is disabled.

The current angle $\theta(\mathbf q)$ and the offset $\theta_{\text{init}}$ / start-of-step angle $\theta_n$ are read from the **base** [AffineBodyRevoluteJoint](./affine_body_revolute_joint.md), which tracks them on the `angle` / `init_angle` edge attributes. The PD joint only consumes these values — it does not own them.

## Requirement

This constitution must be applied to a geometry that already has an [AffineBodyRevoluteJoint](./affine_body_revolute_joint.md) (UID=18) constitution.

## Attributes

On the joint geometry (1D simplicial complex), on **edges** (one edge per joint). The edge inherits all linking and state fields of the base [Affine Body Revolute Joint](./affine_body_revolute_joint.md): `l_geo_id`, `r_geo_id`, `l_inst_id`, `r_inst_id`, `strength_ratio`, `angle`, `init_angle`, and optional `l_position0`, `l_position1`, `r_position0`, `r_position1` when created via Local `create_geometry`.

PD-specific attributes on **edges**:

- `pd/kp`: $k_p$, the proportional (position) gain (N·m/rad)
- `pd/kv`: $k_v$, the derivative (velocity) gain (N·m·s/rad)
- `pd/aim_angle`: $q_{\text{des}}$, the position target in active mode (user-facing frame)
- `pd/aim_velocity`: $v_{\text{des}}$, the velocity target (rad/s); defaults to `0`
- `pd/is_constrained`: enables (`1`) or disables (`0`) the PD actuation
- `is_passive`: passive mode (`1`) holds the current angle; active mode (`0`) drives to the targets
