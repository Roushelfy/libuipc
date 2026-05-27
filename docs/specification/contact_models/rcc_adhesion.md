# RCC Adhesion

**RCC Adhesion** is a contact-based adhesive model that augments IPC contact with normal and tangential adhesion. The model is defined on active contact pairs. Each contact pair stores an adhesion intensity $\beta_k \in [0,1]$, which is updated explicitly at the beginning of each time step and kept fixed during the implicit solve.

## RCCAdhesiveContact

For each active contact pair $k$, define:

- $D_k=d_k^2$: squared distance of the contact pair
- $d_k=\sqrt{D_k}$: distance of the contact pair
- $A_k$: contact area or quadrature weight
- $\beta_k$: adhesion intensity
- $u_k$: relative tangential sliding displacement

The model supports the simplex contact pairs used by IPC contact and friction:

- point-triangle (PT)
- edge-edge (EE)
- point-edge (PE)
- point-point (PP)

### Normal Adhesion

The normal adhesion energy density of contact pair $k$ is:

$$
P_{na,k}=\frac{C_n}{2}\beta_k^2 d_k^2
$$

where $C_n$ is the normal adhesion stiffness. Since the IPC contact pipeline already evaluates squared distance, the implementation uses $D_k=d_k^2$ directly:

$$
E_{na}=\sum_k A_k\frac{C_n}{2}\beta_k^2D_k
$$

For each contact pair,

$$
\nabla E_{na,k}=A_k\frac{C_n}{2}\beta_k^2\nabla D_k
$$

and

$$
\nabla^2E_{na,k}=A_k\frac{C_n}{2}\beta_k^2\nabla^2D_k.
$$

The normal adhesive pressure is:

$$
p_{na,k}=-C_n\beta_k^2d_k.
$$

### Tangential Adhesion

The tangential adhesion energy density of contact pair $k$ is:

$$
P_{ta,k}=\frac{C_t}{2}\beta_k^2\|u_k\|^2
$$

where $C_t$ is the tangential adhesion stiffness. The total tangential adhesion energy is:

$$
E_{ta}=\sum_k A_k\frac{C_t}{2}\beta_k^2\|u_k\|^2.
$$

The relative tangential displacement $u_k$ is evaluated using lagged closest-point coordinates and lagged tangent bases, following the same convention as simplex friction. With the lagged quantities fixed during the current time step,

$$
u_k=J_k(\mathbf{x}-\mathbf{x}^t)
$$

where $J_k$ is the tangential relative displacement Jacobian and $\mathbf{x}^t$ is the position at the beginning of the time step.

For each contact pair,

$$
\nabla E_{ta,k}=A_k C_t\beta_k^2 J_k^T u_k
$$

and

$$
\nabla^2E_{ta,k}=A_k C_t\beta_k^2 J_k^T J_k.
$$

Under the lagged-basis approximation, the tangential adhesion Hessian is positive semi-definite.

### Beta Evolution

The adhesion intensity is updated explicitly at the beginning of each time step:

$$
\beta_k \leftarrow \min(1,\max(0,\beta_k+h\dot\beta_k)).
$$

During the implicit solve, $\beta_k$ is fixed.

Let

$$
p_k=p_{na,k}+p_b
$$

be the total normal pressure. The barrier pressure is:

$$
p_b=-\hat d\frac{\partial b}{\partial D}(D_k)2d_k,
$$

where $D_k=d_k^2$. Therefore,

$$
p_k=\left(-C_n\beta_k^2-\hat d\frac{\partial b}{\partial D}(D_k)2\right)d_k.
$$

When $p_k<0$, adhesion dominates and debonding is triggered:

$$
\dot\beta_k=
\frac{1}{\eta}\min(W-C_n\beta_kd_k^2-C_t\beta_k\|u_k\|^2,0).
$$

When $p_k>0$, contact dominates. Normal compression triggers bonding, while tangential motion may still trigger debonding:

$$
\dot\beta_k=
r\max(p_k-\beta_kp_0,0)
+
\frac{1}{\eta}\min(W-C_t\beta_k\|u_k\|^2,0).
$$

where:

- $W$ is the maximum adhesion energy
- $\eta$ is the viscosity parameter
- $r$ is the bonding rate
- $p_0$ is the compression value for saturation

## Attributes

On contact model:

- `Cn`: $C_n$, normal adhesion stiffness
- `Ct`: $C_t$, tangential adhesion stiffness
- `W`: $W$, maximum adhesion energy
- `eta`: $\eta$, viscosity parameter
- `bonding_rate`: $r$, bonding rate
- `p0`: $p_0$, compression value for saturation
- `initial_beta`: initial adhesion intensity for newly created contact pairs

On contact state:

- `topo`: vertex indices of the contact pair
- `type`: contact pair type, one of PT, EE, PE, or PP
- `beta`: $\beta_k$, adhesion intensity
- `area`: $A_k$, contact area or quadrature weight

On shell vertices (optional, v3+):

- `rcc_sticky_sign` <IndexT>: $-1$, $0$, or $+1$. Default $0$ keeps double-sided adhesion (v2 behaviour). $\pm 1$ enables single-sided adhesion where only the $\pm \hat n$ face of the shell participates in the adhesion energy. See "Single-sided adhesion (oriented shells)" below.

## Notes

RCC Adhesion is additive to IPC barrier contact and friction. It should not disable the barrier term or friction term.

The normal adhesion term should use squared distance $D_k$ directly. The tangential adhesion term should use lagged tangent bases and lagged closest-point coordinates, so that $u_k$ is linear in the current displacement.

The adhesion state should be matched across time steps using stable contact-pair keys. New contact pairs use `initial_beta`. Inactive contact pairs may be removed or kept for a small number of steps, depending on the backend implementation.

## Backend implementation notes (CUDA, v2)

The CUDA backend ports XBow's `RCCAdhesionEnergy3D` (`XBow-main/src/Bow/Energy/FEM/RCCAdhesionEnergy.h`). Differences from the bare formulas above:

- **Adaptive scaling**: $C_n$ and $C_t$ are treated as area-weighted stiffnesses divided by $\hat d$ at energy time, so user-facing $C_n$/$C_t$ can be set in physical units (e.g. Young's-modulus-like). The actual energy is $\tfrac{C_n}{2\hat d}\beta^2 D$ (not $\tfrac{C_n}{2}\beta^2 d^2$), and similarly for tangential. The $\beta$-evolution formulas pick up matching $r_{\text{scale}}$, $W_{\text{scale}}$, and $\eta\cdot W_{\text{scale}}/10$ factors so user-facing $r$/$\eta$/$W$ are normalised quantities.
- **Area $A_k$**: lumped into $C_n$/$C_t$ (libuipc IPC convention); not plumbed as a separate per-pair attribute.
- **Energy scale**: each per-pair energy is multiplied by $dt^2$ to match libuipc's `kt2 = \kappa \cdot dt^2` convention used by the IPC barrier and friction reporters.

### Pair-type scope

Adhesion (energy, gradient, Hessian, and $\beta$ evolution) runs **only on point-triangle (PT) pairs**. PE/PP/EE adhesion and vertex–half-plane adhesion are **disabled**.

**Why**: the libuipc trajectory filter classifies a candidate (vert, tri) pair into PT/PE/PP based on which sub-feature is closest (interior, edge, or vertex). The IPC barrier — which is *repulsive* — handles all three feature types symmetrically, since "push away from the closest sub-feature" is direction-consistent. RCC adhesion is *attractive*, so the gradient becomes "pull toward the closest sub-feature": on faceted meshes (e.g. the diagonal that splits a cube face into two triangles), the same cloth vert hovering above the face interior emits a PT pair against one triangle and a PE pair (against the shared diagonal edge) against the other — and the PE pair's attractive gradient pulls the cloth sideways toward the diagonal. Restricting v2 adhesion to PT pairs eliminates this artifact. The PT gradient itself uses the *plane-projection* form ($g_{PT}$) regardless of where the perpendicular foot lands, so it always pulls perpendicular to the triangle's plane (no sub-feature dispatch inside the PT branch either).

### Single-sided adhesion (oriented shells)

Each shell vertex may carry an optional `rcc_sticky_sign` <IndexT> attribute with values $-1, 0, +1$ (default $0$ — double-sided, identical to v2 behaviour). When non-zero on either endpoint of a PT pair $(P, T)$, the adhesion contribution is **gated** by

$$
\underbrace{\bigl(P - \mathrm{closest}_T(P)\bigr) \cdot \bigl(s_P \cdot \hat n_P\bigr) < 0}_{P\text{-side: } P\text{'s sticky face faces } T}
\quad\lor\quad
\underbrace{\bigl(P - \mathrm{closest}_T(P)\bigr) \cdot \bigl(s_T \cdot \hat n_T\bigr) > 0}_{T\text{-side: } T\text{'s sticky face faces } P},
$$

where $s_P, s_T$ are the sticky signs at $P$ and at any vertex of triangle $T$, and $\hat n_P, \hat n_T$ are the corresponding shell vertex normals. The disjunction reflects "either side's sticky face engaging the contact is enough to bond." When both signs are zero the gate trivially passes (v2 fallback). When the gate fails, the adhesion energy, gradient, Hessian, and $\beta$-evolution all return zero on that pair; only the IPC barrier survives.

$\hat n_P$ and $\hat n_T$ are **lagged**: they are recomputed once per step from the begin-of-step positions and held constant through that step's Newton iterations (same convention as the friction tangent basis). Concretely each is the area-weighted average of the incident-triangle face normals.

**Self-contact / rolled-up tape**: shell-shell PT pairs are emitted in both directions by the trajectory filter — $(P\in A, T\in B)$ and $(P\in B, T\in A)$. As long as one PT pair has at least one side's sticky face engaged, adhesion fires at that contact. So a tape rolled with the sticky face on the inside will bond adjacent turns (outer turn's sticky face touches inner turn's non-sticky face — the outer turn's P-side or the inner turn's T-side passes). Two truly non-sticky faces back-to-back have all gates fail → no bond.

**Shell-vs-rigid**: when one side is a closed body or a rigid that did not call `set_sticky_side`, its sticky sign is 0 and that side does not contribute to the disjunction — the gate is decided entirely by the shell's preference. So a tape with sticky-up bonds only to objects it touches with its sticky face, never with its back face, regardless of which PT direction (cube vert vs cloth tri or vice versa) the trajectory filter happens to emit.

**Frontend API**: `RCCAdhesive::set_sticky_side(geo, sign)` writes the per-vertex attribute across an entire geometry. Per-vertex granularity is also possible by writing the attribute directly.

### Persistence

Per-pair $\beta$ is keyed by a sorted-vertex-tuple `u64` hash of the PT stencil. At end of each step (`Phase A`, called from `RCCBetaEvolutionTimeIntegrator::do_update_state`):

1. For each active PT pair, advance $\beta$ via XBow's debonding/bonding rule using the squared distance $D$ at end-of-step and the tangential displacement $u$ over the step (current positions minus a snapshot taken at start-of-step).
2. Snapshot `(sorted_keys, β)` into a sorted device array.

At the start of each new step (`Phase B`, triggered on the first `do_compute_energy` call after the frame counter advances):

1. Recompute keys for the new active PT pair list (which may have grown/shrunk as pairs entered or left the IPC active band).
2. For each current pair, binary-search the prev-step snapshot. **Hit** → carry $\beta$ forward. **Miss** → new pair, apply XBow's new-pair bonding kick (β starts at 0, gets a single bonding step), clamped against `initial_beta` so users can force a "start fully bonded" sim by setting `initial_beta = 1.0`.

When a stencil flips feature type across steps (PT ↔ PE ↔ PP), it changes key, and the lookup misses → $\beta$ resets via the new-pair kick. This is a known limitation of the v2 key scheme.

### Lifecycle

Phase A is invoked once per step at end-of-step via the existing `TimeIntegrator::do_update_state` hook (same per-step path as plasticity uses). No new SimEngine event was added. Phase B runs inside `do_compute_energy` and uses an internal frame-counter to fire only on the first call per frame; subsequent Newton iterations just read the populated `m_beta_PT` buffer.

### SPD projection

`make_spd` is applied to the PT normal-adhesion Hessian (potentially indefinite for general triangle stencils). The PT tangential Hessian is naturally PSD ($J^T J$ scaled by a non-negative coefficient).

### Deferred

- PE/PP/EE adhesion: requires either a deduplication step in the trajectory filter (so each geometric contact emits exactly one stencil) or a smoothed signed-distance gradient (no sub-feature dispatch).
- Vertex–half-plane adhesion (`IPCVertexHalfPlaneRCCAdhesiveContact`).
- A user-facing knob to toggle XBow's adaptive scaling on/off (current code is hardwired to "on").

## References

- Yu Fang, Minchen Li, Yadi Cao, Xuan Li, Joshuah Wolper, Yin Yang, and Chenfanfu Jiang. Augmented Incremental Potential Contact for Sticky Interactions. IEEE Transactions on Visualization and Computer Graphics, 2024.