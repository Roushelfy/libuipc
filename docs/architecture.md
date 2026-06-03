# Architecture

This document is the stable architecture for RCC bonded point-triangle acceleration. It describes intended ownership and data flow; date-specific evidence lives in [the journal](./development/rcc_adhesion_acceleration_journal.md).

## Motivation

RCC adhesion currently evaluates long-lived point-triangle pairs through the same active contact pipeline used for transient contact. Stable sticky pairs repeatedly pay for collision detection, CCD filtering, contact visibility, friction candidate recording, RCC beta matching, and RCC adhesion assembly even when their topology and adhesion state are unlikely to change.

The performance thesis is to classify stable RCC PT pairs, remove them from the contact/RCC hot path, and replace them with a bonded virtual tetrahedron over the same four global vertices until a release gate fails.

## Core Thesis

Stable adhesive contact should become a complement energy with explicit ownership, not a hidden modification of contact detection. The design is valid only if the locked pair is absent from contact/RCC views and present in exactly one bonded virtual-tet reporter for the same step.

For production bonded PT acceleration, a skipped pair must be backed in the same step by high-kappa ABD-style energy over the same four vertex position DOFs. A locked PT that skips CCD/contact/RCC but does not receive the ABD-style virtual-tet energy in that step is an invalid production state, even if beta history says the pair is stable.

## Baseline Data Flow

```text
AdvanceIPC frame
  -> record previous positions
  -> DCD emits active simplex pairs
  -> record_friction_candidates copies PTs() to friction_PTs()
  -> RCC Phase B matches friction_PTs() against m_prev_keys_PT
  -> normal contact, friction, RCC adhesion assemble from trajectory-filter views
  -> line search calls filter_toi()
  -> RCCBetaEvolutionTimeIntegrator runs Phase A beta update
  -> m_prev_keys_PT / m_prev_beta_PT are sorted for the next step
```

Relevant current anchors:

| Current Anchor | Role |
| --- | --- |
| `src/backends/cuda/engine/advance_ipc.cu` | Frame order, DCD, friction candidate recording, line-search CCD |
| `src/backends/cuda/collision_detection/simplex_trajectory_filter.cu` | `record_friction_candidates()` copies active PTs to friction PTs |
| `src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu` | RCC PT beta buffers, Phase A/Phase B, beta persistence |
| `src/backends/cuda/collision_detection/filters/*simplex_trajectory_filter.cu` | PT CCD broadphase and active-pair emission |
| `src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu` | Existing static vertex-triangle rest-shape and thickness reference |
| `src/backends/cuda/affine_body/constitutions/ortho_potential.cu` | Existing ABD OrthoPotential energy reference |
| `src/backends/cuda/affine_body/constitutions/arap.cu` | Existing ABD ARAP energy reference |

## Target Data Flow

```text
End of step
  -> RCC beta evolves on active PT pairs
  -> bonded-PT classifier tests beta, age, sticky side, gap, slip, rest shape
  -> RCCBondedPTState publishes sorted locked_keys and oriented locked_topos

Next detection/filtering
  -> simplex filter forms PT candidate ids
  -> shared locked-key lookup tests membership before PT CCD broadphase
  -> locked PTs are not emitted to PTs()
  -> record_friction_candidates cannot copy locked PTs to friction_PTs()

Newton assembly
  -> normal contact/friction/RCC assemble only unlocked pairs
  -> bonded virtual-tet reporter assembles high-kappa ABD-style complement energy for locked_topos, Dm_inv, and rest_volume

Release/update
  -> release gate evaluates active locks before reporter ownership is finalized
  -> release gate records reason flags
  -> released pairs carry beta back into RCC persistence
  -> released pairs are removed from bonded reporter input for that step
  -> still-locked pairs update age and diagnostics
```

The classifier and release evaluator may consume overlapping data, but they are different contracts. A sticky-side, gap, slip, or policy input routed into release does not by itself prove that the live lock producer rejects a bad candidate before it becomes a lock.

## Project Structure

```text
docs/
  roadmap.md
  architecture.md
  conventions.md
  rcc_adhesion_acceleration.md
  development/
    rcc_adhesion_acceleration_journal.md
scripts/
  run_rcc_adhesion_acceleration_all_gates.py
  run_rcc_adhesion_acceleration_gates.py
  build_docs.py
src/backends/cuda/
  collision_detection/
  contact_system/
  inter_primitive_effect_system/
apps/tests/
  core/
  backends/cuda/
  sim_case/
```

The `apps/tests` locations hold the current deterministic state, oracle, CUDA bridge, lookup, filter, owner, reporter, and scene gates. Source scans remain boundary checks, not numeric proof.

## Ownership And Boundaries

| Component | Owns | Must Not Own |
| --- | --- | --- |
| `RCCBondedPTState` | Locked membership keys, oriented topologies, beta, age, release flags, rest-shape metrics | Contact force assembly, BVH traversal, frontend geometry objects |
| Simplex trajectory filters | Candidate generation, PT CCD broadphase, active-pair output | Beta evolution, material parameters, virtual-tet energy |
| `IPCSimplexRCCAdhesiveContact` | RCC beta evolution and persistence for active/unlocked PT pairs | Dynamic tet Hessian assembly, filter-specific skip logic |
| Bonded virtual-tet reporter | High-kappa ABD-style complement energy, gradient, Hessian for locked topologies | RCC beta law, broadphase decisions, contact-component accounting |
| Global dynamic topology manager | Aggregation and scattering of complement energy | Classification policy or release policy |
| `RCCBondedPTStateAccessorFeature` | Frontend snapshots of active locked state and counters | Physics decisions, implicit synchronization outside explicit query points, or release-history claims before released snapshots are exposed |
| Bench/report layer | Timers and counters | Physics decisions |

## Runtime Data Model

Use two keys for different jobs:

| Data | Purpose | Required Property |
| --- | --- | --- |
| `locked_keys` | Membership lookup in filters and beta matching | Sorted by key, one entry per locked PT ownership record |
| `locked_topos` | Energy assembly topology | Oriented `(point, tri0, tri1, tri2)` and permuted with `locked_keys` |
| `locked_beta` | Carry adhesion state through lock/release | Matched to RCC beta convention |
| `locked_age` | Enforce minimum stable lifetime before lock/reuse | Incremented only after accepted steps |
| `Dm_inv` | Virtual-tet rest-shape inverse | Finite, conditioned, derived from rest positions |
| `rest_volume` | ABD-style virtual-tet volume scale | Positive and above configured minimum |
| `release_flags` | Device-generated release reasons | Stable enum or bit mask for reports/tests |

The sorted membership key can match current RCC persistence behavior, but it is not enough for energy. Energy must use oriented topology because orientation controls `Dm`, normal direction, and rest volume sign handling.

## Frame Lifecycle Contract

| Stage | Locked Pair Requirement | Test Or Report |
| --- | --- | --- |
| Classification | Candidate accepted only after all lock gates pass | State fixture and candidate/rejected counters; lock-gate parity fixture still planned |
| DCD/PT emission | Locked PT is skipped before PT CCD broadphase | Filter contract and `filter_skip_count` |
| Friction candidate recording | Locked PT cannot enter `friction_PTs()` | Contract test reading trajectory-filter views |
| RCC Phase B | Released pairs can receive carried beta | Beta carry fixture |
| Newton assembly | Locked PT appears only in the ABD-style bonded reporter | Duplicate-suppressed counter, E/G/H oracle, and scene no-penetration observation |
| End-of-step update | Release reasons and beta state are recorded | Backend release fixture now; scene-accessible released snapshots still planned |

## Virtual Tet Rest Shape And Energy

The dynamic reporter uses `SoftVertexTriangleStitch` only for the PT rest-shape convention:

1. Build `Dm = [x1 - x0, x2 - x0, x3 - x0]` from rest positions.
2. If point-plane rest distance is below `min_separate_distance`, offset the rest point along the triangle normal before computing `Dm_inv`.
3. Store `Dm_inv` and positive `rest_volume`.

The production bonded energy must be ABD-style, not Stable Neo-Hookean. A locked pair is removed from CCD/contact/RCC, so the replacement energy is responsible for making the four vertices behave like a stiff bonded patch during that step. The ABD energy is a replacement for the skipped geometric/contact work, not the RCC adhesion law itself; beta remains RCC state and must be restored on release. While a pair is locked it is removed from `friction_PTs()`, so its beta does not evolve: the lock-time beta is frozen and carried until release, and the spec's energy-driven debonding law is replaced for that pair by the geometric release gates (strain/gap/slip) plus sticky-side/policy. This is a deliberate approximation and must be calibrated against the unaccelerated beta evolution before any correctness claim (see the conventions Test Matrix).

For the same four real vertex positions, build:

$$
D_s = [x_1 - x_0, x_2 - x_0, x_3 - x_0], \qquad F = D_s D_m^{-1}.
$$

Treat `F` as the virtual affine transform. The default target model is ABD OrthoPotential:

$$
E = \kappa \, V_0 \, \Delta t^2 \, \|F F^T - I\|_F^2,
$$

where `V0 = rest_volume` and `kappa = rcc_bonded_pt_kappa`. `abd_arap` may be supported as an alternate model only if it has the same CPU and GPU E/G/H oracle coverage. The default production gate value is `kappa >= 1e8` in scene units, with higher values allowed when solver conditioning gates pass.

Implementation rules:

- Do not instantiate a real frontend ABD body for each lock; assemble the ABD-style energy directly into the same four vertex DOFs through `dF/dx`.
- Keep `rcc_bonded_pt_energy_model` explicit; default target model is `abd_ortho`.
- Treat missing, zero, negative, or unsupported production energy settings as a hard diagnostic failure when bonded PT acceleration is asked to skip CCD/contact/RCC.
- Keep bonded PT acceleration default-off until high-kappa ABD oracle, no-penetration scene, release, and benchmark gates pass.
- Apply SPD projection in the Hessian path.
- Do not reintroduce the old Stable Neo-Hookean `rcc_bonded_pt_mu/lambda` reporter as a production path; it did not justify skipping CCD and has been replaced by ABD-style energy.

## CCD Removal Precondition

Skipping CCD for a locked pair is one source of speedup (the per-iteration CCD broadphase/TOI cost; the assembly and linear-system-conditioning wins are independent of it and already active), but it is also the only step that removes the engine's penetration guarantee. The current code does not yet skip CCD: locked PTs are removed only from the DCD active-pair view in `do_filter_active`, while PT CCD broadphase and the `do_filter_toi` TOI line search still process them. That accidental retention is currently the only thing preventing a locked point from tunneling through its triangle, because the bonded ABD energy cannot prevent it on its own.

Invariant (testable): `E = kappa * V0 * dt^2 * ||F F^T - I||^2` depends only on `F F^T`, so it is invariant to the sign of `det(F)`. An inverted configuration where the point has crossed to the mirror side of the triangle has the same energy and gradient as the correct side. The bonded energy is therefore a shape-preservation term, not a non-penetration barrier, and gives no restoring force toward the correct side once inverted.

Consequence: locked PTs may be removed from PT CCD broadphase and `do_filter_toi` only after both of the following pass.

1. The `pt_lift_release` no-penetration scene gate observes no penetration during press/hold/lift with CCD skipped for locked pairs.
2. Inversion/tunneling is handled without CCD, e.g. the flip release reason is evaluated before the bonded reporter consumes its input (not only at end-of-step), or a separated point-triangle barrier is retained for locked pairs.

Until both hold, keep CCD active for locked pairs and do not claim a CCD speedup.

## Release And Beta Carry Contract

Release is the normal exit path for a bonded approximation. It is evaluated from the live locked state, current or predicted positions, and scene policy. If any release condition fires, the pair must stop being reporter input for that step and must re-enter RCC persistence with its last locked beta.

Required release data:

| Data | Purpose |
| --- | --- |
| Released key | Reconnect to RCC PT beta persistence |
| Released oriented topology | Debug ownership and scene assertions |
| Released beta | Avoid adhesion history reset on fallback |
| Released age | Distinguish early churn from long-lived locks |
| Release flags | Explain strain/gap/slip/flip/sticky-side/policy fallback |
| Release counters | Prove one-shot accounting and scene gate behavior |

The first release fixture should be deterministic: one lock stays active, one lock releases by a controlled reason, active bonded buffers are compacted, released buffers preserve key/topology/beta/age/flags alignment, and the RCC previous-beta snapshot can observe the released beta.

Scene gates need one more diagnostic layer: released topology, age, and flags must be available through an accessor/report path, not only through backend owner buffers used by CUDA tests and RCC beta carry.

## Full-Feature Adhesion And Per-Primitive Beta

Status: in progress. This extends RCC adhesion itself so the bonded acceleration can cover every stable contact, not only face-interior point-plane pairs. The journal records the XBow reference reading that motivated it.

### Why

The current RCC adhesion only assembles point-triangle (face-interior) adhesion: the `EE_*`, `PE_*`, and `PP_*` adhesion functions return zero, RCC beta is stored only for PT-classified pairs (`m_beta_PE`/`m_beta_PP` are zero-filled), and there is no contact-area weight. The bonded-PT acceleration inherits the same scope, so a stable contact whose closest feature is an edge or vertex (a cube corner, a cloth fold) gets neither adhesion nor a bond. The off-diagonal-corner exclusion observed on the faceted cube top face is a direct consequence: single-diagonal triangulation makes two corners classify as PE/PP, and those never enter `friction_PTs()`, never evolve beta, never lock.

### Reference model (XBow `RCCAdhesionEnergy3D`)

The reference treats the **vertex-triangle (VT)** and **edge-edge (EE)** pair as the contact primitive and:

1. stores beta per VT/EE primitive — keyed by `(boundary_point, boundary_face)`, carried across closest-feature transitions, not per closest-feature-classified pair;
2. each step classifies the closest feature (`point_triangle_distance_type` -> PP/PE/PT) and assembles adhesion with the **true feature distance** and the matching tangent basis for that classification;
3. weights each pair by the contacting vertex's tributary area (`boundary_point_area / 2`);
4. builds adhesion as an energy op that *inherits* the IPC barrier op's class scaffolding and the same `xi/dHat/kappa` parameter values — but it overrides `precompute()`, clears the inherited PP/PE/PT arrays, and rebuilds them from its OWN bonded pair set (`pair_PT/pair_EE`, grown by a separate spatial-hash broad phase + beta evolution), and recomputes distances independently. So the "sharing" is structural (scaffolding + parameter values), NOT a reused distance buffer and NOT a shared pair set. (Verified by reading XBow `RCCAdhesionEnergy.h:586/848/875` + `IPCSimulator.h:440/492`; the 2026-06-02 distance-reuse analysis in the journal corrects the earlier "reuses the barrier's distances" claim.)

### Target model in libuipc

1. **Primitive, not closest-feature, is the unit.** A VT pair is one adhesive primitive. The closest-feature classification (PP/PE/PT) only selects which distance sub-formula and tangent basis are used this step; it does not create or destroy the primitive.
2. **Beta per primitive.** Beta is keyed by the VT primitive (the orientation-invariant, point-distinct key already used by `PT_pair_key`) and evolves every step from the **true closest-feature distance**, regardless of which sub-formula is active. It is carried across PP<->PE<->PT transitions; no beta pop when the closest feature changes. `m_beta_PE`/`m_beta_PP` stop being zero-filled.
3. **Full-feature adhesion.** The `PE_*`/`PP_*` (and later `EE_*`) adhesion energy/gradient/Hessian use the true point-edge / point-point distance (`point_edge_distance2`, `point_point_distance2`) and the matching friction tangent basis (`point_edge_*`, `point_point_*`), mirroring the existing `PT_*` functions. The assembly skeleton already loops PE/PP and calls these functions; only the formula bodies (previously `return 0`) and per-primitive beta were missing.
4. **Contact area — currently lumped into Cn/Ct, NOT a separate factor.** libuipc has no per-pair area term anywhere: Cn/Ct are raw scalar stiffnesses and the assembled energy is `dt^2·Cn/(2 d_hat)·β²·D` with no `A_k`, exactly like the IPC barrier's `kappa` (which also does not area-weight per pair). The spec records this as the convention — area is *lumped into* Cn/Ct ([rcc_adhesion.md](./specification/contact_models/rcc_adhesion.md): "Area A_k: lumped into Cn/Ct (libuipc IPC convention)"). XBow instead keeps Cn/Ct as per-unit-area densities and multiplies a *separate* per-vertex tributary area `wPT = m_boundary_point_area/4` into the energy, reusing the same area field its barrier uses. So a separate `A_k` is **optional, not a bug fix**: under the lumped convention it would double-count (if users still bake area into Cn). It is justified only for resolution-independent / non-uniform-mesh adhesion (the lumped constant makes adhesion strength scale with vertex count, not physical contact area; and Cn is stored per contact-element-pair, so it physically cannot encode per-vertex area). If pursued, Cn/Ct must be reinterpreted as densities AND the per-vertex area reused from the barrier (do not invent a parallel field). This is the lowest-priority step.
5. **Distance handling — reuse is structural, not a stored buffer.** The barrier recomputes `d^2` and its derivatives in-kernel each pass and stores nothing (verified: `D` is materialized only under `RUNTIME_CHECK`, [ipc_simplex_normal_contact.cu:67-83](../src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu)), so there is no barrier distance to read. Adding a per-pair `d^2`/derivative scratch buffer would be a GPU regression — it trades cheap in-register arithmetic for global-memory traffic (the `Matrix12x12` Hessian alone is 144 floats/pair) on kernels that are launch/bandwidth-bound, not FLOP-bound. The genuinely-redundant, safely-shareable quantities are narrow: the barrier's *flagged* `d^2` across its own energy/assemble passes, and `db_dd2 = dKappaBarrierdD` (the beta law recomputes it bit-identically to the barrier's pressure derivative). RCC's *normal* adhesion deliberately uses the *unflagged* plane-projection distance (to avoid the faceted-cube diagonal-pull artifact), numerically different from the barrier's flagged distance, so it cannot be inherited from the barrier. The real lever is **structural single-compute** — compute `d^2`/basis once per pair within one kernel body, saving kernel launches, list-walks, and stencil gathers (the memory/launch cost that dominates), not the d² FLOPs. The cleanest merge target is **friction**, not the barrier: RCC is already a `SimplexFrictionalContact` subclass over the same lagged `friction_PTs()` list, and its tangential lagged tangent-basis/closest-foot is bit-identical to friction's `PT_friction_basis`. A barrier merge is harder (live `PTs()` vs lagged `friction_PTs()`, plus the flagged/unflagged divergence) and only `db_dd2` + the flagged `d^2` are safely shareable there.
6. **Bonded on the VT primitive; ABD tet stays point-plane.** Once beta exists for every VT primitive, the bonded lock decides on the VT primitive, so corner/edge contacts can bond. The bonded virtual-tet energy itself is unchanged: it remains the point-plane ABD shape energy over `F = Ds Dm_inv`, independent of which adhesion sub-formula classified the pair.

### Barrier invariant (do not change)

The IPC barrier must keep using the **true closest-feature distance** (PP/PE/PT via the flagged dispatch) for non-penetration. Only adhesion, beta evolution, and the bonded lock share that distance. The historical reason the PT *adhesion* used the unflagged plane projection (avoiding a diagonal pull artifact on faceted cubes) is subsumed by this model: PT keeps the plane formula because for a face-interior pair plane == true feature distance, while PE/PP pairs now use their own true feature distance and beta instead of being silently dropped.

### Implementation order

| Step | Change | Independently verifiable by |
| --- | --- | --- |
| 1 | PE/PP true-feature adhesion formulas (replace the `return 0` stubs), mirroring PT with `point_edge_*` / `point_point_*` | Compiles; behaviour-neutral while `m_beta_PE/PP == 0` (call sites early-out on `beta <= 0`); PE/PP E/G/H finite-difference oracle |
| 2 | Per-primitive beta: evolve beta for all VT primitives using the true closest-feature distance; expose `m_beta_PE/PP` | Beta nonzero for edge/corner pairs in a faceted-cube probe; PE/PP oracle must land first |
| 3 | (Optional, lowest priority) per-vertex contact area `A_k` weight — only for resolution-independent / non-uniform-mesh adhesion. Today area is lumped into Cn/Ct (barrier convention, per spec); adding A_k requires reinterpreting Cn/Ct as densities + reusing the barrier's boundary area, else double-count | Adhesion converges under mesh refinement instead of scaling with vertex count |
| 4 | Distance single-compute (structural, NOT a stored buffer): compute `d^2`/basis once per pair in one kernel body — first intra-RCC, then optionally merge with the friction kernel (shared lagged basis/foot); reject a per-pair `d^2` scratch buffer | Fewer kernel launches / list-walks in a profile trace; `db_dd2` shared with the barrier |
| 5 | Bonded consumes VT primitives; corner/edge pairs lock; ABD tet stays point-plane | Faceted-cube corners bond; `pt_lift_release` stays penetration-free |

## Relationship To Existing Systems

- RCC adhesion: provides beta evolution, sticky-side semantics, persistence keys, and (currently PT-only, moving to per-VT-primitive full-feature) adhesion scope.
- Simplex trajectory filters: provide the only high-performance place to skip locked PT work before CCD broadphase.
- Inter-primitive constitutions: provide complement-energy ownership and the existing SVTS rest-shape/thickness convention.
- Affine body constitutions: provide the ABD-style high-stiffness shape energy formulas that the bonded virtual tet should mirror over `F = Ds Dm_inv`.
- Dynamic topology manager: aggregates complement reporter output into global energy, gradient, and Hessian.
- Contact adaptive strategies: should not count bonded virtual tets as contact unless an explicit diagnostic adds comparison accounting.

## Failure Modes

| Failure | Symptom | Required Guard |
| --- | --- | --- |
| Late skip | Contact/RCC still pays CCD and assembly cost | Filter contract requires skip before PT CCD broadphase |
| Duplicate ownership | Pair is both contact/RCC and bonded tet | Duplicate counter and contract failure |
| Key/topology mismatch | Wrong four vertices or wrong orientation assembled | State fixture with key/topology permutation check |
| Singular rest shape | Large `Dm_inv`, unstable Hessian, solver spikes | Determinant, area, volume, and normal validity gates |
| Wrong replacement energy | Locked pair skips CCD but uses soft SNH/prototype energy | ABD-style high-kappa oracle and no-penetration scene gate |
| Release beta pop | Released pair behaves like a new contact | Beta carry fixture and scene report |
| False speed claim | Total frame time hides moved cost | Benchmark protocol requires subsystem timers |

## External References

- [RCC Adhesion specification](./specification/contact_models/rcc_adhesion.md)
- [Soft Vertex Triangle Stitch specification](./specification/constitutions/soft_vertex_triangle_stitch.md)
- [Affine Body specification](./specification/constitutions/affine_body.md)
- [CUDA backend development notes](./development/backend_cuda/index.md)
