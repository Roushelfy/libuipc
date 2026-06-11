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
| `IPCSimplexRCCAdhesiveContact` | RCC beta evolution and persistence for active/unlocked PT pairs (beta mode); candidate stream, lock eligibility, and occlusion in distance-lock mode | Dynamic tet Hessian assembly, filter-specific skip logic |
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
| RCC Phase B | Released pairs can receive carried beta (beta mode; vacuous in distance-lock mode) | Beta carry fixture |
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

Release is the normal exit path for a bonded approximation. It is evaluated from the live locked state, current or predicted positions, and scene policy. If any release condition fires, the pair must stop being reporter input for that step and must re-enter RCC persistence with its last locked beta. (Beta mode only: in distance-lock mode — "Distance-Locked Bonding Without Adhesion Energy" — beta does not exist, locked entries carry the sentinel `1.0`, and the carry is vacuous; the release gates themselves are identical.)

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

## Distance-Locked Bonding Without Adhesion Energy

Status: implemented 2026-06-11 (feasibility verified against source 2026-06-10; journal entries "Distance-Locked Bonding Feasibility" and "Phase 7 Implementation"). All five implementation-order steps below are landed and gate-backed: band oracle `[rcc_bonded_pt][oracle][distance_lock]` 18/2, scene gate `[rcc_bonded_pt][scene][distance_lock]` 604/1, counters contract `[rcc_bonded_pt][state][counters]` 28/1, beta-mode regressions green. This is an alternative bonded mode for scenes that want the virtual-tet bond/release machinery without any soft adhesion force: the adhesion energy is not assembled at all, beta does not exist, and the lock gate is purely geometric — a VT pair locks when its end-of-step distance is within `ξ + c·d_hat` for a user-set fraction `c ∈ [0,1]` (`ξ` = per-pair thickness; `ξ = 0` is the common case and reduces to `d < c·d_hat`). Release gates are byte-for-byte the ones the beta mode uses today. The beta-carry contracts below ("Release And Beta Carry Contract", roadmap Core Principle rule 5) are **beta-mode contracts**: in this mode beta does not exist, the carry is vacuous, and locked entries carry the sentinel `beta = 1.0` purely for format compatibility.

### Why

The soft adhesion energy and the bonded virtual tet solve different problems: the energy models gradual bonding/debonding (beta dynamics), while the bond replaces a stable contact with one stiff ABD complement energy so CCD/contact/RCC can skip it. Some workloads only want the second half — "grab whatever touches, hold it rigidly, release on the existing geometric/force gates" — and for them beta is pure overhead and an extra tuning surface: locking requires beta to integrate up to `rcc_bonded_pt_beta_lock_threshold` over several steps, and beta's Cn/eta/bonding-rate parameters have no meaning when no adhesion force is wanted. A distance criterion (`d < ξ + c·d_hat`) is the natural beta-free lock signal: the DCD activity test already classifies candidates by exactly this kind of band.

### Why not the obvious shortcuts (verified against source)

- `adhesion_enabled = 0` per pair zeroes the energy but also pins beta to 0 in Phase B init and Phase A evolution (`ipc_simplex_rcc_adhesive_contact.cu:477-481, 632-636, 1149-1150`), so the beta lock can never fire — and the policy release gate (`rcc_bonded_pt_rcc_policy_enabled`, `rcc_bonded_pt_system.cu:140-147, 300-305`) force-releases existing locks on disabled pairs. Disabling adhesion this way disables bonding too.
- `Cn = 0` zeroes the assembled energy but breaks beta evolution: `PT_beta_evolve_existing` divides by `denom = eta * W_scale / 10` with `W_scale ∝ Cn` (`codim_ipc_simplex_rcc_adhesive_function.h:57, 91-97`), a 0/0 hazard. There is no existing configuration that yields zero adhesion force while bonds still lock.

### Target model

1. **The lock driver stays Phase A of `IPCSimplexRCCAdhesiveContact`.** It already owns everything the lock needs: the `friction_VTs()` candidate stream, end-of-step positions, per-vertex `d_hats()` (fetched in the same function; `thicknesses()` is exposed by the same `GlobalVertexManager` handle and is a new fetch), contact-element ids, the adhesive tabular, the release context, and the only per-step call to `lock_from_rcc_pt_snapshot`. A parallel beta-free driver would duplicate that ownership (one-owner boundary) and re-plumb the release context for no gain. Consequences: `RCCAdhesive::apply_to` is still required in this mode — it provides the contact-model tabular (`adhesion_enabled`, per-pair `bonded_*` release/lock overrides, sticky-side machinery); `Cn`/`Ct` simply stay 0 and are never read, because beta kernels do not run. And, as for RCC adhesion today, the candidate stream exists only while `contact/friction/enable` is on (the default): `record_friction_candidates` is gated on it (`advance_ipc.cu:57-63`), and the adhesive reporter is a `SimplexFrictionalContact` subclass whose build requires friction anyway.
2. **Mode switch `rcc_bonded_pt_distance_lock`** (IndexT, default 0). When 1: adhesion energy/gradient/Hessian are not assembled — the reporter reports zero pair counts via the existing `friction_pair_counts` hook **and** gates its `do_compute_energy`/`do_assemble` kernel bodies on the mode (the two must change together: the kernels iterate `friction_VTs().size()` but write into subviews sized by the reported counts, so zeroing the counts alone would write out of bounds). Phase B beta init and Phase A beta evolution are skipped together (Phase A's evolve kernel reads `m_beta_PT`, which Phase B sizes), the released-beta carry merge is skipped (its runtime consumer, Phase B matching, no longer runs), and the lock-eligibility kernel switches from beta masking to the distance gate. What keeps running in Phase A: the lagged vertex-normal recompute and the release-context wiring (sticky signs, masks, tabular) — they live outside the beta kernels and feed the sticky-side/policy release gates — plus the occlusion pass (below). `rcc_bonded_pt_enabled` remains the master switch for the bonded machinery itself.
3. **Lock predicate: `D < (ξ + c·d_hat)²`** evaluated in the existing Phase A lock-eligibility kernel (`ipc_simplex_rcc_adhesive_contact.cu:1273-1320`), which already captures the end-of-step positions buffer (and loads all 4 vertex positions whenever the face-interior gate is on); `c = rcc_bonded_pt_distance_lock_ratio`, clamped to `[0,1]` on read (the `rcc_adhesion_normal_offset_coeff` pattern). `D` is the true closest-feature squared distance via `distance::point_triangle_distance2(flag, ...)` with the flag **recomputed from end-of-step positions** (`point_triangle_distance_flag` is a pure function of the 4 positions) — the lock is a fresh decision about end-of-step geometry, so it must not inherit the lagged begin-of-step `ActiveVT.flag`. `ξ = PT_thickness(...)` (sum of the two half-thicknesses) and `d_hat = PT_d_hat(...)` reduce the per-vertex buffers. The thickness offset is required, not optional: DCD only emits candidates in the band `ξ < d < ξ + d_hat` (`D_range`, strict on both ends), so a bare `d < c·d_hat` gate could never fire for pairs with `ξ ≥ c·d_hat`; with `ξ = 0` (the common cloth/tape case) the predicate reduces exactly to the user-facing `d < c·d_hat`. `c = 0` gives `d < ξ`, which IPC never allows — i.e. `c = 0` cleanly disables distance locking, mirroring the `normal_offset_coeff` `c = 0` special case.
4. **Indicator lock-beta keeps the downstream untouched.** The eligibility kernel emits lock-beta `1.0` (all gates pass) or `0.0`, exactly as the face-interior mask does today (`lockbeta(i) = eligible ? beta(i) : 0` becomes `lockbeta(i) = eligible ? 1.0 : 0.0`). The bonded producer's select (`entry.beta >= thr`), rest-shape validation, dedup, age tracking, same-step relock suppression, and the entire release path run unmodified. Threshold semantics in distance mode: the driver passes `min(rcc_bonded_pt_beta_lock_threshold, 1.0)` as the global threshold (a global value `> 1` is a beta-mode setting and must not silently veto every indicator lock), while a per-pair `bonded_lock_threshold > 1` remains a deliberate per-pair lock veto; per-pair values in `(0, 1]` keep their pass/fail meaning. Locked entries therefore carry `beta = 1.0` as a sentinel through `locked_beta()` / `released_beta()` / dump / seed, keeping the bridge and accessor formats unchanged. Seeding a beta-mode dump (`seed_locks` with betas `< 1.0`) into a distance-lock scene works unchanged — `seed_locks` re-thresholds against its own threshold argument.
5. **All other lock eligibility gates compose unchanged.** The face-interior mask (`rcc_bonded_pt_lock_face_interior_only` / `_margin`) lives in the same kernel and ANDs with the distance gate; distance-mode scenes should run with `rcc_bonded_pt_lock_face_interior_only = 1` — without beta's multi-step integration, the default `0` mass-locks edge/corner VTs into skewed sliver tets on first contact (exactly the 2026-06-03 failure). The cross-layer occlusion gate must keep functioning — it is what prevents a multi-layer tape from bonding through an intermediate layer. Today the occlusion test is a block **inside** the Phase B beta-init kernels (pinning beta to 0); in distance mode the same segment-cast (shared helper `VT_occlusion_blocked`, also rewired into the Phase B kernels) is fused into the eligibility kernel and evaluated at **end-of-step** positions with the freshly recomputed vertex normals — not Phase B's frame-open snapshot. That timing shift is deliberate: the lock is a fresh decision about end-of-step geometry (same rationale as the flag recompute), it lets a pair that became clear during the step lock one frame earlier, and it correctly rejects a pair that became occluded during the step (which beta mode would have locked on the stale frame-open test). `adhesion_enabled` keeps meaning "this contact pair may participate in RCC bonding": the eligibility kernel checks `PT_rcc_coeff(...).enabled` explicitly (today the check is implicit via beta pinning), which stays consistent with the unchanged policy release gate. Rejections from the explicit `enabled` check count toward `rcc_bonded_pt_rejected_policy_count` (the same gate the policy row of the Lock Gate table names); occlusion and distance rejections count toward `rcc_bonded_pt_rejected_distance_count`.
6. **Release gates: literally unchanged.** `release_flags_from_current_shape` (strain/gap/slip/force/flip/degenerate/sticky-side/policy) reads no beta anywhere (`rcc_bonded_pt_system.cu:157-362`), so the user requirement "release conditions stay as today" is satisfied structurally, not by re-implementation.
7. **No frontend/pybind changes.** Both keys are plain scene-config entries flowing through the existing `Scene(Json)` path; Python users set `config["rcc_bonded_pt_distance_lock"] = 1` like every other `rcc_bonded_pt_*` key. Setting `Cn`/`Ct > 0` together with `rcc_bonded_pt_distance_lock = 1` is a contradiction (the mode's defining semantic is "no adhesion force"); the build reports a warning and ignores the adhesion coefficients.

### Behavioral differences vs the beta lock (accepted, documented)

- **No temporal hysteresis.** Beta integrates contact history; the distance gate locks on the first end-of-step inside the band. The planned `rcc_bonded_pt_min_lock_age` gate composes naturally when it lands. Until then, choose `c` conservatively, and set the gap release band wider than the lock band — `rcc_bonded_pt_release_gap` is a *growth* threshold relative to the lock-time gap, so `release_gap > c·d_hat` is the conservative sufficient condition — or a pair grazing the lock boundary will churn lock/release every step (each relock rebuilds the rest shape, so churn also means rest-shape drift under sliding).
- **No load-based lock suppression.** In beta mode, tension drives beta down and a force-released bond does not immediately relock. In distance mode a force-released pair still inside the band relocks next step; if peel-off scenarios matter, the force/gap release thresholds carry that burden alone.
- **One-frame rest-shape re-snapshot on fresh locks.** A pair locked at the end of frame `k` is usually still in frame `k+1`'s lagged candidate list (the filter's locked-key compact only affects later detects), so the unconditional indicator relocks it once at the end of `k+1` with a rest shape rebuilt from `k+1` positions — effectively re-referencing the gap-release baseline one frame after locking. Shared with beta mode (where beta stays above threshold and relocks the same way), so not a regression; under fast pressing it slightly weakens "growth from lock-time gap" semantics.
- **Frame-1 locking.** `friction_VTs` is seeded by the frame-1 initial DCD (`advance_ipc.cu:312-315`), so distance locks can already form at the end of frame 1 — useful for asset-loaded scenes that start in contact (the `seed_locks` accessor remains available for exact lock restoration).
- **State accessors.** `RCCBondedPTStateAccessorFeature` works unchanged (sentinel beta 1.0). `RCCAdhesionStateAccessorFeature` (soft beta dump/load) has no state to carry in this mode and reports zero soft pairs; loading soft beta state into a distance-lock scene is a documented no-op.

### Implementation order

| Step | Change | Independently verifiable by |
| --- | --- | --- |
| 1 | Config keys `rcc_bonded_pt_distance_lock` (IndexT 0) + `rcc_bonded_pt_distance_lock_ratio` (Float 0.5, clamped `[0,1]` on read), their read path in `RCCBetaEvolutionTimeIntegrator::do_build`, and a build warning when the mode is on while any tabular `Cn`/`Ct > 0` (contradictory config: adhesion coefficients are ignored) | Config read/clamp unit check; warning observable |
| 2 | Distance-mode eligibility kernel: end-of-step flag recompute + flagged `D` + `D < (ξ + c·d_hat)²` + `adhesion_enabled` + occlusion + face-interior compose; indicator lock-beta; global threshold passed as `min(beta_lock_threshold, 1.0)` | CPU oracle over crafted candidates (inside/outside band, blocked, disabled, face-exterior) vs kernel mask |
| 3 | Energy/beta bypass, atomically: zero `friction_pair_counts` AND gate the `do_compute_energy`/`do_assemble` kernel bodies (the kernels write into subviews sized by the reported counts — counts and kernels must change together or the writes go out of bounds); skip Phase B init + Phase A evolution together (the evolve kernel reads `m_beta_PT`, which Phase B sizes) and the beta carry; extract the occlusion block into a beta-free pass; keep the vertex-normal recompute and release-context wiring | Adhesion energy contributes exactly 0; no NaN; bonded locks still form; occlusion still blocks cross-layer pairs |
| 4 | Scene gate: distance-lock variant of `pt_lift_release` (press/hold/lift; bonds form by distance with `Cn = Ct = 0` and `rcc_bonded_pt_lock_face_interior_only = 1`, carry the cube, release on forced pull; penetration-free) | `[rcc_bonded_pt][scene][distance_lock]` sim gate |
| 5 | Counters + docs: `rcc_bonded_pt_rejected_distance_count` (distance/occlusion rejections; `enabled` rejections go to `rejected_policy_count`), snapshot/lock-gate doc rows flip from Planned to Implemented | Counter visible in `RCCBondedPTCounters`; doc gate green |

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
