# RCC Adhesion Acceleration

This is the focused subsystem note for replacing stable RCC point-triangle adhesion pairs with bonded virtual tetrahedra. The roadmap says what is current; this page says what the subsystem is supposed to do.

## Scope

In scope:

- RCC adhesive vertex-triangle (VT) primitives. The first production milestone locks face-interior point-triangle pairs; Phase 6 extends adhesion and locking to the full VT primitive across all closest-feature classifications (PP/PE/PT) — see "Full-Feature Adhesion And Per-Primitive Beta" below.
- Stable pairs whose beta and motion history make contact topology unlikely to change during the next step.
- CUDA backend trajectory filtering, RCC beta persistence, and complement-energy assembly.
- Dynamic runtime state owned by the backend.

Out of scope for the first production milestone:

- EE (edge-edge) adhesion and locking. PP/PE adhesion is added in Phase 6; EE is a later follow-up.
- Vertex-half-plane adhesion locking.
- Frontend `SoftVertexTriangleStitch` geometry rebuilds for transient pairs.
- Default-on behavior before correctness and benchmark gates pass.

## Algorithm Summary

```text
For each end-of-step RCC PT pair:
  compute stable membership key
  evaluate lock gate
  if lock gate passes:
    build oriented virtual-tet topology
    build conditioned rest shape using SVTS-compatible logic
    store beta, age, Dm_inv, rest volume

For each next-step PT candidate in simplex filters:
  compute membership key
  if key is locked:
    skip PT CCD broadphase and active-pair emission
  else:
    keep current contact pipeline behavior

Before bonded assembly:
  evaluate release gate before final bonded reporter ownership
  record release reason
  carry beta back to RCC persistence
  remove released pairs from bonded reporter input for that step

During assembly:
  contact/RCC sees only unlocked pairs
  bonded reporter assembles one high-kappa ABD-style complement energy per still-locked topology
```

## Current Implementation Snapshot

This table is the short handoff surface. It intentionally separates lock, release, filtering, energy, and scene proof so that one completed layer is not mistaken for another.

| Area | Implemented And Gate-Backed | Still Required Before Production Claims |
| --- | --- | --- |
| Lock/classification | Feature switch, beta threshold, duplicate suppression, SVTS-compatible rest-shape construction, degenerate fresh-lock rejection, active-lock carry/age increment | Minimum-age gate, sticky-side lock gate, normal-gap lock band, tangential-slip lock band, contact/subscene policy lock gate, and their rejection counters |
| Release/fallback | Backend CUDA release for strain, normal gap, tangential slip, sticky-side failure, disabled policy, flip, and degenerate current shape; same-step relock suppression; released beta carry | Scene-accessible released topology, age, flags, and per-reason release counters |
| Filter ownership | Shared locked-key lookup, common active/friction PT compact, and (default-off) pre-CCD skip in all four simplex filter broadphase predicates | Enable the pre-CCD skip by default after the no-penetration scene gate; duplicate-ownership diagnostics |
| Bonded energy | ABD OrthoPotential reporter over `F = Ds Dm_inv`, `rcc_bonded_pt_kappa`, CPU E/G/H oracle, CUDA-vs-CPU reporter oracle | Scene no-penetration observation while CCD/contact/RCC is skipped |
| Scene/benchmark | Legacy RCC cube-cube and cube-cloth lift/release baselines; bonded-mode `pt_lift_release` press/hold/lift no-penetration gate with `rcc_bonded_pt_skip_ccd` on | Forced-pull release/separation and adhesion-off baseline in the scene gate, and benchmark timers |

Release context is not a lock gate. Sticky-side signs, normals, contact masks, subscene masks, and adhesive enable flags are currently routed into the bonded owner for release decisions; the live producer still needs explicit lock-side use of those inputs before scene correctness claims.

## Full-Feature Adhesion And Per-Primitive Beta

The bonded acceleration today only covers face-interior point-triangle contacts because RCC adhesion itself is PT-only: `EE_*`/`PE_*`/`PP_*` adhesion returned zero, beta lived only on PT-classified pairs (`m_beta_PE`/`m_beta_PP` zero-filled), and there was no contact-area weight. A stable edge/corner contact therefore gets neither adhesion nor a bond. Phase 6 rebuilds adhesion on the XBow `RCCAdhesionEnergy3D` model so the bonded lock can cover the full vertex-triangle (VT) primitive. The architecture page is the design of record; this is the subsystem-level checklist and status.

Principles (see [architecture](./architecture.md) "Full-Feature Adhesion And Per-Primitive Beta"):

- The VT pair is the primitive; the closest-feature classification (PP/PE/PT) only chooses the distance sub-formula and tangent basis for that step.
- Beta is per primitive, evolved from the **true closest-feature distance**, carried across PP/PE/PT transitions.
- PE/PP (and later EE) adhesion uses the true `point_edge_distance2` / `point_point_distance2` and matching friction basis, mirroring the `PT_*` functions.
- Contact area is currently **lumped into Cn/Ct** (libuipc IPC convention, matching barrier `kappa`; spec `rcc_adhesion.md:176-177`), not a separate per-pair factor — the assembled energy has no `A_k` term. An explicit per-vertex `A_k` (XBow style, where Cn/Ct stay per-unit-area densities) is optional and only for resolution-independent / non-uniform-mesh adhesion; adding it requires reinterpreting Cn/Ct as densities, else it double-counts.
- Distance handling is structural single-compute, NOT a reused buffer: the barrier recomputes `d^2` in-kernel and stores nothing, and a per-pair `d^2`/derivative buffer would be a GPU regression. Compute `d^2`/basis once per pair in one kernel body (intra-RCC first, then optionally merge with the friction kernel — same lagged `friction_PTs()`, bit-identical lagged basis). Only `db_dd2` and the flagged `d^2` are barrier-shareable; RCC's unflagged plane normal `d^2` is intrinsically different. The IPC barrier keeps the true closest-feature distance for non-penetration.
- The bonded lock decides on the VT primitive; the ABD virtual-tet energy stays point-plane (`F = Ds Dm_inv`).

| Step | Change | Status |
| --- | --- | --- |
| 1 | PE/PP true-feature adhesion formulas replace the `return 0` stubs (point-edge / point-point distance + friction basis, mirroring `PT_*`) | Implemented; behaviour-neutral while `m_beta_PE/PP == 0` (call sites early-out on `beta <= 0`) |
| 1-oracle | PE/PP E/G/H finite-difference oracle | Planned before Step 2 makes the formulas load-bearing |
| 2 | Per-primitive beta evolved from the true closest-feature distance; `m_beta_PE/PP` no longer zero-filled | Planned |
| 3 | (Optional, lowest priority) per-vertex `A_k` weight — area is currently lumped into Cn/Ct; adding A_k needs Cn/Ct reinterpreted as densities + barrier boundary-area reuse, else double-count | Optional |
| 4 | Distance single-compute in one kernel body (intra-RCC, then optional friction-kernel merge); reject a per-pair `d^2` scratch buffer (GPU regression) | Planned |
| 5 | Bonded lock on the VT primitive; corner/edge contacts bond; ABD tet stays point-plane | Planned |

## Distance-Locked Bonding Without Adhesion Energy

A planned alternative bonded mode: no soft adhesion energy at all (and therefore no beta), while the bonded virtual-tet lock/release machinery keeps running. The lock gate becomes purely geometric — a VT pair locks when its end-of-step true closest-feature distance satisfies `d < ξ + c·d_hat` with user coefficient `c ∈ [0,1]` (`ξ` = per-pair thickness; `ξ = 0` reduces to the user-facing `d < c·d_hat`). Release gates are unchanged — they never read beta. The [architecture](./architecture.md) section "Distance-Locked Bonding Without Adhesion Energy" is the design of record; this is the subsystem-level checklist and status.

Principles:

- The lock driver stays Phase A of the adhesive reporter (`RCCAdhesive::apply_to` still required; `Cn`/`Ct` stay 0 and are never read); a parallel beta-free driver would duplicate candidate/release-context ownership. As for RCC adhesion today, the candidate stream requires `contact/friction/enable` on (the default).
- Existing shortcuts do not work: `adhesion_enabled = 0` pins beta to 0 and policy-releases locks; `Cn = 0` is a 0/0 hazard in beta evolution. Hence a dedicated mode switch (which also warns if `Cn`/`Ct > 0` is set alongside it).
- The distance gate runs in the existing Phase A lock-eligibility kernel (which already captures the end-of-step positions buffer) and emits an indicator lock-beta (1.0/0.0), so the bonded producer's select, rest-shape conditioning, age, dedup, and the whole release path run unmodified; the global lock threshold is passed as `min(value, 1.0)` so a beta-mode setting cannot silently veto indicator locks, while a per-pair `bonded_lock_threshold > 1` remains a deliberate veto.
- The flag for the closest-feature distance is recomputed from end-of-step positions (pure function), not taken from the lagged `ActiveVT.flag`.
- Face-interior mask, cross-layer occlusion, and `adhesion_enabled` eligibility compose with the distance gate; the occlusion test is extracted from the Phase B beta-init kernels into a beta-free pass, and distance-mode scenes should set `rcc_bonded_pt_lock_face_interior_only = 1` (without beta's multi-step integration, the default `0` mass-locks edge/corner sliver tets on first contact — the 2026-06-03 failure mode).
- Energy bypass is atomic: zero `friction_pair_counts` AND gate the energy/assemble kernel bodies together (the kernels write into subviews sized by the reported counts); Phase B init and Phase A evolution are skipped together.
- No temporal hysteresis: set the gap release band wider than the lock band — `rcc_bonded_pt_release_gap` is a growth threshold relative to the lock-time gap, so `release_gap > c·d_hat` is the conservative sufficient condition — to avoid lock/release churn; the planned min-age gate composes when it lands.

| Step | Change | Status |
| --- | --- | --- |
| 1 | Config keys `rcc_bonded_pt_distance_lock` + `rcc_bonded_pt_distance_lock_ratio` (clamped `[0,1]`) + contradictory-config warning (adhesion-enabled row with `Cn`/`Ct > 0` while the mode is on) | Implemented |
| 2 | Distance-mode eligibility kernel (end-of-step flag + flagged `D` + `D < (ξ + c·d_hat)²` via shared `VT_distance_lock_band_pass` + enabled/occlusion/sticky/face-interior compose; indicator lock-beta; global threshold clamped `<= 1` in BOTH the producer-call scalar and the per-pair sentinel resolution) | Implemented; CPU oracle `[rcc_bonded_pt][oracle][distance_lock]` 18/2 green |
| 3 | Energy/beta bypass (zero `friction_pair_counts` + gate energy/assemble kernels atomically; skip Phase B init + Phase A evolution together + beta carry; occlusion cast fused into the eligibility kernel via shared `VT_occlusion_blocked`, end-of-step positions; keep vertex-normal recompute and release-context wiring) | Implemented |
| 4 | Scene gate: distance-lock `pt_lift_release` variant with `Cn = Ct = 0`, `rcc_bonded_pt_lock_face_interior_only = 1`, `c = 0.95` (press equilibrium gap ~0.89·d_hat must sit inside the band) | Implemented; `[rcc_bonded_pt][scene][distance_lock]` 604/1 green |
| 5 | `rcc_bonded_pt_rejected_distance_count` counter (distance/occlusion rejections; explicit `enabled` rejections go to `rcc_bonded_pt_rejected_policy_count`) + doc status flip | Implemented; counters in the pybind dict and the `[rcc_bonded_pt][state][counters]` contract fixture |

## Pair Keys

Use two representations:

| Representation | Contents | Use |
| --- | --- | --- |
| Membership key | Point id kept distinct, triangle global vertex ids sorted, then hashed exactly like RCC PT beta persistence | Filter lookup and beta matching |
| Oriented topology | `(point, tri0, tri1, tri2)` global vertex ids | Virtual-tet rest shape and energy assembly |

The membership key can answer "is this four-vertex set locked?" It cannot answer "what oriented tet should be assembled?" Therefore every lock stores both.

## Lifecycle State Machine

The bonded PT lifecycle has four observable states. Tests and reports should be able to tell which state a pair occupies.

| State | Meaning | Required Ownership |
| --- | --- | --- |
| Candidate | Current RCC PT pair is considered for lock | Still owned by contact/RCC until all lock gates pass |
| Locked | Pair is stable enough to skip CCD/contact/RCC | Owned by exactly one ABD-style bonded reporter input |
| Released | A release gate fired for a previously locked pair | Removed from bonded reporter input and returned to RCC beta persistence (beta mode; vacuous in distance-lock mode) |
| Rejected | Candidate failed lock gates or rest-shape conditioning | Remains in normal contact/RCC path |

A pair must not be both `Locked` and active in `PTs()` / `friction_PTs()` for the same Newton iteration. A pair must not be `Released` and still assembled by the bonded reporter in the same step.

## Lock Gate

A candidate can lock only if every condition passes. The target policy and the current implementation status are deliberately shown together because release-side coverage does not imply lock-side coverage.

| Gate | Data Needed | Rejection Counter | Current Status |
| --- | --- | --- | --- |
| Feature enabled | Config | `rcc_bonded_pt_disabled_count` | Implemented as master switch; disabled-count reporting planned |
| PT adhesion scope | Current RCC PT pair | `rcc_bonded_pt_rejected_type_count` | Implemented implicitly by consuming RCC PT snapshots only |
| Beta threshold | `beta >= rcc_bonded_pt_beta_lock_threshold` | `rcc_bonded_pt_rejected_beta_count` | Implemented; low-beta rejected count planned |
| Minimum age | `locked_age` or candidate age | `rcc_bonded_pt_rejected_age_count` | Planned |
| Sticky-side consistency | Lagged normals and `rcc_sticky_sign` | `rcc_bonded_pt_rejected_sticky_count` | Planned for lock; implemented for release only |
| Normal gap band | End-of-step positions | `rcc_bonded_pt_rejected_gap_count` | Planned for lock; implemented for release only |
| Tangential slip band | Lagged closest coordinates/basis | `rcc_bonded_pt_rejected_slip_count` | Planned for lock; implemented for release only |
| Rest-shape quality | Triangle area, normal, `det(Dm)`, rest volume | `rcc_bonded_pt_rejected_degenerate_count` | Implemented as degenerate fresh-lock rejection |
| Contact policy | Contact tabular enable/disable | `rcc_bonded_pt_rejected_policy_count` | Implemented for release; implemented for lock in distance-lock mode (explicit `adhesion_enabled` check + counter); still planned for the beta-mode lock (acts via beta pinning, uncounted) |
| Distance band (distance-lock mode) | End-of-step positions, per-vertex `d_hats`/thicknesses: `D < (ξ + c·d_hat)²` | `rcc_bonded_pt_rejected_distance_count` | Implemented — supersedes the beta threshold when `rcc_bonded_pt_distance_lock` is on (the beta select is trivially satisfied via an indicator lock-beta; see "Distance-Locked Bonding Without Adhesion Energy") |

The conservative rule is: reject on missing data. A missing sticky-side normal, invalid triangle normal, or unavailable beta carry is not a reason to guess.

`rcc_bonded_pt_min_lock_age` is not yet a live config key, so a freshly formed PT pair can lock on its first step where beta crosses the threshold. "Likely to stay bonded" — the user-facing justification for skipping CCD/contact — currently means only `beta >= rcc_bonded_pt_beta_lock_threshold`; the age/sticky/gap/slip/policy lock gates that make that judgment real are still planned (release-side coverage does not imply lock-side coverage).

## Rest Shape

The rest-shape construction follows the existing `SoftVertexTriangleStitch` convention:

1. Use rest positions for point `x0` and triangle vertices `x1`, `x2`, `x3`.
2. Compute triangle normal and signed point-plane distance.
3. If the distance magnitude is below `rcc_bonded_pt_min_separate_distance`, offset the rest point along the triangle normal to create finite thickness.
4. Build `Dm = [x1 - x0, x2 - x0, x3 - x0]`.
5. Reject if triangle area, `abs(det(Dm))`, or rest volume is below threshold.
6. Store `Dm_inv` and positive `rest_volume`.

`min_separate_distance` is a numerical thickness, not a cosmetic gap. It must scale with scene/mesh units and should not be set arbitrarily close to zero.

## Energy Model

The runtime bonded energy is not the same thing as `SoftVertexTriangleStitch` energy. SVTS supplies the PT rest-shape convention above; production bonded PT acceleration must use an ABD-style high-stiffness shape energy.

Hard production contract:

- If a locked PT pair skips CCD/contact/RCC in a step, exactly one ABD-style virtual-tet energy over the same four real vertex DOFs must be assembled in that step.
- The production/default model is `abd_ortho` with `rcc_bonded_pt_kappa >= 1e8` for the first correctness gates.
- A zero-stiffness, missing-reporter, retired SNH, or unsupported-energy path is not allowed to claim non-penetration or performance.
- `SoftVertexTriangleStitch` is not a runtime energy substitute; it is only the rest-shape/thickness convention used to build `Dm_inv` and `rest_volume`.

For one locked topology `(x0, x1, x2, x3)`:

$$
D_s = [x_1 - x_0, x_2 - x_0, x_3 - x_0], \qquad F = D_s D_m^{-1}.
$$

`F` is treated as the affine transform of a virtual tetrahedron whose DOFs are still the four real vertex positions. The target default model is ABD OrthoPotential:

$$
E_{\text{abd\_ortho}} = \kappa \, V_0 \, \Delta t^2 \, \|F F^T - I\|_F^2.
$$

`V0` is `rest_volume`, and `kappa` comes from `rcc_bonded_pt_kappa`. The first production gates should use `kappa >= 1e8` in scene units. `abd_arap` may be added as an alternate model only after matching CPU and GPU oracle coverage:

$$
E_{\text{abd\_arap}} = \kappa \, V_0 \, \Delta t^2 \, \|F - R\|_F^2,
$$

where `R` is the rotation from the polar decomposition of `F`.

The reporter assembles this ABD-style energy directly into the 12 vertex-position DOFs through `dF/dx`. It does not create transient frontend ABD objects or extra affine DOFs. The earlier Stable Neo-Hookean reporter/oracle was a prototype path only and has been replaced for production bonded PT assembly.

## Release Gate

A locked pair releases when any condition fails.

| Release Reason | Condition | Required Follow-Up |
| --- | --- | --- |
| `strain` | Stretch, shear, or volume distortion exceeds threshold | Return to RCC/contact next detection |
| `gap` | Predicted normal gap exceeds release distance | Carry beta and emit release counter |
| `slip` | Tangential slip exceeds release distance | Carry beta and let RCC tangential adhesion handle sliding |
| `flip` | Triangle orientation or virtual tet quality becomes invalid | Release before assembly if detected early |
| `force` | Bond restoring force (`~ kappa * deformation`) exceeds threshold | Carry beta; the criterion that peels a stiff bond on a compliant counterpart |
| `sticky_side` | Sticky-side gate no longer passes | Return to normal contact/RCC policy |
| `policy` | Contact tabular or scene policy disables the pair | Remove lock and report policy release |

Release is not failure. It is the intended fallback when the bonded approximation stops matching contact-like adhesion.

Current implementation status:

| Reason | Status |
| --- | --- |
| `strain` | Implemented on CUDA using `rcc_bonded_pt_release_strain` against `||F F^T - I||` |
| `gap` | Implemented on CUDA using `rcc_bonded_pt_release_gap` against current normal distance growth from the lock-time rest gap reconstructed from `Dm_inv` |
| `slip` | Implemented on CUDA using `rcc_bonded_pt_release_slip` against current closest-foot tangential displacement from the lock-time rest barycentric foot reconstructed from `Dm_inv` |
| `sticky_side` | Implemented on CUDA by routing RCC sticky signs and lagged vertex normals into the bonded owner and reusing the RCC sticky-side gate semantics |
| `policy` | Implemented on CUDA by routing RCC adhesive enable flags plus contact/subscene masks into the bonded owner |
| `force` | Implemented on CUDA using `rcc_bonded_pt_release_force` against the F-space restoring force `4 * kappa * V0 * dt^2 * \|\|C F\|\|` (`C = F F^T - I`). Unlike strain/gap, it is scaled by `kappa`, so it fires on a holding stiff bond and peels compliant-counterpart fixtures (cube-cloth corner pull separates at `release_force=1e-4`, `kappa=5e7`, while holding through press/hold/lift) |
| `flip` / `degenerate` | Implemented on CUDA from current virtual-tet determinant/topology/rest-volume validity |

Release ordering matters:

1. Evaluate release gates against active locks before the bonded reporter consumes its input for the current step.
2. Mark release flags and produce released key/beta/topology/age snapshots.
3. Compact active locked buffers so the released pair is absent from ABD reporter assembly.
4. Merge released beta into RCC PT persistence before the pair is treated as a fresh contact candidate.
5. Count release once; repeated accessor queries or repeated filter-sync calls must not double-count it.

### Beta While Locked

While a pair is locked it is removed from `friction_PTs()`, so Phase A does not evolve its beta: the lock-time beta is frozen and carried until release. This replaces the spec's energy-driven debonding law (beta evolution from accumulated normal/tangential adhesion energy and pressure, see the RCC adhesion spec) with the geometric release gates (strain/gap/slip) plus sticky-side/policy. This is a deliberate approximation: a pair the RCC energy criterion would gradually debond stays rigidly bonded until a geometric threshold trips. Before any correctness claim the geometric release thresholds must be calibrated so the locked-then-released trajectory matches the unaccelerated beta-evolution debond timing within tolerance on a canonical purely-normal and purely-tangential example (conventions Test Matrix `[rcc_bonded_pt][calibration][debond]`).

Known blind spot (2026-06-02 cross-fixture probe, see journal): geometric release (strain/gap) can be self-defeating. A stiff bonded tet (`kappa` 5e7-1e8) absorbs the imposed motion, so the per-tet strain/gap never exceeds threshold when the counterpart is compliant (e.g. FEM cloth bonded to an ABD cube) — the bond holds and the soft side just follows it, so the pair never releases (the cloth-peel and cube-cloth fixtures did not separate even at `release_strain=0.1, release_gap=0.01`). The all-ABD cube-cube case separates only because two stiff actuators force enough gap/strain across the bond.

This is fundamental, not a tuning issue: at fixed `kappa=5e7` with a corner-peel pull, lowering `release_gap` all the way to `0.0002` (0.2 mm) released only 1-2 of ~87 bonds and the cloth still rode up with the cube — and below that the bonds release from solver jitter during hold. The reason is that geometric release measures how far a bond has already *deformed/failed*, but a bond that is doing its job keeps `curr_dist ~ rest_dist` (gap ~ 0) and `F ~ I` (strain ~ 0) by construction, so no threshold above numerical jitter distinguishes "holding well" from "should release"; the compliant cloth also absorbs the pull in its free region before it reaches the bonded edge. A robust release law on compliant counterparts therefore needs a *force/energy* trigger (release when the bond's restoring force, proportional to `kappa * deformation`, exceeds a limit — which fires for a holding stiff bond), the RCC beta criterion evaluated on locked pairs, a lower `kappa`, or a lock gate that excludes soft counterparts.

This force/energy trigger is now implemented as the `force` release reason (`rcc_bonded_pt_release_force`): at fixed `kappa=5e7` the cube-cloth corner pull separates at `release_force=1e-4` (all ~94 bonds peel, cloth drops from y~1.0 to ~0.4) while holding cleanly through press/hold/lift (zero premature release), where geometric release at any threshold could not (gap down to 0.2 mm released only 1-2 of ~87).

## Assembly Contract

The bonded virtual-tet reporter:

- Reports complement energy, not contact energy.
- Assembles energy, gradient, and Hessian over global vertex ids.
- Uses ABD-style high-kappa shape energy over `F = Ds Dm_inv`; default target model is `abd_ortho`.
- Scales by `rest_volume * dt^2`, matching the backend reporter convention.
- Applies SPD projection in the Hessian path.
- Emits report fields for assembled count and timing.

The reporter must not:

- Update beta.
- Run broadphase or CCD.
- Reinsert locked pairs into `PTs()` or `friction_PTs()`.
- Count as contact for adaptive contact-parameter strategies unless a separate diagnostic explicitly asks for that comparison.

## Implementation Targets

| Area | Target Files | Notes |
| --- | --- | --- |
| Host state owner | `include/uipc/core/rcc_bonded_pt_state.h`, `src/core/core/rcc_bonded_pt_state.cpp` | Minimum host contract, counters, release flags, `Dm_inv`, rest volume, and deterministic fixtures |
| CUDA state bridge | `src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.*` | Owns device buffers for locked keys, topologies, beta, age, release flags, `Dm_inv`, rest volume, and host counter snapshots; can be replaced from compact sorted device entries without host roundtrip while rest-shape payloads stay in SoA buffers |
| CUDA owner | `src/backends/cuda/contact_system/rcc_bonded_pt_system.*` | Owns the bridge at runtime, honors `rcc_bonded_pt_enabled`, feeds sorted locked keys to `SimplexTrajectoryFilter`, syncs common active-filter skip counters, compacts RCC Phase A high-beta PTs into locked state after live rest-shape construction and degeneracy rejection, and evaluates backend release reasons. Remaining live lock gates are still planned. |
| Filter helper | `src/backends/cuda/contact_system/rcc_bonded_pt_lookup.h` | Shared device helper for RCC PT key construction, sorted membership lower-bound, and lock lookup |
| Common active filter | `src/backends/cuda/collision_detection/simplex_trajectory_filter.*` | Can compact locked PTs out of active `PTs()` and `friction_PTs()` when supplied sorted locked keys; default is no-op |
| Filter backends | `src/backends/cuda/collision_detection/filters/*simplex_trajectory_filter.cu` | Reject locked PTs in the PT broadphase predicate via the shared `rcc_bonded_pt_candidate_is_locked` helper, gated by default-off `rcc_bonded_pt_skip_ccd` |
| Reporter | `src/backends/cuda/contact_system/rcc_bonded_pt_virtual_tet_reporter.*` | Dynamic ABD-style high-kappa complement reporter, no frontend geometry rebuild |
| RCC integration | `ipc_simplex_rcc_adhesive_contact.cu` | Phase A beta evolution now optionally calls the bonded-PT producer with current positions for lock-time rest-shape construction; sticky-side and policy inputs are routed as release context; released snapshots preserve key/topology/beta/age/flags inside the backend owner, and released key/beta pairs are merged back into RCC persistence through `RCCBondedPTBetaCarryScratch` |
| Tests | `apps/tests/core`, `apps/tests/backends/cuda`, `apps/tests/sim_case` | Follow the test matrix in conventions |
| Benchmarks | `scripts/bench_rcc_adhesion_acceleration.py` | Planned after timers/counters exist |

## Oracles

Required oracles before production use:

| Oracle | Input | Expected Output |
| --- | --- | --- |
| State oracle | Two PT pairs with deterministic beta/age/release flags and rest-shape payloads | Implemented by `uipc_test_core "[rcc_bonded_pt][state]"`: one lock stays active, one release is extracted, stable key/topology/beta/age/release/rest-shape permutation is preserved |
| Rest-shape oracle | Point near triangle plane with known `min_separate_distance` | Implemented by `uipc_test_core "[rcc_bonded_pt][oracle][rest_shape]"`: `Dm_inv`, positive rest volume, point offset, orientation swap, and degenerate-triangle rejection match SVTS rules |
| ABD-style production energy oracle | Single virtual tet with deterministic deformation and high stiffness | Implemented by `uipc_test_core "[rcc_bonded_pt][oracle][abd_energy]"`: proves `abd_ortho` E/G/H, finite differences, SPD projection, and `kappa >= 1e8` conditioning |
| CUDA state bridge oracle | Host state with pending and extracted release paths | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]"`: device buffers preserve key/topology/beta/age/release/rest-shape alignment and counters roundtrip through upload/download |
| CUDA lookup oracle | One locked key, one triangle permutation, two misses, and an empty locked set | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][lookup]"`: lookup uses the existing RCC PT key, treats triangle permutations as the same membership key, keeps the point id distinct, and misses cleanly |
| Common active-filter oracle | Synthetic active PT list with two locked and two unlocked pairs | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][filter]"`: locked pairs are compacted out of `SimplexTrajectoryFilter::PTs()` and stay absent after `record_friction_candidates()` copies to `friction_PTs()` |
| CUDA owner oracle | Host locked state plus synthetic filter active view | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][owner]"`: owner uploads state, feeds sorted keys, syncs filter-skip counters, and downloads counters aligned with active locks |
| Beta/rest producer oracle | Existing locks plus a RCC PT beta snapshot with one refresh, one carry, one new lock, one duplicate, one low-beta reject, and one degenerate high-beta reject | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][owner][producer]"`: device-side producer carries old locks and their rest-shape payloads, refreshes high-beta candidates, increments age, suppresses duplicate keys, builds SVTS-compatible `Dm_inv/rest_volume` for fresh locks, counts degenerate rejects, and leaves low-beta or degenerate candidates unlocked |
| Lock-gate parity oracle | High-beta PT candidates with controlled age, sticky-side, normal-gap, tangential-slip, rest-shape, and policy inputs | Planned: only fully valid candidates become locks, missing data rejects conservatively, and lock rejection counters do not reuse release counters |
| ABD-style reporter oracle | One locked pair with known rest shape and high stiffness | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][reporter][abd_oracle]"`: proves CUDA E/G/H matches the ABD CPU oracle |
| CUDA BVH/radix-sort regression | Bunny sanity mesh through the existing GPU sanity checker | Implemented by `uipc_test_backend_cuda "gpu_sanity_check" -c "bunny"` and `scripts/run_rcc_adhesion_acceleration_cuda_gates.py`: bonded-PT device layout changes must not destabilize `SimplicialSurfaceDistanceCheck` or `InfoStacklessBVH` |
| Release/beta carry oracle | Locked PTs with controlled current deformation/context: one stays locked while another releases by strain, normal gap, tangential slip, sticky-side failure, or disabled policy | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][release]"`: released lock is absent from bonded state, released key/topology/beta/age/flag snapshots stay aligned, release reason/counter is recorded once, same-step relock is suppressed, and released beta is merged into RCC persistence without overwriting newer duplicate beta |
| Pre-CCD filter membership oracle | Synthetic PT candidates resolved through `surf_vertices`/`surf_triangles` with one locked (permuted) key | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]"`: the shared broadphase membership helper rejects the locked candidate (orientation-invariant) and keeps unlocked ones; full scene candidate/TOI-absent integration still planned |
| No-penetration scene observation | PT-rich bonded-mode press/hold/lift with CCD skipped for locked pairs | Implemented by `uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"`: with `rcc_bonded_pt_skip_ccd` on, the cube fixture forms 8 bonded locks and the contact-face gap stays non-negative (min ~+0.019) through press/hold/lift while the lower cube is carried by the bonded energy. Forced-pull release/separation and adhesion-off baseline still planned |

## Scene Gate

The first bonded-mode real-scene gate is `pt_lift_release`: a PT-rich adhesion fixture promoted from visual demos to pass/fail assertions. The 2026-06-01 coarse cube-cube `rcc_adhesion_pick_and_lift` probe is not accepted as the seed because adhesion-on and adhesion-off runs produced matching bottom-cube heights even though RCC adhesion assembly was active. A point-dense variant works: a subdivided contact-face or subdivided tet cube can produce stable adhesion-on/off lift. Prefer a production fixture seeded from `rcc_adhesion_cloth_peel`, `python/examples/rcc_adhesive_oriented_cloth_demo.py`, or a point-dense variant of `python/examples/rcc_adhesive_pick_and_lift_demo.py`.

Current legacy RCC gates now cover the non-accelerated behavior that bonded PT must preserve:

| Gate | Command | What It Proves |
| --- | --- | --- |
| Python subdivided cube and cube-cloth fixtures | `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Existing RCC adhesion can lift during hold and release/separate during pull in PT-rich fixtures |
| Native C++ sim-case fixtures | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | The same cube-cube and cube-cloth behaviors are available as CUDA sim-case gates |

These gates are legacy behavior baselines, not bonded-PT proof. They do not observe locked keys, filter skips, duplicate ownership, release reason flags, ABD reporter ownership, or beta carry through bonded state.

| Phase | Scene Action | Required Observation |
| --- | --- | --- |
| Adhesion-off baseline | Disable RCC adhesive contact while keeping the driver, constraints, and gravity identical | The adhered body or patch does not follow the driver beyond the declared tolerance |
| Legacy RCC baseline | Run current RCC adhesion with bonded PT disabled, including the current Python and C++ legacy scene gates | The fixture shows the reference lift/release behavior for comparison |
| Press/hold | In bonded mode, bring the driver into a PT-rich contact patch under gravity | At least one PT lock is reported, no duplicate ownership is reported |
| Lift | Raise the driver below release thresholds | Adhered geometry follows the legacy RCC baseline within the declared tolerance, locks are reused, the reported energy model is `abd_ortho`, `kappa >= 1e8`, and the ABD-style energy prevents visible penetration while CCD is skipped |
| Forced pull | Increase normal gap or tangential slip past release threshold | Release counter and reason flag are reported once, beta is carried back, released pairs are absent from bonded assembly, and the adhered geometry separates or falls |

The gate must read simulation state or report fields. Writing OBJ sequences is useful for debugging, but it is not sufficient evidence.

## Reports

The host state contract, CUDA state bridge, CUDA owner, beta-threshold producer, and device release path now carry matching `RCCBondedPTCounters` fields, and `SimplexTrajectoryFilter` has a local `rcc_bonded_pt_filter_skipped_count()` for the common active compact path. `RCCBondedPTStateAccessorFeature` can expose active locked state and counters. Released snapshots exist in owner device buffers and are used by backend fixtures and RCC beta carry, but scene-accessible diagnostics do not yet expose released topology, age, flags, or per-reason counts.

Minimum backend report fields before scene gates:

| Field | Meaning | Current Status |
| --- | --- | --- |
| `rcc_bonded_pt_candidate_count` | Candidate PT pairs considered for lock | Implemented in counters |
| `rcc_bonded_pt_locked_count` | Active bonded PT locks | Implemented in counters/accessor |
| `rcc_bonded_pt_released_count` | Locks released this step | Implemented in counters; reason detail still not scene-accessible |
| `rcc_bonded_pt_filter_skip_count` | PT candidates skipped before CCD/contact | Implemented for the common active compact path; pre-CCD filter paths planned |
| `rcc_bonded_pt_duplicate_suppressed_count` | Duplicate ownership prevented | Implemented in counters |
| `rcc_bonded_pt_rejected_degenerate_count` | Rest-shape quality rejection | Implemented in counters |
| `rcc_bonded_pt_rejected_age/sticky/gap/slip/policy_count` | Lock-gate rejection reasons | Planned |
| `rcc_bonded_pt_rejected_distance_count` | Distance-band / occlusion rejections in distance-lock mode | Implemented in counters (with `rcc_bonded_pt_rejected_policy_count` for explicit `adhesion_enabled` lock rejections) |
| `rcc_bonded_pt_release_flags` or per-reason counters | Reason a locked pair returned to RCC/contact | Planned for scene-accessible diagnostics; implemented only in backend owner/test buffers |
| `rcc_bonded_pt_energy_model` | Reported model, expected production value `abd_ortho` unless the test explicitly chooses another model | Config implemented; scene report planned |
| `rcc_bonded_pt_kappa` | Reported ABD-style stiffness used by the bonded reporter | Config implemented; scene report planned |
| `rcc_bonded_pt_assembly_ms` | Bonded virtual-tet assembly time | Planned |

Scene and benchmark gates must fail if these fields are missing.

## Performance Claim Requirements

A valid speed claim must compare:

1. Baseline RCC adhesion with bonded PT disabled.
2. Bonded PT enabled with the same scene, seed, frame range, solver settings, binary, and GPU.
3. Cold setup timing for initial lock build.
4. Cache-hot timing for stable locked sets.
5. Churn timing for controlled lock/release turnover.
6. End-to-end timing plus subsystem timers.

The expected first win should appear in solver iteration count — fewer Newton/PCG iterations from replacing the stiff near-contact log-barrier Hessian of stable adhesive pairs (whose curvature grows as the gap shrinks) with a smooth, SPD-projected high-kappa ABD block — and in contact/RCC assembly time. This conditioning/iteration win is independent of the CCD skip and is active as soon as locked pairs leave the contact assembly; it is not guaranteed, because a too-high `rcc_bonded_pt_kappa` relative to surrounding material stiffness or a preconditioner that does not capture it can worsen conditioning, so iteration counts must be measured and `kappa` swept. PT CCD/filter time only improves once the pre-CCD filter lands. If only total frame time moves, the benchmark is not diagnostic enough.
