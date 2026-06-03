# Roadmap

This roadmap is the current status surface for the RCC bonded point-triangle acceleration project. Historical observations and command logs belong in [the journal](./development/rcc_adhesion_acceleration_journal.md).

Current focus: **the benchmark and the remaining `pt_lift_release` lifecycle assertions** (forced-pull release, adhesion-off baseline). The bonded reporter (Phase 3), the release/fallback machinery (Phase 4), and the pre-CCD filter (Phase 2) are implemented, and the no-penetration scene gate (`[rcc_bonded_pt][scene][pt_lift_release]`) passes: with `rcc_bonded_pt_skip_ccd` on, the cube fixture forms 8 bonded locks and stays penetration-free (contact-face gap min ~+0.019) through press/hold/lift while the lower cube is carried by the bonded energy. There are three independent performance levers, not one: (1) locked pairs are removed from barrier/friction/RCC assembly — active now; (2) replacing their stiff near-contact log-barrier Hessian with a smooth high-kappa ABD block can improve linear-system conditioning and cut Newton/PCG iteration counts — active now and likely the dominant win; (3) PT CCD broadphase/TOI cost — the skip is implemented in all four simplex filters but stays behind default-off `rcc_bonded_pt_skip_ccd`, so there are still **zero CCD savings** by default. The no-penetration precondition is now met for the cube fixture; the default stays off until the benchmark shows a net win and broader fixtures (cloth/patch) also pass. No performance claim until measured (Newton/PCG iteration counts plus per-stage timing, swept over `rcc_bonded_pt_kappa`). The live lock gate is also still beta/rest-shape only, and released reason snapshots are not yet scene-accessible.

Parallel workstream (now the active implementation thread): **Phase 6, full-feature adhesion and per-primitive beta**. The bonded acceleration today only covers face-interior point-triangle contacts because RCC adhesion itself is PT-only — `EE_*`/`PE_*`/`PP_*` adhesion returned zero, beta lived only on PT-classified pairs, and there was no contact-area weight, so a stable edge/corner contact (the off-diagonal cube corner, a cloth fold) gets neither adhesion nor a bond. Phase 6 rebuilds adhesion on the XBow `RCCAdhesionEnergy3D` model: beta per VT primitive, true feature distance per closest-feature classification (PP/PE/PT), contact-area weight, and reuse of the barrier's classified pairs/distances; then the bonded lock decides on the VT primitive while the ABD tet stays point-plane. Step 1 (PE/PP true-feature adhesion formulas, replacing the `return 0` stubs) is implemented and behaviour-neutral until per-primitive beta lands (Step 2). See [architecture](./architecture.md) "Full-Feature Adhesion And Per-Primitive Beta".

## Core Principle

Every implementation decision must serve one goal: **reduce stable RCC point-triangle adhesion cost without changing pair ownership, beta continuity, or non-penetration accounting**.

Non-negotiable rules:

1. A locked PT pair is excluded from CCD/contact/RCC only when one bonded virtual tet owns the same four global vertex degrees of freedom for that step.
2. Locked-pair lookup happens before PT CCD broadphase and before `friction_PTs()` is recorded in every simplex trajectory filter backend.
3. A PT pair cannot be visible through `PTs()` or `friction_PTs()` and through bonded virtual-tet assembly in the same Newton iteration.
4. Lock is allowed only after beta, age, sticky-side, normal-gap, tangential-slip, and rest-shape conditioning gates pass.
5. Release must carry a beta value back to RCC persistence; release cannot silently reset beta or delete pair history.
6. A locked pair that skips CCD/contact/RCC must be backed in the same step by high-kappa ABD-style virtual-tet energy over `F = Ds Dm_inv`; `kappa` must be positive and the production/default gate value is `>= 1e8`. The SVTS Stable Neo-Hookean prototype does not satisfy the production replacement requirement.
7. Numeric changes require CPU or legacy oracles before scene gates and benchmarks.
8. Speed claims are invalid unless cold setup, cache-hot steady state, churn, and end-to-end frame timing are reported with the same build and scene.

Exception: a late post-detection split may be used for a throwaway prototype, but it cannot be called production, cannot support a performance claim, and must be marked experimental in reports.

ABD note: `SoftVertexTriangleStitch` is a rest-shape/thickness reference only. It is not the runtime replacement energy for a locked PT pair that has been removed from CCD/contact/RCC.

## Phase 0: Project Skeleton And Current Gates (Complete)

- [x] Current roadmap exists and names the core principle, phase, blockers, next tasks, current gates, and planned gates.
- [x] Stable architecture is split into [architecture](./architecture.md).
- [x] Stable coding, kernel, data-layout, validation, and benchmark conventions are split into [conventions](./conventions.md).
- [x] Focused subsystem details are split into [RCC adhesion acceleration](./rcc_adhesion_acceleration.md).
- [x] Chronological decisions and command results are recorded in [the journal](./development/rcc_adhesion_acceleration_journal.md).
- [x] A default portable docs/source-gates entry point exists: `scripts/run_rcc_adhesion_acceleration_all_gates.py`.
- [x] Legacy RCC lift/hold/release scene gates exist for current behavior: Python pytest fixtures and C++ `uipc_test_sim_case "[rcc_adhesion][gate]"`.

## Phase 1: State Contract, CPU Oracle, And CUDA State Bridge (Complete For Current Data Contract)

### State Owner

- [x] Implement `RCCBondedPTState` minimum host contract for locked keys, oriented topologies, beta, age, and release flags.
- [x] Extend the state owner and CUDA bridge with rest-shape metrics (`Dm_inv`, `rest_volume`) once the SVTS oracle lands.
- [x] Define feature-disabled default behavior, the `rcc_bonded_pt_enabled` master config key, and the beta lock threshold key.
- [ ] Add remaining age and lock-gate config/report keys under the `rcc_bonded_pt` prefix. Rest-shape threshold keys, production ABD energy keys, and release threshold keys are implemented.
- [x] Add minimum `RCCBondedPTCounters` fields for candidate, locked, released, degenerate-rejected, filter-skipped, and duplicate-suppressed pairs.
- [x] Mirror the host state contract into a CUDA-owned `RCCBondedPTStateBridge` with device buffers and counter roundtrip.
- [x] Add a CUDA `RCCBondedPTSystem` owner that holds the bridge, feeds sorted locked keys into `SimplexTrajectoryFilter`, and syncs common active-filter skip counts.
- [x] Add a device-side RCC Phase A beta-threshold producer that compacts high-beta PTs into the CUDA owner, carries existing locks, increments age, suppresses duplicate candidate keys, builds SVTS-compatible rest shapes for new locks, and rejects degenerate fresh locks.
- [x] Expose `RCCBondedPTCounters` and active locked state through `RCCBondedPTStateAccessorFeature` after the CUDA bridge is wired into the live RCC pipeline.

### Oracles

- [x] Add a deterministic two-pair state fixture: one pair stays locked, one pair releases.
- [x] Add a CPU rest-shape oracle matching `SoftVertexTriangleStitch` construction, including `min_separate_distance`.
- [x] Add ABD-style high-kappa E/G/H oracle over `F = Ds Dm_inv` for a single bonded PT virtual tet.

### Default Validation

- [x] Wire the state fixture into the current validation path.
- [x] Wire the rest-shape CPU oracle into the current validation path.
- [x] Wire the ABD-style high-kappa E/G/H CPU oracle into the current validation path.
- [x] Wire the CUDA state bridge roundtrip into the backend CUDA validation path.
- [ ] Keep source scans as boundary checks only; do not use them as proof of math.

## Phase 2: Filter Integration (Pre-CCD Filter Implemented, Default-Off)

The pre-CCD item below is now implemented: all four simplex filters reject locked PTs inside the PT broadphase predicate, so locked pairs never become candidates (no broadphase emission, no TOI, no active pair). It is gated behind default-off `rcc_bonded_pt_skip_ccd` until the no-penetration scene gate passes. With it off, the active-view compact still removes locked PTs from contact/RCC assembly, and the solver-conditioning win is active regardless.

- [x] Add one shared device helper for locked PT membership lookup.
- [x] Add a CUDA deterministic lookup fixture for RCC PT key semantics, sorted membership search, misses, and empty locked sets.
- [x] Use the helper in the common `SimplexTrajectoryFilter` active-PT compact path, defaulting to no-op when no locked-key owner is connected.
- [x] Prove locked PTs are absent before `record_friction_candidates()` copies `PTs()` into `friction_PTs()` for the common active-view path.
- [x] Reject locked PTs inside the PT broadphase predicate (before candidate emission) in stackless BVH, info stackless BVH, v0 info stackless BVH, and LBVH simplex filters, via the shared `rcc_bonded_pt_candidate_is_locked` helper. This removes locked pairs from broadphase, TOI, and active-pair emission at once. Gated behind default-off `rcc_bonded_pt_skip_ccd`; enable only after the non-penetration scene gate passes (architecture CCD Removal Precondition). Covered by `[rcc_bonded_pt][filter][ccd]`.
- [ ] Add duplicate-accounting diagnostics for pair ownership.

## Phase 3: Bonded Virtual-Tet Reporter (Complete For ABD Ortho Scope)

- [x] Implement dynamic complement-energy reporter for bonded PT virtual tets.
- [x] Store oriented topology, `Dm_inv`, and rest volume in host/CUDA bonded-state buffers.
- [x] Add production config keys `rcc_bonded_pt_energy_model` and `rcc_bonded_pt_kappa`, defaulting to `abd_ortho` and `1e8`.
- [x] Replace runtime reporter math with ABD-style high-kappa energy over `F = Ds Dm_inv`.
- [x] Remove the Stable Neo-Hookean prototype from the production reporter and default configuration.
- [x] Add CPU finite-difference and GPU-vs-CPU oracle coverage for ABD-style E/G/H at `kappa >= 1e8`.
- [x] Add a no-penetration scene observation for CCD-skipped locked pairs before bonded-mode correctness claims. Implemented by `[rcc_bonded_pt][scene][pt_lift_release]`: with `rcc_bonded_pt_skip_ccd` on, the cube fixture forms 8 bonded locks and the contact-face gap stays positive (min ~+0.019) through press/hold/lift. This is a scene/lifecycle gate, not proof supplied by the E/G/H oracle alone.
- [x] Keep the reporter out of contact-component accounting unless an explicit diagnostic requests comparison.

## Phase 4: Release, Fallback, And Scene Gate (Backend Implemented; Downstream Of Pre-CCD Filter)

- [x] Implement device release reason flags for strain, normal gap, tangential slip, sticky-side failure, disabled contact policy, flip, and degenerate current shape.
- [x] Add `rcc_bonded_pt_release_strain`, `rcc_bonded_pt_release_gap`, and `rcc_bonded_pt_release_slip`, defaulting to large disabled thresholds until scenes choose values.
- [x] Carry beta across lock/release transitions by merging released key/beta snapshots back into RCC PT persistence without overwriting newer RCC beta for duplicate keys.
- [x] Compact released locks out of bonded state before bonded reporter assembly input is refreshed, keep released key/topology/beta/age/reason snapshots in backend owner buffers for CUDA assertions and RCC beta carry, and prevent same-step relock of released keys.
- [x] Add deterministic release fixtures before scene work: one locked PT stays active, one releases by controlled strain, released key/beta/flag snapshots remain aligned, counters update once, and beta is visible to RCC persistence.
- [x] Route sticky-side and policy release context from RCC Phase A into the bonded owner without host roundtrips.
- [ ] Expose released topology, age, flags, and per-reason release counts through a scene-accessible accessor/report path; the current public owner path exposes only released key/beta for RCC beta carry.
- [ ] Add live lock-gate parity for minimum age, sticky-side consistency, normal-gap band, tangential-slip band, and contact policy, with rejection counters distinct from release counters.
- [~] Add deterministic `pt_lift_release` scene gate. Press/hold/lift is implemented (`[rcc_bonded_pt][scene][pt_lift_release]`): 8 locks form, the lower cube is carried by the bonded energy with no penetration while CCD is skipped. Still to add: forced-pull release/separation assertions in the same gate.
- [ ] Add adhesion-off baseline for the scene so follow-through cannot be explained by constraints, ground contact, or animator setup.
- [x] Add bonded-mode no-penetration observation during press/hold/lift with CCD skipped (closest contact-face gap), via `[rcc_bonded_pt][scene][pt_lift_release]`.
- [ ] Add unsupported-mode tests for missing diagnostics and disabled feature behavior.

## Phase 5: Benchmark And Default Enablement

- [ ] Add benchmark script with fixed scene, seed, frame range, build metadata, warmups, medians, and correctness checks.
- [ ] Report at least DCD, FilterTOI, contact/RCC assembly, bonded-tet assembly, solver, and frame timing.
- [ ] Compare baseline RCC and bonded-PT mode on the same binary and scene.
- [ ] Keep the feature default-off until correctness, scene, and benchmark gates pass.

## Phase 6: Full-Feature Adhesion And Per-Primitive Beta (In Progress)

This phase extends RCC adhesion to the full VT/EE primitive so the bonded acceleration can cover edge/corner contacts, not only face-interior PT. It is built on the XBow `RCCAdhesionEnergy3D` model (per-primitive beta, true feature distance, area weight, barrier reuse). The bonded virtual-tet energy is unchanged: it stays the point-plane ABD shape energy over `F = Ds Dm_inv`. See [architecture](./architecture.md) "Full-Feature Adhesion And Per-Primitive Beta".

- [x] Step 1: PE/PP true-feature adhesion formulas. Replace the `return 0` `PE_*`/`PP_*` normal and tangential adhesion stubs in `codim_ipc_simplex_rcc_adhesive_function.h` with the true point-edge / point-point feature distance (`point_edge_distance2`, `point_point_distance2`) and matching friction basis, mirroring the `PT_*` functions. Behaviour-neutral: every assembly call site early-outs on `beta <= 0`, and `m_beta_PE/PP` are zero-filled until Step 2.
- [x] PE/PP E/G/H finite-difference oracle: `apps/tests/backends/cuda/rcc_adhesion_oracle.cu`, tag `[rcc_adhesion][oracle][feature_adhesion][cuda]` (PE Vector9/9x9 + PP Vector6/6x6, normal + tangential, FD-checked; 12 assertions). Proves the Step-1 formulas before per-primitive beta makes them load-bearing. Wired into `run_rcc_adhesion_acceleration_cuda_gates.py`.
- [x] Step 2: per-VT-primitive beta + unified assembly (the behavior-flipping core). `_evolve_beta_step_at_end` + Phase B init/match iterate `friction_VTs`, keyed by the orientation-invariant `PT_pair_key` (identical for any VT primitive → beta persists across PP/PE/PT transitions), using the **flagged** closest-feature distance `point_triangle_distance2(flag,...)` + the matching lagged tangent basis (`VT_tangential_rel_dx_sq`). `do_compute_energy`/`do_assemble` run ONE loop over `friction_VTs`: `VT_normal_adhesion_*` (flagged distance → 12-DOF block, auto PT/PE/PP) + `VT_tangential_adhesion_*` (flag-switched PT/PE/PP basis scattered to 12 DOF). The reporter routes all VT primitives to the PT output slot via the new `SimplexFrictionalContact::friction_pair_counts` hook (PT=VT count, EE/PE/PP=0). The sticky + occlusion gates carry over verbatim (full-triangle). `m_beta_EE/PE/PP`, `_sync_disabled_buffers`, and the separate per-feature loops are retired; `m_beta_PT` now spans the whole VT list. Edge/corner contacts now get adhesion. Verified: `[rcc_adhesion][oracle]` 12/4, backend `[rcc_bonded_pt]` 247/14, bunny 4/1, legacy `[rcc_adhesion][gate]` 836/2, bonded `[rcc_bonded_pt][scene][pt_lift_release]` 596/1 — all green, no regression from the behavior flip. Bonded producer kept on the `flag==4`-compacted subset (`m_beta_PT_face`) so bonding stays PT-only until Step 5.
- [ ] Step 3 (optional, lowest priority): per-vertex contact area `A_k` weight. Today area is **lumped into Cn/Ct** by the libuipc IPC convention (spec `rcc_adhesion.md:176-177`, matching barrier `kappa`); the assembled energy has no `A_k` term. A separate A_k is not a bug fix and would double-count under the lumped convention — it is justified only for resolution-independent / non-uniform-mesh adhesion (XBow keeps Cn/Ct as densities and multiplies a per-vertex tributary area reused from the barrier). If done, reinterpret Cn/Ct as densities and reuse the barrier's boundary-point area; do not invent a parallel field.
- [ ] Step 4: distance single-compute (structural, NOT a stored buffer). The barrier recomputes `d^2` in-kernel and stores nothing, and a per-pair `d^2`/derivative scratch buffer would be a GPU regression (memory traffic > the cheap in-register FLOPs). Instead compute `d^2`/basis once per pair in one kernel body: first intra-RCC (reuse the unflagged plane `d^2` across RCC's own normal energy/grad/hess + `db_dd2`), then optionally merge with the friction kernel (RCC is already a friction subclass over the same lagged `friction_PTs()`, and its lagged tangent-basis/foot is bit-identical to friction's). The only barrier-shareable quantities are `db_dd2` and the flagged `d^2`. See [architecture](./architecture.md) "Distance handling". (Corrected from the earlier "reuse the barrier's distances / layer like XBow" framing — XBow does not actually reuse barrier distances; journal 2026-06-02 Distance Reuse Analysis.)
- [x] Step 5: bonded lock decides on the VT primitive. The producer is fed the full `friction_VTs` topologies + per-VT beta (`m_vt_topos` extracted from `friction_VTs`, aligned with `m_beta_PT`), so corner/edge contacts bond, not only face-interior PT. The producer's rest-shape conditioning (`min_separate_distance` normal offset) + `det_dm_min` accept well-conditioned pairs (incl. near-coplanar corners) and reject degenerate ones; the ABD tet stays point-plane (`F = Ds Dm_inv`). The filter's locked-key compact now also removes locked VTs from `friction_VTs` (new `rcc_bonded_pt_unlocked_VT`), so a bonded pair is never both adhered and bonded (no double-count). Verified: `[rcc_bonded_pt][scene][pt_lift_release]` 596/1 stays penetration-free; backend `[rcc_bonded_pt]` 247/14, oracle 12/4, bunny 4/1, legacy `[rcc_adhesion][gate]` 836/2 all green. Headless probe: the faceted subdivided cube now forms 96 bonded locks (was 8 face-only), the cube is carried through the lift, no NaN.
- [x] Behavioral confirmation (headless demo probe; journal 2026-06-02): faceted subdivided cube — the per-VT beta snapshot spans 96 primitives at beta=1.0 vs 8 face-interior bonded locks (edge/corner VTs now carry beta), and adhesion holds the lower cube during lift (botY +0.170→+0.426 vs left behind when off); oriented cloth (single-diagonal) — no diagonal-pull artifact (in-plane drift ON≈OFF, no diagonal bias, cloth stays flat). No NaN. A committed `[rcc_adhesion]` unit/demo gate is still nice-to-have.
- [ ] Optional follow-up: EE adhesion (edge-edge true-feature distance + per-primitive beta), out of the first milestone.

## Blockers

| Blocker | Current Impact | Unblock Condition |
| --- | --- | --- |
| Lock gate is still beta/rest-shape only | The live producer now builds SVTS-compatible rest shapes and rejects degenerate fresh locks. Sticky-side and policy data are routed for release only; they are not yet lock gates. Age, sticky-side, normal-gap, tangential-slip, and policy lock gates are still planned | Add the remaining lock gates and their rejection counters before bonded-mode scene correctness claims |
| Pre-CCD skip is implemented but default-off | The skip mechanism rejects locked PTs in every simplex filter's PT broadphase predicate, but `rcc_bonded_pt_skip_ccd` defaults off because removing CCD removes the last non-penetration guard. So there are no CCD savings by default | No-penetration gate now passes for the cube fixture; flip the default only after the benchmark shows a net win and more fixtures (cloth/patch) pass |
| Released-beta merge can read a stale PT key snapshot | On a step with zero PT candidates `m_prev_keys_PT` is not rebuilt, but the producer still merges released beta into it, so a re-bond can seed from an older beta for one step | Rebuild/maintain `m_prev_keys_PT` on every Phase A step including `n==0`, and add an `n==0`+release fixture |
| Scene-level release diagnostics are incomplete | Device strain/gap/slip/sticky/policy/flip/degenerate release, released snapshots, same-step relock suppression, one-shot counters, and RCC beta carry are implemented, but released topology/age/flags and per-reason counts are not yet exposed through scene-accessible report fields | Add report/accessor fields that scene gates can assert before relying on forced-pull scene release |
| No bonded-PT PT lifecycle scene | Legacy RCC lift/hold/release fixtures are automated; bonded mode still lacks lock/reuse/release assertions, no-penetration observation, complete lock gates, and scene-level release reason fields | Extend the current subdivided-cube and cube-cloth fixtures into `pt_lift_release` after lock-gate parity, scene diagnostics, and no-penetration metrics exist |
| No subsystem timers | Performance claims would collapse into total frame time | Add or expose timing fields before benchmarks |
| ~~Bonding is PT-only~~ (resolved, Phase 6 Step 5) | Resolved: the producer now locks on the full VT primitive (`friction_VTs` topos + per-VT beta), and locked VTs are removed from `friction_VTs`, so corner/edge contacts bond with no double-count. `pt_lift_release` stays penetration-free (596/1); faceted cube forms 96 locks (was 8). The ABD tet remains point-plane | — |

## Playbook Compliance

| Minimum Standard | Status | Evidence Or Required Work |
| --- | --- | --- |
| Current roadmap names principle, phase, next tasks, blockers, and gates | Satisfied | This file is the current status surface |
| Stable architecture and conventions are documented | Satisfied | [architecture](./architecture.md) and [conventions](./conventions.md) |
| Runnable gates and planned gates are separated | Satisfied | Current gates are below; future commands are in planned gates |
| New tests are wired into a default validation path | Partially satisfied | Portable docs/source gates are wired through `run_rcc_adhesion_acceleration_all_gates.py`; local CUDA gates are wired through `run_rcc_adhesion_acceleration_cuda_gates.py`; legacy RCC scene gates are wired into pytest and `sim_case`; bonded-PT state, rest-shape payload, counter, state accessor, CPU oracle, CUDA state bridge, CUDA lookup, common active-filter, CUDA owner, beta-threshold producer with live rest-shape rejection, release/beta carry, ABD GPU virtual-tet reporter oracle, and bunny BVH regression fixtures are implemented; lock-gate parity, scene-accessible release diagnostics, pre-CCD filter, bonded lifecycle scene, and benchmark gates are still planned |
| Numeric or algorithmic claims have CPU or legacy oracles | Satisfied for current scope | State, rest-shape, ABD-style high-kappa virtual-tet E/G/H, and ABD GPU reporter matching gates exist |
| Benchmark claims split cold, cache-hot, churn, and end-to-end timing | Not yet implemented | Phase 5 requires benchmark script, timers, and correctness fields |
| Journals record commands, observed results, and decisions | Satisfied | [journal](./development/rcc_adhesion_acceleration_journal.md) |

## Validation Gates

These commands are runnable today from the repository root and must pass before changing the roadmap status. Note: the portable source/doc gates prove documentation structure and source anchors only — not runtime behavior. Behavioral proof requires the CUDA gate bundle and the listed `uipc_test_*` binaries below.

| Gate | Command | Required Result |
| --- | --- | --- |
| Portable docs/source gates | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Runs source/doc checks, Python syntax checks, and docs build; exits 0 |
| Source/doc boundary gate | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Confirms doc skeleton, nav link, all-gates entry, and current source anchors |
| Python syntax gate | `uv run --no-sync python -m py_compile scripts/run_rcc_adhesion_acceleration_gates.py scripts/run_rcc_adhesion_acceleration_all_gates.py scripts/run_rcc_adhesion_acceleration_cuda_gates.py scripts/build_docs.py` | Exits 0 |
| Docs site build | `uv run --no-sync python scripts/build_docs.py -o /tmp/libuipc-docs-check` | MkDocs and MkDoxy build the docs and API pages |
| Local RCC CUDA gate bundle | `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py --no-build` | Runs bonded-PT core/backend tests and the bunny GPU sanity regression against an existing CUDA build |
| Bonded PT state and counters | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Deterministic zipped key/topology/beta/age/release/rest-shape fixture and counter fixture pass |
| Bonded PT CPU oracles | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle]" -r compact` | Rest-shape conditioning plus ABD-style high-kappa virtual-tet energy, gradient, and Hessian CPU oracles pass |
| Bonded PT CUDA state bridge | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]" -r compact` | Host state, device buffers, rest-shape payloads, pending release flags, extract-release counters, and clear behavior roundtrip |
| Bonded PT state accessor | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][accessor]" -r compact` | Frontend feature wrapper reports locked count, counters, and active locked-state snapshots through the overrider contract |
| Bonded PT CUDA lookup helper | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lookup]" -r compact` | RCC PT key semantics, sorted membership lookup, miss handling, and empty locked-set behavior pass on CUDA |
| Bonded PT common active-filter compact | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter]" -r compact` | Locked PTs are removed from `SimplexTrajectoryFilter::PTs()` and therefore from `friction_PTs()` when sorted locked keys are supplied |
| Bonded PT CUDA owner and beta/rest producer | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` | Owner uploads locked state, feeds sorted keys to the filter, syncs skip counters, and device-side producer carries existing locks, adds high-beta candidates, increments age, suppresses duplicate candidate keys, builds live rest shapes for fresh locks, and rejects degenerate fresh locks |
| Bonded PT release and beta carry | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][release]" -r compact` | Device strain, gap, slip, sticky-side, and policy release compact active locks, expose released key/topology/beta/age/flag snapshots, suppress same-step relock, increment release counters once, and merge released beta back into RCC persistence without overwriting newer duplicate beta |
| Bonded PT pre-CCD filter membership | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]" -r compact` | The shared `rcc_bonded_pt_candidate_is_locked` helper that every simplex filter's PT broadphase predicate calls rejects a locked (orientation-permuted) PT candidate and keeps unlocked ones; empty locked set keeps all |
| Bonded PT no-penetration scene (press/hold/lift) | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]" -r compact` | With `rcc_bonded_pt_skip_ccd` on, the cube fixture forms bonded locks, the contact-face gap stays non-negative through press/hold/lift (no penetration while CCD is skipped), and the lower cube is carried by the bonded energy |
| Bonded PT GPU virtual-tet reporter oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][reporter][abd_oracle]" -r compact` | ABD-style reporter math matches the CPU virtual-tet energy, gradient, and Hessian oracle within tolerance at `kappa >= 1e8` |
| CUDA bunny BVH/radix-sort regression | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Ensures bonded-PT CUDA payload/layout changes do not destabilize the existing `SimplicialSurfaceDistanceCheck` + `InfoStacklessBVH` path |
| Legacy RCC Python lift/release scenes | `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Subdivided cube-cube and cube-cloth lift/hold/release fixtures pass |
| Legacy RCC C++ lift/release scenes | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Two native scene gates pass after the sim case target is built |

## Planned Gates

These commands are target gates for missing code, missing tests, or missing system dependencies. They are not current proof.

| Gate | Target Command | Required Result | Missing Piece |
| --- | --- | --- | --- |
| Lock-gate parity contract | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lock_gate]"` | High-beta candidates lock only when minimum age, sticky-side, normal-gap, tangential-slip, rest-shape, and contact-policy gates pass; each rejection counter is distinct from release counters | Add live lock-gate inputs/config and deterministic producer fixtures |
| Scene diagnostics/accessor contract | `build/bin/uipc_test_core "[rcc_bonded_pt][accessor][release]"` | Scene-accessible diagnostics expose released topology, beta, age, flags, per-reason counts, active lock count, filter skip count, and duplicate suppression without double-counting | Extend core/backend accessor and pybind/native scene access as needed |
| Pre-CCD filter scene contract | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd][scene]"` | In a real scene with `rcc_bonded_pt_skip_ccd` on, a locked PT key is absent from PT candidate and TOI paths in every simplex filter backend | Mechanism implemented (predicate-level membership, gated default-off, unit-tested by `[rcc_bonded_pt][filter][ccd]`); remaining piece is scene-level candidate/TOI instrumentation, coupled to the no-penetration gate |
| PT lift/release scene | `build/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"` | PT-rich fixture locks during press/hold, ABD-style high-kappa energy prevents visible penetration while CCD is skipped, adhered geometry follows during sub-threshold lift, forced pull releases and separates, adhesion-off baseline does not lift, beta carry and zero duplicate ownership are reported | Press/hold/lift implemented and runnable (8 locks, no penetration, lower cube carried, zero duplicates); remaining: forced-pull release/separation and adhesion-off baseline in the same gate |
| Benchmark matrix | `uv run --no-sync python scripts/bench_rcc_adhesion_acceleration.py --scene stable_cloth_peel --frames 40 --warmup 5 --runs 10` | Reports cold, cache-hot, churn, and end-to-end medians with correctness fields | Add benchmark script, timers, and report parser |

## Next Safe Task

The `pt_lift_release` no-penetration observation now passes (`[rcc_bonded_pt][scene][pt_lift_release]`). Next: add the benchmark (Newton/PCG iteration counts + per-stage timing, swept over `rcc_bonded_pt_kappa`, bonded vs baseline on the same scene and binary) to measure whether the conditioning lever is a net win; add the forced-pull release/separation and adhesion-off baseline to the scene gate; then lock-gate parity and scene-accessible release diagnostics. Keep bonded mode and `rcc_bonded_pt_skip_ccd` default-off until the benchmark and broader fixtures justify flipping the default.
