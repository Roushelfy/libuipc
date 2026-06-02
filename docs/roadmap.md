# Roadmap

This roadmap is the current status surface for the RCC bonded point-triangle acceleration project. Historical observations and command logs belong in [the journal](./development/rcc_adhesion_acceleration_journal.md).

Current focus: **Phase 2 pre-CCD filter integration** — the gating milestone for the CCD-cost lever. The bonded reporter (Phase 3) and the release/fallback machinery (Phase 4) are implemented for the current oracle/reporter scope. There are three independent performance levers, not one: (1) locked pairs are removed from barrier/friction/RCC assembly — realized now; (2) replacing their stiff near-contact log-barrier Hessian with a smooth high-kappa ABD block can improve linear-system conditioning and cut Newton/PCG iteration counts — realized now and likely the dominant win; (3) PT CCD broadphase/TOI cost — **zero CCD savings** today, because locked PTs are removed only from the DCD active-pair view in `do_filter_active` while `do_filter_toi` still processes them. Lever (3) needs the pre-CCD filter; levers (1) and (2) are measurable now. No performance claim until measured (Newton/PCG iteration counts plus per-stage timing, swept over `rcc_bonded_pt_kappa`). The live lock gate is also still beta/rest-shape only, released reason snapshots are not yet scene-accessible, and lifecycle scene correctness is unproven.

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

## Phase 2: Filter Integration (Current Gating Milestone)

The remaining pre-CCD item below is the next deliverable and the only source of CCD-broadphase/TOI savings. Until it lands, the active-view compact removes locked PTs from contact/RCC assembly only, not from CCD — so the realized wins are reduced assembly cost and (likely larger) improved solver conditioning / fewer iterations, not reduced CCD cost.

- [x] Add one shared device helper for locked PT membership lookup.
- [x] Add a CUDA deterministic lookup fixture for RCC PT key semantics, sorted membership search, misses, and empty locked sets.
- [x] Use the helper in the common `SimplexTrajectoryFilter` active-PT compact path, defaulting to no-op when no locked-key owner is connected.
- [x] Prove locked PTs are absent before `record_friction_candidates()` copies `PTs()` into `friction_PTs()` for the common active-view path.
- [ ] **Next (gating deliverable):** Use the helper before PT CCD broadphase and before the `do_filter_toi` TOI line search in stackless BVH, info stackless BVH, v0 info stackless BVH, and LBVH simplex filters. This is the only place CCD-broadphase/TOI cost is removed (the assembly and solver-conditioning wins are independent of it and already active). It may be enabled for locked pairs only after the non-penetration scene gate passes (see the architecture CCD-removal precondition).
- [ ] Add duplicate-accounting diagnostics for pair ownership.

## Phase 3: Bonded Virtual-Tet Reporter (Complete For ABD Ortho Scope)

- [x] Implement dynamic complement-energy reporter for bonded PT virtual tets.
- [x] Store oriented topology, `Dm_inv`, and rest volume in host/CUDA bonded-state buffers.
- [x] Add production config keys `rcc_bonded_pt_energy_model` and `rcc_bonded_pt_kappa`, defaulting to `abd_ortho` and `1e8`.
- [x] Replace runtime reporter math with ABD-style high-kappa energy over `F = Ds Dm_inv`.
- [x] Remove the Stable Neo-Hookean prototype from the production reporter and default configuration.
- [x] Add CPU finite-difference and GPU-vs-CPU oracle coverage for ABD-style E/G/H at `kappa >= 1e8`.
- [ ] Add a no-penetration scene observation for CCD-skipped locked pairs before bonded-mode correctness claims. This is a scene/lifecycle gate, not proof supplied by the E/G/H oracle alone.
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
- [ ] Add deterministic `pt_lift_release` scene gate: a PT-rich adhesion fixture under gravity press/hold locks, a sub-threshold lift carries the adhered body or patch, and a stronger pull releases and separates it.
- [ ] Add adhesion-off baseline for the scene so follow-through cannot be explained by constraints, ground contact, or animator setup.
- [ ] Add bonded-mode no-penetration observation during press/hold/lift with CCD skipped, such as maximum signed penetration or closest-point gap.
- [ ] Add unsupported-mode tests for missing diagnostics and disabled feature behavior.

## Phase 5: Benchmark And Default Enablement

- [ ] Add benchmark script with fixed scene, seed, frame range, build metadata, warmups, medians, and correctness checks.
- [ ] Report at least DCD, FilterTOI, contact/RCC assembly, bonded-tet assembly, solver, and frame timing.
- [ ] Compare baseline RCC and bonded-PT mode on the same binary and scene.
- [ ] Keep the feature default-off until correctness, scene, and benchmark gates pass.

## Blockers

| Blocker | Current Impact | Unblock Condition |
| --- | --- | --- |
| Lock gate is still beta/rest-shape only | The live producer now builds SVTS-compatible rest shapes and rejects degenerate fresh locks. Sticky-side and policy data are routed for release only; they are not yet lock gates. Age, sticky-side, normal-gap, tangential-slip, and policy lock gates are still planned | Add the remaining lock gates and their rejection counters before bonded-mode scene correctness claims |
| PT CCD broadphase still sees locked keys | **No CCD-cost savings exist today.** The common active-view compact removes locked PTs from contact/RCC assembly only; concrete filter `candidate_PTs()`/`toi_PTs()` and the `do_filter_toi` line search still process every locked pair. The CCD-cost lever is unrealized; the assembly and solver-conditioning levers are independent and already active | Move lookup before PT CCD broadphase and the TOI line search in all simplex filter backends, gated behind the non-penetration scene observation |
| Released-beta merge can read a stale PT key snapshot | On a step with zero PT candidates `m_prev_keys_PT` is not rebuilt, but the producer still merges released beta into it, so a re-bond can seed from an older beta for one step | Rebuild/maintain `m_prev_keys_PT` on every Phase A step including `n==0`, and add an `n==0`+release fixture |
| Scene-level release diagnostics are incomplete | Device strain/gap/slip/sticky/policy/flip/degenerate release, released snapshots, same-step relock suppression, one-shot counters, and RCC beta carry are implemented, but released topology/age/flags and per-reason counts are not yet exposed through scene-accessible report fields | Add report/accessor fields that scene gates can assert before relying on forced-pull scene release |
| No bonded-PT PT lifecycle scene | Legacy RCC lift/hold/release fixtures are automated; bonded mode still lacks lock/reuse/release assertions, no-penetration observation, complete lock gates, and scene-level release reason fields | Extend the current subdivided-cube and cube-cloth fixtures into `pt_lift_release` after lock-gate parity, scene diagnostics, and no-penetration metrics exist |
| No subsystem timers | Performance claims would collapse into total frame time | Add or expose timing fields before benchmarks |

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
| Pre-CCD filter contract | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]"` | Locked PT key is absent from PT candidate and TOI paths in every simplex filter backend | Wire the lookup helper before PT CCD broadphase in all simplex filters and add candidate/TOI instrumentation |
| PT lift/release scene | `build/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"` | PT-rich fixture locks during press/hold, ABD-style high-kappa energy prevents visible penetration while CCD is skipped, adhered geometry follows during sub-threshold lift, forced pull releases and separates, adhesion-off baseline does not lift, beta carry and zero duplicate ownership are reported | Add lock-gate parity, release diagnostics, no-penetration metric, report fields, and assertion-based scene |
| Benchmark matrix | `uv run --no-sync python scripts/bench_rcc_adhesion_acceleration.py --scene stable_cloth_peel --frames 40 --warmup 5 --runs 10` | Reports cold, cache-hot, churn, and end-to-end medians with correctness fields | Add benchmark script, timers, and report parser |

## Next Safe Task

Move locked-key lookup before PT CCD broadphase and the `do_filter_toi` TOI line search in every simplex filter — this is the only change that produces CCD-cost savings; the assembly and solver-conditioning/iteration-count wins are independent of it and measurable now. Couple it with the `pt_lift_release` no-penetration scene observation (see the architecture CCD-removal precondition): CCD may be skipped for a locked pair only once that gate proves the bonded ABD energy prevents tunneling. Then expose release-reason scene diagnostics, add lock-gate parity, and only then benchmark. Keep bonded mode default-off and make no correctness/performance claims until pre-CCD filtering, the no-penetration gate, lock gates, scene diagnostics, and benchmark gates all pass.
