# Roadmap

This roadmap is the current status surface for the RCC bonded point-triangle acceleration project. Historical observations and command logs belong in [the journal](./development/rcc_adhesion_acceleration_journal.md).

## Core Principle

Every implementation decision must serve one goal: **reduce stable RCC point-triangle adhesion cost without changing pair ownership, beta continuity, or non-penetration accounting**.

Non-negotiable rules:

1. A locked PT pair is excluded from CCD/contact/RCC only when one bonded virtual tet owns the same four global vertex degrees of freedom for that step.
2. Locked-pair lookup happens before PT CCD broadphase and before `friction_PTs()` is recorded in every simplex trajectory filter backend.
3. A PT pair cannot be visible through `PTs()` or `friction_PTs()` and through bonded virtual-tet assembly in the same Newton iteration.
4. Lock is allowed only after beta, age, sticky-side, normal-gap, tangential-slip, and rest-shape conditioning gates pass.
5. Release must carry a beta value back to RCC persistence; release cannot silently reset beta or delete pair history.
6. Numeric changes require CPU or legacy oracles before scene gates and benchmarks.
7. Speed claims are invalid unless cold setup, cache-hot steady state, churn, and end-to-end frame timing are reported with the same build and scene.

Exception: a late post-detection split may be used for a throwaway prototype, but it cannot be called production, cannot support a performance claim, and must be marked experimental in reports.

## Phase 0: Project Skeleton And Current Gates (Complete)

- [x] Current roadmap exists and names the core principle, phase, blockers, next tasks, current gates, and planned gates.
- [x] Stable architecture is split into [architecture](./architecture.md).
- [x] Stable coding, kernel, data-layout, validation, and benchmark conventions are split into [conventions](./conventions.md).
- [x] Focused subsystem details are split into [RCC adhesion acceleration](./rcc_adhesion_acceleration.md).
- [x] Chronological decisions and command results are recorded in [the journal](./development/rcc_adhesion_acceleration_journal.md).
- [x] A default portable docs/source-gates entry point exists: `scripts/run_rcc_adhesion_acceleration_all_gates.py`.
- [x] Legacy RCC lift/hold/release scene gates exist for current behavior: Python pytest fixtures and C++ `uipc_test_sim_case "[rcc_adhesion][gate]"`.

## Phase 1: State Contract, CPU Oracle, And CUDA State Bridge (Current)

### State Owner

- [x] Implement `RCCBondedPTState` minimum host contract for locked keys, oriented topologies, beta, age, and release flags.
- [x] Extend the state owner and CUDA bridge with rest-shape metrics (`Dm_inv`, `rest_volume`) once the SVTS oracle lands.
- [x] Define feature-disabled default behavior, the `rcc_bonded_pt_enabled` master config key, and the beta lock threshold key.
- [ ] Add remaining age/release config keys under the `rcc_bonded_pt` prefix. Rest-shape threshold keys and global virtual-tet material keys are implemented.
- [x] Add minimum `RCCBondedPTCounters` fields for candidate, locked, released, degenerate-rejected, filter-skipped, and duplicate-suppressed pairs.
- [x] Mirror the host state contract into a CUDA-owned `RCCBondedPTStateBridge` with device buffers and counter roundtrip.
- [x] Add a CUDA `RCCBondedPTSystem` owner that holds the bridge, feeds sorted locked keys into `SimplexTrajectoryFilter`, and syncs common active-filter skip counts.
- [x] Add a device-side RCC Phase A beta-threshold producer that compacts high-beta PTs into the CUDA owner, carries existing locks, increments age, suppresses duplicate candidate keys, builds SVTS-compatible rest shapes for new locks, and rejects degenerate fresh locks.
- [x] Expose `RCCBondedPTCounters` and locked/release state through `RCCBondedPTStateAccessorFeature` after the CUDA bridge is wired into the live RCC pipeline.

### Oracles

- [x] Add a deterministic two-pair state fixture: one pair stays locked, one pair releases.
- [x] Add a CPU rest-shape oracle matching `SoftVertexTriangleStitch` construction, including `min_separate_distance`.
- [x] Add a CPU Stable Neo-Hookean E/G/H oracle for a single bonded PT virtual tet.

### Default Validation

- [x] Wire the state fixture into the current validation path.
- [x] Wire the rest-shape CPU oracle into the current validation path.
- [x] Wire the Stable Neo-Hookean E/G/H CPU oracle into the current validation path.
- [x] Wire the CUDA state bridge roundtrip into the backend CUDA validation path.
- [ ] Keep source scans as boundary checks only; do not use them as proof of math.

## Phase 2: Filter Integration

- [x] Add one shared device helper for locked PT membership lookup.
- [x] Add a CUDA deterministic lookup fixture for RCC PT key semantics, sorted membership search, misses, and empty locked sets.
- [x] Use the helper in the common `SimplexTrajectoryFilter` active-PT compact path, defaulting to no-op when no locked-key owner is connected.
- [x] Prove locked PTs are absent before `record_friction_candidates()` copies `PTs()` into `friction_PTs()` for the common active-view path.
- [ ] Use the helper before PT CCD broadphase in stackless BVH, info stackless BVH, v0 info stackless BVH, and LBVH simplex filters.
- [ ] Add duplicate-accounting diagnostics for pair ownership.

## Phase 3: Bonded Virtual-Tet Reporter

- [x] Implement dynamic complement-energy reporter for bonded PT virtual tets.
- [x] Store oriented topology, `Dm_inv`, and rest volume in host/CUDA bonded-state buffers.
- [x] Store global material parameters for bonded virtual-tet assembly through `rcc_bonded_pt_mu` and `rcc_bonded_pt_lambda`, defaulting to zero contribution until explicitly set.
- [x] Match CPU oracle energy, gradient, and Hessian within declared tolerances.
- [x] Keep the reporter out of contact-component accounting unless an explicit diagnostic requests comparison.

## Phase 4: Release, Fallback, And Scene Gate

- [ ] Implement release reason flags for strain, normal gap, tangential slip, rest-shape quality, sticky-side failure, and disabled contact policy.
- [ ] Carry beta across lock/release transitions.
- [ ] Add deterministic `pt_lift_release` scene gate: a PT-rich adhesion fixture under gravity press/hold locks, a sub-threshold lift carries the adhered body or patch, and a stronger pull releases and separates it.
- [ ] Add adhesion-off baseline for the scene so follow-through cannot be explained by constraints, ground contact, or animator setup.
- [ ] Add unsupported-mode tests for missing diagnostics and disabled feature behavior.

## Phase 5: Benchmark And Default Enablement

- [ ] Add benchmark script with fixed scene, seed, frame range, build metadata, warmups, medians, and correctness checks.
- [ ] Report at least DCD, FilterTOI, contact/RCC assembly, bonded-tet assembly, solver, and frame timing.
- [ ] Compare baseline RCC and bonded-PT mode on the same binary and scene.
- [ ] Keep the feature default-off until correctness, scene, and benchmark gates pass.

## Blockers

| Blocker | Current Impact | Unblock Condition |
| --- | --- | --- |
| Lock gate is still beta/rest-shape only | The live producer now builds SVTS-compatible rest shapes and rejects degenerate fresh locks, but age, sticky-side, normal-gap, tangential-slip, and policy gates are still planned | Add the remaining lock gates and their rejection counters before bonded-mode scene correctness claims |
| PT CCD broadphase still sees locked keys | The common active-view compact prevents locked PTs from reaching `friction_PTs()` when keys are supplied, but concrete filter `candidate_PTs()` and `toi_PTs()` are still generated before that compact | Move lookup into the PT candidate/TOI path for all simplex filter backends |
| No bonded-mode release diagnostics | Host release flags and live counter accessors exist, but there is still no device release policy producing gap/slip/strain/sticky-side reasons | Add release kernels, beta carry, and scene assertions that read `RCCBondedPTStateAccessorFeature` |
| No bonded-PT PT lifecycle scene | Legacy RCC lift/hold/release fixtures are automated; bonded mode still lacks lock/reuse/release assertions and report fields | Extend the current subdivided-cube and cube-cloth fixtures into `pt_lift_release` once bonded state, counters, and release instrumentation exist |
| No subsystem timers | Performance claims would collapse into total frame time | Add or expose timing fields before benchmarks |

## Playbook Compliance

| Minimum Standard | Status | Evidence Or Required Work |
| --- | --- | --- |
| Current roadmap names principle, phase, next tasks, blockers, and gates | Satisfied | This file is the current status surface |
| Stable architecture and conventions are documented | Satisfied | [architecture](./architecture.md) and [conventions](./conventions.md) |
| Runnable gates and planned gates are separated | Satisfied | Current gates are below; future commands are in planned gates |
| New tests are wired into a default validation path | Partially satisfied | Portable docs/source gates are wired through `run_rcc_adhesion_acceleration_all_gates.py`; local CUDA gates are wired through `run_rcc_adhesion_acceleration_cuda_gates.py`; legacy RCC scene gates are wired into pytest and `sim_case`; bonded-PT state, rest-shape payload, counter, state accessor, CPU oracle, CUDA state bridge, CUDA lookup, common active-filter, CUDA owner, beta-threshold producer with live rest-shape rejection, GPU virtual-tet reporter oracle, and bunny BVH regression fixtures are implemented; bonded-PT release, pre-CCD filter, and scene/benchmark gates are still planned |
| Numeric or algorithmic claims have CPU or legacy oracles | Satisfied for current scope | State, rest-shape, virtual-tet E/G/H CPU gates, and a GPU reporter E/G/H matching gate exist |
| Benchmark claims split cold, cache-hot, churn, and end-to-end timing | Not yet implemented | Phase 5 requires benchmark script, timers, and correctness fields |
| Journals record commands, observed results, and decisions | Satisfied | [journal](./development/rcc_adhesion_acceleration_journal.md) |

## Validation Gates

These commands are runnable today from the repository root and must pass before changing the roadmap status.

| Gate | Command | Required Result |
| --- | --- | --- |
| Portable docs/source gates | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Runs source/doc checks, Python syntax checks, and docs build; exits 0 |
| Source/doc boundary gate | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Confirms doc skeleton, nav link, all-gates entry, and current source anchors |
| Python syntax gate | `uv run --no-sync python -m py_compile scripts/run_rcc_adhesion_acceleration_gates.py scripts/run_rcc_adhesion_acceleration_all_gates.py scripts/run_rcc_adhesion_acceleration_cuda_gates.py scripts/build_docs.py` | Exits 0 |
| Docs site build | `uv run --no-sync python scripts/build_docs.py -o /tmp/libuipc-docs-check` | MkDocs and MkDoxy build the docs and API pages |
| Local RCC CUDA gate bundle | `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py --no-build` | Runs bonded-PT core/backend tests and the bunny GPU sanity regression against an existing CUDA build |
| Bonded PT state and counters | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Deterministic zipped key/topology/beta/age/release/rest-shape fixture and counter fixture pass |
| Bonded PT CPU oracles | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle]" -r compact` | Rest-shape conditioning plus virtual-tet energy, gradient, and Hessian CPU oracles pass |
| Bonded PT CUDA state bridge | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]" -r compact` | Host state, device buffers, rest-shape payloads, pending release flags, extract-release counters, and clear behavior roundtrip |
| Bonded PT state accessor | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][accessor]" -r compact` | Frontend feature wrapper reports locked count, counters, and state snapshots through the overrider contract |
| Bonded PT CUDA lookup helper | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lookup]" -r compact` | RCC PT key semantics, sorted membership lookup, miss handling, and empty locked-set behavior pass on CUDA |
| Bonded PT common active-filter compact | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter]" -r compact` | Locked PTs are removed from `SimplexTrajectoryFilter::PTs()` and therefore from `friction_PTs()` when sorted locked keys are supplied |
| Bonded PT CUDA owner and beta/rest producer | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` | Owner uploads locked state, feeds sorted keys to the filter, syncs skip counters, and device-side producer carries existing locks, adds high-beta candidates, increments age, suppresses duplicate candidate keys, builds live rest shapes for fresh locks, and rejects degenerate fresh locks |
| Bonded PT GPU virtual-tet reporter oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][reporter][oracle]" -r compact` | Dynamic complement reporter math matches the CPU virtual-tet energy, gradient, and Hessian oracle within tolerance |
| CUDA bunny BVH/radix-sort regression | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Ensures bonded-PT CUDA payload/layout changes do not destabilize the existing `SimplicialSurfaceDistanceCheck` + `InfoStacklessBVH` path |
| Legacy RCC Python lift/release scenes | `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Subdivided cube-cube and cube-cloth lift/hold/release fixtures pass |
| Legacy RCC C++ lift/release scenes | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Two native scene gates pass after the sim case target is built |

## Planned Gates

These commands are target gates for missing code, missing tests, or missing system dependencies. They are not current proof.

| Gate | Target Command | Required Result | Missing Piece |
| --- | --- | --- | --- |
| Pre-CCD filter contract | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]"` | Locked PT key is absent from PT candidate and TOI paths in every simplex filter backend | Wire the lookup helper before PT CCD broadphase in all simplex filters and add candidate/TOI instrumentation |
| PT lift/release scene | `build/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"` | PT-rich fixture locks during press/hold, adhered geometry follows during sub-threshold lift, forced pull releases and separates, adhesion-off baseline does not lift, beta carry and zero duplicate ownership are reported | Add bonded-PT implementation, report counters, release instrumentation, and assertion-based scene |
| Benchmark matrix | `uv run --no-sync python scripts/bench_rcc_adhesion_acceleration.py --scene stable_cloth_peel --frames 40 --warmup 5 --runs 10` | Reports cold, cache-hot, churn, and end-to-end medians with correctness fields | Add benchmark script, timers, and report parser |

## Next Safe Task

Expose live bonded-PT counters/release diagnostics through the reporting path, then implement release gates that carry beta back to RCC persistence. Keep bonded mode default-off and avoid correctness/performance claims until release, scene, report counters, pre-CCD filtering, and benchmark gates all pass.
