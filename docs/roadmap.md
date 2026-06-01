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

## Phase 1: State Contract And CPU Oracle (Current)

### State Owner

- [x] Implement `RCCBondedPTState` minimum host contract for locked keys, oriented topologies, beta, age, and release flags.
- [ ] Extend the state owner with rest-shape metrics once the SVTS oracle lands.
- [ ] Define feature-disabled default behavior and explicit config keys under the `rcc_bonded_pt` prefix.
- [ ] Add debug/report counters for candidate, locked, released, degenerate-rejected, filter-skipped, and duplicate-suppressed pairs.

### Oracles

- [x] Add a deterministic two-pair state fixture: one pair stays locked, one pair releases.
- [x] Add a CPU rest-shape oracle matching `SoftVertexTriangleStitch` construction, including `min_separate_distance`.
- [ ] Add a CPU Stable Neo-Hookean E/G/H oracle for a single bonded PT virtual tet.

### Default Validation

- [x] Wire the state fixture into the current validation path.
- [x] Wire the rest-shape CPU oracle into the current validation path.
- [ ] Wire the Stable Neo-Hookean E/G/H CPU oracle into the default validation path.
- [ ] Keep source scans as boundary checks only; do not use them as proof of math.

## Phase 2: Filter Integration

- [ ] Add one shared device helper for locked PT membership lookup.
- [ ] Use the helper in stackless BVH, info stackless BVH, v0 info stackless BVH, and LBVH simplex filters.
- [ ] Prove locked PTs are absent before `record_friction_candidates()` copies `PTs()` into `friction_PTs()`.
- [ ] Add duplicate-accounting diagnostics for pair ownership.

## Phase 3: Bonded Virtual-Tet Reporter

- [ ] Implement dynamic complement-energy reporter for bonded PT virtual tets.
- [ ] Store `Dm_inv`, rest volume, material parameters, and oriented topology in device buffers.
- [ ] Match CPU oracle energy, gradient, and Hessian within declared tolerances.
- [ ] Keep the reporter out of contact-component accounting unless an explicit diagnostic requests comparison.

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
| No backend-owned bonded PT state integration | A host `RCCBondedPTState` contract exists, but filter/reporter work has no live CUDA-owned source of locked keys | Finish rest-shape/oracle work, then mirror the contract into the CUDA backend |
| No CPU oracle | Numeric correctness cannot be separated from GPU implementation bugs | Add rest-shape and E/G/H oracle fixtures |
| No bonded-PT runtime counters | Current legacy RCC scene gates can prove lift/release behavior, but not bonded-pair ownership | Add `rcc_bonded_pt_*` counters and release flags |
| No bonded-PT PT lifecycle scene | Legacy RCC lift/hold/release fixtures are automated; bonded mode still lacks lock/reuse/release assertions and report fields | Extend the current subdivided-cube and cube-cloth fixtures into `pt_lift_release` once bonded state, counters, and release instrumentation exist |
| No subsystem timers | Performance claims would collapse into total frame time | Add or expose timing fields before benchmarks |

## Playbook Compliance

| Minimum Standard | Status | Evidence Or Required Work |
| --- | --- | --- |
| Current roadmap names principle, phase, next tasks, blockers, and gates | Satisfied | This file is the current status surface |
| Stable architecture and conventions are documented | Satisfied | [architecture](./architecture.md) and [conventions](./conventions.md) |
| Runnable gates and planned gates are separated | Satisfied | Current gates are below; future commands are in planned gates |
| New tests are wired into a default validation path | Partially satisfied | Portable docs/source gates are wired through `run_rcc_adhesion_acceleration_all_gates.py`; legacy RCC scene gates are wired into pytest and `sim_case`; the bonded-PT state fixture is implemented; bonded-PT CPU oracle/filter gates are not implemented |
| Numeric or algorithmic claims have CPU or legacy oracles | Partially satisfied | State and rest-shape CPU gates exist; Phase 1 still requires virtual-tet E/G/H CPU oracle |
| Benchmark claims split cold, cache-hot, churn, and end-to-end timing | Not yet implemented | Phase 5 requires benchmark script, timers, and correctness fields |
| Journals record commands, observed results, and decisions | Satisfied | [journal](./development/rcc_adhesion_acceleration_journal.md) |

## Validation Gates

These commands are runnable today from the repository root and must pass before changing the roadmap status.

| Gate | Command | Required Result |
| --- | --- | --- |
| Portable docs/source gates | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Runs source/doc checks, Python syntax checks, and docs build; exits 0 |
| Source/doc boundary gate | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Confirms doc skeleton, nav link, all-gates entry, and current source anchors |
| Python syntax gate | `uv run --no-sync python -m py_compile scripts/run_rcc_adhesion_acceleration_gates.py scripts/run_rcc_adhesion_acceleration_all_gates.py scripts/build_docs.py` | Exits 0 |
| Docs site build | `uv run --no-sync python scripts/build_docs.py -o /tmp/libuipc-docs-check` | MkDocs and MkDoxy build the docs and API pages |
| Bonded PT state contract | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Deterministic zipped key/topology/beta/age/release fixture passes |
| SVTS rest-shape CPU oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][rest_shape]" -r compact` | Point-triangle rest-shape conditioning, orientation swap, `Dm_inv`, and rest volume match SVTS rules |
| Legacy RCC Python lift/release scenes | `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Subdivided cube-cube and cube-cloth lift/hold/release fixtures pass |
| Legacy RCC C++ lift/release scenes | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Two native scene gates pass after the sim case target is built |

## Planned Gates

These commands are target gates for missing code, missing tests, or missing system dependencies. They are not current proof.

| Gate | Target Command | Required Result | Missing Piece |
| --- | --- | --- | --- |
| CPU virtual-tet E/G/H oracle | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][oracle]"` | GPU bonded-tet E/G/H match CPU oracle within tolerance | Add reporter and E/G/H oracle fixture |
| Filter contract | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter]"` | Locked PT key is absent from `PTs()` and `friction_PTs()` in every simplex filter backend | Add locked-key lookup and instrumentation |
| PT lift/release scene | `build/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"` | PT-rich fixture locks during press/hold, adhered geometry follows during sub-threshold lift, forced pull releases and separates, adhesion-off baseline does not lift, beta carry and zero duplicate ownership are reported | Add bonded-PT implementation, report counters, release instrumentation, and assertion-based scene |
| Benchmark matrix | `uv run --no-sync python scripts/bench_rcc_adhesion_acceleration.py --scene stable_cloth_peel --frames 40 --warmup 5 --runs 10` | Reports cold, cache-hot, churn, and end-to-end medians with correctness fields | Add benchmark script, timers, and report parser |

## Next Safe Task

Implement the Stable Neo-Hookean virtual-tet E/G/H CPU oracle next. Do not touch filter kernels until there is an authoritative locked-key/topology state owner and CPU oracles for rest-shape and virtual-tet E/G/H construction.
