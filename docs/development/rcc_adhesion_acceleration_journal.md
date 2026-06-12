# RCC Adhesion Acceleration Journal

This journal records decisions, commands, and observed results for the bonded point-triangle RCC acceleration project.

## 2026-05-30 Documentation Gate

### Context

The design target is to accelerate stable RCC adhesion pairs by removing long-lived point-triangle pairs from the IPC/RCC contact pipeline and replacing them with a bonded virtual tetrahedron over the same four global vertex degrees of freedom.

### Source Observations

- `IPCSimplexRCCAdhesiveContact` already owns PT beta buffers (`m_beta_PT`, `m_prev_keys_PT`) and updates them through `_evolve_beta_step_at_end()`.
- `RCCBetaEvolutionTimeIntegrator` is the existing end-of-step hook for beta evolution.
- `SimplexTrajectoryFilter::record_friction_candidates()` copies active `PTs()` into `friction_PTs()`, so locked pairs must be removed before this copy.
- PT CCD broadphase calls exist in the stackless, info-stackless, v0 info-stackless, and LBVH simplex trajectory filters.
- `SoftVertexTriangleStitch` already builds vertex-triangle tetrahedra, applies `min_separate_distance`, stores `Dm_inv` and rest volume, and uses SPD-projected Stable Neo-Hookean Hessians. The rest-shape convention remains useful; the energy-model choice was later corrected to ABD-style high stiffness.
- Inter-primitive constitutions report complement energy, which is the right ownership model for a bonded virtual tet that replaces contact/RCC work but is not contact itself.

### Decisions

- Treat `SoftVertexTriangleStitch` as the rest-shape/thickness reference, not as the runtime container. Dynamic adhesion pairs should not rebuild frontend geometry. Its Stable Neo-Hookean energy is not the final bonded-PT replacement model.
- Store sorted membership keys for filtering and oriented topologies for tet energy.
- Make early filter skip a production requirement because late removal does not save the expensive PT CCD/contact path.
- Keep the first executable gate lightweight and always runnable: source/doc anchors only. Numeric and performance gates are planned separately.

### Commands

| Command | Result |
| --- | --- |
| `python3 scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Initial non-uv check before the development environment was switched to uv commands. |
| `python3 -m py_compile scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Initial non-uv syntax check before the development environment was switched to uv commands. |
| `command -v mkdocs && mkdocs --version` | Failed because `mkdocs` was not installed in this environment. Docs build was not counted as a current validation gate. |
| `./scripts/run_rcc_adhesion_acceleration_gates.py` | Passed after marking the script executable. |

## 2026-05-30 UV Docs Environment

### Context

The development workflow should use the repository root uv environment.

### Commands

| Command | Result |
| --- | --- |
| `uv pip list --python ./.venv/bin/python \| rg 'mkdocs\|mkdoxy\|properdocs'` | Passed. `mkdocs`, `mkdocs-material`, `mkdocs-literate-nav`, `mkdocs-video`, `mkdoxy`, and `properdocs` are installed in `./.venv`. |
| `command -v doxygen || true; doxygen --version 2>/dev/null || true` | `doxygen` is not installed on `PATH`. |
| `uv run --no-sync python scripts/build_docs.py -o /tmp/libuipc-docs-check` | Failed in MkDoxy with `Invalid Doxygen binary path: doxygen`. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Current source/doc gate runs through uv. |
| `uv run --no-sync python -m py_compile scripts/run_rcc_adhesion_acceleration_gates.py scripts/build_docs.py` | Passed. Current Python scripts compile through uv. |
| `uv run --no-sync python scripts/build_docs.py -o /tmp/libuipc-docs-check` after fixing `scripts/build_docs.py` | Failed with exit code 1 because `doxygen` is still missing. This confirms the script now propagates build failure. |

### Decision

`scripts/build_docs.py` now exits with the MkDocs return code, so documentation build failures can be used as strict gates once `doxygen` is installed.

## 2026-05-30 Strict Playbook Rewrite

### Problem addressed

- The first documentation pass was useful but did not strictly follow the playbook's lifetime split.
- Architecture, conventions, and status lived under `docs/development/`, which made them look like journal material instead of stable handoff surfaces.
- There was no single all-gates entry point for current validation.

### Implemented

- Created top-level [roadmap](../roadmap.md), [architecture](../architecture.md), [conventions](../conventions.md), and [subsystem](../rcc_adhesion_acceleration.md) docs.
- Kept this journal as the only chronological `docs/development/` page for the feature.
- Removed obsolete parallel development docs for status, architecture, and conventions.
- Added `scripts/run_rcc_adhesion_acceleration_all_gates.py` as the default current-gates entry point.
- Tightened `scripts/run_rcc_adhesion_acceleration_gates.py` to enforce the playbook skeleton, nav links, all-gates entry, obsolete-doc removal, and source anchors.

### Validation

| Check | Result |
| --- | --- |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Checked strict playbook skeleton, nav links, all-gates entry, obsolete-doc removal, and current source anchors. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Ran source/doc gate and Python syntax gate through the default current-gates entry point. |

### Decision

- The docs now satisfy the implemented parts of the playbook skeleton. Numeric oracles, C++/CUDA tests, scene gates, and benchmark gates remain explicitly not implemented in the roadmap.

### Next

- Run current uv gates, update this entry with observed results, then keep Phase 1 focused on the state contract and CPU oracle.

### Next

Implement `RCCBondedPTState` and the CPU/GPU state fixture before touching filter kernels.

## 2026-06-01 PT Scene Gate Update

### Context

The roadmap had a planned lifecycle scene gate, but it did not name the concrete real-scene behavior expected from the bonded PT acceleration. The first gate should validate stable point-triangle adhesion on/off behavior, not merely a visually plausible two-body animation.

### Decisions

- Promote the first lifecycle scene target to `pt_lift_release`.
- Require a PT-rich fixture under gravity, an adhesion-off baseline, COM/patch-height follow assertions during sub-threshold lift, release/separation assertions during forced pull, beta-carry reporting, and zero duplicate ownership.
- Treat adhesion-off as RCC adhesive contact disabled; current RCC adhesion with bonded PT disabled is a separate legacy baseline for correctness comparison.
- Prefer a seed based on `rcc_adhesion_cloth_peel`, `python/examples/rcc_adhesive_oriented_cloth_demo.py`, or a patch-on-cube variant of `python/examples/rcc_adhesive_pick_and_lift_demo.py`; do not require a closed cube-cube setup.
- Treat OBJ output as debugging evidence only; the gate must fail automatically from simulation state and report fields.
- Move docs site build into current gates now that `doxygen` is available on `PATH`.

### Commands

| Command | Result |
| --- | --- |
| `doxygen --version` | Passed. Reported `1.9.8`. |
| `uv run --no-sync python scripts/build_docs.py -o /tmp/libuipc-docs-check` | Passed. MkDocs/MkDoxy built the docs and API pages. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Ran source/doc checks, Python syntax checks, and docs site build. |

## 2026-06-01 Existing Pick-And-Lift Probe

### Context

Before implementing bonded PT acceleration, the existing `rcc_adhesion_pick_and_lift` demo was run to see whether it can serve as the first concrete `pt_lift_release` scene seed.

### Commands

| Command | Result |
| --- | --- |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case --list-tests "[rcc_bonded_pt]"` | Passed. Reported 0 matching test cases, so `pt_lift_release` does not exist yet. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "rcc_adhesion_pick_and_lift" -r compact` | Passed. Ran both `lift_with_adhesion` and `lift_without_adhesion` sections and wrote OBJ sequences. |
| OBJ center-height scan at frames 0, 60, 100, 150, 200, 260 | `lift_with_adhesion` and `lift_without_adhesion` had matching bottom-cube center heights; frame 260 bottom center was about `0.169626` in both sections while top center was about `0.900000`. |
| Temporary C++ retune to match the Python pick-and-lift bottom height | Built and ran, but still produced matching adhesion-on/off bottom heights. The source retune was not retained. |

### Decision

The existing cube-cube pick-and-lift demo is useful as a harness shape, but it is not a valid seed for the lifecycle gate as-is. RCC adhesion assembly appears in the log, but the geometry still does not produce a stable on/off lift difference. Prefer a PT-friendly contact patch or explicit vertex-triangle fixture over a symmetric face-face cube setup.

## 2026-06-01 PT-Friendly Scene Seed Probe

### Context

The Python `rcc_adhesive_pick_and_lift_demo.py` and `rcc_adhesive_oriented_cloth_demo.py` examples are known to produce visible adhesion effects when run directly. The C++ seed should mirror that PT-friendly behavior closely enough to become an automated gate.

### Commands

| Command | Result |
| --- | --- |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "rcc_adhesion_cloth_peel" -r compact` | Passed. Ran both `peel_with_adhesion` and `peel_without_adhesion` sections and wrote OBJ sequences. |
| Log scan for PT activity | Passed. The run reported nonzero `SimplexTrajectoryFilter PTs` and friction PT candidates, so the scene exercises point-triangle contact paths. |
| OBJ average-height scan at frames 0, 50, 110, 180, 250, 300 | `peel_with_adhesion` kept the two cloth patches close and lifted them together: frame 300 averages were about `0.194587` and `0.200415`. `peel_without_adhesion` separated strongly: frame 300 averages were about `-0.005566` and `0.206003`. |

### Decision

Use a PT-rich lift/release fixture for the first scene gate. The existing cloth-peel sim case is a better seed than the cube-cube pick-and-lift case because it already shows a stable adhesion-on/off difference and exercises PT filters. The production gate still needs explicit lock/reuse/release counters, beta-carry checks, and pass/fail assertions rather than OBJ post-processing.

## 2026-06-01 Pick-And-Lift Point-Density Probe

### Context

The Python `rcc_adhesive_pick_and_lift_demo.py` visibly responds to adhesion, but the required scene gate needs a numeric adhered-body lift difference, not just visible deformation. A point-density probe was run to check whether the coarse 8-corner cube is the limiting factor.

### Commands

| Command | Result |
| --- | --- |
| `timeout 12s python/.venv/bin/python python/examples/rcc_adhesive_pick_and_lift_demo.py` | Started the Polyscope/OpenGL demo successfully and entered the UI loop. |
| Headless import of `python/examples/rcc_adhesive_pick_and_lift_demo.py::build_demo` for adhesion-on/off, 260 frames | The original coarse Python demo did not lift the lower cube: frame 260 lower average height was `0.169626` for both adhesion-on and adhesion-off. Adhesion-on did visibly deform/stretch the upper cube: frame 260 upper range was about `[0.692324, 1.107676]` versus adhesion-off `[0.749995, 1.050005]`. |
| Temporary regular subdivided tet cube probe, grid `n=2`, 260 frames | Produced a stable lift difference. Frame 260 lower average height was `0.582975` with adhesion-on and `0.169751` with adhesion-off. |
| Temporary regular subdivided tet cube probe, grid `n=3`, 260 frames | Also produced a stable lift difference. Frame 260 lower average height was `0.578885` with adhesion-on and `0.169814` with adhesion-off. |
| Temporary face-only `+Y` surface-point hack, `FACE_N=2`, 260 frames | Also produced a strong on/off difference: frame 260 lower geometry range was about `[0.757229, 1.220090]` with adhesion-on and `[0.019626, 0.319626]` with adhesion-off. This construction manually appends surface triangles after labeling and is not a production fixture candidate. |

### Decision

Do not reject the pick-and-lift family outright. Reject only the coarse 8-corner cube as the gate seed. For a production `pt_lift_release` fixture, prefer a real subdivided tet cube or patch-on-cube geometry so the contact face has enough point-triangle samples while remaining a valid closed mesh. Avoid the face-only append hack in tests because it bypasses normal closure labeling and can create overlapping surface triangles.

## 2026-06-01 Legacy RCC Lift/Release Gates

### Context

The visual Python adhesive demos were promoted into assertion-based fixtures before implementing bonded PT acceleration. These gates preserve the current RCC adhesion behavior and give the future bonded implementation a concrete baseline: lift during stable adhesion, maintain near-complete contact before pull, and release/separate during a stronger pull.

### Implemented

- Added `python/examples/rcc_adhesive_subdivided_cube_lift_release_demo.py` for a point-dense cube-cube lift, hold, and pull sequence.
- Added `python/examples/rcc_adhesive_cube_cloth_lift_release_demo.py` for a cube pressing cloth, lifting it, then pulling one cloth point down to release.
- Added `python/tests/sim_case/test_rcc_adhesive_lift_release.py` with pytest assertions for cube-cube and cube-cloth hold/release behavior.
- Added native C++ sim-case gates in `apps/tests/sim_case/rcc_adhesion_lift_release_gate.cpp` with `[rcc_adhesion][gate][cuda]` tags.
- Stabilized the cube-cube gate around `adhesion_w = 3.8` so the hold stage has beta near one while the pull stage releases instead of leaving high-beta residual pairs.

### Commands

| Command | Result |
| --- | --- |
| `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Passed. Reported `2 passed in 44.74s`. |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed the sim-case source glob so the new C++ gate was included. |
| `cmake --build build/cuda_mixed_fused_pcg --target sim_case -j2` | Passed. Rebuilt the native sim-case target. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case --list-tests "[rcc_adhesion][gate]"` | Passed. Reported two matching tests: `rcc_adhesion_subdivided_cube_lift_hold_release_gate` and `rcc_adhesion_cube_cloth_lift_hold_release_gate`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Passed. Reported `All tests passed (836 assertions in 2 test cases)`. |

### Decision

Treat these as current legacy RCC behavior gates, not as proof of bonded PT acceleration. At this point the bonded path still needed `RCCBondedPTState`, key/topology/beta/age/release fixtures, SVTS-compatible rest-shape oracles, high-kappa ABD-style virtual tet E/G/H oracles, `rcc_bonded_pt_*` counters, and release reason fields before the final `pt_lift_release` gate could claim pair ownership correctness.

## 2026-06-01 Minimal Bonded PT State Contract

### Context

The first bonded PT implementation step needs a small state owner before any CUDA filter or reporter integration. The state fixture should prove that sorted keys stay zipped with oriented topologies, beta, age, and release flags.

### Implemented

- Added `RCCBondedPTState` with host-side arrays for `locked_keys`, `locked_topos`, `locked_beta`, `locked_age`, and `release_flags`.
- Added release flag constants and release extraction so released pairs carry their key, topology, beta, age, and release reason out of the active lock set.
- Added `rcc_bonded_pt_state_keeps_payloads_zipped` under `[rcc_bonded_pt][state]`.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed the core test source glob. |
| `cmake --build build/cuda_mixed_fused_pcg --target core -j2` | Passed. Built `uipc_core` and `uipc_test_core` with the new state contract. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Passed. Reported `All tests passed (29 assertions in 1 test case)`. |

### Decision

Keep this as a host-side contract for now. The next step is the SVTS-compatible rest-shape CPU oracle; CUDA device buffers and filter/reporter integration should wait until state, rest-shape, and production ABD-style E/G/H oracles are all executable.

## 2026-06-01 SVTS Rest-Shape CPU Oracle

### Context

The bonded PT virtual tet must build the same rest shape as `SoftVertexTriangleStitch`: condition the point-triangle separation with `min_separate_distance`, orient the tetrahedron to positive determinant, and produce `Dm_inv` plus positive rest volume.

### Implemented

- Added `build_rcc_bonded_pt_rest_shape_svts()` as a CPU reference for SVTS-compatible point-triangle rest-shape construction.
- Added tests for the plane-degenerate point case that requires `min_separate_distance` offset and tri0/tri1 orientation swap.
- Added a degenerate-triangle rejection test.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg && cmake --build build/cuda_mixed_fused_pcg --target core -j2` | Failed initially at link because the new oracle used Eigen `cross`, `determinant`, and `inverse` without explicit `Eigen/Geometry` and `Eigen/LU` includes. |
| `cmake --build build/cuda_mixed_fused_pcg --target core -j2` after adding explicit Eigen includes | Passed. Built `uipc_core` and `uipc_test_core`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][rest_shape]" -r compact` | Passed. Reported `All tests passed (11 assertions in 2 test cases)`. |

### Decision

Use this rest-shape oracle as the reference for future dynamic bonded PT lock construction. The next historical implementation step used a Stable Neo-Hookean virtual-tet oracle, but the production plan was later corrected to require ABD-style high-kappa energy.

## 2026-06-01 Prototype Stable Neo-Hookean Virtual Tet E/G/H CPU Oracle

### Context

After rest-shape construction is deterministic, the bonded PT reporter needs a CPU reference for the virtual tet complement energy. This historical slice matched the `SoftVertexTriangleStitch` Stable Neo-Hookean path, including `F = Ds * Dm_inv`, `dFdx`, `rest_volume * dt^2` scaling, and SPD projection of the F-space Hessian for the default Hessian path. It is now classified as a prototype oracle, not the production acceptance oracle.

### Implemented

- Added `build_rcc_bonded_pt_virtual_tet_oracle()` for CPU energy, 12-vector gradient, and 12x12 Hessian.
- Used the SVTS simplified Stable Neo-Hookean energy density, not the separate log-form `StableNeoHookean3D` formula.
- Added finite-difference checks for gradient and raw Hessian, plus a default projected-Hessian positive-semidefinite check.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target core -j2` | Failed initially because `Matrix9x12` is a CUDA-backend-local alias and was not visible in core. Fixed by using a local `Matrix<Float, 9, 12>` alias in the CPU oracle implementation. |
| `cmake --build build/cuda_mixed_fused_pcg --target core -j2` after the alias fix | Passed. Built `uipc_core` and `uipc_test_core`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][energy]" -r compact` | Passed. Reported `All tests passed (9 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle]" -r compact` | Passed. Reported `All tests passed (20 assertions in 3 test cases)`. |

### Decision

The CPU oracle layer was sufficient for the next prototype implementation step. After the 2026-06-02 ABD correction, it is no longer sufficient for production bonded-mode claims; an ABD-style high-kappa oracle must replace or supersede it.

## 2026-06-01 Minimum Bonded PT Counters

### Context

The bonded PT path needs reportable counters before filter or reporter integration. Without candidate, lock, release, rejection, skip, and duplicate accounting, scene gates cannot prove that a PT pair is owned by exactly one path.

### Implemented

- Added `RCCBondedPTCounters` to the host state contract with `candidate_count`, `locked_count`, `released_count`, `degenerate_rejected_count`, `filter_skipped_count`, and `duplicate_suppressed_count`.
- Kept `locked_count` synchronized with the active lock set and incremented `released_count` when released entries are extracted.
- Added explicit record methods for candidate, degeneracy rejection, filter skip, and duplicate suppression events.
- Added `[rcc_bonded_pt][state][counters]` assertions.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target core -j2` | Passed. Built `uipc_core` and `uipc_test_core`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Passed. Reported `All tests passed (48 assertions in 2 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state][counters]" -r compact` | Passed. Reported `All tests passed (16 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (68 assertions in 5 test cases)`. |

### Decision

The requested pre-filter pieces are now present in host-side contracts and CPU oracles. The next implementation step should create the CUDA-owned state/reporting bridge before any simplex filter skips are enabled.

## 2026-06-01 CUDA Bonded PT State Bridge

### Context

After the host state, counters, rest-shape oracle, and virtual-tet E/G/H oracle passed, the next safe step was to mirror the state contract into CUDA-owned buffers without changing filter behavior yet. This gives filter and reporter integration a deterministic device-side owner for locked keys, oriented topologies, beta, age, release flags, and counter snapshots.

### Implemented

- Added `RCCBondedPTStateBridge` under `src/backends/cuda/contact_system`.
- Mirrored `locked_keys`, `locked_topos`, `locked_beta`, `locked_age`, and `release_flags` into `muda::DeviceBuffer` storage.
- Added host upload/download roundtrip and clear behavior.
- Added `RCCBondedPTState::set_counters()` for backend-controlled counter restore while always re-syncing `locked_count` with active locks.
- Added `[rcc_bonded_pt][backend_state][cuda]` assertions covering pending release flags, extracted release counters, and device buffer extent alignment.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed CUDA backend and backend test source globs. |
| `cmake --build build/cuda_mixed_fused_pcg --target core backend_cuda -j2` | Passed. Built `uipc_core`, `uipc_test_core`, `libuipc_backend_cuda`, and `uipc_test_backend_cuda`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Passed. Reported `All tests passed (54 assertions in 2 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]" -r compact` | Passed. Reported `All tests passed (44 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (74 assertions in 5 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (44 assertions in 1 test case)`. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc anchors now include the CUDA bridge files and the docs site builds successfully. |

### Decision

The CUDA bridge is now a tested state transport layer, not a live simulation owner. The next step should add the shared locked-key membership lookup helper and a CUDA filter contract fixture while keeping production filter skipping disabled until live `rcc_bonded_pt_*` reports are available.

## 2026-06-01 CUDA Bonded PT Lookup Helper

### Context

Before any simplex filter skips are enabled, all filter backends need one shared membership primitive so the RCC PT key semantics cannot drift. The existing RCC beta persistence key keeps the point id distinct and sorts only the three triangle vertices, so the bonded-PT lookup helper must reuse that exact key.

### Implemented

- Added `rcc_bonded_pt_lookup.h` under `src/backends/cuda/contact_system`.
- Wrapped the existing RCC PT key function for host/device use.
- Added inline sorted-key lower-bound and `is_locked` helpers over `muda::CBufferView<U64>`.
- Added `[rcc_bonded_pt][lookup][cuda]` assertions covering triangle-vertex permutation, point-id distinction, sorted lookup hits, miss handling, topology alignment through the bridge, and empty locked-set behavior.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed the backend CUDA test source glob. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` | Failed initially because the RCC adhesive function header was not self-contained for `muda::CDense2D`, and the helper used viewer-style `operator()` on a raw `CBufferView`. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` after adding the missing dense viewer include and using `CBufferView::operator[]` | Passed. Built `uipc_test_backend_cuda`; only the pre-existing unused `xi2` warning in the adhesive header remained. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lookup]" -r compact` | Passed. Reported `All tests passed (17 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (61 assertions in 2 test cases)`. |

### Decision

The lookup primitive is ready as a contract for filter integration, but it still does not change live simulation behavior. The next safe step is to feed this helper into the PT-producing simplex filters with instrumentation proving that locked keys disappear before `friction_PTs()` is recorded.

## 2026-06-01 Common Active PT Filter Compact

### Context

All four simplex filter backends pass their active PT view through `SimplexTrajectoryFilter` before `record_friction_candidates()` copies it into `friction_PTs()`. A common compact at that layer can prove pair ownership for active/contact/RCC assembly without duplicating code in stackless BVH, info stackless BVH, v0 info stackless BVH, and LBVH.

This is still later than the final performance target. It does not skip PT candidate generation or PT CCD broadphase yet.

### Implemented

- Added a sorted locked-key view to `SimplexTrajectoryFilter::Impl`, defaulting to an empty no-op.
- Added `filter_rcc_bonded_pt_locked_active_pairs()` using the shared lookup helper plus CUB `DeviceSelect`.
- Replaced active `PTs()` with the unlocked compacted view before `record_friction_candidates()`.
- Added an exposed skip count for the common active compact path.
- Added `[rcc_bonded_pt][filter][cuda]` assertions proving two locked PTs are removed from active `PTs()` and remain absent from `friction_PT` after the friction candidate copy.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` | Failed initially because CUB extended host/device lambdas do not allow init-capture. Fixed by capturing a local `locked_keys` variable. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` after the capture fix | Failed in the new test because `SimplexTrajectoryFilter` needs the backend-common include root. Fixed by adding `${PROJECT_SOURCE_DIR}/src` to the backend CUDA test target include path. |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed the backend CUDA test target after the include-path change. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` | Passed. Built `libuipc_backend_cuda` and `uipc_test_backend_cuda`. Existing CUDA warnings remained in unrelated include chains. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter]" -r compact` | Passed. Reported `All tests passed (11 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lookup]" -r compact` | Passed. Reported `All tests passed (17 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (72 assertions in 3 test cases)`. |

### Decision

The common active/contact path now has a tested locked-PT compact hook, but live simulation still needs a bonded-PT owner to feed sorted keys into it. The next safe step is live owner wiring and report counters; after that, move the same membership check earlier into the concrete PT candidate/TOI paths to get the intended CCD broadphase speedup.

## 2026-06-01 CUDA Bonded PT Owner

### Context

The CUDA bridge and common active-filter compact were still disconnected: tests could feed sorted locked keys by hand, but no runtime owner held the bridge or owned counter synchronization. The next step was to add a small backend owner while keeping the feature disabled by default and avoiding fake lock generation.

### Implemented

- Added `rcc_bonded_pt_enabled` to the default scene config with default `0`.
- Added `RCCBondedPTSystem` under `src/backends/cuda/contact_system`.
- The owner holds `RCCBondedPTStateBridge`, keeps a host counter snapshot, binds an optional `SimplexTrajectoryFilter`, feeds sorted locked keys when enabled, and syncs common active-filter skip counts back into counters.
- Added `[rcc_bonded_pt][owner][cuda]` assertions covering upload, key feed, filter compact, skip-counter sync, download, and clear behavior.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Failed inside the sandbox because vcpkg could not write `/home/zhaofeng/work/vcpkg/buildtrees/vcpkg-running.lock`. |
| `cmake -S . -B build/cuda_mixed_fused_pcg` with filesystem escalation | Passed. Refreshed source globs for the new owner and test files. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` | Passed. Built core, CUDA backend, and backend CUDA tests; only existing local architecture warnings appeared. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` in the sandbox | Failed with `cudaErrorNoDevice`; the sandbox could not see the GPU. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` with GPU escalation | Passed. Reported `All tests passed (12 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` with GPU escalation | Passed. Reported `All tests passed (84 assertions in 4 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (74 assertions in 5 test cases)`. |

### Decision

The live owner plumbing now exists, but it still needs a real producer. The next safe step is to connect RCC end-of-step PT beta/age/rest-shape gates to `RCCBondedPTSystem` so the owner receives real locked pairs, release flags, and counters.

## 2026-06-01 RCC Phase A Beta Producer

### Context

The CUDA owner could hold bonded PT state, but live RCC still did not populate it. The first live producer slice should avoid host roundtrips and avoid claiming release or virtual-tet correctness before those systems exist.

### Implemented

- Added `rcc_bonded_pt_beta_lock_threshold` to the default scene config with default `1.0`.
- Added a zipped `RCCBondedPTDeviceEntry` path so `RCCBondedPTStateBridge` can replace its device buffers from sorted GPU entries.
- Added `RCCBondedPTSystem::lock_from_rcc_pt_snapshot()`.
- The producer compacts RCC Phase A PT pairs whose evolved beta meets the threshold, suppresses duplicate candidate keys, refreshes existing locks, carries prior locks that are absent from the current friction list, increments age, and feeds the updated sorted keys back to `SimplexTrajectoryFilter`.
- Wired `IPCSimplexRCCAdhesiveContact::_evolve_beta_step_at_end()` to call the producer when `rcc_bonded_pt_enabled` is true.
- Added `[rcc_bonded_pt][owner][producer][cuda]` assertions covering refresh, carry, new lock, low-beta rejection, duplicate suppression, age update, and active PT filtering from produced keys.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` | Failed initially because the kernel captured a dense viewer where the shared lower-bound helper expects `CBufferView`; fixed by capturing both the raw view and viewer. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` after the capture fix | Failed in the new test because `Approx` was unqualified; fixed by using `Catch::Approx`. |
| `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda -j2` after the test fix | Passed. Built `libuipc_backend_cuda` and `uipc_test_backend_cuda`; only existing local architecture warnings and unrelated CUDA warnings appeared. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` with GPU escalation | Passed. Reported `All tests passed (32 assertions in 2 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` with GPU escalation | Passed. Reported `All tests passed (104 assertions in 5 test cases)`. |

### Decision

The live owner now has a device-side beta producer, but bonded mode is still not a correctness-complete acceleration. Existing locks are intentionally carried until release logic lands; the next safe steps are rest-shape storage/quality gates, release reason accounting, and an ABD-style bonded virtual-tet reporter.

## 2026-06-02 CUDA BVH Regression Gate

### Context

After adding rest-shape payloads to bonded PT state, the local tape-winding demo aborted with `cudaErrorIllegalAddress` during a later GPU sanity/BVH path. The failing stack pointed at `SimplicialSurfaceDistanceCheck` and `InfoStacklessBVH`, but the known-good commit `f6985e2729ebc79f94087648871f70a7bb4ecbf7` and committed branch head both passed the bunny sanity check in a clean temporary worktree. The regression was introduced by the uncommitted bonded-PT payload changes, not by the old BVH implementation.

### Root Cause

`RCCBondedPTDeviceEntry` had grown to include `Matrix3x3 Dm_inv` and `rest_volume`. The CUDA producer sorts bonded PT candidates with `thrust::sort_by_key`, so CUB treated the entire entry as the radix-sort value. In a clean repro this failed device link with a CUB onesweep radix-sort kernel exceeding the shared-memory limit; in the current build it surfaced later as an illegal address in the BVH sanity path.

### Implemented

- Kept `RCCBondedPTDeviceEntry` compact: key, oriented topology, beta, age, and release flags only.
- Kept `Dm_inv` and `rest_volume` in separate SoA buffers owned by `RCCBondedPTStateBridge`.
- During bridge replacement, copied the previous sorted key/rest-shape buffers before resizing and recovered existing rest-shape payloads by key; at this point fresh locks still defaulted to identity `Dm_inv` and zero rest volume until live rest-shape construction landed.
- Added `scripts/run_rcc_adhesion_acceleration_cuda_gates.py`, defaulting to `-j8`, to run the local RCC CUDA validation bundle.
- Added a source/doc gate that rejects `Matrix3x3`, `Dm_inv`, or `rest_volume` inside `RCCBondedPTDeviceEntry`.
- Promoted `uipc_test_backend_cuda "gpu_sanity_check" -c "bunny"` to a current CUDA regression gate.

### Commands

| Command | Result |
| --- | --- |
| `/tmp/libuipc-f698-build-nosync/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` at `f6985e2729ebc79f94087648871f70a7bb4ecbf7` | Passed. Baseline reported `All tests passed (4 assertions in 1 test case)`. |
| Same bunny gate at committed branch head `777323b0` in the clean temporary worktree | Passed. This isolated the regression to current uncommitted changes or local build products. |
| Temporary build with the uncommitted fat `RCCBondedPTDeviceEntry` patch applied | Failed at CUDA device link: CUB radix-sort value type used too much shared data. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` after compacting the device entry | Passed. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Passed. Reported `Distance(PT): CPU=0, GPU=275` and `All tests passed (4 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (120 assertions in 5 test cases)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8` (`ninja: no work to do`), then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc gate, Python syntax gate, and docs build all completed. |
| `python/.venv/bin/python` import smoke for `rcc_adhesive_tape_winding_demo.py --set SOLVER_PROFILE=quick` followed by `build_demo(True)` | Passed. Built the CUDA demo world at frame 0 with `valid=True`, confirming the startup crash path is gone. |

### Decision

Do not carry virtual-tet rest-shape matrices through sort values. Future bonded PT implementation should keep the hot classification/compact path key-centric and SoA-backed, then assemble/report virtual-tet energy from aligned SoA buffers after ownership is established. The bunny GPU sanity gate stays current because bonded PT CUDA changes can otherwise break unrelated CUB/BVH codegen paths.

## 2026-06-02 Live Producer Rest Shapes

### Context

After the CUDA BVH regression fix, the state and bridge could safely carry `Dm_inv` and rest volume, but fresh locks created by the live RCC Phase A producer still received placeholder rest-shape payloads. That blocked the bonded virtual-tet reporter because there was no real reference shape to consume.

### Implemented

- Added default scene config keys `rcc_bonded_pt_min_separate_distance = 1e-6` and `rcc_bonded_pt_det_dm_min = 1e-12`.
- Extended `RCCBondedPTSystem::lock_from_rcc_pt_snapshot()` to accept the current global positions from Phase A.
- Built SVTS-compatible lock-time rest shapes on CUDA for fresh high-beta PT locks, including point-plane offset, positive-orientation topology swap, `Dm_inv`, and rest volume.
- Rejected fresh high-beta candidates whose triangle/rest tet is degenerate and incremented `degenerate_rejected_count`.
- Kept radix-sort values compact: rest-shape payloads stay in SoA buffers keyed by sorted membership key.
- Preserved previous topology and rest-shape payloads for refreshed existing locks so a changed PT emission order cannot mismatch old `Dm_inv`.
- Updated the producer CUDA fixture to compare fresh GPU rest-shape output against the CPU SVTS oracle and to assert degenerate rejects stay unlocked.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` | Passed. Built the CUDA backend and backend test binary. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner][producer]" -r compact` | Passed after updating the expected fresh topology to the CPU oracle's positive orientation. Reported `All tests passed (30 assertions in 1 test case)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8`, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. Backend bonded-PT reported `All tests passed (124 assertions in 5 test cases)`. |

### Decision

The live producer now has enough rest-shape data for the next implementation slice: a bonded virtual-tet complement reporter plus a GPU-vs-CPU E/G/H oracle. The first implemented reporter used the SNH prototype path; production still requires ABD-style high-kappa energy. Remaining lock gates, release flags, scene counters, pre-CCD filtering, and benchmarks are still required before bonded mode can make correctness or performance claims.

## 2026-06-02 Prototype SNH Bonded Virtual-Tet Reporter Oracle

### Context

With live CUDA rest-shape construction in place, the next safe slice was to assemble a bonded PT virtual-tet complement energy from the existing owner buffers and compare the GPU math against the CPU virtual-tet E/G/H oracle. This implemented and validated the SNH prototype path; it does not satisfy the high-kappa ABD replacement requirement for CCD-skipped locked pairs.

### Implemented

- Added `RCCBondedPTVirtualTetReporter` as a `DyTopoEffectReporter` with `EnergyComponentFlags::Complement`.
- Added a config-gated creator so the reporter is only instantiated when `rcc_bonded_pt_enabled` is set and a dynamic topology manager is available.
- Added read-only locked-topology, `Dm_inv`, and rest-volume accessors on `RCCBondedPTSystem`.
- Added global material config keys `rcc_bonded_pt_mu` and `rcc_bonded_pt_lambda`, both defaulting to `0.0` so bonded owner/filtering can remain enabled without silently adding virtual-tet stiffness.
- Shared one CUDA evaluator between dense oracle buffers and runtime prototype doublet/triplet assembly.
- Added `[rcc_bonded_pt][reporter][oracle][cuda]` to compare GPU energy, gradient, and SPD-projected Hessian against the CPU oracle.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed CMake globs for the new CUDA reporter and backend test file. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` | Failed initially because the device evaluator guessed the internal muda viewer type. Fixed by templating the evaluator on the position viewer type. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` after the viewer fix | Passed. Built `libuipc_backend_cuda` and `uipc_test_backend_cuda`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][reporter][oracle][cuda]" -r compact` | Passed. Reported `All tests passed (5 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (129 assertions in 6 test cases)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8`, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc checks, Python syntax checks, and docs build all completed. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_sim_case -j8 && build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Passed. Reported `All tests passed (836 assertions in 2 test cases)`. |

### Decision

The prototype bonded virtual-tet reporter math is covered by a GPU-vs-CPU oracle, but the feature is still not correctness-complete and the energy model is not the production target. ABD-style high-kappa energy, release diagnostics, beta carry on release, live counter reporting, pre-CCD filtering, and bonded-mode scene/benchmark gates remain required before enabling or making performance claims.

## 2026-06-02 Bonded PT State Accessor

### Context

Host state and CUDA owner counters existed, but scene gates still needed a frontend-readable way to inspect live bonded PT ownership. The common active-filter skip counter also needed idempotent synchronization so repeated diagnostics could not double-count the same filter pass.

### Implemented

- Added `RCCBondedPTStateAccessorFeature` to expose `locked_pair_count()`, `counters()`, and `dump_state()`.
- Added a CUDA accessor SimSystem that inserts the feature and proxies to `RCCBondedPTSystem`.
- Added a filter generation counter to `SimplexTrajectoryFilter` and made owner skip-counter sync consume each generation once.
- Added a core accessor contract fixture and extended the CUDA owner fixture to assert repeated sync calls do not double-count filter skips.

### Commands

| Command | Result |
| --- | --- |
| `cmake -S . -B build/cuda_mixed_fused_pcg` | Passed. Refreshed CMake globs for the new core and CUDA accessor sources. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_core uipc_test_backend_cuda -j8` | Passed after fixing the accessor fixture to query counters once. Built core and CUDA backend test binaries. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (89 assertions in 6 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (130 assertions in 6 test cases)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8`, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. |

### Decision

Live counters and active locked-state snapshots are now accessible to frontend/native scene gates. Released history still needs its own diagnostics. The next missing correctness slice is a real release policy that sets release flags, carries beta back to RCC, and proves the lifecycle in a bonded-mode scene.

## 2026-06-02 ABD Energy Plan Correction

### Context

The bonded PT roadmap and implemented reporter had drifted toward using the `SoftVertexTriangleStitch` Stable Neo-Hookean formulas as the runtime replacement energy. That contradicts the original design requirement: once a locked PT pair skips CCD/contact/RCC, the four involved vertex DOFs must be held by a high-stiffness ABD-style bonded energy, with an initial target stiffness of `1e8` or higher in scene units.

### Findings

- The SVTS rest-shape convention is still useful for `min_separate_distance`, positive orientation, `Dm_inv`, and `rest_volume`.
- The current `rcc_bonded_pt_mu/lambda` reporter and oracle are only a prototype/regression path. They do not justify skipping CCD in a production bonded mode.
- The production energy should be assembled over `F = Ds Dm_inv` as an ABD-style virtual affine transform, preferably defaulting to `abd_ortho` with optional `abd_arap`.
- The implementation should assemble the ABD-style energy directly into the same four vertex-position DOFs through `dF/dx`; it should not create transient frontend ABD bodies or extra affine DOFs.

### Implemented

- Rewrote the architecture energy section to separate SVTS rest-shape construction from ABD-style runtime energy.
- Updated conventions with target config keys `rcc_bonded_pt_energy_model` and `rcc_bonded_pt_kappa`, and marked `rcc_bonded_pt_mu/lambda` as prototype-only.
- Moved the roadmap current task to correcting the bonded virtual-tet energy model.
- Added planned gates for ABD CPU E/G/H, ABD CUDA reporter E/G/H, and a bonded-mode no-penetration scene observation.
- Updated the source/doc gate so future document checks require the ABD-style energy plan.

### Commands

| Command | Result |
| --- | --- |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Source/doc gate accepted the corrected ABD-style roadmap anchors. |
| `uv run --no-sync python -m py_compile scripts/run_rcc_adhesion_acceleration_gates.py scripts/run_rcc_adhesion_acceleration_all_gates.py scripts/run_rcc_adhesion_acceleration_cuda_gates.py scripts/build_docs.py` | Passed. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc checks, Python syntax checks, and docs site build completed; existing MkDocs nav/API warnings remained informational. |

### Decision

The next safe implementation task is to replace the prototype SNH virtual-tet oracle and reporter with ABD-style high-kappa energy, then rerun the CUDA and scene gates before continuing release or benchmark work.

## 2026-06-02 ABD Virtual-Tet Oracle And Reporter

### Context

After the plan correction, the next implementation slice replaced the prototype SNH virtual-tet energy with ABD-style OrthoPotential over `F = Ds Dm_inv`. This is the production energy model required before a locked PT pair can skip CCD/contact/RCC.

### Implemented

- Replaced `RCCBondedPTVirtualTetInput` material parameters with `energy_model = ABDOrtho` and `kappa`, defaulting to `1e8`.
- Replaced the CPU oracle energy with `kappa * ||F F^T - I||^2`, assembled through the existing `dF/dx` mapping into the 12 vertex DOFs.
- Added a high-kappa CPU finite-difference gate under `[rcc_bonded_pt][oracle][abd_energy]`.
- Replaced the CUDA bonded reporter's SVTS function call with ABD OrthoPotential, including explicit row-major ABD affine layout to column-major FEM `dFdx` permutation.
- Replaced default config keys `rcc_bonded_pt_mu/lambda` with `rcc_bonded_pt_energy_model = "abd_ortho"` and `rcc_bonded_pt_kappa = 1e8`.
- Added source/doc checks preventing the retired SNH config keys and SVTS reporter function from reappearing in the production path.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_core -j8 && build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle]" -r compact` | Passed after switching high-kappa gradient comparison to relative error. Reported `All tests passed (22 assertions in 3 test cases)`. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_core uipc_test_backend_cuda -j8` | Passed. Built core and CUDA backend test binaries with existing CUDA warnings only. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (91 assertions in 6 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (130 assertions in 6 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][abd_energy]" -r compact` | Passed. Reported `All tests passed (11 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][reporter][abd_oracle]" -r compact` | Passed. Reported `All tests passed (5 assertions in 1 test case)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Passed. Reported `All tests passed (4 assertions in 1 test case)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8`, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc checks, Python syntax checks, and docs site build completed; existing MkDocs nav/API warnings remained informational. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_sim_case -j8 && build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Passed. Reported `All tests passed (836 assertions in 2 test cases)`. |

### Decision

The bonded virtual-tet energy model now matches the original high-stiffness ABD requirement. Remaining correctness blockers are release/beta carry, no-penetration scene observation while CCD is skipped, pre-CCD filtering in every concrete simplex filter, and benchmark timing.

## 2026-06-02 ABD Contract Documentation Hardening

### Context

After correcting the implementation from the prototype SNH virtual-tet energy to ABD OrthoPotential, the stable docs still needed a stricter handoff contract. The important risk is future drift: an agent could read old SVTS/SNH history, treat it as a valid replacement energy, and then skip CCD/contact/RCC without the high-stiffness ABD energy originally required by the design.

### Implemented

- Updated the roadmap current phase to Phase 4: release, fallback, and bonded-mode scene gates.
- Tightened the core rule: any locked PT pair that skips CCD/contact/RCC must be backed in the same step by high-kappa ABD-style virtual-tet energy over `F = Ds Dm_inv`, with production/default gates requiring `kappa >= 1e8`.
- Clarified that `SoftVertexTriangleStitch` is only the rest-shape and thickness reference. Its Stable Neo-Hookean energy is historical/prototype evidence only and must not re-enter the production reporter path.
- Added a lifecycle state machine and release ordering contract: released pairs must be removed from bonded reporter input, must carry beta back to RCC persistence, and must be counted once.
- Added a release/beta-carry planned gate before the bonded-mode scene gate.
- Clarified that E/G/H oracle success is necessary but not sufficient for non-penetration; the bonded-mode scene gate must observe penetration/gap or an equivalent fixture-specific bound.
- Strengthened the source/doc gate so it checks the ABD/SVTS boundary, high-kappa requirement, current Phase 4 status, release ordering, and absence of the obsolete "ABD material keys are still planned" wording.

### Commands

| Command | Result |
| --- | --- |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Source/doc gate accepted the hardened ABD, release, and Phase 4 anchors. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Ran source/doc checks, Python syntax checks, and docs build. Existing MkDocs nav/API informational warnings remained. |

### Decision

The docs now treat the ABD virtual-tet energy as a non-negotiable production precondition rather than one possible material choice. The next implementation slice remains release/beta carry, followed by no-penetration scene observation and pre-CCD filtering.

## 2026-06-02 Strain Release And Beta Carry

### Context

After ABD OrthoPotential became the production bonded virtual-tet energy, the next lifecycle gap was release. A locked pair that fails a release gate must leave active bonded state, must not immediately re-lock from the same high-beta RCC snapshot, and must carry its last locked beta back into RCC PT persistence.

### Implemented

- Added `rcc_bonded_pt_release_strain`, defaulting to `1e30` so release remains disabled until a test or scene selects a threshold.
- Added device release evaluation for strain, flip, and degenerate current virtual-tet state in `RCCBondedPTSystem`.
- Added released key/topology/beta/age/flag snapshots and kept them zipped after release extraction.
- Compact released entries out of the active locked state before the CUDA owner refreshes bonded reporter/filter input.
- Suppress same-step relock for keys released by the current owner update.
- Added `RCCBondedPTBetaCarryScratch` to merge released key/beta pairs back into RCC PT persistence without overwriting newer duplicate beta values already present in the RCC snapshot.
- Added deterministic backend fixtures for one strained release plus one persistent lock, and for beta-carry merge-without-overwrite behavior.
- Strengthened the source/doc gate with anchors for release config, released snapshots, RCC beta carry, and the release fixtures.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` | Passed. Built the CUDA backend test binary; existing CUDA warnings remained. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][release]" -r compact` | Passed. Reported `All tests passed (24 assertions in 2 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (156 assertions in 8 test cases)`. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Source/doc gate accepted the release config, released snapshot, RCC beta-carry, and release-fixture anchors. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc checks, Python syntax checks, and docs site build completed; existing MkDocs nav/API informational warnings remained. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8` with no work to do, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. |

### Decision

The first release policy slice is implemented and covered: strain/flip/degenerate release, active-lock compaction, released snapshot alignment, same-step relock suppression, one-shot release counters, and released beta carry. Remaining release reasons are normal gap, tangential slip, sticky-side failure, and disabled contact policy. Scene-level no-penetration observation, pre-CCD filtering in all concrete filters, and benchmarks are still required before bonded PT acceleration can claim full production correctness or speedup.

## 2026-06-02 Gap And Slip Release

### Context

After the strain release slice, forced-pull scene gates still needed normal-gap and tangential-slip release reasons. These reasons can be evaluated from the current locked payload without adding new fat state: `Dm_inv` reconstructs the lock-time virtual tet rest `Dm`, which provides the rest point-plane gap and rest closest-foot barycentric coordinates.

### Implemented

- Added `rcc_bonded_pt_release_gap` and `rcc_bonded_pt_release_slip`, both defaulting to `1e30` so release stays disabled unless a test or scene selects thresholds.
- Extended the device release evaluator to reconstruct rest `Dm` from `Dm_inv`.
- Added normal-gap release as growth of current point-triangle normal distance beyond the lock-time rest gap.
- Added tangential-slip release as current closest-foot barycentric displacement from the lock-time rest foot, measured in the current triangle tangent metric.
- Kept the evaluator device-only and SoA-based; no host roundtrip or extra sorted payload was added.
- Added independent backend fixtures for gap and slip release while keeping the stay/release two-lock lifecycle checks.
- Updated roadmap, subsystem docs, conventions, and source/doc gates to mark gap/slip release implemented while leaving sticky-side and policy release planned.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` | Passed. Built the CUDA backend test binary; existing CUDA warnings remained. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][release]" -r compact` | Passed. Reported `All tests passed (58 assertions in 4 test cases)`. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed. Source/doc gate accepted gap/slip config, implementation, and fixture anchors. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (190 assertions in 10 test cases)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8`, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc checks, Python syntax checks, and docs site build completed; existing MkDocs nav/API informational warnings remained. |

### Decision

Normal-gap and tangential-slip release are now covered as deterministic CUDA gates. The remaining release reasons are sticky-side failure and disabled contact policy, which require explicitly routing RCC sticky/policy inputs into the bonded owner rather than inferring them from geometry alone.

## 2026-06-02 Sticky And Policy Release

### Context

After strain, gap, and slip release were implemented, the remaining backend release reasons were sticky-side failure and disabled contact policy. These reasons must use the same RCC sticky/policy inputs that the contact path uses; a geometry-only guess would not prove consistency with the current RCC discretization.

This slice also exposed a documentation risk: release-side sticky/policy coverage is not the same as lock-side sticky/policy coverage. The live lock producer is still beta/rest-shape only, so the docs now keep release coverage, lock-gate parity, scene diagnostics, pre-CCD filtering, and benchmarks as separate gates.

### Implemented

- Added `RCCBondedPTReleaseContext` so the bonded owner can receive sticky signs, lagged vertex normals, contact element ids, subscene element ids, contact/subscene masks, and RCC adhesive enable tables without a host roundtrip.
- Routed the release context from RCC Phase A in `ipc_simplex_rcc_adhesive_contact.cu`.
- Added device release checks for sticky-side failure using RCC sticky-side semantics.
- Added device release checks for disabled contact/subscene/RCC adhesive policy.
- Added deterministic backend fixtures for one persistent lock plus one sticky-side release and one policy release.
- Hardened roadmap, subsystem, architecture, conventions, and source/doc gate wording so future work cannot confuse implemented backend release reasons with planned live lock gates or scene-accessible diagnostics.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8 && build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][release]" -r compact` | Passed. Reported `All tests passed (92 assertions in 6 test cases)`. |
| `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. Reported `All tests passed (224 assertions in 12 test cases)`. |
| `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py` | Passed. Built with `-j8` with no work to do, then passed core bonded-PT, backend bonded-PT, and bunny GPU sanity gates. The bunny sanity case reported `All tests passed (4 assertions in 1 test case)`. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed. Source/doc checks, Python syntax checks, and docs site build completed; existing MkDocs/Doxygen informational warnings remained. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed after the documentation hardening. Source/doc gate accepted the lock-vs-release, scene diagnostics, and sticky/policy anchors. |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Passed after the documentation hardening. Source/doc checks, Python syntax checks, and docs site build completed; existing MkDocs/Doxygen informational warnings remained. |

### Decision

All backend device release reasons currently planned for the bonded owner are implemented and covered by deterministic CUDA fixtures. The path is still not production-complete: released topology/age/flags and per-reason counts need scene-accessible diagnostics, the live lock producer still needs age/sticky/gap/slip/policy lock gates and rejection counters, locked keys still need to be removed before PT CCD broadphase in every concrete filter backend, and the bonded-mode `pt_lift_release` scene plus benchmark matrix remain planned.

## 2026-06-02 Review Findings And Roadmap Re-Sequencing

### Context

An adversarially-verified review (six dimensions: design fidelity, physics/math, GPU performance, code-vs-plan conformance, test-gate integrity, commit discipline) was run against the branch to check whether following the existing roadmap to the end would actually deliver the original intent: skip CCD, skip contact/adhesion force, and replace each stable PT pair with a high-kappa ABD virtual-tet energy.

### Source Observations

- The ABD energy is mathematically correct: `sym::abd_ortho_potential::E` expands to `kappa * ||F F^T - I||^2`, matching the doc form and the independent CPU oracle (`build_rcc_bonded_pt_virtual_tet_oracle`), which is cross-checked against the GPU reporter. Row-major(ABD `q`) to column-major(FEM `vec(F)`) gradient/Hessian permutations and `make_spd` placement are correct.
- The "skip CCD" leg of the intent does not exist in code. `filter_rcc_bonded_pt_locked_active_pairs()` runs in `do_filter_active` and mutates only the DCD active `PTs`; `do_filter_toi` and the concrete BVH `candidate_AllP_AllT_pairs` are never filtered, so locked pairs still pay full CCD broadphase and TOI every Newton iteration. The contact/RCC-assembly skip is realized (via `friction_PTs()`); more importantly, removing the locked pairs' stiff near-contact log-barrier and adhesion Hessian blocks from the linear system can improve conditioning and cut Newton/PCG iteration counts — likely the dominant win, active now, and independent of the (still-unrealized) CCD skip. The earlier "CCD is the headline/only win" framing was wrong; CCD-broadphase/TOI cost is one lever among at least three (assembly, conditioning/iterations, CCD).
- The fresh-lock predicate is `beta >= threshold` only; age/sticky/gap/slip/policy are evaluated only as release reasons against already-locked entries, never as lock gates. `rcc_bonded_pt_min_lock_age` is not a live config key.
- Locking freezes beta (the locked pair leaves `friction_PTs()`, so Phase A does not evolve it) and substitutes geometric release proxies for the spec's energy-driven debonding law.
- The bonded ABD energy is reflection-invariant (depends only on `F F^T`), so it is not a non-penetration barrier; CCD TOI being still active is currently the only tunneling guard.
- Latent issues confirmed against code: (1) on a zero-PT-candidate step, `m_prev_keys_PT` is not rebuilt but released beta is still merged into it, so a re-bond can seed from a stale beta for one step; (2) the membership key is a 64-bit hash with no topology re-check on match, so a hash collision can silently drop a real contact pair (~1e-8 across ~1e6 pairs, noted in code); (3) `replace_from_sorted_device_entries` silently falls back to identity `Dm_inv`/zero rest volume on a key miss, which is unreachable by construction but oracle-invisible if key derivation ever drifts.
- The only always-runnable gate is a source/doc string scan plus a syntax check; it proves no runtime behavior. The producer's real manager-sourced release context (sticky/policy/normals) is fed only by synthetic test contexts, never end-to-end.

### Decisions

- Keep the doc-based methodology; it is the project's strongest asset and the review confirmed docs track code accurately.
- Re-sequence the roadmap: pre-CCD filter integration (Phase 2 remaining item) is now the current gating milestone, because it is the only source of CCD savings. Release/scene/benchmark work is downstream of it. The current-focus line and Next Safe Task were rewritten; Phase 4 heading re-labelled "Backend Implemented; Downstream Of Pre-CCD Filter".
- Add a CCD-removal precondition to the architecture: CCD may be skipped for locked pairs only after the `pt_lift_release` no-penetration gate passes and inversion/tunneling is handled without CCD; the reflection-invariance of the ABD energy is stated as a testable invariant.
- Document beta-freeze-while-locked as a deliberate approximation and require a calibration gate (`[rcc_bonded_pt][calibration][debond]`) against the unaccelerated beta-evolution debond timing before any correctness claim.
- Add conventions hot-path rules for a producer steady-state early-out and a negative disabled-release sentinel (instead of `1e30`), plus a `rcc_bonded_pt_producer_ms` benchmark timer, so the steady-state producer cost is measured rather than hidden.
- Record the stale-snapshot lifecycle bug as a roadmap blocker with an `n==0`+release fixture requirement; record the hash-collision and silent-fallback items as known robustness hardening.
- Reframe the performance story as three independent levers — assembly skip, linear-system conditioning / iteration count, and CCD-broadphase/TOI cost — not "CCD is the only/headline win". The conditioning lever (replacing stable adhesive pairs' stiff near-contact log-barrier Hessian with a smooth high-kappa ABD block) is likely dominant and active now. The benchmark must therefore measure Newton/PCG iteration counts (already available via the per-iter solver telemetry, commit `05644f78`), not just CCD/assembly time, and sweep `rcc_bonded_pt_kappa`, since kappa trades bond rigidity against conditioning and too-high kappa can worsen it.

### Commands

| Command | Result |
| --- | --- |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed after updating doc anchors for the re-sequenced phases, the CCD-removal precondition, the beta-while-locked approximation, and the new producer/sentinel conventions. |

## 2026-06-02 Pre-CCD Filter Implementation

### Context

The re-sequenced roadmap made the pre-CCD filter the gating deliverable for the CCD-cost lever. The mechanism removes locked PTs from PT CCD broadphase, TOI, and active-pair emission in one place, while staying default-off because removing CCD also removes the last non-penetration guard (architecture CCD Removal Precondition).

### Source Observations

- `QueryBuffer` (LBVH `AtomicCountingLBVH::QueryBuffer`, stackless `StacklessBVH::QueryBuffer`) exposes only a read-only `view()` with a private logical size, so compacting the PT candidate buffer after broadphase is not possible through its public interface.
- Every backend's PT broadphase predicate already has the resolved global point id `V = Vs(i)` and triangle `F = Fs(j)`, so the locked-pair check belongs inside the predicate (before candidate emission) — which is exactly "before PT CCD broadphase".

### Implemented

- Added the shared device helper `rcc_bonded_pt_candidate_is_locked(locked_keys, V, F)` to `rcc_bonded_pt_lookup.h`, reusing the existing orientation-invariant membership lookup.
- Rejected locked PTs inside the PT broadphase predicate of all four simplex filters (LBVH, stackless BVH, info stackless BVH, v0 info stackless BVH), gated by a captured `rcc_skip_ccd` flag and `rcc_locked_keys`.
- Added `DetectInfo::rcc_bonded_pt_locked_keys()` / `rcc_bonded_pt_skip_ccd()` accessors and a base-Impl `rcc_bonded_pt_skip_ccd` flag with a public setter.
- Plumbed `rcc_bonded_pt_skip_ccd` config (default `0`) through `RCCBondedPTSystem` into the filter (bind/feed paths).
- Added the deterministic GPU fixture `[rcc_bonded_pt][filter][ccd]`: a locked pair recorded with a permuted triangle is rejected (orientation-invariant), unlocked candidates are kept, and an empty locked set keeps all.

### Decisions

- Keep `rcc_bonded_pt_skip_ccd` default-off until the `pt_lift_release` no-penetration scene gate passes. The mechanism is correct and tested at the membership level; the full scene-level candidate/TOI-absent assertion is coupled to that gate.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_backend_cuda -j8` | Passed. Only pre-existing `xi2`/nested-class warnings. |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_core -j8` | Passed. |
| `uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]" -r compact` | Passed. `All tests passed (6 assertions in 1 test case)`. |
| `uipc_test_backend_cuda "[rcc_bonded_pt]" -r compact` | Passed. `All tests passed (230 assertions in 13 test cases)`. |
| `uipc_test_core "[rcc_bonded_pt]" -r compact` | Passed. `All tests passed (91 assertions in 6 test cases)`. |
| `uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Passed. BVH/radix regression unaffected. |

## 2026-06-02 Pre-CCD No-Penetration Scene Gate

### Context

The pre-CCD skip is implemented but default-off because removing CCD removes the last non-penetration guard for locked pairs, and the ABD virtual-tet energy is reflection-invariant (it cannot by itself prevent tunneling). The architecture CCD Removal Precondition requires a scene gate that observes no penetration with `rcc_bonded_pt_skip_ccd` on before the skip can be enabled safely. This is also a genuine experiment: it could have shown that the ABD energy alone is insufficient.

### Implemented

- Added `[rcc_bonded_pt][scene][pt_lift_release]` to `apps/tests/sim_case/rcc_adhesion_lift_release_gate.cpp`, reusing the proven `cube_gate::build_scene` PT-rich subdivided cube-cube fixture.
- Enabled `rcc_bonded_pt_enabled=1`, `rcc_bonded_pt_skip_ccd=1`, `rcc_bonded_pt_beta_lock_threshold=0.9`, `rcc_bonded_pt_kappa=1e8`.
- Read bonded state through `RCCBondedPTStateAccessorFeature` and measured the contact-face gap each frame from contact through the lift hold.
- Asserts: at least one bonded lock forms; the contact-face gap stays `>= -0.01` (no visible penetration, `PenTol < d_hat`); the lower cube is lifted (`bottom_y > BottomLiftY - 0.06`) through the bonded energy.

### Observed Result

The gate passes (596 assertions). Captured diagnostics at the lift hold:

- `locked_count = 8` (from `candidate_count = 16`), `duplicate_suppressed = 0`, `degenerate_rejected = 0` — 8 bonded locks, no duplicate ownership.
- `min_gap = +0.0188` at frame 63 (just after contact) — the contact-face gap never went negative, i.e. zero penetration with a comfortable margin (the `-0.01` tolerance was never approached). The faces settle near `d_hat` (0.02).
- `filter_skipped = 0` — corroborates the pre-CCD path: with `skip_ccd` on, locked pairs are removed at the broadphase predicate and never reach the active-view compact (which would otherwise report ~8).
- `bottom_y = 0.581` (target lift `BottomLiftY = 0.60`) — the lower cube was carried to within 0.02 of the full lift purely through the bonded ABD energy, with CCD skipped for the locked pairs.

So for this fixture the ABD virtual-tet energy holds the bond non-penetrating once CCD is skipped. The legacy `[rcc_adhesion][gate]` (836 assertions, 2 cases) still passes, confirming no regression.

### Decision

The CCD Removal Precondition is satisfied for the cube fixture, but `rcc_bonded_pt_skip_ccd` stays default-off: one fixture is evidence, not a license to flip a default. Next is the benchmark (to measure whether the conditioning/iteration lever is a net win) and broader fixtures (cloth/patch) plus the forced-pull release/separation and adhesion-off baseline in the scene gate.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target uipc_test_sim_case -j8` | Passed. |
| `uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]" -r compact` | Passed. `All tests passed (596 assertions in 1 test case)`. |
| `uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Passed. `All tests passed (836 assertions in 2 test cases)` (legacy regression). |
| `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Passed after updating doc/source anchors for the scene gate. |

## 2026-06-02 Python Bonded-PT Accessor And Viewer Demo

### Context

Bonded-PT state was readable only from C++ tests. To inspect and visualize the locked virtual tets interactively, the accessor needed Python bindings and a viewer demo. This also exercises the full press/hold/lift/pull lifecycle end-to-end in a real scene for the first time.

### Implemented

- Added `RCCBondedPTStateAccessorFeature::dump_locked_tet_world_positions()` (core feature + a non-pure overrider default so the test mock is unaffected). The CUDA overrider gathers the four world positions per locked tet from `GlobalVertexManager::positions()` indexed by `locked_topos`.
- Bound `RCCBondedPTStateAccessorFeature` in pyuipc: `locked_pair_count()`, `counters()` (dict), and `dump_locked_tet_world_positions()` (`[M, 4, 3]` numpy).
- Added bonded params to `rcc_adhesive_subdivided_cube_lift_release_demo.build_demo` (default-off) and a dedicated viewer `python/examples/rcc_bonded_pt_lift_pull_viewer.py` that enables bonded + skip_ccd + finite release thresholds, draws the bonded tets as a polyscope curve network, and shows live lock counts.

### Observed Result (headless)

End-to-end press/hold/lift/pull with `rcc_bonded_pt_skip_ccd` on, `release_gap = 0.04`, `release_strain = 0.6`:

- HOLD (frame 240): 8 locks; dump returns shape `(8, 4, 3)`; contact-face gap min `+0.0193` (no penetration); bottom cube lifted to `0.581`.
- PULL (frame 400): all 8 locks released (`released_count = 8`); contact gap `+0.73`; bottom cube fell to `0.170`; cube separation `gap_y = 1.030` — the bonds release on the hard pull and the cubes separate cleanly.

This is the first end-to-end scene observation of the bonded lock -> hold -> release -> separate lifecycle (with CCD skipped during the locked phase). It is demo/inspection evidence, not yet a pass/fail assertion gate; the forced-pull release/separation C++ gate is still planned.

### Commands

| Command | Result |
| --- | --- |
| `cmake --build build/cuda_mixed_fused_pcg --target core backend_cuda pyuipc -j8` | Passed; pyuipc auto-installed into the venv. |
| headless `build_demo(bonded=True, skip_ccd=True, release_gap=0.04, release_strain=0.6)`, 400 frames | 8 locks at hold (gap_min `+0.019`), 8 released at pull (separated, gap_y `1.03`). |
| `uipc_test_core "[rcc_bonded_pt]"` / `uipc_test_backend_cuda "[rcc_bonded_pt]"` | Passed (91/6, 230/13) — accessor change is regression-free. |

## 2026-06-02 Cross-Fixture Bonded-PT Probe (Cloth Demos)

### Context

Broaden bonded-PT + skip_ccd validation beyond the all-ABD cube fixture to the FEM-cloth demos: oriented-cloth pickup, cube-cloth lift/release, and cloth peel — softer, denser, mixed body types (NeoHookean shell bonded to an ABD cube).

### Implemented

- Added bonded params (default-off) to `build_demo` in `rcc_adhesive_oriented_cloth_demo.py`, `rcc_adhesive_cube_cloth_lift_release_demo.py`, and `rcc_adhesive_cloth_peel_demo.py` (same additive pattern as the subdivided-cube demo).
- Added `scripts/probe_rcc_bonded_pt_demos.py`: runs each fixture in its own subprocess with bonded + skip_ccd and reports stability, lock trajectory, released count, and a geometry-agnostic penetration proxy (the bonded points' signed distance to their own virtual-tet triangle plane).

### Observed Results (bonded + skip_ccd)

| Fixture | kappa | valid | max_locked | released | min_abs_sep | macro behavior |
| --- | --- | --- | --- | --- | --- | --- |
| oriented_cloth (cube picks up cloth) | 1e8 | yes | 111 | 0 | 0.0016 | cloth lifted with the cube (mean Y 0.30 -> 1.01) |
| cube_cloth lift/release | 5e7 | yes | 88 | 0 | 0.021 (~d_hat) | held; on pull the cloth rode **up** with the cube, did not separate |
| cloth_peel | 1e8 | yes | 233 | 0 | 0.0031 | bonds held the cloth flat; the peel was **suppressed** (no edge lift) |

- All three run **stably** end-to-end (no NaN/blowup): bonded + skip_ccd does not destabilize the FEM NeoHookean cloth.
- Locks form **abundantly** (88-233; cloth contact is denser than the cube case).
- **No penetration**: bonded-tet plane separation stays ~d_hat and never collapses.
- **Release/separation did NOT fire** (released ~0) even with aggressive thresholds: `release_strain=0.1, release_gap=0.01` gave only 1/88 release on cube_cloth (the externally pulled-down vertex) and 0/233 on cloth_peel.

### Finding

Geometric-proxy release (strain/gap) has a structural blind spot: a stiff ABD bond (kappa 5e7-1e8) **absorbs** the imposed motion, so the per-tet strain/gap never grows past threshold when the counterpart is **compliant** (FEM cloth, weak constraint). The bond then holds and the soft side simply follows it. The all-ABD cube-cube case separates only because two stiff SoftTransformConstraint actuators force enough gap/strain across the bond. This sharpens the phys-1 concern: geometric release is not a reliable substitute for the spec's energy-driven beta debonding on compliant fixtures.

### Decision

Establishment, stability, and non-penetration **generalize** across fixtures (good). Release/separation on compliant-counterpart fixtures is an **open problem** needing release-law redesign — e.g. evaluating the RCC beta criterion on locked pairs, a force/strain-rate release, lower kappa, or a lock gate that excludes soft counterparts. Keep `rcc_bonded_pt_skip_ccd` default-off; do not claim release correctness on cloth.

### Commands

| Command | Result |
| --- | --- |
| `python/.venv/bin/python scripts/probe_rcc_bonded_pt_demos.py` | All 3 stable; locks 111/88/233; released 0/0/0; min_abs_sep 0.0016/0.021/0.003. |
| tuning `release_strain=0.1, release_gap=0.01` on cube_cloth/cloth_peel | released 1/88 and 0/233 — release still largely suppressed (confirms the blind spot). |

## 2026-06-02 Corner-Peel Release Threshold Sweep (Fixed kappa)

### Context

Follow-up to the cross-fixture finding: can the cube-cloth fixture be made to peel/separate by tuning the release threshold alone, without lowering `kappa`? Changed the cube-cloth demo pull target to a cloth **corner** (outside the cube footprint, so unbonded) to create a peel front at the bonded-patch edge, then swept release thresholds at fixed `kappa=5e7`.

### Observed Result

| release_strain / release_gap | released @240 (hold/lift) | released @400 (post-pull) | cloth_y @400 |
| --- | --- | --- | --- |
| 0.3 / 0.02 ... 0.05 / 0.001 | 0 | **0** | ~1.0 (rode up with cube) |
| 0.02 / 0.0005 | 0 | 1 / 86 | 0.99 |
| 0.01 / 0.0002 (0.2 mm) | 0 | 2 / 87 | 0.99 |

Even at `release_gap=0.0002` (0.2 mm) only 1-2 of ~87 bonds released and the cloth still rode up with the cube — no separation. Below that, bonds would release from solver jitter during hold (no clean window).

### Finding

At fixed `kappa`, **no release threshold peels the cloth**, because geometric release (strain/gap) measures how far a bond has already *deformed/failed*, but a holding stiff bond keeps `curr_dist ~ rest_dist` (gap ~ 0) and `F ~ I` (strain ~ 0) by construction. The compliant cloth also absorbs the corner pull in its free region (`gap_max ~ 0.36-0.42` is the hanging corner) before it loads the bonded edge. This is the strongest evidence yet for phys-1: geometric release is the wrong tool for "stiff bond + compliant counterpart"; it needs a force/energy trigger (release when the bond restoring force, ~ `kappa * deformation`, exceeds a limit — which fires for a holding stiff bond), the RCC beta criterion on locked pairs, or a lower `kappa`.

### Decision

Keep the corner pull in the demo (cleaner peel test). Do not pursue threshold-only release on compliant fixtures; a force/energy-based release law is the right next design step if cloth release is required.

## 2026-06-02 Force/Energy Release Criterion

### Context

The corner-peel sweep showed geometric release (strain/gap) cannot peel a stiff bond on a compliant counterpart at any threshold, because a holding bond has ~0 deformation by construction. The fix is a force/energy trigger: `force ~ kappa * deformation` is appreciable for a holding stiff bond (tiny deformation amplified by `kappa`) where strain/gap are not.

### Implemented

- Added release reason `RCCBondedPTReleaseForce` (`1u << 7`) and config `rcc_bonded_pt_release_force` (default `1e30` = disabled).
- In `release_flags_from_current_shape`, compute the F-space restoring force `4 * kappa * rest_volume * dt^2 * ||C F||` (`C = F F^T - I`, reusing the strain block's `C`/`F`) and flag `force` when it exceeds the threshold.
- Threaded `kappa`/`dt`/force threshold into the producer via `set_release_force_config` (read from `rcc_bonded_pt_release_force`, `rcc_bonded_pt_kappa`, `dt` in `do_build`).
- Added unit test `[rcc_bonded_pt][release][force]`; exposed `release_force` in the cube-cloth `build_demo`; the cube-cloth viewer uses `release_force=1e-4` (geometric release disabled) for the corner pull.

### Observed Result (cube-cloth corner pull, kappa=5e7 fixed)

| release_force | @240 hold/lift | @400 pull | cloth_y | result |
| --- | --- | --- | --- | --- |
| 1.0 / 0.1 / 0.01 | rel=0 | rel=0 | 1.01 | stuck (rode up) |
| 0.001 | rel=0 | rel=17 | 0.95 | partial peel |
| 1e-4 | rel=0 | **rel=94 (all)** | **0.40** | **fully separated** |

`rel@240=0` for every threshold (no premature release during hold/lift) — a clean release window that geometric release at any threshold could not provide.

### Validation

`[rcc_bonded_pt][release][force]` passes (17 assertions); full release suite 109/7; backend `[rcc_bonded_pt]` 247/14; core 91/6; bunny BVH regression OK.

### Decision

The force/energy criterion resolves the compliant-counterpart release blind spot (phys-1) and is the recommended release trigger for stiff bonds; geometric strain/gap remain available. The threshold is scene-dependent (scales with `kappa`, `V0`, `dt`) and stays default-off (`1e30`).

## 2026-06-02 Full-Feature Adhesion Plan And PE/PP Formulas

### Context

The bonded acceleration only ever covers face-interior point-triangle contacts. Investigating why two off-diagonal corners on the faceted cube top face never bonded traced back to RCC adhesion itself being PT-only: the single-diagonal triangulation makes those corners classify as PE/PP, and PE/PP/EE adhesion is disabled, beta is PT-only, so they never enter `friction_PTs()`, never evolve beta, never lock. The advisor's framing is that the mesh **vertex-triangle (VT)** and **edge-edge (EE)** pair is the contact primitive; the PP/PE/PT split is only the closest-feature sub-formula. We downloaded the RCC adhesion paper's reference implementation (XBow `src/Bow/Energy/FEM/RCCAdhesionEnergy.h`) to see how it handles PP/PE.

### Source Observations

- XBow `RCCAdhesionEnergy3D`: beta stored per VT primitive keyed by `(boundary_point, boundary_face)`, carried across feature transitions; each step `point_triangle_distance_type` classifies PP/PE/PT and uses the **true feature distance** + matching tangent basis; pairs weighted by `boundary_point_area / 2`; the adhesion op extends the IPC barrier op (shares classified pairs, distances, kappa).
- libuipc `codim_ipc_simplex_rcc_adhesive_function.h`: `PT_*` adhesion uses the unflagged plane projection (`point_triangle_distance2`, deliberately, to avoid a diagonal-pull artifact); `EE_*`/`PE_*`/`PP_*` normal and tangential adhesion all `return 0` (V1 disabled).
- `ipc_simplex_rcc_adhesive_contact.cu`: beta is PT-only (`m_beta_EE/PE/PP.fill(0)`); the PE/PP **assembly skeleton already exists** — it loops `friction_PEs()`/`friction_PPs()`, reads `m_beta_PE/PP`, early-outs on `beta <= 0`, and calls the `PE_*`/`PP_*` functions for energy (947-995) and grad/hess (1216-1311). Only the formula bodies and per-primitive beta were missing.
- Distance + friction utils for PE/PP all exist: unflagged `point_edge_distance2`/`point_point_distance2` (+ `_gradient` -> Vector9/6, + `_hessian` -> 9x9/6x6) via `distance_flagged.h`, and `point_edge_*`/`point_point_*` basis/closest/`tan_rel_dx`/`jacobi` in `friction_utils.h`. The barrier path uses the flagged (true closest-feature) distance for non-penetration.

### Decisions

- Adopt the XBow model: per-VT-primitive beta + true feature distance (PP/PE/PT) + area weight, built on the barrier's classified pairs/distances (reuse `d^2` and derivatives). Keep the barrier on the true closest-feature distance for non-penetration. The bonded lock decides on the VT primitive; the ABD tet stays point-plane (`F = Ds Dm_inv`).
- Implement in five independently-verifiable steps (architecture "Full-Feature Adhesion And Per-Primitive Beta", roadmap Phase 6). Start with Step 1 (PE/PP formulas) because it is self-contained and behaviour-neutral until beta is wired.
- Documented the plan into architecture (new section + relationship update), roadmap (Phase 6 + current-focus note + blocker), conventions (rules 15/16 + adhesion sub-formula oracle rule), and the subsystem doc (scope + Phase 6 table).

### Implemented

- Step 1: replaced the `return 0` `PE_normal/PE_tangential/PP_normal/PP_tangential` adhesion energy/gradient/gradient_hessian stubs with the true point-edge / point-point feature-distance formulas, mirroring `PT_*` exactly (PE: Vector9/9x9 via `point_edge_distance2*` and `point_edge_*` friction basis; PP: Vector6/6x6 via `point_point_distance2*` and `point_point_*`). Updated the EE normal/tangential disable comments (EE still disabled; PE/PP now implemented). Confirmed argument order against the call sites (`Cn/Ct, beta, d_hat, dt, [prev...], [curr...]`) and that PE normal Hessian is SPD-projected by the caller while PP/tangential blocks are naturally PSD.
- Behaviour is unchanged: `m_beta_PE/PP` are still zero-filled and every call site early-outs on `beta <= 0`, so the new formulas are not reached until Step 2.

### Commands

- `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` — source/doc gate (with new Phase 6 anchors).
- CUDA backend rebuild to confirm the device formulas compile.

## 2026-06-02 Distance Reuse Analysis (Workflow)

### Context

Question for Step 4: can RCC adhesion/bonded reuse the distance the barrier already computed instead of recomputing it? My earlier framing (and the docs I wrote earlier today) said "reuse the barrier's classified pairs/distances; the libuipc analogue of XBow layering the adhesion op on the barrier op." Ran a 5-reader workflow (barrier / friction / rcc-adhesive / reporter-arch / XBow) + verified the load-bearing fact myself.

### Source Observations

- Barrier (normal contact) RECOMPUTES `d^2`/grad/hess in-kernel every pass and stores nothing: `D` is materialized only under `if constexpr(RUNTIME_CHECK)` ([ipc_simplex_normal_contact.cu:67-83](../../src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu)); production passes only the `flag` to `PT_barrier_energy`, which recomputes `D` internally. The only per-pair distance buffer (`per_pt_dist2`) is telemetry — full-pair (not flagged), reduced to a scalar, exposed through no reporter Info.
- XBow's "reuse" is structural only, and NOT even a shared pair set: `RCCAdhesionEnergy3D : IpcEnergyOp3D` and barrier are SEPARATE `make_shared` instances (distinct PP/PE/PT storage); RCC overrides `precompute()`, clears the inherited arrays, and rebuilds them from its own bonded `pair_PT/pair_EE`; it recomputes distances independently. Only `xi/dHat/kappa` VALUES are shared (XBow `RCCAdhesionEnergy.h:586/848/875`, `IPCSimulator.h:440/492/501`).
- libuipc RCC already shares MORE structurally than XBow: it iterates the same lagged `friction_PTs()` lists and writes additively into the shared friction output buffers.
- Lists differ by design: barrier iterates live `PTs()` (current-step DCD); friction + RCC adhesion iterate lagged `friction_PTs()`. RCC's lagged tangential basis/foot is bit-identical to friction's `PT_friction_basis` (same helper, prev positions, list order).
- RCC normal uses the deliberately UNFLAGGED plane-projection `d^2` (`codim_ipc_simplex_rcc_adhesive_function.h:393-403`), numerically different from the barrier's flagged `d^2`; cannot be inherited from the barrier. Per Newton iteration a bonded PT pair recomputes current-position `d^2` ~5x across 4-5 launches.

### Decisions

- Reject a shared per-pair `d^2`/derivative scratch buffer: on launch/bandwidth-bound kernels it trades cheap in-register arithmetic for global-memory traffic (Hessian = 144 floats/pair) — a likely regression.
- Reframe Step 4 as structural single-compute: (a) intra-RCC reuse `D` across RCC's own normal energy/grad/hess + `db_dd2` (zero cross-reporter coupling); (b) optionally merge with the FRICTION kernel (same lagged list, bit-identical basis) — preferred over a barrier merge (which needs list reconciliation + only `db_dd2`/flagged `d^2` shareable).
- Corrected architecture (XBow reference item 4, target-model item 5, Step-4 table), roadmap (Phase 6 Step 4), and the subsystem doc to match. The earlier "reuse the barrier's distances / layer like XBow" wording was inaccurate.

### Commands

- Workflow `rcc-distance-reuse-analysis` (6 agents); independent read of `ipc_simplex_normal_contact.cu` + `codim_ipc_simplex_normal_contact_function.h` to confirm the no-store fact.
- `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` — source/doc gate after the corrections.

## 2026-06-02 Contact Area Weight Check (Workflow)

### Context

Question for Step 3: does libuipc already fold the per-vertex contact area into Cn/Ct, making a separate `A_k` weight redundant? Ran a 2-reader workflow (libuipc coeff trace + XBow area weighting) and verified the spec note myself.

### Source Observations

- libuipc: Cn/Ct are raw scalar stiffnesses; no area/mass/volume factor at any stage. `RCCAdhesiveCoeff` is 8 Floats + enabled ([rcc_adhesive_coeff.h:16-26](../../src/backends/cuda/contact_system/rcc_adhesive_coeff.h)); frontend writes them verbatim ([rcc_adhesive.cpp:114-115](../../src/constitution/rcc_adhesive.cpp)); tabular copies verbatim ([ipc_simplex_rcc_adhesive_contact.cu:205-216]); energy is `dt^2·Cn/(2 d_hat)·β²·D`, no `A_k` ([codim_ipc_simplex_rcc_adhesive_function.h:416]). Stencil coeffs are arithmetic averages (PT /3, PE /2, EE /4), same as barrier `kappa`. The barrier itself also does not area-weight per pair.
- Spec records the convention explicitly ([docs/specification/contact_models/rcc_adhesion.md:176-177](../specification/contact_models/rcc_adhesion.md)): "Cn and Ct are treated as area-weighted stiffnesses divided by d_hat ... Area A_k: lumped into Cn/Ct (libuipc IPC convention); not plumbed as a separate per-pair attribute."
- XBow: Cn/Ct are per-unit-area densities; area is a SEPARATE per-vertex `m_boundary_point_area` (face_area/3 onto each of 3 boundary verts, `IPCSimulator.h:109/118`), multiplied as `wPT` into the energy (`RCCAdhesionEnergy.h:1111`) and folded into tangential `mu_lambda = β²·area/dHat`. The 3D `/4` = `/2` (double-count) × `/2` (PT emplaced twice as EE). The same area field also weights XBow's ordinary barrier (`IpcEnergy3D.cpp:334`) — shared infra, nearly free.

### Decisions

- Verdict: area is NOT computed into Cn/Ct by any code (nothing to double-count yet), but by libuipc's documented convention it is *meant* to be lumped into Cn/Ct — consistent with barrier `kappa`. So a separate `A_k` is optional, not redundant-with-existing-code and not a bug fix.
- Consequence of the lumped convention: per-pair-uniform stiffness → adhesion scales with vertex count (not physical area) and non-uniform meshes pull unevenly; Cn is per-contact-element-pair so it cannot encode per-vertex area at all.
- Re-scoped Step 3 as optional / lowest priority: keep the lumped convention (uniform meshes), OR add per-vertex `A_k` for resolution-independent / non-uniform-mesh adhesion — but then Cn/Ct must be reinterpreted as densities and the area reused from the barrier's boundary-point area (no parallel field), else double-count. Corrected the Step-3 wording in architecture, roadmap, and the subsystem doc (the earlier "defaults to 1 until area exists" framing was inaccurate).

### Commands

- Workflow `rcc-area-weight-check` (3 agents); independent read of `docs/specification/contact_models/rcc_adhesion.md:170-185` to confirm the lumped-area note.
- `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` — source/doc gate after the corrections.

## 2026-06-02 Step 2 Plan And Step 0 PE/PP Oracle

### Context

Starting Phase 6 Step 2 (per-primitive beta). User chose the faithful VT-primitive model (align with advisor/XBow: judge on the VT pair, classification only picks the distance sub-formula). A planning workflow (`rcc-step2-vt-primitive-plan`, 4 agents) verified feasibility and produced the ordered plan.

### Source Observations

- The filter REDUCES each VT candidate to one of PT/PE/PP and the reduced PE/PP lists are NOT pure VT-reductions — they fuse VT, EE, and CodimPE degeneracies (lbvh_simplex_trajectory_filter.cu AllP_AllT vs AllE_AllE/CodimPE blocks). So unioning the reduced lists does NOT reconstruct the VT set; the correct construction is one entry per active `AllP_AllT` candidate, emitted before the dim-switch, carrying the full `(point,t0,t1,t2)` + a feature flag (= `degenerate_point_triangle` dim).
- The beta evolution law `PT_beta_evolve_existing(...,D,u_sq,blocked)` is feature-agnostic; only D and u_sq carry geometry. The sticky gate + occlusion gate are full-triangle based and carry over unchanged for any VT pair. The pair key `PT_pair_key(topo)` is identical for any VT primitive (pure function of the 4 vertex ids).

### Decisions

- Option 2 (VT-primitive). Ordered steps: (0) PE/PP FD oracle [decision-free]; (1) additive `active_VTs{topo,flag}` + `friction_VTs` across the 4 filters + interface [behavior-neutral]; (2) per-VT beta over `friction_VTs` keyed by `PT_pair_key`, flag-driven feature distance, gates carry over [behavior-neutral]; (3) single VT assembly loop, flag-switched to the Step-1 formulas [behavior FLIPS]; (4) retire `m_beta_PE/PP/EE` + zero-fill + separate loops; (5) producer kept on the `flag==4` subset so bonding stays byte-for-byte unchanged (real Step 5 later).
- Risk noted for Step 3: enabling PE/PP adhesion re-introduces the "diagonal-pull" the unflagged-plane PT formula was chosen to avoid (a face-interior-hovering vertex also emits a PE toward the shared diagonal). Must be validated on the faceted-cube fixture, not assumed.

### Implemented (Step 0)

- `apps/tests/backends/cuda/rcc_adhesion_oracle.cu`: GPU finite-difference E/G/H oracle for the Step-1 PE/PP adhesion device functions (PE Vector9/9x9, PP Vector6/6x6, normal + tangential), launching 1-thread kernels and central-differencing the energy/gradient. Tag `[rcc_adhesion][oracle][feature_adhesion][cuda]`. FD the raw (un-projected) Hessian; tangential perturbs only current DOFs (lagged basis fixed); beta passed >0 directly.
- Wired into `run_rcc_adhesion_acceleration_cuda_gates.py`; added source/doc gate anchors; ticked roadmap.

### Commands

- `cmake build/cuda_mixed_fused_pcg` (reconfigure for GLOB) + `cmake --build ... --target backend_cuda` — compiles + links clean.
- `uipc_test_backend_cuda "[rcc_adhesion][oracle][feature_adhesion]"` → All tests passed (12 assertions in 4 test cases).

### Implemented (Step 1)

- Added `struct ActiveVT { Vector4i topo; IndexT flag; }` (collision_detection/simplex_trajectory_filter.h) and an additive `active_VTs` list emitted by all four simplex filters' `filter_active`, written from the `AllP_AllT` (VT-candidate) kernel right after `degenerate_point_triangle` (full `vIs` topo + `dim` flag), before the reduction switch — one entry per active VT candidate regardless of closest feature. Compacted by a 5th `DeviceSelect` (predicate `topo(0)!=-1`) and emitted via the new `FilterActiveInfo::VTs(...)`.
- Trajectory filter: new `VTs()` accessor, `friction_VT` snapshot in `record_friction_candidates` (lagged, same phase as the reduced friction lists), `friction_VTs()` accessor, cleared in `do_clear_friction_candidates`. Consumer interface: `SimplexFrictionalContact::BaseInfo::friction_VTs()` forwards it.
- Additive only: barrier/friction reduced lists untouched; nothing consumes `friction_VTs` yet, so behaviour is unchanged.
- Important: the reduced PE/PP lists fuse VT + EE + CodimPE degeneracies, so `active_VTs` is emitted from the VT-candidate kernel (not reconstructed from the reduced lists) — its count is exactly the active VT candidates.

### Commands (Step 1)

- `cmake --build ... --target backend_cuda` — all 4 filters + interface compile + link clean.
- `scripts/run_rcc_adhesion_acceleration_cuda_gates.py --no-build` → core 91/6, backend 247/14, adhesion oracle 12/4, bunny 4/1 (no regression).

## 2026-06-02 Step 2/3 Per-VT-Primitive Beta And Unified Assembly

### Context

The behavior-flipping core of Phase 6: move beta + assembly from the face-interior `friction_PTs` to the full per-VT-primitive `friction_VTs`, so edge/corner contacts (the off-diagonal cube corners that started this) get adhesion.

### Implemented

- Function header `codim_ipc_simplex_rcc_adhesive_function.h`: added `VT_normal_adhesion_*` (uses the FLAGGED `point_triangle_distance2(flag,...)` dispatch, which routes PT->plane / PE->point-edge / PP->point-point and emits a 12-DOF gradient/Hessian directly — same machinery the barrier uses, so one path covers all features), `VT_tangential_adhesion_*` (switches on `degenerate_point_triangle` dim, calls the Step-1 PT/PE/PP tangential helpers on the reduced sub-stencil and scatters the Vector12/9/6 result into the 12-DOF block), and `VT_tangential_rel_dx_sq` for beta evolution. Coeff/d_hat aggregated over the full VT primitive (`PT_rcc_coeff`/`PT_d_hat`/`PT_contact_coeff`) so evolution and assembly stay consistent.
- Base `SimplexFrictionalContact`: added a `friction_pair_counts(pt,ee,pe,pp)` virtual hook (default = friction list sizes) called by the (still-final) extent functions. RCC overrides it to `{friction_VTs size, 0, 0, 0}` so all VT primitives ride the PT output slot as 12-DOF blocks; the friction reporter is unaffected.
- `ipc_simplex_rcc_adhesive_contact.cu`: `ActiveVT` now carries the `Vector4i` flag (lagged, fixed across the step). Phase B init/match, `_phase_b_if_new_frame`, `_compute_curr_keys_PT`, and `_evolve_beta_step_at_end` all iterate `friction_VTs` (key = `PT_pair_key(topo)`, identical for any VT primitive → beta persists across PP/PE/PT transitions; flagged feature distance + `VT_tangential_rel_dx_sq`). `do_compute_energy`/`do_assemble` collapsed to ONE VT loop. Sticky + occlusion gates carry over verbatim (full-triangle). Retired `m_beta_EE/PE/PP`, `_sync_disabled_buffers`, and the separate per-feature energy/assemble loops. Bonded producer kept PT-only via a `flag==4`-compacted `m_beta_PT_face` (DeviceSelect.Flagged) fed with `friction_PTs()` — byte-for-byte bonding behavior preserved (Step 5 will let it consume the full VT list).
- The 4 filters now emit `ActiveVT{vIs, flag}` (the Vector4i flag).

### Decisions

- Normal adhesion uses the flagged `point_triangle_distance2` path (provably equal to scattering the per-feature Step-1 PE/PP normal grad, and simpler); the Step-1 PE/PP normal functions stay (oracle-tested) and the PE/PP tangential functions are reused by the VT tangential wrapper.
- Coeff/d_hat aggregated over the full VT primitive regardless of flag (R7): treats the VT as the unit, keeps evolution/assembly consistent, avoids per-flag coeff branching.
- `make_spd` applied uniformly to the normal Hessian block (PP normal is already PSD, so it is a harmless no-op there).

### Observed Result / Validation

- `backend_cuda` + `uipc_test_sim_case` build + link clean. All gates green, no regression from the behavior flip: `[rcc_adhesion][oracle][feature_adhesion]` 12/4; backend `[rcc_bonded_pt]` 247/14; bunny 4/1; legacy `[rcc_adhesion][gate]` 836/2; bonded `[rcc_bonded_pt][scene][pt_lift_release]` 596/1; source/doc gate green (new VT anchors).
- The legacy scenes passing means the diagonal-pull risk (R-physics: enabling PE/PP adhesion re-introduces a sideways pull for face-interior-hovering verts near a diagonal) did not break the existing fixtures. A dedicated edge/corner-adhesion demo/unit test (proving the new coverage directly, not just no-regression) is the recommended next behavioral proof.

### Commands

- `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda uipc_test_sim_case` — compiles + links (one round of fixes: materialize Eigen diff expressions before `*_tan_rel_dx`).
- `uipc_test_backend_cuda "[rcc_adhesion][oracle]"/"[rcc_bonded_pt]"/"gpu_sanity_check -c bunny"`; `uipc_test_sim_case "[rcc_adhesion][gate]"/"[rcc_bonded_pt][scene][pt_lift_release]"`.
- `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py`.

## 2026-06-02 Headless Demo Verification (Corner Adhesion + Diagonal Pull)

### Context

Built `pyuipc` (and copied the fresh `libuipc_backend_cuda.so`/`libuipc_core.so` into the venv `_native/` — the stale-wheel trap) and probed the faceted-cube + oriented-cloth demos headless to directly confirm the Step 2/3 behavior, since the gates prove no-regression but not the new coverage.

### Observed Result

- Subdivided faceted cube (`rcc_adhesive_subdivided_cube_lift_release_demo`), adhesion ON + bonded ON: at hold the per-VT beta snapshot is **96 primitives all at beta=1.0** (`dump_pt_state()`, frac09=1.0) while the bonded producer locks only **8** (the face-interior `flag==4` subset). 96 >> 8 ⇒ the edge/corner VTs (previously beta-less) now carry beta — corner adhesion is active. During lift the lower cube follows (botY +0.170 -> +0.426, contact gap held ~0.019); adhesion OFF it is left behind (botY +0.170, gap opens to +0.275). Contact-face gaps are uniform (min≈mean≈max) ⇒ no diagonal distortion. No NaN over 160 frames.
- Oriented cloth (`rcc_adhesive_oriented_cloth_demo`, single-diagonal triangulation), adhesion ON vs OFF, hold phase: in-plane drift `|dXZ|mean` 0.00301 (ON) vs 0.00261 (OFF) — essentially equal and dominated by physical deformation (grid spacing ~0.029). `diagRMS < antiRMS` for both ⇒ **no bias toward the +x+z diagonal**: enabling PE/PP adhesion did NOT reintroduce the diagonal-pull artifact the unflagged-plane PT formula was originally chosen to avoid. Cloth stays flat (Yspan ~0.0596 ON==OFF) and follows the cube up during lift. No NaN.

### Decision

Step 2/3 confirmed end to end in real scenes: corner/edge adhesion active, no diagonal-pull artifact, stable, no regression. The bonded path is unchanged (8 locks). Probes were headless throwaway scripts (`/tmp`), not committed.

### Commands

- `cmake --build build/cuda_mixed_fused_pcg --target pyuipc`; `cp .../bin/libuipc_backend_cuda.so libuipc_core.so python/.venv/.../uipc/_native/`.
- `python/.venv/bin/python` headless probes over `rcc_adhesive_subdivided_cube_lift_release_demo` and `rcc_adhesive_oriented_cloth_demo` (`dump_pt_state`, `locked_pair_count`, cube height/gap stats, cloth XZ drift).

## 2026-06-02 Step 5 Bonded Lock On The VT Primitive

### Context

Final Phase 6 step: let the bonded producer lock on the full VT primitive (corner/edge contacts bond, not only face-interior PT). The ABD virtual-tet energy stays point-plane.

### Implemented

- RCC `_evolve_beta_step_at_end`: feed the producer the full `friction_VTs` topologies (extracted into `m_vt_topos`, aligned 1:1 with the per-VT `m_beta_PT`) instead of the `flag==4`-compacted face subset. The producer matches existing locks by key (`PT_pair_key(topo)`, identical for any VT), adds high-beta candidates, and conditions/rejects rest shapes (`build_rest_shape` offsets a near-coplanar corner point along the normal by `min_separate_distance`, or rejects via `det_dm_min`). Removed the `m_beta_PT_face`/`m_vt_face_flags`/`m_beta_PT_face_count` plumbing.
- Filter `filter_rcc_bonded_pt_locked_active_pairs`: now also compacts locked VTs out of the active `VTs` view (new `rcc_bonded_pt_unlocked_VT` + count, predicate `!rcc_bonded_pt_is_locked(keys, v.topo)`), restructured so the VT compact runs whenever locked keys exist (not gated on `PTs.size()`). This removes a latent double-count: a bonded VT was still in `friction_VTs` (left additive in Step 1) and would have been both adhered and bonded; now a locked pair leaves `friction_VTs` exactly as a locked PT leaves `friction_PTs`.

### Observed Result / Validation

- All gates green: backend `[rcc_bonded_pt]` 247/14, `[rcc_adhesion][oracle]` 12/4, bunny 4/1, legacy `[rcc_adhesion][gate]` 836/2, and the key bonded no-penetration scene `[rcc_bonded_pt][scene][pt_lift_release]` 596/1 (corner/edge bonding allowed, still penetration-free). Source/doc gate green.
- Headless probe (faceted subdivided cube, adhesion+bonded on, skip_ccd off): **96 bonded locks** (was 8 face-only in Step 2/3) — the full VT primitive set (incl. edge/corner) now bonds; lower cube carried through the lift (botY +0.170 -> +0.427, contact gap held ~0.019, uniform); adhesion-off separates (gap -> +0.275). No NaN.

### Decision

Step 5 complete. Bonding now decides on the VT primitive; the ABD tet stays point-plane; no double-count (locked VTs leave adhesion). Phase 6 functional milestone (Steps 0-5) done. Remaining are optional perf items (Step 3-area weight, Step 4 distance single-compute) and EE adhesion.

### Commands

- `cmake --build build/cuda_mixed_fused_pcg --target backend_cuda uipc_test_sim_case`; cp `libuipc_backend_cuda.so` -> venv.
- `uipc_test_backend_cuda "[rcc_bonded_pt]"/"[rcc_adhesion][oracle]"/"gpu_sanity_check -c bunny"`; `uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"/"[rcc_adhesion][gate]"`; headless cube probe; `run_rcc_adhesion_acceleration_gates.py`.

## 2026-06-03 Step 5 Fix — Bond Only Face-Interior VTs (Spurious Skewed Tets)

### Context

Running `python/examples/rcc_bonded_pt_lift_pull_viewer.py` showed many spurious bonded tets — apexes sticking out to the side, looking detached from their triangle.

### Source Observations (headless probe)

Dumped `dump_locked_tet_world_positions()` and measured, per locked tet, the point's perpendicular distance to the triangle plane AND whether the point's projection lands inside the triangle (barycentric). At hold: 96 locks, but only **8 had the projection inside** the triangle; **88 were outside** (projection laterally beyond the triangle footprint). Perpendicular distance was small (<= 0.019). So the artifact is not large perpendicular distance — it is the point projecting OUTSIDE the triangle, making the point-plane tet a skewed sliver whose apex sticks out sideways.

### Root Cause

Step 5 fed the producer the full `friction_VTs` with the real per-VT beta, so every VT primitive with beta>=threshold bonded. A point near a shared edge/vertex generates VT candidates against EVERY nearby triangle (one it is over = face-interior, plus several it is only edge/vertex-adjacent to). The bonded ABD tet is a point-PLANE bond, which is geometrically sound only when the point projects inside the triangle. For the edge/vertex-adjacent triangles the point projects outside -> 88 skewed sliver tets. `build_rest_shape`/`det_dm_min` did not reject them: a point offset to `min_separate_distance` above the plane but laterally outside still forms a non-degenerate (just skewed) tet.

### Fix

Mask the producer's lock-beta to 0 on non-face-interior VTs: in the topo-extraction kernel compute `dim = degenerate_point_triangle(vt.flag, off)` and feed `lockbeta = (dim==4) ? m_beta_PT : 0` (new `m_vt_lock_beta`). Only face-interior (projection-inside) VTs cross the lock threshold. The real per-VT `m_beta_PT` (adhesion on all features) is untouched, so edge/corner VTs keep their adhesion. Edge/corner contacts therefore adhere but do not bond — correct, since a point-plane stiff tet cannot soundly replace a point-near-edge/vertex contact.

### Observed Result / Validation

- Headless probe: 8 locks, all projection-inside (`inside=8/8, outside=0`), no tets with point-plane dist > 0.05; 88 edge/corner VTs adhere at beta=1.0 (removed-from-friction count: 96 total = 8 bonded + 88 adhered); lower cube carried through the lift; no NaN. Adhesion-off separates.
- All gates green: backend `[rcc_bonded_pt]` 247/14, `[rcc_adhesion][oracle]` 12/4, bunny 4/1, `[rcc_bonded_pt][scene][pt_lift_release]` 596/1, legacy `[rcc_adhesion][gate]` 836/2, source/doc gate.

### Decision

The bonded ABD tet is a point-plane bond and only face-interior (closest-feature dim==4) VTs may bond; edge/corner VTs adhere only. This is the geometrically-correct reading of "bond on the VT primitive, ABD tet stays point-plane".

## 2026-06-10 Distance-Locked Bonding Feasibility (Workflow)

### Context

Requested option: keep the bonded virtual-tet machinery but use **no adhesion energy at all**. With no adhesion there is no beta, so the lock criterion changes from `beta >= rcc_bonded_pt_beta_lock_threshold` to a pure distance test — lock a VT pair when its distance is below a user coefficient `c ∈ [0,1]` times `d_hat`. Release conditions must stay exactly as today. A 5-reader parallel source investigation (lock/release lifecycle, adhesion energy + beta lifecycle, config/frontend plumbing, distance availability, docs conventions) verified feasibility before writing the design.

### Source Observations

- The only per-step lock driver is Phase A of `IPCSimplexRCCAdhesiveContact` (`_evolve_beta_step_at_end`, driven by `RCCBetaEvolutionTimeIntegrator::do_update_state`); the only other `lock_from_rcc_pt_snapshot` caller is the asset-load seed accessor. The reporter self-unregisters (throws in `do_build`) when the contact tabular lacks `Cn` — so "no RCCAdhesive applied" currently means "no lock producer at all" (`ipc_simplex_rcc_adhesive_contact.cu:154-168, 1323-1328, 1496-1499`).
- The Phase A lock-eligibility kernel (`ipc_simplex_rcc_adhesive_contact.cu:1273-1320`) captures the end-of-step positions viewer unconditionally (`:1280`) and loads all 4 VT vertex positions when the face-interior barycentric gate is on (`:1289-1303`, default off), masking a lock-beta (`lockbeta(i) = eligible ? beta(i) : 0`); per-vertex `d_hats()` are fetched 150 lines earlier in the same function (`:1120`; `thicknesses()` is exposed by the same `GlobalVertexManager` handle and would be a new fetch), and `PT_d_hat`/`PT_thickness` reductions are the standard per-pair pattern (`utils/primitive_d_hat.h`, `utils/codim_thickness.h:40` — `PT_thickness` is the SUM of the two half-thicknesses). A distance gate there needs zero new buffers — consistent with the structural single-compute rule (no per-pair `d^2` buffer anywhere in the codebase).
- The bonded-side lock select reads only `entry.beta >= thr && release_flags == None && !released-this-call` (`rcc_bonded_pt_system.cu:605-634`); `thr` can be overridden per pair by the tabular `bonded_lock_threshold`. An indicator lock-beta (1.0/0.0) reuses this machinery unchanged.
- All release gates (`release_flags_from_current_shape`, `rcc_bonded_pt_system.cu:157-362`: strain/gap/slip/force/flip/degenerate/sticky-side/policy) read **no beta** — "release unchanged" is structural.
- Existing disable paths cannot express the mode: per-pair `adhesion_enabled=0` pins beta to 0 (Phase B `:477-481, 632-636`; Phase A `:1149-1150`) so the beta lock never fires, AND the policy release gate (`rcc_bonded_pt_rcc_policy_enabled`, `rcc_bonded_pt_system.cu:140-147, 300-305`) force-releases locks on disabled pairs; `Cn=0` divides by zero in `PT_beta_evolve_existing` (`denom = eta*W_scale/10`, `W_scale ∝ Cn`, `codim_ipc_simplex_rcc_adhesive_function.h:57, 91-97`).
- Candidates: `friction_VTs()` is recorded each frame whenever `contact/friction/enable` is on — the default; the recording lambda is gated on `m_friction_enabled` (`advance_ipc.cu:57-63`, frame 1 seeded by an initial DCD at `:313-315`) — and is independent of adhesion settings. This is not a new prerequisite: the adhesive reporter is a `SimplexFrictionalContact` subclass whose build already requires friction. The DCD activity band is `ξ < d < ξ + d_hat`, strict on both ends (`D_range`/`is_active_D`, `codim_thickness.h:101-114`), so a bare `d < c·d_hat` gate is unreachable for pairs with `ξ >= c·d_hat` — the predicate must be `d < ξ + c·d_hat` (reduces to the requested `d < c·d_hat` at `ξ = 0`).
- `ActiveVT.flag` is the lagged begin-of-step classification; `point_triangle_distance_flag` is a pure function of the 4 positions, so an end-of-step flag recompute is cheap and matches the lock's end-of-step semantics.
- Beta is otherwise opaque payload: entry struct, bridge `locked_beta()`, released-beta carry, accessor dump/seed — all keep working with a sentinel value; `merge_released_beta`'s only consumer (Phase B matching) does not run in the new mode.

### Decisions

- Lock driver stays Phase A of the adhesive reporter; `RCCAdhesive::apply_to` remains required (tabular provides `adhesion_enabled`, per-pair `bonded_*` overrides, sticky machinery); `Cn`/`Ct` stay 0 and are never read because beta kernels are skipped (verified: the energy/assemble kernels early-out on `beta <= 0` before any `Cn` read, and the only `Cn` denominators live in the skipped beta-evolution path). The mode warns at build when `Cn`/`Ct > 0` is set alongside it (contradictory config, coefficients ignored).
- New scene-config keys (Planned): `rcc_bonded_pt_distance_lock` (IndexT, 0) and `rcc_bonded_pt_distance_lock_ratio` (Float, 0.5, clamped `[0,1]` on read like `rcc_adhesion_normal_offset_coeff`).
- Lock predicate `D < (ξ + c·d_hat)²` on the flagged true closest-feature distance with the flag recomputed from end-of-step positions; emitted as indicator lock-beta so the bonded producer's select, rest-shape conditioning, age/dedup, and the whole release path run unmodified. Locked entries carry sentinel beta 1.0. The global lock threshold is passed as `min(rcc_bonded_pt_beta_lock_threshold, 1.0)` so a beta-mode value `> 1` cannot silently veto indicator locks; a per-pair `bonded_lock_threshold > 1` stays a deliberate per-pair veto. `seed_locks` round-trips across modes (it re-thresholds against its own argument).
- Face-interior mask, cross-layer occlusion, and `adhesion_enabled` eligibility compose with the distance gate. The occlusion test currently lives INSIDE the Phase B beta-init kernels (early-returns on `!rcc.enabled`/sticky-gate precede it); in distance mode it is extracted into a beta-free pass over positions/shell-triangles/sticky-signs/lagged-normals (verified beta-independent), replicating those early-outs. The eligibility kernel checks `enabled` explicitly instead of inheriting it through beta pinning. Distance-mode scenes should set `rcc_bonded_pt_lock_face_interior_only = 1`: without beta's multi-step integration, the default `0` mass-locks edge/corner sliver tets on first contact (the 2026-06-03 failure mode).
- Adhesion energy bypass is atomic: zero `friction_pair_counts` AND gate the `do_compute_energy`/`do_assemble` kernel bodies together — the kernels iterate `friction_VTs().size()` but write into subviews sized by the reported counts, so zeroing counts alone writes out of bounds (verified against `simplex_frictional_contact.cu:44-168`; zero counts are harmless to OTHER reporters, the dytopo offsets are scan-based per-reporter subviews). Phase B init and Phase A evolution are skipped together (the evolve kernel reads `m_beta_PT`, sized by Phase B), plus the beta carry. The lagged vertex-normal recompute and release-context wiring (sticky signs, masks, tabular) keep running — they live outside the beta kernels and feed the sticky-side/policy release gates.
- Documented caveats: no temporal hysteresis (`rcc_bonded_pt_release_gap` is a growth threshold relative to the lock-time gap, so `release_gap > c·d_hat` is the conservative anti-churn condition; min-age gate composes when it lands), no load-based relock suppression after force release, frame-1 locking possible.
- Beta-carry contracts scoped instead of violated: the Release And Beta Carry Contract (architecture), roadmap Core Principle rules 4/5, the Frame Lifecycle Phase B row, the lifecycle Released row, and the `locked_beta` data-layout row all gained explicit "(beta mode; vacuous/sentinel in distance-lock mode)" qualifiers.
- Design of record: [architecture](../architecture.md) "Distance-Locked Bonding Without Adhesion Energy"; checklist + Lock Gate row + Reports row in [rcc_adhesion_acceleration](../rcc_adhesion_acceleration.md); Phase 7 + Planned Gates row in [roadmap](../roadmap.md); keys + Test Matrix row in [conventions](../conventions.md). No code change in this slice.

### Commands

- 5-reader parallel source investigation (workflow `rcc-distance-lock-feasibility-read`), findings cross-checked inline against `rcc_bonded_pt_system.cu`, `ipc_simplex_rcc_adhesive_contact.cu`, `simplex_trajectory_filter.cu`, `advance_ipc.cu`, `codim_thickness.h`.
- 5-agent adversarial verification pass (workflow `rcc-distance-lock-verify`): 4 claim verifiers re-derived every load-bearing claim and file:line citation from source; 1 doc critic checked the five doc locations for internal/contract consistency. Corrections folded back in: friction-enable prerequisite (recording is NOT unconditional), conditional position loads in the eligibility kernel, `thicknesses()` is a new fetch, counts+kernels must change atomically, occlusion extraction (not "keep the kernel"), threshold `min(value, 1.0)`, beta-carry contract scoping, face-interior recommendation, growth-threshold churn advice.
- `python3 scripts/run_rcc_adhesion_acceleration_gates.py` — source/doc gate green after the doc additions.

## 2026-06-11 Phase 7 Implementation — Distance-Locked Bonding

### Context

Implemented all five Phase 7 steps from the 2026-06-10 feasibility design: distance-locked bonding with no adhesion energy (`rcc_bonded_pt_distance_lock` + `rcc_bonded_pt_distance_lock_ratio`), with the lock driver staying Phase A of `IPCSimplexRCCAdhesiveContact`. A 4-dimension adversarial diff review (correctness-vs-design / GPU / physics / style, each finding re-verified by a dedicated refuter agent) ran on the implementation before landing; its confirmed findings were folded back in.

### Source Observations

- One real bug found and fixed during bring-up: the producer's lock select replaces the global threshold with the per-pair `bonded_lock_threshold` whenever the release context provides the tabular (`use_tabular`, hard-coded on in Phase A), and the per-pair sentinel `-1` resolves to the RAW global `rcc_bonded_pt_beta_lock_threshold` — so a beta-mode global value `> 1` silently vetoed every indicator lock even though the producer-call scalar was clamped. Fix: clamp the sentinel fallback `G_LOCK` to `<= 1` in `_rebuild_adhesive_tabular` when distance mode is on; explicit per-pair values keep their raw meaning (`> 1` = deliberate veto). The scene gate deliberately sets the global to `1.5` to pin this.
- First scene-gate run also exposed a band-vs-physics mismatch: the cube fixture's press equilibrium gap is ~0.0178 (the IPC barrier at `d_hat = 0.02` carries the press load well before contact), so `c = 0.5` (band 0.01) rejected ALL 6144 candidates (`distance_rejected_count == candidate_count`, zero locks) — the counter pipeline diagnosed it directly. The gate runs at `c = 0.95` (band 0.019), which still leaves the `(0.019, 0.02)` DCD-active-but-outside-band window that the approach transits (keeps the rejection counter meaningful).
- A stale-binary race also burned one debug round: a source edit made while a build was running produced an `.o` newer than the edit, so the next incremental build skipped recompiling it — re-touching the file healed it. Symptom signature: counters consistent with the OLD predicate.
- The legacy `[rcc_adhesion][gate]` suite flaked once (668/669 on one run, 836/2 green on two other runs of the same binary) — tolerance-edge flake of the physics fixture, not a regression; beta-mode regressions (`[rcc_bonded_pt]` backend 265/16, `pt_lift_release` 596/1, oracle 12/4) are green.

### Decisions

- Eligibility kernel gate order (cheap→expensive): `adhesion_enabled` (→ `policy_rejected_count`), sticky-side parity (uncounted; beta mode never bonds sticky-fail pairs because Phase B pins their beta — distance mode replicates the behavior explicitly), face-interior foot (shared `VT_lock_face_interior_pass`, also rewired into the beta-mode mask), distance band (shared `VT_distance_lock_band_pass`, structurally false at `c <= 0`; → `distance_rejected_count`), fused occlusion cast (shared `VT_occlusion_blocked`, extracted from the two Phase B kernels and rewired there too — review caught the third hand copy; → `distance_rejected_count`).
- Occlusion timing in distance mode is end-of-step positions + freshly recomputed normals (not Phase B's frame-open snapshot) — deliberate, recorded in the architecture section: the lock is a fresh end-of-step decision; a pair that became occluded during the step is now correctly rejected, one that became clear locks one frame earlier.
- Energy bypass is atomic (`friction_pair_counts` = 0 AND gated `do_compute_energy`/`do_assemble` bodies); Phase B init + Phase A evolution + beta carry skipped together; vertex-normal recompute and release-context wiring kept (they feed the unchanged sticky/policy release gates); `m_blocked_PT`/`m_last_seen_frame` are beta-mode-only state (comments annotated).
- Rejection counters live in `RCCBondedPTCounters` (`distance_rejected_count`, `policy_rejected_count`; report names `rcc_bonded_pt_rejected_{distance,policy}_count`), fed via `RCCBondedPTSystem::add_lock_rejection_counts` from persistent `DeviceVar`s (no per-step alloc churn), exposed in the pybind dict, covered by the `[rcc_bonded_pt][state][counters]` contract fixture (28/1).
- Contradictory-config warning fires only for adhesion-ENABLED rows with `Cn`/`Ct > 0`.

### Commands

- Build: H100 (`hpc-low`), full `cmake --build` RelWithDebInfo, clean.
- Gates green: `[rcc_bonded_pt][oracle][distance_lock]` 18/2 (band predicate vs exact brute-force closest-distance reference — all probe feet on grid nodes — plus face-foot gate incl. degenerate triangle); `[rcc_bonded_pt][scene][distance_lock]` 604/1 (bonds by distance with `Cn = Ct = 0`, threshold-clamp pin at global 1.5, carried lift, sentinel beta 1.0 in the accessor dump, live rejection counter, forced-pull release via unchanged gates, zero locks at full separation, penetration-free with CCD skipped); `[rcc_bonded_pt][state][counters]` 28/1; regressions `[rcc_bonded_pt]` 265/16, `[rcc_bonded_pt][scene][pt_lift_release]` 596/1, `[rcc_adhesion][oracle]` 12/4, `[rcc_adhesion][gate]` 836/2.
- `python3 scripts/run_rcc_adhesion_acceleration_gates.py` — source/doc gate green after the status flips.

## 2026-06-11 Distance-Lock Wind/Drop/Unwind Probes (temflex175-2turn-e5e7-dhat2-cnct1)

### Context

First end-to-end asset pipeline runs of the implemented Phase 7 mode on the soft 2-turn tape preset (H100, headless EGL): wind -> saved asset -> drop, then wind asset -> unwind (peel).

### Source Observations

- Wind, c = 0.5 (default): ZERO locks, save-time max|v| = 0.33 m/s (the released roll was unspooling — nothing holds it in this mode without bonds). Wound layers rest at pitch LAYER_THICKNESS = 2.5e-4, i.e. d − ξ ≈ 1.6e-4 = 0.89·d_hat (d_hat = 2·thickness = 1.8e-4), so a 0.5·d_hat band is unreachable — the same press-equilibrium lesson as the cube scene gate, diagnosed directly by `distance_rejected_count == candidate_count`.
- Wind, c = 0.95: 1379 locks, save-time max|v| = 1.0e-5 m/s; settle1 max|v| ~ 1e-9 (the bonded roll is effectively rigidified — quieter than the beta-mode wind). 2630 frames in ~106 s wall-clock. Sentinel beta 1.0 round-trips the asset dump (`frac>0.9 = 100%`).
- Drop: auto-detects the mode from the asset, reseeds all 1379 locks, survives hold/pull/top/freefall intact, lands as a coherent roll. PNGs/mp4 under `output/distlock_run/`.
- Unwind (peel): does NOT peel — the documented "no load-based lock suppression" caveat, now with data. Lock-count trace during the pull: 1396 at seed -> RISES to ~1568 mid-pull -> stable 1524 at settle; force releases (RCC_RELEASE_FORCE = 3e-7 from the asset) are cancelled by next-step relocks while the peel-front pairs are still inside the band, and the pull itself presses more pairs into the band. Visually the slack segment straightens but the wound body never opens. Beta mode peels here because tension drives beta down and a released pair does not immediately relock.

### Decisions

- Distance-lock mode is fit for wind/drop/hold-style workloads (grab, hold rigidly, release on gross separation); peel-style workloads need beta mode — or a future released-pair cooldown / load-based lock suppression if peel-under-distance-lock becomes a requirement. Recorded as-is; no code change.
- `rcc_adhesive_tape_unwind_demo.py` gained the same DISTANCE_LOCK wiring as the drop demo (CLI > asset > default) and a per-progress-line bonded-lock count in the headless record path (the live peel-front signal that made this diagnosis one log read).

### Commands

- `run_wind_drop.sh` (output/distlock_run): wind + drop, `--set DISTANCE_LOCK=1 --set DISTANCE_LOCK_RATIO=0.95 --set LOCK_FACE_INTERIOR_ONLY=1`.
- `rcc_adhesive_tape_unwind_demo.py --preset default --asset .../temflex175-2turn-e5e7-dhat2-cnct1-distlock.npz --set RECORD_DIR=... --set RECORD_ZOOM=1.2` — UNWIND_OK (sim stable), peel stalled per the lock trace above.

## 2026-06-11 Tension-Only Release + Band-Edge Rest (xi + d_hat)

### Context

Two structural fixes driven by the rod-wind seam crash and the release/relock churn diagnosis (drop release-force sweep on the CCD-wound b09 asset):

- The `force`/`strain` release criteria used direction-blind norms (`||C F||`, `||C||`): a compressed or sheared bond accumulated "release force" exactly like a stretched one. The deterministic rod-wind crash at the spiral seam (tape row 70 = end of the innermost turn pressed against the hub outer wall at the root azimuth, `V-F=(1854,673,769,768)`) was a bond released BY COMPRESSION while inside the thickness shell — the next DCD pass saw an unlocked pair at `D < xi^2` and aborted. Identical signature with CCD on (rodwind24) and with auto-skip (rodwind25), proving the asserting pair was unlocked.
- Rest shapes froze the creation-time geometry, so a released pair was still inside the lock band and relocked next frame: release/relock churn. In beta mode this produced the "self-healing creep" of the drop sweep (rf <= 1e-9: locked count oscillating 400 <-> 3400 while the coil slid apart into a loose loop); in distance mode it is the documented peel stall.

### Changes

- `release_flags_from_current_shape` (rcc_bonded_pt_system.cu): hoisted the rest/current normal-gap computation (was gap/slip-only) and gated the `strain` and `force` flags on tension (`|curr_dist| - |rest_dist| > 0`). Compression and pure shear never release; `gap` (tensile by construction), `slip`, `flip`, `degenerate`, `sticky_side`, `policy` are unchanged.
- `build_rest_shape` / `rest_shape_is_valid`: new `rest_height_target` input. The producer passes `PT_thickness(topo) + d_hat` (per pair, from `GlobalVertexManager::thicknesses()` + `GlobalContactManager::d_hat()` via new Impl slots): the rest point is placed at exactly the band edge, side-preserving. A tension release therefore happens at a gap past the band edge — outside both the lock band and the candidate set — so released pairs cannot relock in place. Fallback (no contact manager / thickness data): legacy `min_separate_distance` clamp.
- Core oracle mirror (`build_rcc_bonded_pt_rest_shape_svts`): same `rest_height_target` rule, keeping the backend-vs-oracle contract.
- Born-compressed bonds (creation gap < band edge by definition of the lock) push outward toward the band edge after locking; the tension gate makes this birth compression structurally unable to fire a release.

### Tests

- `[rcc_bonded_pt][oracle][rest_shape]` band_edge_target: conditioned point lands at exactly the target height on its own side, with matching rest volume.
- New `[rcc_bonded_pt][scene][tension_release]` cube gate: press (compression overload ~0.5 in F-space units, threshold 1e-3 -> the old criterion would release everything on contact) -> zero releases; lift/hang carried; forced pull releases by force; zero locks at the end with no relock.

### Drop sweep baseline (pre-change, beta mode, CCD-wound b09 asset, SKIP_CCD=0)

- rf 1e-7 / 3e-8: clean (locked 3802/3808 flat, intact roll hangs).
- rf 1e-8: passes with churn (HOLD releases ~650, heals; locked ~4150 at top).
- rf 3e-9 / 1e-9 / 1e-10: coil disintegrates into a loose loop/fold but hangs by self-healing relock churn. Minimum usable release force ~1e-8 under OLD semantics; the new tension gate is expected to move this floor down (hang tension is what matters, press/compression no longer counts) — re-sweep pending.

### Follow-up: side-face bonds made the plane-distance opening measure immortal

First gate runs of the tension gate failed `end locked == 0` (48 of 96 survived) in
both the tension gate and the distance-lock gate. Survivor dump (topo/age/current
plane distance vs rest): all survivors age=349 (born at press), plane distance
frozen at ~7e-5 while the cubes had FULLY separated (end gap_y = 1.03). Diagnosis:
these are SIDE-FACE bonds — a rim vertex locked against the other cube's vertical
side plane (the beta gates do not run the face-interior lock gate). When the pull
separates the cubes vertically, such a pair tears apart TANGENTIALLY to its
triangle plane: the point-plane distance never grows, so the plane-based opening
measure said "compressed" forever and the tension gate vetoed every release —
regardless of force ("拉的力再大也没用": the force criterion is only evaluated
after tension=true; force is not an input to the opening test).

Fix: the opening measure now uses the TRUE point-triangle closest distance
(`point_triangle_closest_point`) against the rest height, for both the tension
gate and the `gap` criterion (which had the same blind spot — plane distance —
since its introduction; pre-change scenes never noticed because the
direction-blind force release cleaned up torn side pairs by overload). A pair
torn apart in ANY direction registers growing separation and releases; a pressed
pair (true distance below rest) still never does.

Note: an earlier hypothesis in this entry's drafting blamed AOP compression
snap-through ("crushed zombie bonds"); the survivor dump disproved it — the 7e-5
was a side-pair plane distance, not a crushed gap. The AOP compression branch
non-convexity (force peak at lambda = 1/sqrt(3), flat basin at lambda -> 0)
remains a real property worth a compression-floor guard if crush is ever
observed, but it was NOT the failure mechanism here.

## 2026-06-12 Seam-Pair Forensics — Opening Metric Must Be Region-Clamped

### Context

After the tension gate landed, two scene gates failed identically: forced pull
separates the cubes fully (`gap_y = 1.03`) yet exactly 48 of 96 locks survive
forever (`released_count` frozen at 48). Survivor dump (state accessor +
locked-tet world positions): all age = 349 (born at press, never relocked),
all reporting `curr ~ -7e-5` against `rest = -0.02`.

### Diagnosis

The 48 survivors are NOT interface pairs — they are seam pairs between the two
cubes' COPLANAR side faces (the cubes share a footprint, so their side faces
are flush; near the interface corner a side vertex of one cube lies in the
side-face plane of the other at plane distance ~0). Under vertical separation
their point-PLANE distance never changes: a plane-distance opening test reads
"compressed" forever, so the tension gate vetoes force release permanently.
The legacy direction-blind force release happened to free them because
tangential tearing also accumulates `||C F||` — the veto closed exactly that
accidental escape hatch.

Second finding from the same forensics: the friction helper
`point_triangle_closest_point` is an UNCLAMPED least-squares plane projection,
so `(P - closest).norm()` IS the plane distance — an opening metric built on
it is a no-op for seam pairs.

### Fix

`release_flags_from_current_shape` measures opening as the growth of the TRUE
region-clamped point-triangle distance (`distance::point_triangle_distance_flag`
+ flagged `point_triangle_distance2`, the same machinery as the distance-lock
eligibility kernel) on BOTH ends: current shape and rest shape. Rest-side
clamping keeps an edge-born rest foot at zero opening at rest instead of
spurious tension. Tangentially torn pairs now register opening and release by
force/gap; genuinely pressed pairs still never release.

### Also Ruled Out On The Way

- Newton-cost theory of the slow suite: per-gate probe (INFO stream + summary)
  shows all three bonded gates at mean 1.6 iters/frame, zero line-search-max,
  ~10 s each on RTX PRO 6000; the only slow gate is the pre-existing pure-beta
  cube-cloth lift (78 s, lift-phase mean 18.1). Born-compression from the
  band-edge rest costs nothing measurable -> rest stays `xi + d_hat` (the
  `xi + min(c+eps,1)*d_hat` variant buys ~nothing and loses the
  exits-candidate-set guarantee).
- sm_120 dev build (`build-rtx/`, RTX PRO 6000 Blackwell, rtx partitions = same
  14 nodes at three preemption tiers) mirrors `build/` and passes the fast rcc
  suites; scene-gate work and sweeps migrate to RTX nodes.

### Post-Fix Verification And Boundary Re-Sweep

- Gates after the region-clamped opening fix: `[rcc_bonded_pt][scene][distance_lock]` 604/1, `[tension_release]` 421/1, `[pt_lift_release]` 596/1; `uipc_test_backend_cuda "rcc_*"` 277/20; `uipc_test_core "rcc_*"` 131/8 (all on the sm_120 build, RTX PRO 6000).
- Drop-lift boundary under the new semantics (CCD-wound b09 asset, SKIP_CCD=0): clean hold at release_force 1e-5 / 3e-6 (zero events), 1e-6 (~28 ambient releases, <1%), 3e-7 (~770 releases = 20% attrition, still churn-free and full-height hang); cascade transition between 3e-7 and 3e-8 — at 3e-8 and below, mass release -> structural push-back into the band -> beta>=0.9 relock -> creep ratchet (the coil pays out and sags). Minimum usable release force ~3e-7, recommended 1e-6+. Thresholds now measure net tension above the band-edge prestress (ambient scale 4*kappa*V0*dt^2*2*(d_hat-pitch)/rest ~ 1.3e-6 for this asset), NOT absolute bond load — old-semantics values (1e-7..1e-8) are not comparable.
- Open item (pre-existing suspect): running ALL sim_case rcc tests in ONE process (`uipc_test_sim_case "rcc_*"`) fails 3 beta-demo tests (cloth_peel, smoke, pick_and_lift) that pass when run individually (smoke 31/31 and pick_and_lift 522/522 verified). cloth_peel could not be verified solo: it grinds at ~160 s/frame in its hold phase on RTX PRO 6000 (two 13x13 cloths pressed to 0.012 with d_hat=0.02 -> every vertex in band, beta attraction vs contact repulsion never converges; full section would take ~10 h) -- its own pathology, and irrelevant to the bonded change by code path: the scene never enables rcc_bonded_pt, so RCCBondedPTSystem never builds and none of the changed kernels execute. These tests predate the bonded work, do not enable rcc_bonded_pt, and there is no record of this 8-test batch ever being green in-process — suspected cross-Engine state interference in the test harness, tracked separately from the bonded semantics change.

### In-Shell Locked Pairs vs The Collision Pipeline (rod-wind end-to-end)

Running the rod-wind demo end-to-end on the new semantics surfaced three
independent faults, each killed by the next deeper one:

1. **Seed index remap.** Saved bonded-lock topos are GLOBAL vertex indices
   from the winding scene, whose backend layout is `[hub 1056 | tape 1980]`
   — ABD bodies are numbered BEFORE FEM geometry regardless of Python
   creation order (decoded from the asset itself: lock indices use only the
   hub's outer-ring verts, period-96 half-used gap pattern; zero
   tri-straddlers at the 1056 boundary). The rod demo inserts extra ABD
   bodies (rod, carrier) into that block, shifting every tape index. Fix:
   remap `idx >= hub_nv -> idx + inserted_nv` before `seed_locks`; the
   backend additionally rejects (instead of device-asserting on) seeded
   triangles with non-uniform vertex thickness — the signature of a wrong
   layout — via a negative rest-height sentinel.
2. **DCD classify assert.** Band-edge rest makes "locked pair inside the
   thickness shell" a LEGAL state, but the AllP-AllT classify kernel's
   `D > xi` assert runs BEFORE the locked-pair post-filter and aborted the
   process the moment the orbit peel front compressed a seam lock (true
   distance a hair under xi; D prints are squared). Fix: all four BVH
   trajectory filters skip locked candidates at the source (also covers the
   PE/PP regions the post-filter never stripped);
   `rcc_bonded_pt_locked_keys()` moved from DetectInfo to BaseInfo so the
   FilterActive path can see it.
3. **CCD TOI degeneracy / Zeno.** With SKIP_CCD=0 an in-shell locked pair
   makes the thickness-xi ACCD root-find degenerate (toi=0 -> engine
   assert). A fractional retarget (stop at c*current_distance) fixes the
   assert but creates Zeno: the global line-search alpha is min over all
   pairs, so one compressed lock clamps the WHOLE system to ~160 s/frame.
   Resolution (design decision): locked pairs get CCD thickness ZERO — TOI
   only blocks an actual surface crossing; keeping the pair off the surface
   is the bond energy's job, i.e. kappa/d_hat must supply the equivalent
   barrier. Tuning rounds re-wind the asset, then re-run rod-wind.
4. **First tuning round** (`-dhat3-k3e8`: D_HAT_RATIO 2->3, kappa 1e8->3e8,
   re-wound): wind clean (3556 locks, beta 0.988, settled to 1e-5 m/s);
   rod-wind runs ~5 frames/s vs 0.25 before — the deep-compression slow
   zone is gone. Release thresholds for this asset need re-calibration
   (prestress scale moved ~4-6x); the old boundary table applies to the
   dhat2/k1e8 asset only.
5. **Fold separation + carrier orbit (in progress).** The folded free end
   tip-peels open after squeeze release (recoil moment vs the low
   rf=1e-7); lowering the rod 3 rows hit the hub — keep auto height.
   Orbit redesigned: the hub's STC disengages at orbit entry and an
   STC-driven ABD carrier (R = half the hub hole, length 4*HUB_HEIGHT)
   through the hub hole sweeps the circle; orbit radius auto-adds the hole
   slack so the tape stays taut. Validation run in flight.

### Revolute Bearing Lands The Full Rod-Wind Sequence

The carrier-contact bearing ground Newton regardless of lock mode/count
(983-lock dlock and 3556-lock beta assets both stalled at orbit frames
940-1140; kappa x10 on the bonds did not help): the hub's spin against a
rigid-rigid line contact + friction is a nearly flat energy direction.
Fix: `CARRIER_JOINT=1` (default) replaces the hub-carrier contact with an
`AffineBodyRevoluteJoint` on the carrier axis and disables that contact
pair — a bearing is a joint, not a contact. Result: first-ever full
2460-frame completions, in parallel —

- jointA (dlock asset, kappa 3e8): fold ADDS ~46 locks (low bending 500
  fixed the tip-peel), orbit peels 1042 -> 940 smoothly, zero frames over
  200 iters.
- jointB (beta asset 3556 locks, bend 500 runtime): completes with mild
  release/relock oscillation (1471 -> 1168), transient spikes only.
- jointC (k3e9 asset, kappa 3e9): completes too (659 -> 438, more
  aggressive peel at the stiffer kappa), zero frames over 200 iters.
- Contact-mode control with kappa 3e9: still grinding at the same zone —
  killed.

Drop cross-check on the new assets (rf=1e-7, kappa 3e8): dlock asset holds
980/983 locks; beta asset releases 2/3 (prestress rescale) — per-asset
release-force re-calibration confirmed necessary.

### Speed Sweep (wind + 1-turn rod-wind, joint bearing, 8 configs)

Timing of the 1710-frame 1-turn sequence, RTX PRO 6000, node-to-node noise
~±15% (same-config repeat: 766 vs 885 s):

| config                  | rod time | notes |
|-------------------------|----------|-------|
| dhat2 + kappa1e8 (s8)   | 766/885s | fast, healthy (x2 measurements) |
| dhat3 + kappa1e8 (s4)   | 842s     | fast, healthy — RECOMMENDED |
| dhat4 + kappa3e8 (s3)   | 1017s    | |
| dhat3 + kappa3e8 (s1)   | 1049s    | old baseline |
| interior0 (s7)          | 1112s    | knob is a wind-time no-op (983 locks unchanged) |
| dhat2 + kappa3e8 (s2)   | 1413s    | small d_hat is NOT free: steeper barrier curvature |
| dhat3 + kappa1e9 (s5)   | 1548s    | slowest + over-releases (1023 -> 745) |
| ratio 0.8 (s6)          | (842s)   | DEGENERATE: 4 locks wound — lock ratio must stay ~0.9-0.95 |

Kappa is monotone in cost (1e8 < 3e8 < 1e9, 20%/45% gaps) and 1e8 is also
the healthiest dynamically. d_hat is non-monotone (2x slower at kappa 3e8).
Production recipe: D_HAT_RATIO=3, RCC_KAPPA=1e8, DISTANCE_LOCK_RATIO=0.95,
BENDING_STIFFNESS=500, CARRIER_JOINT=1. Full 10-turn helical validation
(WRAP_PITCH = half tape width) running on this recipe.

Infra note: parallel sweeps exposed a cross-node pip race (setuptools
builds inside the shared NFS source dir -> "dist-info File exists"); jobs
now copy the python dir to node-local /tmp before pip install.

### Speed Sweep Extension (lower kappa / lower d_hat)

| config            | rod time | health |
|-------------------|----------|--------|
| dhat2 + kappa3e7  | 598s     | locks grow 1204->1429, no release events |
| kappa1e7 (dhat3)  | 629s     | locks grow monotonically — release DEAD |
| kappa3e7 (dhat3)  | 676s     | similar growth pattern |
| dhat1.5 + kappa1e8| 907s     | slower than dhat3 — d_hat sweet spot is 2-3x |

Kappa curve flattens below 3e7 (1e9:1548 / 3e8:1049 / 1e8:842 / 3e7:676 /
1e7:629). BUT: bond force scale ~ kappa, so at fixed RCC_RELEASE_FORCE=1e-7
the release gate stops firing somewhere below kappa 1e8 — the 1-turn runs
show pure relock growth. Speed at kappa<=3e7 is only usable after
re-calibrating the release force down with kappa. Recommendation stands:
kappa 1e8 + dhat3 (842s, peel intact) as the no-recalibration recipe;
kappa 3e7 + dhat2 (598s, -43% vs old baseline) once rf is re-swept.

### Speed Sweep Wave 3 + Correction

CORRECTION to the wave-2 health reading: the aggregate locked count mixes
peel RELEASES with NEW locks formed as tape winds onto the rod — net growth
does NOT mean release is dead. Frame review of the kappa3e7+dhat2 2-turn
every-frame video (rodwind_s12full.mp4) confirms peeling is healthy.

Wave 3 (1-turn timing, joint bearing):

| config            | rod time |
|-------------------|----------|
| dhat2 + kappa3e6  | 464s  (0.27 s/frame) |
| dhat2 + kappa1e7  | 492s  |
| dhat2 + kappa3e7  | 598s  (wave 2) |
| dhat2.5 + kappa3e7| 652s  |

Kappa speed curve flattens below 1e7 toward a ~450-460s fixed-cost floor;
d_hat monotone 2 < 2.5 < 3 at low kappa. Speed recipe: dhat2 + kappa
1e7-3e7. Physical-quality check of the soft-kappa roll (layer slip / sag)
still by eye, not by counters.
