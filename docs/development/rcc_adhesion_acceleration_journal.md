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
- `SoftVertexTriangleStitch` already builds vertex-triangle tetrahedra, applies `min_separate_distance`, stores `Dm_inv` and rest volume, and uses SPD-projected Stable Neo-Hookean Hessians.
- Inter-primitive constitutions report complement energy, which is the right ownership model for a bonded virtual tet that replaces contact/RCC work but is not contact itself.

### Decisions

- Treat `SoftVertexTriangleStitch` as the numerical reference, not as the runtime container. Dynamic adhesion pairs should not rebuild frontend geometry.
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

Treat these as current legacy RCC behavior gates, not as proof of bonded PT acceleration. At this point the bonded path still needed `RCCBondedPTState`, key/topology/beta/age/release fixtures, SVTS-compatible rest-shape and virtual tet E/G/H oracles, `rcc_bonded_pt_*` counters, and release reason fields before the final `pt_lift_release` gate could claim pair ownership correctness.

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

Keep this as a host-side contract for now. The next step is the SVTS-compatible rest-shape CPU oracle; CUDA device buffers and filter/reporter integration should wait until state, rest-shape, and E/G/H oracles are all executable.

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

Use this rest-shape oracle as the reference for future dynamic bonded PT lock construction. The next implementation step is the virtual tet Stable Neo-Hookean energy, gradient, and Hessian CPU oracle.

## 2026-06-01 Virtual Tet E/G/H CPU Oracle

### Context

After rest-shape construction is deterministic, the bonded PT reporter needs a CPU reference for the virtual tet complement energy. This oracle should match the `SoftVertexTriangleStitch` Stable Neo-Hookean path, including `F = Ds * Dm_inv`, `dFdx`, `rest_volume * dt^2` scaling, and SPD projection of the F-space Hessian for the default Hessian path.

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

The CPU oracle layer is now sufficient for the next implementation step. Before touching filter kernels, add observable `rcc_bonded_pt_*` counters and release flag plumbing so later scene and benchmark gates can prove pair ownership.

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
