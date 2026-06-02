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
