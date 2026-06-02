# Conventions

These conventions are enforceable rules for RCC bonded point-triangle acceleration. If a rule cannot be tested, reported, or reviewed in code, rewrite the rule before implementing around it.

## Hot-Path Rules

1. Do not rebuild frontend geometry to represent transient bonded pairs.
2. Do not host-copy PT pair lists during Newton iterations.
3. Do not allocate per-candidate buffers inside PT candidate kernels.
4. Do not log per-pair diagnostics in production kernels.
5. Do not implement backend-specific locked-key semantics; all simplex filters must call one shared lookup helper.
6. Do not benchmark a path that removes locked PTs after contact/RCC assembly as if it were the production path.
7. Do not enable bonded PT acceleration by default until state, oracle, filter, scene, and benchmark planned gates pass.
8. Do not put `Matrix3x3`, rest volume, or other fat payloads in a CUB radix-sort value type; keep sorted device entries small and store rest-shape payloads in SoA buffers.

## Data Layout Rules

Runtime state uses structure-of-arrays device buffers.

| Buffer | Type | Rule |
| --- | --- | --- |
| `locked_keys` | `DeviceBuffer<U64>` | Sorted membership keys used for filter lookup |
| `locked_topos` | `DeviceBuffer<Vector4i>` | Oriented topologies zipped/permuted with `locked_keys` |
| `locked_beta` | `DeviceBuffer<Float>` | Beta carried from RCC persistence and back on release |
| `locked_age` | `DeviceBuffer<IndexT>` | Count of accepted stable steps, not wall-clock frames |
| `Dm_inv` | `DeviceBuffer<Matrix3x3>` | Built only after rest-shape conditioning passes |
| `rest_volume` | `DeviceBuffer<Float>` | Positive and above minimum volume |
| `release_flags` | `DeviceBuffer<U32>` | Bit mask or enum, stable enough for tests and reports |

Sorting keys alone is forbidden. Any sort of `locked_keys` must carry or recover the permutation for `locked_topos`, `locked_beta`, `locked_age`, `Dm_inv`, `rest_volume`, and release metadata. The current CUDA producer sorts a compact `RCCBondedPTDeviceEntry` value containing only key/topology/beta/age/release flags, then recovers rest-shape payloads by sorted key in the bridge.

## Naming Rules

Use the `rcc_bonded_pt` prefix for feature names, config keys, counters, tests, and benchmark fields.

| Element | Convention | Example |
| --- | --- | --- |
| Config | `rcc_bonded_pt_<name>` | `rcc_bonded_pt_beta_lock_threshold` |
| Counter | `rcc_bonded_pt_<event>_count` | `rcc_bonded_pt_filter_skip_count` |
| Timer | `rcc_bonded_pt_<stage>_ms` | `rcc_bonded_pt_assembly_ms` |
| C++ test tag | `[rcc_bonded_pt][layer]` | `[rcc_bonded_pt][filter]` |
| Python script | `scripts/<verb>_rcc_adhesion_acceleration.py` | `scripts/bench_rcc_adhesion_acceleration.py` |

Target config keys are not live API until implemented and tested.

| Key | Meaning | Default Until Gates Pass |
| --- | --- | --- |
| `rcc_bonded_pt_enabled` | Master switch | Implemented, default `false` |
| `rcc_bonded_pt_beta_lock_threshold` | Minimum beta to consider locking | Implemented, default `1.0` |
| `rcc_bonded_pt_min_lock_age` | Consecutive accepted steps before lock | Planned |
| `rcc_bonded_pt_min_separate_distance` | Rest-shape thickness floor | Implemented, default `1e-6` |
| `rcc_bonded_pt_det_dm_min` | Minimum absolute rest determinant | Implemented, default `1e-12` |
| `rcc_bonded_pt_mu` | Global virtual-tet shear modulus for bonded reporter | Implemented, default `0.0` |
| `rcc_bonded_pt_lambda` | Global virtual-tet first Lamé parameter for bonded reporter | Implemented, default `0.0` |
| `rcc_bonded_pt_release_gap` | Normal release distance | Planned |
| `rcc_bonded_pt_release_slip` | Tangential release distance | Planned |
| `rcc_bonded_pt_release_strain` | Deformation release threshold | Planned |

## Validation Rules

- Source scans enforce documentation structure and dependency boundaries only.
- Unit and contract tests must use deterministic synthetic fixtures.
- Numeric energy, gradient, and Hessian claims require a CPU or legacy oracle.
- Scene gates must report pair counts and ownership fields, not just "simulation ran".
- Current legacy RCC scene gates are behavior baselines only: they prove adhesion lift/hold/release still works in the existing pipeline, but they do not prove bonded-PT pair ownership until `rcc_bonded_pt_*` counters exist.
- The first lifecycle scene is `pt_lift_release`: a PT-rich fixture under gravity must lock during press/hold, adhered geometry must follow during sub-threshold lift, a stronger pull must release and separate it, and an adhesion-off baseline must not lift the adhered geometry. A subdivided contact-face cube, patch-on-cube, or cloth patch is acceptable; the original 8-corner cube is too sparse for this gate.
- Benchmark gates must include correctness checks after timing.
- A failed gate with a clear root cause must be recorded in the journal.

## Test Matrix

| Invariant | Layer | Target Command | Current Status |
| --- | --- | --- | --- |
| Playbook docs exist and point at current source anchors | Source/doc gate | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Implemented |
| Portable docs/source gates have one entry point | Default gate | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_all_gates.py` | Implemented |
| Local CUDA gates have one entry point and include the bunny BVH regression case | Local CUDA gate | `python3 scripts/run_rcc_adhesion_acceleration_cuda_gates.py --no-build` | Implemented |
| Existing RCC subdivided cube and cube-cloth adhesion lift/hold/release behavior is stable | Legacy scene gate | `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Implemented |
| Native sim-case RCC lift/hold/release behavior is stable | Legacy scene gate | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | Implemented |
| Locked key, oriented topology, beta, age, release flags, and rest-shape payload stay zipped through sort | Unit fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Implemented |
| Candidate, lock, release, reject, filter-skip, and duplicate counters are observable in the state contract | Unit fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state][counters]" -r compact` | Implemented |
| Rest-shape construction matches SVTS behavior | CPU oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][rest_shape]" -r compact` | Implemented |
| Bonded virtual tet E/G/H match CPU reference | CPU oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][energy]" -r compact` | Implemented |
| Host bonded PT state and rest-shape payload roundtrip through CUDA device buffers | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]" -r compact` | Implemented |
| Locked-key membership lookup matches RCC PT persistence semantics | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lookup]" -r compact` | Implemented |
| Locked PT is absent from common active/friction PT views when sorted keys are supplied | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter]" -r compact` | Implemented |
| CUDA owner feeds locked keys and syncs filter-skip counters | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` | Implemented |
| RCC Phase A high-beta PTs populate the CUDA owner with live rest-shape construction | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner][producer]" -r compact` | Implemented |
| Bonded virtual-tet reporter E/G/H matches CPU reference | Backend CUDA oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][reporter][oracle]" -r compact` | Implemented |
| Bonded PT device payloads do not corrupt unrelated BVH/radix-sort CUDA paths | Backend CUDA regression | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Implemented |
| Locked PT is absent before PT CCD broadphase in every concrete simplex filter | Contract test | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]"` | Planned |
| Released pair carries beta back to RCC | Integration test | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][release]"` | Planned |
| PT lift/release scene locks, reuses, releases, separates, and reports no duplicates | Scene gate | `build/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]"` | Planned |
| Stable scene improves hot-path timing without hiding setup cost | Benchmark gate | `uv run --no-sync python scripts/bench_rcc_adhesion_acceleration.py --scene stable_cloth_peel --frames 40 --warmup 5 --runs 10` | Planned |

## Benchmark Protocol

No speedup is accepted unless the benchmark record includes:

| Field | Required Value |
| --- | --- |
| Commit | Git SHA |
| Build | Build directory, binary path, CMake cache summary |
| GPU/driver | Device name, driver version, CUDA runtime |
| Scene/seed | Scene name, frame range, seed, mesh scale |
| Baseline mode | RCC without bonded PT acceleration |
| Test mode | RCC with bonded PT acceleration |
| Warmup/runs | Warmup count and measured run count |
| Cold setup | First classification and buffer build time |
| Cache-hot | Stable locked set, no symbolic churn |
| Churn | Controlled lock/release turnover |
| End-to-end | Fixed frame range and solver settings |
| Timers | DCD, FilterTOI, contact/RCC assembly, bonded-tet assembly, solver, frame |
| Correctness | Pair accounting and scene invariant after timing |

Minimum reported timer names:

| Timer | Meaning |
| --- | --- |
| `dcd_ms` | Discrete collision detection |
| `filter_toi_ms` | CCD trajectory filtering |
| `contact_assembly_ms` | IPC normal/friction/RCC assembly |
| `rcc_bonded_pt_assembly_ms` | Bonded virtual-tet assembly |
| `solver_ms` | Solver time |
| `frame_ms` | End-to-end frame time |

## Review Checklist

- The roadmap current phase and next task are obvious.
- Current gates are runnable today and pass in uv.
- Planned gates are concrete but not presented as proof.
- Source scans are not used as numeric proof.
- Every new counter or config key uses the `rcc_bonded_pt` prefix.
- Every simplex filter backend shares locked-key lookup semantics.
- Every release reason is observable in a report or test.
- Benchmark tables separate cold setup, cache-hot steady state, churn, and end-to-end timing.
