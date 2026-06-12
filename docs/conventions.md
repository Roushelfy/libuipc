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
9. Do not use the SVTS Stable Neo-Hookean prototype energy as the production replacement for a locked PT pair that skips CCD.
10. Do not allow a locked PT pair to skip CCD/contact/RCC unless one high-kappa ABD-style virtual-tet energy is assembled for the same four vertex DOFs in that step, or an explicit debug mode records that non-penetration is not being claimed.
11. Do not treat zero or missing bonded stiffness as a harmless default in production mode. If bonded PT acceleration skips CCD/contact/RCC, `rcc_bonded_pt_energy_model` and `rcc_bonded_pt_kappa` must be reported and valid.
12. Do not assemble a released pair in the bonded reporter for the same step in which its release flag is produced.
13. Give the end-of-step bonded producer a steady-state early-out: when no pair is released and no new pair locks (no membership change), do not re-sort, re-merge, or rebuild the bridge — keep the existing locked buffers. A stable locked set must not pay O(locked) sorts/scans every step.
14. Disable release reasons with a negative threshold sentinel so the `>= 0.0` guards short-circuit, not with a large positive value such as `1e30` (which still runs the per-lock 3x3 inverse and closest-point work every step). Keep the degeneracy and finiteness guards unconditional even when release thresholds are disabled.
15. RCC adhesion distance must match the closest-feature classification (full-feature adhesion). PT uses the plane projection (which equals the true distance for a face-interior pair), PE uses `point_edge_distance2`, PP uses `point_point_distance2`. Do not assemble an edge/vertex contact with the plane formula, and do not silently drop it by leaving the sub-formula at `return 0`.
16. Beta is stored and evolved **per VT primitive**, not per closest-feature-classified pair. Carry beta across PP/PE/PT transitions; do not reset it when the closest feature changes. `m_beta_PE`/`m_beta_PP` must evolve, not stay zero-filled, once per-primitive beta is on. The bonded lock decides on the VT primitive; the ABD virtual-tet energy stays point-plane (`F = Ds Dm_inv`) regardless of which adhesion sub-formula classified the pair.

## Data Layout Rules

Runtime state uses structure-of-arrays device buffers.

| Buffer | Type | Rule |
| --- | --- | --- |
| `locked_keys` | `DeviceBuffer<U64>` | Sorted membership keys used for filter lookup |
| `locked_topos` | `DeviceBuffer<Vector4i>` | Oriented topologies zipped/permuted with `locked_keys` |
| `locked_beta` | `DeviceBuffer<Float>` | Beta carried from RCC persistence and back on release (beta mode; sentinel `1.0` in distance-lock mode) |
| `locked_age` | `DeviceBuffer<IndexT>` | Count of accepted stable steps, not wall-clock frames |
| `Dm_inv` | `DeviceBuffer<Matrix3x3>` | Built only after rest-shape conditioning passes |
| `rest_volume` | `DeviceBuffer<Float>` | Positive and above minimum volume |
| `release_flags` | `DeviceBuffer<U32>` | Bit mask or enum, stable enough for tests and reports |
| released snapshots | Backend owner SoA buffers; scene accessor planned | Key, topology, beta, age, and flags zipped after release extraction |

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
| `rcc_bonded_pt_energy_model` | Production virtual-tet energy model, currently `abd_ortho` | Implemented, default `abd_ortho`; required when locked pairs skip CCD/contact/RCC |
| `rcc_bonded_pt_kappa` | ABD-style virtual-tet stiffness | Implemented, default `1e8`; scene correctness gates require `>= 1e8` unless they explicitly test failure/diagnostic behavior |
| `rcc_bonded_pt_release_gap` | Normal release distance | Implemented, default `1e30` to leave release disabled until a scene/test selects a threshold |
| `rcc_bonded_pt_release_slip` | Tangential release distance | Implemented, default `1e30` to leave release disabled until a scene/test selects a threshold |
| `rcc_bonded_pt_release_strain` | ABD deformation release threshold | Implemented, default `1e30` to leave release disabled until a scene/test selects a threshold |
| `rcc_bonded_pt_release_force` | Bond restoring-force release threshold (`~ kappa * deformation`), tension-gated: only fires once the true point-triangle distance exceeds the rest gap (a compressed/sheared bond never force-releases) | Implemented, default `1e30` (disabled); the trigger that peels a stiff bond on a compliant counterpart, where strain/gap cannot |
| `rcc_bonded_pt_skip_ccd` | Reject locked PTs before PT CCD broadphase in the simplex filters | Implemented; default `-1` = auto (skip whenever bonded is enabled — a locked pair is owned by the ABD virtual tet, so the CCD thickness check is redundant and aborts on over-compression). `0`/`1` explicitly override. Trade-off: skipping removes the last non-penetration guard for locked pairs (the ABD energy is reflection-invariant and cannot prevent tunneling), so set `0` if a scene needs CCD kept on bonded pairs |
| `rcc_bonded_pt_lock_face_interior_only` | Gate bonding by the point's foot vs the triangle | Implemented; default `0` = ALL VTs may bond (rest-shape builder still drops degenerate tets, but expect some skewed slivers). `1` = bond only when the point's perpendicular foot lies inside the triangle, or on / within `rcc_bonded_pt_lock_face_margin` of its boundary (face / edge / corner all qualify); exclude only feet far outside the triangle ("face-exterior" slivers). Closest-feature dim is always 2/3/4 (vertex/edge/face) — the foot is always *on* the closed triangle — so the inside-vs-outside distinction uses the barycentric foot, not the dim |
| `rcc_bonded_pt_lock_face_margin` | Barycentric margin for the face gate | Implemented, default `0.5`. How far outside the triangle the perpendicular foot may be (in barycentric units) and still bond when `rcc_bonded_pt_lock_face_interior_only=1`. `0` = strictly inside / on-boundary; larger admits feet farther past an edge before they count as "face-exterior" |
| `rcc_adhesion_normal_offset_coeff` | SOFT-adhesion normal-energy minimum offset coefficient `c∈[0,1]` (not bonded-specific — uses the `rcc_adhesion_` prefix because it applies to every soft adhesive PT pair, locked or not) | Implemented, **default `0.5`** (band center `d* = ξ + d̂/2`). Moves the normal-energy minimum to `d* = ξ + c·d̂` (ξ = per-pair thickness, sum of the two half-thicknesses), so the soft adhesion is a gentle spring to a natural gap instead of pulling surfaces into the C-IPC barrier wall at `d=ξ`. The default 0.5 cut mean Newton iters ~3.5× on the soft tape drop; `c=0` restores the legacy minimum at gap `d=0` (bitwise identical), `c=1` rests at the band outer edge (best conditioning, weakest adhesion). Clamped to `[0,1]` on read. Only the assembled solver energy is offset; the β bonding/release law keeps its raw-distance proxy (see `docs/specification/contact_models/rcc_adhesion.md`) |
| `rcc_bonded_pt_distance_lock` | Distance-locked bonding mode: disable soft adhesion energy and beta entirely; lock by end-of-step distance instead of beta (release gates unchanged) | Implemented, default `0` (beta mode). See [architecture](./architecture.md) "Distance-Locked Bonding Without Adhesion Energy" |
| `rcc_bonded_pt_distance_lock_ratio` | Lock-band coefficient `c∈[0,1]` for distance-locked bonding: lock when `d < ξ + c·d̂` (true closest-feature distance, end-of-step; `ξ = 0` reduces to `d < c·d̂`) | Implemented, default `0.5`, clamped to `[0,1]` on read; `c<=0` is structurally false in the predicate (never locks). The global `rcc_bonded_pt_beta_lock_threshold` is additionally clamped to `<= 1` in this mode (both the producer scalar and the per-pair sentinel resolution), so a beta-mode threshold cannot veto indicator locks; an explicit per-pair `bonded_lock_threshold > 1` remains a deliberate veto |

Retired prototype keys must not be reintroduced as production acceptance criteria:

| Key | Current Role | Required Follow-Up |
| --- | --- | --- |
| `rcc_bonded_pt_mu` | Retired Stable Neo-Hookean prototype reporter parameter | Keep out of the production reporter |
| `rcc_bonded_pt_lambda` | Retired Stable Neo-Hookean prototype reporter parameter | Keep out of the production reporter |

ABD/SVTS boundary:

| Allowed | Forbidden In Production |
| --- | --- |
| Use SVTS logic to condition point-triangle rest shape, `min_separate_distance`, `Dm_inv`, and positive `rest_volume` | Use SVTS Stable Neo-Hookean energy as the replacement for skipped CCD/contact/RCC |
| Use ABD OrthoPotential over `F = Ds Dm_inv` and scatter through `dF/dx` to four vertex-position DOFs | Create transient frontend ABD or SVTS geometries for per-frame locks |
| Add alternate ABD models after CPU and GPU E/G/H oracle coverage | Accept alternate energy models without a named report field and oracle |

## Validation Rules

- Source scans enforce documentation structure and dependency boundaries only.
- Unit and contract tests must use deterministic synthetic fixtures.
- Numeric energy, gradient, and Hessian claims require a CPU or legacy oracle.
- Each new adhesion sub-formula (PE/PP/EE energy, gradient, Hessian) requires its own finite-difference oracle before it becomes load-bearing. A `return 0` -> real-formula change is behaviour-neutral only while the corresponding beta buffer is zero-filled; once beta is per-primitive the formula carries force and must be oracle-backed first.
- Energy oracles must name the model they validate. A Stable Neo-Hookean oracle is only proof for the debug/prototype path; production requires an ABD-style high-kappa oracle.
- E/G/H oracle success does not by itself prove non-penetration after CCD is skipped. A bonded-mode scene gate must observe penetration/gap or an equivalent fixture-specific geometric bound.
- Release tests must assert both sides of the lifecycle: active locks stay zipped after compaction, and released snapshots carry key/topology/beta/age/flags back toward RCC persistence.
- Scene gates must report pair counts and ownership fields, not just "simulation ran".
- Current legacy RCC scene gates are behavior baselines only: they prove adhesion lift/hold/release still works in the existing pipeline, but they do not enable bonded PT, read bonded counters, assert lock/release diagnostics, or prove pair ownership.
- Release context is not a lock gate. A release fixture using sticky-side or policy inputs does not prove that the live lock producer rejects bad sticky-side, gap, slip, age, or policy candidates.
- The first lifecycle scene is `pt_lift_release`: a PT-rich fixture under gravity must lock during press/hold, adhered geometry must follow during sub-threshold lift with ABD-style bonded energy active, a stronger pull must release and separate it, and an adhesion-off baseline must not lift the adhered geometry. A subdivided contact-face cube, patch-on-cube, or cloth patch is acceptable; the original 8-corner cube is too sparse for this gate.
- Any bonded-mode scene that skips CCD must include a non-penetration observation, such as maximum signed gap/penetration, closest-point separation, or a fixture-specific geometric bound during hold and lift.
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
| Distance-lock mode bonds by `d < ξ + c·d_hat` with zero adhesion energy and unchanged release | Scene gate | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][distance_lock]" -r compact` | Implemented (604/1 green; band oracle `[rcc_bonded_pt][oracle][distance_lock]` 18/2 green) |
| Locked key, oriented topology, beta, age, release flags, and rest-shape payload stay zipped through sort | Unit fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state]" -r compact` | Implemented |
| Candidate, lock, release, reject, filter-skip, and duplicate counters are observable in the state contract | Unit fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][state][counters]" -r compact` | Implemented |
| Frontend can read live bonded PT counters and state snapshots | Feature fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][accessor]" -r compact` | Implemented |
| Rest-shape construction matches SVTS behavior | CPU oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][rest_shape]" -r compact` | Implemented |
| ABD-style high-kappa virtual tet E/G/H match CPU reference | CPU oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_core "[rcc_bonded_pt][oracle][abd_energy]" -r compact` | Implemented |
| Host bonded PT state and rest-shape payload roundtrip through CUDA device buffers | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]" -r compact` | Implemented |
| Locked-key membership lookup matches RCC PT persistence semantics | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lookup]" -r compact` | Implemented |
| Locked PT is absent from common active/friction PT views when sorted keys are supplied | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter]" -r compact` | Implemented |
| CUDA owner feeds locked keys and syncs filter-skip counters | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner]" -r compact` | Implemented |
| RCC Phase A high-beta PTs populate the CUDA owner with live rest-shape construction | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][owner][producer]" -r compact` | Implemented |
| Live lock producer applies minimum-age, sticky-side, normal-gap, tangential-slip, rest-shape, and policy gates with distinct rejection counters | Backend CUDA fixture | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][lock_gate]"` | Planned |
| Strain, gap, slip, sticky-side, and policy release compact active locks, preserve released key/topology/beta/age/flag alignment, suppress same-step relock, and carry beta back to RCC persistence | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][release]" -r compact` | Implemented |
| Scene-accessible diagnostics expose released topology, beta, age, flags, and per-reason counts without double-counting | Accessor/report fixture | `build/bin/uipc_test_core "[rcc_bonded_pt][accessor][release]"` | Planned |
| ABD-style bonded reporter E/G/H matches CPU reference at `kappa >= 1e8` | Backend CUDA oracle | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][reporter][abd_oracle]" -r compact` | Implemented |
| Production path excludes retired SNH config and reporter functions | Source/doc gate | `uv run --no-sync python scripts/run_rcc_adhesion_acceleration_gates.py` | Implemented |
| Bonded PT device payloads do not corrupt unrelated BVH/radix-sort CUDA paths | Backend CUDA regression | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "gpu_sanity_check" -c "bunny" -r compact` | Implemented |
| Locked PT is rejected before PT CCD broadphase in every concrete simplex filter (membership decision) | Backend CUDA fixture | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_backend_cuda "[rcc_bonded_pt][filter][ccd]" -r compact` | Implemented (unit-level membership shared by all four backend predicates); full scene candidate/TOI-absent integration coupled to the no-penetration gate |
| PT lift/release scene locks, reuses, avoids penetration under ABD-style energy, releases, separates, and reports no duplicates | Scene gate | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_bonded_pt][scene][pt_lift_release]" -r compact` | Press/hold/lift + no-penetration implemented (8 locks, contact-face gap min ~+0.019, lower cube carried, zero duplicates); forced-pull release/separation and adhesion-off baseline still planned |
| Stable scene improves hot-path timing without hiding setup cost | Benchmark gate | `uv run --no-sync python scripts/bench_rcc_adhesion_acceleration.py --scene stable_cloth_peel --frames 40 --warmup 5 --runs 10` | Planned |
| Locked-pair geometric release reproduces the unaccelerated beta-evolution debond timing within tolerance on a canonical purely-normal and purely-tangential example | Calibration gate | `build/bin/uipc_test_sim_case "[rcc_bonded_pt][calibration][debond]"` | Planned |
| Bonded producer skips sort/merge/bridge-rebuild on a no-membership-change step | Backend CUDA fixture | `build/bin/uipc_test_backend_cuda "[rcc_bonded_pt][producer][steady_state]"` | Planned |

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
| Timers | DCD, FilterTOI, contact/RCC assembly, bonded-tet assembly, producer, solver, frame |
| Solver iterations | Newton iteration count and total/average PCG iterations per step; the primary expected win is fewer iterations from replacing stiff near-contact log-barrier Hessian blocks with the smooth high-kappa ABD block |
| Kappa sweep | Report metrics across `rcc_bonded_pt_kappa` values; kappa trades bond rigidity (correctness) against conditioning (iteration count), and too-high kappa can worsen conditioning |
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
| `rcc_bonded_pt_producer_ms` | End-of-step lock/release classification, compaction, sort, and bridge rebuild |

## Review Checklist

- The roadmap current phase and next task are obvious.
- Current gates are runnable today and pass in uv.
- Planned gates are concrete but not presented as proof.
- Source scans are not used as numeric proof.
- Every new counter or config key uses the `rcc_bonded_pt` prefix.
- Every simplex filter backend shares locked-key lookup semantics.
- Lock-gate status and release-gate status are not conflated.
- Every release reason is observable in a report or test.
- Benchmark tables separate cold setup, cache-hot steady state, churn, and end-to-end timing.
