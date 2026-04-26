# SOCU Mixed Solver Integration Journal

This journal records implementation progress for
[`socu_mixed_solver_integration_plan.md`](socu_mixed_solver_integration_plan.md).
It is intentionally separate from the plan: the plan describes intended design,
while this file records what has actually landed, what was verified, and what
remains open.

## 2026-04-24

### Milestone 0: Standalone Ordering Lab

Status: **Complete for the standalone graph-ordering checkpoint.**

The first milestone is now implemented as an opt-in experiment under
`experiments/socu_ordering_lab`. It does not touch `cuda_mixed` runtime code,
does not call `socu_native`, and only answers the M0 question: given an atom
graph and a `32` or `64` DoF block size, which chain ordering keeps weighted
graph edges inside the current block or adjacent blocks.

The root CMake switch is:

```cmake
UIPC_BUILD_SOCU_ORDERING_LAB=ON
```

The experiment defines these targets:

```text
socu_ordering_lab_core
socu_ordering_lab_test
socu_ordering_bench
```

The current implementation supports synthetic presets, graph JSON input, and
extra weighted edge CSV input. The implemented presets are `rod`, `cloth_grid`,
`tet_block`, `shuffled_cloth_grid`, and `shuffled_tet_block`. The CLI command is:

```bash
socu_ordering_bench order \
  --preset shuffled_cloth_grid \
  --orderer auto \
  --block-size auto \
  --report report.json \
  --summary-csv summary.csv \
  --mapping-csv mapping.csv
```

The implemented orderers are:

```text
original
rcm
nvidia_symrcm
metis_nd
metis_kway_rcm
```

`nvidia_symrcm` is optional. If CUDAToolkit provides `cusolver` and `cusparse`,
the lab links `CUDA::cusolver` and `CUDA::cusparse` and calls
`cusolverSpXcsrsymrcmHost`. If those libraries are not available, the candidate
fails cleanly with a `fallback_reason` while the rest of the experiment remains
usable.

The report now records all candidate metrics instead of only the winner. The
core metrics include total atom count, total DoFs, block count, block
utilization, near-band and off-band edge counts, weighted near-band and
off-band ratios, max block distance, average block distance, edge chain span,
ordering time, and permutation validity. The mapping CSV records
`old_atom`, `chain_atom`, `block`, `block_offset`, and `dof_count`.

The CLI also supports process-local timing experiments:

```bash
socu_ordering_bench order \
  --preset shuffled_cloth_grid \
  --orderer nvidia_symrcm \
  --block-size 64 \
  --warmup 1 \
  --repeat 5 \
  --report nvidia_warm.json
```

When `--warmup` or `--repeat` is used, the JSON report includes
`timing_repeats`, with per-candidate average, min, and max ordering time.

### Verification

The lab was configured with:

```bash
cmake -S . -B build-socu-ordering-lab \
  -DCMAKE_TOOLCHAIN_FILE=/home/zhaofeng/course/15-740_project/ChampSim/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DUIPC_USING_LOCAL_VCPKG=ON \
  -DUIPC_BUILD_SOCU_ORDERING_LAB=ON \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=OFF \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_BENCHMARKS=OFF
```

The verified build and test commands were:

```bash
cmake --build build-socu-ordering-lab \
  --target socu_ordering_bench socu_ordering_lab_test -j 8

./build-socu-ordering-lab/Release/bin/socu_ordering_lab_test

ctest --test-dir build-socu-ordering-lab \
  -R socu_ordering_lab --output-on-failure
```

The latest test result was:

```text
All tests passed (2573 assertions in 12 test cases)
100% tests passed, 0 tests failed out of 1
```

The tests cover graph canonicalization, malformed permutation rejection,
`rod` zero-off-band behavior for bandwidth-friendly orderers, auto ordering
quality compared with original ordering on shuffled graphs, `32` and `64`
candidate reporting, METIS `perm/iperm` contract validation, optional NVIDIA
RCM availability, JSON report shape, summary CSV output, and mapping CSV row
counts.

### Timing Notes

On this machine, CUDAToolkit was found and `SOL_HAS_CUSOLVER_RCM=1` was used.
For `shuffled_cloth_grid --block-size 64`, process-local handle reuse changed
the observed NVIDIA RCM timing substantially:

```text
nvidia_symrcm cold repeat=5:
  avg ~= 47.41 ms
  max ~= 236.82 ms
  min ~= 0.056 ms

nvidia_symrcm warmup=1 repeat=5:
  avg ~= 0.061 ms
  min ~= 0.054 ms
  max ~= 0.076 ms

local rcm repeat=5:
  avg ~= 0.045 ms
  min ~= 0.033 ms
  max ~= 0.076 ms
```

For `shuffled_cloth_grid`, warmed-up `rcm` and `nvidia_symrcm` produced the same
weighted off-band ratio at block size `64`:

```text
rcm 64:
  weighted_off_band_ratio = 0.0
  ordering_time_ms ~= 0.047

nvidia_symrcm 64:
  weighted_off_band_ratio = 0.0
  ordering_time_ms ~= 0.086
```

For `shuffled_tet_block`, warmed-up `nvidia_symrcm` was slightly better by the
current scorer but also slightly slower:

```text
rcm 64:
  weighted_off_band_ratio ~= 0.1700
  ordering_time_ms ~= 0.071

nvidia_symrcm 64:
  weighted_off_band_ratio ~= 0.1583
  ordering_time_ms ~= 0.096
```

### Completion Criteria

M0 is considered complete because the lab now has:

- an opt-in standalone CMake target under `experiments`;
- pure graph ordering data structures independent of runtime solver code;
- deterministic synthetic graph presets plus JSON and extra-edge CSV inputs;
- multiple orderer candidates, including RCM, METIS, and optional NVIDIA RCM;
- shared scoring and report generation for all candidates;
- JSON report, CSV summary, and mapping CSV output;
- repeat/warmup timing support for fair cuSolver host-path measurement;
- Catch2 and CTest coverage for core M0 contracts.

This completion does **not** include Milestone 1 physical geometry reorder,
Milestone 2 runtime contact classification, or any `cuda_mixed` solver
abstraction work. Direct surface or tet mesh file import is also not part of
the current checkpoint; graph JSON is the supported generic input path for M0.

### Next Work

At the M0 close, the next planned step was Milestone 1: implement an init-time
reorder lab that separates solver-owned mirror reorder from physical geometry
reorder. That work needed to test real
`AttributeCollection::reorder(chain_to_old)` semantics and explicitly rewrite
edge, triangle, and tetrahedron topology indices.

### Milestone 1: Init-Time Reorder Lab

Status: **Complete for the standalone mirror-vs-physical reorder checkpoint.**

The ordering lab now includes an init-time geometry reorder path under the same
`experiments/socu_ordering_lab` target. This remains a lab-only implementation:
it does not touch `cuda_mixed` runtime code, does not register a runtime solver
contract, and does not change assembly flow.

The new public experiment surface is:

```text
include/sol/reorder.h
src/reorder.cpp
tests/reorder_tests.cpp
```

The new CLI command is:

```bash
socu_ordering_bench reorder \
  --preset shuffled_cloth_grid \
  --ordering order.json \
  --mode physical \
  --report physical.json \
  --summary-csv physical.csv \
  --mapping-csv physical_mapping.csv
```

`--ordering` accepts the JSON report produced by `socu_ordering_bench order`.
The `reorder` command supports two modes:

```text
mirror
physical
```

`mirror` keeps the geometry untouched and only evaluates the selected chain
mapping as solver-owned metadata. `physical` copies the geometry, creates
verification metadata on vertices, calls `AttributeCollection::reorder()` with
the selected `chain_to_old` mapping, and then rewrites edge, triangle, and
tetrahedron topologies through `old_to_chain`.

The physical path validates the following contracts:

- `chain_to_old` is passed as the `AttributeCollection::reorder()` New2Old map;
- vertex metadata proves that new vertex `i` came from old vertex
  `chain_to_old[i]`;
- topology indices are rewritten from old vertex IDs to chain vertex IDs;
- vertex, edge, triangle, and tetrahedron counts are preserved;
- edge length, triangle area, and tetrahedron volume invariants are preserved;
- mirror and physical block classifications agree;
- physical reorder turns chain traversal into contiguous vertex memory.

The current reorder report records counts, topology validity, metadata
validity, ordering summary, before/after chain-memory stride, mirror and
physical block classifications, and geometric invariant errors.

### Milestone 1 Verification

The lab was rebuilt with:

```bash
cmake --build build-socu-ordering-lab \
  --target socu_ordering_bench socu_ordering_lab_test -j 8
```

The latest test result is:

```text
All tests passed (3028 assertions in 17 test cases)
100% tests passed, 0 tests failed out of 1
```

The direct test commands were:

```bash
./build-socu-ordering-lab/Release/bin/socu_ordering_lab_test

ctest --test-dir build-socu-ordering-lab \
  -R socu_ordering_lab --output-on-failure
```

A CLI smoke test was run with:

```bash
socu_ordering_bench order \
  --preset shuffled_cloth_grid \
  --orderer rcm \
  --block-size 64 \
  --report /tmp/socu_m1_verify/order.json \
  --summary-csv /tmp/socu_m1_verify/order.csv \
  --mapping-csv /tmp/socu_m1_verify/mapping.csv

socu_ordering_bench reorder \
  --preset shuffled_cloth_grid \
  --ordering /tmp/socu_m1_verify/order.json \
  --mode mirror \
  --report /tmp/socu_m1_verify/mirror.json \
  --summary-csv /tmp/socu_m1_verify/mirror.csv

socu_ordering_bench reorder \
  --preset shuffled_cloth_grid \
  --ordering /tmp/socu_m1_verify/order.json \
  --mode physical \
  --report /tmp/socu_m1_verify/physical.json \
  --summary-csv /tmp/socu_m1_verify/physical.csv \
  --mapping-csv /tmp/socu_m1_verify/physical_mapping.csv
```

For the physical smoke test, the key result was:

```text
counts_preserved = true
topology_indices_valid = true
original_vertex_id_complete = true
chain_metadata_valid = true
chain_index_stride_before = 12025
chain_index_stride_after = 191
mirror_near_band_edges = 521
mirror_off_band_edges = 0
physical_near_band_edges = 521
physical_off_band_edges = 0
max_edge_length_error = 0
max_triangle_area_error = 0
max_tet_volume_error = 0
```

### Milestone 1 Completion Criteria

M1 is considered complete because the lab now has:

- real `SimplicialComplex` presets matching the graph presets;
- a solver-owned mirror reorder mode;
- a physical geometry reorder mode using `AttributeCollection::reorder()`;
- topology remapping for edge, triangle, and tetrahedron vertex indices;
- metadata checks proving New2Old semantics for `chain_to_old`;
- geometric invariant checks for rods, cloth grids, and tet blocks;
- mirror-vs-physical block classification consistency checks;
- CLI JSON and CSV reports for reorder experiments;
- Catch2 and CTest coverage for the M1 contracts.

This completion does **not** include Milestone 2 contact primitive
classification, mesh file import, or any `cuda_mixed` runtime integration.

### Next Work

The next planned step is Milestone 2: add contact primitive and contribution
classification inside the same experiment so reordered topology can be tested
against near-band, mixed, and adversarial contact cases before any runtime
solver integration.

### Commit Handoff

This checkpoint is ready to commit as the standalone M0/M1 experiment. The
intended commit scope is limited to:

```text
CMakeLists.txt
docs/development/backend_cuda/index.md
docs/development/backend_cuda/socu_mixed_solver_integration_journal.md
experiments/socu_ordering_lab/**
```

Known unrelated working-tree entries are intentionally excluded from this
checkpoint:

```text
external/muda
.codex
```

The next implementation step should start from Milestone 2 in
`experiments/socu_ordering_lab`: introduce contact primitives and contribution
classification, keep the work lab-only, and preserve the same JSON/CSV plus
Catch2 verification style used by M0 and M1.

### Milestone 2: Contact Simulation Lab

Status: **Complete for the standalone contact-classification checkpoint.**

The ordering lab now includes simulated runtime-contact classification under
the same `experiments/socu_ordering_lab` target. This remains lab-only: it does
not touch the real collision pipeline, does not change `cuda_mixed`, and does
not assemble a structured solver buffer yet.

The new public experiment surface is:

```text
include/sol/contact.h
src/contact.cpp
tests/contact_tests.cpp
```

The new CLI command is:

```bash
socu_ordering_bench contact \
  --ordering order.json \
  --scenario adversarial \
  --count 24 \
  --report contact.json \
  --summary-csv contact.csv
```

`--ordering` accepts the JSON report produced by `socu_ordering_bench order`.
The contact command supports these scenarios:

```text
near_band
mixed
adversarial
from_file
```

`from_file` reads a contact CSV with this shape:

```text
kind,stiffness,atom0,atom1,atom2,atom3
PP,1.0,0,1
PE,2.0,0,80,81
PT,3.0,0,80,81,82
EE,3.0,0,1,80,81
```

The primitive kinds currently supported are `PP`, `PE`, `PT`, and `EE`.
Each primitive is expanded into unordered pairwise Hessian block
contributions across its participating atoms. Classification is then reported
at two levels:

1. primitive level, where a primitive is near-band only if all expanded
   contributions are near-band;
2. expanded contribution level, where each block pair is independently
   classified by `abs(block_i - block_j) <= 1`.

The report records active contact counts, near/off/mixed primitive counts,
near/off contribution counts, primitive ratios, contribution ratios, weighted
near-band contribution norm, weighted off-band dropped norm, contact classify
time, frame-boundary reorder time, estimated absorbed contribution count,
estimated dropped contribution count, and the fixed-permutation contract for
the frame.

The frame contract is explicit in the report:

```text
newton_iteration_reorder_count = 0
permutation_fixed_within_frame = true
```

### Milestone 2 Verification

The lab was rebuilt with:

```bash
cmake --build build-socu-ordering-lab \
  --target socu_ordering_bench socu_ordering_lab_test -j 8
```

The latest test result is:

```text
All tests passed (3072 assertions in 22 test cases)
100% tests passed, 0 tests failed out of 1
```

The direct test commands were:

```bash
./build-socu-ordering-lab/Release/bin/socu_ordering_lab_test

ctest --test-dir build-socu-ordering-lab \
  -R socu_ordering_lab --output-on-failure
```

A CLI smoke test was run with:

```bash
socu_ordering_bench order \
  --preset shuffled_tet_block \
  --orderer rcm \
  --block-size 32 \
  --report /tmp/socu_m2_verify/order.json \
  --summary-csv /tmp/socu_m2_verify/order.csv \
  --mapping-csv /tmp/socu_m2_verify/mapping.csv

socu_ordering_bench contact \
  --ordering /tmp/socu_m2_verify/order.json \
  --scenario adversarial \
  --count 24 \
  --report /tmp/socu_m2_verify/contact.json \
  --summary-csv /tmp/socu_m2_verify/contact.csv
```

For the adversarial smoke test, the key result was:

```text
active_contact_count = 24
near_band_contact_count = 0
mixed_contact_count = 24
off_band_contact_count = 24
near_band_contribution_count = 48
off_band_contribution_count = 96
off_band_ratio = 1.0
contribution_off_band_ratio = 0.6666666667
weighted_off_band_ratio = 0.6666666667
contact_classify_time_ms ~= 0.0084
frame_boundary_reorder_time_ms = 0
newton_iteration_reorder_count = 0
permutation_fixed_within_frame = true
```

### Milestone 2 Completion Criteria

M2 is considered complete because the lab now has:

- standalone contact primitive data structures;
- synthetic `near_band`, `mixed`, and `adversarial` contact scenarios;
- recorded contact primitive CSV input through `from_file`;
- primitive-level and expanded-contribution-level classification;
- weighted contribution accounting so stiff off-band contacts are visible;
- separate contact classify time and frame-boundary reorder time fields;
- an explicit no-Newton-reorder frame contract;
- JSON and CSV report output;
- Catch2 and CTest coverage for near-band absorption, mixed primitive
  accounting, adversarial off-band pressure, high-stiffness weighted drops,
  and CSV input.

This completion does **not** include real collision pipeline integration,
structured Hessian sinks, `socu_native`, or `cuda_mixed` runtime changes.

### Next Work

The next planned step is Milestone 3: add the `LinearSolver` abstraction inside
`cuda_mixed` as a no-regression checkpoint. That work should keep the current
PCG and fused PCG algorithms unchanged, add selected-solver validation, and
avoid introducing any `socu_native` or structured assembly path yet.

## 2026-04-25

### Milestone 3 Baseline: fp64 No-Regression Benchmark

Before changing the `cuda_mixed` linear solver abstraction, the `fp64` build was
rebuilt and a single asset benchmark was recorded as the no-regression baseline.

The rebuild command was:

```bash
cmake --build build/build_impl_fp64 \
  --target backend_cuda_mixed pyuipc uipc_test_backend_cuda_mixed uipc_test_sim_case \
  --parallel 8
```

The benchmark command was:

```bash
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
  apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/particle.json \
  --levels fp64 \
  --build fp64=build/build_impl_fp64 \
  --config Release \
  --run_root output/benchmarks/mixed/uipc_assets/socu_m3_baseline_fp64 \
  --perf \
  --timers
```

The recorded baseline is:

```text
asset = particle_rain
level = fp64
frames = 80
warmup_frames = 10
wall_time = 1.590889507 s
end_to_end_wall_time = 1.764613934 s
avg_frame_time = 19.886118838 ms/frame

Pipeline mean = 21.070949050 ms
Simulation mean = 21.067777313 ms
Build Linear System mean = 1.576213125 ms
Assemble Linear System mean = 0.291090225 ms
Convert Matrix mean = 1.148941438 ms
Solve Global Linear System mean = 5.928472675 ms
Solve Linear System mean = 4.203382688 ms
FusedPCG mean = 4.073189888 ms

newton_iteration_count mean = 2.6875
line_search_iteration_count mean = 3.975
pcg_iteration_count mean = 14.9375
```

This benchmark uses the default `linear_system/solver = fused_pcg` path. After
M3 lands, the same command should be rerun into a separate run root and compared
against these numbers. Some run-to-run noise is expected, but the solver
selection refactor should not introduce systematic extra work in `Build Linear
System`, `Solve Linear System`, or `FusedPCG`.

### Milestone 3 Implementation

M3 added the `LinearSolver` abstraction as a no-regression checkpoint for
`cuda_mixed`.

The implementation keeps the existing sparse assembly and PCG algorithms intact:

- `LinearSolver` now owns the common solver interface, selected solver system
  pointer, assembly requirements, optional iteration counter name, and final
  `SimSystem::do_build()` bridge.
- `IterativeSolver` now derives from `LinearSolver` and preserves the existing
  `spmv()`, `apply_preconditioner()`, `accuracy_statisfied()`, and `ctx()`
  behavior.
- `GlobalLinearSystem` collects `SimSystemSlotCollection<LinearSolver>`,
  validates that exactly one solver is selected at init, and stores
  `selected_linear_solver`.
- `LinearFusedPCG` remains the default `linear_system/solver = fused_pcg` path.
  `LinearPCG` now explicitly selects only `linear_system/solver = linear_pcg`.
- `global_linear_system.h` only forward-declares `LinearSolver`; the concrete
  solver header is included from `.cu` files to avoid a `GlobalLinearSystem` to
  `LinearSolver` header cycle.

The contract tests were extended with:

- static inheritance checks for `SimSystem -> LinearSolver -> IterativeSolver`;
- default `fused_pcg` solver selection smoke;
- explicit `linear_pcg` solver selection smoke;
- invalid solver selection smoke.

M3 deliberately does not add `socu_native`, `socu_approx`, structured assembly
sinks, or any runtime use of the SOCU ordering lab output.

### Milestone 3 Verification

The `fp64` build and contract tests passed with:

```bash
cmake --build build/build_impl_fp64 \
  --target backend_cuda_mixed uipc_test_backend_cuda_mixed uipc_test_sim_case \
  --parallel 8

ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|sim_case_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

The final test result was:

```text
backend_cuda_mixed_contract: passed
backend_cuda_mixed_source_contract_scan: passed
sim_case_cuda_mixed_contract: passed
```

For Python benchmark runs, `scripts/after_build_pyuipc.py` was rerun after the
native rebuild. The copied package library and build output library matched:

```text
build/build_impl_fp64/python/src/uipc/_native/libuipc_backend_cuda_mixed.so
build/build_impl_fp64/Release/bin/libuipc_backend_cuda_mixed.so
sha256 = 0dabe9ab02af4d7319e7315be532b2020cbccae7e7cdf93d61ba164f44f110d7
```

The final M3 `fp64` benchmark run was:

```bash
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
  apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/particle.json \
  --levels fp64 \
  --build fp64=build/build_impl_fp64 \
  --config Release \
  --run_root output/benchmarks/mixed/uipc_assets/socu_m3_after_fp64_final \
  --perf \
  --timers
```

The recorded M3 result was:

```text
asset = particle_rain
level = fp64
frames = 80
warmup_frames = 10
wall_time = 1.668819212 s
end_to_end_wall_time = 1.852555498 s
avg_frame_time = 20.860240150 ms/frame

Pipeline mean = 22.214535025 ms
Simulation mean = 22.210836288 ms
Build Linear System mean = 1.757508688 ms
Assemble Linear System mean = 0.303369150 ms
Convert Matrix mean = 1.323632275 ms
Solve Global Linear System mean = 6.187157838 ms
Solve Linear System mean = 4.154478725 ms
FusedPCG mean = 4.070428175 ms

newton_iteration_count mean = 2.6875
line_search_iteration_count mean = 3.975
pcg_iteration_count mean = 14.9375
```

The first baseline and first M3 comparison showed noticeable wall-time movement,
so a paired repeat check was run to separate code cost from machine/GPU run
state. The same old checkpoint was rebuilt in `/tmp/libuipc_m3_base`, then the
old and M3 builds were run in alternating order.

The repeat averages were:

```text
old repeat avg_frame_time = 20.850079688 ms/frame
M3 repeat avg_frame_time = 20.648711587 ms/frame
delta = -0.965790554 %

old repeat end_to_end_wall_time = 1.829858880 s
M3 repeat end_to_end_wall_time = 1.817689780 s
delta = -0.665029452 %
```

The solver iteration counters were unchanged in all runs:

```text
newton_iteration_count mean = 2.6875
line_search_iteration_count mean = 3.975
pcg_iteration_count mean = 14.9375
```

M3 is therefore considered complete as a no-regression abstraction checkpoint:
the solver selection refactor changes ownership and validation, but does not
add systematic extra PCG work or change the existing default fused PCG behavior.

### Next Work

The next planned step is Milestone 4: add a `SocuApproxSolver` skeleton and
stub gate for `linear_system/solver = "socu_approx"`. The first M4 commit should
still avoid calling `socu_native`; it should only validate configuration,
report explicit gate-fail reasons, prove that default `fused_pcg` is unaffected,
and make explicit `socu_approx` selection fatal when the stub cannot run.

### Milestone 4 Implementation

M4 added a `SocuApproxSolver` skeleton as a gated experimental solver. It does
not call `socu_native`, does not assemble structured chain buffers, and does not
produce a solve direction yet.

The implementation added:

- `src/backends/cuda_mixed/linear_system/socu_approx_report.h`
- `src/backends/cuda_mixed/linear_system/socu_approx_solver.h`
- `src/backends/cuda_mixed/linear_system/socu_approx_solver.cu`

The only new solver selection value is:

```text
linear_system/solver = "socu_approx"
```

The default solver remains `fused_pcg`. When `socu_approx` is not selected, the
new solver exits through the normal unused-system `SimSystemException` path, so
default runtime behavior is unchanged. When `socu_approx` is explicitly
selected, gate failures throw a non-`SimSystemException` `uipc::Exception`, so
the reason is fatal and cannot be swallowed by the system registry as an unused
solver.

M4 added default config fields:

```text
linear_system/socu_approx/ordering_report = ""
linear_system/socu_approx/min_near_band_ratio = 0.0
linear_system/socu_approx/max_off_band_ratio = 1.0
```

The skeleton reads the ordering report JSON produced by the standalone ordering
lab and validates the basic mapping schema:

```text
block_size
chain_to_old
old_to_chain
atom_to_block
atom_block_offset
atom_dof_count
block_to_atom_range
```

The currently implemented fatal gate reasons are:

```text
ordering_missing
ordering_report_invalid
unsupported_precision_contract
unsupported_block_size
ordering_quality_too_low
structured_provider_missing
socu_approx_stub_no_direction
```

Additional reason names are reserved for later milestones:

```text
socu_disabled
socu_mathdx_unsupported
contact_off_band_ratio_too_high
```

For a valid report in an `fp64` build, M4 intentionally stops at
`structured_provider_missing`, because Milestone 5 is the first milestone that
will add a structured assembly dry run.

### Milestone 4 Verification

The `fp64` build was reconfigured so the new `socu_approx_solver.cu` file was
included by the backend glob, then rebuilt with:

```bash
cmake -S . -B build/build_impl_fp64

cmake --build build/build_impl_fp64 \
  --target backend_cuda_mixed uipc_test_backend_cuda_mixed uipc_test_sim_case \
  --parallel 8
```

The contract tests passed with:

```bash
ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|sim_case_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

The final test result was:

```text
backend_cuda_mixed_contract: passed
backend_cuda_mixed_source_contract_scan: passed
sim_case_cuda_mixed_contract: passed
```

The tests now cover:

- static inheritance: `SocuApproxSolver` derives from `LinearSolver`;
- default `fused_pcg` selection remains valid;
- explicit `linear_pcg` selection remains valid;
- invalid solver selection still fails;
- explicit `socu_approx` with no ordering report fails with `ordering_missing`;
- explicit `socu_approx` with block size other than `32` or `64` fails with
  `unsupported_block_size`;
- explicit `socu_approx` with low ordering quality fails with
  `ordering_quality_too_low`;
- explicit `socu_approx` with a valid ordering report stops at
  `structured_provider_missing`.

The source check confirmed that the M4 runtime files do not include or link
`socu_native`.

The Python package native copy was refreshed after the rebuild. The copied
package library and build output library matched:

```text
build/build_impl_fp64/python/src/uipc/_native/libuipc_backend_cuda_mixed.so
build/build_impl_fp64/Release/bin/libuipc_backend_cuda_mixed.so
sha256 = 98efb14b1bf88b7d6eb9653e2d4f243444b76cd348579f42c5af5648909bc09c
```

The default `fused_pcg` `fp64` benchmark was then rerun:

```bash
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
  apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/particle.json \
  --levels fp64 \
  --build fp64=build/build_impl_fp64 \
  --config Release \
  --run_root output/benchmarks/mixed/uipc_assets/socu_m4_after_fp64_final \
  --perf \
  --timers
```

The recorded M4 default-path result was:

```text
asset = particle_rain
level = fp64
frames = 80
warmup_frames = 10
wall_time = 1.352571868 s
end_to_end_wall_time = 1.499347954 s
avg_frame_time = 16.907148350 ms/frame

Pipeline mean = 17.873403913 ms
Simulation mean = 17.869898288 ms
Build Linear System mean = 1.456503350 ms
Assemble Linear System mean = 0.282735038 ms
Convert Matrix mean = 0.999943888 ms
Solve Global Linear System mean = 5.023033075 ms
Solve Linear System mean = 3.450615763 ms
FusedPCG mean = 3.371987937 ms

newton_iteration_count mean = 2.6875
line_search_iteration_count mean = 3.975
pcg_iteration_count mean = 14.9375
```

The M4 run was faster than the M3 recorded run, but prior paired runs showed
substantial device/runtime variation. The important no-regression facts are that
the default solver remained `fused_pcg`, the iteration counters are unchanged,
and the new `SocuApproxSolver` only takes the unused-system path unless
explicitly selected.

M4 is therefore considered complete as a skeleton/gate checkpoint.

### Next Work

The next planned step is Milestone 5: add the structured assembly dry run. That
work should introduce the ordering-aware provider and structured sink needed to
pack logical `diag`, first `offdiag`, and `rhs` buffers, classify near/off-band
contact contribution, and report pack quality and timing. Milestone 5 should
still avoid invoking `socu_native`; it should only prove that the structured
surrogate layout can be generated and compared against the standalone lab
statistics.

### Milestone 5 Implementation

M5 added the first structured dry-run path for explicit
`linear_system/solver = "socu_approx"`. This is still not a performance path:
it does not call `socu_native`, does not skip the legacy full sparse assembly in
the default solver path, and does not produce a real linear solve direction.

The implementation added the runtime abstraction boundary:

- `src/backends/cuda_mixed/linear_system/structured_chain_provider.h`

The new header defines the logical provider contract for later real structured
assembly:

```text
StructuredChainShape
StructuredDofSlot
StructuredContributionStats
StructuredQualityReport
StructuredAssemblySink
StructuredChainProvider
```

`SocuApproxSolver` now builds an ordering-backed structured provider from the
lab ordering JSON. The provider expands each ordered atom into explicit
`old_dof -> padded chain_dof -> block/lane` slots and inserts explicit padding
slots. Padding slots have no old DoF, are not scatter-written, and are included
in the dry-run report so later kernels do not need implicit lane conventions.

M5 also added two config fields:

```text
linear_system/socu_approx/contact_report = ""
linear_system/socu_approx/dry_run_report = ""
```

When `socu_approx` is explicitly selected with a valid ordering report, the
gate now passes initialization instead of failing at
`structured_provider_missing`. During solve, the solver performs a CPU-side
dry-run pack report and writes JSON with:

```text
mode = structured_dry_run
block_size
ordering_dof_count
structured_slot_count
padding_slot_count
block_utilization
layout.diag_block_count
layout.first_offdiag_block_count
layout.rhs_scalar_count
layout.diag_scalar_count
layout.first_offdiag_scalar_count
layout.blocks[]
contact near/off-band counts and contribution ratios
timing.dry_run_pack_time_ms
status.direction_available = false
status.reason = socu_approx_stub_no_direction
```

For this checkpoint, the contact contribution statistics are read from the
standalone lab contact report. Near-band contribution counts are propagated into
the structured dry-run report, while off-band contribution counts are reported
as dropped. The real contact/dytopo reporter is not yet writing directly into a
structured sink.

The solve phase deliberately returns a zero direction after writing the report.
The report still states that no real direction is available. This avoids
throwing through the CUDA/C++ solve stack while preserving an explicit,
opt-in-only dry-run behavior. The default `fused_pcg` path remains unchanged and
does not instantiate this path except through the normal unused-system registry
flow.

### Milestone 5 Verification

The `fp64` build was rebuilt with:

```bash
cmake --build build/build_impl_fp64 \
  --target backend_cuda_mixed uipc_test_backend_cuda_mixed uipc_test_sim_case \
  --parallel 8
```

The targeted contract tests passed with:

```bash
ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|sim_case_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

The final test result was:

```text
backend_cuda_mixed_contract: passed
backend_cuda_mixed_source_contract_scan: passed
sim_case_cuda_mixed_contract: passed
```

The M5 contract coverage now includes:

- default `fused_pcg` selection remains valid;
- explicit `linear_pcg` selection remains valid;
- invalid solver selection still fails;
- explicit `socu_approx` with no ordering report fails with `ordering_missing`;
- explicit `socu_approx` with block size other than `32` or `64` fails with
  `unsupported_block_size`;
- explicit `socu_approx` with low ordering quality fails with
  `ordering_quality_too_low`;
- explicit `socu_approx` with valid `n=32` and `n=64` ordering reports runs the
  structured dry-run, writes `dry_run.json`, and keeps
  `status.reason = socu_approx_stub_no_direction`;
- the dry-run report records correct one-block `diag/rhs` layout for the test
  scene, explicit padding counts, block utilization, near/off-band contact
  contribution counts, and dry-run pack timing.

The `n=32` dry-run contract produced:

```text
block_size = 32
ordering_dof_count = 12
structured_slot_count = 32
padding_slot_count = 20
block_utilization = 0.375
layout.diag_block_count = 1
layout.first_offdiag_block_count = 0
layout.rhs_scalar_count = 12
layout.diag_scalar_count = 144
near_band_contribution_count = 4
off_band_contribution_count = 2
```

The `n=64` dry-run contract similarly produced one block with
`padding_slot_count = 52` and `block_utilization = 0.1875`.

The source check confirmed that the M5 runtime files still do not reference
`socu_native`.

After the final rebuild, the Python package native copy was refreshed with
`scripts/after_build_pyuipc.py`. The copied package library and build output
library matched:

```text
build/build_impl_fp64/python/src/uipc/_native/libuipc_backend_cuda_mixed.so
build/build_impl_fp64/Release/bin/libuipc_backend_cuda_mixed.so
sha256 = 0fcad518d9555eaa4dfc14c247de051e97b9db106c0cf95fac8caac78e2478bf
```

The default `fused_pcg` `fp64` benchmark was rerun with the final M5 code:

```bash
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
  apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/particle.json \
  --levels fp64 \
  --build fp64=build/build_impl_fp64 \
  --config Release \
  --run_root output/benchmarks/mixed/uipc_assets/socu_m5_after_fp64_final \
  --perf \
  --timers
```

The recorded M5 default-path result was:

```text
asset = particle_rain
level = fp64
frames = 80
warmup_frames = 10
wall_time = 1.285899446 s
end_to_end_wall_time = 1.429956108 s
avg_frame_time = 16.073743075 ms/frame

Pipeline mean = 16.975558113 ms
Simulation mean = 16.972240750 ms
Build Linear System mean = 1.326953850 ms
Assemble Linear System mean = 0.286441425 ms
Convert Matrix mean = 0.933967775 ms
Solve Global Linear System mean = 4.843001200 ms
Solve Linear System mean = 3.416679325 ms
FusedPCG mean = 3.340769963 ms
SpMV mean = 0.706344813 ms
Apply Preconditioner mean = 0.471451300 ms

newton_iteration_count mean = 2.6875
line_search_iteration_count mean = 3.975
pcg_iteration_count mean = 14.9375
```

The iteration counters match the M3/M4 records exactly. The frame time is within
the machine/runtime variation already observed in paired M3/M4 checks. M5 is
therefore considered complete as a structured dry-run checkpoint, not as a
performance checkpoint.

### Next Work

The next planned step is synthetic `socu` solve: keep using the structured
provider boundary, but introduce a synthetic SPD block-tridiagonal system and a
controlled `socu_native` call. That step should validate `n=32` and `n=64`,
finite residual, descent direction, stream compatibility, and repeated-run
resource stability before any real `cuda_mixed` FEM/ABD/contact contribution is
sent to `socu_native`.

## M6 Dependency Checkpoint: `socu_native` Submodule Wiring

Before implementing the synthetic `socu` solve, the external solver dependency
was introduced as a submodule at:

```text
external/socu-native-cuda -> 6195c060526a63705e488c36eb87f0fe9ba2d6c0
```

The `.gitmodules` entry records the upstream origin:

```text
https://github.com/Roushelfy/socu-native-cuda
```

The local checkout was populated from `~/work/socu-native-cuda` because the
non-interactive shell could not authenticate against GitHub over HTTPS. No files
inside the submodule were modified.

The root CMake now exposes:

```text
UIPC_WITH_SOCU_NATIVE = AUTO | ON | OFF
```

The default is `AUTO`. With `UIPC_WITH_CUDA_MIXED_BACKEND=ON`, `AUTO` enables
the submodule when `external/socu-native-cuda/CMakeLists.txt` is present and
skips it when the submodule is missing. `ON` makes the dependency mandatory and
fails configure if the submodule is absent or if `cuda_mixed` is disabled. `OFF`
keeps the existing no-socu build path.

The submodule is added with `EXCLUDE_FROM_ALL`, and its own tests are suppressed
while it is consumed by libuipc. The `cuda_mixed` target links `socu_native`
only when the target exists, and publishes `UIPC_WITH_SOCU_NATIVE=1`; otherwise
it publishes `UIPC_WITH_SOCU_NATIVE=0`.

A small compile-time contract was added to
`apps/tests/backends/cuda_mixed/policy_contract.cu`:

- `UIPC_WITH_SOCU_NATIVE` must always be defined as `0` or `1`;
- when it is `1`, the backend test must be able to include
  `socu_native/common.h` and see `socu_native::ProblemShape`.

Validation was performed on `build/build_impl_fp64`:

```bash
cmake -S . -B build/build_impl_fp64
cmake --build build/build_impl_fp64 --target backend_cuda_mixed --parallel 8
ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|sim_case_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

The configure step reported:

```text
socu_native integration enabled from /home/zhaofeng/work/libuipc/external/socu-native-cuda
```

The build compiled `external/socu-native-cuda/libsocu_native.a`, linked
`libuipc_backend_cuda_mixed.so`, and rebuilt `uipc_test_backend_cuda_mixed`.
The targeted tests passed:

```text
backend_cuda_mixed_contract: passed
backend_cuda_mixed_source_contract_scan: passed
sim_case_cuda_mixed_contract: passed
```

This checkpoint does not call `socu_native` from runtime yet. It only establishes
the dependency boundary needed for the next synthetic solve milestone.

## M0-M5 Strict Audit and Completion Patch

This pass rechecked Milestones 0 through 5 against the current integration plan
and tightened the two places where the implementation had drifted from the
plan's stricter wording.

### Audit Result

Milestone 0 is implemented as an opt-in experiment under
`experiments/socu_ordering_lab`. The root CMake option
`UIPC_BUILD_SOCU_ORDERING_LAB` defaults to `OFF` and adds only the experiment
directory when enabled. The lab exposes `socu_ordering_lab_core`,
`socu_ordering_lab_test`, and `socu_ordering_bench`. The CLI supports the
`order`, `reorder`, and `contact` subcommands and can emit JSON reports, CSV
summaries, and mapping CSV files. The ordering path covers deterministic RCM,
METIS-based candidates, NVIDIA symrcm when cuSolver is available, block sizes
`32` and `64`, and the common rod, cloth-grid, and tet-block presets.

Milestone 1 is implemented inside the same standalone experiment. The physical
reorder path passes `chain_to_old` as the `New2Old` mapping to
`AttributeCollection::reorder()`, then rewrites edge, triangle, and tetrahedron
topology through `old_to_chain`. The mirror path keeps the original geometry
order and only applies the solver-owned mapping. The test coverage checks
preserved counts, legal topology indices, geometric invariants, mirror versus
physical block classification, original-id metadata, and improved chain-order
memory stride.

Milestone 2 is implemented in the standalone lab. Contact classification is
reported at both primitive and expanded contribution levels. The lab covers
`near_band`, `mixed`, `adversarial`, and CSV-input scenarios; it records
near/off-band counts, contribution ratios, weighted dropped norm, contact
classify time, frame-boundary reorder time, and a fixed zero
Newton-iteration reorder count.

Milestone 3 is now stricter than the earlier checkpoint. `LinearSolver` exposes
the plan-shaped `AssemblyRequirements` fields:

```text
needs_dof_extent
needs_gradient_b
needs_full_sparse_A
needs_structured_chain
needs_preconditioner
```

`IterativeSolver` returns the full sparse matrix plus preconditioner
requirements, while `SocuApproxSolver` returns gradient plus structured-chain
requirements and explicitly does not request full sparse `A` or a
preconditioner. `GlobalLinearSystem` now reads the selected solver's
requirements before assembly, runs a gradient-only assembly path when full
sparse `A` is not needed, skips triplet-to-BCOO conversion and preconditioner
assembly for `socu_approx`, and skips sparse debug dump when no sparse matrix
was built. The default PCG path still requests and receives the original full
sparse assembly.

Milestone 4 is now strict about the optional dependency boundary. When
`linear_system/solver = "socu_approx"` is selected but the build does not have
`UIPC_WITH_SOCU_NATIVE=1`, the solver fails fast with `socu_disabled` and an
actionable message. When `socu_approx` is not selected, the unused solver path
still exits through the normal unused-system mechanism and does not affect the
default `fused_pcg` path.

Milestone 5 now has an actual CPU-side structured dry-run sink instead of only
report-level counters. The sink packs full padded logical buffers for `rhs`,
`diag`, and first `offdiag`; copies the live assembled RHS into
`old_dof -> block/lane` slots; initializes padding diagonal entries; writes
same-block structured contributions into the diagonal buffer; writes adjacent
block contributions into the first off-diagonal buffer; and records off-band
contributions as dropped without writing them into the structured buffers. The
dry-run JSON now reports active RHS scalar count, full padded scalar counts,
nonzero counts, structured diag/first-offdiag write counts, off-band drop
count, structured contact absolute sums, RHS absolute sum, and dry-run pack
time.

The current Milestone 5 runtime path still consumes standalone contact-report
JSON for structured synthetic contributions. Real contact/dytopo reporters and
FEM/ABD subsystems do not yet write directly into `StructuredAssemblySink`;
that remains future real-surrogate assembly work, not part of this dry-run
checkpoint. This boundary is intentional: M5 proves block layout, dry-run
packing, near/off-band classification behavior, and reporting without calling
`socu_native`.

### Verification

The build machine had 32 logical cores and about 56 GiB available memory at the
start of this verification pass. CUDA compilation was run with `-j4` to avoid
the memory spike observed with Ninja's default parallelism.

The experiment targets were configured, built, and tested with:

```bash
cmake -S . -B build/build_impl_fp64 -DUIPC_BUILD_SOCU_ORDERING_LAB=ON
ninja -C build/build_impl_fp64 -j8 socu_ordering_bench socu_ordering_lab_test
ctest --test-dir build/build_impl_fp64 -R socu_ordering_lab --output-on-failure
```

The result was:

```text
socu_ordering_lab: passed
```

The CUDA mixed implementation was rebuilt without rebuilding the ordinary
`cuda` backend:

```bash
ninja -C build/build_impl_fp64 -j4 \
  cuda_mixed backend_cuda_mixed \
  apps/tests/sim_case/CMakeFiles/sim_case.dir/86_cuda_mixed_precision_contracts.cpp.o
ctest --test-dir build/build_impl_fp64 -R 'backend_cuda_mixed' --output-on-failure
```

The result was:

```text
backend_cuda_mixed_contract: passed
backend_cuda_mixed_source_contract_scan: passed
```

The modified sim-case contract source
`apps/tests/sim_case/86_cuda_mixed_precision_contracts.cpp` was compile-checked
as an object. The full `uipc_test_sim_case` runtime was not rebuilt in this
pass because a dry-run Ninja query showed that linking it from the current cache
would rebuild the ordinary `cuda` backend, which was outside this verification
scope. The newly added M5 sim-case sections should be run in a full backend
validation pass when rebuilding the ordinary `cuda` backend is acceptable.

The final whitespace check passed:

```bash
git diff --check
```

## M6 Checkpoint: Synthetic `socu_native` Solve Smoke

Milestone 6 now has a backend-level synthetic solve smoke under
`apps/tests/backends/cuda_mixed/socu_native_synthetic_solve.cu`. This test does
not use a real assembled Hessian yet. It constructs a synthetic SPD
block-tridiagonal problem through `socu_native::generate_random_spd_block_tridiag`
and exercises the embedding API directly:

```text
query_solver_capability
describe_problem_layout
create_solver_plan
factor_and_solve_inplace_async
```

The test uses the planned first performance policy:

```text
SolverBackend::NativePerf
PerfBackend::MathDx
MathMode::Auto
GraphMode::Off
```

It validates both `n=32` and `n=64` layouts, checks `diag/off_diag/rhs` element
counts from `describe_problem_layout`, runs three repeated solves on one
long-lived plan, synchronizes only at the test boundary, computes a host-side
residual with `socu_native::residual_norm`, and checks that the resulting
direction satisfies a descent condition for `g = -rhs`.

The test has explicit runtime preflights:

- `UIPC_WITH_SOCU_NATIVE=1`;
- a CUDA device is available;
- `ActivePolicy::StoreScalar == ActivePolicy::SolveScalar`;
- the configured MathDx manifest path exists;
- a Warp native runtime library is discoverable through `SOCU_NATIVE_WARP_SO`
  or a repo-local `socu_native` virtual environment.

Without those artifacts, the test skips instead of failing the normal
`backend_cuda_mixed` contract. With artifacts present, it performs the real
`socu_native` async solve.

For this local `fp64` verification, the build tree was wired to the already
generated MathDx LTO artifacts from the sibling `socu-native-cuda` checkout:

```bash
ln -s /home/zhaofeng/work/socu-native-cuda/build/mathdx_lto \
  build/build_impl_fp64/mathdx_lto
```

The CUDA mixed backend test target was rebuilt with controlled parallelism:

```bash
ninja -C build/build_impl_fp64 -j4 backend_cuda_mixed
```

The ordinary no-env contract path passes and skips the synthetic runtime solve
when Warp is not configured:

```bash
ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

Result:

```text
backend_cuda_mixed_contract: passed
backend_cuda_mixed_source_contract_scan: passed
```

The real M6 solve path was then run with the Warp runtime from the sibling
checkout:

```bash
SOCU_NATIVE_WARP_SO=/home/zhaofeng/work/socu-native-cuda/.venv/lib/python3.13/site-packages/warp/bin/warp.so \
  ./build/build_impl_fp64/Release/bin/uipc_test_backend_cuda_mixed \
  "cuda_mixed_socu_native_synthetic_solve_smoke" -s
```

Result:

```text
All tests passed (84 assertions in 1 test case)
n=32 repeat residual ~= 9.99e-14, relative residual ~= 2.18e-15
n=64 repeat residual ~= 1.56e-13, relative residual ~= 2.46e-15
```

This completes the synthetic `socu_native` call-chain smoke for `fp64` on this
machine. It still does not mean `SocuApproxSolver` solves a real simulation
frame. The next checkpoint should connect the M5 dry-run packed
`diag/off_diag/rhs` buffers to a guarded `socu_native` solve path, then scatter
the direction back into the solver-owned displacement buffer and validate the
direction before line search.

## M7a Checkpoint: Guarded Surrogate Solve Path

Milestone 7a now has the first guarded `SocuApproxSolver` solve path. The config
surface gained:

```text
linear_system/socu_approx/mode = "solve" | "dry_run"
linear_system/socu_approx/damping_shift = 1.0
linear_system/socu_approx/descent_eta = 1e-12
linear_system/socu_approx/max_relative_residual = 1e-4
```

The default explicit `socu_approx` mode is now `solve`; existing M5-style tests
set `mode = "dry_run"` to preserve the dry-run-only behavior. In solve mode,
the solver performs MathDx artifact preflight during build, keeps a persistent
runtime object, and lazily creates the `socu_native::SolverPlan` on first solve
rather than during system initialization. The lazy plan creation matters because
Warp/MathDx touches CUDA context state; creating it during `do_build()` was too
early for full simulation startup.

The solve path now uses the M5 structured pack as actual input. It copies the
assembled global RHS, maps active old DoFs into chain lanes, absorbs any
structured near-band contributions from the contact report, applies diagonal
damping to every structured lane, expands first-offdiag storage to the full
`socu_native::describe_problem_layout()` size, and calls:

```text
factor_and_solve_inplace_async()
```

After the solve, it computes the surrogate residual, relative residual, descent
dot product for `g = -rhs`, gradient norm, and direction norm. Invalid,
non-finite, non-descent, or high-residual directions fail fast with
`direction_invalid`; `socu_native` exceptions are reported as
`socu_runtime_error`. Valid directions are scattered back to
`GlobalLinearSystem::x`, and the existing line-search path receives that
direction normally. There is still no implicit fallback to `fused_pcg`.

The JSON report mode is now either:

```text
structured_dry_run
structured_surrogate_solve
```

The report records pack time, `socu_factor_solve_time_ms`, scatter time,
damping shift, residuals, descent values, and final direction availability.

The sim-case contract source now contains an M7 world smoke section that writes
a sparse 64-block ordering for the single ABD tet scene and validates the M7
report after one frame. That source was compile-checked as an object. The full
`uipc_test_sim_case` binary was not rebuilt here because a Ninja dry run showed
that it would rebuild the ordinary `cuda` backend:

```text
ninja -C build/build_impl_fp64 -n uipc_test_sim_case
# first steps include 228 ordinary cuda backend rebuild/link actions
```

The current M7a boundary is important: this checkpoint connects the real
simulation gradient and the structured surrogate solve/scatter path, but the
Hessian surrogate is still damping plus structured contributions supplied
through the current report/sink path. Real FEM/ABD subsystem-native Hessian
writes into `StructuredAssemblySink` are still future work; they should be the
next strict checkpoint before claiming a real structured `\hat H` assembly
path.

### Verification

The CUDA mixed target and the updated sim-case contract source were built with
controlled parallelism:

```bash
cmake -S . -B build/build_impl_fp64
ninja -C build/build_impl_fp64 -j4 \
  backend_cuda_mixed \
  apps/tests/sim_case/CMakeFiles/sim_case.dir/86_cuda_mixed_precision_contracts.cpp.o
```

The normal backend contract path passed:

```bash
ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

The MathDx/Warp-enabled backend contract path also passed:

```bash
SOCU_NATIVE_WARP_SO=/home/zhaofeng/work/socu-native-cuda/.venv/lib/python3.13/site-packages/warp/bin/warp.so \
  ctest --test-dir build/build_impl_fp64 \
  -R 'backend_cuda_mixed_contract|backend_cuda_mixed_source_contract_scan' \
  --output-on-failure
```

The whitespace check passed:

```bash
git diff --check
```

### 2026-04-25 Addendum: Mixed-Only M7a World Contract and Runtime Smokes

The M7a world smoke is now executable without rebuilding the ordinary `cuda`
backend. A new test target was added under `apps/tests/sim_case`:

```text
sim_case_cuda_mixed_only
```

It compiles only `86_cuda_mixed_precision_contracts.cpp`, links only
`uipc::backend::cuda_mixed` plus `muda::muda`, and registers this focused CTest:

```text
sim_case_cuda_mixed_only_linear_solver
```

The registered Catch filter is the exact test case
`86_cuda_mixed_linear_solver_selection_smoke`, so it covers default
`linear_fused_pcg`, explicit `linear_pcg`, invalid solver rejection, M5 dry-run
packing, structured near/off-band contact writes, and the M7a surrogate solve
world section without pulling unrelated sim-case files or the ordinary CUDA
backend.

The M7a optional-environment behavior was tightened. If `socu_native`, MathDx
artifacts, or `SOCU_NATIVE_WARP_SO` are unavailable, the M7 section now emits a
Catch warning and returns instead of marking the entire CTest as skipped. This
keeps the default contract runnable on machines without the optional native
runtime. When `SOCU_NATIVE_WARP_SO` is provided, the same test executes the real
M7 surrogate solve.

The new target was dry-run checked before building:

```bash
ninja -C build/build_impl_fp64 -n sim_case_cuda_mixed_only
```

The dry-run only listed:

```text
catch_main.cpp.o
86_cuda_mixed_precision_contracts.cpp.o
uipc_test_sim_case_cuda_mixed_only
```

No `src/backends/cuda/` or `uipc_backend_cuda` build steps were present.

The focused target was built with controlled parallelism:

```bash
ninja -C build/build_impl_fp64 -j4 sim_case_cuda_mixed_only
```

The default no-Warp contract path passed:

```bash
ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$|^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

The strict MathDx/Warp-enabled path also passed:

```bash
SOCU_NATIVE_WARP_SO=/home/zhaofeng/work/socu-native-cuda/.venv/lib/python3.13/site-packages/warp/bin/warp.so \
  ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$|^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

The generated M7a report for
`linear_solver_socu_approx_m7_surrogate_solve` recorded:

```text
mode = structured_surrogate_solve
block_size = 32
block_count = 64
active_rhs_scalar_count = 12
surrogate_relative_residual = 0
descent_dot ~= -1.37e-16
dry_run_pack_time_ms ~= 0.17
socu_factor_solve_time_ms ~= 2.09
scatter_time_ms ~= 0.04
status.reason = none
direction_available = true
```

This completes a guarded world-level M7a surrogate solve validation for the
current fp64 build while keeping the ordinary CUDA backend out of the targeted
test cycle. It does not complete strict M7b because the surrogate Hessian is
not yet assembled by FEM/ABD/joint/constraint/coupling subsystem kernels
through a structured device sink.

Additional Python runtime smokes were run against the same fp64 build using
`UIPC_MODULE_DIR=build/build_impl_fp64/Release/bin` and
`PYTHONPATH=build/build_impl_fp64/python/src`. These examples all completed
with `cuda_mixed`:

```text
cuda_mixed_particle_ground_viewer.py --smoke-frames 3
cuda_mixed_abd_bdf2_viewer.py --smoke-frames 3
cuda_mixed_fem_mas_hybrid_viewer.py --smoke-frames 2
cuda_mixed_prismatic_joint_limit_viewer.py --smoke-frames 2
cuda_mixed_revolute_joint_limit_viewer.py --smoke-frames 2
```

The `uipc_assets` benchmark harness was also exercised on a small real asset:

```bash
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
  apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/smoke.json \
  --scene cube_ground \
  --levels fp64 \
  --build fp64=build/build_impl_fp64 \
  --config Release \
  --run_root output/benchmarks/mixed/uipc_assets/socu_m7_cube_ground_smoke \
  --resume
```

Result:

```text
run finished: output/benchmarks/mixed/uipc_assets/socu_m7_cube_ground_smoke
run_meta.failures = []
perf frames = 40
quality frames = 20
```

Because this run used only the `fp64` level, the benchmark report does not
produce cross-level regression rows. It still validates that the real asset
runner, Python bindings, `cuda_mixed` backend load path, timer output,
solution dumps, and report generation remain healthy after the M0-M7 changes.

At this point, M0-M6 and M7a are complete for their documented checkpoint
boundaries: M0-M2 remain experiment/lab artifacts, M3-M5 establish the
no-regression solver abstraction and structured dry-run path, M6 validates
direct `socu_native` synthetic solve calls, and M7a validates the guarded
world-level surrogate solve/scatter path. Strict M7b remains open: replace the
current damping/report-driven surrogate Hessian with real subsystem-native
structured Hessian writes before treating `socu_approx` as a production
solver.

## 2026-04-25 Correction: Strict M7b Route

The previous wording around M7 was too strong. The implementation should now
be treated as a clean M7a baseline, not as strict M7 completion. In particular,
the current `SocuApproxSolver` consumes the real global gradient and calls
`socu_native`, but it does not yet receive real FEM/ABD/joint/constraint/
coupling Hessian contributions through a device structured assembly sink.

The strict M7b route should follow the main integration plan rather than a
triplet-projection shortcut. `socu_approx` must keep `needs_full_sparse_A =
false`; production solve must not build full sparse A, reverse-engineer BCOO,
or expose triplet scratch as a structured side channel. A legacy sparse
projection may exist only as a test oracle.

The next implementation checkpoint is therefore:

1. Introduce an explicit Newton assembly mode with at least `FullSparse` and
   `GradientStructuredHessian`, and propagate the selected solver requirements
   before linear subsystem, dytopo, and contact assembly.
2. Add a zero-behavior `TripletAssemblySink` wrapper around the existing full
   sparse path and verify no-regression before structured writes are enabled.
3. Add a compile-time/static `StructuredDeviceAssemblySink` for a strict
   first scene family, with device-side old-DoF-to-chain-slot mapping and
   deterministic writes into `diag`, `first_offdiag`, and `rhs`.
4. Gate unsupported active subsystem or contact Hessian contributions with an
   explicit fatal reason such as `M8_contact_runtime_not_supported`; do not
   silently fall back to gradient-only or full sparse.
5. Keep pack time, solve time, write counts, dropped/off-band counts, and
   strict gate reason in the report so tests can distinguish M7b success from
   M7a fallback behavior.

This correction also rules out the temporary idea of assembling triplets for
`needs_structured_chain` inside `GlobalLinearSystem` and passing those triplets
to `SocuApproxSolver`. That path would be useful only as an oracle/debug mode;
it is not the strict runtime design.

### First Corrective Patch

The first code correction makes the Newton assembly mode explicit without yet
claiming structured Hessian assembly support. `LinearSolver::AssemblyRequirements`
now carries a `NewtonAssemblyMode`. The existing iterative solvers declare
`FullSparse`, while `SocuApproxSolver` declares `GradientStructuredHessian`.
`GlobalLinearSystem` validates that these mode declarations match the older
boolean requirements, so an inconsistent solver cannot accidentally request a
full sparse path while pretending to be structured, or vice versa.

`advance_ipc` and `advance_al` now pass the selected linear solver assembly
mode into `GlobalDyTopoEffectManager`. Because runtime contact/dytopo structured
Hessian assembly is reserved for M8, any active dytopo/contact reporter under
`GradientStructuredHessian` now fails explicitly with
`M8_contact_runtime_not_supported`. This is intentionally conservative: it
prevents `socu_approx` from silently running a gradient-only contact path and
keeps strict M7b scoped to subsystem-native structured writes.

The same M8 contact gate is also checked during `SocuApproxSolver::do_build()`
when `contact/enable = true`, so ordinary scene initialization reports the
unsupported state before the Newton loop reaches dytopo assembly. A focused
sim-case section now asserts that `socu_approx` rejects runtime contact with
the `M8_contact_runtime_not_supported` reason.

Verification for this corrective patch:

```bash
cmake --build build/build_impl_fp64 --target sim_case_cuda_mixed_only -j2
ctest --test-dir build/build_impl_fp64 \
  -R '^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$' \
  --output-on-failure
```

All three checks passed in the fp64 build. The full `cuda_mixed` target rebuild
was triggered by the new assembly-mode header and completed with the existing
non-fatal warnings only.

### Second Corrective Patch: Sink Boundary Split

The next correction removes the host-side virtual `StructuredAssemblySink` from
the `StructuredChainProvider` runtime contract. That virtual sink was a useful
M5 dry-run convenience, but it violated the strict plan's requirement that the
hot assembly path use compile-time/static sinks and avoid device-side virtual
dispatch or type erasure. The provider now exposes ordering metadata and an
assembly entrypoint without taking a host virtual sink object.

A new `src/backends/cuda_mixed/utils/assembly_sink.h` file introduces the Phase
2A `TripletAssemblySink`. It is a thin compile-time wrapper around the existing
`DoubletVectorAssembler` and `TripletMatrixAssembler`, with inline
`write_gradient()` and `write_hessian_half()` helpers. This patch does not yet
switch FEM/ABD kernels over to the sink; that migration must be done
subsystem-by-subsystem with no-regression checks.

The old CPU dry-run sink remains local to `SocuApproxSolver` as
`HostStructuredDryRunSink`/`CpuStructuredDryRunSink`. It is now explicitly a
report/test helper for M5/M7a, not the runtime structured assembly contract for
strict M7b.

Verification for this sink-boundary patch:

```bash
cmake --build build/build_impl_fp64 \
  --target uipc_test_backend_cuda_mixed sim_case_cuda_mixed_only -j2
ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$|^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
```

The focused backend and mixed-only sim-case checks passed. The compile emitted
only the existing non-fatal virtual hiding warnings in the contract compile
unit.

### Phase 2A Initial FEM Triplet Sink Migration

The first real `TripletAssemblySink` call site has been introduced on the FEM
side. `FEM3DConstitution::ComputeGradientHessianInfo` and
`FiniteElementElastics::ComputeGradientHessianInfo` now expose a `sink()`
accessor that returns the static `TripletAssemblySink`. The
`StableNeoHookean3D` constitution uses this sink for its gradient and
upper-triangular Hessian writes instead of constructing
`DoubletVectorAssembler` and `TripletMatrixAssembler` directly inside the
kernel.

This is intentionally narrow: it proves the sink can be captured in CUDA
kernels and still writes through the existing triplet machinery, but it does
not yet migrate every FEM constitution or introduce structured writes. The
legacy sparse path remains the only production path for this constitution.

Verification:

```bash
cmake --build build/build_impl_fp64 \
  --target uipc_test_backend_cuda_mixed sim_case_cuda_mixed_only -j2
ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$|^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
```

The focused checks passed. A Python FEM smoke was attempted with
`python/examples/cuda_mixed_fem_mas_hybrid_viewer.py --smoke-frames 2`, but the
system `python3` environment is missing `numpy`, so that smoke was not run in
this pass.

A follow-up run used the repository virtual environment and passed:

```bash
UIPC_MODULE_DIR=build/build_impl_fp64/Release/bin \
PYTHONPATH=build/build_impl_fp64/python/src \
python/.venv/bin/python python/examples/cuda_mixed_fem_mas_hybrid_viewer.py \
  --smoke-frames 2
```

### Phase 2A Additional FEM Constitution Sink Migration

The next sink migration pass extends the static `TripletAssemblySink` pattern to
more FEM constitution writers that already followed the
`DoubletVectorAssembler` + `TripletMatrixAssembler` shape. `ARAP3D`,
`DiscreteShellBending`, and `KirchhoffRodBending` now write gradients and
upper-triangular Hessians through `info.sink()` instead of constructing the
assemblers directly in each kernel.

`FiniteElementExtraConstitution::ComputeGradientHessianInfo` now forwards the
same sink accessor from `FiniteElementElastics::ComputeGradientHessianInfo`, so
extra FEM constitutions can use the common compile-time sink boundary without a
separate adapter. This still preserves the legacy full sparse triplet path; it
does not yet introduce the strict M7b structured device sink for `diag` and
`first_offdiag`.

Verification:

```bash
cmake --build build/build_impl_fp64 \
  --target uipc_test_backend_cuda_mixed sim_case_cuda_mixed_only -j2
ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$|^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only '[fem]'
build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only '[shell]'
UIPC_MODULE_DIR=build/build_impl_fp64/Release/bin \
PYTHONPATH=build/build_impl_fp64/python/src \
python/.venv/bin/python python/examples/cuda_mixed_fem_mas_hybrid_viewer.py \
  --smoke-frames 2
```

All checks passed. The `[fem]` filter covered the mixed FEM hybrid and rod
contract cases, and the `[shell]` filter covered the shell contract case.

### Phase 2A Remaining FEM Constitution Sink Migration

The remaining FEM constitution writers that still constructed
`DoubletVectorAssembler` or `TripletMatrixAssembler` directly have now been
migrated to the same static `TripletAssemblySink` boundary. This pass covers
the codimension and plastic shell family:

```text
HookeanSpring1D
NeoHookeanShell2D
StrainLimitingBaraffWitkinShell2D
PlasticDiscreteShellBending
StressPlasticDiscreteShellBending
```

`Codim1DConstitution::ComputeGradientHessianInfo` and
`Codim2DConstitution::ComputeGradientHessianInfo` now expose `sink()` in the
same style as the 3D and extra FEM paths. A source scan now finds no remaining
direct `DoubletVectorAssembler` or `TripletMatrixAssembler` use under
`src/backends/cuda_mixed/finite_element`.

Verification:

```bash
cmake --build build/build_impl_fp64 \
  --target uipc_test_backend_cuda_mixed sim_case_cuda_mixed_only -j2
ctest --test-dir build/build_impl_fp64 \
  -R '^backend_cuda_mixed_contract$|^backend_cuda_mixed_source_contract_scan$|^sim_case_cuda_mixed_only_linear_solver$' \
  --output-on-failure
build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only '[fem]'
build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only '[shell]'
build/build_impl_fp64/Release/bin/uipc_test_sim_case '20_spring_fixed_point'
build/build_impl_fp64/Release/bin/uipc_test_sim_case '19_shell_fixed_point'
build/build_impl_fp64/Release/bin/uipc_test_sim_case '33_discrete_shell_bending'
build/build_impl_fp64/Release/bin/uipc_test_sim_case \
  '76_discrete_shell_bending_strain_plastic_open_strip'
build/build_impl_fp64/Release/bin/uipc_test_sim_case \
  '81_stress_plastic_discrete_shell_bending_open_strip'
```

All checks passed. The ordinary `uipc_test_sim_case` checks were limited to the
smallest direct constitution coverage points and their output was redirected
when needed to avoid excessive terminal output.

## 2026-04-26

### Strict M7 Alignment Repair Pass

This pass fixes the strict-alignment gaps identified against
`socu_mixed_solver_integration_plan.md` and narrows the active solve path to the
planned Level 1 surface: ABD-only, complete DoF coverage, no runtime
contact/dytopo/FEM/offdiag subsystems, equal store/solve scalar type, and block
size `32` or `64`. The default `fused_pcg` solver remains unchanged.

`GlobalLinearSystem` now routes solvers that request a structured chain through
a separate structured assembly path after RHS assembly, instead of requiring the
legacy sparse triplet/BCOO path. The new `LinearSolver` hooks let a solver
prepare structured workspace, expose a device structured sink, finalize the
chain, and receive line-search feedback. `DiagLinearSubsystem` now has explicit
structured assembly capability hooks; unsupported subsystems fail with a clear
reason instead of silently contributing nothing.

`SocuApproxSolver` now owns device-side structured workspace for `diag`,
`first_offdiag`, RHS, solution storage, mappings, validation sums, and pre-factor
matrix snapshots. The production solve branch no longer loads `contact_report`
JSON or uses the CPU dry-run sink. It uses the mixed backend stream, calls
`socu_native::factor_and_solve_inplace_async`, validates residual and descent
against the pre-factor `diag/offdiag` snapshot, and scatters the direction on
device. The dry-run `contact_report` path remains debug-only.

The structured provider contract now requires complete old-DoF to chain-DoF
coverage and reports active/padding counts, duplicate/missing old DoFs,
duplicate/missing chain DoFs, block utilization, near/off-band ratios, and
drop-norm ratio. Any incomplete, duplicated, or padding-overlapping mapping is
a gate failure. Default SOCU gate thresholds in the scene config were changed
from permissive to strict: near-band ratio `0.90`, off-band ratio `0.10`, block
utilization `0.65`, and off-band drop-norm ratio `0.05`.

ABD now implements the first strict structured assembly path by reusing the ABD
kinetic and constitution Hessian computation, writing full `12x12` body
Hessians into the structured diagonal sink, and rejecting reporters/joints plus
non-empty dytopo/contact effect buffers. Empty dytopo receivers do not block an
ABD-only scene. FEM and runtime contact remain explicitly unsupported for this
solve mode.

The report schema now records the stricter gate fields requested by the plan,
including dtype, backend policy, MathDx manifest/cache preflight status, RHS
sign convention, stream source, direction norm, descent, residual, and
line-search feedback. The MathDx preflight now checks the manifest path,
runtime cache directory, artifact presence, dtype/block/nrhs/function match,
and device-arch coverage before creating a plan.

Verification:

```bash
cmake --build build/build_impl_fp64 --target cuda_mixed -- -j2
cmake --build build/build_impl_fp64 --target backend_cuda_mixed -- -j2
./build/build_impl_fp64/Release/bin/uipc_test_backend_cuda_mixed \
  "[cuda_mixed][contract]"
cmake --build build/build_impl_fp64 --target sim_case_cuda_mixed_only -- -j2
./build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only \
  "86_cuda_mixed_linear_solver_selection_smoke"
```

The focused backend contract and linear-solver smoke tests passed. The strict
M7 solve and coverage-gate runtime sections were compiled and reached their
availability checks, but were skipped on this machine because `warp.so` was not
discoverable; the skip message recommends setting `SOCU_NATIVE_WARP_SO`.
