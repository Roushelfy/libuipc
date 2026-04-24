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
