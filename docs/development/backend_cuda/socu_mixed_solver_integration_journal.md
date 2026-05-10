# SOCU Mixed Solver Notes

The staged implementation journal has been retired. The supported behavior is
now documented in
[`socu_mixed_solver_integration_plan.md`](socu_mixed_solver_integration_plan.md).

Current policy:

- Keep `socu_native` integration, `socu_approx_solver`, structured sink
  assembly, and direct local linear build writes as the only runtime path.
- Keep ordinary IPC contact assembly free of SOCU structured sink captures;
  SOCU contact assembly uses separate direct structured contact kernels.
- Do not ship contact vertex-slot tables or write-plan caches in the current
  runtime path; revisit them only as measured follow-up optimizations.
- Treat `wrecking_ball fused_pcg --frames 20` as the ordinary contact baseline;
  it must complete all 20 frames after any SOCU contact split.
- Keep `experiments/socu_ordering_lab` as a standalone ordering lab.
- Treat ordering quality and off-band ratios as report diagnostics, not solve
  gates.
- Use `socu_rt1_contact_hessian` and `socu_rt1_full_hessian` wrecking-ball
  variants for runtime reorder graph-source diagnosis; keep `socu_rt1` as the
  topology default.
- Do not add dense matrix eigen diagnostics or full Hessian triplet fallback to
  the SOCU runtime path.

Future SOCU changes should update the final behavior document instead of adding
new staged journal entries.

## Measured Follow-Ups

### Structured Contact Sink Assembly Cost

`socu_rt1_full_hessian` can pass the wrecking-ball 20-frame acceptance case, but
the timer trace shows that contact structured Hessian assembly dominates the
runtime while `socu_native` factor/solve is comparatively small. The next
performance investigation keeps the current direct structured contact kernels
intact and measures isolated sink overhead before reintroducing larger
precomputed plans.

Investigation order:

1. Counter-off timing. Run the same full-Hessian runtime reorder case with
   `debug_validation`, `debug_timing`, and `report_each_solve` disabled so
   `StructuredContactAssemblySink::counters` is null and near/off-band counter
   atomics are skipped. Compare `Assemble Contact` duration and frame time
   against the current diagnostic run.
2. Vertex-slot table. If counter removal is not the main cost, prototype a
   reorder-level device table indexed by global vertex that stores the already
   resolved FEM/ABD kind, fixed state, old DoF, ABD body, and ABD Jacobian
   index. This should remove repeated `global vertex -> old DoF` lookup inside
   every contact half-block without changing Hessian values or write semantics.
3. Block-level sink writes. Classify FEM/FEM, ABD/FEM, and ABD/ABD atom-blocks
   once per 3x3 block instead of calling scalar status logic for every scalar.
   Keep counters optional so production timing does not pay per-scalar atomic
   cost.
4. Compact Hessian reuse. Only after the cheaper items are measured, consider a
   narrow compact contact Hessian workspace for full-Hessian runtime reorder:
   compute contact Hessians during the graph probe, collect ordering weights
   from those values, then scatter the cached Hessians after the reorder install.
   This avoids recomputing contact Hessians in the final structured pass, but it
   should stay an explicit measured experiment because it trades compute for
   extra memory traffic and workspace lifetime management.

Initial counter-off result:

- Baseline:
  `python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_rt1_full_hessian --frames 20`
  with the script's default SOCU diagnostics enabled.
  - `wall_time_s = 434.75`
  - `Assemble Contact = 347.13s / 764 calls`
  - `Assemble Structured DyTopo Hessian = 347.15s / 122 calls`
  - `Solve Linear System = 0.45s / 69 calls`
- Counter-off diagnostic run:
  `SOCU_REPORT_COUNTERS=0 python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_rt1_full_hessian --frames 20`
  disables the diagnostic flags that allocate `StructuredContactAssemblySink`
  counters.
  - `wall_time_s = 448.46`
  - `Assemble Contact = 365.37s / 776 calls`
  - `Assemble Structured DyTopo Hessian = 365.39s / 124 calls`
  - `Solve Linear System = 0.35s / 70 calls`

The counter-off run completed all 20 frames, but did not reduce contact assembly
time. Frame 19 took one extra Newton/structured build in that run, so the total
wall time is not a perfect apples-to-apples comparison; however, per contact
assembly call also did not improve. This makes counter atomics a secondary issue
for the current full-Hessian bottleneck. The next useful optimization target is
therefore repeated vertex mapping / band classification and, after that,
avoiding the probe/final double Hessian computation.

### Rejected Vertex-Slot Table Prototype

2026-05-03: A prototype reorder-level `StructuredContactVertexSlot` table was
implemented and tested, but rejected. The table cached FEM/ABD kind, fixed
state, old DoF, ABD body, ABD Jacobian index, and chain metadata, and
`StructuredContactAssemblySink::map_vertex()` used it before falling back to the
old range lookup path.

Build notes:

- `libuipc_backend_cuda_mixed.so` required `CMAKE_CUDA_ARCHITECTURES=120-real`
  in this build directory to avoid CUDA 12.8 device-link segmentation faults on
  RTX 5090.
- `ipc_simplex_normal_contact.cu.o` still needed a one-off non-fast-compile
  object build to avoid the known ptxas ICE.
- `86_cuda_mixed_linear_solver_selection_smoke` passed after the prototype.

Functional rejection:

- `SOCU_REPORT_COUNTERS=0 python/examples/cuda_mixed_wrecking_ball_compare.py
  --variant socu_rt1_full_hessian --frames 20 --output
  output/examples/cuda_mixed_wrecking_ball_compare_vertex_slots` failed during
  frame 13.
- The first failure was a `segmental_reduce` device bounds check:
  `Dense1D[out:segmental_reduce]: out of range, index=(-1) m_dim=(1280)`,
  followed by `cudaErrorLaunchFailure`.
- Adding a device synchronization after the slot rebuild did not fix the
  failure; the synced run failed at the same frame 13 point under
  `output/examples/cuda_mixed_wrecking_ball_compare_vertex_slots_sync`.

Decision: revert the vertex-slot code and do not commit it. The failure suggests
that caching only per-global-vertex mapping is not a semantics-free drop-in for
the current structured contact path, or that stale/invalid cached chain metadata
can poison downstream reductions when the contact set changes. Revisit this only
with a matrix-diff harness or a narrower cache that is consumed by a block-level
writer with explicit validity checks.

### Rejected Block-Level Structured Sink Prototype

2026-05-03: A block-level `StructuredContactAssemblySink` prototype was
implemented and then reverted before commit. The prototype classified 3x3 atom
blocks once, wrote diag/first-offdiag storage directly when the three DoFs were
contiguous inside one SOCU block, aggregated scalar counters by block, and fell
back to the existing scalar path whenever runtime graph collection was active or
the layout assumptions were not met.

Build and short tests:

- `libuipc_backend_cuda_mixed.so` linked successfully with the same CUDA 12.8
  build workarounds used above: build directory configured for `120-real`,
  `NVCC_APPEND_FLAGS='--Ofast-compile=max'`, and a one-off
  `ipc_simplex_normal_contact.cu.o` build with `-arch=sm_120` for the known
  ptxas ICE.
- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed.
- `uipc_test_backend_cuda_mixed "[cuda_mixed][contract]" -s` passed.

Functional rejection / blocker:

- The required baseline
  `SOCU_REPORT_COUNTERS=0 python/examples/cuda_mixed_wrecking_ball_compare.py
  --variant fused_pcg --frames 20 --output
  output/examples/cuda_mixed_wrecking_ball_compare_step2_fused_pcg` failed in
  frame 9 after `IPCVertexHalfPlaneNormalContact` reported `PHs: 320` and
  `DyTopo Hess3x3 count: 320`.
- The device failure was
  `Dense1D[out:segmental_reduce]: out of range, index=(-1) m_dim=(320)`,
  followed by `cudaErrorLaunchFailure`.
- The block-level sink patch was then reverted and the backend was fully
  relinked. The same `fused_pcg --frames 20` run still failed at the same frame
  9 PH contact point under
  `output/examples/cuda_mixed_wrecking_ball_compare_step2_reverted_fused_pcg`.

Decision: do not commit the block-level sink code. The failure is not caused by
the block-level structured sink patch, but the current FullSparse fused-PCG
baseline is not healthy enough to accept SOCU structured sink performance work.
Before retrying block-level writes, fix the FullSparse PH contact
`segmental_reduce index=-1` regression and rerun the fused-PCG 20-frame
acceptance case. The prototype patch is intentionally left out of the working
tree; during this experiment it was saved only as the local scratch file
`/tmp/socu_step2_structured_sink.patch`.

### Step 0 Clean Baseline for Structured Contact Retest

2026-05-03: The structured contact optimization retest was restarted from a
clean build policy: `CMAKE_CUDA_ARCHITECTURES=120-real`,
`UIPC_CUDA_ARCHITECTURES=120-real`, and no `NVCC_APPEND_FLAGS`. This avoids the
previous `--Ofast-compile=max` build-artifact ambiguity.

Static checks:

- `git diff --check` passed.
- `matrix_converter.inl`, `fast_segmental_reduce.inl`,
  `structured_contact_assembly_sink.h`, and `global_dytopo_effect_manager.cu`
  had no diff.
- `StructuredContactVertexSlot` and `--Ofast-compile=max` remained only in
  journal text, not active source.

Functional baseline:

- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed: 165 assertions.
- `SOCU_REPORT_COUNTERS=0 ... --variant fused_pcg --frames 20` passed with
  `final_frame=20`, `wall_time_s=4.240158434011391`, and
  `mean_frame_ms=95.27113550066133`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_full_hessian --frames 20`
  passed with `final_frame=20`, `wall_time_s=5.9441658739960985`, and
  `mean_frame_ms=198.3644207510224`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_contact_hessian --frames 20`
  reached frame 16 and failed at the known light direction validation issue:
  `finite=false`, `nonfinite_count=6888`, `p_norm=0`.

Frame 13-20 timing baseline:

- `socu_rt1_full_hessian` all `Assemble Contact`: `1.035016538s / 516 calls =
  2.005846 ms/call`.
- `socu_rt1_full_hessian` structured-chain `Assemble Contact`:
  `0.965702684s / 344 calls = 2.807275 ms/call`.
- `fused_pcg` FullSparse `Assemble Contact`: `0.339493971s / 180 calls =
  1.886078 ms/call`.

Decision: use this as the baseline for the next conservative vertex-slot
retest. A step is accepted only if functionality does not regress and
`socu_rt1_full_hessian` structured contact assembly improves by at least 5%.

### Step 1 Conservative Vertex Slot Table Retest

2026-05-03: A conservative device vertex-slot table was tested without
`NVCC_APPEND_FLAGS` and with the build directory configured for `120-real`.
The slot table cached only per-global-vertex FEM/ABD kind, fixed state, old DoF,
body, and local vertex. It did not cache final chain addresses, band
classification, or ABD Jacobian values.

Functional results:

- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed: 165 assertions.
- `SOCU_REPORT_COUNTERS=0 ... --variant fused_pcg --frames 20` passed with
  `final_frame=20`, `wall_time_s=5.19305259900284`, and
  `mean_frame_ms=135.31200174620608`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_full_hessian --frames 20`
  passed with `final_frame=20`, `wall_time_s=7.362715014023706`, and
  `mean_frame_ms=235.6725092002307`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_contact_hessian --frames 20`
  reached `>>> Begin Frame: 16` and failed in the same known light direction
  validation mode: `finite=false`, `nonfinite_count=6888`, `p_norm=0`.

Frame 13-20 `socu_rt1_full_hessian` timing:

- all `Assemble Contact`: `1.354156915s / 528 calls = 2.564691 ms/call`.
- structured-chain `Assemble Contact`: `1.280427552s / 352 calls =
  3.637578 ms/call`.

Decision: reject and revert the Step 1 source changes. Compared with the Step 0
structured-chain baseline of `2.807275 ms/call`, the vertex-slot table regressed
to `3.637578 ms/call` instead of improving by at least 5%. The likely cause is
that the added slot-table rebuild and extra indirection did not remove enough of
the hot sink work to offset the additional load/branch pressure. The active
source tree was restored to the Step 0 code shape; this journal entry records
the rejected measurement only.

### Step 2 Block-Level Structured Writes Compile Rejection

2026-05-03: A block-level structured sink prototype was attempted directly on
top of the Step 0 source shape, after Step 1 was rejected. The prototype kept
runtime graph collection on the existing scalar path and only changed the real
matrix-write path: 3x3 atom-pair classification, direct diag/first-offdiag
writes when the three DoFs were contiguous inside one SOCU block, aggregated
scalar counters by atom pair, and scalar fallback for invalid layout or active
runtime collector.

Compile rejection:

- The user build was run with `NVCC_APPEND_FLAGS` unset, as required by the
  clean retest plan.
- CUDA compilation of
  `contact_system/contact_models/ipc_simplex_frictional_contact_structured.cu.o`
  consumed the available memory and filled the 256 GiB swap space.
- No `--Ofast-compile=max` workaround was used.

Decision: reject and revert this Step 2 prototype before functional or timing
tests. The patch made `structured_contact_assembly_sink.h` too template-heavy
for the largest structured frictional contact translation unit. The active
source tree was restored to the Step 0 code shape. The rejected prototype was
saved as `/tmp/socu_step2_block_level_sink_oom.patch` for reference only.

Next retry should not put block-level ABD/FEM/ABD projection helpers into the
shared sink header. A safer shape is either a much narrower FEM/FEM-only
fast-path experiment, or contact-kernel-local non-template direct-write helpers
inside selected `*_structured.cu` files so the frictional TU does not instantiate
all block-level variants through the shared header.

### Step 3 Approximate Graph Probe Source

2026-05-03: Graph-probe specialization was narrowed to the experimental
approximate weighted graph source. The exact `contact_hessian` and
`full_hessian` semantics were not changed: exact weighted probes still compute
the contact Hessian and collect the existing exact graph weights. The new
experimental sources are:

- `contact_weight_approx`
- `full_weight_approx`

Approximate weights use only contact coefficients:

- normal contact: `abs(kappa * dt * dt)`
- frictional contact: `abs(kappa * mu * dt * dt)`

Functional results:

- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed: 193 assertions.
- `SOCU_REPORT_COUNTERS=0 ... --variant fused_pcg --frames 20` passed with
  `final_frame=20`, `wall_time_s=4.511972936999882`, and
  `mean_frame_ms=105.5394023999952`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_full_hessian --frames 20`
  passed with `final_frame=20`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_full_weight_approx --frames
  20` passed with `final_frame=20`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_contact_hessian --frames 20`
  reached frame 16 and failed in the known light direction validation mode.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_contact_weight_approx
  --frames 20` also reached frame 16 and failed in the same known mode.

Frame 13-20 timing:

- Exact `socu_rt1_full_hessian` structured-chain `Assemble Contact`:
  `1.123955734s / 352 calls = 3.193056 ms/call` in the first run.
- Approximate `socu_rt1_full_weight_approx` structured-chain
  `Assemble Contact`: `0.809257999s / 352 calls = 2.299028 ms/call` in the
  paired first run.
- A later serial repeat had more timing noise and different Newton/contact
  counts, but kept the same direction: exact `3.718594 ms/call`, approximate
  `2.930782 ms/call`.

Decision: accept only the experimental approximate graph source. The exact
`full_hessian` path did not improve over the Step 0 baseline, so exact graph
source specialization is not accepted. The approximate `full_weight_approx`
source is kept as an opt-in benchmark/research path because it passed 20 frames
and was consistently faster than exact `full_hessian` under the same Step 3
build. It does not replace the default graph source.

Follow-up verification after restoring the pre-narrow Step 3 shape:

- `git diff --check` passed.
- `matrix_converter.inl` and `fast_segmental_reduce.inl` had no diff.
- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed: 193 assertions.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_full_weight_approx --frames
  20` passed with `final_frame=20`, `wall_time_s=6.894915459999538`, and
  `mean_frame_ms=243.63898474998678`.
- Frame 13-20 `full_weight_approx` structured-chain `Assemble Contact`:
  `0.928372598s / 344 calls = 2.698758 ms/call`.

This follow-up run did not reproduce the earlier best `2.299028 ms/call`
measurement, but it still remained slightly faster than the Step 0
`full_hessian` baseline of `2.807275 ms/call`. Treat this graph source as an
experimental option rather than a default replacement; further acceptance should
use multiple serial runs and median timing.

Five additional serial runs of `socu_rt1_full_weight_approx --frames 20` on the
same pre-narrow Step 3 build all completed `final_frame=20`:

| run | wall time | mean frame | frame 13-20 structured-chain contact |
| --- | ---: | ---: | ---: |
| 1 | `6.2491455569997925s` | `207.75674609990347 ms` | `2.273468 ms/call` |
| 2 | `5.656144983000559s` | `207.37094015007642 ms` | `2.241910 ms/call` |
| 3 | `5.66265281699998s` | `210.10174875000303 ms` | `2.272197 ms/call` |
| 4 | `5.565006402000108s` | `208.29522045000886 ms` | `2.248593 ms/call` |
| 5 | `5.467283701999804s` | `203.79408369990415 ms` | `2.288049 ms/call` |

Median structured-chain contact time: `2.272197 ms/call`. Mean:
`2.264843 ms/call`. Compared with the Step 0 `full_hessian` baseline
`2.807275 ms/call`, the median improvement is `19.06%` and the mean
improvement is `19.32%`.

Decision update: accept the experimental `full_weight_approx` /
`contact_weight_approx` graph sources as opt-in variants. They remain
non-default, but the multi-run median clears the 5% performance threshold.

### Step 4 Compact Hessian Cache Experiment

2026-05-04: Added the benchmark-only runtime graph source
`full_hessian_cached`. It keeps exact `full_hessian` graph weights, but when a
runtime reorder is successfully installed it reuses the contact half-block
Hessians computed during the graph probe for the immediately following final
structured contact assembly. The cache is one-shot and frame-local; default
`full_hessian`, `topology`, and approximate graph sources are unchanged.

Implementation shape:

- Contact graph probe appends compact half-block records
  `(global_i, global_j, mirror_diag_block, H3x3)` while collecting exact graph
  weights.
- Final structured assembly replays those cached records through
  `StructuredContactAssemblySink::write_contact_half_block`.
- The replay launcher lives in `structured_contact_hessian_cache.cu`, so the
  shared sink header no longer owns the `ParallelFor` replay kernel body.
- Cache overflow, empty cache, stale frame, or a skipped runtime reorder falls
  back to the existing recompute path.

Validation:

- `git diff --check` passed.
- `matrix_converter.inl` and `fast_segmental_reduce.inl` had no diff.
- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed: 207 assertions.
- `SOCU_REPORT_COUNTERS=0 ... --variant fused_pcg --frames 20` passed with
  `final_frame=20`, `wall_time_s=4.197232440999869`, and
  `mean_frame_ms=105.11000325004716`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_full_hessian_cached
  --frames 20` passed with `final_frame=20`.
- `SOCU_REPORT_COUNTERS=0 ... --variant socu_rt1_contact_hessian --frames 20`
  reached frame 16 and failed in the known light direction validation mode.
- Debug dumps for frame 1 Newton iterations 0 and 1 matched exactly between
  `socu_rt1_full_hessian` and `socu_rt1_full_hessian_cached`: same nonzero
  structure and zero value difference for `A_structured.1.0.mtx` and
  `A_structured.1.1.mtx`.

Frame 13-20 timing, excluding one exact run with negative timer duration:

| variant | runs | wall median | mean frame median | contact + replay median | structured DyTopo median | Build Linear System median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `socu_rt1_full_hessian` | 3 | `6.382151s` | `212.377556 ms` | `2.240269 ms/call` | `13.095898 ms/call` | `61.231031 ms/call` |
| `socu_rt1_full_hessian_cached` | 3 | `5.659183s` | `213.919838 ms` | `1.060268 ms/call` | `4.417649 ms/call` | `35.551243 ms/call` |

The compact cache reduces the contact-plus-replay median by `52.67%` and the
structured DyTopo median by `66.27%` relative to the paired exact
`full_hessian` runs. Total wall time also improved in the median run, although
`full_hessian_cached` performed more Build Linear System calls in these 20-frame
runs (`61` median vs `43`), so it remains an opt-in benchmark source rather
than replacing `full_hessian`.

Decision: accept `full_hessian_cached` as an experimental graph source. It is
not a default path. Follow-up work should investigate why cached runs take more
structured solves despite matching the first dumped matrices exactly.

### Step 4 Diagonal Cache Fix

2026-05-04: Follow-up validation found that the first cached structured matrix
matched exact `full_hessian`, but frame 13 Newton iteration 1 diverged even
when the runtime graph and selected ordering were identical. The missing piece
was `StructuredContactAssemblySink::write_hessian(global_vertex, H)`: the graph
probe recorded scalar graph weights for this single-vertex diagonal contact
path, but the compact cache only stored half-block writes. The cache therefore
replayed an incomplete final contact matrix.

Fix:

- Add the diagonal `write_hessian(global_vertex, H)` block to the compact
  Hessian cache during weighted graph probes.
- Keep the cache append before the scalar graph recording path. Moving it after
  scalar graph recording slightly reduced floating-point ordering perturbation
  in debug dumps, but did not eliminate it and was not kept because it offered
  no measured benefit.
- Add debug-only per-Newton runtime ordering dumps as
  `runtime_ordering.<frame>.<newton_iter>.json`, enabled by
  `SOCU_DEBUG_RUNTIME_ORDERING=1` in the wrecking-ball comparison script.

Validation with counters disabled:

| variant | final frame | wall time | mean frame | Build/Solve calls | Newton sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `socu_rt1_full_hessian` | `20` | `6.292579s` | `176.071150 ms` | `65 / 65` | `45` |
| `socu_rt1_full_hessian_cached` | `20` | `4.851380s` | `171.110829 ms` | `69 / 69` | `49` |

Frame 13-20 structured contact comparison:

- exact final structured contact:
  `0.921110s / 312 calls = 2.952 ms/call`
- cached final structured contact plus replay:
  `0.416651s / 172 calls + 0.056771s / 43 replay calls = 0.473422s`
- final structured contact plus replay duration improved by about `48.6%`.

Decision update: keep the diagonal cache fix and keep `full_hessian_cached` as
an opt-in experimental graph source. The large extra-iteration issue from the
initial cache experiment was caused by the missing diagonal records; after the
fix, 20-frame total solve/build counts are close enough for continued
benchmarking.

### Offband-Safe Contact Matrix Experiment Plan

2026-05-04: The next stability experiment will test whether the known frame-16
`contact_hessian` direction-validation failure is caused by partially dropping
off-band contact Hessian blocks. The current default remains unchanged: exact
contact Hessians are assembled into the SOCU structured matrix and off-band
entries are dropped according to the existing band policy. New policies must be
opt-in benchmark variants only.

Motivation:

- A complete contact stencil Hessian may be positive semidefinite after the
  existing contact Hessian projection, but keeping only an arbitrary subset of
  its off-diagonal blocks is not guaranteed to preserve positive
  semidefiniteness.
- A diagonal or lumped-diagonal contact contribution with nonnegative weights is
  positive semidefinite, so it should not introduce an indefinite contribution
  into the total Hessian.
- The experiment should separate ordering/probe approximation from matrix
  approximation. Changing both at once would make frame-16 failures harder to
  diagnose.

Planned experiment order:

1. Exact offband diagonalize / lump.
   Keep computing the exact contact Hessian. If every half-block in a contact
   stencil can be written into the current SOCU band, write the exact stencil as
   before. If any half-block is off-band, do not write a partial off-diagonal
   stencil. Instead, write a conservative vertex-diagonal contribution. Test
   two opt-in variants: `socu_rt1_full_hessian_diag` keeps only the exact
   per-vertex diagonal block, while `socu_rt1_full_hessian_diag_lump` writes a
   nonnegative row-sum / `abs_sum(H)` lump to scalar diagonal entries.
2. Approximate diagonal matrix mode.
   Add opt-in variants such as `socu_rt1_full_weight_approx_diag` and
   `socu_rt1_contact_weight_approx_diag`. Final structured contact assembly
   does not compute the exact contact Hessian. Normal contact writes
   `abs(kappa * dt * dt) * scale * I3`; frictional contact writes
   `abs(kappa * mu * dt * dt) * scale * I3`. Each stencil writes only
   per-vertex diagonal blocks, no off-diagonal coupling. This path is expected
   to be cheap enough that a compact Hessian cache is not useful initially.
3. Hybrid exact-band else approximate diagonal.
   Cheaply classify the stencil under the current ordering before computing a
   Hessian. If all half-blocks are in band, write exact contact Hessian values.
   If any half-block is off-band, skip exact off-diagonal coupling and write
   the approximate diagonal contribution. This is more faithful than always
   approximate, and safer than partial off-band dropping.
4. Approximate cache.
   Keep this as the lowest-priority follow-up. Approximate diagonal assembly is
   expected to be dominated by contact traversal and classification rather than
   Hessian math. Only add a cache if profiling shows repeated approximate
   traversal/classification is still a measurable bottleneck.

Validation requirements for each variant:

- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s`.
- `SOCU_REPORT_COUNTERS=0 ... --variant fused_pcg --frames 20` must still
  complete `final_frame=20`.
- The new variant must run `--frames 20`; the key question is whether it gets
  past `>>> Begin Frame: 16` without the known direction-validation NaN.
- If the 20-frame run passes, run `--frames 100` and record final frame, Newton
  sum, Build/Solve counts, line-search behavior, and whether the diagonal mode
  appears stable but too soft.
- Dump frame 13 and/or frame 16 structured matrices for representative runs and
  compare symmetry, off-band / near-band counters, minimum diagonal, direction
  validation status, and, where practical, sparse factorization or minimum
  eigenvalue estimates.
- With `SOCU_REPORT_COUNTERS=0`, compare total wall time, Build Linear System,
  Assemble Structured Chain, Assemble Structured DyTopo Hessian, Assemble
  Contact, Replay Structured Contact Hessian Cache, and Solve Linear System
  against both `socu_rt1_full_hessian_cached` and `fused_pcg`.

Acceptance policy:

- Keep each mode opt-in until it proves both stable and useful.
- The first implementation target is the exact `diag` / `diag_lump` pair,
  because together they answer the narrowest question: whether partial
  off-band dropping is the source of the frame-16 non-SPD / NaN behavior.
- If either exact diagonal policy passes 100 frames without a major Newton-count
  increase, proceed to the cheaper approximate diagonal matrix modes for
  performance.

### Offband Diag / Diag-Lump Prototype Retained Opt-In

2026-05-05: Implemented the opt-in matrix fallback policies
`contact_offband_policy = diag` and `diag_lump`. The prototype keeps default
`drop` unchanged, disables compact Hessian cache replay whenever the policy is
not `drop`, and adds wrecking-ball variants for both `full_hessian` and
`contact_hessian`.

Validation:

- `git diff --check` passed before testing.
- `uipc_test_sim_case_cuda_mixed_only
  "86_cuda_mixed_linear_solver_selection_smoke" -s` passed: 207 assertions.
- `SOCU_REPORT_COUNTERS=0 ... --variant fused_pcg --frames 20` passed.

20-frame results:

- `socu_rt1_full_hessian_diag` reached `final_frame=20`.
- `socu_rt1_full_hessian_diag_lump` reached `final_frame=20`.
- `socu_rt1_contact_hessian_diag` still failed at frame 16 with the known light
  direction validation failure:
  `finite=false`, `nonzero=false`, `descent=false`,
  `nonfinite_count=6888`, `p_norm=0`.
- `socu_rt1_contact_hessian_diag_lump` failed at the same frame 16 point with
  the same validation mode.

100-frame timing, counters disabled:

| variant | final frame | wall time | Newton sum | Build/Solve calls | Build Linear System | Assemble Structured Chain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fused_pcg` | `100` | `17.619s` | `375` | `475 / 475` | `3.304 ms/call` | n/a |
| `socu_rt1_full_hessian` | `100` | `35.031s` | `314` | `414 / 414` | `58.420 ms/call` | `57.835 ms/call` |
| `socu_rt1_full_hessian_cached` | `100` | `29.948s` | `312` | `412 / 412` | `44.766 ms/call` | `44.180 ms/call` |
| `socu_rt1_full_hessian_diag` | `100` | `36.013s` | `312` | `412 / 412` | `63.148 ms/call` | `62.545 ms/call` |
| `socu_rt1_full_hessian_diag_lump` | `100` | `35.132s` | `312` | `412 / 412` | `57.268 ms/call` | `56.683 ms/call` |

Decision: keep the prototype as an opt-in diagnostic / future tuning path, but
do not make either policy a default. Because default `drop` remains unchanged
and `full_hessian_cached` only replays the compact cache when the policy is
`drop`, the existing cached performance path should not be affected unless the
user explicitly selects `diag` or `diag_lump`. The full-Hessian fallback
variants did not provide a stability advantage over exact `full_hessian` and
were slower than `full_hessian_cached`. More importantly, applying the same
policies to `contact_hessian` did not fix the frame-16 direction-validation
failure. This suggests the known `contact_hessian` failure is not solved by
simply replacing partially off-band contact stencils with exact diagonal or
row-sum lumped diagonal contributions.

Next direction: investigate what is missing from the `contact_hessian` runtime
graph source relative to `full_hessian`, because `full_hessian` and
`full_hessian_cached` pass while `contact_hessian` still produces a zero /
nonfinite direction at frame 16.

### Frame-16 No-Contact SPD Check

2026-05-05: Added a debug-only structured-chain checkpoint before structured
contact assembly. When `SOCU_DEBUG_DUMP=1` is enabled, the solver now writes
both the existing final structured problem dump and a `_no_contact` dump taken
after chain / dyTopo base assembly but before contact Hessian assembly.

Run:

```bash
SOCU_DEBUG_DUMP=1 SOCU_REPORT_COUNTERS=1 \
PYTHONPATH=build/build_impl_fp64/python/src \
LD_LIBRARY_PATH=build/build_impl_fp64/python/src/uipc/_native:${LD_LIBRARY_PATH:-} \
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
python/examples/cuda_mixed_wrecking_ball_compare.py \
  --variant socu_rt1_contact_hessian \
  --frames 17 \
  --output output/examples/wrecking_ball_no_contact_dump
```

The run reached the known frame-16 Newton-1 failure and produced:

- `problem_no_contact.16.1.bin`
- `problem.16.1.bin`

CPU reference checks:

```text
problem_no_contact.16.1.bin: factor passed
problem_no_contact.16.1.bin: factor_and_solve passed, residual=2.615262e-13
problem.16.1.bin: CPU reference LLT failed
```

Conclusion: at the failing frame/Newton, the structured chain/base matrix before
contact assembly is still LLT-factorable. The non-SPD matrix appears only after
the contact structured contribution is assembled. This narrows the frame-16
`contact_hessian` failure to contact contribution / contact-induced ordering
effects rather than the pre-contact structured chain matrix.

Follow-up check for `socu_rt1_contact_hessian_diag` at the same failure point:

```text
A_structured_no_contact.16.1.mtx: diag neg=0, min=1
A_structured.16.1.mtx: diag neg=0, min=1
contact-only diagonal delta: neg=0, min=0, max=151126.5462167073
problem_no_contact.16.1.bin: CPU LLT factor passed
problem.16.1.bin: CPU reference LLT failed
```

So the `diag` off-band policy does not introduce negative diagonal entries
either. Its frame-16 failure is still a whole-matrix SPD issue after contact
assembly, not a negative-diagonal issue.

Correction: the run above did not actually enable the opt-in policy. The Python
variant wrote `contact_offband_policy` into the initial config dictionary, but
the scene config did not contain that path, so the solver still reported
`contact_offband_policy = drop`.

After creating the scene-config path explicitly, a rerun of
`socu_rt1_contact_hessian_diag --frames 17` completed frame 17. The corrected
frame-16 Newton-1 report shows:

```text
runtime_reorder.contact_offband_policy = diag
contact_offband_diag_fallback_count = 6904
structured_off_band_drop_count = 0
dropped_hessian_contribution_count = 0
contribution_off_band_ratio = 0
problem_no_contact.16.1.bin: CPU LLT factor passed
problem.16.1.bin: CPU LLT factor passed
```

Corrected conclusion: the user's SPD argument was right. Once `diag` is
actually enabled, stencils that would otherwise partially drop off-band contact
couplings are replaced by exact diagonal blocks, and the frame-16
`contact_hessian` LLT failure disappears at least through the 17-frame debug
run. The remaining work is to rerun the 20/100-frame timing and stability tests
with the fixed variant configuration.

### Corrected Offband Policy Sweep

2026-05-05: Invalidated the earlier offband-policy timing section because
`diag` / `diag_lump` were not actually enabled in the scene config. The Python
benchmark now explicitly creates
`linear_system/socu_approx/contact_offband_policy` on the post-`Scene(config)`
config object, so the solver receives the intended policy.

Retest setup:

- No `SOCU_DEBUG_DUMP`.
- `SOCU_REPORT_COUNTERS=0`.
- Graph sources: `topology`, `contact_hessian`.
- Runtime intervals: init-only (`0`), `1`, `2`, `5`, `10`.
- Off-band policies: `diag`, `diag_lump`.
- Output:
  - `output/examples/wrecking_ball_offband_policy_sweep_20f`
  - `output/examples/wrecking_ball_offband_policy_sweep_100f`

20-frame sweep: all 20 combinations reached `final_frame=20`; no NaN was
observed, so no debug dump rerun was needed.

100-frame sweep:

| graph | interval | policy | final | wall time | Newton | Build ms/call | Chain ms/call | Contact ms/call |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| topology | init | diag | 100 | 18.210s | 465 | 17.062 | 16.513 | 1.963 |
| topology | init | diag_lump | 100 | 15.399s | 400 | 16.036 | 15.479 | 1.863 |
| topology | 1 | diag | 100 | 27.746s | 507 | 33.239 | 32.691 | 2.134 |
| topology | 1 | diag_lump | 100 | 20.881s | 436 | 27.116 | 26.561 | 1.549 |
| topology | 2 | diag | 100 | 23.957s | 479 | 28.767 | 28.215 | 2.509 |
| topology | 2 | diag_lump | 100 | 18.273s | 414 | 22.488 | 21.925 | 1.873 |
| topology | 5 | diag | 100 | 22.820s | 473 | 25.527 | 24.971 | 2.682 |
| topology | 5 | diag_lump | 100 | 17.696s | 401 | 21.970 | 21.397 | 2.305 |
| topology | 10 | diag | 100 | 21.827s | 472 | 23.512 | 22.958 | 2.605 |
| topology | 10 | diag_lump | 100 | 15.442s | 400 | 16.727 | 16.164 | 1.840 |
| contact_hessian | init | diag | 100 | 17.650s | 465 | 17.252 | 16.696 | 1.986 |
| contact_hessian | init | diag_lump | 100 | 15.205s | 400 | 16.237 | 15.674 | 1.886 |
| contact_hessian | 1 | diag | 100 | 45.348s | 489 | 70.489 | 69.934 | 2.543 |
| contact_hessian | 1 | diag_lump | 100 | 39.595s | 436 | 68.395 | 67.838 | 2.224 |
| contact_hessian | 2 | diag | 100 | 34.541s | 504 | 46.764 | 46.207 | 2.762 |
| contact_hessian | 2 | diag_lump | 100 | 26.805s | 428 | 41.375 | 40.823 | 2.119 |
| contact_hessian | 5 | diag | 100 | 26.751s | 501 | 31.328 | 30.781 | 2.698 |
| contact_hessian | 5 | diag_lump | 100 | 22.755s | 440 | 29.087 | 28.534 | 2.290 |
| contact_hessian | 10 | diag | 100 | 25.639s | 513 | 28.491 | 27.936 | 2.849 |
| contact_hessian | 10 | diag_lump | 100 | 22.195s | 447 | 28.050 | 27.482 | 2.543 |

Observations:

- All corrected `diag` and `diag_lump` variants are stable through 100 frames.
- `diag_lump` is consistently faster than `diag` in this scene, largely because
  it reduces Newton/build counts.
- Runtime `contact_hessian` reordering is significantly more expensive than
  runtime `topology` reordering at the same interval. The rt1 contact-hessian
  path is the slowest tested configuration.
- Among the tested runtime-reorder configurations, the best 100-frame wall time
  is `topology + interval 10 + diag_lump` at `15.442s`, close to init-only
  `diag_lump` at `15.399s`.
- `contact_hessian + diag_lump` is stable, but slower than topology for all
  tested runtime intervals in this 100-frame wrecking-ball run.

### Extended Graph Source / Interval Sweep

2026-05-05: Extended the corrected sweep with `full_hessian` ordering and
larger runtime reorder intervals. Conditions remained:

- No `SOCU_DEBUG_DUMP`.
- `SOCU_REPORT_COUNTERS=0`.
- 100 frames.
- Policies: `diag`, `diag_lump`.
- Output:
  `output/examples/wrecking_ball_offband_policy_sweep_extended_100f`.

All extended combinations reached `final_frame=100`; no NaN was observed, so no
debug dump rerun was needed.

| graph | interval | policy | wall time | Newton | Build ms/call | Chain ms/call | Contact ms/call |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| full_hessian | init | diag | 18.893s | 465 | 17.109 | 16.557 | 1.967 |
| full_hessian | init | diag_lump | 15.041s | 400 | 16.112 | 15.541 | 1.869 |
| full_hessian | 1 | diag | 33.231s | 410 | 58.639 | 58.075 | 1.906 |
| full_hessian | 1 | diag_lump | 33.169s | 410 | 58.640 | 58.074 | 1.909 |
| full_hessian | 2 | diag | 31.120s | 492 | 40.457 | 39.904 | 2.525 |
| full_hessian | 2 | diag_lump | 26.090s | 417 | 39.994 | 39.435 | 2.028 |
| full_hessian | 5 | diag | 33.778s | 670 | 27.985 | 27.442 | 2.651 |
| full_hessian | 5 | diag_lump | 31.238s | 621 | 27.828 | 27.281 | 2.304 |
| full_hessian | 10 | diag | 38.044s | 832 | 25.809 | 25.267 | 2.702 |
| full_hessian | 10 | diag_lump | 33.736s | 790 | 22.950 | 22.402 | 2.219 |
| full_hessian | 15 | diag | 38.318s | 895 | 22.804 | 22.937 | 2.576 |
| full_hessian | 15 | diag_lump | 34.316s | 798 | 23.839 | 23.296 | 2.464 |
| full_hessian | 20 | diag | 61.182s | 1432 | 23.855 | 23.315 | 2.640 |
| full_hessian | 20 | diag_lump | 26.268s | 611 | 22.329 | 21.786 | 2.294 |
| full_hessian | 25 | diag | 61.996s | 1415 | 24.177 | 23.643 | 2.768 |
| full_hessian | 25 | diag_lump | 34.707s | 809 | 23.852 | 23.309 | 2.462 |
| full_hessian | 50 | diag | 52.104s | 1256 | 22.595 | 22.060 | 2.603 |
| full_hessian | 50 | diag_lump | 36.321s | 910 | 20.833 | 20.285 | 2.352 |
| topology | 15 | diag | 21.602s | 472 | 23.115 | 22.561 | 2.630 |
| topology | 15 | diag_lump | 17.178s | 399 | 21.798 | 21.233 | 2.479 |
| topology | 20 | diag | 21.785s | 475 | 24.681 | 24.126 | 2.813 |
| topology | 20 | diag_lump | 14.979s | 400 | 16.112 | 15.546 | 1.813 |
| topology | 25 | diag | 22.389s | 478 | 25.548 | 24.994 | 2.909 |
| topology | 25 | diag_lump | 15.165s | 400 | 16.381 | 15.812 | 1.857 |
| topology | 50 | diag | 17.554s | 465 | 17.271 | 16.715 | 1.967 |
| topology | 50 | diag_lump | 14.941s | 400 | 16.088 | 15.525 | 1.844 |
| contact_hessian | 15 | diag | 24.127s | 485 | 28.501 | 27.949 | 2.921 |
| contact_hessian | 15 | diag_lump | 18.704s | 410 | 22.721 | 22.156 | 2.316 |
| contact_hessian | 20 | diag | 22.610s | 487 | 24.025 | 23.467 | 2.560 |
| contact_hessian | 20 | diag_lump | 20.145s | 444 | 24.230 | 23.674 | 2.524 |
| contact_hessian | 25 | diag | 22.242s | 474 | 24.562 | 24.003 | 2.470 |
| contact_hessian | 25 | diag_lump | 20.227s | 429 | 24.879 | 24.282 | 2.558 |
| contact_hessian | 50 | diag | 19.344s | 431 | 23.337 | 22.769 | 2.563 |
| contact_hessian | 50 | diag_lump | 17.920s | 416 | 21.101 | 20.520 | 2.269 |

Updated observations:

- `full_hessian` ordering is stable with both fallback policies, but runtime
  `full_hessian` reordering is not competitive in this scene. The rt1
  `full_hessian` variants spend about `58 ms/call` in structured chain
  assembly.
- Large `full_hessian` intervals can severely increase Newton count. The worst
  cases here are `full_hessian + interval 20/25 + diag`, with more than 1400
  Newton iterations over 100 frames.
- `full_hessian + init + diag_lump` is fast (`15.041s`) and stable, but it does
  not exercise runtime reordering.
- For runtime reordering, the best measured configurations are still
  topology-based with `diag_lump`: interval 50 (`14.941s`), interval 20
  (`14.979s`), and interval 25 (`15.165s`).
- `contact_hessian + diag_lump` remains stable at larger intervals, but it is
  slower than topology at every matching large interval tested here.

### Fused PCG Comparison For Corrected Sweep

Same command family and environment as the corrected 100-frame sweeps:

```bash
SOCU_REPORT_COUNTERS=0 \
PYTHONPATH=build/build_impl_fp64/python/src \
LD_LIBRARY_PATH=build/build_impl_fp64/python/src/uipc/_native:${LD_LIBRARY_PATH:-} \
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
python/examples/cuda_mixed_wrecking_ball_compare.py \
  --variant fused_pcg \
  --frames 100 \
  --output output/examples/wrecking_ball_offband_policy_sweep_extended_100f
```

`fused_pcg` reached `final_frame=100` with:

```text
wall_time = 17.035s
Newton count = 470
Build Linear System = 3.114 ms/call
Assemble Contact = 1.316 ms/call
Solve Linear System = 7.964 ms/call
```

Fastest corrected SOCU variants versus this fused-PCG baseline:

| variant | wall time | vs fused_pcg | Newton | Build ms/call | Contact ms/call | Solve ms/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `socu_rt50_topology_diag_lump` | 14.941s | -12.3% | 400 | 16.088 | 1.844 | 2.262 |
| `socu_rt20_topology_diag_lump` | 14.979s | -12.1% | 400 | 16.112 | 1.813 | 2.259 |
| `socu_init_full_hessian_diag_lump` | 15.041s | -11.7% | 400 | 16.112 | 1.869 | 2.330 |
| `socu_rt25_topology_diag_lump` | 15.165s | -11.0% | 400 | 16.381 | 1.857 | 2.275 |
| `socu_init_contact_hessian_diag_lump` | 15.205s | -10.7% | 400 | 16.237 | 1.886 | 2.365 |

Takeaway: the best corrected SOCU configurations are about 10-12% faster than
fused PCG on this 100-frame wrecking-ball run, mainly because SOCU has fewer
Newton/build calls and a much cheaper solve stage. Fused PCG still assembles
the system much faster per build; SOCU only wins when the solve-stage saving
and reduced Newton count overcome the heavier structured build.

## 2026-05-05 Milestone 0 Baseline Freeze

Milestone 0 was rerun before starting the backend split. No implementation
files were changed for this milestone; the only active code changes are the
benchmark/debug script corrections and plan/journal updates.

Static and smoke:

```bash
git diff --check -- \
  docs/development/backend_cuda/socu_mixed_solver_integration_plan.md \
  python/examples/cuda_mixed_wrecking_ball_compare.py \
  scripts/debug_socu_repro.py

build/build_impl_fp64/RelWithDebInfo/bin/uipc_test_sim_case_cuda_mixed_only \
  "86_cuda_mixed_linear_solver_selection_smoke" -s
```

Result: `git diff --check` passed. The linear solver selection smoke passed
with `207 assertions in 1 test case`.

100-frame baseline command template:

```bash
SOCU_REPORT_COUNTERS=0 \
PYTHONPATH=build/build_impl_fp64/python/src \
LD_LIBRARY_PATH=build/build_impl_fp64/python/src/uipc/_native:${LD_LIBRARY_PATH} \
apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
python/examples/cuda_mixed_wrecking_ball_compare.py \
  --variant <variant> \
  --frames 100 \
  --output output/examples/cuda_mixed_wrecking_ball_compare_milestone0
```

Recorded outputs:

```text
output/examples/cuda_mixed_wrecking_ball_compare_milestone0/fused_pcg/result.json
output/examples/cuda_mixed_wrecking_ball_compare_milestone0/socu_rt50_topology_diag_lump/result.json
output/examples/cuda_mixed_wrecking_ball_compare_milestone0/logs/socu_rt50_topology_diag_lump_100.log
```

Results:

| variant | final frame | wall time | mean frame |
| --- | ---: | ---: | ---: |
| `fused_pcg` | 100 | 18.806s | 162.696 ms |
| `socu_rt50_topology_diag_lump` | 100 | 16.459s | 143.558 ms |

Acceptance: both `fused_pcg` and the best current topology `diag_lump`
candidate reached frame 100. This freezes the current baseline before any
Milestone 1 backend split work.

## 2026-05-06 Milestone 1 Backend Split Scaffold

Milestone 1 was implemented with the full-copy split plan:

- Copied the current `src/backends/cuda_mixed` tree to
  `src/backends/cuda_mixed_socu`.
- Added the `cuda_mixed_socu` backend target behind
  `UIPC_WITH_CUDA_MIXED_SOCU_BACKEND`.
- Left the original `src/backends/cuda_mixed` source tree unchanged.
- Excluded `linear_system/linear_fused_pcg.cu` from the copied SOCU backend.
- Added a SOCU-only init guard so `cuda_mixed_socu` rejects any solver other
  than `socu_approx`.
- Added `--backend cuda_mixed|cuda_mixed_socu` support to the wrecking-ball
  comparison script.

Build was performed without `NVCC_APPEND_FLAGS` or `--Ofast-compile=max`:

```bash
unset NVCC_APPEND_FLAGS
cmake -S . -B build/build_impl_fp64 \
  -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DUIPC_CUDA_ARCHITECTURES=120-real \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_WITH_CUDA_MIXED_SOCU_BACKEND=ON
ninja -C build/build_impl_fp64 -j1 libuipc_backend_cuda_mixed.so
ninja -C build/build_impl_fp64 -j1 libuipc_backend_cuda_mixed_socu.so
ninja -C build/build_impl_fp64 -j1 uipc_test_sim_case_cuda_mixed_only
```

Static/config checks:

```text
git diff --check: passed
python3 -m py_compile python/examples/cuda_mixed_runtime.py \
  python/examples/cuda_mixed_wrecking_ball_compare.py: passed
CMake target generation: cuda_mixed_socu and
  libuipc_backend_cuda_mixed_socu.so present
Ninja command scan: cuda_mixed_socu contains UIPC_CUDA_MIXED_SOCU_ONLY=1 and
  does not compile linear_fused_pcg
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_sim_case_cuda_mixed_only "86_cuda_mixed_linear_solver_selection_smoke" -s` | passed, 207 assertions |
| `cuda_mixed fused_pcg --frames 20` | `final_frame=20`, `wall_time=4.174s`, `mean_frame=94.260 ms` |
| `cuda_mixed_socu socu_init_topology_diag_lump --frames 20` | `final_frame=20`, `wall_time=3.918s`, `mean_frame=75.999 ms` |
| `cuda_mixed_socu socu_rt20_topology_diag_lump --frames 20` | `final_frame=20`, `wall_time=3.448s`, `mean_frame=72.316 ms` |
| `cuda_mixed_socu fused_pcg --frames 1` | rejected during init with `cuda_mixed_socu only supports linear_system/solver='socu_approx', got 'fused_pcg'` |

Acceptance: Milestone 1 passes. The copied SOCU backend loads and runs the
current structured SOCU baseline, while the original `cuda_mixed` backend still
runs fused PCG. Milestone 2 can now restore the original `cuda_mixed` directory
to the recorded pre-SOCU baseline without removing the SOCU implementation.

## 2026-05-06 Milestone 2 cuda_mixed Restore

Milestone 2 restored the original `src/backends/cuda_mixed` directory to the
pre-SOCU source baseline:

```text
bca61ac6 Add cuda_mixed linear solver abstraction
```

The copied `src/backends/cuda_mixed_socu` tree remains the SOCU implementation
surface. `socu_native` CMake integration is now gated by
`UIPC_WITH_CUDA_MIXED_SOCU_BACKEND`, so the restored `cuda_mixed` backend no
longer links or configures SOCU native code. The mixed backend policy/sim tests
were restored to their pre-SOCU shape, and SOCU-specific backend contract tests
were moved under `apps/tests/backends/cuda_mixed_socu`.

Build handoff was performed without `NVCC_APPEND_FLAGS` or
`--Ofast-compile=max`:

```bash
unset NVCC_APPEND_FLAGS
cmake -S . -B build/build_impl_fp64 \
  -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DUIPC_CUDA_ARCHITECTURES=120-real \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_WITH_CUDA_MIXED_SOCU_BACKEND=ON
ninja -C build/build_impl_fp64 -j1 \
  libuipc_backend_cuda_mixed.so \
  libuipc_backend_cuda_mixed_socu.so \
  uipc_test_backend_cuda_mixed \
  uipc_test_backend_cuda_mixed_socu \
  uipc_test_sim_case_cuda_mixed_only
```

Static/config checks:

```text
git diff --check: passed
python3 -m py_compile python/examples/cuda_mixed_wrecking_ball_compare.py: passed
SOCU reference scan over src/backends/cuda_mixed,
  apps/tests/backends/cuda_mixed, and
  apps/tests/sim_case/86_cuda_mixed_precision_contracts.cpp: no matches
matrix_converter.inl / fast_segmental_reduce.inl: no diff
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_backend_cuda_mixed "[cuda_mixed][contract]" -s` | passed, 1 assertion |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -s` | passed, 102 assertions |
| `uipc_test_sim_case_cuda_mixed_only "86_cuda_mixed_linear_solver_selection_smoke" -s` | passed, 9 assertions |
| `--variant fused_pcg --frames 20 --backend auto` | `backend=cuda_mixed`, `final_frame=20`, `wall_time=4.820s`, `mean_frame=112.914 ms` |
| `--variant socu_init_topology_diag_lump --frames 20 --backend auto` | `backend=cuda_mixed_socu`, `final_frame=20`, `wall_time=3.624s`, `mean_frame=71.318 ms` |
| `--variant socu_rt20_topology_diag_lump --frames 20 --backend auto` | `backend=cuda_mixed_socu`, `final_frame=20`, `wall_time=3.599s`, `mean_frame=77.860 ms` |
| `--variant fused_pcg --frames 1 --backend cuda_mixed_socu` | rejected during init with `cuda_mixed_socu only supports linear_system/solver='socu_approx', got 'fused_pcg'` |

Acceptance: Milestone 2 passes. The original `cuda_mixed` tree is again a
FullSparse/fused-PCG backend with no effective SOCU source or test dependency,
while the copied `cuda_mixed_socu` backend keeps the current SOCU solver and
continues to run the selected 20-frame baselines.

## 2026-05-06 Milestone 3 SOCU-Native Storage Skeleton

Milestone 3 added an isolated SOCU-native matrix storage builder under the
copied SOCU backend only:

```text
src/backends/cuda_mixed_socu/linear_system/socu_native_matrix_builder.h
apps/tests/backends/cuda_mixed_socu/socu_native_matrix_builder.cu
```

The builder owns native `D`, `E`, `rhs`, and block metadata buffers, computes
the same recursive off-diagonal layout shape expected by `socu_native`, exposes
a device write view for scalar and 3x3 writes, and provides a host snapshot
helper for debug/unit tests. It is not wired into the default structured SOCU
solver path yet.

Build handoff was performed by the user without `NVCC_APPEND_FLAGS` or
`--Ofast-compile=max`:

```bash
unset NVCC_APPEND_FLAGS
ninja -C build/build_impl_fp64 -j1 libuipc_backend_cuda_mixed_socu.so
ninja -C build/build_impl_fp64 -j1 uipc_test_backend_cuda_mixed_socu
```

Static checks:

```text
git diff --check: passed
SocuNativeMatrixBuilder reference scan over src/backends/cuda_mixed and
  apps/tests/backends/cuda_mixed: no matches
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_builder]" -s` | passed, 226 assertions |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -s` | passed, 328 assertions |
| `socu_rt20_topology_diag_lump --frames 20 --backend auto` | `backend=cuda_mixed_socu`, `final_frame=20`, `wall_time=4.182s`, `mean_frame=75.353 ms` |

For the Python wrecking-ball check, the system `python3` and repository
`.venv` did not have `numpy`/`matplotlib`. The check was run with
`uv run --no-project --with numpy --with matplotlib` plus the existing build
`PYTHONPATH`/`LD_LIBRARY_PATH`, avoiding an editable `pyuipc` rebuild.

Acceptance: Milestone 3 passes. The native storage skeleton can be written from
device kernels, downloaded for inspection, and consumed by `socu_native`
`NativeProof` on a synthetic SPD diagonal system. The current structured SOCU
runtime remains the default path and still passes the 20-frame SOCU sanity case.

## 2026-05-06 M0-M3 Closure and SOCU-Only PCG Cleanup

This closure pass tightened the split-backend state after M2/M3:

- `cuda_mixed` remains the restored FullSparse/fused-PCG baseline.
- `cuda_mixed_socu` now physically removes PCG/fused-PCG solver exposure:
  `linear_pcg`, `linear_fused_pcg`, their shared `iterative_solver`, and the
  old SpMV helper were removed from the SOCU backend.
- The temporary CMake source filter for `linear_fused_pcg.cu` was removed
  because the file no longer exists in the SOCU backend.
- `GlobalLinearSystem` in `cuda_mixed_socu` no longer carries the private
  PCG-only SpMV / preconditioner-apply / residual-accuracy bridge.
- The SOCU-side mixed precision docs/contracts now describe `PcgAuxScalar` as
  a legacy preconditioner type name and no longer list PCG/fused-PCG solver
  files as SOCU backend components.

Build handoff was performed by the user without `NVCC_APPEND_FLAGS` or
`--Ofast-compile=max`:

```bash
unset NVCC_APPEND_FLAGS
ninja -C build/build_impl_fp64 -j1 \
  libuipc_backend_cuda_mixed.so \
  libuipc_backend_cuda_mixed_socu.so \
  uipc_test_backend_cuda_mixed \
  uipc_test_backend_cuda_mixed_socu \
  uipc_test_sim_case_cuda_mixed_only
```

Static checks:

```text
git diff --check: passed
python3 -m py_compile python/examples/cuda_mixed_wrecking_ball_compare.py: passed
SOCU reference scan over src/backends/cuda_mixed,
  apps/tests/backends/cuda_mixed, and
  apps/tests/sim_case/86_cuda_mixed_precision_contracts.cpp: no matches
PCG/fused-PCG source scan over src/backends/cuda_mixed_socu and
  apps/tests/backends/cuda_mixed_socu: no LinearFusedPCG, linear_fused_pcg,
  LinearPCG, linear_pcg, IterativeSolver, iterative_solver, Spmv, or
  linear_system/spmv matches
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_backend_cuda_mixed "[cuda_mixed][contract]" -s` | passed, 1 assertion |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -s` | passed, 328 assertions |
| `uipc_test_sim_case_cuda_mixed_only "86_cuda_mixed_linear_solver_selection_smoke" -s` | passed, 9 assertions |
| `--variant fused_pcg --frames 1 --backend cuda_mixed_socu` | rejected during init with `cuda_mixed_socu only supports linear_system/solver='socu_approx', got 'fused_pcg'` |

100-frame closure commands:

```bash
LD_LIBRARY_PATH=build/build_impl_fp64/RelWithDebInfo/bin:build/build_impl_fp64/python/src/uipc/_native:$LD_LIBRARY_PATH \
PYTHONPATH=build/build_impl_fp64/python/src \
SOCU_REPORT_COUNTERS=0 \
uv run --no-project --with numpy --with matplotlib python \
  python/examples/cuda_mixed_wrecking_ball_compare.py \
  --variant fused_pcg --frames 100 --backend cuda_mixed \
  --output output/examples/cuda_mixed_wrecking_ball_compare_m2_m3_closure

LD_LIBRARY_PATH=build/build_impl_fp64/RelWithDebInfo/bin:build/build_impl_fp64/python/src/uipc/_native:$LD_LIBRARY_PATH \
PYTHONPATH=build/build_impl_fp64/python/src \
SOCU_REPORT_COUNTERS=0 \
uv run --no-project --with numpy --with matplotlib python \
  python/examples/cuda_mixed_wrecking_ball_compare.py \
  --variant socu_rt50_topology_diag_lump --frames 100 --backend cuda_mixed_socu \
  --output output/examples/cuda_mixed_wrecking_ball_compare_m2_m3_closure
```

100-frame results:

| variant/backend | final frame | wall time | mean frame | result path |
| --- | ---: | ---: | ---: | --- |
| `fused_pcg` / `cuda_mixed` | 100 | 19.805 s | 174.166 ms | `output/examples/cuda_mixed_wrecking_ball_compare_m2_m3_closure/fused_pcg/result.json` |
| `fused_pcg` / `cuda_mixed` rerun | 100 | 19.564 s | 173.001 ms | `output/examples/cuda_mixed_wrecking_ball_compare_m2_m3_closure_rerun/fused_pcg/result.json` |
| `socu_rt50_topology_diag_lump` / `cuda_mixed_socu` | 100 | 18.378 s | 163.263 ms | `output/examples/cuda_mixed_wrecking_ball_compare_m2_m3_closure/cuda_mixed_socu/socu_rt50_topology_diag_lump/result.json` |

M0 frozen fused-PCG baseline was `final_frame=100`, `wall_time=18.806 s`,
and `mean_frame=162.696 ms`. The post-cleanup fused-PCG runs still reach frame
100, but both measured wall time and mean frame are slower than the M0 timing
by more than the 2% performance gate. This cleanup did not modify
`src/backends/cuda_mixed` code, and the slowdown reproduced across two
post-cleanup runs, so it is recorded as a baseline timing drift / environment
follow-up rather than a SOCU-backend code regression. The correctness gate for
the restored `cuda_mixed` baseline remains satisfied; the timing gate should be
rechecked before using these numbers as a performance baseline for later
milestones.

Acceptance: the SOCU-only backend no longer exposes PCG/fused-PCG
implementation code, `cuda_mixed_socu` rejects `fused_pcg` clearly at init,
both backend contract suites pass, and both 100-frame closure simulations reach
frame 100.

## 2026-05-06 Milestone 3 Native Storage Refinement

This pass extended the M3 native matrix storage skeleton without migrating any
production structured assembly path:

- Added a host helper for mapping recursive off-diagonal level/block pairs to
  the flat SOCU-native offdiag storage block index.
- Added generic device writes for recursive offdiag scalar and 3x3 row-major
  block updates.
- Carried `ordering_epoch` through `SocuNativeBlockMeta` so future descriptor
  rebuild tests can verify epoch propagation.
- Added a layout contract test comparing the local storage descriptor against
  `socu_native::describe_problem_layout()` for several horizon, block-size, and
  rhs combinations.

Build handoff was performed by the user without `NVCC_APPEND_FLAGS` or
`--Ofast-compile=max`.

Static checks:

```text
git diff --check: passed
SocuNativeMatrixBuilder reference scan over src/backends/cuda_mixed and
  apps/tests/backends/cuda_mixed: no matches
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_builder]" -r compact` | passed, 719 assertions in 4 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` | passed, 821 assertions in 9 cases |

Acceptance: M3 remains a storage/test skeleton only. The generic recursive
offdiag write path is covered by device-write tests, the local descriptor is
checked against SOCU native's layout description, and no `cuda_mixed` baseline
source was modified.

## 2026-05-06 Milestone 4 Descriptor Infrastructure, First Slice

This pass started M4 with reusable host-side SOCU descriptor infrastructure
without migrating any production provider to the native matrix builder:

- Added `SocuNativeDofDescriptor` and `SocuNativeVertexDescriptor` helpers for
  old-DoF, block/lane, FEM/ABD kind, fixed state, ABD body/J index, and epoch
  metadata.
- Added band classification helpers for scalar DoF pairs, vertex half-blocks,
  and stencil half-blocks. The classification mirrors the current structured
  sink band model: same block is diagonal, adjacent blocks are first offdiag,
  and longer distances are off-band.
- Added descriptor table epoch helpers and unit tests for reorder epoch changes.
- `SocuApproxSolver` now rebuilds the old-DoF descriptor table after each
  successful init-time or runtime ordering install, increments
  `descriptor_epoch`, and writes that epoch into the solve report JSON. The
  current structured assembly path does not consume the descriptor table yet.

Build handoff was performed by the user without `NVCC_APPEND_FLAGS` or
`--Ofast-compile=max`:

```bash
unset NVCC_APPEND_FLAGS
ninja -C build/build_impl_fp64 -j1 \
  libuipc_backend_cuda_mixed_socu.so \
  uipc_test_backend_cuda_mixed_socu
```

Static checks:

```text
git diff --check: passed
matrix_converter.inl / fast_segmental_reduce.inl diff scan: no changes
SocuNativeDescriptor reference scan over src/backends/cuda_mixed and
  apps/tests/backends/cuda_mixed: no matches
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_descriptor]" -r compact` | passed, 70 assertions in 4 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` | passed, 891 assertions in 13 cases |
| `socu_rt50_topology_diag_lump --frames 20 --backend cuda_mixed_socu` | `final_frame=20`, `wall_time=4.049s`, `mean_frame=73.139 ms` |

Acceptance: M4 is still infrastructure-only. Descriptor construction and
classification are covered for FEM/FEM, FEM/ABD, ABD/FEM, ABD/ABD, fixed
vertices, off-band pairs, and reorder epochs. The default structured SOCU path
continues to use the existing structured sink and still passes the 20-frame
sanity run.

## 2026-05-07 Milestone 4 Descriptor Device Rebuild And Link Closure

This pass closed the remaining M4 descriptor/device-build issues and kept the
native builder work isolated from legacy structured contact assembly:

- Added `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY`. In this mode the SOCU backend
  excludes legacy structured contact model translation units matching
  `contact_system/contact_models/*_structured.cu`, while the non-structured
  contact model callers compile with `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=1` so
  accidental fallback calls fail clearly instead of silently using removed code.
- Forced shared CUDA runtime linkage for the SOCU backend and backend test
  target with `CUDA_RUNTIME_LIBRARY Shared`.
- Fixed a CUDA DSO registration/link interaction by changing the backend's
  `socu_native` link dependency from `PUBLIC` to `PRIVATE`. The test target now
  inherits only `socu_native` interface include directories and compile
  definitions; it no longer pulls `external/socu-native-cuda/libsocu_native.a`
  into the test executable's CUDA device-link step. The previous duplicate
  static CUDA archive linkage was the root cause of the
  `cudaErrorInvalidResourceHandle` seen when launching the backend DSO's
  descriptor rebuild kernel from the directly linked test executable.
- Reworked the `socu_native` synthetic smoke test to use a local deterministic
  SPD block-tridiagonal fixture and local residual check. This preserves the
  solver contract test without requiring the test executable to link the
  `socu_native` host-side problem generator symbols directly.

Build-artifact checks:

```text
UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON configure:
  cuda_mixed_socu native-only build: 468 -> 464 source entries

build.ninja scan for
  src/backends/cuda_mixed_socu/CMakeFiles/cuda_mixed_socu.dir/contact_system/contact_models/.*_structured.cu.o:
  no matches

structured_contact_hessian_cache.cu.o unresolved registration:
  U __cudaRegisterLinkedBinary_6dcc94e7_35_structured_contact_hessian_cache_cu_82f2d525_2706635

cuda_mixed_socu cmake_device_link.o registration:
  T __cudaRegisterLinkedBinary_6dcc94e7_35_structured_contact_hessian_cache_cu_82f2d525_2706635

test executable CUDA device-link/final-link:
  no external/socu-native-cuda/libsocu_native.a

git diff --check: passed
```

Functional results:

| check | result |
| --- | --- |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_descriptor]" -s` | passed, 124 assertions in 6 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_builder]"` | passed, 840 assertions in 5 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native][m6]"` | passed, 84 assertions in 1 case |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 1066 assertions in 16 cases |

Acceptance: M4 descriptor infrastructure is now considered complete for the
planned infrastructure scope. Host descriptor construction, pair/stencil
classification, reorder epoch behavior, device descriptor rebuild, comparison
against the current structured sink classification, and native storage/solver
contracts are covered by always-on backend contract tests. The production
structured SOCU path remains unchanged and still does not consume the native
descriptor table for matrix assembly. Provider migration, contact stencil
descriptors, native graph/reorder collection, and native contact writes remain
future milestones rather than hidden M4 requirements.

## 2026-05-07 Milestone 5 Native Diagonal/RHS Infrastructure Slice

This pass starts M5 with the smallest provider-write surface that can be
validated without switching production assembly to the native builder:

- Made `SocuNativeVertexDescriptor::mapped()` and `writable()` callable from
  device code.
- Added descriptor-aware write helpers to `SocuNativeMatrixView`:
  - active DoF descriptor -> scalar diagonal entry.
  - active DoF descriptor -> packed RHS entry.
  - writable vertex descriptor -> scalar diagonal entry.
  - writable vertex descriptor -> row-major dense diagonal block.
  - writable vertex descriptor -> scalar/vector RHS entries.
- Fixed, unmapped, inactive, and padding descriptors are skipped by the helper
  layer before touching native storage.
- Added an M5 provider-style CUDA fixture that writes the same deterministic
  diagonal block through `SocuNativeMatrixView` and the current
  `StructuredDeviceMatrixSink`, then compares downloaded `D/E`. RHS packing is
  compared against an explicit expected packed vector because the current
  structured matrix sink does not own gradient/RHS packing.

Functional results:

| check | result |
| --- | --- |
| `ninja -C build/build_impl_fp64 -j4 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m5]" -s` | passed, 76 assertions in 1 case |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_builder]"` | passed, 916 assertions in 6 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 1142 assertions in 17 cases |
| `git diff --check` | passed |

Acceptance status: this is an infrastructure slice, not the full M5 provider
migration. It proves the native builder can accept descriptor-driven
diagonal/RHS writes and match the current structured sink for diagonal matrix
storage, while preserving all existing SOCU contracts. The production structured
SOCU assembly path remains unchanged and disabled from the new helpers by
default. Full M5 still needs actual mass/inertia/regularization provider
migration, a runtime dual-assembly diff option for `D/E/rhs`, and the 20-frame
and 100-frame native-enabled scene gates.

## 2026-05-07 Milestone 5 Native Diagonal/RHS Production Path

This pass completes M5 under the refined provider boundary now documented in
the plan: native initialization covers solver-owned diagonal workspace writes
(`damping_shift`/regularization and padding identity) plus packed RHS writes.
FEM/ABD kinetic, inertia, and shape Hessian assembly remain in the existing
structured chain/base provider path and are deferred to M6, where the provider
API itself moves to native writes.

Implementation notes:

- Added default scene config keys
  `linear_system/socu_approx/native_diag_rhs` and
  `linear_system/socu_approx/debug_compare_native_diag_rhs`. Without default
  schema entries, Python-side config assignments are ignored before they reach
  `SocuApproxSolver`.
- Added `SOCU_NATIVE_DIAG_RHS=1` and `SOCU_NATIVE_DIAG_RHS_DIFF=1` hooks to
  `python/examples/cuda_mixed_wrecking_ball_compare.py`.
- Added `initialize_socu_native_diag_rhs_workspace(...)` and
  `compare_socu_native_diag_rhs_workspace(...)`. The production native path
  clears `D/E/rhs`, writes damping/padding through `SocuNativeMatrixView`, packs
  RHS from `SocuNativeDofDescriptor`, and copies `rhs_original`.
- Added runtime compare buffers and solver report fields for native diag/RHS
  parity. In diff mode, the solver assembles both legacy and native init paths
  and throws if any `D/E/rhs` mismatch exceeds tolerance.
- Kept native diag/RHS opt-in and default-off. The legacy structured path is
  still the fallback and comparison source.

Build note:

For full scene acceptance the build was reconfigured with
`UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF`, because the current contact fallback
still requires legacy structured contact TUs. The full fallback target rebuilt
successfully with `ninja -C build/build_impl_fp64 -j1
RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu`; the old
`ipc_simplex_frictional_contact_structured.cu` TU dominated compile time and
memory, so single-job build remains the safer path for this configuration.

Functional results:

| check | result |
| --- | --- |
| `ninja -C build/build_impl_fp64 -j4 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed after adding the scene config schema keys |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m5]" -s` | passed, 98 assertions in 2 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 1164 assertions in 18 cases |
| `SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 ... cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_topology_diag_lump --frames 20` | `final_frame=20`, report shows `native_diag_rhs_enabled=true`, `native_diag_rhs_diff_enabled=true`, `native_diag_rhs_diff_mismatch_count=0`, and all three diff abs sums are `0.0` |
| `SOCU_REPORT_COUNTERS=0 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 ... --frames 5` | passed, `final_frame=5`; verifies the diff path does not depend on debug/report counters being enabled |
| default 100-frame `socu_rt50_topology_diag_lump` | `final_frame=100`, `wall_time_s=17.216184758988675`, `mean_frame_ms=152.9752541182097` |
| native diag/RHS 100-frame `socu_rt50_topology_diag_lump` | `final_frame=100`, `wall_time_s=16.569126201968174`, `mean_frame_ms=148.57842522033025` |

Performance gate: native diag/RHS was `-2.87%` in mean frame time relative to
the default path on this run, so the 2% regression gate passes. Since this is an
opt-in path and not a default behavior change, the default solver remains
unchanged while M6 migrates the larger chain/base Hessian providers.

Acceptance: M5 is complete for the documented diagonal/RHS production scope.
The native builder is now used by an opt-in production path, has a runtime
legacy diff gate, passes provider and full backend contracts, and reaches both
20-frame parity and 100-frame performance acceptance on the chosen
`topology + diag_lump` wrecking-ball variant.

## 2026-05-07 Milestone 6 Native Chain/Base Hessian Provider

This pass migrates the always-present non-contact chain/base Hessian write
surface behind an opt-in native path while keeping the provider API stable:

- Extended `StructuredDeviceMatrixSink` with a descriptor-backed native D/E
  writer. It uses `SocuNativeDofDescriptor` block/lane addresses and preserves
  the existing diagonal and first-offdiagonal storage orientation.
- Added a mirror compare mode to the same sink. A primary native write can be
  mirrored into legacy structured buffers, or a primary legacy write can be
  mirrored into native buffers, without recording runtime ordering twice.
- Added `StructuredAssemblyPhase` to `StructuredAssemblyInfo`. FEM/ABD
  subsystem assembly runs in the `ChainBase` phase; DyTopo/contact assembly
  runs in the `Contact` phase and remains legacy structured fallback.
- Added solver config/report fields
  `linear_system/socu_approx/native_chain_base_hessian` and
  `linear_system/socu_approx/debug_compare_native_chain_base_hessian`.
- Added runtime compare buffers and a pre-contact checkpoint diff. The diff
  compares `D/E/rhs` after chain/base assembly and before contact Hessian
  assembly, so contact does not pollute the M6 parity signal.
- Added wrecking-ball environment hooks `SOCU_NATIVE_CHAIN_BASE=1`,
  `SOCU_NATIVE_CHAIN_BASE_DIFF=1`, and `SOCU_CONTACT_ENABLE=0`. The contact
  switch is only for no-contact native-only smoke validation; default scene
  behavior is unchanged.

Functional results:

| check | result |
| --- | --- |
| `git diff --check` | passed |
| direct object build for `scene_default_config.cpp.o`, `global_linear_system.cu.o`, `socu_approx_solver.cu.o`, and `socu_native_matrix_builder.cu.o` | passed |
| native-only reconfigure/build, `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; device link completed; legacy structured contact TUs excluded |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]" -s` | passed, 89 assertions in 1 case |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` in native-only build | passed, 1253 assertions in 19 cases |
| `SOCU_CONTACT_ENABLE=0 SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_CHAIN_BASE_DIFF=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 ... --frames 1` | passed; report shows native chain/base enabled, diff enabled, mismatch count `0`, and `D/E/rhs` diff sums all `0.0` |
| same no-contact native-only scene with `--frames 20` and report counters off | passed, `final_frame=20`, `wall_time_s=2.857510956004262`, `mean_frame_ms=30.889609490986913` |

Fallback/contact build note:

The contact-enabled fallback build was attempted with
`UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF` and `-j1`, but
`ipc_simplex_frictional_contact_structured.cu` reached about `55GB` RSS and
entered heavy swap during `cicc` (observed swap usage about `97GB`). The build
was interrupted to avoid destabilizing the workstation. A native-only
contact-enabled 1-frame smoke reaches the expected explicit unsupported path:
`SOCU native-only build excludes legacy vertex-half-plane frictional structured
contact assembly`.

Acceptance status: M6 implementation and no-contact native chain/base parity are
complete, including provider tests, runtime diff wiring, report fields, and
device link validation. The full contact-enabled 20-frame `topology +
diag_lump` regression and the 100-frame performance gate are not yet rigorously
closed, because they require the fallback build to finish the legacy structured
contact TU. Until that build/scene gate passes, keep
`native_chain_base_hessian` opt-in and default-off.

### M6b Fast Native Chain/Base Target Start

The parity path above is correct but still shaped like the legacy structured
sink: each scalar write classifies old DoF pairs at the moment of insertion.
The performance-oriented builder work is therefore split into M6b.

Implemented first M6b slice:

- Added a descriptor-validated ABD `12x12` fast path for
  `add_dense_block_upper_subblocks_fixed<3, 4>`.
- The fast path requires the 12 old DoFs for one ABD body to be active and
  located inside one native SOCU block. It writes directly to native `D` using
  the prevalidated per-DoF lanes and only mirrors scalar writes into the debug
  compare buffer. Lane contiguity is deliberately not required: the real RCM
  ordering commonly reverses or permutes ABD lanes within a block.
- If native storage is disabled, the body spans unsupported blocks, or runtime
  ordering collection is enabled, ABD assembly falls back to the existing scalar
  structured sink.
- The M6 plan now separates M6a parity from M6b fast targets and places native
  contact hotspot migration under M8.

Validation:

| check | result |
| --- | --- |
| native-only reconfigure/build, `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; the already-built `-j1` objects were reused after switching to `-j2`; device link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6b]"` | passed, 304 assertions in 1 case; the fixture now uses reverse lane order, the fast-path status flag was `1`, native primary/legacy mirror/legacy reference `D` matched, and counters reported hit `1`, miss `0`, fallback `0` |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]"` | passed, 393 assertions in 2 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` in native-only build | passed, 1557 assertions in 20 cases |
| `SOCU_CONTACT_ENABLE=0 SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_CHAIN_BASE_DIFF=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 SOCU_REPORT_COUNTERS=1 ... --frames 1` | passed; report timing shows native chain/base enabled, diff enabled, mismatch count `0`, all `D/E/rhs` diff sums `0.0`, and target counters hit `492`, miss `82`, scalar fallback `82` |
| same no-contact native-only scene with `--frames 20` and report counters on | passed, `final_frame=20`, `wall_time_s=2.5994132080231793`, `mean_frame_ms=30.93939629616216`; final report still shows mismatch count `0`, all diff sums `0.0`, and target counters hit `492`, miss `82`, scalar fallback `82` |
| restore CMake cache with `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF` | passed; `CMakeCache.txt` reports `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY:BOOL=OFF` |

Acceptance status: this first M6b slice is correct as a guarded ABD base Hessian
fast target and is actually exercised by the no-contact wrecking-ball scene. The
ordering probe showed 168 of 191 ABD bodies are same-block but none are
contiguous; the arbitrary-lane target is therefore the right first target. The
follow-up entry below closes the observed cross-block miss class with an
adjacent-block target. This is still not the complete high-performance native
builder: FEM block targets, general first-offdiag target precomputation, and
native contact builder work remain open under the M6b/M8 split.

### M6b ABD Adjacent-Block Target

Implemented the next ABD `12x12` fast-path slice:

- The dense ABD target now classifies the whole body once as same-block,
  adjacent-block, or miss.
- Same-block bodies keep writing directly to native `D` with arbitrary lanes.
- Adjacent-block bodies gather per-local-DoF `block/lane` descriptors, write
  same-block pieces to `D`, and write cross-boundary pieces to first-offdiag
  `E` with the same SOCU orientation as the scalar native sink.
- Counters now distinguish
  `native_chain_base_same_block_dense_hit_count`,
  `native_chain_base_adjacent_dense_hit_count`,
  `native_chain_base_dense_miss_count`, and
  `native_chain_base_scalar_fallback_count`.

Validation:

| check | result |
| --- | --- |
| `git diff --check` | passed |
| native-only incremental build, `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6b]" -s` | passed, 711 assertions in 2 cases; same-block and adjacent-block fixtures both matched legacy `D/E` and reported the expected hit/miss/fallback counters |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]"` | passed, 800 assertions in 3 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider]"` | passed, 898 assertions in 5 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_descriptor]"` | passed, 124 assertions in 6 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` in native-only build | passed, 1964 assertions in 21 cases |
| `SOCU_CONTACT_ENABLE=0 SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_CHAIN_BASE_DIFF=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 SOCU_REPORT_COUNTERS=1 ... --frames 20` | passed, `final_frame=20`, `wall_time_s=2.8978309520171024`, `mean_frame_ms=30.47051320609171`; report shows native chain/base diff mismatch `0`, all diff sums `0.0`, same-block hits `492`, adjacent hits `82`, dense misses `0`, scalar fallbacks `0` |

Acceptance status: the ABD base Hessian portion of M6b is now much tighter than
the previous same-block-only slice. The real no-contact scene exercises both
the arbitrary-lane same-block target and the adjacent-block first-offdiag
target, and the previous 82 dense misses/scalar fallbacks are gone. Remaining
M6b work is still FEM `3x3`/pair targets, general first-offdiag target
precomputation, diff-off performance timing, and keeping production native
single-write mode clean. Native contact builder work remains M8.

### M6b Assembly Timing Gate

Added a debug-timing replacement for the old coarse `Assemble Structured Chain`
signal:

- `chain_base_assembly_time_ms` measures chain/base structured provider work
  with CUDA events on the mixed backend stream.
- `native_chain_base_assembly_time_ms` mirrors that value only when native
  chain/base Hessian writes are enabled, so baseline/native reports are easy to
  separate.
- `contact_assembly_time_ms` records structured DyTopo contact assembly time
  when that phase runs. It is `0.0` in no-contact validation.
- Timing is only active when `debug_timing` is enabled. Diff-off production
  runs with `SOCU_REPORT_COUNTERS=0` do not pay the extra event synchronize.

Validation:

| check | result |
| --- | --- |
| native-only reconfigure/build, `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; device link and shared library link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 1964 assertions in 21 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]"` | passed, 800 assertions in 3 cases |
| no-contact 20-frame native timing run, native chain/base + native diag/RHS enabled, diff disabled, counters/timing enabled | passed; final report shows `chain_base_assembly_time_ms=0.6652160286903381`, `native_chain_base_assembly_time_ms=0.6652160286903381`, same-block hits `492`, adjacent hits `82`, dense misses `0`, scalar fallbacks `0` |
| no-contact 20-frame structured baseline timing run, native disabled, counters/timing enabled | passed; final report shows `chain_base_assembly_time_ms=0.8285120129585266`, `native_chain_base_assembly_time_ms=0.0` |
| no-contact 100-frame structured baseline, diff/counters disabled | passed, `final_frame=100`, `wall_time_s=3.823726774950046`, `mean_frame_ms=17.808828111737967` |
| no-contact 100-frame native diff-off, native chain/base + native diag/RHS enabled, counters disabled | passed, `final_frame=100`, `wall_time_s=3.726719599973876`, `mean_frame_ms=17.589850779622793` |

Acceptance status: the ABD fast target now has a usable performance signal.
The last-solve chain/base assembly timer improved by about `20%`
(`0.8285ms -> 0.6652ms`) on the no-contact wrecking-ball scene, while the
end-to-end 100-frame run improved by about `1.2%`. The small frame-level delta
is expected because factor/solve, Newton loop overhead, logging, and scene
pipeline work dominate this no-contact benchmark. Future M6b FEM/first-offdiag
target work should use `chain_base_assembly_time_ms` as the primary acceptance
signal and keep end-to-end frame time as a secondary sanity check.

### M6b/M6c FEM Vertex-Local 3x3 Target

Implemented the first FEM-style native target slice:

- Added a same-block dense `3x3` native writer for descriptor-backed
  `BlockDim=3` local Hessian writes.
- The target gathers the three old DoF descriptors once, accepts arbitrary lane
  order inside the native SOCU block, and writes the full `3x3` block directly
  to native `D`.
- `LocalAssemblySink<BlockDim=3>::add_structured_block()` now attempts this
  target for non-fixed same-vertex/same-block writes when native chain/base
  Hessian is enabled and runtime ordering collection is inactive.
- Unsupported cases fall back to the existing scalar structured sink and record
  scalar fallback. New report counters are
  `native_chain_base_diag3x3_hit_count` and
  `native_chain_base_diag3x3_miss_count`.

Validation:

| check | result |
| --- | --- |
| native-only reconfigure/build, `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; device link and test executable link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6c]" -s` | passed, 288 assertions in 1 case; same-block arbitrary-lane fixture reported hit `1`, miss `0`, fallback `0`, and inactive-descriptor fixture reported hit `0`, miss `1`, fallback `1`; native primary, debug compare, and scalar legacy `D` matched |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]"` | passed, 1088 assertions in 4 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 2252 assertions in 22 cases |
| no-contact 1-frame native smoke, `SOCU_CONTACT_ENABLE=0 SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_REPORT_COUNTERS=1 ... --variant socu_init --frames 1` | passed; report JSON includes the new fields and shows ABD counters unchanged: same-block `492`, adjacent `82`, dense miss `0`, scalar fallback `0`, `diag3x3_hit=0`, `diag3x3_miss=0` for the ABD-only scene |

Acceptance status: this closes the vertex-local FEM `3x3` same-block target
slice, but it is not yet a full FEM high-performance builder. FEM element pair
targets, first-offdiag precomputed targets, and a scene-level FEM native hit
rate/performance gate remain open M6b work.

### M6b/M6c FEM Pair 3x3 First-Offdiag Target

Implemented the next FEM native target slice:

- Added descriptor-backed `3x3` pair writes for non-fixed
  `LocalAssemblySink<BlockDim=3>` cross-vertex Hessian blocks.
- Same-native-block vertex pairs write the full `3x3` block directly to native
  `D` with arbitrary lane order and debug compare parity.
- Adjacent-native-block vertex pairs write directly to first-offdiag `E` using
  the same orientation as the scalar native sink, including reverse provider
  pair order.
- Off-band, inactive, fixed-policy, runtime-ordering, and native-disabled cases
  keep the scalar structured fallback. Runtime ordering intentionally disables
  the fast target so graph collection remains scalar.
- Added report counters
  `native_chain_base_pair3x3_same_block_hit_count`,
  `native_chain_base_pair3x3_adjacent_hit_count`, and
  `native_chain_base_pair3x3_miss_count`.

Validation:

| check | result |
| --- | --- |
| native-only reconfigure/build, `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; device link and test executable link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6c]" -s` | passed, 2936 assertions in 2 cases; vertex-local diag and pair fixtures matched native primary/debug compare/legacy scalar buffers |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]"` | passed, 3736 assertions in 5 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 4900 assertions in 23 cases |
| no-contact FEM tower native gate, `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_REPORT_COUNTERS=1 ... cuda_mixed_abd_fem_tower_viewer.py --backend cuda_mixed_socu --solver socu_approx --levels 6 --smoke-frames 1 --disable-contact` | passed; report shows `diag3x3_hit=84`, `pair3x3_same_block_hit=77`, `pair3x3_adjacent_hit=13`, `pair3x3_miss=0`, `scalar_fallback=0`, and `chain_base_assembly_time_ms=0.8841919898986816` |
| same no-contact FEM tower structured baseline, native chain/base disabled | passed; report shows `chain_base_assembly_time_ms=1.4704960584640503` |
| same no-contact FEM tower native mirror-diff gate with `SOCU_NATIVE_CHAIN_BASE_DIFF=1 SOCU_NATIVE_DIAG_RHS_DIFF=1` | passed; native chain/base and diag/RHS mismatch counts are `0`, and all `D/E/rhs` diff abs sums are `0.0` |

Acceptance status: the FEM pair/first-offdiag sink-side target slice is correct
and is exercised by an actual FEM scene. On the chosen `levels=6` no-contact
ABD/FEM tower gate, native chain/base assembly improved from about `1.47ms` to
about `0.884ms` for the final solve, a roughly `40%` reduction in the isolated
chain/base assembly timer. This still is not the final provider-built target
table design: the current slice classifies once per local `3x3` block inside
the sink from DoF descriptors. The next performance refinement is explicit
provider/stencil target arrays that carry `left_block`, `row_lane`, `col_lane`,
and transpose/orientation flags into the kernel hot loop.

### M6 Closure: Native Chain/Base Complete

M6 is now closed for its native chain/base scope. Contact-enabled topology
acceptance for the native contact writer remains M8 because contact Hessian
build is explicitly M8 work. The legacy structured-contact fallback path is
kept as a compatibility/reference route, but it is not the implementation scope
that closes M6.

Important build observation:

- A full `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF` rebuild was attempted again
  with `ninja -C build/build_impl_fp64 -j1
  RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu`.
- The build reached
  `contact_system/contact_models/ipc_simplex_frictional_contact_structured.cu`.
  During `cicc`, RSS climbed to about `54GB`, available memory dropped to tens
  of MB, and swap usage climbed to about `16GB`.
- The build was interrupted to avoid destabilizing the machine. This confirms
  the earlier observation that the legacy frictional structured contact TU is a
  contact-stack compilation problem, not a native chain/base correctness gap.
- On 2026-05-08, the full fallback build was later completed with
  `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF`, allowing the structured-contact
  compatibility gates below to run.

Final M6 validation:

| check | result |
| --- | --- |
| native-only reconfigure/build, `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON` then `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; native-only source exclusion `469 -> 465`, device link and test executable link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_provider][m6]"` | passed, 3736 assertions in 5 cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, 4900 assertions in 23 cases |
| no-contact 20-frame topology gate, `SOCU_CONTACT_ENABLE=0 SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_CHAIN_BASE_DIFF=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 SOCU_REPORT_COUNTERS=1 ... cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_topology_diag_lump --frames 20 --backend cuda_mixed_socu` | passed, `final_frame=20`, `mean_frame_ms=41.2303806951968`; report shows native chain/base diff mismatch `0`, all `D/E/rhs` diff sums `0.0`, same-block dense hits `492`, adjacent dense hits `82`, dense misses `0`, scalar fallback `0`, and `chain_base_assembly_time_ms=0.7747520208358765` |
| no-contact 100-frame structured baseline, native chain/base disabled, counters/diff disabled | passed, `final_frame=100`, `wall_time_s=4.650163705984596`, `mean_frame_ms=24.125422997167334` |
| no-contact 100-frame native diff-off, native diag/RHS and native chain/base enabled, counters/diff disabled | passed, `final_frame=100`, `wall_time_s=3.678715430025477`, `mean_frame_ms=18.035593961831182` |
| no-contact ABD/FEM tower native mirror-diff gate, `cuda_mixed_abd_fem_tower_viewer.py --backend cuda_mixed_socu --solver socu_approx --levels 6 --smoke-frames 1 --disable-contact` with native chain/base/diag-RHS diff enabled | passed; report shows native chain/base mismatch `0`, all diff sums `0.0`, `diag3x3_hit=84`, `pair3x3_same_block_hit=77`, `pair3x3_adjacent_hit=13`, `pair3x3_miss=0`, `scalar_fallback=0`, and `chain_base_assembly_time_ms=0.83651202917099` |

Acceptance status: M6 is complete for native chain/base Hessian/RHS assembly.
The accepted implementation provides descriptor-backed native parity, debug
mirror diff, production single-write mode when diff is disabled, ABD same and
adjacent `12x12` fast targets, FEM vertex-local and pair `3x3` targets, real
scene hit-rate gates, and no-contact performance improvement. Contact-enabled
`topology + diag_lump` native-contact acceptance is M8, where the contact target
table and native contact writer will remove the dependency on the legacy
frictional structured contact TU.

### M6 Supplemental Full Fallback Contact Compatibility

2026-05-08: After the full fallback build completed, the contact-enabled
compatibility gates were run against the `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF`
artifact. These gates exercise legacy structured contact assembly together with
the M6 native chain/base path; they do not replace the M8 native contact writer
acceptance.

Build/product checks:

- `build/build_impl_fp64/CMakeCache.txt` reports
  `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY:BOOL=OFF`.
- `build/build_impl_fp64/RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu`
  exists, and the legacy structured contact object files are present for
  simplex frictional/normal and vertex-half-plane frictional/normal contact.
- `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact`
  passed, `4900` assertions in `23` test cases.

Supplemental contact-enabled validation:

| check | result |
| --- | --- |
| 20-frame contact-enabled topology mirror-diff gate, `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_CHAIN_BASE_DIFF=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_DIAG_RHS_DIFF=1 SOCU_REPORT_COUNTERS=1 ... cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_topology_diag_lump --frames 20 --backend cuda_mixed_socu` | passed, `final_frame=20`, `wall_time_s=4.479084960010368`, `mean_frame_ms=82.0732523483457`; report shows status `structured band direction solved and scattered: provider=abd_only, scope=multi_provider`, native chain/base diff mismatch `0`, all native chain/base `D/E/rhs` diff sums `0.0`, same-block dense hits `492`, adjacent dense hits `82`, dense misses `0`, scalar fallback `0`, `chain_base_assembly_time_ms=1.0244799852371216`, and `contact_assembly_time_ms=6.956448078155518` |
| 100-frame contact-enabled structured baseline, native chain/base disabled and counters/diff disabled | passed, `final_frame=100`, `wall_time_s=17.490785808011424`, `mean_frame_ms=154.2077547806548` |
| 100-frame contact-enabled native chain/base diff-off, `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1`, counters/diff disabled | passed, `final_frame=100`, `wall_time_s=17.48229211801663`, `mean_frame_ms=155.182085702545` |
| 100-frame contact-enabled native chain/base counter gate, `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_REPORT_COUNTERS=1`, diff disabled | passed, `final_frame=100`, `wall_time_s=18.19558243697975`, `mean_frame_ms=161.3195193867432`; report shows native chain/base and native diag/RHS enabled, diff disabled, same-block dense hits `492`, adjacent dense hits `82`, dense misses `0`, scalar fallback `0`, `chain_base_assembly_time_ms=0.9599360227584839`, `native_chain_base_assembly_time_ms=0.9599360227584839`, and `contact_assembly_time_ms=24.102815628051758` |

Interpretation:

- The full fallback artifact is now usable for compatibility/regression checks.
- M6 native chain/base remains exact under contact-enabled scenes: the 20-frame
  mirror-diff gate has zero mismatch and zero diff sums, and both 20-frame and
  100-frame counter reports show zero dense miss/scalar fallback for the ABD
  chain/base targets.
- The 100-frame native/contact mean time is roughly the same as structured
  baseline in this small scene, and the counter-enabled run is slower because it
  records debug counters. This is acceptable for M6; reducing contact assembly
  time is still M8.

### M7 Deferred, M8 Native Contact Kickoff

2026-05-08: M7 is intentionally deferred until after M8. The reason is that the
current accepted scenes and fallback contact gates show contact assembly as the
next SOCU integration hotspot, while constraints/joints/external forces can stay
on structured fallback without blocking the topology/contact acceptance path.

M8 first slice implemented:

- Added `src/backends/cuda_mixed_socu/linear_system/socu_native_contact_targets.h`.
- Introduced `SocuNativeContactStencilTarget`,
  `SocuNativeContactStencilPolicy`, and `SocuNativeContactWriteMode`.
- Contact target classification uses per-DoF descriptors directly rather than
  `SocuNativeVertexDescriptor::active`, so arbitrary-lane vertices remain
  eligible even when the old contiguous-range descriptor view would mark them
  inactive. This preserves the M6 lesson from real RCM ordering.
- Added policy helpers for exact in-band writes, `drop` off-band classification,
  and whole-stencil `diag`/`diag_lump` fallback selection.
- Added `apps/tests/backends/cuda_mixed_socu/socu_native_contact_targets.cu`
  covering arbitrary-lane exact targets, adjacent first-offdiag orientation,
  whole-stencil `diag_lump` fallback, drop/off-band classification,
  skipped/fixed vertices, and ABD metadata.

Validation:

| check | result |
| --- | --- |
| `cmake -S . -B build/build_impl_fp64` | passed; new test source picked up by the globbed test target |
| `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; compiled `socu_native_contact_targets.cu` and relinked the test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact][m8]" -s` | passed, `53` assertions in `4` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` | passed, `4953` assertions in `27` test cases |

M8 second slice implemented:

- Added
  `src/backends/cuda_mixed_socu/linear_system/socu_native_contact_targets.cu`.
- Added `rebuild_socu_native_simplex_contact_targets(...)`, a narrow CUDA
  rebuild entry point for PT/EE/PE/PP target arrays.
- The device rebuild emits one `SocuNativeContactStencilTarget` per upper
  half-block, in the same local order used by
  `StructuredContactAssemblySink::write_hessian_half`.
- Device classification uses current `old_to_chain` directly, so arbitrary-lane
  FEM/ABD descriptors remain eligible even when the contiguous vertex descriptor
  flag is false.
- Whole-stencil `diag`/`diag_lump` policy is selected before exact writes; with
  `drop`, only off-band half-blocks are marked `DropOffBand`, preserving the old
  scalar drop behavior.

Second-slice validation:

| check | result |
| --- | --- |
| `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; built `socu_native_contact_targets.cu`, relinked `libuipc_backend_cuda_mixed_socu.so`, and relinked the SOCU test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact][m8]" -r compact` | passed, `85` assertions in `5` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` | passed, `4985` assertions in `28` test cases |

Next M8 slice: consume the simplex target table in the normal-contact exact
in-band writer, then add matrix diff against the legacy structured contact
reference before enabling the path in scene gates.

M8 third slice implemented:

- Extended `SocuNativeContactStencilTarget` with the ordered row/column global
  vertex ids and the legacy `mirror_diag_block` decision. This preserves the
  `StructuredContactAssemblySink::write_hessian_half` upper-half ordering and
  same-block mirror rule in the target table itself.
- Added `src/backends/cuda_mixed_socu/linear_system/socu_native_contact_writer.h`.
  The exact writer consumes `SocuNativeContactStencilTarget` records and writes
  projected FEM/FEM, FEM/ABD, ABD/FEM, and ABD/ABD half-blocks through a
  `StructuredDeviceAssemblySink`. With native matrix fields enabled, the same
  writer writes native SOCU `D/E`; with legacy fields, it writes the legacy
  structured `D/E` layout.
- The ABD projection is implemented by scalar Jacobian weights rather than
  building device-side Eigen 12x12 temporaries. This keeps the small M8 contract
  test from becoming another heavyweight structured-contact TU.
- Added a mixed FEM/ABD PT simplex matrix-diff contract. It rebuilds the PT
  target table, consumes all 10 exact in-band half-block targets, writes native
  `D/E`, writes legacy-layout `D/E`, and checks both native and debug-compare
  buffers against the legacy layout.

Third-slice validation:

| check | result |
| --- | --- |
| `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; rebuilt `socu_native_contact_targets.cu` and relinked the SOCU test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact][m8]" -r compact` | passed, `6289` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` | passed, `11189` assertions in `29` test cases |

Current boundary: this proves the target-table exact writer and matrix layout
diff for simplex normal-style in-band blocks. The production
`ipc_simplex_normal_contact_structured.cu` path still writes through the legacy
structured sink until contact-phase native matrix/compare wiring is added to the
runtime scene path.

M8 fourth slice implemented:

- Added split production native exact contact TUs for simplex normal PT, EE, PE,
  and PP contacts:
  `ipc_simplex_normal_contact_native_{pt,ee,pe,pp}.cu`, plus a small dispatcher
  and shared inline write helper.
- `IPCSimplexNormalContact` now checks the rebuilt native target-table views in
  `SimplexNormalContact::ContactInfo`. When all simplex normal target tables
  are present and the pass is not the approximate-weight graph probe, the
  structured Hessian branch dispatches to the native exact writer.
- The native production path is intentionally all-or-legacy per contact. If
  every half-block target for a contact is `ExactInBand` or `Skipped`, the
  kernel consumes the target table and writes exact projected half-blocks. If
  any half-block is `DropOffBand`, `DiagFallback`, `DiagLumpFallback`, missing,
  or out of range, the whole contact falls back to the current structured sink.
  This preserves existing off-band policy behavior until native `diag` and
  `diag_lump` fallback writes are implemented.
- The split TUs keep compile cost bounded compared with inlining the new writer
  into the existing heavy structured contact implementation.

Fourth-slice validation:

| check | result |
| --- | --- |
| `cmake -S . -B build/build_impl_fp64` | passed; globbed build picked up the new native simplex normal contact files |
| `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; compiled the dispatcher plus PT/EE/PE/PP native TUs and relinked `libuipc_backend_cuda_mixed_socu.so` and the SOCU test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact][m8]" -r compact` | passed, `6289` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` | passed, `11189` assertions in `29` test cases |
| 20-frame contact-enabled smoke, `cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_topology_diag_lump --frames 20 --backend cuda_mixed_socu` | passed, `final_frame=20`, `wall_time_s=3.5400211589876562`, `mean_frame_ms=71.38351279718336`; the run reached real simplex contact frames without breaking the new dispatch |

Instrumentation note:

- A speculative attempt to add new global
  `NativeContactExactStencilHit`/`NativeContactLegacyFallbackStencil` counter
  slots through the shared structured counter header caused a broad rebuild
  that pulled the old `ipc_simplex_frictional_contact_structured.cu` TU back
  into `cicc`. That compile ran for over an hour and peaked around 50 GB RSS, so
  the counter-slot/report change was reverted.
- Existing scalar contact counters in `SocuNativeContactExactWriter` remain.
  Future native-contact hit-rate reporting should use a narrower debug buffer or
  a native-only report path that does not force the old structured frictional
  implementation to rebuild.

Current boundary after the fourth slice:

- Simplex normal production kernels now consume native target tables for exact
  in-band stencils.
- Contact assembly is not yet true SOCU native single-write in the runtime
  scene path, because the contact phase still passes the current
  `StructuredDeviceAssemblySink` storage. Native contact matrix/compare storage
  wiring is the next required step before declaring the contact builder fully
  native.
- Off-band `diag`/`diag_lump`, PH normal, and frictional contact families still
  use legacy structured fallback behavior.

M8 fifth slice implemented:

- Added opt-in contact-phase native matrix plumbing:
  `linear_system/socu_approx/native_contact_hessian`.
- Added opt-in contact-phase mirror diff:
  `linear_system/socu_approx/debug_compare_native_contact_hessian`.
- `GlobalLinearSystem::StructuredAssemblyInfo::sink()` now installs native
  matrix fields for the `Contact` phase, independently from the chain/base
  phase. This lets contact providers write either primary native storage or a
  native compare workspace through the same `StructuredDeviceAssemblySink`.
- The contact diff path reuses the opposite-layout Hessian compare workspace
  that is populated during chain/base assembly. As a result, a runtime frame can
  compare full native-vs-legacy `D/E/rhs` after contact assembly, not just a
  synthetic contact-only fixture.
- `cuda_mixed_wrecking_ball_compare.py` and
  `cuda_mixed_abd_fem_tower_viewer.py` now expose the new switches through
  `SOCU_NATIVE_CONTACT=1` and `SOCU_NATIVE_CONTACT_DIFF=1`.
- In native-only builds, excluded fallback contact callers now no-op when their
  contact set is empty. They still throw if an unsupported family has real
  contacts. This keeps native-only scene gates from failing on empty PH or
  frictional contact families.

Fifth-slice validation:

| check | result |
| --- | --- |
| full fallback rebuild attempt, `ninja -C build/build_impl_fp64 -j2 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` with `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF` | intentionally stopped; changing `global_linear_system.h` invalidated broad objects and pulled `ipc_simplex_frictional_contact_structured.cu` into `cicc` again |
| native-only configure, `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON` | passed; source exclusion reported `479 -> 475` source entries |
| native-only build, `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; split simplex normal native TUs compiled serially, device link and test executable link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact][m8]" -r compact` in native-only build | passed, `6289` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]" -r compact` in native-only build | passed, `11189` assertions in `29` test cases |
| native-only 20-frame scene attempt with `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_CONTACT_DIFF=1` | progressed past empty unsupported contact families; stopped at frame 9 when the scene produced `320` vertex-half-plane normal contacts, which are still intentionally unsupported in native-only mode |
| restore fallback cache, `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF` | passed; `CMakeCache.txt` reports `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY:BOOL=OFF` |

Current boundary after the fifth slice:

- Contact-phase native primary/compare storage is wired and compiles.
- The default fallback artifact was not relinked after the `global_linear_system.h`
  change because doing so requires the old heavy structured frictional contact
  TU. The CMake cache is restored to fallback mode, but the last linked binary
  was produced by the native-only validation build.
- The next runtime acceptance blocker is PH normal contact. The topology scene
  reaches PH contacts before simplex normal contacts, so a simplex-only
  native-only scene gate is not enough for the current accepted scene.

M8 sixth slice implemented:

- Added
  `rebuild_socu_native_vertex_half_plane_contact_targets(...)`, reusing the
  generic target-table rebuild kernel with `StencilSize=1`. For PH contacts the
  target table intentionally consumes only `PH(0)` as the active vertex stencil;
  `PH(1)` remains the half-plane index used by the contact model.
- Extended `VertexHalfPlaneNormalContact::ContactInfo` with a PH native target
  view and added a cached PH target buffer in the wrapper implementation.
- Moved PH topology and approximate-weight graph probes into small free helper
  kernels. This avoids the nvcc extended-lambda restriction on private member
  functions and keeps probe behavior aligned with the old structured PH normal
  path.
- Added `ipc_vertex_half_plane_normal_contact_native.{h,cu}`. The exact kernel
  computes the same `PH_barrier_gradient_hessian` as the old structured path,
  then consumes the PH target table through the shared
  `SocuNativeContactExactWriter` helper with `StencilSize=1`.
- Updated `IPCVertexHalfPlaneNormalContact` so its structured Hessian branch
  dispatches to the native exact PH writer whenever a PH target table is
  present. In native-only builds it still throws if a real PH contact appears
  without native targets.
- Extended the M8 contact target contract with a PH device-rebuild check,
  covering both FEM and ABD single-vertex PH targets and proving that the
  ignored half-plane slot does not affect the native target record.

Sixth-slice validation:

| check | result |
| --- | --- |
| `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON` | passed; source exclusion reported `481 -> 477` entries |
| `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed after moving PH probe kernels out of the private wrapper method; device link and shared library link completed |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact]" -s` | passed, `6312` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, `11212` assertions in `29` test cases |
| native-only scene attempt with `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_CONTACT=1 SOCU_NATIVE_CONTACT_DIFF=1` | progressed through frame 9 with `320` PH normal contacts and converged in three Newton iterations; stopped at frame 10 when `320` PH frictional contacts appeared and the native-only build correctly rejected the still-legacy PH frictional structured path |

Current boundary after the sixth slice:

- PH normal is no longer the native-only runtime blocker for the topology
  scene. Simplex normal and PH normal now both have production native exact
  target-table write paths.
- The next acceptance blocker is PH frictional contact. The topology scene hits
  PH frictional before simplex frictional, so M8 should migrate PH frictional
  next if the goal is to reach the full native-only contact acceptance gate.

M8 seventh slice implemented:

- Added PH frictional native target-table plumbing to
  `VertexHalfPlaneFrictionalContact`. The wrapper now handles topology and
  approximate-weight probes before model dispatch, rebuilds a single-vertex PH
  target table for exact writes, and passes that table through
  `ContactInfo`.
- Added `ipc_vertex_half_plane_frictional_contact_native.{h,cu}`. The exact
  kernel computes the same `PH_friction_gradient_hessian` as the existing
  structured path, applies `make_spd`, and consumes the shared PH target table
  through `SocuNativeContactExactWriter` with `StencilSize=1`.
- Updated `IPCVertexHalfPlaneFrictionalContact` so native-ready structured
  Hessian assembly dispatches to the PH frictional native exact writer. In
  native-only builds, real PH frictional contacts without native targets still
  throw.

Seventh-slice validation:

| check | result |
| --- | --- |
| `git diff --check` | passed |
| `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON` | passed; source exclusion reported `483 -> 479` entries |
| `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; compiled the PH frictional native TU, device link, shared library, and test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact]"` | passed, `6312` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, `11212` assertions in `29` test cases |
| native-only topology gate with `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_CONTACT=1 SOCU_NATIVE_CONTACT_DIFF=1`, `socu_rt50_topology_diag_lump --frames 20` | passed the previous frame-10 PH frictional blocker; progressed through frame 13 and began frame 14; stopped when simplex frictional produced `18342` Grad3 contributions and native-only rejected the still-legacy simplex frictional structured path |

Current boundary after the seventh slice:

- PH normal and PH frictional both have production native exact paths and no
  longer block the native-only topology gate.
- The next observed blocker is simplex frictional contact at frame 14. This is
  the largest remaining contact migration because PT/EE/PE/PP frictional
  stencils require `12x12`, `9x9`, and `6x6` Hessian kernels plus the existing
  `make_spd` and mollifier logic.

M8 eighth slice implemented:

- Added simplex frictional native target-table plumbing to
  `SimplexFrictionalContact`. The wrapper now records cheap frictional weights
  for graph probes, rebuilds PT/EE/PE/PP native target tables for exact
  structured Hessian assembly, and passes those views through `ContactInfo`.
- Added split simplex frictional native exact writers:
  `ipc_simplex_frictional_contact_native.{h,cu}` and
  `ipc_simplex_frictional_contact_native_{pt,ee,pe,pp}.cu`.
- PT, EE, PE, and PP keep the same generated IPC frictional Hessian path as the
  legacy structured implementation, including `make_spd`; EE keeps the
  existing mollifier branch that writes zero Hessian when mollification is
  required. The write side consumes `SocuNativeContactExactWriter` through the
  target table.
- PP was briefly implemented as a scalar exact-in-band writer to reduce
  `cicc` memory pressure, but was restored to the generated `6x6`
  `PP_friction_gradient_hessian` path so its computation matches the other
  native simplex frictional families and the legacy CUDA/structured path.
- Updated `IPCSimplexFrictionalContact` so native-ready structured Hessian
  assembly dispatches to the simplex frictional native exact writer. In
  native-only builds, real simplex frictional contacts without target tables
  still throw.

Eighth-slice validation:

| check | result |
| --- | --- |
| `git diff --check` | passed |
| `cmake -S . -B build/build_impl_fp64 -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON` | passed; native-only CMake cache is enabled |
| `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; compiled the split simplex frictional native TUs, device link, shared library, and test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact]"` | passed, `6312` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, `11212` assertions in `29` test cases |
| native-only topology gate with `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_CONTACT=1 SOCU_NATIVE_CONTACT_DIFF=1 SOCU_REPORT_COUNTERS=1`, `socu_rt50_topology_diag_lump --frames 20` | passed the previous frame-14 simplex frictional blocker and completed `final_frame=20`; summary reported `wall_time_s=3.790876034006942` and `mean_frame_ms=78.18641975754872` |

Current boundary after the eighth slice:

- The native-only contact path now covers PH normal, PH frictional, simplex
  normal, and simplex frictional exact in-band writes well enough to complete
  the 20-frame topology gate with contact mirror diff enabled.
- M8 is not complete yet. Remaining acceptance work: native off-band
  `diag`/`diag_lump` fallback consumption, compile-resource validation for the
  restored PP generated path, 100-frame topology gate, and a performance
  comparison against the structured contact baseline.

M8 direct-writer cleanup:

- The original M8 plan stated that production contact writes should use
  precomputed native block/lane targets and avoid per-scalar band
  classification, but the first implementation still used
  `SocuNativeContactExactWriter` as an adapter around
  `StructuredDeviceAssemblySink::add_hessian_scalar_status`. That preserved
  fallback/debug behavior, but it also kept legacy structured sink semantics in
  the native writer and contributed to high compile pressure for PP frictional.
- The plan now explicitly separates exact native writes, native off-band policy
  writes, and legacy structured fallback. Legacy fallback remains an outer
  dispatch path while M8 is incomplete; it is no longer a writer-internal
  behavior.
- `SocuNativeContactExactWriter` has been simplified to an exact-only direct
  primary writer: it requires native contact matrix storage, writes primary
  values through SOCU-native `D/E`, and returns without writing when targets are
  missing or not exact/skipped. It may still update the configured debug compare
  workspace, but production primary writes no longer fall back to structured
  contact storage inside the writer.
- Native contact dispatch now requires both ready target tables and installed
  native matrix storage. Default/fallback builds that do not enable
  `native_contact_hessian` continue to use the explicit legacy structured
  contact path outside native-only mode.
- This is V1 of the direct writer. It still uses native DoF descriptors while
  expanding ABD/FEM projections so arbitrary ABD lane orders remain correct.
  The high-performance V2 target table should precompute the final row/column
  lanes and projection weights so the writer only performs projected `3x3`
  block additions into `D/E`.

Direct-writer cleanup validation:

| check | result |
| --- | --- |
| `git diff --check` | passed |
| `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; rebuilt the touched contact wrappers, device link, backend shared library, core config, and test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact]"` | passed, `3238` assertions in `6` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, `8138` assertions in `29` test cases |
| native-only topology gate with `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_CONTACT=1 SOCU_NATIVE_CONTACT_DIFF=1 SOCU_REPORT_COUNTERS=1`, `socu_rt50_topology_diag_lump --frames 20` | passed; completed `final_frame=20` with `native_contact_hessian_enabled=true`, contact mirror diff enabled, and `native_contact_hessian_diff_mismatch_count=0` |

Two cleanup fixes were needed for the direct-writer gate:

- Empty contact families now return early in native-only structured assembly.
  This preserves the old no-op behavior for unsupported families before any
  real contacts exist, while still throwing when a nonempty contact family lacks
  native storage or target tables.
- `linear_system/socu_approx/native_contact_hessian` and
  `debug_compare_native_contact_hessian` are now registered in the default
  scene config. Before this fix the example set the keys, but the solver could
  not find them, so the report showed `native_contact_hessian_enabled=false`
  and native-only dispatch rejected real PH normal contacts at frame 9.

M8 V2 direct-lane target slice:

- Added precomputed row/column lane arrays to exact native contact target
  records. Each exact half-block target can now tell the writer the final
  native `D` or first-offdiag `E` lanes for up to `12` local DoFs per side.
- Updated the exact native contact writer so primary writes use those
  precomputed lanes when `direct_lanes_valid=true`. The FEM/FEM direct path can
  now write native `D/E` without native DoF descriptors, `old_to_chain`, or
  per-scalar pair classification in the writer.
- Kept a V1 descriptor-assisted path for targets whose scalar half-blocks do
  not map to one uniform `D`/first-offdiag destination. ABD projection weights
  are still expanded in the writer, so this is a correctness-oriented first
  slice of V2 rather than the final high-performance schema.
- Added a synthetic arbitrary-lane writer contract that constructs native
  matrix storage without native DoF descriptors and verifies direct diagonal
  and transposed first-offdiag writes from the target lanes alone.

V2 direct-lane validation:

| check | result |
| --- | --- |
| `git diff --check` | passed before the validation build |
| `ninja -C build/build_impl_fp64 -j1 RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu` | passed; rebuilt the target-schema users, device link, backend shared library, and test executable |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact][v2]" -s` | passed, `30` assertions in `1` test case |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_native_contact]"` | passed, `3282` assertions in `7` test cases |
| `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` | passed, `8182` assertions in `30` test cases |
| native-only topology gate with `SOCU_NATIVE_CHAIN_BASE=1 SOCU_NATIVE_DIAG_RHS=1 SOCU_NATIVE_CONTACT=1 SOCU_NATIVE_CONTACT_DIFF=1 SOCU_REPORT_COUNTERS=1`, `socu_rt50_topology_diag_lump --frames 20` | passed; completed `final_frame=20`, `wall_time_s=53.931154954014346`, `mean_frame_ms=2588.105514511699`, `native_contact_hessian_enabled=true`, contact mirror diff enabled, and `native_contact_hessian_diff_mismatch_count=0` |

Current V2 boundary:

- Do not claim a performance win from this slice. The target record is larger
  because it stores simple fixed-size lane arrays, and the measured topology
  run above had debug counters plus mirror diff enabled.
- Next V2 work should precompute ABD projection weights, compact or split the
  target schema so contact kernels do not pay unnecessary target-table
  bandwidth, and then rerun diff-off 100-frame structured-vs-native performance
  gates.
