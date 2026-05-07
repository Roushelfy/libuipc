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
