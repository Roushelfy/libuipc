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
