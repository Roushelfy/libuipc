# SOCU Mixed Structured Direct Solver

This document records the current `cuda_mixed` SOCU integration. Historical
checkpoint notes have been retired from this file so the documented behavior
matches the production path.

## Scope

`socu_approx` is a structured-band direct direction solver for the
`cuda_mixed` backend. Local linear build providers assemble directly into a
`StructuredAssemblySink`; the solver does not build, filter, or redistribute a
full Hessian triplet for the SOCU path.

The assembled matrix contains the diagonal block band and the first
off-diagonal block band accepted by `socu_native`. If a scene's true Hessian is
fully represented by that band, the solve is exact for the assembled Newton
system. If contributions fall outside the band, they are recorded and dropped
from the SOCU matrix, and the solve is explicitly a structured-band
approximation.

The default structured scope is:

```text
linear_system/socu_approx/structured_scope = multi_provider
```

`single_provider` remains available for strict provider isolation checks.

## Runtime Path

The final solve path is:

1. A `StructuredChainProvider` owns the block-chain layout and old-to-chain
   mappings.
2. Local ABD, FEM, ABD-FEM coupling, joint, constraint, and contact reporters
   write Hessian and RHS contributions directly into the structured sink when
   the selected solver requests SOCU assembly.
3. The runtime uploads validated mappings once, creates the `socu_native` plan
   during solver build, and reuses the plan in Newton solves.
4. Each solve assembles the structured band, calls `socu_native`
   `factor_and_solve`, runs lightweight direction validation, and scatters the
   direction back to the global vector.

The fused PCG path keeps using the normal full sparse assembly route. Shared
local Hessian evaluation code may feed either sink, but the SOCU path writes
its structured destination directly.

Runtime Hessian-based RCM reordering is optional and disabled by default. When
enabled, a matching frame first runs a graph-only structured probe, records the
atom-pair graph from structured Hessian writes, including off-band writes before
they are dropped from the structured matrix, reorders the merged graph on the
CPU, and installs the new structured runtime/plan before the final structured
assembly and solve. A runtime reorder failure leaves the previous ordering
active and is reported as a diagnostic.

Current-frame probing must be treated as a per-linear-build concern, not only a
per-frame concern. Contact sets can change after the first Newton solve in a
frame; for example, `wrecking_ball` frame 13 has only PH contacts in Newton
iteration 0, then activates EE/PP simplex contacts in Newton iteration 1. If
runtime reorder probes only once at the start of the frame, the later EE/PP
body-pair graph is absent from the ordering, strong off-band blocks are dropped,
and the SOCU direct direction can become invalid even though the current-frame
ordering report for the first solve shows `off_band_ratio = 0`.

## Contact Write Plan Cache

Large IPC contact Hessian kernels can become register-bound when they compute
the contact Hessian and also inline the full structured contact sink. The
current stable path splits the largest structured contact writes through
compact Hessian workspaces before the structured scatter pass. The production
default uses that stable compact path for EE/PP simplex normal contact and keeps
PT/PE on the direct structured sink path.

The write-plan cache is staged so unverified direct-write expansion does not
replace the stable path:

1. Reorder-level vertex slots. After each init-time ordering install or runtime
   reorder, build a device table indexed by global vertex. Each entry records
   the vertex kind, fixed state, ABD body, old DoF begin, chain block, chain
   local offset, and the ABD Jacobian index. Contact kernels then
   avoid repeated `global_vertex -> old_dof -> chain` lookups.
2. Contact-set write plans. After contact detection updates the active
   PT/EE/PE/PP/PH stencils, build a lightweight device write plan for each
   half-Hessian vertex pair under the current ordering. The plan records the
   FEM/ABD projection kind, diag/first-offdiag/off-band status, target block
   and local offsets, transpose direction, and skip/fixed state.
3. Debug validation. `debug_contact_write_plan_validate = 1` builds PT/EE/PE/PP
   simplex normal write plans and reports plan skip/near/off-band counters
   without changing the default write path.
4. Stable planned compact scatter. EE/PP simplex normal contact use
   `build plan -> compact Hessian workspace -> planned scatter` by default.
   PT/PE remain on the direct structured sink path unless the debug validator is
   enabled.
5. Experimental planned direct write.
   `experimental_contact_planned_direct_write = 1` tries EE/PP simplex normal
   planned direct writes, reusing the same write plan while bypassing the
   compact Hessian workspace. This is deliberately off by default.

The experimental target shape is:

```text
H = compute_contact_hessian(...)
write_by_plan(contact_write_plan[i], H)
```

instead of redoing vertex mapping, old-to-chain lookup, band classification, and
fixed/off-band handling inside the large Hessian kernel. ABD projection values
still depend on current ABD Jacobian data, but the write location and projection
kind are fixed for a given contact stencil and ordering.

PH, frictional contact, and PT/PE planned direct writes are not production
defaults. Expansion requires all of the following gates: `wrecking_ball
socu_rt1` reaches frame 16, planned and old structured matrices match in band,
ptxas register/spill usage does not regress without a runtime win, and
contact-heavy benchmarks are not slower by more than 5%.

## Failure And Report Semantics

Hard failures are limited to invalid or unsupported states:

- missing or malformed ordering data
- unsupported block size
- incomplete DoF coverage or invalid mapping
- unsupported precision contract
- unavailable or failing `socu_native` runtime
- non-finite, non-descent, or residual-invalid directions
- configured line-search rejection fallback

The following quality fields are diagnostics only and do not reject a solve:

- `min_block_utilization`
- `min_near_band_ratio`
- `max_off_band_ratio`
- `max_off_band_drop_norm_ratio`

Ordering quality, low block utilization, and runtime off-band contributions are
written into the report. Runtime off-band contributions do not disable the
direction; the report status explains that the solve continued with the
in-band structured matrix.

## Configuration

Typical strict structured solve configuration:

```json
{
  "linear_system": {
    "solver": "socu_approx",
    "socu_approx": {
      "structured_scope": "multi_provider",
      "ordering_source": "init_time",
      "ordering_orderer": "rcm",
      "ordering_block_size": "64",
      "damping_shift": 0.0,
      "runtime_reorder_frame_interval": 0,
      "runtime_reorder_edge_capacity": 0,
      "runtime_reorder_graph_source": "topology",
      "debug_contact_write_plan_validate": 0,
      "experimental_contact_planned_direct_write": 0
    }
  }
}
```

`damping_shift = 0.0` and `ordering_block_size = "64"` are the defaults.

`ordering_source` only supports `init_time`; external ordering report mode has
been removed from the solver path. `ordering_orderer` only supports `rcm` in
the runtime solver. `generated_ordering_report` may still be set to write the
init-time ordering diagnostics for inspection.

`runtime_reorder_frame_interval = 0` means init-time ordering only. A positive
interval rebuilds ordering on frames satisfying `frame % interval == 0` before
that frame's first structured solve, then assembles once with the new ordering.
`runtime_reorder_graph_source` selects the graph source: `topology` uses cached
base topology plus current contact topology with unit weights,
`contact_hessian` keeps base topology at weight 1 and weights current contacts
by accumulated absolute contact Hessian contribution, and `full_hessian` uses a
graph-only full Hessian probe for diagnostics. Runtime reorder preserves the
currently installed block size. The collector capacity defaults to an automatic
value; setting `runtime_reorder_edge_capacity > 0` overrides it. Collector
overflow, empty graphs, invalid mappings, or
`socu_native` plan creation failures do not abort the solve; they are reported
and the previous ordering remains active.

SOCU approx now exposes only the strict structured direct solve path. The
previous assembly-only validation path has been removed; structured assembly
diagnostics are reported from the real solve path through
`linear_system/socu_approx/report`.

## Report Fields

The solve report records:

- layout size, block utilization, active and padding DoF counts
- ordering quality diagnostics and configured thresholds
- structured write counters for diagonal, first off-diagonal, and off-band
  dropped contributions
- RHS norm, residual, relative residual, descent dot, gradient norm, direction
  norm, and direction validation thresholds
- plan/timing/report counters when enabled
- line-search feedback when available
- optional `runtime_reorder` diagnostics: enabled state, interval, capacity,
  collecting frame, last applied frame, raw/unique edge counts, overflow count,
  apply status, and failure detail

The report does not include dense matrix eigen summaries or pre-factor matrix
downloads. Direction validation is the required lightweight correctness check
on the solve path.

## Ordering Lab

`experiments/socu_ordering_lab` remains available as a standalone lab target
for ordering studies. It is not part of the runtime hot path.

## Verification

Primary fp64 build:

```bash
cmake --build build/build_impl_fp64 --target libuipc_backend_cuda_mixed.so pyuipc uipc_test_sim_case_cuda_mixed_only -j 8
```

Contract smoke:

```bash
build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only "86_cuda_mixed_linear_solver_selection_smoke" -s
```

Recommended asset regression checks cover:

- `abd_fem_tower` with zero damping and strict structured solve residual near
  machine precision when all contributions are in band
- `abd_external_force` and `cube_ground` with default diagnostics enabled
- `fem_bouncing_cubes` long frame runs with runtime off-band reporting and no
  abort
- `fem_link_drop` with init-time off-band diagnostics and no quality gate
