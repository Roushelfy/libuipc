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

## Contact Structured Writes

Ordinary IPC contact assembly keeps the pre-SOCU shape: gradient/Hessian kernels
write the original doublet and triplet buffers and do not capture the structured
contact sink. SOCU contact Hessian assembly is split into separate structured
translation units so the ordinary kernels do not inherit SOCU register pressure
or template instantiations.

The production structured path is deliberately direct:

```text
H = compute_contact_hessian(...)
structured_sink.write_hessian_half(stencil, H)
```

Simplex normal and frictional PT/EE/PE/PP contacts, plus vertex-half-plane
normal/frictional PH contacts, use this compute-and-write path. The structured
sink maps the current global vertex to FEM/ABD old DoFs, projects ABD blocks,
classifies diag/first-offdiag/off-band writes, and records runtime-reorder graph
edges when the runtime ordering collector is active.

The previous reorder-level vertex-slot table, contact-set write plans, planned
scatter path, and planned-direct-write debug switches are not part of the
current runtime. They can be revisited only as a separate optimization after
the direct structured path is stable on `wrecking_ball socu_rt1 --frames 20`
through `>>> Begin Frame: 16`, with any later known direction-validation NaN
tracked separately.

The ordinary `fused_pcg` path remains the contact assembly baseline. Any SOCU
contact split must keep `wrecking_ball fused_pcg --frames 20` passing through
frame 20; a fused-PCG failure before frame 20 is a refactor blocker, not an
acceptable SOCU validation limitation.

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
      "runtime_reorder_graph_source": "topology"
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
graph-only full Hessian probe for diagnostics. Experimental graph sources are
`contact_weight_approx`, `full_weight_approx`, and `full_hessian_cached`.
Approximate sources use cheap contact coefficients for ordering weights. The
cached full Hessian source replays the contact Hessian half-blocks computed by
the graph probe for the immediately following structured contact assembly.
Runtime reorder preserves the currently installed block size. The collector
capacity defaults to an automatic value; setting
`runtime_reorder_edge_capacity > 0` overrides it. Collector overflow, empty
graphs, invalid mappings, or
`socu_native` plan creation failures do not abort the solve; they are reported
and the previous ordering remains active.

The wrecking-ball comparison script exposes the default topology runtime
variant as `socu_rt1`, plus explicit diagnostics `socu_rt1_contact_hessian` and
`socu_rt1_full_hessian`. It also exposes experimental variants for
`full_weight_approx` and `full_hessian_cached`. They use the same frame
interval and differ only in the runtime reorder graph source, so frame-16
failures can be compared without changing the ordinary `fused_pcg` baseline.

Future off-band contact experiments must remain opt-in and must not replace the
default exact structured contact assembly without measurement. The planned
stability sequence is:

1. `diag` / `diag_lump`: compute the exact contact Hessian, write it unchanged
   when the entire contact stencil fits in the current SOCU band, and otherwise
   replace the partial off-band stencil with either the exact per-vertex
   diagonal contribution (`diag`) or a conservative nonnegative lumped-diagonal
   contribution (`diag_lump`).
2. `approx_diag`: use cheap normal/frictional contact weights to write only
   per-vertex diagonal blocks in the final structured matrix. This is a matrix
   approximation, not merely a runtime ordering graph approximation.
3. `hybrid`: use exact structured contact writes for fully in-band stencils and
   approximate diagonal writes for stencils that would otherwise be partially
   dropped.
4. Approximate cache: only consider this after profiling shows approximate
   diagonal assembly is still dominated by repeated traversal or
   classification.

These policies are intended to diagnose whether partial off-band contact block
dropping breaks positive semidefiniteness. Validation must compare the new
variant against `fused_pcg`, `socu_rt1_full_hessian_cached`, and the exact
`contact_hessian` / `full_hessian` diagnostics before any default behavior
changes.

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
