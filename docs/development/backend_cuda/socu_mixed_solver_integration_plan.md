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
      "ordering_orderer": "auto_stable",
      "ordering_block_size": "auto",
      "damping_shift": 1e-6
    }
  }
}
```

`damping_shift = 0.0` is supported. The default remains `1e-6`.

`ordering_source` only supports `init_time`; external ordering report mode has
been removed from the solver path. `generated_ordering_report` may still be set
to write the init-time ordering diagnostics for inspection.

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
