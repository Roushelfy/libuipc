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
- Do not add dense matrix eigen diagnostics or full Hessian triplet fallback to
  the SOCU runtime path.

Future SOCU changes should update the final behavior document instead of adding
new staged journal entries.
