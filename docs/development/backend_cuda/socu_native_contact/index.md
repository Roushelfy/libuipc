# SOCU Native Contact Handoff

This directory is the current agent-facing entry point for the SOCU native
contact assembly redesign. The long redesign plan and integration journal remain
the detailed historical record; this directory should stay short, current, and
gate-oriented.

## Current Status

Current phase: **M6.7 docs and gates hardening**.

Accepted foundation:

- M1 to M6.6 established compact symbolic side/program plans, source-id indexed
  contact programs, native executor buckets, hot-block strategies, Wrecking Ball
  scene smoke gates, and the opt-in direct evaluator.
- M6.6 direct evaluator is accepted as a development and performance gate path,
  not as a default cutover.

Open blockers before M8 cutover:

- Direct evaluator and executor implementation is still header-heavy.
- Direct production source-scan isolation needs stronger structural checks.
- The M6.7 unsupported-direct-program guard uses a host readback before scatter;
  M8 must remove it from the peak direct path or record measured acceptance.
- Performance gates do not yet have the required repeated median table,
  baseline artifact and cold/cache-hit/topology-churn split. A fresh
  direct-vs-triplet_compat partial-rebuild median exists, but it is not a
  cutover benchmark.
- `fixed_mapping_epoch` and `vertex_projection_epoch` producer ownership must be
  documented and tested.

Local validation snapshot, 2026-05-14:

- Configure and build passed with vcpkg at `/home/zhaofeng/work/vcpkg`, CUDA
  compiler `/usr/local/cuda-12.8/bin/nvcc`, and
  `python/.venv/bin/python`.
- `--mode source-scan` passed.
- `--mode contract` passed after fixing native matrix builder fixture
  synchronization around host uploads, memset initialization, and host copies.
- `external/socu-native-cuda` is initialized at gitlink
  `45292fdb942b2424c26d4db327019b765754accf`; CMake reports socu_native
  integration enabled.
- Builder-generated ABD side lanes now have a contract fixture that checks
  legacy projection components and `ABDJacobi::x_bar()` weights from real
  builder output.
- `native_contact_scalar_diag_compat` now has a builder contract proving that
  `Diag` fallback emits `DiagScalarFem` and `DiagScalarAbd`, not only key/report
  fields.
- `hybrid` now has a low-level unsupported-source contract: direct evaluation
  emits per-program unsupported flags, `direct`/`direct_compare` reject them,
  and `hybrid` replaces unsupported program Hessians through a counted triplet
  fallback path.
- Contact source enumeration now routes plan input, topology stamp, direct
  sources, direct-compare references, and `triplet_compat` references through a
  shared host source catalog helper.
- Direct evaluator per-family parity now has a CUDA contract fixture covering
  PT/EE/PE/PP/PH direct Hessians against a triplet oracle.
- Wrecking Ball 20-frame scene gates passed for `direct` and `direct_compare`.
  A 3-run partial-rebuild median showed `direct` contact assembly at
  `10.2849 ms` and `triplet_compat` at `13.1288 ms`; this is useful evidence,
  not a cache-hot/default-cutover claim.
- The contract gate still skips the MathDx LTO synthetic solve smoke when
  `build/socu_native_contact/mathdx_lto/manifest.json` is absent.

## Reading Order

1. [roadmap.md](roadmap.md) - current phase, next work, runnable gates, planned
   gates.
2. [build_and_run.md](build_and_run.md) - configure, build, C++ gate, uv Python,
   and scene commands.
3. [architecture.md](architecture.md) - stable data flow and peak performance
   target.
4. [conventions.md](conventions.md) - rules future agents must preserve.
5. [testing.md](testing.md) - validation ladder and invariant coverage.
6. [benchmark_protocol.md](benchmark_protocol.md) - performance acceptance
   protocol.

Historical context:

- [SOCU Native Assembly Builder Redesign Plan](../socu_native_assembly_builder_redesign_plan.md)
  keeps the detailed milestone design and acceptance notes.
- [SOCU Mixed Solver Integration Journal](../socu_mixed_solver_integration_journal.md)
  records chronological decisions, commands, and observed results.

## Default Gates

Fast gates:

```bash
uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode contract

uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode source-scan
```

Report gate:

```bash
uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode report \
  --reports output/examples/wrecking_ball_native_direct \
  --report-evaluator direct \
  --require-no-triplets \
  --require-no-direct-fallbacks
```

Scene gates are intentionally opt-in because they run the Wrecking Ball example.
The gate runner reads the configured Python from CMake and injects build package
paths as described in [build_and_run.md](build_and_run.md).

```bash
uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode scene \
  --scene-evaluator direct \
  --output output/examples/socu_native_contact_scene_gates
```

## Non-Negotiable Principle

Every implementation decision must serve one goal: **make contact assembly a
cacheable symbolic plan plus a direct native numeric replay, with legacy triplets
only as an oracle or explicit compatibility bridge**.

Rules that must stay testable:

1. Direct production replay must not call reporter triplet assembly.
2. Native executor hot paths must not re-enter the structured contact sink.
3. Source ids must have one consistent ordering across builder, evaluator,
   topology stamps, and compatibility triplet views.
4. Performance claims require repeated reports and subsystem timing fields.
5. Default cutover is blocked until M8 gates pass.
