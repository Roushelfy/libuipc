# SOCU Native Contact Testing

The validation ladder should make it hard for future agents to accidentally
turn the direct native path back into a legacy triplet or structured sink path.
Build and runtime setup lives in [build_and_run.md](build_and_run.md).

## Test Ladder

| Layer | Purpose | Current Entry |
| --- | --- | --- |
| Unit/contract | Prove small local invariants and report semantics | `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"` |
| Source scan | Prevent forbidden dependencies and direct-path regressions | `cuda_mixed_socu_contact_assembly_plan_source_scan` |
| Integration | Prove solver wiring, defaults, cache keys, and report fields | `[cuda_mixed_socu][contract][socu_approx]` tests |
| Scene | Prove Wrecking Ball native contact replay reaches frame 20 | `cuda_mixed_wrecking_ball_compare.py` |
| Benchmark | Prove speed claims with repeated medians | `analyze_socu_native_contact_reports.py` plus recorded runs |

## Invariant Matrix

| Invariant | Unit/Contract | Source Scan | Scene/Benchmark |
| --- | --- | --- | --- |
| Source ids are dense and source-local ids map to O(1) programs | `cuda_mixed_socu_contact_source_id_dense_contract`, `cuda_mixed_socu_contact_assembly_plan_*` | Builder source scan rejects host-copy topology patterns | Wrecking Ball final report has valid source/program counts |
| Side plan and program plan invalidate separately | `cuda_mixed_socu_contact_plan_cache_split_layers` | Solver scan requires split cache stats mapping | Reports separate side/program cache state |
| Direct production path avoids reporter triplet assembly | Report defaults and evaluator path tests | Direct branch scan allows triplets only in `direct_compare` and `triplet_compat` | Direct scene gate requires `native_contact_hessian_triplet_ms == 0` |
| Direct compare remains an oracle | Report field tests | Scan requires direct compare launcher and error recorder | Direct compare scene gate requires zero mismatch |
| Executor does not depend on legacy target/sink/classification | Executor and writer parity tests | Executor/writer source isolation scans | Native replay path remains `native_plan` |
| Empty contact topology is a native no-op | Empty plan contract tests | Scan requires `native_contact_empty_plan_replay` | Empty or zero-contact scenes do not fall back to legacy contact TUs |
| Hot-block strategies preserve writer semantics | Hot-block and executor tests | Scan requires strategy parser and report fields | High-contention scenes compare timing/counters |
| Builder-generated ABD lanes match legacy projection | `cuda_mixed_socu_contact_assembly_plan_side_table_active_set` checks real builder lane component/weight for nontrivial `x_bar` | Source scan keeps the ABD projection convention documented | Direct/compare scenes cannot replace this unit gate |
| Scalar diag compatibility changes builder output | `cuda_mixed_socu_contact_assembly_plan_scalar_diag_compatibility` checks `DiagScalarFem` and `DiagScalarAbd` task emission | Source scan keeps the scalar diag convention documented | Report gate checks scalar diag counts when the mode is benchmarked |
| Hybrid fallback is explicit and counted | Planned unsupported-source fixture checks direct reject vs hybrid fallback counters | Source scan keeps hybrid blocker in roadmap | Report analyzer checks direct fallback counters |
| Cache-key producers cover mapping/projection changes | Planned producer fixture checks nonzero epoch ownership or descriptor-epoch coverage | Source scan keeps cache-key producer blocker in roadmap | Cache-churn reports distinguish side and program rebuilds |
| Direct evaluator parity covers each family | Planned PT/EE/PE/PP/PH fixtures compare direct vs triplet or CPU oracle | Source scan keeps direct parity blocker in roadmap | `direct_compare` scene remains a broad smoke, not a unit oracle |
| Source catalog order is shared | Planned source catalog fixture compares builder/direct/triplet/topology order | Source scan keeps source catalog blocker in roadmap | Wrecking Ball source counts remain secondary smoke |

## Required Fast Gates

Run these before editing native contact implementation:

```bash
uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode contract

uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode source-scan
```

Current local status, 2026-05-14: both `--mode source-scan` and
`--mode contract` pass on `build/socu_native_contact` with
`external/socu-native-cuda` initialized. The contract gate still skips the
MathDx LTO synthetic solve smoke when
`build/socu_native_contact/mathdx_lto/manifest.json` is absent.

Run this after producing reports:

```bash
uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode report \
  --reports output/examples/<native-contact-run> \
  --report-evaluator direct \
  --require-no-triplets \
  --require-no-direct-fallbacks
```

## Planned Scene Gates

Scene gates are not part of the fast default because they require a working
CUDA runtime, Wrecking Ball assets, and enough time for 20-frame runs.

```bash
uv run --project python python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode scene \
  --scene-evaluator direct \
  --output output/examples/socu_native_contact_direct

uv run --project python python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode scene \
  --scene-evaluator direct_compare \
  --output output/examples/socu_native_contact_direct_compare
```

Direct production reports must satisfy:

- `native_contact_replay_path == "native_plan"`;
- `native_contact_evaluator_path == "direct"`;
- `native_contact_hessian_triplet_ms == 0`;
- `native_contact_direct_unsupported_program_count == 0`;
- `native_contact_direct_fallback_program_count == 0`;
- `native_contact_direct_eval_ms > 0`;
- `native_contact_executor_scatter_ms > 0`.

Direct compare reports must satisfy:

- `native_contact_replay_path == "native_plan"`;
- `native_contact_evaluator_path == "direct_compare"`;
- `native_contact_direct_compare_mismatch_count == 0`.
