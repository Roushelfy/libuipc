# SOCU Native Contact Benchmark Protocol

Benchmark claims are accepted only when they are reproducible and split by the
question being answered.

## Required Metadata

Every accepted benchmark table must record:

- git branch and commit;
- binary path;
- CMake cache path or the exact configure command;
- GPU, driver, and CUDA toolkit;
- scene variant, frame count, and seed when available;
- environment variables;
- warmup policy;
- report directory;
- analyzer command.

## Required Timing Fields

Tables must include these fields when present:

- `native_contact_plan_build_ms`;
- `native_contact_side_plan_build_ms`;
- `native_contact_program_plan_build_ms`;
- `native_contact_direct_eval_ms`;
- `native_contact_hessian_triplet_ms`;
- `native_contact_direct_compare_ms`;
- `native_contact_executor_scatter_ms`;
- `native_contact_hot_reduce_ms`;
- `native_contact_numeric_ms`;
- `contact_assembly_time_ms`.

Tables should also group by:

- `native_contact_probe_cache_state`;
- `native_contact_replay_cache_state`;
- `native_contact_final_cache_state`;

`native_contact_plan_build_ms` is the aggregate symbolic build wall time.
`native_contact_side_plan_build_ms` and
`native_contact_program_plan_build_ms` are split substage timings and must not
be produced by copying the same aggregate value into both columns.

## Benchmark Questions

Do not mix these questions into one number:

| Question | Required Comparison |
| --- | --- |
| Cold rebuild cost | First native plan report vs legacy/triplet baseline |
| Cache-hit steady state | Native replay reports where side and program plans hit |
| Topology churn | Program plan rebuild with side plan hit |
| Side-key churn | Side and program plan rebuild after ordering/descriptor change |
| Direct vs bridge | `direct` vs `triplet_compat` on same scene/config |
| Diagnostic correctness | `direct_compare` mismatch and abs-error fields |
| End-to-end impact | Frame time and contact assembly time with same scene/frame count |

## Repetition Rule

Performance acceptance requires at least three runs with the same binary,
scene, frame count, and environment. Report the median. Do not accept a speedup
from a single best run.

Example analyzer command:

```bash
uv run --no-project python scripts/analyze_socu_native_contact_reports.py \
  output/examples/socu_native_contact_direct \
  --require-native-plan \
  --require-evaluator direct \
  --require-no-triplets \
  --min-samples 3 \
  --format markdown
```

## Direct Production Acceptance

A direct production report must have:

- `native_contact_replay_path == "native_plan"`;
- `native_contact_evaluator_path == "direct"`;
- `native_contact_hessian_triplet_ms == 0`;
- `native_contact_direct_unsupported_program_count == 0`;
- `native_contact_direct_fallback_program_count == 0`.

Fallback programs are allowed only for a roadmap-approved `hybrid` gate, and the
fallback/unsupported counters must be part of the accepted report.

## Direct Compare Acceptance

A direct compare report must have:

- `native_contact_replay_path == "native_plan"`;
- `native_contact_evaluator_path == "direct_compare"`;
- `native_contact_direct_compare_mismatch_count == 0`;
- recorded `native_contact_direct_compare_max_abs_error`;
- recorded `native_contact_direct_compare_sum_abs_error`.

## Cutover Rule

M8 cutover cannot be accepted on correctness alone. The direct native path must
beat `triplet_compat` and the frozen legacy target-table baseline on the same
workload, with repeated medians and complete metadata.
