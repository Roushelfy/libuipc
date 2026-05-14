# SOCU Native Contact Roadmap

## Core Principle

Every implementation decision must serve one goal: **make contact assembly a
cacheable symbolic plan plus direct native numeric replay, with legacy triplets
only as an oracle or explicit compatibility bridge**.

Non-negotiable rules:

1. Direct production replay does not assemble reporter triplets.
2. Native executor hot paths do not re-enter legacy structured contact sinks.
3. Source ids stay dense and consistent across all source consumers.
4. Performance claims use repeated medians and subsystem timing fields.
5. Default cutover waits for M8 gates.

## Phase Summary

### M1 to M6.6: Foundation (Complete)

- [x] Compact symbolic side/program plan types and cache keys.
- [x] Dense source ids, source-to-program maps, and O(1) program lookup.
- [x] Global and demand-filled side coverage.
- [x] Executor buckets for exact, diag, diag-lump, drop, and hot-block paths.
- [x] Wrecking Ball 20-frame native-plan smoke gates.
- [x] Opt-in direct evaluator and `direct_compare` oracle path.

### M6.7: Docs And Gate Hardening (Current)

- [x] Add agent-facing architecture, conventions, testing, and benchmark docs.
- [x] Add generic report analyzer for native contact reports.
- [x] Add default gate runner for contract, source-scan, report, and opt-in
  scene gates.
- [x] Strengthen source scans for direct production isolation.
- [x] Record M6.7 contract-closure blockers for ABD projection, scalar diag
  compatibility, hybrid fallback, cache-key producer epochs, direct per-family
  parity, and source catalog drift.
- [x] Configure and build `uipc_test_backend_cuda_mixed_socu` locally with
  vcpkg, explicit CUDA compiler, and uv Python venv.
- [x] Run the source-scan gate on the local build artifact.
- [x] Make the full contract gate green on the local build artifact.
- [x] Close the builder-generated ABD projection contract with a real builder
  lane fixture using nontrivial `ABDJacobi::x_bar()` weights.
- [x] Close the scalar diag compatibility behavior contract with a real builder
  fixture that emits `DiagScalarFem` and `DiagScalarAbd`.
- [x] Close the hybrid fallback accounting contract with unsupported direct
  source flags, direct-mode rejection, and counted per-program triplet fallback.
- [x] Introduce the host contact source catalog helper and route plan input,
  topology stamp, direct sources, direct-compare references, and `triplet_compat`
  references through the same source order.
- [x] Capture a fresh 3-run Wrecking Ball direct/triplet_compat median table,
  and a 20-frame direct_compare correctness gate.

Contract-closure blockers:

| Blocker | Why It Blocks Cutover | Required Gate |
| --- | --- | --- |
| Cache-key producer epochs | Mapping and ABD projection changes must not reuse stale plans | Producer contract documents whether descriptor epoch covers mapping/projection; otherwise nonzero epoch changes rebuild both layers |

Closed M6.7 contract items:

| Item | Gate |
| --- | --- |
| Builder-generated ABD projection lanes | `uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_assembly_plan_side_table_active_set"` checks builder output lanes for `component = q < 3 ? q : (q - 3) / 3` and `weight = q < 3 ? 1 : x_bar((q - 3) % 3)` |
| Scalar diagonal compatibility behavior | `uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_assembly_plan_scalar_diag_compatibility"` proves `native_contact_scalar_diag_compat` changes emitted tasks to `DiagScalarFem` and `DiagScalarAbd` |
| Hybrid fallback accounting | `uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_direct_evaluator_flags_unsupported_sources"` proves unsupported direct source detection and fallback Hessian replacement |
| Shared source catalog | Source-scan gate requires `collect_socu_contact_source_catalog` and rejects the old per-consumer source id counters |
| Direct evaluator per-family parity | `uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_direct_evaluator_per_family_triplet_parity"` builds real native programs and compares PT/EE/PE/PP/PH direct Hessians against a triplet oracle |

Acceptance gates:

- Contract binary passes `[cuda_mixed_socu][contract]`.
- Source-scan tests prove direct production replay is isolated from reporter
  triplet assembly except in diagnostic `direct_compare` and counted `hybrid`
  fallback.
- Report analyzer rejects direct production reports that still spend time in
  `native_contact_hessian_triplet_ms`.
- Report analyzer rejects direct production reports with nonzero direct
  unsupported or fallback counters unless the roadmap accepts that mode.

### M7: Replay Observability (Next)

- [ ] Make probe, replay, and final cache states observable in reports.
- [ ] Split timing so side-plan and program-plan rebuild costs are not
  double-counted by aggregate build timing.

Acceptance gates:

- Contract test mutates source ordering and proves every consumer observes the
  same `source_id`.
- Reports distinguish cold rebuild, cache hit, topology churn, and side-key
  churn.
- Cache-hit Wrecking Ball reports show native replay without symbolic rebuild.

### M8: Cutover Decision (Future)

- [ ] Isolate heavy direct evaluator and executor implementation from broad
  headers or record measured compile-resource acceptance.
- [ ] Remove the M6.7 unsupported-program host readback from the production
  direct path, or record measured acceptance if it remains as a safety guard.
- [ ] Add native-only build graph checks for `compile_commands.json`,
  `build.ninja`, or dry-run Ninja output.
- [ ] Prove direct native replay is faster than `triplet_compat` and the frozen
  old target-table baseline on the same workload.
- [ ] Decide whether to switch defaults from off/triplet_compat to native
  direct mode.

Acceptance gates:

- Direct Wrecking Ball gate has `native_contact_hessian_triplet_ms == 0`.
- `direct_compare` has zero mismatch count on the accepted scene mix.
- 3-run median table is recorded with build, GPU, driver, and seed metadata.
- No M8 cutover is accepted on correctness-only evidence.

### M9: Peak Performance Shape (Future)

- [ ] Fuse direct local Hessian evaluation with native scatter for selected
  family/bucket combinations.
- [ ] Remove dense per-program Hessian global-memory traffic from production
  fused paths.
- [ ] Optimize owner-reduce or cached microblock strategies for hot native
  blocks.
- [ ] Consider CUDA Graph capture only after symbolic shape and launch
  parameters are stable.

Acceptance gates:

- Fused path matches `direct_compare` or legacy oracle within accepted tolerance.
- Fused path beats non-fused direct on cache-hot steady-state median.
- Hot-block strategy improves or preserves correctness and timing on repeated
  high-contention scenes.

## Runnable Gates

| Gate | Command | Required Result |
| --- | --- | --- |
| Contract | `uv run --no-project python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode contract` | Catch tests with `[cuda_mixed_socu][contract]` pass |
| Source scan | `uv run --no-project python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode source-scan` | Source isolation tests pass |
| Report direct | `uv run --no-project python scripts/analyze_socu_native_contact_reports.py <reports> --require-native-plan --require-evaluator direct --require-no-triplets --format markdown` | Native direct reports have no triplet timing |
| Report compare | `uv run --no-project python scripts/analyze_socu_native_contact_reports.py <reports> --require-native-plan --require-evaluator direct_compare --require-direct-compare-zero --format markdown` | Direct compare mismatch count is zero |
| Report direct strict | `uv run --no-project python scripts/analyze_socu_native_contact_reports.py <reports> --require-native-plan --require-evaluator direct --require-no-triplets --require-no-direct-fallbacks --format markdown` | Native direct reports have no unsupported or fallback programs |
| ABD projection builder contract | `build/socu_native_contact/RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_assembly_plan_side_table_active_set"` | Real builder output ABD lanes match legacy projection components and weights |
| Scalar diag builder contract | `build/socu_native_contact/RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_assembly_plan_scalar_diag_compatibility"` | Real builder output emits `DiagScalarFem` and `DiagScalarAbd` when compat is enabled |
| Hybrid fallback direct contract | `build/socu_native_contact/RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_direct_evaluator_flags_unsupported_sources"` | Unsupported direct sources are flagged and can be replaced by fallback Hessians |
| Direct per-family parity | `build/socu_native_contact/RelWithDebInfo/bin/uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_direct_evaluator_per_family_triplet_parity"` | PT/EE/PE/PP/PH direct evaluator output matches the triplet oracle |

## Planned Gates

| Gate | Target Command | Missing Piece |
| --- | --- | --- |
| Scene direct | `uv run --no-project python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode scene --scene-evaluator direct --output <run-dir>` | Requires local CUDA build, configured Python, and Wrecking Ball assets |
| Scene direct compare | `uv run --no-project python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode scene --scene-evaluator direct_compare --output <run-dir>` | Requires local CUDA build, configured Python, and Wrecking Ball assets |
| Build graph isolation | `uv run --no-project python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode build-graph` | Build graph scanner not implemented yet |
| Fused direct eval+scatter | Future benchmark command | Fused kernels are not implemented |
| CUDA Graph replay | Future benchmark command | Graph capture path is not implemented |
