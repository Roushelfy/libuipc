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
- [ ] Capture a fresh 3-run Wrecking Ball direct/direct_compare/triplet table.

Contract-closure blockers:

| Blocker | Why It Blocks Cutover | Required Gate |
| --- | --- | --- |
| Scalar diagonal compatibility behavior | `native_contact_scalar_diag_compat` must change emitted tasks, not only key/report fields | Builder contract proves `DiagScalarFem` and `DiagScalarAbd` are emitted when the flag is enabled |
| Hybrid fallback accounting | `hybrid` must not be an uncounted alias for `direct` | Unsupported direct fixture increments unsupported/fallback counters in `hybrid`; `direct` rejects or reports an error |
| Cache-key producer epochs | Mapping and ABD projection changes must not reuse stale plans | Producer contract documents whether descriptor epoch covers mapping/projection; otherwise nonzero epoch changes rebuild both layers |
| Direct evaluator per-family parity | Scene gates do not isolate PT/EE/PE/PP/PH formula bugs | Unit fixtures compare direct vs triplet/CPU oracle for PT, EE, PE, PP, and PH |
| Shared source catalog | Repeated source ordering can drift across builder/direct/triplet/topology | Source catalog contract proves all consumers share one enumeration order |

Closed M6.7 contract items:

| Item | Gate |
| --- | --- |
| Builder-generated ABD projection lanes | `uipc_test_backend_cuda_mixed_socu "cuda_mixed_socu_contact_assembly_plan_side_table_active_set"` checks builder output lanes for `component = q < 3 ? q : (q - 3) / 3` and `weight = q < 3 ? 1 : x_bar((q - 3) % 3)` |

Acceptance gates:

- Contract binary passes `[cuda_mixed_socu][contract]`.
- Source-scan tests prove direct production replay is isolated from reporter
  triplet assembly except in diagnostic `direct_compare`.
- Report analyzer rejects direct production reports that still spend time in
  `native_contact_hessian_triplet_ms`.
- Report analyzer rejects direct production reports with nonzero direct
  unsupported or fallback counters unless the roadmap accepts that mode.

### M7: Source Enumeration And Replay Observability (Next)

- [ ] Introduce one host-side contact source enumeration helper.
- [ ] Route builder inputs, topology stamps, direct sources, and triplet
  compatibility sources through the same ordering contract.
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

## Planned Gates

| Gate | Target Command | Missing Piece |
| --- | --- | --- |
| Scene direct | `uv run --project python python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode scene --scene-evaluator direct --output <run-dir>` | Requires local CUDA build, Python env vars, and Wrecking Ball assets |
| Scene direct compare | `uv run --project python python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode scene --scene-evaluator direct_compare --output <run-dir>` | Requires local CUDA build, Python env vars, and Wrecking Ball assets |
| Build graph isolation | `uv run --no-project python scripts/run_socu_native_contact_gates.py --build build/socu_native_contact --mode build-graph` | Build graph scanner not implemented yet |
| Scalar diag builder contract | Future Catch test in `[cuda_mixed_socu][contract]` | Needs builder fixture that forces diag fallback with scalar compatibility enabled |
| Hybrid fallback counter contract | Future Catch test in `[cuda_mixed_socu][contract]` | Needs unsupported direct source fixture |
| Direct per-family parity | Future Catch tests or CUDA fixtures | Needs PT/EE/PE/PP/PH direct-vs-oracle fixtures |
| Source catalog consistency | Future Catch/source-scan test | Needs shared source catalog helper |
| Fused direct eval+scatter | Future benchmark command | Fused kernels are not implemented |
| CUDA Graph replay | Future benchmark command | Graph capture path is not implemented |
