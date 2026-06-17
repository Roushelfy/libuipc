# RCC Adhesion — Optimization Status

Tracking the 10 optimization candidates from the 4-agent code audit, **re-prioritized
by actual measurement** (CPU `uipc.Timer` tree + `ncu gpu__time_duration` on a rod-wind
orbit window, asset `speed-r150-bend5k`, distance-lock mode).

## Measurement baseline (before any optimization)

`ncu` per-kernel GPU duration, rod-wind orbit window (top of ~25 ms profiled):

| %GPU | kernel | note |
|---|---|---|
| 26.2 | `InfoStacklessBVH::stacklessSelf` | collision broadphase (NOT RCC) |
| **14.7** | **`RCCBondedPTVirtualTetReporter::compute_energy`** | bonded virtual-tet energy |
| 8.7 | `DiscreteShellBending::grad_hessian` | FEM |
| 8.6 | `InfoStacklessBVH::stacklessOther` | collision broadphase |
| **6.5** | **`RCCBondedPTVirtualTetReporter::assemble`** | bonded grad/hessian |
| … | `filter_active` 0.6%, `filter_toi` 0.3% | RCC trajectory filter |

CPU timer (per frame ~0.4 ms CPU scope; GPU is async): linear solver dominates CPU
launch overhead (~740 PCG iters/solve → SpMV 2230 launches/frame). Adhesion CPU
stages < 4%. **The CPU timer cannot see the occlusion GPU cost — ncu is the arbiter.**

## Status table

| # | item | status | evidence |
|---|---|---|---|
| ① | occlusion cast → BVH | **REFUTED — skip** | ncu < 0.3% GPU. Audit's "10^9 tests" was ~100× over: actual n_tris≈4000 × ~1867 locks ≈ 7.5M, negligible on GPU. |
| ② | producer host↔device syncs | **TODO** | structural; producer not a top GPU kernel but the 4+ syncs bubble the async pipeline. Risk: medium (producer hot path). |
| ③ | filter `has_locks` guard + drop redundant post-filter | **TODO (next)** | low risk; universal broadphase tax + real double-work in lock scenes. |
| ④ | lock-set re-sort → incremental merge | **TODO-low** | cub radix sorts ≈ 1.4% GPU. Bug-prone; low payoff. |
| ⑤ | `_rebuild_adhesive_tabular` dirty-flag | **NEGLIGIBLE** | `m_N` = contact-element count (~5), so the "O(N²) host vector" is ~25 structs, not N_vertices². Audit overstated. ~20 `find()`s/frame are host-trivial. |
| ⑥ | share energy/grad/hess basis | **LOW / N/A** | this is the SOFT-adhesion path; in distance-lock mode soft energy is not assembled at all. Beta-mode only. |
| ⑦ | reporter gradient_only skips Hessian | **DONE** | commit 76e80918. + bare-Float energy eval. |
| ⑧ | 4-filter dedup → shared helper | **TODO (maintainability)** | no perf; the patch already drifted (lbvh lost a debug block). |
| ⑨ | c>1 VT-range vs AABB under-reach | **REFUTED — sound** | point AABB AND triangle AABB each expand by full `d_hat`, so combined broadphase reach ≈ 2·d_hat covers any scale ≤ 2 (the clamp ceiling). c=1.5 far locks are real. |
| ⑩ | misc constant factors | **PARTIAL** | DONE: the 12×12 struct local-mem spill (the real cost, via ⑦). TODO-small: `eigen::inverse` per existing lock/frame in `release_flags_from_current_shape` (store Dm instead). PT_rcc_coeff copies / buffer-resize churn: negligible. |

## Key lesson

The audit's complexity-based ranking (① occlusion as "single highest value") was **wrong
on the constants** — the only super-linear term had small actual counts. The real RCC GPU
cost was the bonded virtual-tet reporter, and there the bottleneck was **occupancy
(a 1152-byte Matrix12x12 return-struct spilling to local memory), not arithmetic**:
returning a bare `Float` from the energy path cut `compute_energy` 1818→270 µs/call
(−85 %, ~12 % of total GPU). Measure GPU kernels with ncu before committing to a rewrite;
CPU scope timers and big-O estimates both misled here.

## Remaining work order (evidence-based)

1. ③ filter `has_locks` guard + remove redundant post-filter (low risk, universal).
2. ⑩ producer `eigen::inverse` → stored Dm (small, producer hot path).
3. ⑧ 4-filter dedup (code health).
4. ② producer host-sync removal (structural, only if re-profiling justifies the risk).
5. ④ incremental lock merge (lowest payoff; likely skip).
