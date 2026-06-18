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
| ③ | filter `has_locks` guard | **DONE** | commit ed6ab3b2. NOTE: the "broadphase tax" was smaller than the audit claimed — `lower_bound` over EMPTY keys is O(1) (0 probes), not log(N); the guard only saves a `PT_pair_key` hash per candidate in non-adhesion scenes. Post-filter NOT removed: it is already gated by `size()==0` and is only 0.6% GPU in lock scenes (removal = correctness risk for ~0 gain). |
| ④ | lock-set re-sort → incremental merge | **TODO-low** | cub radix sorts ≈ 1.4% GPU. Bug-prone; low payoff. |
| ⑤ | `_rebuild_adhesive_tabular` dirty-flag | **NEGLIGIBLE** | `m_N` = contact-element count (~5), so the "O(N²) host vector" is ~25 structs, not N_vertices². Audit overstated. ~20 `find()`s/frame are host-trivial. |
| ⑥ | share energy/grad/hess basis | **LOW / N/A** | this is the SOFT-adhesion path; in distance-lock mode soft energy is not assembled at all. Beta-mode only. |
| ⑦ | reporter gradient_only skips Hessian | **DONE** | commit 76e80918. + bare-Float energy eval. |
| ⑧ | 4-filter dedup → shared helper | **TODO (maintainability)** | no perf; the patch already drifted (lbvh lost a debug block). |
| ⑨ | c>1 VT-range vs AABB under-reach | **REFUTED — sound** | point AABB AND triangle AABB each expand by full `d_hat`, so combined broadphase reach ≈ 2·d_hat covers any scale ≤ 2 (the clamp ceiling). c=1.5 far locks are real. |
| ⑩ | misc constant factors | **DONE (the one that mattered)** | the 12×12 struct local-mem spill (the real cost) fixed via ⑦. Re-assessed the rest as NEGLIGIBLE: `eigen::inverse` per lock/frame ≈ 30 flops × 1867 = 56K flops/frame (producer not even top-28 GPU); PT_rcc_coeff copies + buffer-resize churn similarly noise. Not worth touching the producer hot path. |

## Key lesson

The audit's complexity-based ranking (① occlusion as "single highest value") was **wrong
on the constants** — the only super-linear term had small actual counts. The real RCC GPU
cost was the bonded virtual-tet reporter, and there the bottleneck was **occupancy
(a 1152-byte Matrix12x12 return-struct spilling to local memory), not arithmetic**:
returning a bare `Float` from the energy path cut `compute_energy` 1818→270 µs/call
(−85 %, ~12 % of total GPU). Measure GPU kernels with ncu before committing to a rewrite;
CPU scope timers and big-O estimates both misled here.

## Final state

The optimization pass is effectively complete. The single real win — ⑦/⑩ bonded-energy
struct spill, **−85 % compute_energy, ~12 % of total GPU** — is landed (76e80918). ③ is a
clean universal hygiene guard (ed6ab3b2). Everything else is REFUTED (①⑨), NEGLIGIBLE
(⑤⑥⑩-rest), or risk-without-measured-payoff (②④⑧).

Deferred, would only revisit with new evidence:
- ② producer host-sync removal — structural refactor of the producer hot path
  (over-allocate + device-side counts). The 4 syncs/frame are pipeline bubbles a CPU
  timer can't size; needs a Nsight Systems timeline to justify the risk.
- ⑧ 4-filter dedup — pure maintainability (the patch has drifted); a 4-kernel refactor
  with zero perf upside, deferred to avoid churning the production hot path.
- ④ incremental lock merge — lowest payoff (~1.4 % radix sort), bug-prone; skip.

## Solver: linear tolerance & preconditioner (rod-wind, 2-turn, 123 frames)

Separate from the GPU-kernel audit above: the linear PCG solve is the other big wall-clock
lever. PCG converges on the relative `rᵀM⁻¹r ≤ tol_rate·rz0` test (`tol_rate` = `LIN_TOL_RATE`),
`max_iter = 2·DoF`. Measured on the `tape_abd002_nodal002w` profile, all runs complete 123
frames / 2460 Newton solves:

| config | preconditioner | LIN_TOL | PCG iters (mean) | Newton iters/solve | result |
|---|---|---|---|---|---|
| baseline | ABD/FEM block-diagonal Jacobi | 1e-4 | 462.9 | 9.5 | ✅ |
| MAS | FEM multilevel additive Schwarz (part=16) | 1e-4 | 405.6 (−12 %) | 9.2 | ✅ |
| **default** | block-diagonal Jacobi | **1e-3** | **399.0 (−14 %)** | 9.4 | ✅ −8 % wall |
| (probe) | block-diagonal Jacobi | 1e-2 | 6.5 | **509.8** 💥 | ❌ stalls, 52 max-iter hits |

Conclusions:
- **`LIN_TOL_RATE` 1e-4 → 1e-3 is a free −8 % wall win**, no Newton/stability penalty
  (9.5 → 9.4 iters/solve). The bonded/contact conditioning front-loads PCG iterations, so a
  tighter final tolerance only shaves the converged tail. Now the profile default.
- **1e-2 is past the cliff**: the linear direction is too inaccurate, Newton degrades to
  ~510 iters/solve and hits the 1024 cap (52 non-converged steps) — ~40× the per-frame Newton
  work. Rejected.
- **MAS net-negative for this scene.** Verified by code read (`mas_preconditioner_engine.cu`,
  `fem_mas_preconditioner.cu`): it IS a genuine multilevel method (`MAX_LEVELS=6`, Galerkin
  `H_L = R_L H R_L^T` per level via `scatter_hessian_to_clusters`; additive — not a
  multiplicative V-cycle — combining levels by injection `collect_final_Z`). The full Hessian
  including **contact + bonded** triplets does enter, and distant vertices ARE coupled at
  coarser levels. But (1) the coarsening hierarchy is built **once** from the **static
  rest-mesh element graph** (`add_edge` over tets/tris/codim edges), not the dynamic
  contact/bond graph — so two contacting-but-mesh-distant coil layers only aggregate at very
  coarse levels, giving weak capture of their stiff coupling; (2) FEM↔ABD bonds are entirely
  outside the FEM hierarchy (ABD verts have no `mesh_part`); (3) the multilevel apply costs
  more per PCG iter than diagonal. Result: only −12 % iters, net **+7 % wall**. Loosening to
  1e-3 with the cheap diagonal preconditioner beats MAS@1e-4 on both PCG mean (399 < 406) and
  per-apply cost. `TAPE_PARTITION>0` knob retained in the demo for re-evaluation on other scenes.

### Contact-aware MAS partition — upper-bound experiment (tested, rejected)

Follow-up question: would re-anchoring the MAS hierarchy on the *current contact/bond*
topology (instead of the static rest mesh) help? A code feasibility study confirmed the
hierarchy is already rebuilt every Newton iter (`reorder_realtime`, which even takes a
`cp_num` contact-count arg — dead/reserved plumbing); the only static thing is the neighbor
graph, restored from a rest-mesh `_init` snapshot. So contact-awareness = feed an augmented
neighbor graph. The blocker for the *ideal* version is that fine-cluster bank assignment comes
from the static `mesh_part` partition, not the graph — so true co-clustering needs re-partitioning.

We measured the **upper bound** (ideal static contact-aware *fine* partition): captured the
1057 tape-tape bonded PT pairs from the wound-roll asset → 1777 cross-layer FEM-FEM edges (each
spanning ~71 grid rows ≈ one coil turn; 0 overlap with rest-mesh edges), injected them into
`mesh_partition`'s METIS adjacency via the env-gated `UIPC_MESH_PARTITION_EXTRA_EDGES` hook, and
re-ran the 2-turn rod-wind (`speed-r150-bend5k`, `RCC_KAPPA=3e7`) at 1e-3:

| config | partition | PCG mean | Newton/solve | wall (2460 frames) |
|---|---|---|---|---|
| diagonal@1e-3 | — | 399 | 9.4 | 938 s |
| MAS rest-mesh@1e-3 | static rest graph | 328.5 (−18 %) | 8.8 | 976 s (+4 %) |
| **MAS contact-augmented@1e-3** | rest + 1777 contact edges | **292.4 (−27 %)** | 9.9 | **1027 s (+9 %)** |

**Verdict: confirmed on iterations, rejected on wall.** Co-clustering the contacting layers
*does* cut PCG iterations exactly as predicted (−27 % vs diagonal, −11 % beyond static MAS —
the break-even iteration target was even exceeded). **But the wall got monotonically *worse***
(938 → 976 → 1027 s): the denser contact-augmented clustering raises the MAS per-apply cost more
than the iteration savings recover (C is slower than B despite 11 % fewer iters). The bottleneck
was never the iteration count — it is the multilevel-apply cost, and contact-awareness pushes
that the wrong way. This init-locks static version is +9 % wall vs the free diagonal@1e-3.
(The per-frame question is refined in the follow-up below, which uses a better-targeted
late-ORBIT snapshot + a per-phase breakdown — the short answer is per-frame's ceiling is
≈ parity, still not a clear win.) diagonal@1e-3 remains best.
(Apparatus: `output/distlock_run/{capture_edges.py,run_experiment.sh}` + the
`UIPC_MESH_PARTITION_EXTRA_EDGES` hook in `mesh_partition.cpp`, kept for re-evaluation on
contact-stiffness-dominated scenes where the per-apply premium might pay off.)

### Follow-up: per-phase breakdown + late-ORBIT-matched partition (refines the above)

The init-locks partition above was mis-targeted: 97 % of the solver cost is in ORBIT (the
new-wind phase), where the *dynamic* rod-wind contacts — not the asset's roll bonds — dominate.
So we (a) bucketed PCG iters/wall by phase, and (b) re-ran with a partition built from the LIVE
locks captured at a late-ORBIT frame (frame 2000, 2003 locks → 2018 edges, via the demo
`DUMP_LOCKS_AT_FRAME`/`DUMP_LOCKS_OUT` hook). Per-phase wall (s) on the 2-turn rod-wind:

| phase | diag | MAS rest | MAS init-aug | MAS late-ORBIT-aug |
|---|---|---|---|---|
| 3_WRAP (fold) | 14 | 25 | 71 | **96** ← mismatch catastrophe |
| 7_ORBIT_early | 472 | 495 | 482 | **468** |
| 8_ORBIT_late | 363 | 368 | 378 | 371 |
| **ORBIT total** | **835 (iters 9.0M)** | 863 | 860 | **840 (+0.6 %, iters 6.2M −31 %)** |
| **full-run total** | **938** | 976 (+4 %) | 1027 (+9 %) | 1014 (+8 %) |

Two refinements to the verdict:
- **In its matched phase (ORBIT, 97 % of cost), contact-aware MAS is essentially break-even on
  wall (+0.6 %) while cutting iters −31 %** — when the partition matches the active contacts, the
  multilevel per-apply tax IS nearly recovered. This is the one genuinely encouraging signal.
- A *static* partition can only match one phase: the late-ORBIT partition wins ORBIT but is
  catastrophic in WRAP (96 s vs 14 s, 6.8×, badly mismatched to the folding phase). That WRAP
  blow-up accounts for nearly all of its +8 % overall loss.

A **per-frame** partition would match each phase and avoid the WRAP-type mismatches. Idealized
per-phase-best ≈ 917 s vs diagonal 938 s ≈ **−2 % wall** — but that is the *ceiling* (perfect
matching, zero rebuild cost). The win comes from FINE-level re-partitioning, so a real per-frame
loop pays per-frame METIS + contact D2H + engine re-init (~5–15 ms/frame × 2460 ≈ 12–37 s),
which is the same order as the ~20 s idealized win. **Net: per-frame's realistic ceiling is
≈ parity with diagonal@1e-3, not a clear win** — so a multi-day per-frame implementation is not
justified for this scene. The user's intuition (static partition is phase-mismatched; per-frame
fixes it) is correct and the ORBIT break-even is real, but the headroom is too thin to bank.
diagonal@1e-3 remains the pragmatic best. Phase tool: `output/distlock_run/phase_wall.py`.
