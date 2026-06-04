#!/usr/bin/env python3
"""Sweep `rcc_adhesion_normal_offset_coeff` and rank by solver conditioning.

The soft RCC point-triangle NORMAL adhesion energy is `E = (Cn/2d_hat) beta^2 D`
(D = gap^2), minimized at gap d=0 and monotone increasing across the C-IPC band
d in (xi, xi+d_hat). It therefore pulls the surfaces into the stiff barrier wall
at d=xi -> poorly conditioned Newton solve. The offset coefficient `c` moves the
minimum to d* = xi + c*d_hat (c=0.5 = band center), turning the normal term into
a gentle spring to a natural gap. `c=0` disables the offset (legacy, min at d=0).

This harness runs the headless drop demo (PROFILE path) once per `c`, in a fresh
subprocess each (clean CUDA engine state), parses each `timer_frames.json` via
`uipc.stats.SimulationStats`, and prints a `c`-vs-conditioning table:

  * mean Newton iterations / frame   (primary: nonlinear-solve hardness)
  * mean SpMV / frame                (secondary: ~PCG iterations, linear cond.)
  * mean Line Search / frame         (secondary: energy-landscape shape)
  * ms / frame                       (tertiary, noisy)

Run with BONDED off by default so the soft RCC normal energy is actually
exercised (locked pairs bypass it).

Examples:
    python python/examples/rcc_adhesion_offset_sweep.py
    python python/examples/rcc_adhesion_offset_sweep.py \
        --asset temflex175_2turn_soft_default --frames 90 \
        --coeffs 0 0.25 0.5 0.75 1.0
    # confirm no regression with bonding on:
    python python/examples/rcc_adhesion_offset_sweep.py --bonded --coeffs 0 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(REPO, "python", "examples")
DROP = os.path.join(EXAMPLES, "rcc_adhesive_tape_drop_demo.py")

# (label shown, alias key resolved against the timer tree)
METRICS = [
    ("newton", "newton_iteration"),  # primary
    ("spmv", "spmv"),                # secondary A (~PCG iterations)
    ("line_search", "line_search"),  # secondary B
]


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset", default="temflex175_2turn_soft_default",
                   help="tape asset to drop (default: a soft 2-turn roll)")
    p.add_argument("--coeffs", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="offset coefficients c to sweep")
    p.add_argument("--frames", type=int, default=90,
                   help="PROFILE_FRAMES per run (default 90)")
    p.add_argument("--warmup", type=int, default=0, help="PROFILE_WARMUP")
    p.add_argument("--bonded", action="store_true",
                   help="run with bonding ON (regression check); default OFF")
    p.add_argument("--out", default=None,
                   help="sweep output root (default output/rcc_adhesion_offset_sweep)")
    p.add_argument("--python", default=sys.executable, help="python interpreter")
    return p.parse_args(argv)


def run_one(args, c):
    out_dir = os.path.join(args.out, f"c_{c:g}")
    cmd = [args.python, DROP, "--asset", args.asset,
           "--set", f"BONDED={1 if args.bonded else 0}",
           "--set", f"RCC_ADHESION_NORMAL_OFFSET_COEFF={c}",
           "--set", "PROFILE=1",
           "--set", f"PROFILE_FRAMES={args.frames}",
           "--set", f"PROFILE_WARMUP={args.warmup}",
           "--set", f"PROFILE_DIR={out_dir}"]
    print(f"\n=== c={c:g} (bonded={args.bonded}) ===\n  {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    # PROFILE writes <out_dir>/<asset>_<nobond|bonded>/{benchmark,timer_frames}.json
    hits = glob.glob(os.path.join(out_dir, "*", "timer_frames.json"))
    return rc, (os.path.dirname(hits[0]) if hits else None)


def metrics_for(result_dir):
    from uipc.stats import SimulationStats
    tf = os.path.join(result_dir, "timer_frames.json")
    st = SimulationStats.load_timer_frames_json(tf)
    # get_values does EXACT name matching; resolve alias -> actual timer name.
    aliases = [a for _, a in METRICS]
    resolved, _ = st._resolve_keys(aliases)
    name_of = dict(zip(aliases, resolved))
    out = {}
    for label, alias in METRICS:
        _, vals = st.get_values(name_of[alias], metric="count")
        out[f"mean_{label}"] = float(vals.mean()) if len(vals) else math.nan
        out[f"max_{label}"] = float(vals.max()) if len(vals) else math.nan
    bm_path = os.path.join(result_dir, "benchmark.json")
    if os.path.exists(bm_path):
        bm = json.load(open(bm_path))
        nf = max(int(bm.get("num_frames", 0)), 1)
        out["ms_per_frame"] = 1000.0 * float(bm.get("wall_time", 0.0)) / nf
        out["frames"] = int(bm.get("num_frames", 0))
    else:
        out["ms_per_frame"] = math.nan
        out["frames"] = 0
    return out


def print_table(rows):
    hdr = (f"\n{'c':>6}  {'ok':>3}  {'mean_newton':>11}  {'max_newton':>10}  "
           f"{'mean_spmv':>9}  {'mean_ls':>8}  {'ms/frame':>9}  {'frames':>6}")
    print(hdr)
    print("-" * len(hdr))
    for c, ok, m in rows:
        if not ok or m is None:
            print(f"{c:6g}  {'NO':>3}  {'(run failed / no timer_frames.json)':>50}")
            continue
        print(f"{c:6g}  {'OK':>3}  {m['mean_newton']:11.2f}  {m['max_newton']:10.0f}  "
              f"{m['mean_spmv']:9.1f}  {m['mean_line_search']:8.2f}  "
              f"{m['ms_per_frame']:9.1f}  {m['frames']:6d}")


def recommend(rows):
    ok = [(c, m) for c, ok_, m in rows if ok_ and m is not None
          and not math.isnan(m["mean_newton"])]
    base = next((m for c, m in ok if c == 0.0), None)
    if not ok:
        print("\n[recommend] no successful runs.")
        return
    # best = smallest c whose mean_newton is within 2% of the global minimum
    best_newton = min(m["mean_newton"] for _, m in ok)
    near = [(c, m) for c, m in sorted(ok) if m["mean_newton"] <= best_newton * 1.02]
    c_star, m_star = near[0]
    print(f"\n[recommend] best conditioning at c={c_star:g} "
          f"(mean Newton {m_star['mean_newton']:.2f}/frame, "
          f"mean SpMV {m_star['mean_spmv']:.1f}/frame).")
    if base and c_star != 0.0:
        dn = 100.0 * (1 - m_star["mean_newton"] / base["mean_newton"]) \
            if base["mean_newton"] else float("nan")
        ds = 100.0 * (1 - m_star["mean_spmv"] / base["mean_spmv"]) \
            if base["mean_spmv"] else float("nan")
        print(f"            vs c=0 baseline: Newton {dn:+.1f}%, SpMV {ds:+.1f}%.")
        meets = (dn >= 20.0 and ds >= 15.0
                 and m_star["max_newton"] <= base["max_newton"])
        print(f"            default-flip criteria (Newton>=20%% & SpMV>=15%% & "
              f"no max_newton regression): {'MET' if meets else 'not met'} "
              f"-> {'propose default c=%g' % c_star if meets else 'keep default 0 (opt-in)'}.")


def main(argv=None):
    args = parse_cli(argv)
    if args.out is None:
        args.out = os.path.join(REPO, "output", "rcc_adhesion_offset_sweep"
                                + ("_bonded" if args.bonded else ""))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for c in args.coeffs:
        rc, rdir = run_one(args, c)
        ok = (rc == 0) and rdir is not None
        m = None
        if ok:
            try:
                m = metrics_for(rdir)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] c={c:g}: failed to parse metrics: {exc}")
                ok = False
        rows.append((c, ok, m))
    print_table(rows)
    recommend(rows)
    print(f"\n[sweep] artifacts under {args.out}/")


if __name__ == "__main__":
    main()
