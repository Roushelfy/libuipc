#!/usr/bin/env python3
"""
Sweep the drop demo across a list of solver-precision profiles, saving
each run as a separate MP4. Useful for `which solver tol settings do I
actually need?` ablation.

Each profile runs the SAME loaded wound-tape asset through the SAME
drop physics — only the libuipc solver knobs change. Output structure:

    <output-root>/
      quick/    pngs/frame_NNNNNN.png   output.mp4   sim.log   metrics.csv   summary.json
      default/  ...
      high/     ...
      extreme/  ...
      paranoid/ ...

Each profile's `sim.log` is libuipc's INFO-level stdout for that run
(captured live + saved). `metrics.csv` has one row per (frame, Newton iter)
with the residuals and min PT contact distance the solver actually saw.
`summary.json` has one entry per simulated frame with the converged
exit values.

Requirements (one-time install):
    sudo apt-get install ffmpeg        # video encoder
    # polyscope's EGL backend handles offscreen rendering; no xvfb needed.

Run me:
    python python/examples/rcc_adhesive_tape_drop_precision_sweep.py \
        --asset wound_2t_preci_3_2 \
        --profiles default high extreme

    # All profiles:
    python python/examples/rcc_adhesive_tape_drop_precision_sweep.py \
        --asset wound_2t_preci_3_2

    # Print profiles only, don't run:
    python python/examples/rcc_adhesive_tape_drop_precision_sweep.py --list

Profiles (lowest → highest precision):
    quick     libuipc defaults (1e-3 / 0.05 m/s) — fastest, blurriest
    default   1e-4 / 5e-3                        — typical demo grade
    high      1e-5 / 5e-4                        — serious experiment
    extreme   1e-6 / 5e-5                        — paper-grade; practical max
    paranoid  1e-7 / 5e-6                        — fp64 noise floor; diminishing returns

Polyscope's EGL backend renders offscreen — no display required.
ffmpeg then combines the PNG sequence into MP4.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time


# ---------------------------------------------------------------------
# Precision profiles
# ---------------------------------------------------------------------
# Each profile is a dict of (cli-key → value) passed as `--set KEY=VAL`
# to the drop demo. See `tape_asset_lib.SOLVER_KEYS` for the mapping
# to libuipc's nested scene config (`linear_system/tol_rate`,
# `newton/velocity_tol`, etc.).
PROFILES = {
    "quick": dict(
        LIN_TOL_RATE=1e-3,
        NEWTON_VELOCITY_TOL=0.05,
        NEWTON_TRANSRATE_TOL=0.1,
        NEWTON_MAX_ITER=1024,
        LINE_SEARCH_MAX_ITER=8,
    ),
    "default": dict(
        LIN_TOL_RATE=1e-4,
        NEWTON_VELOCITY_TOL=5e-3,
        NEWTON_TRANSRATE_TOL=1e-2,
        NEWTON_MAX_ITER=1024,
        LINE_SEARCH_MAX_ITER=8,
    ),
    "high": dict(
        LIN_TOL_RATE=1e-4,
        NEWTON_VELOCITY_TOL=5e-4,
        NEWTON_TRANSRATE_TOL=1e-3,
        NEWTON_MAX_ITER=2048,
        LINE_SEARCH_MAX_ITER=16,
    ),
    "extreme": dict(
        LIN_TOL_RATE=1e-4,
        NEWTON_VELOCITY_TOL=5e-5,
        NEWTON_TRANSRATE_TOL=1e-3,
        NEWTON_MAX_ITER=4096,
        LINE_SEARCH_MAX_ITER=64,
    ),
    "paranoid": dict(
        LIN_TOL_RATE=1e-7,
        NEWTON_VELOCITY_TOL=5e-6,
        NEWTON_TRANSRATE_TOL=1e-5,
        NEWTON_MAX_ITER=8192,
        LINE_SEARCH_MAX_ITER=64,
    ),
}


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DROP_DEMO = os.path.join(
    REPO_ROOT, "python", "examples", "rcc_adhesive_tape_drop_demo.py"
)


# ---------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------
# Match patterns against the INFO-level libuipc log emitted by the drop
# demo. The four shapes:
#
#   "============================= [*] Frame F Newton Iteration N ============================"
#   "    [*] <uipc::backend::cuda::MaxTranslationChecker> Residual/AbsTol: X/Y"
#   "    [*] <uipc::backend::cuda::ABDToleranceChecker> Residual/AbsTol: X/Y"
#   "[time] [info] ContactPairMinDist: PT=X, count=N"
#   "[time] [info] Newton Iteration Converged with Iteration Count: N, Bound: [a, b]"
#   "[time] [info] >>> Begin Frame: F"
#
# All four log lines are produced unconditionally at INFO level — the
# sweep just needs to pipe stdout through the parser.
_RE_BEGIN_FRAME  = re.compile(r">>> Begin Frame:\s*(\d+)")
_RE_FRAME_ITER   = re.compile(r"Frame\s+(\d+)\s+Newton Iteration\s+(\d+)")
_RE_MAX_RES      = re.compile(r"MaxTranslationChecker[^\n]*Residual/AbsTol:\s*(\S+?)/(\S+)")
_RE_ABD_RES      = re.compile(r"ABDToleranceChecker[^\n]*Residual/AbsTol:\s*(\S+?)/(\S+)")
_RE_MIN_PT_DIST  = re.compile(r"ContactPairMinDist:\s*PT=(\S+?),\s*count=(\d+)")
_RE_CONVERGED    = re.compile(r"Newton Iteration Converged with Iteration Count:\s*(\d+)")


def _parse_float(s: str) -> float:
    """Parse a float; treat 'inf'/'nan' as math.inf / math.nan."""
    s = s.strip().rstrip(",")
    if s in ("inf", "+inf", "infinity"):
        return math.inf
    if s == "-inf":
        return -math.inf
    if s == "nan":
        return math.nan
    return float(s)


def parse_sim_log(log_path: str) -> tuple[list[dict], list[dict]]:
    """Parse the libuipc INFO-level log into per-iter rows + per-frame
    summaries.

    Returns:
        rows: list of {frame, iter, max_disp, max_disp_tol, abd_res,
              abd_tol, min_pt_dist, min_pt_dist_count}.
        frame_summaries: list of {frame, newton_iters, exit_max_disp,
              exit_max_disp_tol, exit_abd_res, exit_abd_tol,
              exit_min_pt_dist, min_pt_dist_over_frame}.
    """
    rows: list[dict] = []
    frame_summaries: list[dict] = []

    current_frame: int | None = None
    last_min_pt: float = math.inf      # min_pt at end of current iter (last-write-wins per iter)
    last_min_pt_count: int = 0
    frame_min_pt: float = math.inf     # min over all iters in the frame
    frame_rows: list[dict] = []

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = _RE_MIN_PT_DIST.search(line)
            if m:
                pt = _parse_float(m.group(1))
                last_min_pt = pt
                last_min_pt_count = int(m.group(2))
                if pt < frame_min_pt:
                    frame_min_pt = pt
                continue

            m = _RE_BEGIN_FRAME.search(line)
            if m:
                current_frame  = int(m.group(1))
                last_min_pt    = math.inf
                last_min_pt_count = 0
                frame_min_pt   = math.inf
                frame_rows     = []
                continue

            m = _RE_FRAME_ITER.search(line)
            if m:
                row = dict(
                    frame=int(m.group(1)),
                    iter=int(m.group(2)),
                    max_disp=None,
                    max_disp_tol=None,
                    abd_res=None,
                    abd_tol=None,
                    min_pt_dist=last_min_pt,
                    min_pt_dist_count=last_min_pt_count,
                )
                frame_rows.append(row)
                last_min_pt = math.inf
                last_min_pt_count = 0
                continue

            m = _RE_MAX_RES.search(line)
            if m and frame_rows:
                frame_rows[-1]["max_disp"]     = _parse_float(m.group(1))
                frame_rows[-1]["max_disp_tol"] = _parse_float(m.group(2))
                continue

            m = _RE_ABD_RES.search(line)
            if m and frame_rows:
                frame_rows[-1]["abd_res"] = _parse_float(m.group(1))
                frame_rows[-1]["abd_tol"] = _parse_float(m.group(2))
                continue

            m = _RE_CONVERGED.search(line)
            if m and current_frame is not None:
                exit_iters = int(m.group(1))
                rows.extend(frame_rows)
                if frame_rows:
                    last = frame_rows[-1]
                    frame_summaries.append(dict(
                        frame=current_frame,
                        newton_iters=exit_iters,
                        exit_max_disp=last["max_disp"],
                        exit_max_disp_tol=last["max_disp_tol"],
                        exit_abd_res=last["abd_res"],
                        exit_abd_tol=last["abd_tol"],
                        exit_min_pt_dist=last["min_pt_dist"],
                        min_pt_dist_over_frame=frame_min_pt if frame_min_pt < math.inf else None,
                    ))
                current_frame  = None
                frame_rows     = []
                last_min_pt    = math.inf
                last_min_pt_count = 0
                frame_min_pt   = math.inf
                continue

    return rows, frame_summaries


def write_metrics_csv(rows: list[dict], path: str) -> None:
    fields = ["frame", "iter", "max_disp", "max_disp_tol",
              "abd_res", "abd_tol", "min_pt_dist", "min_pt_dist_count"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_summary_json(frame_summaries: list[dict], path: str) -> None:
    safe = []
    for s in frame_summaries:
        ss = dict(s)
        for k, v in list(ss.items()):
            if isinstance(v, float) and math.isinf(v):
                ss[k] = None
        safe.append(ss)
    with open(path, "w") as fh:
        json.dump(safe, fh, indent=2)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def print_profiles():
    print("Available profiles (lowest → highest precision):\n")
    head = f"  {'name':<10s} {'LIN_TOL':>9s} {'VEL_TOL':>9s} {'TRANS_TOL':>10s} {'MAX_ITER':>9s} {'LS_MAX':>7s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, knobs in PROFILES.items():
        print(f"  {name:<10s} {knobs['LIN_TOL_RATE']:>9.1e} "
              f"{knobs['NEWTON_VELOCITY_TOL']:>9.1e} "
              f"{knobs['NEWTON_TRANSRATE_TOL']:>10.1e} "
              f"{knobs['NEWTON_MAX_ITER']:>9d} "
              f"{knobs['LINE_SEARCH_MAX_ITER']:>7d}")


def _tee_stream(src, dst_file, dst_stdout):
    """Pump bytes from `src` (subprocess.stdout) into `dst_file` AND
    `dst_stdout` line-buffered. Runs on its own thread."""
    try:
        for raw in iter(src.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            dst_file.write(text)
            dst_file.flush()
            dst_stdout.write(text)
            dst_stdout.flush()
    finally:
        try:
            src.close()
        except Exception:
            pass


def run_profile(asset: str, name: str, knobs: dict, output_root: str,
                record_every: int, record_zoom: float,
                drop_height: float | None,
                skip_encode: bool) -> tuple[bool, float]:
    out_dir   = os.path.join(output_root, name)
    png_dir   = os.path.join(out_dir, "pngs")
    mp4_path  = os.path.join(out_dir, "output.mp4")
    log_path  = os.path.join(out_dir, "sim.log")
    csv_path  = os.path.join(out_dir, "metrics.csv")
    json_path = os.path.join(out_dir, "summary.json")
    os.makedirs(png_dir, exist_ok=True)

    # LOG_LEVEL is forced to "info" so the new per-iter metric lines
    # (ContactPairMinDist, Residual/AbsTol for both checkers) are present
    # in the log. Sweep script users who want quieter output should run
    # the demo standalone instead.
    set_args = ["--set", f"RECORD_DIR={png_dir}",
                "--set", f"RECORD_EVERY={record_every}",
                "--set", f"RECORD_ZOOM={record_zoom}",
                "--set", "LOG_LEVEL=info"]
    for k, v in knobs.items():
        set_args += ["--set", f"{k}={v}"]
    if drop_height is not None:
        set_args += ["--set", f"DROP_HEIGHT={drop_height}"]

    cmd = [sys.executable, DROP_DEMO,
           "--asset", asset,
           *set_args]

    print(f"\n========== PROFILE: {name} ==========")
    print(f"  knobs: {knobs}")
    print(f"  cmd:   {' '.join(cmd)}")
    print(f"  log:   {log_path}\n", flush=True)

    t0 = time.time()
    rc: int
    with open(log_path, "wb") as log_fh:
        # Wrap log_fh to also flush text-decoded output to parent stdout.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # unbuffered — let the tee thread handle line boundaries
        )

        # Tee: subprocess stdout -> log_fh + sys.stdout (binary-safe).
        class _DualWriter:
            def __init__(self, f): self.f = f
            def write(self, text):
                self.f.write(text.encode("utf-8", errors="replace"))
            def flush(self):
                self.f.flush()

        t = threading.Thread(
            target=_tee_stream,
            args=(proc.stdout, _DualWriter(log_fh), sys.stdout),
            daemon=True,
        )
        t.start()
        rc = proc.wait()
        t.join(timeout=5.0)
    dt = time.time() - t0

    if rc != 0:
        print(f"!!! profile '{name}' exited {rc} after {dt:.0f}s !!!")
        # Still try to parse whatever logs we did capture.
    else:
        print(f"  sim done in {dt:.0f}s")

    # Parse the log → CSV + JSON
    try:
        rows, frame_summaries = parse_sim_log(log_path)
        write_metrics_csv(rows, csv_path)
        write_summary_json(frame_summaries, json_path)
        print(f"  metrics: {len(rows)} iter-rows over {len(frame_summaries)} frames")
        print(f"           → {csv_path}")
        print(f"           → {json_path}")
    except Exception as e:
        print(f"  !!! log parse failed: {e}")

    if rc != 0:
        return False, dt

    if skip_encode:
        return True, dt

    ff = ["ffmpeg", "-y", "-framerate", "30",
          "-i", os.path.join(png_dir, "frame_%06d.png"),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
          mp4_path]
    print(f"  encoding mp4: {' '.join(ff)}")
    enc_rc = subprocess.run(ff).returncode
    if enc_rc != 0:
        print(f"  !!! ffmpeg exited {enc_rc} — PNGs left in {png_dir}")
    else:
        print(f"  mp4 → {mp4_path}")
    return True, dt


def aggregate_summary(out_dir: str) -> dict | None:
    """Read <out_dir>/summary.json and compute aggregate stats."""
    path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            entries = json.load(fh)
    except Exception:
        return None
    if not entries:
        return None

    iters   = [e["newton_iters"] for e in entries if e.get("newton_iters") is not None]
    max_d   = [e["exit_max_disp"] for e in entries if e.get("exit_max_disp") is not None]
    abd     = [e["exit_abd_res"] for e in entries if e.get("exit_abd_res") is not None]
    min_pt  = [e["min_pt_dist_over_frame"] for e in entries
               if e.get("min_pt_dist_over_frame") is not None]

    def safe_median(xs): return statistics.median(xs) if xs else None
    def safe_quant(xs, q):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        k = int(round((len(xs_sorted) - 1) * q))
        return xs_sorted[k]

    return dict(
        frames=len(entries),
        avg_iters   = (sum(iters) / len(iters)) if iters else None,
        p95_iters   = safe_quant(iters, 0.95),
        max_iters   = max(iters) if iters else None,
        min_pt      = min(min_pt) if min_pt else None,
        med_max_d   = safe_median(max_d),
        med_abd     = safe_median(abd),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset", default="wound_2t_preci_3_2",
                   help="asset name under output/rcc_adhesive_tape_winding/")
    p.add_argument("--profiles", nargs="*", default=list(PROFILES),
                   help="precision profiles to run (default: all)")
    p.add_argument("--output-root", default=os.path.join(REPO_ROOT, "output",
                                                          "drop_precision_sweep"))
    p.add_argument("--record-every", type=int, default=10,
                   help="capture every N sim frames (default 10)")
    p.add_argument("--record-zoom", type=float, default=5.0,
                   help="optical zoom (FoV = 45°/zoom). Default 5 → ~9° FoV")
    p.add_argument("--drop-height", type=float, default=None,
                   help="DROP_HEIGHT in m for the drop demo (default uses preset)")
    p.add_argument("--skip-encode", action="store_true",
                   help="skip the ffmpeg step; leave PNGs only")
    p.add_argument("--list", action="store_true",
                   help="print available profiles and exit")
    args = p.parse_args()

    if args.list:
        print_profiles()
        return

    for name in args.profiles:
        if name not in PROFILES:
            print(f"unknown profile '{name}'; available: {list(PROFILES)}", file=sys.stderr)
            sys.exit(2)

    if not args.skip_encode and not have("ffmpeg"):
        print("ffmpeg not found. Install with: sudo apt-get install ffmpeg",
              file=sys.stderr)
        print("(Or pass --skip-encode to just dump PNGs.)", file=sys.stderr)
        sys.exit(3)

    print(f"output root: {args.output_root}")
    print(f"asset:       {args.asset}")
    print(f"profiles:    {args.profiles}")
    print(f"every N:     {args.record_every}")

    results = []
    for name in args.profiles:
        ok, dt = run_profile(
            args.asset, name, PROFILES[name], args.output_root,
            args.record_every, args.record_zoom,
            args.drop_height, args.skip_encode,
        )
        results.append((name, ok, dt))

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------
    print("\n========== SUMMARY ==========")
    head = (f"  {'STATUS':<6s}  {'PROFILE':<10s} {'WALL(s)':>8s}  "
            f"{'FRAMES':>6s} {'AVG_IT':>7s} {'P95_IT':>7s} {'MAX_IT':>7s}  "
            f"{'MIN_PT(m)':>11s} {'MED_MAXD':>11s} {'MED_ABD':>11s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, ok, dt in results:
        status = "OK " if ok else "FAIL"
        agg = aggregate_summary(os.path.join(args.output_root, name))
        if agg is None:
            print(f"  {status:<6s}  {name:<10s} {dt:>8.0f}  "
                  f"{'-':>6s} {'-':>7s} {'-':>7s} {'-':>7s}  "
                  f"{'-':>11s} {'-':>11s} {'-':>11s}")
            continue
        def f8(x, p=".4e"):
            return ("-" if x is None else f"{x:{p}}")
        def fi(x):
            return ("-" if x is None else f"{x:.0f}")
        print(f"  {status:<6s}  {name:<10s} {dt:>8.0f}  "
              f"{agg['frames']:>6d} {fi(agg['avg_iters']):>7s} "
              f"{fi(agg['p95_iters']):>7s} {fi(agg['max_iters']):>7s}  "
              f"{f8(agg['min_pt']):>11s} {f8(agg['med_max_d']):>11s} "
              f"{f8(agg['med_abd']):>11s}")

    print("\nPer-profile artifacts:")
    for name, ok, _dt in results:
        out_dir = os.path.join(args.output_root, name)
        print(f"  {name:<10s} → {out_dir}/  (sim.log / metrics.csv / summary.json / pngs/ / output.mp4)")


if __name__ == "__main__":
    main()
