#!/usr/bin/env python3
"""
Sweep the drop demo across a list of solver-precision profiles, saving
each run as a separate MP4. Useful for `which solver tol settings do I
actually need?` ablation.

Each profile runs the SAME loaded wound-tape asset through the SAME
drop physics — only the libuipc solver knobs change. Output structure:

    <output-root>/
      quick/    pngs/frame_NNNNNN.png   output.mp4
      default/  pngs/...                output.mp4
      high/     ...
      extreme/  ...
      paranoid/ ...

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
import os
import shutil
import subprocess
import sys
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
        LIN_TOL_RATE=1e-5,
        NEWTON_VELOCITY_TOL=5e-4,
        NEWTON_TRANSRATE_TOL=1e-3,
        NEWTON_MAX_ITER=2048,
        LINE_SEARCH_MAX_ITER=16,
    ),
    "extreme": dict(
        LIN_TOL_RATE=1e-6,
        NEWTON_VELOCITY_TOL=5e-5,
        NEWTON_TRANSRATE_TOL=1e-4,
        NEWTON_MAX_ITER=4096,
        LINE_SEARCH_MAX_ITER=32,
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


def run_profile(asset: str, name: str, knobs: dict, output_root: str,
                record_every: int, record_zoom: float, log_level: str,
                drop_height: float | None,
                skip_encode: bool) -> tuple[bool, float]:
    out_dir  = os.path.join(output_root, name)
    png_dir  = os.path.join(out_dir, "pngs")
    mp4_path = os.path.join(out_dir, "output.mp4")
    os.makedirs(png_dir, exist_ok=True)

    set_args = ["--set", f"RECORD_DIR={png_dir}",
                "--set", f"RECORD_EVERY={record_every}",
                "--set", f"RECORD_ZOOM={record_zoom}",
                "--set", f"LOG_LEVEL={log_level}"]
    for k, v in knobs.items():
        set_args += ["--set", f"{k}={v}"]
    if drop_height is not None:
        set_args += ["--set", f"DROP_HEIGHT={drop_height}"]

    cmd = [sys.executable, DROP_DEMO,
           "--asset", asset,
           *set_args]

    print(f"\n========== PROFILE: {name} ==========")
    print(f"  knobs: {knobs}")
    print(f"  cmd:   {' '.join(cmd)}\n", flush=True)

    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0

    if rc != 0:
        print(f"!!! profile '{name}' exited {rc} after {dt:.0f}s !!!")
        return False, dt

    print(f"  sim done in {dt:.0f}s")

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
    p.add_argument("--log-level", default="warn",
                   help="libuipc log level (warn / info / debug)")
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
            args.record_every, args.record_zoom, args.log_level,
            args.drop_height, args.skip_encode,
        )
        results.append((name, ok, dt))

    print("\n========== SUMMARY ==========")
    for name, ok, dt in results:
        status = "OK " if ok else "FAIL"
        print(f"  {status}  {name:<10s} {dt:>7.0f}s  →  "
              f"{os.path.join(args.output_root, name, 'output.mp4')}")


if __name__ == "__main__":
    main()
