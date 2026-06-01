#!/usr/bin/env python3
"""
Sweep adhesion coefficients (Cn = Ct = k, k ∈ {2, 4, 6, 8}) × precision
(quick, default, high) on the `temflex175-2turn-soft` preset. Builds 12
wound-tape assets via the wind demo, then re-runs each through the
drop demo. Each combo produces:

    output/rcc_adhesive_tape_winding/temflex175_2turn_soft_k{K}_{PROFILE}.npz
    output/rcc_adhesive_tape_winding_record/temflex175_2turn_soft_k{K}_{PROFILE}/
        pngs/   sim.log   output.mp4
    output/rcc_adhesive_tape_drop_record/temflex175_2turn_soft_k{K}_{PROFILE}/
        pngs/   sim.log   output.mp4

Drop precision auto-matches wind (saved in asset params via
SOLVER_PROFILE). Crash-safe: skips any combo whose asset / mp4 already
exists, so re-running resumes.

Total compute estimate at typical libuipc speeds: ~5-6 hours.

Run me:
    python python/examples/rcc_adhesive_2turn_soft_cnct_sweep.py
    # or only certain ks / profiles:
    python ... --only-k 1 2 3 --only-profile default
    # do only wind or only drop:
    python ... --skip-drop
    python ... --skip-wind
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIND_DEMO = os.path.join(REPO_ROOT, "python", "examples",
                         "rcc_adhesive_tape_winding_demo.py")
DROP_DEMO = os.path.join(REPO_ROOT, "python", "examples",
                         "rcc_adhesive_tape_drop_demo.py")
WIND_REC  = os.path.join(REPO_ROOT, "output", "rcc_adhesive_tape_winding_record")
DROP_REC  = os.path.join(REPO_ROOT, "output", "rcc_adhesive_tape_drop_record")
ASSET_DIR = os.path.join(REPO_ROOT, "output", "rcc_adhesive_tape_winding")

K_VALUES_DEFAULT = [2, 4, 6, 8]
PROFILES_DEFAULT = ["quick", "default", "high"]
PRESET           = "temflex175-2turn-soft"


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def asset_path(k: int, profile: str) -> str:
    return os.path.join(ASSET_DIR, f"temflex175_2turn_soft_k{k}_{profile}.npz")


def asset_name(k: int, profile: str) -> str:
    return f"temflex175_2turn_soft_k{k}_{profile}"


def _ffmpeg(png_dir: str, mp4_path: str) -> int:
    cmd = ["ffmpeg", "-y", "-framerate", "30",
           "-i", os.path.join(png_dir, "frame_%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           mp4_path]
    return subprocess.run(cmd,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode


def run_wind(k: int, profile: str, force: bool = False) -> tuple[bool, float]:
    name      = asset_name(k, profile)
    npz_path  = asset_path(k, profile)
    out_dir   = os.path.join(WIND_REC, name)
    png_dir   = os.path.join(out_dir, "pngs")
    mp4_path  = os.path.join(out_dir, "output.mp4")
    log_path  = os.path.join(out_dir, "sim.log")

    if not force and os.path.exists(npz_path) and os.path.exists(mp4_path):
        print(f"  [skip] wind {name}: asset + mp4 already present")
        return True, 0.0

    os.makedirs(png_dir, exist_ok=True)

    cmd = [sys.executable, "-u", WIND_DEMO,
           "--preset", PRESET,
           "--asset",  name,
           "--set", f"ADH_CN={k}",
           "--set", f"ADH_CT={k}",
           "--set", f"SOLVER_PROFILE={profile}",
           "--set", f"RECORD_DIR={png_dir}",
           "--set", "RECORD_EVERY=10",
           "--set", "VEL_LOG_EVERY=500",
           "--set", "LOG_LEVEL=warn"]

    print(f"  [wind] k={k:2d}  profile={profile:<7s}  → {name}")
    print(f"         cmd: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    with open(log_path, "wb") as logfh:
        rc = subprocess.run(cmd, stdout=logfh,
                            stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0

    if rc != 0:
        print(f"  !!! wind exited {rc} after {dt:.0f}s  (log: {log_path})")
        return False, dt

    enc_rc = _ffmpeg(png_dir, mp4_path)
    if enc_rc != 0:
        print(f"  !! ffmpeg exited {enc_rc}  (PNGs left in {png_dir})")
    print(f"         done in {dt/60:.1f} min  → {mp4_path}", flush=True)
    return True, dt


def run_drop(k: int, profile: str, force: bool = False) -> tuple[bool, float]:
    name      = asset_name(k, profile)
    npz_path  = asset_path(k, profile)
    out_dir   = os.path.join(DROP_REC, name)
    png_dir   = os.path.join(out_dir, "pngs")
    mp4_path  = os.path.join(out_dir, "output.mp4")
    log_path  = os.path.join(out_dir, "sim.log")

    if not os.path.exists(npz_path):
        print(f"  [skip] drop {name}: asset missing ({npz_path})")
        return False, 0.0
    if not force and os.path.exists(mp4_path):
        print(f"  [skip] drop {name}: mp4 already present")
        return True, 0.0

    os.makedirs(png_dir, exist_ok=True)

    # SOLVER_PROFILE is read from the asset's saved params, so drop
    # precision auto-matches wind.
    cmd = [sys.executable, "-u", DROP_DEMO,
           "--asset", name,
           "--set", f"RECORD_DIR={png_dir}",
           "--set", "RECORD_EVERY=10",
           "--set", "LOG_LEVEL=warn"]

    print(f"  [drop] k={k:2d}  profile={profile:<7s}  → {name}")
    print(f"         cmd: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    with open(log_path, "wb") as logfh:
        rc = subprocess.run(cmd, stdout=logfh,
                            stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0

    if rc != 0:
        print(f"  !!! drop exited {rc} after {dt:.0f}s  (log: {log_path})")
        return False, dt

    enc_rc = _ffmpeg(png_dir, mp4_path)
    if enc_rc != 0:
        print(f"  !! ffmpeg exited {enc_rc}  (PNGs left in {png_dir})")
    print(f"         done in {dt/60:.1f} min  → {mp4_path}", flush=True)
    return True, dt


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only-k", nargs="*", type=int, default=None,
                   help="only run these k values (default: 1..10)")
    p.add_argument("--only-profile", nargs="*", default=None,
                   choices=PROFILES_DEFAULT,
                   help="only run these profiles (default: quick default high)")
    p.add_argument("--skip-wind", action="store_true",
                   help="skip wind phase (use existing assets)")
    p.add_argument("--skip-drop", action="store_true",
                   help="skip drop phase")
    p.add_argument("--force", action="store_true",
                   help="ignore skip-if-exists and rerun every combo")
    args = p.parse_args()

    if not have("ffmpeg"):
        print("ffmpeg not found. Install with: sudo apt-get install ffmpeg",
              file=sys.stderr)
        sys.exit(2)

    ks       = args.only_k or K_VALUES_DEFAULT
    profiles = args.only_profile or PROFILES_DEFAULT

    print(f"=== 2-turn-soft Cn=Ct sweep ===")
    print(f"  ks:       {ks}")
    print(f"  profiles: {profiles}")
    print(f"  total combos: {len(ks)} × {len(profiles)} = {len(ks) * len(profiles)}")
    print(f"  skip-wind:    {args.skip_wind}")
    print(f"  skip-drop:    {args.skip_drop}")
    print(f"  force:        {args.force}")
    print()

    t_start = time.time()
    wind_done, drop_done = 0, 0
    wind_total_dt, drop_total_dt = 0.0, 0.0

    # k-outer, profile-inner so adjacent rows compare default vs high
    # for the same k.
    for k in ks:
        for profile in profiles:
            if not args.skip_wind:
                ok, dt = run_wind(k, profile, force=args.force)
                wind_total_dt += dt
                if ok:
                    wind_done += 1
            if not args.skip_drop:
                ok, dt = run_drop(k, profile, force=args.force)
                drop_total_dt += dt
                if ok:
                    drop_done += 1

    wall = time.time() - t_start
    print(f"\n=== summary ===")
    print(f"  wind: {wind_done}/{len(ks) * len(profiles)} ok  "
          f"({wind_total_dt/60:.1f} min)")
    print(f"  drop: {drop_done}/{len(ks) * len(profiles)} ok  "
          f"({drop_total_dt/60:.1f} min)")
    print(f"  wall: {wall/60:.1f} min")


if __name__ == "__main__":
    main()
