#!/usr/bin/env python3
"""Headless cross-fixture probe for RCC bonded-PT acceleration.

Runs several RCC adhesion example fixtures with bonded-PT + skip_ccd enabled and
reports, per fixture: solver stability (is_valid throughout), bonded-lock
formation/release, and a geometry-agnostic penetration proxy -- the signed
distance from each bonded point to its own virtual-tet triangle plane (read from
RCCBondedPTStateAccessorFeature.dump_locked_tet_world_positions()).

This is an exploratory validation, NOT a pass/fail gate. Each fixture runs in its
own subprocess so engine/GPU state cannot leak between runs.

    python/.venv/bin/python scripts/probe_rcc_bonded_pt_demos.py
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np

EXAMPLES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "python", "examples")
)
sys.path.append(EXAMPLES)


DEMOS = [
    dict(
        mod="rcc_adhesive_oriented_cloth_demo",
        kwargs=dict(adhesion_on=True, oriented=True, bonded=True, skip_ccd=True,
                    beta_lock_threshold=0.9, kappa=1.0e8,
                    release_strain=0.5, release_gap=0.03),
        frames=280, checkpoints={100: "hold", 200: "lift", 279: "settled"},
    ),
    dict(
        mod="rcc_adhesive_cube_cloth_lift_release_demo",
        kwargs=dict(adhesion_on=True, bonded=True, skip_ccd=True,
                    beta_lock_threshold=0.85, kappa=5.0e7,
                    release_strain=0.5, release_gap=0.03),
        frames=400, checkpoints={100: "hold", 200: "lift", 360: "pull", 399: "settled"},
    ),
    dict(
        mod="rcc_adhesive_cloth_peel_demo",
        kwargs=dict(adhesion_on=True, bonded=True, skip_ccd=True,
                    beta_lock_threshold=0.9, kappa=1.0e8,
                    release_strain=0.6, release_gap=0.04),
        frames=280, checkpoints={30: "settle", 230: "peel", 279: "lifted"},
    ),
]


def _bonded_tet_seps(acc):
    """(min_abs_plane_sep, min_signed, max_signed, lock_count) for bonded points
    vs their own triangle plane; None if no locks."""
    pts = np.asarray(acc.dump_locked_tet_world_positions(), dtype=np.float64)
    if pts.ndim != 3 or pts.shape[0] == 0:
        return None
    p, t0, t1, t2 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    n = np.cross(t1 - t0, t2 - t0)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    sd = np.full(p.shape[0], np.nan)
    sd[ok] = np.einsum("ij,ij->i", (p - t0)[ok], n[ok]) / nn[ok]
    sdf = sd[np.isfinite(sd)]
    if sdf.size == 0:
        return (float("nan"), float("nan"), float("nan"), int(pts.shape[0]))
    return (float(np.abs(sdf).min()), float(sdf.min()), float(sdf.max()), int(pts.shape[0]))


def _surf_y(sim):
    v = np.asarray(sim["scene_io"].simplicial_surface().positions().view()).reshape(-1, 3)
    return round(float(v[:, 1].min()), 3), round(float(v[:, 1].max()), 3), round(float(v[:, 1].mean()), 3)


def _run_one(idx: int) -> None:
    import importlib
    from uipc.core import RCCBondedPTStateAccessorFeature

    d = DEMOS[idx]
    mod = importlib.import_module(d["mod"])
    sim = mod.build_demo(**d["kwargs"])
    world = sim["world"]
    acc = world.features().find(RCCBondedPTStateAccessorFeature)
    if acc is None:
        print(f"  ERROR: bonded accessor missing (rebuild pyuipc)", flush=True)
        return

    valid_through, invalid_frame = True, None
    max_locked = 0
    g_min_abs, g_min_signed = math.inf, math.inf

    while world.frame() < d["frames"]:
        world.advance()
        if not world.is_valid():
            valid_through, invalid_frame = False, world.frame()
            break
        world.retrieve()
        f = world.frame()
        max_locked = max(max_locked, acc.locked_pair_count())
        seps = _bonded_tet_seps(acc)
        if seps is not None and math.isfinite(seps[0]):
            g_min_abs = min(g_min_abs, seps[0])
            g_min_signed = min(g_min_signed, seps[1])
        if f in d["checkpoints"]:
            c = dict(acc.counters())
            sep = seps[0] if seps else None
            print(f"  [{d['checkpoints'][f]:8s} f={f:>3}] locked={acc.locked_pair_count():>3} "
                  f"released={int(c.get('released_count',0)):>3} "
                  f"cand={int(c.get('candidate_count',0)):>4} "
                  f"min_abs_sep={None if sep is None else round(sep,5)} "
                  f"surf_y={_surf_y(sim)}", flush=True)

    c = dict(acc.counters())
    print(f"  SUMMARY valid_through={valid_through} invalid_frame={invalid_frame} "
          f"max_locked={int(max_locked)} released_total={int(c.get('released_count',0))} "
          f"min_abs_sep={None if not math.isfinite(g_min_abs) else round(g_min_abs,5)} "
          f"min_signed_sep={None if not math.isfinite(g_min_signed) else round(g_min_signed,5)}",
          flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        _run_one(int(sys.argv[1]))
        return
    for i, d in enumerate(DEMOS):
        print(f"\n=== {d['mod']} (bonded + skip_ccd, kappa={d['kwargs']['kappa']:.0e}) ===", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), str(i)], check=False)


if __name__ == "__main__":
    main()
