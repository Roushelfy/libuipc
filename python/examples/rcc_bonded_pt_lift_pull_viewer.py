"""
RCC bonded-PT acceleration viewer.

Same subdivided cube-cube press / hold / lift / pull scene as
``rcc_adhesive_subdivided_cube_lift_release_demo.py``, but with bonded-PT
acceleration enabled (``rcc_bonded_pt_enabled``) and the pre-CCD skip on
(``rcc_bonded_pt_skip_ccd``): stable high-beta point-triangle pairs are replaced
by a stiff ABD virtual tetrahedron and removed from CCD / contact / RCC.

The bonded virtual tets (the "ABD bonds") are drawn as a red polyscope curve
network, and live lock counts are shown in the panel. Watch the bonds appear
during press/hold, persist through the lift (the lower cube is carried purely by
the bonded energy with CCD skipped), and then disappear during the final pull as
the bond releases and the cubes separate.

Run:
    python python/examples/rcc_bonded_pt_lift_pull_viewer.py

    # Restrict bonded virtual tets to point projections inside triangle faces.
    python python/examples/rcc_bonded_pt_lift_pull_viewer.py --face-interior-vts-only
    python python/examples/rcc_bonded_pt_lift_pull_viewer.py --set LOCK_FACE_INTERIOR_ONLY=1

    # Default: allow all PT virtual tets, including edge/corner cases.
    python python/examples/rcc_bonded_pt_lift_pull_viewer.py --all-vts
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This example requires `polyscope`. Install it with `pip install polyscope`."
    ) from exc

try:
    from uipc.core import RCCBondedPTStateAccessorFeature
except ImportError as exc:
    raise SystemExit(
        "This example requires the bonded-PT accessor binding. Rebuild the "
        "Python package: `cmake --build <build> --target pyuipc`."
    ) from exc

# Reuse the proven subdivided cube-cube fixture + phase functions.
sys.path.append(os.path.dirname(__file__))
import rcc_adhesive_subdivided_cube_lift_release_demo as base  # noqa: E402

TOTAL_FRAMES = base.TOTAL_FRAMES
BETA_LOCK_THRESHOLD = 0.9
KAPPA = 1.0e8
# Default barycentric margin for the face gate (only used with
# --face-interior-vts-only). Override with `--set LOCK_FACE_MARGIN=<float>`.
LOCK_FACE_MARGIN = 0.5
# Finite release thresholds so the final pull stretches the bond past the
# normal-gap / strain limit and the lock releases (cubes separate).
RELEASE_GAP = 0.04
RELEASE_STRAIN = 0.6


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    vt_mode = p.add_mutually_exclusive_group()
    vt_mode.add_argument(
        "--face-interior-vts-only",
        action="store_true",
        help="only lock face-interior point-triangle VTs",
    )
    vt_mode.add_argument(
        "--all-vts",
        action="store_true",
        help="allow all point-triangle VTs to lock (default)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override: LOCK_FACE_INTERIOR_ONLY=0|1, LOCK_FACE_MARGIN=<float> "
             "(how far outside the triangle the foot may be, barycentric units; "
             "only used with --face-interior-vts-only)",
    )
    args = p.parse_args(argv)

    args.lock_face_margin = LOCK_FACE_MARGIN
    for kv in args.overrides:
        if "=" not in kv:
            p.error(f"--set expects KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        key = key.strip()
        if key in ("LOCK_FACE_INTERIOR_ONLY", "rcc_bonded_pt_lock_face_interior_only"):
            args.face_interior_vts_only = _parse_bool(value)
        elif key in ("LOCK_FACE_MARGIN", "rcc_bonded_pt_lock_face_margin"):
            try:
                args.lock_face_margin = float(value)
            except ValueError:
                p.error(f"LOCK_FACE_MARGIN expects a float, got {value!r}")
        else:
            p.error(f"unsupported --set key for this viewer: {key}")
    return args


def _parse_bool(value: str) -> bool:
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    try:
        return float(s) != 0.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}") from exc


def _new_sim(lock_face_interior_only: bool = False,
             lock_face_margin: float = LOCK_FACE_MARGIN):
    return base.build_demo(
        adhesion_on=True,
        bonded=True,
        skip_ccd=True,
        lock_face_interior_only=lock_face_interior_only,
        lock_face_margin=lock_face_margin,
        beta_lock_threshold=BETA_LOCK_THRESHOLD,
        kappa=KAPPA,
        release_strain=RELEASE_STRAIN,
        release_gap=RELEASE_GAP,
    )


def bonded_accessor(sim):
    return sim["world"].features().find(RCCBondedPTStateAccessorFeature)


def bond_geometry(acc):
    """Return (nodes [4M,3], edges [E,2]) for the locked virtual tets, or None."""
    if acc is None:
        return None
    pts = np.asarray(acc.dump_locked_tet_world_positions(), dtype=np.float64)
    if pts.ndim != 3 or pts.shape[0] == 0:
        return None
    m = pts.shape[0]
    nodes = pts.reshape(-1, 3)
    edges = []
    for i in range(m):
        b = 4 * i
        p, t0, t1, t2 = b, b + 1, b + 2, b + 3
        # the bond (point -> each triangle vertex) plus the triangle outline
        edges += [[p, t0], [p, t1], [p, t2], [t0, t1], [t1, t2], [t2, t0]]
    return nodes, np.asarray(edges, dtype=np.int64)


def run_demo():
    args = parse_cli()
    state = {"face_interior_only": bool(args.face_interior_vts_only),
             "lock_face_margin": float(args.lock_face_margin)}
    sim = _new_sim(state["face_interior_only"], state["lock_face_margin"])

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    try:
        ps.set_transparency_mode("pretty")
    except Exception:
        pass

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "rcc_bonded_cubes",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(1.0)
    try:
        mesh.set_transparency(0.55)
    except Exception:
        pass

    ui = {"run": False}

    def update_bonds() -> int:
        bg = bond_geometry(bonded_accessor(sim))
        if ps.has_curve_network("rcc_bonded_pt_bonds"):
            ps.remove_curve_network("rcc_bonded_pt_bonds")
        if bg is None:
            return 0
        nodes, edges = bg
        net = ps.register_curve_network("rcc_bonded_pt_bonds", nodes, edges)
        net.set_radius(0.004)
        net.set_color((1.0, 0.15, 0.1))
        return nodes.shape[0] // 4

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        verts = merged.positions().view().reshape(-1, 3)
        tris = merged.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != verts.shape[0]:
            ps.remove_surface_mesh("rcc_bonded_cubes")
            mesh = ps.register_surface_mesh("rcc_bonded_cubes", verts, tris)
            mesh.set_edge_width(1.0)
            try:
                mesh.set_transparency(0.55)
            except Exception:
                pass
        else:
            mesh.update_vertex_positions(verts)
        update_bonds()

    def step_once():
        if sim["world"].frame() >= TOTAL_FRAMES:
            ui["run"] = False
            return
        sim["world"].advance()
        if not sim["world"].is_valid():
            ui["run"] = False
            return
        sim["world"].retrieve()
        update_visual()

    def reset():
        nonlocal sim
        sim = _new_sim(state["face_interior_only"], state["lock_face_margin"])
        update_visual()

    def on_update():
        if psim.Button("run / pause"):
            ui["run"] = not ui["run"]
        psim.SameLine()
        if psim.Button("step"):
            step_once()
        psim.SameLine()
        if psim.Button("reset"):
            reset()
        psim.SameLine()
        vt_label = "face-only" if state["face_interior_only"] else "all"
        if psim.Button(f"VTs: {vt_label}"):
            state["face_interior_only"] = not state["face_interior_only"]
            ui["run"] = False
            reset()

        if ui["run"]:
            step_once()

        frame = min(sim["world"].frame(), TOTAL_FRAMES)
        target_y, phase = base.picker_y(frame)
        bottom_y, top_y, gap_y = base.cube_height_stats(sim)
        contact = base.cube_contact_gap_stats(sim)

        acc = bonded_accessor(sim)
        locked = int(acc.locked_pair_count()) if acc is not None else 0
        counters = acc.counters() if acc is not None else {}

        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}   Phase: {phase}")
        psim.Text(f"Picker target Y: {target_y:+.3f}")
        psim.Text(f"Bottom avg Y: {bottom_y:+.3f}   gap Y: {gap_y:+.3f}")
        psim.Text(
            "Contact gap: "
            f"min={contact['gap_min']:+.4f} mean={contact['gap_mean']:+.4f} "
            "(>=0 means no penetration)"
        )
        psim.Separator()
        psim.Text(f"Bonded locks (red bonds): {locked}")
        psim.Text(
            "VT lock mode: "
            f"{'face-interior only' if state['face_interior_only'] else 'all VTs'}"
        )
        if counters:
            psim.Text(
                f"candidates={counters.get('candidate_count', 0)} "
                f"released={counters.get('released_count', 0)} "
                f"filter_skipped={counters.get('filter_skipped_count', 0)} "
                f"dup={counters.get('duplicate_suppressed_count', 0)}"
            )
        psim.TextWrapped(
            "Bonded PT pairs are skipped from CCD/contact/RCC and held by a stiff "
            "ABD virtual tet. Expect locks to grow during press/hold, persist "
            "through the lift, and release during the final pull."
        )
        if frame >= TOTAL_FRAMES:
            psim.Text("Sim done. Hit `reset` to rebuild.")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
