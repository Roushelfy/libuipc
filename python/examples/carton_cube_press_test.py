#!/usr/bin/env python3
"""carton_cube_press_test.py — NO-TAPE motion test of the SPINDLE + press-CUBE.

Drives the carton (flaps close per the seal choreography), the SPINDLE (a
roll-sized cylinder, on the real _spindle_center path) and the press-CUBE (on the
real _cube_center path) — but with NO tape — so the two trajectories and their
STRICT ALTERNATION (spindle and cube never move at the same time) can be checked
cheaply before committing to a full tape run.

Everything is driven by the SAME trajectory functions the tape demo uses
(imported from carton_tape_seal_v2_demo), so what you see here is exactly the
motion the real run will follow.
"""

import os
import sys

import numpy as np

from uipc import Logger, Engine, World, Scene, SceneIO, view, builtin
from uipc.geometry import linemesh
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyRevoluteJoint,
    AffineBodyDrivingRevoluteJoint,
    AffineBodyRevoluteJointLimit,
    SoftTransformConstraint,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir

import carton_tape_seal_v2_demo as V   # trajectories (_spindle_center, _cube_center), schedule, helpers
from carton_articulated_close_demo import (
    _box_VF, _make_sc, _build_base_mesh, _build_flap_mesh, _hinge_edge,
    _coord_out, _tilt_transform,
    FLAPS, TILT_DEG, ABD_KAPPA, JOINT_STRENGTH, DRIVE_STRENGTH, LIMIT_STRENGTH,
    OUT_DEG, PAIR1_IN_DEG, PAIR2_IN_DEG, D_HAT, HALF, BASE,
)

# spindle stand-in (a roll-sized cylinder so clearances read realistically)
SPIN_R, SPIN_LEN = 0.042, 0.05


def _cube_phase(f):
    if f < V.S3:          return "1:spindle→−Z (cube static)"
    if f < V.S2b:         return "2:CUBE −Z press (spindle static)"
    if f < V.S5:          return "3:spindle full run (cube static)"
    return "4:CUBE →+Z press + leave (spindle static)"


def build_demo(cfg: dict):
    def _cf(k, d):  return float(cfg.get(k, d))

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = D_HAT
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
    scene = Scene(config)

    abd = AffineBodyConstitution()
    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)

    # ---- carton BASE (fixed) -----------------------------------------
    base_sc = _build_base_mesh()
    base_sc.instances().resize(1)
    abd.apply_to(base_sc, ABD_KAPPA)
    view(base_sc.instances().find(builtin.is_fixed))[:] = 1
    carton_elem = tabular.create("carton"); carton_elem.apply_to(base_sc)
    base_obj = scene.objects().create("carton_base")
    base_slot, _ = base_obj.geometries().create(base_sc)

    # ---- carton FLAPS ------------------------------------------------
    tilt_rad = np.radians(TILT_DEG)
    flap_slots, flap_scs = [], []
    flap_objs = scene.objects().create("carton_flaps")
    for (name, axis, hin, coord, _kind, sign) in FLAPS:
        coord_o = _coord_out(coord)
        fsc = _build_flap_mesh(axis, hin, coord_o)
        fsc.instances().resize(1)
        abd.apply_to(fsc, ABD_KAPPA)
        view(fsc.instances().find(builtin.is_fixed))[:] = 0
        carton_elem.apply_to(fsc)
        if tilt_rad != 0.0:
            view(fsc.transforms())[0] = _tilt_transform(axis, hin, coord_o, -sign * tilt_rad)
        slot, _ = flap_objs.geometries().create(fsc)
        flap_slots.append(slot); flap_scs.append(fsc)

    # ---- SPINDLE (roll-sized cylinder, STC-driven along _spindle_center) --
    spin_sc = V._make_rod_abd_sc(SPIN_R, SPIN_LEN, 20, (0.0, 0.0, 0.0))
    spin_sc.instances().resize(1)
    abd.apply_to(spin_sc, 1.0e8, 1000.0)
    spin_elem = tabular.create("spindle"); spin_elem.apply_to(spin_sc)
    SoftTransformConstraint().apply_to(spin_sc, np.array([1.0e8, 1.0e8]))
    Ms0 = np.eye(4); Ms0[:3, :3] = V._RY90; Ms0[:3, 3] = V._spindle_center(0)
    view(spin_sc.transforms())[0] = V._mat4_to_uipc(Ms0)
    spin_obj = scene.objects().create("spindle")
    spin_slot, _ = spin_obj.geometries().create(spin_sc)

    # ---- PRESS-CUBE (STC-driven along _cube_center) ------------------
    cube_sc = _make_sc(*_box_VF((0.0, 0.0, 0.0), (2 * V.CUBE_HX, 2 * V.CUBE_HY, 2 * V.CUBE_HZ)))
    cube_sc.instances().resize(1)
    abd.apply_to(cube_sc, V.CUBE_KAPPA, V.CUBE_DENSITY)
    cube_elem = tabular.create("cube"); cube_elem.apply_to(cube_sc)
    SoftTransformConstraint().apply_to(cube_sc, np.array([V.CUBE_STC_ETA, V.CUBE_STC_ETA]))
    Mc0 = np.eye(4); Mc0[:3, 3] = V._cube_center(0)
    view(cube_sc.transforms())[0] = V._mat4_to_uipc(Mc0)
    cube_obj = scene.objects().create("press_cube")
    cube_slot, _ = cube_obj.geometries().create(cube_sc)

    # ---- carton hinge joints + flap-closing choreography (÷100 limit) -
    jV, jE = [], []
    for i, (name, axis, hin, coord, _kind, _sign) in enumerate(FLAPS):
        p0, p1 = _hinge_edge(axis, hin, _coord_out(coord))
        jV += [p0, p1]; jE.append([2 * i, 2 * i + 1])
    joint_mesh = linemesh(np.array(jV, dtype=np.float32), np.array(jE, dtype=np.int32))
    AffineBodyRevoluteJoint().apply_to(joint_mesh, [base_slot] * len(FLAPS), flap_slots, JOINT_STRENGTH)
    signs = np.array([f[5] for f in FLAPS], dtype=np.float64)
    ia = joint_mesh.edges().find("init_angle")
    if ia is not None: view(ia)[:] = -signs * tilt_rad
    AffineBodyDrivingRevoluteJoint().apply_to(joint_mesh, np.full(len(FLAPS), DRIVE_STRENGTH))
    isp = joint_mesh.edges().find("is_passive")
    if isp is not None: view(isp)[:] = 0
    phi_lo = np.array([-(PAIR1_IN_DEG if f[1] == "X" else PAIR2_IN_DEG) for f in FLAPS])
    b_lo = -signs * np.radians(phi_lo); b_hi = -signs * np.radians(np.full(len(FLAPS), +OUT_DEG))
    flap_limit = _cf("LIMIT_STRENGTH", LIMIT_STRENGTH * 0.01)
    AffineBodyRevoluteJointLimit().apply_to(joint_mesh, np.minimum(b_lo, b_hi),
                                            np.maximum(b_lo, b_hi), np.full(len(FLAPS), flap_limit))
    joint_obj = scene.objects().create("hinges")
    joint_obj.geometries().create(joint_mesh)

    def drive_flaps(info):
        geo = info.geo_slots()[0].geometry()
        c, a = V._seal_choreo(info.frame())
        view(geo.edges().find("driving/is_constrained"))[:] = c
        view(geo.edges().find("aim_angle"))[:] = a
    scene.animator().insert(joint_obj, drive_flaps)

    def animate_spindle(info):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        M = np.eye(4); M[:3, :3] = V._RY90; M[:3, 3] = V._spindle_center(f)
        view(geo.instances().find(builtin.aim_transform))[0] = V._mat4_to_uipc(M)
    scene.animator().insert(spin_obj, animate_spindle)

    def animate_cube(info):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        M = np.eye(4); M[:3, 3] = V._cube_center(f)
        view(geo.instances().find(builtin.aim_transform))[0] = V._mat4_to_uipc(M)
    scene.animator().insert(cube_obj, animate_cube)

    world.init(scene)
    return {"engine": engine, "world": world, "scene": scene, "scene_io": SceneIO(scene),
            "cube_slot": cube_slot, "spin_slot": spin_slot, "world_": world}


def run():
    cfg = V._parse_set_args()
    Logger.set_level(Logger.Level.Warn)
    sim = build_demo(cfg)
    total = int(cfg.get("FRAMES", V._cube_keyframes()[-1][0] + 40))
    record_dir = cfg.get("RECORD_DIR") or os.path.join(AssetDir.output_path(__file__), "frames")
    every_n = int(cfg.get("RECORD_EVERY", 10))
    zoom = float(cfg.get("RECORD_ZOOM", 1.0))

    gq_v = np.array([[-0.6, 0, -0.6], [0.6, 0, -0.6], [0.6, 0, 0.6], [-0.6, 0, 0.6]], dtype=np.float64)
    gq_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    def setup_extras(ps):
        gm = ps.register_surface_mesh("ground", gq_v, gq_f)
        gm.set_color((0.55, 0.55, 0.58)); gm.set_transparency(0.5)

    def on_progress(f, t):
        sc = np.asarray(view(sim["spin_slot"].geometry().transforms())[0]).reshape(4, 4)[:3, 3]
        cc = np.asarray(view(sim["cube_slot"].geometry().transforms())[0]).reshape(4, 4)[:3, 3]
        print(f"[motion] frame {f}/{t} ({f/t*100:.1f}%) {_cube_phase(max(f-1,0))} "
              f"spindle=({sc[0]:+.3f},{sc[1]:+.3f},{sc[2]:+.3f}) "
              f"cube=({cc[0]:+.3f},{cc[1]:+.3f},{cc[2]:+.3f})", flush=True)

    print(f"[motion] NO-TAPE spindle+cube motion test. phases@[descend-end S3={V.S3}, "
          f"stick-end S2b={V.S2b}, run-end S5={V.S5}]. TOTAL={total} -> {record_dir}", flush=True)
    L.record_demo_to_pngs(sim=sim, total_frames=total, output_dir=record_dir,
                          every_n=every_n, up_dir="y_up", mesh_name="carton_motion",
                          setup_extras_fn=setup_extras, zoom=zoom, on_progress=on_progress,
                          bbox_extent_override=max(BASE * 2.0, 0.75))


if __name__ == "__main__":
    run()
