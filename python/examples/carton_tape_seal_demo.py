"""
carton_tape_seal_demo.py — mime taping a carton shut.

Builds on:
  - carton_articulated_close_demo.py : the articulated cardboard carton +
    its flap geometry/joints (imported wholesale; the close schedule is
    re-authored here so the OUTER pair is held pressed, not released).
  - rcc_adhesive_tape_rod_wind_demo.py : the rigid ABD spindle + the
    SoftTransformConstraint (STC) "drive a rigid body along a path" pattern.

Flap layering / which seam we tape:
  - The ±Z (FRONT) pair (hinge axis X) closes flat FIRST (act B) and is held
    shut by its 90° limit — the bottom layer.
  - The ±X (BACK / OUTER) pair (hinge axis Z, taller walls) folds ON TOP and
    is PRESSED CLOSED and HELD by its motor for the whole run (NOT released
    back to 60° as in the carton demo). The two outer flaps meet along the
    Z axis at x=0 — THIS is the seam we tape.
  - Seam: runs along Z at x=0, top y≈HIN_OUTER; endpoints z=±FLAP_W/2.

Sequence (spindle-only v1, NO TAPE — kinematic trajectory preview):
  1. CLOSE  — front pair flat, back pair pressed flat & held.
  2. APPROACH — spindle (axis ∥ X, i.e. PERPENDICULAR to the Z-seam) moves to
     just above one END of the seam (the −Z end).
  3. RUN    — translates slowly along +Z to the far (+Z) end, riding just
     above the seam ("lay tape over the seam").
  4. CROSS+DOWN — crosses past the +Z edge and descends the +Z side face.
spindle↔carton contact is DISABLED in v1 so the path is a clean preview.

Run (inside a gs-srun GPU container):
    python python/examples/carton_tape_seal_demo.py \
        --set RECORD_DIR=/mnt/home/zhaofeng/workspace/libuipc/output/carton_seal
"""

from __future__ import annotations

import os
import sys

import numpy as np

from uipc import (
    Logger, Matrix4x4, Engine, World, Scene, SceneIO, Animation, view, builtin,
)
from uipc.geometry import trimesh, linemesh, label_surface
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

import carton_articulated_close_demo as C
from carton_articulated_close_demo import (
    _build_base_mesh, _build_flap_mesh, _hinge_edge, _coord_out, _tilt_transform, _ramp,
    FLAPS, TILT_DEG, ABD_KAPPA, JOINT_STRENGTH, DRIVE_STRENGTH, LIMIT_STRENGTH,
    OUT_DEG, PAIR1_IN_DEG, PAIR2_IN_DEG, CLOSED_DEG,
    A_END, B_END, C_END, PRESS_END,
    D_HAT, THK, WALL_H, OFFSET, BASE, HALF, HIN_INNER, HIN_OUTER, ZC, FLAP_W, INCH,
)


# ----------------------------------------------------------------------
# Spindle helpers (copied from rcc_adhesive_tape_rod_wind_demo.py)
# ----------------------------------------------------------------------
def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


def _make_rod_abd_sc(R, length, n_sides, center):
    """Closed cylinder trimesh, axis +z, centered at `center`."""
    cx, cy, cz = center
    z0, z1 = cz - 0.5 * length, cz + 0.5 * length
    ang = np.arange(n_sides) * (2.0 * np.pi / n_sides)
    ring = np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], axis=1)
    v0 = np.column_stack([ring, np.full(n_sides, z0)])
    v1 = np.column_stack([ring, np.full(n_sides, z1)])
    verts = np.vstack([v0, v1, [[cx, cy, z0]], [[cx, cy, z1]]]).astype(np.float64)
    ic0, ic1 = 2 * n_sides, 2 * n_sides + 1
    tris = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        a, b = i, j
        c, d = n_sides + i, n_sides + j
        tris += [[a, b, d], [a, d, c]]
        tris += [[b, a, ic0], [c, d, ic1]]
    tris = np.asarray(tris, dtype=np.int32)
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() < 0.0:
        tris = tris[:, ::-1].copy()
    sc = trimesh(verts, tris)
    label_surface(sc)
    return sc


# rotate the cylinder's native +Z axis to +X (so the spindle is perpendicular
# to the Z-running seam). R_y(90°): (0,0,1) -> (1,0,0).
_RY90 = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]], dtype=np.float64)


# ----------------------------------------------------------------------
# Re-authored close schedule: front (±Z) pair closes flat then is limit-held;
# BACK (±X) pair is pushed in, pressed flat, and HELD pressed (never released).
# ----------------------------------------------------------------------
CLOSE_X_DEG = 90.0     # back-pair pressed-flat hold angle (deg from vertical)


def _seal_choreo(frame):
    """Per-edge (is_constrained[4], aim_rad[4]). aim_rad = −sign·radians(φ)."""
    is_c = np.zeros(len(FLAPS), dtype=np.int32)
    aim  = np.zeros(len(FLAPS), dtype=np.float64)
    for i, (name, axis, hin, coord, kind, sign) in enumerate(FLAPS):
        driven, phi = False, 0.0
        if axis == "X":                                    # ±Z FRONT pair
            if A_END <= frame < B_END:                     # close flat, then limit-held
                driven, phi = True, _ramp(frame, A_END, B_END, +OUT_DEG, -CLOSED_DEG)
        else:                                              # ±X BACK pair (held pressed)
            if B_END <= frame < C_END:
                driven, phi = True, _ramp(frame, B_END, C_END, +OUT_DEG, -PAIR2_IN_DEG)
            elif C_END <= frame < PRESS_END:               # press from 60° to flat
                driven, phi = True, _ramp(frame, C_END, PRESS_END, -PAIR2_IN_DEG, -CLOSE_X_DEG)
            elif frame >= PRESS_END:                        # HOLD pressed flat (no release)
                driven, phi = True, -CLOSE_X_DEG
        if driven:
            is_c[i] = 1
            aim[i]  = -sign * np.radians(phi)
    return is_c, aim


# ----------------------------------------------------------------------
# Spindle geometry + taping trajectory (BACK-pair seam: along Z at x=0).
# ----------------------------------------------------------------------
SP_R       = 0.005
SP_LEN     = 0.06
SP_SIDES   = 16
SP_KAPPA   = 1.0e8
SP_DENSITY = 1000.0
STC_ETA    = 1.0e8

SEAM_X   = 0.0                              # seam at x=0
SEAM_Y   = HIN_OUTER + 0.5 * THK            # top of the flat back-pair (~0.214)
SEAM_Z0  = -FLAP_W / 2.0                    # start end (−Z), ≈ −0.146
SEAM_Z1  = +FLAP_W / 2.0                    # far end   (+Z), ≈ +0.146
# Leave roughly a tape-width of room between the spindle and the seam/side face
# so the tape (later wound on the spindle) has space to feed down onto the box.
TAPE_SPACE = 0.045                          # ≈ scotch tape width
CLEAR    = TAPE_SPACE                       # spindle ride height above the seam
SIDE_Z   = HALF + TAPE_SPACE                # spindle offset outboard of the +Z face
DOWN_Y   = 0.5 * (THK + WALL_H)             # how far down the +Z face

SP_PAD      = 50
SP_APPROACH = 80
SP_DESCEND  = 60
SP_RUN      = 240
SP_CROSS    = 100
SP_END      = 40

# Carton is closed once the back pair is pressed flat (PRESS_END) + a short
# settle; no guided-return phase here, so this is well before the carton demo's
# 510. Give a settle margin before the spindle starts.
CLOSE_DONE = PRESS_END + 70
S0 = CLOSE_DONE
S1 = S0 + SP_PAD
S2 = S1 + SP_APPROACH
S3 = S2 + SP_DESCEND
S4 = S3 + SP_RUN
S4b = S4 + SP_CROSS // 2
S5 = S4 + SP_CROSS
TOTAL_FRAMES = S5 + SP_END

PARK = np.array([0.0, SEAM_Y + 0.24, 0.0])
_W1  = np.array([SEAM_X, SEAM_Y + CLEAR + 0.06, SEAM_Z0])   # high above −Z end
_W2  = np.array([SEAM_X, SEAM_Y + CLEAR, SEAM_Z0])          # ride height, −Z end
_W3  = np.array([SEAM_X, SEAM_Y + CLEAR, SEAM_Z1])          # run to +Z end
_W4  = np.array([SEAM_X, SEAM_Y + CLEAR, SIDE_Z])           # cross past +Z edge
_W5  = np.array([SEAM_X, DOWN_Y, SIDE_Z])                   # down the +Z face

_KEYS = [(S1, PARK), (S2, _W1), (S3, _W2), (S4, _W3), (S4b, _W4), (S5, _W5),
         (TOTAL_FRAMES, _W5)]


def _spindle_center(f: int) -> np.ndarray:
    if f <= _KEYS[0][0]:
        return _KEYS[0][1].copy()
    for (fa, pa), (fb, pb) in zip(_KEYS, _KEYS[1:]):
        if f <= fb:
            return np.array([_ramp(f, fa, fb, pa[k], pb[k]) for k in range(3)])
    return _KEYS[-1][1].copy()


def _phase(f: int) -> str:
    if f < S0:  return "close"
    if f < S1:  return "settle"
    if f < S2:  return "approach"
    if f < S3:  return "descend"
    if f < S4:  return "run-seam"
    if f < S5:  return "cross+down"
    return "hold"


# ----------------------------------------------------------------------
def build_demo(cfg: dict):
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
    tilt_rad = np.radians(TILT_DEG)

    drive_strengths = np.full(len(FLAPS), DRIVE_STRENGTH, dtype=np.float64)
    limit_strengths = np.full(len(FLAPS), LIMIT_STRENGTH, dtype=np.float64)
    phi_lo = np.array([-(PAIR1_IN_DEG if f[1] == "X" else PAIR2_IN_DEG) for f in FLAPS])
    phi_hi = np.full(len(FLAPS), +OUT_DEG)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    default_elem = tabular.default_element()

    # ---- carton BASE (fixed) -----------------------------------------
    base_sc = _build_base_mesh()
    base_sc.instances().resize(1)
    abd.apply_to(base_sc, ABD_KAPPA)
    view(base_sc.instances().find(builtin.is_fixed))[:] = 1
    base_obj = scene.objects().create("carton_base")
    base_slot, _ = base_obj.geometries().create(base_sc)

    # ---- carton FLAPS (free, hinged) ---------------------------------
    flap_slots = []
    flap_objs = scene.objects().create("carton_flaps")
    for (name, axis, hin, coord, _kind, sign) in FLAPS:
        coord_o = _coord_out(coord)
        fsc = _build_flap_mesh(axis, hin, coord_o)
        fsc.instances().resize(1)
        abd.apply_to(fsc, ABD_KAPPA)
        view(fsc.instances().find(builtin.is_fixed))[:] = 0
        if tilt_rad != 0.0:
            view(fsc.transforms())[0] = _tilt_transform(axis, hin, coord_o, -sign * tilt_rad)
        slot, _ = flap_objs.geometries().create(fsc)
        flap_slots.append(slot)

    # ---- hinge joints (revolute + driving motor + soft limit) --------
    jV, jE = [], []
    for i, (name, axis, hin, coord, _kind, _sign) in enumerate(FLAPS):
        p0, p1 = _hinge_edge(axis, hin, _coord_out(coord))
        jV.append(p0); jV.append(p1)
        jE.append([2 * i, 2 * i + 1])
    joint_mesh = linemesh(np.array(jV, dtype=np.float32), np.array(jE, dtype=np.int32))

    revolute = AffineBodyRevoluteJoint()
    revolute.apply_to(joint_mesh, [base_slot] * len(FLAPS), flap_slots, JOINT_STRENGTH)

    signs = np.array([f[5] for f in FLAPS], dtype=np.float64)
    init_angle = joint_mesh.edges().find("init_angle")
    if init_angle is not None:
        view(init_angle)[:] = -signs * tilt_rad

    driving = AffineBodyDrivingRevoluteJoint()
    driving.apply_to(joint_mesh, drive_strengths)
    is_passive = joint_mesh.edges().find("is_passive")
    if is_passive is not None:
        view(is_passive)[:] = 0

    b_lo = -signs * np.radians(phi_lo)
    b_hi = -signs * np.radians(phi_hi)
    limit = AffineBodyRevoluteJointLimit()
    limit.apply_to(joint_mesh, np.minimum(b_lo, b_hi), np.maximum(b_lo, b_hi), limit_strengths)

    joint_obj = scene.objects().create("hinges")
    joint_slot, _ = joint_obj.geometries().create(joint_mesh)

    def drive_flaps(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_c = view(geo.edges().find("driving/is_constrained"))
        aim = view(geo.edges().find("aim_angle"))
        c, a = _seal_choreo(info.frame())
        is_c[:] = c
        aim[:] = a

    scene.animator().insert(joint_obj, drive_flaps)

    # ---- SPINDLE (rigid ABD cylinder, axis ∥ X, STC-driven) ----------
    # Baked at origin; held with axis along X (R_y(90°)). v1: NO tape and
    # spindle↔carton contact DISABLED (clean kinematic preview).
    spindle_sc = _make_rod_abd_sc(SP_R, SP_LEN, SP_SIDES, (0.0, 0.0, 0.0))
    abd.apply_to(spindle_sc, SP_KAPPA, SP_DENSITY)
    spindle_elem = tabular.create("spindle")
    spindle_elem.apply_to(spindle_sc)
    tabular.insert(spindle_elem, default_elem, 0.5, 1.0e9, enable=False)
    stc = SoftTransformConstraint()
    stc.apply_to(spindle_sc, np.array([STC_ETA, STC_ETA], dtype=np.float64))
    # start it parked (axis ∥ X) so it doesn't fly out of the box origin
    M0 = np.eye(4, dtype=np.float64)
    M0[:3, :3] = _RY90
    M0[:3, 3] = PARK
    view(spindle_sc.transforms())[0] = _mat4_to_uipc(M0)
    spindle_obj = scene.objects().create("spindle")
    spindle_slot, _ = spindle_obj.geometries().create(spindle_sc)

    def animate_spindle(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.instances().find(builtin.is_constrained))
        aim = view(geo.instances().find(builtin.aim_transform))
        is_c[0] = 1
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = _RY90
        M[:3, 3] = _spindle_center(f)   # origin-baked ⇒ b = c
        aim[0] = _mat4_to_uipc(M)

    scene.animator().insert(spindle_obj, animate_spindle)

    world.init(scene)

    return {
        "engine": engine, "world": world, "scene": scene,
        "scene_io": SceneIO(scene),
        "base_slot": base_slot, "flap_slots": flap_slots,
        "joint_slot": joint_slot, "spindle_slot": spindle_slot,
    }


def _parse_set_args():
    argv = list(sys.argv[1:])
    cfg, i = {}, 0
    while i < len(argv):
        if argv[i] == "--set" and i + 1 < len(argv) and "=" in argv[i + 1]:
            k, v = argv[i + 1].split("=", 1)
            cfg[k.strip()] = v.strip()
            i += 2
        else:
            i += 1
    return cfg


def run():
    cfg = _parse_set_args()
    Logger.set_level(Logger.Level.Warn)

    sim = build_demo(cfg)
    total_frames = int(cfg.get("FRAMES", TOTAL_FRAMES))
    record_dir = cfg.get("RECORD_DIR") or os.path.join(
        AssetDir.output_path(__file__), "frames")
    every_n = int(cfg.get("RECORD_EVERY", 5))
    zoom = float(cfg.get("RECORD_ZOOM", 1.1))

    gq_v = np.array([[-0.4, 0.0, -0.4], [0.4, 0.0, -0.4],
                     [0.4, 0.0, 0.4], [-0.4, 0.0, 0.4]], dtype=np.float64)
    gq_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def setup_extras(ps_mod):
        gm = ps_mod.register_surface_mesh("ground", gq_v, gq_f)
        gm.set_color((0.55, 0.55, 0.58))
        gm.set_transparency(0.5)

    def on_progress(f, t):
        sp = sim["spindle_slot"].geometry()
        c = np.asarray(view(sp.transforms())[0]).reshape(4, 4)[:3, 3]
        print(f"[seal] frame {f}/{t} ({f/t*100:.1f}%)  phase={_phase(max(f-1,0))}  "
              f"spindle=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})", flush=True)

    print(f"[seal] back-pair seam: Z∈[{SEAM_Z0:+.3f},{SEAM_Z1:+.3f}] x=0 y={SEAM_Y:.4f}; "
          f"back pair HELD pressed flat ({CLOSE_X_DEG:.0f}°) from frame {PRESS_END}. "
          f"phases@[S0{S0},S1{S1},S2{S2},S3{S3},S4{S4},S4b{S4b},S5{S5}] TOTAL={total_frames}. "
          f"recording → {record_dir}")

    L.record_demo_to_pngs(
        sim=sim, total_frames=total_frames, output_dir=record_dir,
        every_n=every_n, up_dir="y_up", mesh_name="carton_seal",
        setup_extras_fn=setup_extras, zoom=zoom, on_progress=on_progress,
        bbox_extent_override=max(BASE * 1.5, 0.5))


if __name__ == "__main__":
    run()
