"""
Articulated cardboard carton — gravity-droop flap demo.

A paper box modelled with affine-rigid bodies (ABD):
  - BASE  = closed bottom + 4 side walls, merged into ONE fixed rigid
            body (its sub-plates never self-collide, so overlaps at the
            corners are harmless).
  - 4 TOP FLAPS, each its own ABD body, hinged to the top edge of a
    wall by an AffineBodyRevoluteJoint. The hinge axis is free; gravity
    swings each flap and a soft AffineBodyRevoluteJointLimit catches it.

Geometry follows a real carton:
  - 12 in × 12 in footprint, 8 in tall.
  - The hinge / flap of each wall is shifted slightly OUTBOARD of the
    wall (HINGE_OUT) so the flap can swing fully down the OUTSIDE of the
    wall (to ~180°) instead of jamming on the wall at ~90°.
  - Frame 0 = fully OPEN (all four flaps standing vertical).

What we want: with a *gentle* hinge the lid cannot hold itself up, so
under gravity each flap sags ~60° from vertical and rests there.

Two cooperating joints per hinge make this work:
  - A *soft* AffineBodyRevoluteJointLimit sets the rest angles. It is
    barrier-like, so even a very low LIMIT_STRENGTH holds an angle firmly,
    yet the motor can press a flap PAST it and, on release, it springs
    back. φ = angle from vertical (+ outward, − inward); both pairs may
    hang OUTWARD to OUT_DEG (≈60°), the ±Z pair may close fully inward
    (−PAIR1_IN_DEG = −90°), the ±X pair only to −PAIR2_IN_DEG (−60°).
  - The DRIVING joint is a stiff "motor" (effective stiffness ~100× the
    nominal strength·mass — a weak one aiming at a fixed angle can't make
    a gravity droop, it just pins). We use that stiffness as the "hand"
    that pushes/presses flaps on cue; it is `is_constrained = 1` ONLY
    while actively driving, otherwise the hinge is free.

The DEFAULT run is a four-act show (CHOREO):
  A HANG : all free; gravity swings every flap OUTWARD into the 60° limit
           (they hang out and wobble).
  B CLOSE: the motor folds the ±Z pair INWARD to closed (90°); released,
           the 90° limit + gravity hold it shut.
  C PUSH : the motor pushes the ±X pair INWARD to its 60° limit, then
           releases → it rests / wobbles at inward 60°.
  D PRESS: the motor presses the ±X pair toward closed (PRESS_DEG), then
           eases it SLOWLY back to the 60° limit and lets go (a free release
           would fling it over the unstable vertical — see _choreo).

Upright is an UNSTABLE equilibrium (zero net torque, so the solver stays
put), so each flap is BUILT tilted slightly OUTWARD by TILT_DEG to give a
finite starting gravity torque for act A. The joint `init_angle` is offset
by the build tilt so the joint angle reads 0 at vertical. The hinge/flap
is shifted OUTBOARD (HINGE_OUT) so a flap can swing fully down the OUTSIDE
of the wall (to ~180°) instead of jamming on it at ~90°.

Headless render (EGL) to a PNG sequence, same machinery the tape
drop/wind demos use (`tape_asset_lib.record_demo_to_pngs`).

Run (inside a gs-srun GPU container):
    python python/examples/carton_articulated_close_demo.py \
        --set RECORD_DIR=/mnt/home/zhaofeng/workspace/libuipc/output/carton
    # tune without editing the file:
    #   --set OUT_DEG=60 PAIR1_IN_DEG=90 PAIR2_IN_DEG=60 CLOSED_DEG=90
    #   --set DRIVE_STRENGTH=10 LIMIT_STRENGTH=1.0 TILT_DEG=15
    #   --set CHOREO=0 FREE_HINGE=1   (just gravity droop to the limit)
    #   --set CHOREO=0               (driving motor holds upright)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from uipc import (
    Logger,
    Engine,
    World,
    Scene,
    SceneIO,
    Animation,
    view,
    builtin,
)
from uipc.geometry import (
    trimesh,
    linemesh,
    ground,
    label_surface,
)
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyRevoluteJoint,
    AffineBodyDrivingRevoluteJoint,
    AffineBodyRevoluteJointLimit,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ----------------------------------------------------------------------
# Parameters (SI metres; the carton is specified in inches).
# ----------------------------------------------------------------------
INCH          = 0.0254
BASE          = 12.0 * INCH      # footprint side (X and Z), full outer
WALL_H        = 8.0  * INCH      # wall height of the INNER (±Z) pair
THK           = 0.003            # plate thickness — real corrugated cardboard
                                 # (~3 mm single-wall)
D_HAT         = 0.0006           # IPC contact band — must stay below every
                                 # clearance below (THK, OFFSET-THK, MID_GAP,
                                 # HINGE_GAP, CORNER_GAP) or contacts jam
OFFSET        = 0.005            # extra height of the OUTER (±X) walls so their
                                 # flaps fold ON TOP of the inner pair. Needs
                                 # OFFSET > THK so the outer flap clears the
                                 # inner one all through its swing; the residual
                                 # OFFSET-THK = 2 mm > D_HAT is the resting gap.
MID_GAP       = 0.003            # gap left between the two opposing flaps where
                                 # they meet over the middle. As small as
                                 # possible WITHOUT jamming: 3 mm = 5·D_HAT, so
                                 # the coplanar inner (and outer) edges never
                                 # enter each other's contact band.
CORNER_GAP    = 0.003            # shrink flap width by this each side so a
                                 # vertical flap clears its perpendicular
                                 # neighbour at the corners (> D_HAT)
HINGE_GAP     = 0.0015           # lift each flap's bottom edge (and its hinge
                                 # axis) this far above the wall top, so the
                                 # flap and the base are NOT coincident at
                                 # frame 0 (a 0-distance IPC contact would be
                                 # singular). Joints don't require the bodies
                                 # to touch — cf. sim_case/72. (> 2·D_HAT)
ABD_KAPPA     = 1.0e8            # ABD stiffness
JOINT_STRENGTH = 300.0           # revolute-hinge strength ratio (stiff: keeps
                                 # the hinge axis rigid — does NOT resist
                                 # rotation about the axis)

# --- Two joints per hinge --------------------------------------------------
# A weak DRIVING joint aiming at a fixed angle does NOT make a gravity droop:
# that joint is a stiff "motor" (effective stiffness ~100× nominal strength·
# mass), so any non-tiny strength rigidly pins its aim. We exploit exactly that
# stiffness — the motor is the "hand" that PUSHES / PRESSES flaps on cue (it is
# only `is_constrained = 1` while actively driving). The rest angles come from
# a soft AffineBodyRevoluteJointLimit (barrier-like: even a low strength holds
# firmly, yet the motor can press past it and, on release, it springs back).
DRIVE_STRENGTH = 10.0           # motor stiffness when pushing (>> LIMIT_STRENGTH
                                # so a press overpowers the limit)
# Soft-limit strength. LOW on purpose: (1) act A/C settle with a visible
# wobble, (2) the spring-back in act D is gentle. It must NOT be so deep a
# press that the elastic rebound carries the flap over the vertical (unstable)
# hill to the OUTWARD side — the limit bounds are fixed at world.init (the
# backend reads them once) and there is no damping knob, so the only levers are
# LIMIT_STRENGTH and the press depth PRESS_DEG (kept below ~80° past so the
# rebound energy stays under the gravity hill ≈ 0.05 J).
LIMIT_STRENGTH = 0.3

# Per-direction soft-limit angles (degrees from vertical; + OUTWARD, − INWARD).
# Both flap pairs may hang OUTWARD to OUT_DEG. Inward, the ±Z pair may close
# fully (PAIR1_IN_DEG = 90° = flat/closed) while the ±X pair only reaches
# PAIR2_IN_DEG (60°) — so it rests inward at 60° and springs back there after a
# press.
OUT_DEG        = 60.0            # outward hang cap (all flaps)
PAIR1_IN_DEG   = 90.0           # ±Z pair inward cap = fully closed
PAIR2_IN_DEG   = 60.0           # ±X pair inward cap = inward rest
CLOSED_DEG     = 90.0           # ±Z fully-closed target (its limit, no overshoot)
PRESS_DEG      = 88.0           # how far the motor presses the ±X pair toward
                                # closed before easing it back (guided return)

# Upright is an UNSTABLE equilibrium (zero net torque, solver stays put), so
# each flap is BUILT tilted slightly OUTWARD by TILT_DEG to give a finite
# starting gravity torque; it then swings outward into the limit.
TILT_DEG       = 15.0
# Push the hinge / flap this far OUTBOARD of the wall centre-line so a flap can
# swing fully outward (down the OUTSIDE of the wall, to ~180°) instead of
# jamming on the wall at ~90°. Needs flap inner face clear of the wall outer
# face when hanging: HINGE_OUT > THK/2 + D_HAT past the wall, i.e. ≳ THK.
HINGE_OUT      = THK + 2.0 * D_HAT

# Derived geometry
HALF          = BASE / 2.0
# Flap fold length: reach to MID_GAP/2 short of the box centre-line. The hinge
# sits at the wall centre-line (HALF - THK/2), so the folded flap's inner edge
# lands at (HALF - THK/2) - FLAP_LEN = MID_GAP/2 from centre → opposing flaps
# leave exactly MID_GAP between them. Just under half the opening, so no jam.
FLAP_LEN      = (HALF - THK / 2.0) - MID_GAP / 2.0
FLAP_W        = BASE - 2.0 * THK - 2.0 * CORNER_GAP   # flap width (along hinge)
ZC            = HALF - THK / 2.0      # ±Z wall centre-line in z
XC            = HALF - THK / 2.0      # ±X wall centre-line in x
# Hinge heights = wall top + HINGE_GAP (the flap floats just above its wall).
HIN_INNER     = THK + WALL_H + HINGE_GAP            # inner pair (±Z walls)
HIN_OUTER     = THK + WALL_H + OFFSET + HINGE_GAP   # outer pair (±X walls, taller)

# Flap identifiers.  `sign` is the rotation direction (about the hinge axis)
# that folds the flap INWARD, i.e. toward the box centre / toward closing; it
# drives both the inward-droop seed velocity and the angle-sign convention. The
# geometric guess below is verified from the run diagnostics (a flap drooping
# inward moves its centroid TOWARD the box axis) and flipped per-flap if wrong.
#  name        axis    hinge_y   coord  kind  sign
FLAPS = [
    ("front_+Z", "X", HIN_INNER, +ZC, "z", -1.0),  # hinge along X at z=+ZC
    ("back_-Z",  "X", HIN_INNER, -ZC, "z", +1.0),  # hinge along X at z=-ZC
    ("right_+X", "Z", HIN_OUTER, +XC, "x", +1.0),  # hinge along Z at x=+XC
    ("left_-X",  "Z", HIN_OUTER, -XC, "x", -1.0),  # hinge along Z at x=-XC
]

# ---- choreography timeline (dt = 0.01) — four acts ----
#  A HANG : all free; gravity swings every flap OUTWARD into the 60° limit.
#  B CLOSE: motor folds the ±Z pair INWARD to closed (90°); it stays (limit).
#  C PUSH : motor pushes the ±X pair INWARD to its 60° limit, then releases
#           → it rests / wobbles at inward 60°.
#  D PRESS: motor presses the ±X pair toward closed (PRESS_DEG), then eases
#           SLOWLY back to the 60° limit and releases (a gentle, controlled
#           "let-go": the soft limit alone is a stiff barrier with no damping,
#           so a free release would fling the flap over the unstable vertical
#           to the OUTWARD side — the motor walks it back so it settles inward).
PH_HANG   = 90
PH_CLOSE  = 70
PH_PUSH   = 70
PH_REST   = 45
PH_PRESS  = 55
PH_RETURN = 120     # slow, motor-guided return to the 60° rest
PH_SETTLE = 60      # free at the end, resting inward at 60°
A_END     = PH_HANG
B_END     = A_END + PH_CLOSE
C_END     = B_END + PH_PUSH
REST_END  = C_END + PH_REST
PRESS_END = REST_END + PH_PRESS
RETURN_END = PRESS_END + PH_RETURN
TOTAL_FRAMES = RETURN_END + PH_SETTLE


# ----------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------
def _box_VF(center, dims):
    """Axis-aligned box as (V[8,3], F[12,3]).  Winding is fixed up later
    by flip_inward_triangles, so the orientation here is irrelevant."""
    cx, cy, cz = center
    hx, hy, hz = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0
    V = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float64)
    F = np.array([
        [0, 3, 2], [0, 2, 1],   # -z
        [4, 5, 6], [4, 6, 7],   # +z
        [0, 1, 5], [0, 5, 4],   # -y
        [3, 7, 6], [3, 6, 2],   # +y
        [0, 4, 7], [0, 7, 3],   # -x
        [1, 2, 6], [1, 6, 5],   # +x
    ], dtype=np.int32)
    return V, F


def _concat_boxes(boxes):
    """Concatenate a list of (V, F) into one (V, F)."""
    Vs, Fs, off = [], [], 0
    for V, F in boxes:
        Vs.append(V)
        Fs.append(F + off)
        off += V.shape[0]
    return np.vstack(Vs), np.vstack(Fs)


def _make_sc(V, F):
    """Trimesh + surface labelling.  `label_triangle_orient` /
    `flip_inward_triangles` only work on tetmeshes, so the box winding in
    `_box_VF` is authored outward-facing directly and we just label the
    surface (same as sim_case/72's trimesh-cube ABD setup)."""
    sc = trimesh(V, F)
    label_surface(sc)
    return sc


def _build_base_mesh():
    """Bottom plate + four walls, concatenated into one mesh.

    Walls sit ON the bottom plate (y starts at THK) and the ±X walls are
    inset in z so the four walls butt together at the corners without
    overlapping volumes.  The ±X walls are OFFSET taller than the ±Z
    walls — that height stagger is what lets the ±X flaps fold on top.
    """
    boxes = []
    # bottom
    boxes.append(_box_VF((0.0, THK / 2.0, 0.0), (BASE, THK, BASE)))
    # ±Z walls (inner pair, height WALL_H) — full span in x
    for sz in (+1.0, -1.0):
        boxes.append(_box_VF(
            (0.0, THK + WALL_H / 2.0, sz * ZC),
            (BASE, WALL_H, THK)))
    # ±X walls (outer pair, height WALL_H + OFFSET) — inset in z
    inner_span = BASE - 2.0 * THK
    for sx in (+1.0, -1.0):
        boxes.append(_box_VF(
            (sx * XC, THK + (WALL_H + OFFSET) / 2.0, 0.0),
            (THK, WALL_H + OFFSET, inner_span)))
    V, F = _concat_boxes(boxes)
    return _make_sc(V, F)


def _build_flap_mesh(axis, hinge_y, coord_val):
    """One flap as a thin plate standing VERTICAL (the open pose), with
    its hinge edge at the bottom (y = hinge_y) and extending +FLAP_LEN up.
    `axis` is the hinge axis direction ('X' or 'Z'); the flap width FLAP_W
    runs along it.  `coord_val` is the wall centre-line (z for X-axis
    flaps, x for Z-axis flaps)."""
    cy = hinge_y + FLAP_LEN / 2.0
    if axis == "X":           # hinge ∥ X, flap in the z = coord_val plane
        center = (0.0, cy, coord_val)
        dims   = (FLAP_W, FLAP_LEN, THK)
    else:                     # axis == "Z": hinge ∥ Z, flap in x = coord_val
        center = (coord_val, cy, 0.0)
        dims   = (THK, FLAP_LEN, FLAP_W)
    V, F = _box_VF(center, dims)
    return _make_sc(V, F)


def _hinge_edge(axis, hinge_y, coord_val):
    """Two world-space endpoints of a hinge axis (length FLAP_W)."""
    h = FLAP_W / 2.0
    if axis == "X":
        return ([-h, hinge_y, coord_val], [+h, hinge_y, coord_val])
    else:
        return ([coord_val, hinge_y, -h], [coord_val, hinge_y, +h])


def _coord_out(coord_val):
    """Wall centre-line coord shifted OUTBOARD by HINGE_OUT, so the flap and its
    hinge sit just outside the wall and the flap can swing fully outward (down
    the OUTSIDE of the wall to ~180°) instead of jamming on the wall at ~90°."""
    return coord_val + np.sign(coord_val) * HINGE_OUT


def _tilt_transform(axis, hinge_y, coord_val, angle):
    """4×4 instance transform that rigidly rotates the (vertically-built) flap
    by `angle` about its hinge axis, i.e. R about the line through hinge point
    p with unit axis â:  T = Translate(p)·R(â,angle)·Translate(−p).

    Sign convention: `+sign·θ` folds the flap INWARD, `−sign·θ` OUTWARD (the
    same rotation sense as the joint's signed angle). The caller cancels the
    build tilt by setting the joint `init_angle` to the SAME angle, so aim = 0
    rests the spring at the upright pose regardless of the tilt direction."""
    if axis == "X":
        a = np.array([1.0, 0.0, 0.0])
        p = np.array([0.0, hinge_y, coord_val])
    else:  # "Z"
        a = np.array([0.0, 0.0, 1.0])
        p = np.array([coord_val, hinge_y, 0.0])
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])           # skew(â):  [â]ₓ·v = â × v
    R = (np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K))  # Rodrigues
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = p - R @ p
    return T


def _smooth(t):
    """Cosine ease 0→1 (zero velocity at both ends)."""
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _ramp(frame, f0, f1, v0, v1):
    """Smoothly ramp v0→v1 as frame goes f0→f1 (clamped outside)."""
    if frame <= f0:
        return v0
    if frame >= f1:
        return v1
    return v0 + (v1 - v0) * _smooth((frame - f0) / float(f1 - f0))


def _choreo(frame):
    """Four-act motor schedule → per-edge (is_constrained[4], aim_rad[4]).

    φ = signed angle from vertical (+ outward, − inward); the motor rests a flap
    at jang = aim and jang = −sign·φ, so aim_rad = −sign·radians(φ). The motor
    is active (is_constrained = 1) ONLY while it is driving; otherwise the hinge
    is free and the soft limit governs.
      ±Z pair (axis 'X'): act B drives  +OUT → −CLOSED  (fold flat).
      ±X pair (axis 'Z'): act C drives  +OUT → −PAIR2_IN (push to inward rest),
                          act D presses −PAIR2_IN → −PRESS_DEG, then eases
                          −PRESS_DEG → −PAIR2_IN (slow guided return) and frees.
    """
    is_c = np.zeros(len(FLAPS), dtype=np.int32)
    aim  = np.zeros(len(FLAPS), dtype=np.float64)
    for i, (name, axis, hin, coord, kind, sign) in enumerate(FLAPS):
        driven, phi = False, 0.0
        if axis == "X":                                    # ±Z pair
            if A_END <= frame < B_END:
                driven, phi = True, _ramp(frame, A_END, B_END, +OUT_DEG, -CLOSED_DEG)
        else:                                              # ±X pair
            if B_END <= frame < C_END:
                driven, phi = True, _ramp(frame, B_END, C_END, +OUT_DEG, -PAIR2_IN_DEG)
            elif REST_END <= frame < PRESS_END:
                driven, phi = True, _ramp(frame, REST_END, PRESS_END, -PAIR2_IN_DEG, -PRESS_DEG)
            elif PRESS_END <= frame < RETURN_END:
                driven, phi = True, _ramp(frame, PRESS_END, RETURN_END, -PRESS_DEG, -PAIR2_IN_DEG)
        if driven:
            is_c[i] = 1
            aim[i]  = -sign * np.radians(phi)
    return is_c, aim


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

    # ---- tunables (overridable via --set) -----------------------------
    tilt_rad = np.radians(float(cfg.get("TILT_DEG", TILT_DEG)))
    # choreo = the four-act show (DEFAULT). The motor (driving joint) is the
    # "hand" that pushes/presses on cue; the soft limit sets the rest angles.
    choreo_mode = cfg.get("CHOREO", "1") == "1"
    free_hinge  = cfg.get("FREE_HINGE", "0") == "1"   # pure free-hinge (no motor)
    use_limit   = cfg.get("LIMIT", "1") == "1"

    drive_strengths = np.full(
        len(FLAPS), float(cfg.get("DRIVE_STRENGTH", DRIVE_STRENGTH)), dtype=np.float64)
    if "LIMIT_STRENGTHS" in cfg:                 # per-flap limit-strength sweep
        limit_strengths = np.array(
            [float(x) for x in cfg["LIMIT_STRENGTHS"].split(",")], dtype=np.float64)
    else:
        limit_strengths = np.full(
            len(FLAPS), float(cfg.get("LIMIT_STRENGTH", LIMIT_STRENGTH)), dtype=np.float64)

    # Per-edge soft-limit φ-range (degrees from vertical; + outward, − inward).
    # Both pairs hang outward to OUT_DEG; the ±Z pair (axis 'X') may close fully
    # (−PAIR1_IN_DEG), the ±X pair (axis 'Z') only to −PAIR2_IN_DEG.
    out_deg  = float(cfg.get("OUT_DEG", OUT_DEG))
    p1_in    = float(cfg.get("PAIR1_IN_DEG", PAIR1_IN_DEG))
    p2_in    = float(cfg.get("PAIR2_IN_DEG", PAIR2_IN_DEG))
    phi_lo = np.array([-(p1_in if f[1] == "X" else p2_in) for f in FLAPS])  # inward bound
    phi_hi = np.full(len(FLAPS), +out_deg)                                  # outward bound

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)          # friction 0.5, stiffness 1 GPa

    # ---- BASE (fixed) -------------------------------------------------
    # NOTE: abd.apply_to creates the per-instance `is_fixed` attribute, so
    # resize the instance buffer BEFORE applying (mirrors the revolute demo).
    base_sc = _build_base_mesh()
    base_sc.instances().resize(1)
    abd.apply_to(base_sc, ABD_KAPPA)
    view(base_sc.instances().find(builtin.is_fixed))[:] = 1
    base_obj = scene.objects().create("carton_base")
    base_slot, _ = base_obj.geometries().create(base_sc)

    # ---- FLAPS (free, gravity-droop) ---------------------------------
    flap_slots = []
    flap_objs  = scene.objects().create("carton_flaps")
    for (name, axis, hin, coord, _kind, sign) in FLAPS:
        coord_o = _coord_out(coord)              # hinge/flap sit outboard of wall
        fsc = _build_flap_mesh(axis, hin, coord_o)
        fsc.instances().resize(1)
        abd.apply_to(fsc, ABD_KAPPA)
        view(fsc.instances().find(builtin.is_fixed))[:] = 0
        # Start tilted OUTWARD by TILT (off the unstable upright, finite gravity
        # torque). Outward (not inward) so the four flaps splay APART and never
        # intersect at the corners — inward tilt makes perpendicular flaps
        # overlap and fails world.init's penetration check. init_angle below
        # cancels the tilt so the joint angle still reads 0 at upright.
        #   inward  rotation = +sign·θ  →  outward = −sign·θ
        if tilt_rad != 0.0:
            view(fsc.transforms())[0] = _tilt_transform(axis, hin, coord_o, -sign * tilt_rad)
        slot, _ = flap_objs.geometries().create(fsc)
        flap_slots.append(slot)

    # ---- HINGE joints: base (left) ↔ each flap (right) ----------------
    jV, jE = [], []
    for i, (name, axis, hin, coord, _kind, _sign) in enumerate(FLAPS):
        p0, p1 = _hinge_edge(axis, hin, _coord_out(coord))
        jV.append(p0); jV.append(p1)
        jE.append([2 * i, 2 * i + 1])
    joint_mesh = linemesh(np.array(jV, dtype=np.float32),
                          np.array(jE, dtype=np.int32))

    revolute = AffineBodyRevoluteJoint()
    revolute.apply_to(joint_mesh,
                      [base_slot] * len(FLAPS),   # left bodies (all the base)
                      flap_slots,                 # right bodies (the flaps)
                      JOINT_STRENGTH)

    # The bind pose is the (outward-tilted) frame-0 configuration, so offset
    # init_angle by that same build-tilt angle (−sign·tilt): then with aim = 0
    # the spring rests at the UPRIGHT pose, not at the tilted bind.
    signs = np.array([f[5] for f in FLAPS], dtype=np.float64)
    init_angle = joint_mesh.edges().find("init_angle")
    if init_angle is not None:
        view(init_angle)[:] = -signs * tilt_rad

    driving = AffineBodyDrivingRevoluteJoint()
    driving.apply_to(joint_mesh, drive_strengths)   # stiff motor (the pushing hand)

    # Active driving (not passive resist-only).
    is_passive = joint_mesh.edges().find("is_passive")
    if is_passive is not None:
        view(is_passive)[:] = 0

    # Soft joint limit. The limit acts on the joint angle `jang`, and
    # jang = −sign·φ where φ is the angle from vertical (+ outward, − inward).
    # So a φ-range [phi_lo, phi_hi] maps to jang ∈ sorted(−sign·phi_lo, −sign·phi_hi).
    if use_limit:
        b_lo = -signs * np.radians(phi_lo)
        b_hi = -signs * np.radians(phi_hi)
        lowers = np.minimum(b_lo, b_hi)
        uppers = np.maximum(b_lo, b_hi)
        limit = AffineBodyRevoluteJointLimit()
        limit.apply_to(joint_mesh, lowers, uppers, limit_strengths)

    joint_obj = scene.objects().create("hinges")
    joint_slot, _ = joint_obj.geometries().create(joint_mesh)

    def drive_flaps(info: Animation.UpdateInfo):
        geo  = info.geo_slots()[0].geometry()
        is_c = view(geo.edges().find("driving/is_constrained"))
        aim  = view(geo.edges().find("aim_angle"))
        if choreo_mode:
            # Four-act show: the motor pushes/presses on cue, the soft limit
            # sets the rest angles (see _choreo / the timeline constants).
            c, a = _choreo(info.frame())
            is_c[:] = c
            aim[:]  = a
        elif free_hinge:
            # Pure free hinge: gravity alone moves the flaps; the soft limit
            # catches them at their bound.
            is_c[:] = 0
        else:
            # Driving motor holds the upright pose (aim = 0).
            is_c[:] = 1
            aim[:]  = 0.0

    scene.animator().insert(joint_obj, drive_flaps)

    world.init(scene)

    return {
        "engine": engine, "world": world, "scene": scene,
        "scene_io": SceneIO(scene),
        "base_slot": base_slot, "flap_slots": flap_slots,
        "joint_slot": joint_slot,
    }


# ----------------------------------------------------------------------
def _diagnostics(sim, frame, total):
    """Per-flap droop diagnostics: droop angle FROM VERTICAL (deg) + centroid.

    `droop` is the geometric tilt of the hinge→centroid vector away from the
    +Y (upright) axis — convention-independent, so it reads 0° at upright and
    ≈90° flat regardless of the joint's signed `angle`.  A flap drooping OUTWARD
    moves its centroid AWAY from the box axis; toward it would flag a sign flip.
    `jang` is the joint's own signed angle (deg) for cross-checking."""
    jgeo = sim["joint_slot"].geometry()
    jang = jgeo.edges().find("angle")
    jang = np.degrees(np.asarray(view(jang)).reshape(-1)) if jang is not None else None
    parts = [f"frame {frame}/{total}"]
    for i, (name, axis, hin, coord, kind, sign) in enumerate(FLAPS):
        geo = sim["flap_slots"][i].geometry()
        # ABD: positions() are LOCAL/rest; the body pose lives in the affine
        # instance transform. Compose to get the world-space centroid.
        loc = np.asarray(geo.positions().view()).reshape(-1, 3)
        T = np.asarray(view(geo.transforms())[0]).reshape(4, 4)
        world = (loc @ T[:3, :3].T) + T[:3, 3]
        c = world.mean(axis=0)
        # hinge midpoint (the centroid hangs FLAP_LEN/2 along the flap from it)
        co = _coord_out(coord)
        hmid = np.array([0.0, hin, co]) if axis == "X" else np.array([co, hin, 0.0])
        d = c - hmid
        droop = np.degrees(np.arctan2(np.hypot(d[0], d[2]), d[1]))
        # signed angle from vertical (+ outward, − inward) from the joint angle
        phi = (-sign * jang[i]) if jang is not None else float("nan")
        parts.append(f"[{name}] phi={phi:+6.1f}° |droop|={droop:4.0f}° "
                     f"cen=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})")
    print("  ".join(parts), flush=True)


def _parse_set_args(argv=None):
    """Minimal `--set KEY=VALUE` parser (kept independent of the tape
    presets machinery)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = {}
    i = 0
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

    total_frames = int(cfg.get("FRAMES", cfg.get("SETTLE_FRAMES", TOTAL_FRAMES)))

    record_dir = cfg.get("RECORD_DIR") or os.path.join(
        AssetDir.output_path(__file__), "frames")
    every_n = int(cfg.get("RECORD_EVERY", 5))
    zoom    = float(cfg.get("RECORD_ZOOM", 1.2))

    # visualization-only ground quad at y = 0
    gq_v = np.array([[-0.4, 0.0, -0.4], [0.4, 0.0, -0.4],
                     [0.4, 0.0, 0.4], [-0.4, 0.0, 0.4]], dtype=np.float64)
    gq_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def setup_extras(ps_mod):
        gm = ps_mod.register_surface_mesh("ground", gq_v, gq_f)
        gm.set_color((0.55, 0.55, 0.58))
        gm.set_transparency(0.5)

    print(f"[carton] base {BASE/INCH:.0f}×{BASE/INCH:.0f}×{WALL_H/INCH:.0f} in, "
          f"THK={THK*1e3:.1f} mm, flap_len={FLAP_LEN/INCH:.2f} in, "
          f"mid_gap={MID_GAP*1e3:.1f} mm, hinge offset={OFFSET*1e3:.1f} mm, "
          f"d_hat={D_HAT*1e3:.2f} mm, {total_frames} frames")
    if cfg.get("CHOREO", "1") == "1":
        print(f"[carton] CHOREO (4 acts): A hang-out {OUT_DEG:.0f}° | "
              f"B close ±Z to {PAIR1_IN_DEG:.0f}° | C push ±X to {PAIR2_IN_DEG:.0f}° | "
              f"D press ±X to {PRESS_DEG:.0f}° then guided return to {PAIR2_IN_DEG:.0f}°. "
              f"motor={cfg.get('DRIVE_STRENGTH', DRIVE_STRENGTH)} "
              f"limit={cfg.get('LIMIT_STRENGTHS', cfg.get('LIMIT_STRENGTH', LIMIT_STRENGTH))} "
              f"acts@frames[A{A_END},B{B_END},C{C_END},rest{REST_END},press{PRESS_END},ret{RETURN_END}]")
    elif cfg.get("FREE_HINGE", "0") == "1":
        print(f"[carton] FREE hinge + soft limit (out={OUT_DEG}°, "
              f"limit={cfg.get('LIMIT_STRENGTH', LIMIT_STRENGTH)})")
    else:
        print(f"[carton] DRIVING mode: aim=upright(0)")
    print(f"[carton] recording → {record_dir}")

    L.record_demo_to_pngs(
        sim=sim, total_frames=total_frames, output_dir=record_dir,
        every_n=every_n, up_dir="y_up", mesh_name="carton",
        setup_extras_fn=setup_extras, zoom=zoom,
        on_progress=lambda f, t: _diagnostics(sim, f, t),
        bbox_extent_override=BASE * 1.3)


if __name__ == "__main__":
    run()
