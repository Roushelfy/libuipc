"""
carton_tape_seal_v2_demo.py — tape the carton seam with the real scotch3850 roll.

v2 of carton_tape_seal_demo.py: instead of a bare spindle, the spindle becomes a
CARRIER that holds the wound scotch3850 roll (asset scotch3850_3turn.npz) on a
revolute bearing (so the roll can spin and pay tape off). The carrier is
STC-driven along the SAME seal path as v1 (approach the −Z seam end → run +Z →
cross +Z edge → down +Z face). RCC adhesion bonds the tape to the carton, so the
free tail sticks at the −Z end and the tape lays/sticks along the seam.

Bodies: carton base(fixed) + 4 flaps(hinged, back pair held pressed) + hub(ring,
free, spins) + carrier(rigid, STC-driven) + tape(NeoHookean shell, from asset).
Adhesion: tape↔tape + tape↔hub (restored from asset) + tape↔carton (grows on
press). Backend lock state re-seeded after world.init with the ABD-vs-FEM index
remap (hub created FIRST so only tape indices shift).

NOTE: high-risk integration (pay-off vs snap, ride-height geometry, mixed d_hat).
`--set FRAMES=480` for a short dry-run (close+approach+stick only). Knobs:
RIDE_DY, SP_RUN, ASSET.

Run (gs-srun GPU):
    python python/examples/carton_tape_seal_v2_demo.py \
        --set RECORD_DIR=/mnt/.../output/carton_seal_v2
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
    NeoHookeanShell,
    DiscreteShellBending,
    SoftPositionConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)
from uipc.core import RCCAdhesionStateAccessorFeature, RCCBondedPTStateAccessorFeature

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir

import carton_articulated_close_demo as C
from carton_articulated_close_demo import (
    _build_base_mesh, _build_flap_mesh, _hinge_edge, _coord_out, _tilt_transform, _ramp,
    _box_VF, _make_sc,
    FLAPS, TILT_DEG, ABD_KAPPA, JOINT_STRENGTH, DRIVE_STRENGTH, LIMIT_STRENGTH,
    OUT_DEG, PAIR1_IN_DEG, PAIR2_IN_DEG, CLOSED_DEG,
    A_END, B_END, C_END, PRESS_END,
    D_HAT, THK, WALL_H, OFFSET, BASE, HALF, HIN_INNER, HIN_OUTER, ZC, FLAP_W, INCH,
)

DEFAULT_ASSET = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                             "distlock_run", "scotch3850_3turn.npz")


# ---- helpers copied from rod-wind (self-contained) -------------------
def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


def _make_rod_abd_sc(R, length, n_sides, center):
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


def _make_tape_topology_tris(NX, NZ):
    n_width = NZ + 1
    def vid(i, j): return i * n_width + j
    tris = []
    for i in range(NX):
        for j in range(NZ):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            tris += [[a, b, c], [a, c, d]]
    return np.asarray(tris, dtype=np.int32)


def _make_tape_sc(positions, tris):
    sc = trimesh(positions.astype(np.float64), tris)
    label_surface(sc)
    return sc


def _straight_rest_positions(R_anchor, length, width, NX, NZ):
    dy, dz = length / NX, width / NZ
    nw = NZ + 1
    verts = np.empty(((NX + 1) * nw, 3), dtype=np.float64)
    for i in range(NX + 1):
        for j in range(nw):
            verts[i * nw + j] = (R_anchor, i * dy, j * dz - 0.5 * width)
    return verts


# +Z build axis -> +X (spindle/carrier perpendicular to the Z-seam)
_RY90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
# scotch asset native frame (hub axis +Z, tail +Y) -> hub axis −X, tail −Y.
# PROPER rotation (det=+1; a reflection makes the hub ABD determinant negative
# -> affine_body_dynamics D>=0 assert). This is R_y(180)·[+Z->+X,+Y->−Y] so the
# roll faces the OTHER way (tape pays off trailing the +Z traverse, not leading):
# columns = images of e_x,e_y,e_z: +X->−Z, +Y->−Y (tail down), +Z->−X (hub axis).
_R_MOUNT = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=np.float64)


def _mount_roll(tape_pos, hub_T, R_mount, hub_center_world):
    hub_c0 = hub_T[:3, 3].copy()
    tape_new = (tape_pos - hub_c0) @ R_mount.T + hub_center_world
    hub_T_new = hub_T.copy()
    hub_T_new[:3, :3] = R_mount @ hub_T[:3, :3]
    hub_T_new[:3, 3] = hub_center_world
    return tape_new, hub_T_new


# ---- back-pair-held-pressed close schedule (from v1) -----------------
CLOSE_X_DEG = 90.0


def _seal_choreo(frame):
    is_c = np.zeros(len(FLAPS), dtype=np.int32)
    aim = np.zeros(len(FLAPS), dtype=np.float64)
    for i, (name, axis, hin, coord, kind, sign) in enumerate(FLAPS):
        driven, phi = False, 0.0
        if axis == "X":
            if A_END <= frame < B_END:
                driven, phi = True, _ramp(frame, A_END, B_END, +OUT_DEG, -CLOSED_DEG)
        else:
            if B_END <= frame < C_END:
                driven, phi = True, _ramp(frame, B_END, C_END, +OUT_DEG, -PAIR2_IN_DEG)
            elif C_END <= frame < PRESS_END:
                driven, phi = True, _ramp(frame, C_END, PRESS_END, -PAIR2_IN_DEG, -CLOSE_X_DEG)
            elif frame >= PRESS_END:
                driven, phi = True, -CLOSE_X_DEG
        if driven:
            is_c[i] = 1
            aim[i] = -sign * np.radians(phi)
    return is_c, aim


# ---- seam geometry (back ±X pair, along Z at x=0) --------------------
SEAM_X = 0.0
SEAM_Y = HIN_OUTER + 0.5 * THK     # ≈ 0.2127
SEAM_Z0 = -FLAP_W / 2.0            # −Z end (start)
SEAM_Z1 = +FLAP_W / 2.0            # +Z end (far)
SIDE_Z = HALF + 0.045
DOWN_Y = 0.5 * (THK + WALL_H)

# carrier / spindle
SP_KAPPA, SP_DENSITY = 1.0e8, 1000.0
HUB_MASS_DENSITY = 1000.0
# axis stiffness: STC_ETA drives the spindle along its commanded path; a stiffer
# (×100) roll won't pay out tape, so it loads the axis hard — the drive must be
# strong enough to keep the spindle ON the trajectory or the roll lags and stalls.
STC_ETA = 1.0e8
CARRIER_JOINT_STRENGTH = 2000.0

# press-cube: a rigid block (slightly larger than the tape width 0.048) that
# presses the tape onto the −Z wall at the seam start and the +Z wall at the end
# — replaces the SPC tape-presses. Its path is defined after the schedule below.
CUBE_HX, CUBE_HY, CUBE_HZ = 0.032, 0.05, 0.015     # -> 0.064 x 0.10 x 0.03
CUBE_Y_PRESS  = 0.15                                # centre height while pressing
CUBE_Y_HIGH   = 0.35                                # over-the-top transit height (clear of the box top 0.206)
CUBE_PRESS_IN     = 0.006                           # −Z press aim depth into the wall (force ≈ η·depth)
CUBE_PRESS_IN_POS = 0.012                           # +Z (final) press: deeper → firmer, to ensure the tape sticks
CUBE_STC_ETA  = 1.0e5    # LOW on purpose: the cube only needs to gently reach the wall to close
#                          the tape gap (bond forms on distance) → yields to contact → easy to solve
CUBE_KAPPA, CUBE_DENSITY = 1.0e8, 1000.0
CUBE_Z_NEG  = -(HALF + CUBE_HZ) + CUBE_PRESS_IN         # press pose at the −Z wall
CUBE_Z_POS  = +(HALF + CUBE_HZ) - CUBE_PRESS_IN_POS     # press pose at the +Z wall (deeper)
CUBE_Z_PRE  = HALF + 0.07                           # stand-off outside a wall
CUBE_Z_PARK = 0.55                                  # far-out park (clear of flaps)
CUBE_X_SWING = 0.45                                 # swing wide around the box


def build_demo(cfg: dict):
    def _cf(k, d):  return float(cfg.get(k, d))
    def _ci(k, d):  return int(cfg.get(k, d))

    ASSET = cfg.get("ASSET", DEFAULT_ASSET)
    hub_T, tape_pos, params, pair_state, tape_vel = L.load_tape_asset(ASSET)

    def _p(k):  return float(params[k])
    HUB_R_OUTER, HUB_R_INNER = _p("HUB_R_OUTER"), _p("HUB_R_INNER")
    HUB_HEIGHT = _p("HUB_HEIGHT")
    N_TURNS = _p("N_TURNS"); LAYER_THICKNESS = _p("LAYER_THICKNESS")
    TAPE_LENGTH, TAPE_WIDTH = _p("TAPE_LENGTH"), _p("TAPE_WIDTH")
    TAPE_NX, TAPE_NZ = int(params["TAPE_NX"]), int(params["TAPE_NZ"])

    def _res(k): return float(L.resolve_param(cfg, params, k))
    D_HAT_TAPE = _res("D_HAT"); TAPE_THICKNESS = _res("TAPE_THICKNESS")
    TAPE_YOUNGS = _res("TAPE_YOUNGS"); BENDING_STIFFNESS = _res("BENDING_STIFFNESS")
    TAPE_POISSON = _res("TAPE_POISSON"); TAPE_MASS_DENSITY = _res("TAPE_MASS_DENSITY")
    ADH_CN, ADH_CT, ADH_W, ADH_ETA = (_res(k) for k in ("ADH_CN", "ADH_CT", "ADH_W", "ADH_ETA"))
    ADH_BONDING_RATE = _res("ADH_BONDING_RATE"); ADH_INITIAL_BETA = _res("ADH_INITIAL_BETA")

    R_roll = HUB_R_OUTER + N_TURNS * LAYER_THICKNESS + TAPE_THICKNESS
    # Bake the roll PARKED high above the box CENTRE — clear of the OPEN flaps at
    # frame 0 (baking it at the −Z seam end intersects the still-standing −Z flap
    # → world.init penetration reject). The carrier drives it down to the seam
    # only AFTER the carton closes. HUB_C0 must equal the path's PARK.
    HUB_C0 = PARK_POS.copy()
    tape_pos_up, hub_T_up = _mount_roll(tape_pos, hub_T, _R_MOUNT, HUB_C0)
    centers0 = tape_pos_up.reshape(-1, TAPE_NZ + 1, 3).mean(axis=1)
    tail_dir = centers0[-1] - centers0[0]
    print(f"[seal2] roll R={R_roll*1e3:.1f}mm W={TAPE_WIDTH*1e3:.0f}mm, parked hub center={HUB_C0}, "
          f"ride_Y={_RIDE_Y:.4f}, tail tip={centers0[-1]} (dir {tail_dir})", flush=True)

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
    # bonded-PT, mirror the asset
    config["rcc_bonded_pt_enabled"] = 1
    config["rcc_bonded_pt_skip_ccd"] = 0
    config["rcc_bonded_pt_beta_lock_threshold"] = 0.9
    config["rcc_bonded_pt_lock_face_interior_only"] = 1
    config["rcc_bonded_pt_kappa"] = _res("RCC_KAPPA")
    config["rcc_bonded_pt_release_force"] = _res("RCC_RELEASE_FORCE")
    config["rcc_bonded_pt_distance_lock"] = 1
    config["rcc_bonded_pt_distance_lock_ratio"] = float(params.get("DISTANCE_LOCK_RATIO", 1.5))
    config["rcc_adhesion_normal_offset_coeff"] = _cf("RCC_ADHESION_NORMAL_OFFSET_COEFF", 0.5)
    scene = Scene(config)

    abd = AffineBodyConstitution()
    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)

    # ---- HUB FIRST (clean lock remap: only tape indices shift) -------
    hub_sc = L.make_ring_hub(R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
                             height=HUB_HEIGHT, n_radial=48, center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, ABD_KAPPA, HUB_MASS_DENSITY)
    hub_elem = tabular.create("hub"); hub_elem.apply_to(hub_sc)
    view(hub_sc.transforms())[0] = _mat4_to_uipc(hub_T_up)
    hub_obj = scene.objects().create("hub")
    hub_slot, _ = hub_obj.geometries().create(hub_sc)

    # ---- carton BASE (fixed) -----------------------------------------
    base_sc = _build_base_mesh()
    base_sc.instances().resize(1)
    abd.apply_to(base_sc, ABD_KAPPA)
    view(base_sc.instances().find(builtin.is_fixed))[:] = 1
    carton_elem = tabular.create("carton")
    carton_elem.apply_to(base_sc)
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

    # ---- CARRIER (rigid, STC-driven, axis ∥ X, baked at origin) ------
    CARRIER_R = _cf("CARRIER_R", 0.5 * HUB_R_INNER)
    CARRIER_LEN = _cf("CARRIER_LEN", 2.4 * HUB_HEIGHT)
    carrier_sc = _make_rod_abd_sc(CARRIER_R, CARRIER_LEN, 16, (0.0, 0.0, 0.0))
    abd.apply_to(carrier_sc, SP_KAPPA, SP_DENSITY)
    carrier_elem = tabular.create("carrier"); carrier_elem.apply_to(carrier_sc)
    _stc_eta = _cf("STC_ETA", STC_ETA)
    print(f"[seal2] axis stiffness: STC_ETA={_stc_eta:.1e}, "
          f"bearing_strength={_cf('CARRIER_JOINT_STRENGTH', CARRIER_JOINT_STRENGTH):.0f}", flush=True)
    stc = SoftTransformConstraint(); stc.apply_to(carrier_sc, np.array([_stc_eta, _stc_eta]))
    M0 = np.eye(4); M0[:3, :3] = _RY90; M0[:3, 3] = HUB_C0
    view(carrier_sc.transforms())[0] = _mat4_to_uipc(M0)
    carrier_obj = scene.objects().create("carrier")
    carrier_slot, _ = carrier_obj.geometries().create(carrier_sc)

    # ---- PRESS-CUBE (rigid, STC-driven; created before tape so the lock
    #      remap shift stays clean: only tape indices move) ---------------
    cube_sc = _make_sc(*_box_VF((0.0, 0.0, 0.0), (2 * CUBE_HX, 2 * CUBE_HY, 2 * CUBE_HZ)))
    cube_sc.instances().resize(1)
    abd.apply_to(cube_sc, CUBE_KAPPA, CUBE_DENSITY)
    cube_elem = tabular.create("cube"); cube_elem.apply_to(cube_sc)
    _cube_eta = _cf("CUBE_STC_ETA", CUBE_STC_ETA)
    SoftTransformConstraint().apply_to(cube_sc, np.array([_cube_eta, _cube_eta]))
    Mc0 = np.eye(4); Mc0[:3, 3] = _cube_center(0)
    view(cube_sc.transforms())[0] = _mat4_to_uipc(Mc0)
    cube_obj = scene.objects().create("press_cube")
    cube_slot, _ = cube_obj.geometries().create(cube_sc)

    # ---- TAPE (current = mounted wound, rest = straight strip) -------
    tris = _make_tape_topology_tris(TAPE_NX, TAPE_NZ)
    current_sc = _make_tape_sc(tape_pos_up, tris)
    R_anchor = HUB_R_OUTER + TAPE_THICKNESS + 0.5 * D_HAT_TAPE
    rest_sc = _make_tape_sc(_straight_rest_positions(R_anchor, TAPE_LENGTH, TAPE_WIDTH, TAPE_NX, TAPE_NZ), tris)
    nhs, dsb = NeoHookeanShell(), DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    for sc in (current_sc, rest_sc):
        nhs.apply_to(sc, moduli, mass_density=TAPE_MASS_DENSITY, thickness=TAPE_THICKNESS)
        dsb.apply_to(sc, BENDING_STIFFNESS)
    tape_elem = tabular.create("tape"); tape_elem.apply_to(current_sc)
    spc = SoftPositionConstraint(); spc.apply_to(current_sc, _cf("SPC_STRENGTH", 3000.0))
    RCCAdhesive.set_sticky_side(current_sc, 0)
    if tape_vel is not None and tape_vel.shape == tape_pos.shape:
        if current_sc.vertices().find("velocity") is None:
            current_sc.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        view(current_sc.vertices().find("velocity"))[:] = (tape_vel @ _R_MOUNT.T).reshape(-1, 3, 1)
    tape_obj = scene.objects().create("tape")
    tape_slot, _ = tape_obj.geometries().create(current_sc, rest_sc)

    # ---- contact pairs ------------------------------------------------
    tabular.insert(tape_elem, tape_elem, 0.5, 1.0e9)
    tabular.insert(tape_elem, hub_elem, 0.5, 1.0e9)
    tabular.insert(tape_elem, carton_elem, 0.5, 1.0e9)
    tabular.insert(tape_elem, carrier_elem, 0.5, 1.0e9)
    tabular.insert(carrier_elem, hub_elem, 0.5, 1.0e9, enable=False)   # bearing = joint
    tabular.insert(carrier_elem, carton_elem, 0.5, 1.0e9)
    tabular.insert(hub_elem, carton_elem, 0.5, 1.0e9)
    tabular.insert(cube_elem, tape_elem, 0.5, 1.0e9)      # explicit entries so the adhesion-OFF
    tabular.insert(cube_elem, carton_elem, 0.5, 1.0e9)    # set() below has a pair to target

    # ---- adhesion: tape↔tape, tape↔hub (restored), tape↔carton (grow) -
    # tape↔carton bonds get a MUCH higher release force (×TAPE_CARTON_RF_MULT,
    # default 100 → 1e-5 vs the asset's global 1e-7) so laid tape STAYS stuck to
    # the box instead of releasing under the roll's motion. The roll's internal
    # tape-tape / tape-hub bonds keep the global release force (rf=-1 inherits).
    RF_GLOBAL = _res("RCC_RELEASE_FORCE")
    TC_RF = RF_GLOBAL * _cf("TAPE_CARTON_RF_MULT", 1000.0)
    # internal roll cohesion (tape↔tape and tape↔hub) release force: default
    # inherits the global (mult=1 → rf=-1, exact prior path); >1 makes the wound
    # roll harder to unspool.
    TT_MULT = _cf("TAPE_TAPE_RF_MULT", 1.0)
    TT_RF = -1.0 if TT_MULT == 1.0 else RF_GLOBAL * TT_MULT
    adhesive = RCCAdhesive()
    adhesive.default_model(tabular, Cn=0.0, Ct=0.0, W=0.0, eta=ADH_ETA,
                           bonding_rate=0.0, p0=0.0, initial_beta=0.0, enabled=False)
    for A, B, b0, br, rf in [(tape_elem, tape_elem, ADH_INITIAL_BETA, ADH_BONDING_RATE, TT_RF),
                             (tape_elem, hub_elem, ADH_INITIAL_BETA, ADH_BONDING_RATE, TT_RF),
                             (tape_elem, carton_elem, 0.0, max(ADH_BONDING_RATE, 1.0), TC_RF)]:
        adhesive.set(tabular, A, B, Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
                     bonding_rate=br, p0=0.0, initial_beta=b0, enabled=True)
        adhesive.set_bonded(tabular, A, B, lock_threshold=-1.0, release_strain=-1.0,
                            release_gap=-1.0, release_slip=-1.0, release_force=rf)
    # the press-CUBE only presses — it must NOT stick to the tape (or the box), or
    # lifting it would peel the tape off the wall. Explicitly disable its adhesion
    # (no set_bonded on these pairs either → no distance-lock forms).
    for B in (tape_elem, carton_elem):
        adhesive.set(tabular, cube_elem, B, Cn=0.0, Ct=0.0, W=0.0, eta=ADH_ETA,
                     bonding_rate=0.0, p0=0.0, initial_beta=0.0, enabled=False)
    _tt_rf_eff = RF_GLOBAL if TT_RF < 0 else TT_RF
    print(f"[seal2] tape<->carton release_force={TC_RF:.2e} "
          f"(x{_cf('TAPE_CARTON_RF_MULT',100.0):.0f} of global {RF_GLOBAL:.1e}); "
          f"tape<->tape/hub release_force={_tt_rf_eff:.2e} (x{TT_MULT:.0f}); cube adhesion OFF", flush=True)

    # (no sim ground — base is fixed, tape is carrier-held; the carton base
    # bottom sits at y=0 so a half-plane there would fail the init distance
    # check. The visualization ground quad is added in run().)

    # ---- carton hinge joints -----------------------------------------
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
    b_lo, b_hi = -signs * np.radians(phi_lo), -signs * np.radians(np.full(len(FLAPS), +OUT_DEG))
    _flap_limit = _cf("LIMIT_STRENGTH", LIMIT_STRENGTH * 0.01)   # ÷100: softer flap limit
    AffineBodyRevoluteJointLimit().apply_to(joint_mesh, np.minimum(b_lo, b_hi),
                                            np.maximum(b_lo, b_hi), np.full(len(FLAPS), _flap_limit))
    joint_obj = scene.objects().create("hinges")
    joint_slot, _ = joint_obj.geometries().create(joint_mesh)

    # after the taping motion finishes, RELEASE the flap-top hold and let the
    # tape alone try to keep the seam closed (RELEASE_AT default = motion end).
    release_at = int(cfg.get("RELEASE_AT", TOTAL_FRAMES))
    def drive_flaps(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = info.frame()
        c, a = _seal_choreo(f)
        if f >= release_at:
            c = np.zeros_like(c)      # stop pressing the flaps closed
        view(geo.edges().find("driving/is_constrained"))[:] = c
        view(geo.edges().find("aim_angle"))[:] = a
    print(f"[seal2] flap-hold RELEASE_AT frame {release_at} "
          f"(motion end {TOTAL_FRAMES}); settle beyond that", flush=True)
    scene.animator().insert(joint_obj, drive_flaps)

    # ---- carrier↔hub revolute bearing (created LAST, ∥ X through HUB_C0)
    half = 0.5 * HUB_HEIGHT
    bV = np.array([HUB_C0 + [-half, 0, 0], HUB_C0 + [+half, 0, 0]], dtype=np.float32)
    bearing_mesh = linemesh(bV, np.array([[0, 1]], dtype=np.int32))
    AffineBodyRevoluteJoint().apply_to(bearing_mesh, [carrier_slot], [hub_slot],
                                       _cf("CARRIER_JOINT_STRENGTH", CARRIER_JOINT_STRENGTH))
    # The backend requires the driving + limit joint lists to align 1:1 with the
    # revolute list (4 carton hinges + this bearing = 5). Give the bearing a
    # FREE-SPIN dummy driving (strength 0, never constrained) and a wide dummy
    # limit (strength 0) so the counts/indices match.
    AffineBodyDrivingRevoluteJoint().apply_to(bearing_mesh, np.array([0.0]))
    _bisp = bearing_mesh.edges().find("is_passive")
    if _bisp is not None: view(_bisp)[:] = 0
    _bisc = bearing_mesh.edges().find("driving/is_constrained")
    if _bisc is not None: view(_bisc)[:] = 0
    AffineBodyRevoluteJointLimit().apply_to(bearing_mesh, np.array([-100.0]),
                                            np.array([100.0]), np.array([0.0]))
    bearing_obj = scene.objects().create("bearing")
    bearing_obj.geometries().create(bearing_mesh)

    # ---- carrier STC drive along the seal path -----------------------
    def animate_carrier(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        M = np.eye(4); M[:3, :3] = _RY90; M[:3, 3] = _spindle_center(f)
        view(geo.instances().find(builtin.aim_transform))[0] = _mat4_to_uipc(M)
    scene.animator().insert(carrier_obj, animate_carrier)

    # ---- tape free-END press onto the −Z box side during the stick dwell -----
    # Spindle trajectory is UNCHANGED; instead, while the carrier is parked at the
    # −Z seam end (STICK phase S3..S2b), SPC-drive a patch of the tape's free-end
    # rows down onto the −Z wall's outer face so the tape end bonds to the box
    # side. Then release — the tape↔carton distance-lock holds it as the roll
    # traverses. (The tape already has an SPC applied.)
    def _vidt(i, j): return i * (TAPE_NZ + 1) + j
    K_PRESS   = int(_ci("TAPE_PRESS_ROWS", 8))          # last K rows = the wall-drape patch
    N_FLOP    = int(_ci("TAPE_FLOP_ROWS", 4))           # free-TIP rows for anti-flop + wrap (few → don't
    #                                                     fight the wound bonds → no descend blowup)
    Z_FACE    = -(HALF + 0.003)                          # just outside the −Z wall face (z=−0.1524)
    Y_TOP     = THK + WALL_H - 0.012                     # top of the press band, below the −Z wall top
    DS_Y      = TAPE_LENGTH / TAPE_NX
    _press_rows = list(range(max(TAPE_NX - K_PRESS, 0), TAPE_NX + 1))
    _flop_rows  = list(range(max(TAPE_NX - N_FLOP, 0), TAPE_NX + 1))

    use_cube = int(_cf("USE_CUBE_PRESS", 1)) != 0
    HANG = 0.02                                          # gap below the roll where the free end hangs

    def _drive_end(is_c, aim, live, rows, yfn, zval):
        for ridx, i in enumerate(rows):
            for j in range(TAPE_NZ + 1):
                k = _vidt(i, j); is_c[k] = 1
                aim[k] = np.array([live[k, 0], yfn(ridx), zval], dtype=np.float64).reshape(3, 1)

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim = view(geo.vertices().find(builtin.aim_position))
        is_c[:] = 0
        live = np.asarray(view(geo.positions())).reshape(-1, 3)
        if not use_cube:                                 # legacy: SPC drapes+presses the −Z wall
            if S3 <= f < S2b:
                _drive_end(is_c, aim, live, _press_rows, lambda r: Y_TOP - r * DS_Y, Z_FACE)
            return
        # cube mode: the SPC keeps the tape END rigidly tied to the spindle during the
        # initial motion (aim = initial pos + spindle displacement → exact relative pos,
        # no flop, and no fight with the wound bonds since it matches their rest), then
        # ramps it onto the −Z wall (spindle static) for the cube to press.
        if S1 <= f < S3:                                 # approach+descend: RIGID follow the spindle
            disp = _spindle_center(f) - PARK_POS
            for i in _press_rows:
                for j in range(TAPE_NZ + 1):
                    k = _vidt(i, j); is_c[k] = 1
                    aim[k] = (tape_pos_up[k] + disp).reshape(3, 1)
        elif S3 <= f < S3 + 40:                          # stick: ramp the end onto the −Z wall while the
            disp3 = _spindle_center(S3) - PARK_POS       # cube is still static; RELEASE at S3+40 (when the
            t = min((f - S3) / 40.0, 1.0)                # cube starts moving in) so the cube alone handles it
            for ridx, i in enumerate(_press_rows):
                for j in range(TAPE_NZ + 1):
                    k = _vidt(i, j); is_c[k] = 1
                    pf = tape_pos_up[k] + disp3          # rigid-follow pose at S3
                    pw = np.array([tape_pos_up[k][0], Y_TOP - ridx * DS_Y, -HALF])  # −Z wall face
                    aim[k] = (pf + (pw - pf) * t).reshape(3, 1)
        # f >= S3+40: SPC released — the cube presses/holds the tape end onto the −Z wall
    scene.animator().insert(tape_obj, animate_tape)

    # ---- press-cube STC drive: press −Z wall @ stick dwell, +Z wall @ end,
    #      then leave the box (path defined below, tied to the schedule) ----
    def animate_cube(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        M = np.eye(4); M[:3, 3] = _cube_center(f)
        view(geo.instances().find(builtin.aim_transform))[0] = _mat4_to_uipc(M)
    scene.animator().insert(cube_obj, animate_cube)

    world.init(scene)

    # ---- restore β + bonded locks (remap: hub first, tape shifts) -----
    if pair_state is not None:
        keys, betas = pair_state
        acc = world.features().find(RCCAdhesionStateAccessorFeature)
        if acc is not None and len(betas) > 0:
            acc.load_pt_state(keys, betas)
            print(f"[seal2] restored β: n={len(betas)}", flush=True)
    lp = L.load_tape_locked_pairs(ASSET)
    if lp is not None:
        topos, lbetas = lp
        hub_nv = len(np.asarray(view(hub_sc.positions())).reshape(-1, 3))
        base_nv = len(np.asarray(view(base_sc.positions())).reshape(-1, 3))
        flaps_nv = sum(len(np.asarray(view(s.positions())).reshape(-1, 3)) for s in flap_scs)
        carrier_nv = len(np.asarray(view(carrier_sc.positions())).reshape(-1, 3))
        cube_nv = len(np.asarray(view(cube_sc.positions())).reshape(-1, 3))
        shift = base_nv + flaps_nv + carrier_nv + cube_nv   # ABD created between hub and tape
        topos = np.asarray(topos).copy()
        topos = np.where(topos >= hub_nv, topos + shift, topos)
        bpt = world.features().find(RCCBondedPTStateAccessorFeature)
        if bpt is not None:
            bpt.seed_locks(topos, lbetas, 0.9)
            print(f"[seal2] seeded {len(lbetas)} locks (tape shift +{shift}); "
                  f"locked_pair_count={bpt.locked_pair_count()}", flush=True)

    return {"engine": engine, "world": world, "scene": scene, "scene_io": SceneIO(scene),
            "tape_slot": tape_slot, "carrier_slot": carrier_slot, "hub_slot": hub_slot,
            "world_": world}


# ---- carrier path: park -> approach(-Z end) -> slow descend -> stick(SPC press)
#      -> CONSTANT-SPEED traverse: seam(+Z) -> out(past +Z edge) -> down(+Z face).
SP_PAD, SP_APPROACH = 50, 100
SP_DESCEND = 540        # slower descend (≈0.66x traverse speed; gentler contact)
SP_STICK   = 190       # spindle holds at −Z end: SPC positions the tape end, then the cube presses + lifts
SP_END     = 60
RIDE_DY_DEFAULT = 0.045                 # carrier centre above SEAM_Y (roll bottom ~at seam)
TRAVERSE_SPEED  = 0.0007                # m/frame — CONSTANT over the whole seam->out->down pass
OUT_DZ  = 0.10                          # how far out PAST the +Z wall face (more out)
DOWN_Y  = 0.05                          # bottom of the +Z-face descent (more down; box top 0.206)
PARK_Y  = SEAM_Y + 0.32                 # roll parked high, clear of the open flaps
PARK_POS = np.array([SEAM_X, PARK_Y, 0.0])


def _build_traj(ride_dy):
    """(Re)compute carrier waypoints, per-segment frames (∝ length => constant
    speed), phase breakpoints and TOTAL_FRAMES for a given ride height."""
    global _RIDE_Y, PARK, _W1, _W2, _W3, _W4, _W5, SIDE_Z, SP_RUN, SP_OUT, SP_DOWN, \
        S0, S1, S2, S3, S2b, S4, S4b, S5, TOTAL_FRAMES
    _RIDE_Y = SEAM_Y + ride_dy
    SIDE_Z  = HALF + OUT_DZ
    PARK = PARK_POS
    _W1 = np.array([SEAM_X, PARK_Y,  SEAM_Z0])       # high above −Z end
    _W2 = np.array([SEAM_X, _RIDE_Y, SEAM_Z0])       # ride, −Z end (stick here)
    _W3 = np.array([SEAM_X, _RIDE_Y, SEAM_Z1])       # +Z end (along the seam)
    _W4 = np.array([SEAM_X, _RIDE_Y, SIDE_Z])        # out past the +Z edge
    _W5 = np.array([SEAM_X, DOWN_Y,  SIDE_Z])        # down the +Z face
    SP_RUN  = max(round(abs(SEAM_Z1 - SEAM_Z0) / TRAVERSE_SPEED), 1)   # seam
    SP_OUT  = max(round(abs(SIDE_Z - SEAM_Z1)   / TRAVERSE_SPEED), 1)  # out
    SP_DOWN = max(round(abs(_RIDE_Y - DOWN_Y)   / TRAVERSE_SPEED), 1)  # down
    S0  = PRESS_END + 70
    S1  = S0 + SP_PAD
    S2  = S1 + SP_APPROACH
    S3  = S2 + SP_DESCEND
    S2b = S3 + SP_STICK
    S4  = S2b + SP_RUN          # reaches +Z end
    S4b = S4 + SP_OUT           # reaches out-past-edge
    S5  = S4b + SP_DOWN         # reaches bottom of +Z face
    TOTAL_FRAMES = S5 + SP_END


_build_traj(RIDE_DY_DEFAULT)


# ---- press-cube path: park −Z → PRESS −Z wall (stick dwell) → swing wide in
#      +X around the box → PRESS +Z wall (after the roll descends) → leave. -----
def _cube_pos(x, z, y=CUBE_Y_PRESS):
    return np.array([x, y, z], dtype=np.float64)


def _cube_keyframes():
    # STRICT ALTERNATION: the cube moves ONLY while the spindle is static (the stick
    # dwell S3..S2b, and the post-S5 hold). It is held fixed during the spindle's
    # approach/descend (phase 1) and its seam/cross/descend (phase 3).
    return [
        (0,          _cube_pos(0.0,   -CUBE_Z_PARK)),                 # FAR −Z park — STATIC through phase 1
        (S3 + 40,    _cube_pos(0.0,   -CUBE_Z_PARK)),                 # still static while the SPC drapes the end
        # -- phase 2: spindle static at −Z end; cube moves --
        (S3 + 90,    _cube_pos(0.0,    CUBE_Z_NEG)),                  # move IN + 1st PRESS-down (−Z wall)
        (S3 + 150,   _cube_pos(0.0,    CUBE_Z_NEG)),                  # hold (tape↔wall bond forms)
        (S2b,        _cube_pos(0.0,   -CUBE_Z_PRE)),                  # LIFT to the −Z stand-off
        # -- phase 3: cube STATIC standby while the spindle does seam/cross/descend --
        (S5 + 40,    _cube_pos(0.0,   -CUBE_Z_PRE)),                  # hold (+40 = brief settle after the spindle stops)
        # -- phase 4: spindle static at +Z bottom; cube goes OVER THE TOP, presses, leaves --
        (S5 + 90,    _cube_pos(0.0,   -CUBE_Z_PRE, CUBE_Y_HIGH)),     # LIFT up above the box
        (S5 + 180,   _cube_pos(0.0,    CUBE_Z_PRE, CUBE_Y_HIGH)),     # translate OVER THE TOP to +Z
        (S5 + 240,   _cube_pos(0.0,    CUBE_Z_PRE)),                  # descend to the +Z stand-off
        (S5 + 300,   _cube_pos(0.0,    CUBE_Z_POS)),                  # 2nd PRESS-down (+Z wall, deeper)
        (S5 + 420,   _cube_pos(0.0,    CUBE_Z_POS)),                  # HOLD 120f (tape↔wall bond forms)
        (S5 + 480,   _cube_pos(0.0,    CUBE_Z_PRE)),                  # LIFT off the +Z wall
        (S5 + 540,   _cube_pos(0.0,    CUBE_Z_PRE, CUBE_Y_HIGH)),     # LIFT up
        (S5 + 630,   _cube_pos(0.0,   -CUBE_Z_PRE, CUBE_Y_HIGH)),     # LEAVE over the top, back to −Z
    ]


def _cube_center(f):
    kfs = _cube_keyframes()
    if f <= kfs[0][0]:
        return kfs[0][1].copy()
    for (f0, p0), (f1, p1) in zip(kfs, kfs[1:]):
        if f < f1:
            t = (f - f0) / (f1 - f0) if f1 > f0 else 1.0
            return p0 + (p1 - p0) * t
    return kfs[-1][1].copy()


CUBE_END = None   # last cube keyframe frame; set in run() once the schedule is final


def _lin(f, f0, f1, a, b):
    t = 0.0 if f1 <= f0 else min(max((f - f0) / (f1 - f0), 0.0), 1.0)
    return a + (b - a) * t


def _spindle_center(f):
    if f < S1:  return PARK.copy()
    if f < S2:  return _lin(f, S1, S2, PARK, _W1)     # approach
    if f < S3:  return _lin(f, S2, S3, _W1, _W2)      # descend (slow, linear)
    if f < S2b: return _W2.copy()                     # stick dwell (hold; SPC presses end)
    if f < S4:  return _lin(f, S2b, S4, _W2, _W3)     # seam   (constant speed)
    if f < S4b: return _lin(f, S4, S4b, _W3, _W4)     # out    (same speed)
    if f < S5:  return _lin(f, S4b, S5, _W4, _W5)     # down   (same speed)
    return _W5.copy()


def _phase(f):
    if f < S0:  return "close"
    if f < S2:  return "approach"
    if f < S3:  return "descend"
    if f < S2b: return "stick"
    if f < S4:  return "run-seam"
    if f < S4b: return "out"
    if f < S5:  return "down"
    return "hold"


def _parse_set_args():
    argv = list(sys.argv[1:]); cfg, i = {}, 0
    while i < len(argv):
        if argv[i] == "--set" and i + 1 < len(argv) and "=" in argv[i + 1]:
            k, v = argv[i + 1].split("=", 1); cfg[k.strip()] = v.strip(); i += 2
        else:
            i += 1
    return cfg


def run():
    cfg = _parse_set_args()
    Logger.set_level(Logger.Level.Warn)
    # let RIDE_DY override the ride height (recomputes the whole constant-speed path)
    if "RIDE_DY" in cfg:
        _build_traj(float(cfg["RIDE_DY"]))
    sim = build_demo(cfg)
    total_frames = int(cfg.get("FRAMES", TOTAL_FRAMES))
    record_dir = cfg.get("RECORD_DIR") or os.path.join(AssetDir.output_path(__file__), "frames")
    every_n = int(cfg.get("RECORD_EVERY", 10))
    zoom = float(cfg.get("RECORD_ZOOM", 1.0))

    gq_v = np.array([[-0.4, 0, -0.4], [0.4, 0, -0.4], [0.4, 0, 0.4], [-0.4, 0, 0.4]], dtype=np.float64)
    gq_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    def setup_extras(ps):
        gm = ps.register_surface_mesh("ground", gq_v, gq_f)
        gm.set_color((0.55, 0.55, 0.58)); gm.set_transparency(0.5)

    def on_progress(f, t):
        cs = np.asarray(view(sim["carrier_slot"].geometry().transforms())[0]).reshape(4, 4)[:3, 3]
        bpt = sim["world_"].features().find(RCCBondedPTStateAccessorFeature)
        lk = int(bpt.locked_pair_count()) if bpt is not None else -1
        tp = sim["tape_slot"].geometry().positions().view().reshape(-1, 3)
        print(f"[seal2] frame {f}/{t} ({f/t*100:.1f}%) phase={_phase(max(f-1,0))} "
              f"carrier=({cs[0]:+.3f},{cs[1]:+.3f},{cs[2]:+.3f}) locked={lk} "
              f"tape_y∈[{tp[:,1].min():.3f},{tp[:,1].max():.3f}]", flush=True)

    print(f"[seal2] scotch3850 roll on carrier; seam Z∈[{SEAM_Z0:+.3f},{SEAM_Z1:+.3f}] x=0 y={SEAM_Y:.4f}. "
          f"phases@[S0{S0},approach{S2},descend{S3},stick{S2b},run{S4},cross{S5}] TOTAL={total_frames}. -> {record_dir}", flush=True)
    L.record_demo_to_pngs(sim=sim, total_frames=total_frames, output_dir=record_dir,
                          every_n=every_n, up_dir="y_up", mesh_name="carton_seal_v2",
                          setup_extras_fn=setup_extras, zoom=zoom, on_progress=on_progress,
                          bbox_extent_override=max(BASE * 1.5, 0.5))


if __name__ == "__main__":
    run()
