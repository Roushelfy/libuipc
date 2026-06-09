"""
Wind a sticky tape onto a fixed deformable rod (Option B — hub-orbit).

A pre-wound tape roll (loaded from a wind asset, like the drop demo) lies
on its SIDE on the ground (hub axis horizontal, +z). A two-end-fixed
DEFORMABLE rod (a thin tube shell, the libuipc analogue of the
wire_harnessing wire) is suspended in the air, axis parallel to the roll
(+z). The sequence:

  HOLD   : everything settles on the ground under gravity.
  PRESS  : SoftPositionConstraint lifts the tape's free-end ROW straight
           up and presses it onto the underside of the rod, where the
           tape-rod RCC adhesion fires (β rises).
  BOND   : hold the free end on the rod so the bond strengthens; the hub
           (free ABD) hangs below, suspended by the now-anchored tape.
  ORBIT  : the free-end SPC releases; the hub is driven by a
           SoftTransformConstraint on a CIRCLE around the rod axis —
           translation ONLY, orientation pinned (no self-rotation). Each
           orbit wraps one turn onto the rod; tape is paid out / peeled
           off the roll (soft-adhesion debond, or bonded-PT release if
           --bonded). N_TURNS orbits = N turns.
  SETTLE : hub held at the final orbit point; the wrap relaxes.

Almost everything is a knob (CLI `--set KEY=VALUE`, or a preset). The
defaults are a starting point to iterate from, not a tuned result.

Run (interactive):
    python/.venv/bin/python python/examples/rcc_adhesive_tape_rod_wind_demo.py \
        --asset output/rcc_tape_wind_drop/e5e7_dhat2_abd002_nodal002w_cnct1/wound_asset.npz

Headless render:
    ... --set RECORD_DIR=output/rod_wind/run1
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ModuleNotFoundError:
    ps = None
    psim = None

from uipc import (
    Logger, Matrix4x4, Engine, World, Scene, SceneIO, Animation, view, builtin,
)
from uipc.geometry import ground
from uipc.core import RCCAdhesionStateAccessorFeature, RCCBondedPTStateAccessorFeature
from uipc.constitution import (
    AffineBodyConstitution,
    NeoHookeanShell,
    DiscreteShellBending,
    SoftPositionConstraint,
    SoftTransformConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ----------------------------------------------------------------------
# CLI / preset resolution (re-uses UNWIND_PRESETS — same material +
# adhesion semantics as the wound asset).
# ----------------------------------------------------------------------
_CFG = L.parse_tape_cli(L.UNWIND_PRESETS)


def _cfg_f(key, default):  return float(_CFG.get(key, default))
def _cfg_i(key, default):  return int(_CFG.get(key, default))


# ---- asset to load (the wound roll = the tape supply) ----
# Default: the e5e8-cnct1 wound roll (soft adhesion, firmly settled:
# β>0.9 on ~99.8% of pairs, 3767 bonded locks). Override with
# `--asset <path-to.npz>`.
_DEFAULT_ASSET = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_winding", "e5e8-cnct1.npz")
ASSET_IN_PATH = _CFG.get("__asset_arg__") or _DEFAULT_ASSET

# ---- materials / adhesion (asset overrides these in build_demo) ----
HUB_KAPPA         = _cfg_f("HUB_KAPPA", 1.0e8)
HUB_MASS_DENSITY  = _cfg_f("HUB_MASS_DENSITY", 1000.0)

# ---- deformable rod (faithful to wire_harnessing's wire) ----
ROD_R             = _cfg_f("ROD_R", 0.003)          # tube radius (m)
ROD_LENGTH        = _cfg_f("ROD_LENGTH", 0.41)      # chord length (m)
ROD_SEGMENTS      = _cfg_i("ROD_SEGMENTS", 64)      # rings along axis
ROD_SIDES         = _cfg_i("ROD_SIDES", 12)         # cross-section polygon
ROD_YOUNGS        = _cfg_f("ROD_YOUNGS", 5.0e7)     # 50 MPa
ROD_POISSON       = _cfg_f("ROD_POISSON", 0.4)
ROD_DENSITY       = _cfg_f("ROD_DENSITY", 1000.0)
ROD_THICKNESS     = _cfg_f("ROD_THICKNESS", 5.0e-4)
ROD_BENDING       = _cfg_f("ROD_BENDING", 1.0e-2)
ROD_PIN_STRENGTH  = _cfg_f("ROD_PIN_STRENGTH", 1.0e6)
# rod placement (y-up world). Axis +z, centered at (ROD_CX, ROD_Y, ROD_CZ).
ROD_Y             = _cfg_f("ROD_Y", 0.16)           # height of rod axis (m)
ROD_CX            = _cfg_f("ROD_CX", 0.0)
ROD_CZ            = _cfg_f("ROD_CZ", 0.0)

# ---- orbit (hub motion) ----
HANG_RADIUS       = _cfg_f("HANG_RADIUS", -1.0)     # orbit radius (m); <0 ⇒ auto = the hub's natural risen hang radius, captured at orbit entry
N_TURNS           = _cfg_f("N_TURNS", 3.0)          # number of orbits = turns wound
WRAP_PITCH        = _cfg_f("WRAP_PITCH", 0.0)       # axial advance per turn (m). 0=stacked, >0=helical
ORBIT_DIR         = _cfg_f("ORBIT_DIR", 1.0)        # +1 / -1 orbit sense

# ---- press / drape onto the rod ----
# Anchor a row BACK from the tip (so the tip drapes OVER the rod) and press
# that row onto the rod. The rows beyond the anchor stay free → drape over.
PRESS_BACK        = _cfg_i("PRESS_BACK", 15)        # # of overhang rows; fold is always a 180° half-circle, this sets its RADIUS (r=PRESS_BACK*ds_len/π)

# ---- adhesion-to-rod (tape ↔ rod). Defaults mirror tape-tape unless set ----
STICKY            = _cfg_i("STICKY", 0)             # 0=double-sided (robust), ±1=single

# ---- timeline (dt = 0.01) ----
#   HOLD  → settle on ground
#   PRESS → DRAPE: lift the anchor row onto the rod top; the tip overhangs
#           over the far side (the tape goes up & over the rod).
#   FOLD  → curl the overhanging tip in a half-circle around the rod and
#           press it down onto the incoming tape below (tape–tape bond) →
#           a secure first wrap.
#   BOND  → hold the wrap so the bond strengthens.
#   ORBIT → release the tape SPC; the hub orbits → more turns.
HOLD_FRAMES       = _cfg_i("HOLD_FRAMES", 40)
PRESS_FRAMES      = _cfg_i("PRESS_FRAMES", 120)
FOLD_FRAMES       = _cfg_i("FOLD_FRAMES", 150)      # curl the overhang around the rod onto the incoming
BOND_FRAMES       = _cfg_i("BOND_FRAMES", 150)
ORBIT_FRAMES      = _cfg_i("ORBIT_FRAMES", 1500)
SETTLE_FRAMES     = _cfg_i("SETTLE_FRAMES", 150)

SPC_STRENGTH      = _cfg_f("SPC_STRENGTH", 1.0e9)
STC_ETA_P         = _cfg_f("STC_ETA_P", 1.0e8)      # hub translation strength
STC_ETA_A         = _cfg_f("STC_ETA_A", 1.0e8)      # hub rotation strength (>0 ⇒ no self-rotation)

_T0 = HOLD_FRAMES
_Tp = _T0 + PRESS_FRAMES        # drape done
_Tf = _Tp + FOLD_FRAMES         # fold done
_T2 = _Tf + BOND_FRAMES         # bond done → orbit start
_T3 = _T2 + ORBIT_FRAMES
TOTAL_FRAMES      = _T3 + SETTLE_FRAMES


def phase_at(f: int) -> str:
    if f < _T0: return "hold"
    if f < _Tp: return "press"
    if f < _Tf: return "fold"
    if f < _T2: return "bond"
    if f < _T3: return "orbit"
    return "settle"


# ----------------------------------------------------------------------
# Helpers (tape topology / asset standing — copied from the drop demo so
# this script is self-contained).
# ----------------------------------------------------------------------
def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


def _make_tape_topology_tris(NX, NZ):
    """Identical triangle layout to the wind/drop demos so
    set_sticky_side tags the same face."""
    n_width = NZ + 1
    def vid(i, j): return i * n_width + j
    tris = []
    for i in range(NX):
        for j in range(NZ):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            tris.append([a, b, c])
            tris.append([a, c, d])
    return np.asarray(tris, dtype=np.int32)


def _make_tape_sc(positions, tris):
    from uipc.geometry import trimesh, label_surface
    sc = trimesh(positions.astype(np.float64), tris)
    label_surface(sc)
    return sc


def _straight_rest_positions(R_anchor, length, width, NX, NZ):
    """The straight-tangent layout the wind demo used as the FEM rest
    reference (REST_FROM_WIND=1)."""
    dy = length / NX
    dz = width / NZ
    verts = np.empty(((NX + 1) * (NZ + 1), 3), dtype=np.float64)
    n_width = NZ + 1
    for i in range(NX + 1):
        for j in range(NZ + 1):
            verts[i * n_width + j] = (R_anchor, i * dy, j * dz - 0.5 * width)
    return verts


def _lay_on_side(tape_pos, hub_T, hub_R_outer, ground_clearance, target_xz):
    """Keep the asset's native orientation (hub axis +z, horizontal) — the
    roll lies on its SIDE — and translate so its lowest point sits
    `ground_clearance` above y=0 and its hub center is at target_xz=(x, z)."""
    hub_center = hub_T[:3, 3].copy()
    hub_bottom_y = float(hub_center[1]) - hub_R_outer
    min_y = min(float(tape_pos[:, 1].min()), hub_bottom_y)
    shift = np.array([target_xz[0] - hub_center[0],
                      ground_clearance - min_y,
                      target_xz[1] - hub_center[2]], dtype=np.float64)
    tape_new = tape_pos + shift
    hub_T_new = hub_T.copy()
    hub_T_new[:3, 3] += shift
    return tape_new, hub_T_new


def smooth_lerp(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * t))


# ----------------------------------------------------------------------
def build_demo(adhesion_on: bool = True,
               bonded: bool = False,
               beta_lock_threshold: float = 0.9,
               kappa: float = 1.0e8,
               release_force: float = 1.0e30):
    L.apply_log_level(_CFG, default="warn")

    if not os.path.isfile(ASSET_IN_PATH):
        raise SystemExit(
            f"Asset not found: {ASSET_IN_PATH}\n"
            f"Pass an existing wound roll with --asset <path-to wound_asset.npz>, "
            f"or run the wind demo to make one.")

    hub_T, tape_pos, params, pair_state, tape_vel = L.load_tape_asset(ASSET_IN_PATH)
    HUB_R_OUTER = float(params["HUB_R_OUTER"])
    HUB_R_INNER = float(params["HUB_R_INNER"])
    HUB_HEIGHT  = float(params["HUB_HEIGHT"])
    TAPE_LENGTH = float(params["TAPE_LENGTH"])
    TAPE_WIDTH  = float(params["TAPE_WIDTH"])
    TAPE_NX     = int(params["TAPE_NX"])
    TAPE_NZ     = int(params["TAPE_NZ"])

    def _res(key):  return L.resolve_param(_CFG, params, key)
    D_HAT             = _res("D_HAT")
    TAPE_THICKNESS    = _res("TAPE_THICKNESS")
    TAPE_YOUNGS       = _res("TAPE_YOUNGS")
    BENDING_STIFFNESS = _res("BENDING_STIFFNESS")
    TAPE_POISSON      = _res("TAPE_POISSON")
    TAPE_MASS_DENSITY = _res("TAPE_MASS_DENSITY")
    ADH_CN            = _res("ADH_CN")
    ADH_CT            = _res("ADH_CT")
    ADH_W             = _res("ADH_W")
    ADH_ETA           = _res("ADH_ETA")
    ADH_BONDING_RATE  = _res("ADH_BONDING_RATE")
    ADH_INITIAL_BETA  = _res("ADH_INITIAL_BETA")

    print(f"[rod-wind] asset={os.path.relpath(ASSET_IN_PATH)} "
          f"tape ({TAPE_NX+1}×{TAPE_NZ+1}), L={TAPE_LENGTH:.3f} m, "
          f"hub R∈[{HUB_R_INNER:.4f},{HUB_R_OUTER:.4f}]")

    # ---- lay the roll on its side on the ground, centered under the rod
    GROUND_CLEARANCE = TAPE_THICKNESS + 0.5 * D_HAT
    tape_pos_lay, hub_T_lay = _lay_on_side(
        tape_pos, hub_T, HUB_R_OUTER, GROUND_CLEARANCE, (ROD_CX, ROD_CZ))
    R_fixed = hub_T_lay[:3, :3].copy()   # hub orientation pinned during the orbit

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
    _off = _CFG.get("RCC_ADHESION_NORMAL_OFFSET_COEFF")
    if _off is not None:
        config["rcc_adhesion_normal_offset_coeff"] = float(_off)
    if bonded:
        config["rcc_bonded_pt_enabled"] = 1
        config["rcc_bonded_pt_skip_ccd"] = int(_CFG.get("SKIP_CCD", -1))
        config["rcc_bonded_pt_beta_lock_threshold"] = beta_lock_threshold
        config["rcc_bonded_pt_lock_face_interior_only"] = (
            1 if L.cfg_flag(_CFG, "LOCK_FACE_INTERIOR_ONLY", default=False) else 0)
        config["rcc_bonded_pt_energy_model"] = "abd_ortho"
        config["rcc_bonded_pt_kappa"] = float(_CFG.get("RCC_KAPPA", kappa))
        config["rcc_bonded_pt_release_force"] = release_force
    L.apply_solver_overrides(config, _CFG, params=params)
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    dsb = DiscreteShellBending()
    spc = SoftPositionConstraint()
    stc = SoftTransformConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    hub_contact  = tabular.default_element()         # hub + ground
    tape_contact = tabular.create("tape")
    rod_contact  = tabular.create("rod")
    tabular.insert(tape_contact, tape_contact, 0.5, 1.0e9)
    tabular.insert(tape_contact, hub_contact,  0.5, 1.0e9)
    tabular.insert(tape_contact, rod_contact,  0.5, 1.0e9)
    tabular.insert(rod_contact,  hub_contact,  0.5, 1.0e9)
    tabular.insert(rod_contact,  rod_contact,  0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            tabular, Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0, enabled=False)
        for (A, B, b0) in [(tape_contact, tape_contact, ADH_INITIAL_BETA),
                           (tape_contact, hub_contact,  ADH_INITIAL_BETA),
                           (tape_contact, rod_contact,  0.0)]:
            adhesive.set(tabular, A, B,
                         Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
                         bonding_rate=ADH_BONDING_RATE, p0=0.0,
                         initial_beta=b0, enabled=True)
        if bonded:
            _tt_rf = float(_CFG.get("BONDED_TAPE_RELEASE_FORCE", -1.0))
            _th_rf = float(_CFG.get("BONDED_HUB_RELEASE_FORCE", -1.0))
            for (A, B, rf) in [(tape_contact, tape_contact, _tt_rf),
                               (tape_contact, hub_contact,  _th_rf),
                               (tape_contact, rod_contact,  -1.0)]:
                adhesive.set_bonded(tabular, A, B,
                                    lock_threshold=-1.0, release_strain=-1.0,
                                    release_gap=-1.0, release_slip=-1.0,
                                    release_force=rf)

    # ---- hub (ABD; free during hold/press/bond, STC-driven during orbit) ----
    hub_sc = L.make_ring_hub(R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
                             height=HUB_HEIGHT, n_radial=48, center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    hub_contact.apply_to(hub_sc)
    stc.apply_to(hub_sc, np.array([STC_ETA_P, STC_ETA_A], dtype=np.float64))
    view(hub_sc.transforms())[0] = _mat4_to_uipc(hub_T_lay)
    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- deformable rod (tube shell, both end rings pinned) ----
    rod_sc, rod_pin = L.make_rod_tube_shell(
        R=ROD_R, length=ROD_LENGTH, segments=ROD_SEGMENTS, n_sides=ROD_SIDES,
        center=(ROD_CX, ROD_Y, ROD_CZ))
    rod_moduli = ElasticModuli2D.youngs_poisson(ROD_YOUNGS, ROD_POISSON)
    nhs.apply_to(rod_sc, rod_moduli, mass_density=ROD_DENSITY, thickness=ROD_THICKNESS)
    dsb.apply_to(rod_sc, ROD_BENDING)
    rod_contact.apply_to(rod_sc)
    spc.apply_to(rod_sc, ROD_PIN_STRENGTH)
    # static pin: set is_constrained=1 + aim=rest on the two end rings, and
    # never animate the rod → ends stay clamped for the whole sim.
    rod_isc = view(rod_sc.vertices().find(builtin.is_constrained))
    rod_aim = view(rod_sc.vertices().find(builtin.aim_position))
    rod_rest = np.asarray(view(rod_sc.positions())).reshape(-1, 3)
    for k in rod_pin:
        rod_isc[int(k)] = 1
        rod_aim[int(k)] = rod_rest[int(k)].reshape(3, 1)
    rod_obj = scene.objects().create("rod")
    rod_geo, _ = rod_obj.geometries().create(rod_sc)

    # ---- tape (current = wound-laid; rest = wind's straight strip) ----
    tris = _make_tape_topology_tris(TAPE_NX, TAPE_NZ)
    current_sc = _make_tape_sc(tape_pos_lay, tris)
    R_anchor_rest = HUB_R_OUTER + TAPE_THICKNESS + 0.5 * D_HAT
    rest_positions = _straight_rest_positions(
        R_anchor_rest, TAPE_LENGTH, TAPE_WIDTH, TAPE_NX, TAPE_NZ)
    rest_sc = _make_tape_sc(rest_positions, tris)

    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    for sc in (current_sc, rest_sc):
        nhs.apply_to(sc, moduli, mass_density=TAPE_MASS_DENSITY, thickness=TAPE_THICKNESS)
        dsb.apply_to(sc, BENDING_STIFFNESS)
    tape_contact.apply_to(current_sc)
    spc.apply_to(current_sc, SPC_STRENGTH)
    if adhesion_on:
        RCCAdhesive.set_sticky_side(current_sc, STICKY)

    if tape_vel is not None and tape_vel.shape == tape_pos.shape:
        # lay-on-side keeps orientation ⇒ velocity unchanged (pure translation)
        if current_sc.vertices().find("velocity") is None:
            current_sc.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        view(current_sc.vertices().find("velocity"))[:] = tape_vel.reshape(-1, 3, 1)

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(current_sc, rest_sc)

    # ---- ground ----
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    # ---- free-end DRAPE + FOLD press ----
    # PRESS (drape): drive ONLY the anchor row (PRESS_BACK rows from the tip)
    #   up to the rod TOP. The tip rows stay free → the tape goes up & over
    #   the rod, tip overhanging the far side (the orbit1 frame-40 look).
    # FOLD: curl the overhang — drive rows anchor_i..NX onto the rod surface,
    #   laid by arc length STARTING AT THE TOP (θ=π) and wrapping in ORBIT_DIR
    #   (r̂(θ)=(sinθ,-cosθ): θ=π→top, θ=2π→bottom). PRESS_BACK≈6 ⇒ ~half-circle
    #   so the tip folds down onto the incoming tape (tape–tape bond) at the
    #   rod bottom. Held through BOND, released at ORBIT.
    def vid(i, j): return i * (TAPE_NZ + 1) + j
    band   = TAPE_THICKNESS + 0.5 * D_HAT
    wrap_R = ROD_R + band
    ds_len = TAPE_LENGTH / TAPE_NX
    anchor_i = max(TAPE_NX - PRESS_BACK, 1)
    rod_top_y = ROD_Y + wrap_R
    anchor_ids = [vid(anchor_i, j) for j in range(TAPE_NZ + 1)]
    # FOLD targets: the overhang folds as a clean HALF-CIRCLE (hairpin).
    # The tip ALWAYS travels 180° and lands on the incoming tape below the
    # rod; a longer overhang only makes the half-circle RADIUS bigger
    # (r = overhang_len / π), not the angle. The anchor (driven to the rod
    # top in PRESS) is the TOP of the diameter; the tip is the BOTTOM of the
    # diameter (= straight below the rod, on the incoming). Bulge = ORBIT_DIR.
    L_over  = max(PRESS_BACK, 1) * ds_len
    r_fold  = L_over / np.pi
    cy_fold = (ROD_Y + wrap_R) - r_fold         # α=0 ⇒ anchor at rod top
    fold_ids, fold_xy, fold_alpha = [], [], []
    for k in range(anchor_i, TAPE_NX + 1):
        alpha = ((k - anchor_i) * ds_len) / r_fold   # 0 (top/anchor) … π (tip on incoming)
        tx = ROD_CX + ORBIT_DIR * r_fold * np.sin(alpha)
        ty = cy_fold + r_fold * np.cos(alpha)
        for j in range(TAPE_NZ + 1):
            fold_ids.append(vid(k, j))
            fold_xy.append((tx, ty))
            fold_alpha.append(alpha)
    fold_ids   = np.asarray(fold_ids,   dtype=np.int64)
    fold_xy    = np.asarray(fold_xy,    dtype=np.float64)
    fold_alpha = np.asarray(fold_alpha, dtype=np.float64)
    drape_state = {"start": None}
    fold_state  = {"start": None}

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim  = view(geo.vertices().find(builtin.aim_position))
        is_c[:] = 0
        if f < _T0 or f >= _T2:          # hold + (orbit/settle): tape unconstrained
            drape_state["start"] = None
            fold_state["start"] = None
            return
        live = np.asarray(view(geo.positions())).reshape(-1, 3)
        if f < _Tp:                       # DRAPE: anchor row → rod top
            if drape_state["start"] is None:
                drape_state["start"] = live[anchor_ids].copy()
            t = min((f - _T0) / max(PRESS_FRAMES, 1), 1.0)
            s = drape_state["start"]
            for jj, k in enumerate(anchor_ids):
                tgt = np.array([smooth_lerp(s[jj, 0], ROD_CX, t),
                                smooth_lerp(s[jj, 1], rod_top_y, t),
                                s[jj, 2]], dtype=np.float64)
                is_c[k] = 1
                aim[k] = tgt.reshape(3, 1)
        else:                             # FOLD (progressive lay-down) then hold through BOND
            # Wrap front sweeps α: 0→π. A row is pinned to its arc target only
            # once the front reaches its α (laid down); rows ahead of the front
            # stay free (trailing) → the tape wraps AROUND the rod, never pulled
            # straight through it. By BOND (t≥1) all rows are laid and held.
            if fold_state["start"] is None:
                fold_state["start"] = live[fold_ids, 2].copy()   # capture z (width)
            t = min((f - _Tp) / max(FOLD_FRAMES, 1), 1.0)
            alpha_front = np.pi * t
            sz = fold_state["start"]
            for idx in range(len(fold_ids)):
                if fold_alpha[idx] > alpha_front + 1e-9:
                    continue                   # not yet reached by the wrap front → free
                k = int(fold_ids[idx])
                tx, ty = fold_xy[idx]
                is_c[k] = 1
                aim[k] = np.array([tx, ty, sz[idx]], dtype=np.float64).reshape(3, 1)

    scene.animator().insert(tape_obj, animate_tape)

    # ---- hub orbit: SoftTransformConstraint, translation-only circle.
    # During PRESS/BOND the hub is FREE and rises with the lifted tape,
    # ending up hanging just below the rod. At orbit entry we capture that
    # natural (radius, angle) and sweep the hub around the rod axis from
    # exactly there — no jump, no fighting the hang. Orientation is pinned
    # to R_fixed ⇒ NO self-rotation. φ=0 ⇒ straight below the rod.
    MIN_ORBIT_R = ROD_R + HUB_R_OUTER + 0.003
    orbit_state = {"init": None}

    def animate_hub(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.instances().find(builtin.is_constrained))
        aim  = view(geo.instances().find(builtin.aim_transform))
        if f < _T2:                       # free before the orbit (rises with the lift)
            is_c[0] = 0
            orbit_state["init"] = None
            return
        cur = np.asarray(view(geo.transforms())[0]).reshape(4, 4)
        c0 = cur[:3, 3]
        if orbit_state["init"] is None:
            dx, dy = float(c0[0] - ROD_CX), float(c0[1] - ROD_Y)
            r_nat = float(np.hypot(dx, dy))
            r = HANG_RADIUS if HANG_RADIUS > 0 else max(r_nat, MIN_ORBIT_R)
            phi0 = float(np.arctan2(dx, -dy))   # so point(φ0)=current center
            # Capture the hub's ACTUAL current orientation (it may have tilted
            # while free during press/bond). Holding this — not the stale
            # build-time R_fixed — makes the STC engage at zero initial error
            # (no snap / no Newton blow-up) and pins orientation ⇒ no self-rotation.
            orbit_state["init"] = (r, phi0, float(c0[2]), cur[:3, :3].copy())
            print(f"[rod-wind] orbit start: r={r*1e3:.1f} mm "
                  f"(natural {r_nat*1e3:.1f} mm), φ0={np.degrees(phi0):.0f}°", flush=True)
        r, phi0, z0, R_hold = orbit_state["init"]
        frac = min((f - _T2) / max(ORBIT_FRAMES, 1), 1.0)
        sweep = ORBIT_DIR * 2.0 * np.pi * N_TURNS * (0.5 - 0.5 * np.cos(np.pi * frac))
        phi = phi0 + sweep
        c = np.array([ROD_CX + r * np.sin(phi),
                      ROD_Y - r * np.cos(phi),
                      z0 + WRAP_PITCH * sweep / (2.0 * np.pi)], dtype=np.float64)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_hold                # hold the hub's entry orientation ⇒ no snap, no self-rotation
        M[:3, 3] = c
        is_c[0] = 1
        aim[0] = _mat4_to_uipc(M)

    scene.animator().insert(hub_obj, animate_hub)

    world.init(scene)

    # ---- restore β + bonded locks (after init, before first advance) ----
    if adhesion_on and pair_state is not None:
        keys, betas = pair_state
        acc = world.features().find(RCCAdhesionStateAccessorFeature)
        if acc is not None and len(betas) > 0:
            acc.load_pt_state(keys, betas)
            print(f"[rod-wind] restored β: n={len(betas)}, "
                  f"frac>0.9={(betas > 0.9).mean():.2%}")
    if bonded:
        locked_pairs = L.load_tape_locked_pairs(ASSET_IN_PATH)
        bpt = world.features().find(RCCBondedPTStateAccessorFeature)
        if locked_pairs is not None and bpt is not None:
            topos, lbetas = locked_pairs
            bpt.seed_locks(topos, lbetas, beta_lock_threshold)
            print(f"[rod-wind] seeded {len(lbetas)} bonded locks.")

    return {"engine": engine, "world": world, "scene": scene,
            "scene_io": SceneIO(scene), "hub_geo": hub_geo,
            "tape_geo": tape_geo, "rod_geo": rod_geo, "params": params}


def run_demo():
    bonded = L.cfg_flag(_CFG, "BONDED", default=False)
    sim = build_demo(adhesion_on=True, bonded=bonded,
                     beta_lock_threshold=_cfg_f("RCC_BETA_LOCK_THRESHOLD", 0.9),
                     kappa=_cfg_f("RCC_KAPPA", 1.0e8),
                     release_force=_cfg_f("RCC_RELEASE_FORCE", 1.0e30) if bonded else 1.0e30)

    record_dir = _CFG.get("RECORD_DIR")
    if record_dir:
        L.record_demo_to_pngs(
            sim=sim, total_frames=TOTAL_FRAMES, output_dir=record_dir,
            every_n=_cfg_i("RECORD_EVERY", 10), up_dir="y_up",
            mesh_name="rod_wind", zoom=_cfg_f("RECORD_ZOOM", 1.5),
            on_progress=lambda f, t: print(
                f"  frame {f}/{t} [{phase_at(f)}]", flush=True))
        return

    if ps is None or _CFG.get("HEADLESS"):
        # headless without a record dir: just step + report (CI / quick check)
        w = sim["world"]
        for _ in range(min(TOTAL_FRAMES, _cfg_i("MAX_FRAMES", TOTAL_FRAMES))):
            w.advance()
            if not w.is_valid():
                print(f"[rod-wind] INVALID at frame {w.frame()}"); break
            w.retrieve()
        print(f"[rod-wind] done at frame {w.frame()}/{TOTAL_FRAMES}")
        return

    ps.init(); ps.set_ground_plane_mode("none"); ps.set_up_dir("y_up")
    surface = sim["scene_io"].simplicial_surface()
    mesh = ps.register_surface_mesh(
        "rod_wind", surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3))
    ui = {"run": False}

    def step_once():
        if sim["world"].frame() >= TOTAL_FRAMES:
            ui["run"] = False; return
        sim["world"].advance()
        if not sim["world"].is_valid():
            ui["run"] = False; return
        sim["world"].retrieve()
        nonlocal mesh
        m = sim["scene_io"].simplicial_surface()
        v = m.positions().view().reshape(-1, 3)
        t = m.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != v.shape[0]:
            ps.remove_surface_mesh("rod_wind")
            mesh = ps.register_surface_mesh("rod_wind", v, t)
        else:
            mesh.update_vertex_positions(v)

    def on_update():
        if psim.Button("run / pause"): ui["run"] = not ui["run"]
        psim.SameLine()
        if psim.Button("step"): step_once()
        if ui["run"]: step_once()
        f = min(sim["world"].frame(), TOTAL_FRAMES)
        psim.Text(f"Frame {f}/{TOTAL_FRAMES}  phase={phase_at(f)}")

    ps.set_user_callback(on_update); ps.show()


if __name__ == "__main__":
    run_demo()
