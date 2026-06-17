"""
Stick a tape's free end onto a fixed deformable rod and fold it over
(first slice of the rod-wind sequence; the orbit/winding comes later).

A pre-wound tape roll (loaded from a wind asset) is stood UPRIGHT: the
whole asset is rotated about the hub axis (+z) so the protruding free
segment points straight UP, sticky face toward the rod. A two-end-pinned
DEFORMABLE rod (thin tube shell) is suspended with its axis +z at the
height of the upper-middle of the protruding segment, a small horizontal
gap away on the sticky side. The sequence:

  PIN      : hub held by a SoftTransformConstraint at its stand-up pose;
             the free-end ROW held by a SoftPositionConstraint. The rest
             of the tape settles under gravity between the two anchors.
  APPROACH : hub STC aim and free-end-row SPC aim translate horizontally
             toward the rod by the same vector, until the tape's sticky
             face meets the rod surface (IPC band) at the rod's height.
  WRAP     : hub frozen at the contact pose; the tape rows in contact
             with the rod are SPC-frozen where they touched; the rows
             from the free end down to the contact band rotate rigidly
             180 deg about the ROD AXIS — the free end folds over the
             rod and hangs on the far side.
  SQUEEZE  : below the rod, the folded strip and the incoming strip are
             SPC-pinched toward each other into the adhesion band →
             tape–tape bond closes the loop around the rod.
  SETTLE   : all constraints held; adhesion/bonds strengthen.
  FREE     : tape SPCs release in STAGES, farthest-from-rod first (an
             instant full release lets the stiff fold's recoil slam the
             strips together); fully free by the end — the wrap stays
             on the rod iff the bond took.
  ORBIT    : the hub sweeps a circle around the rod axis (translation
             only, orientation pinned) — each orbit winds one more turn
             onto the rod, paying tape off the roll.
  SETTLE2  : hub held at the final orbit point; the winding relaxes.

Almost everything is a knob (CLI `--set KEY=VALUE`, or a preset). The
defaults are a starting point to iterate from, not a tuned result.

Run (interactive):
    python/.venv/bin/python python/examples/rcc_adhesive_tape_rod_wind_demo.py \
        --asset output/rcc_adhesive_tape_winding/temflex175-2turn-e5e7-dhat2-cnct1-distlock.npz

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
from uipc.geometry import ground, trimesh, label_surface, mesh_partition
from uipc.core import RCCAdhesionStateAccessorFeature, RCCBondedPTStateAccessorFeature
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyRevoluteJoint,
    NeoHookeanShell,
    DiscreteShellBending,
    SoftPositionConstraint,
    SoftTransformConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)
from uipc.geometry import linemesh

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
# Default: the temflex175 2-turn beta-mode roll. The orbit pays tape off the
# roll by PEELING, which beta-mode bonds support (β drops under tension); a
# distance-locked roll cannot be peeled (released pairs relock inside the
# band) — pass a distlock asset only for the stick/fold phases.
_DEFAULT_ASSET = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_winding", "temflex175-2turn-e5e8-dhat2-cnct1.npz")
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
ROD_ABD           = _cfg_i("ROD_ABD", 1)            # 1 = rod is ONE rigid ABD body (closed cylinder, is_fixed); 0 = deformable tube shell, end rings pinned
ROD_ABD_KAPPA     = _cfg_f("ROD_ABD_KAPPA", 1.0e8)
# rod placement (y-up world). Axis +z. The X position is AUTO: the rod sits
# a horizontal gap ROD_GAP off the tape's sticky face (toward the roll
# center). The height is AUTO: ROD_HEIGHT_FRAC of the way up the protruding
# free segment ("middle, slightly above"). ROD_Y >= 0 overrides the height.
ROD_Y             = _cfg_f("ROD_Y", -1.0)           # rod axis height (m); <0 ⇒ auto
ROD_GAP           = _cfg_f("ROD_GAP", 0.03)         # initial tape-face → rod-surface gap (m)
ROD_HEIGHT_FRAC   = _cfg_f("ROD_HEIGHT_FRAC", 0.6)  # rod height along the protruding segment (0=root, 1=tip)
ROD_CZ            = _cfg_f("ROD_CZ", 0.0)

# ---- wrap (180° fold of the free end about the rod axis) ----
FOLD_ROWS         = _cfg_i("FOLD_ROWS", 5)          # rows from the tip that rotate rigidly; the rows between them and the frozen contact row stay FREE and bend naturally around the rod (<0 ⇒ every row above the contact band)
WRAP_DIR          = _cfg_f("WRAP_DIR", 0.0)         # ±1 rotation sense about +z; 0 ⇒ auto (over the rod top, away from the tape side)
FOLD_CLEAR        = _cfg_f("FOLD_CLEAR", 0.005)     # extra fold radius (m): the rotating rows arc this far OUTSIDE their natural radius about the rod axis, so the fold never presses onto the rod; ramps in with the wrap

# ---- orbit (hub circles the rod axis to wind more turns) ----
N_TURNS           = _cfg_f("N_TURNS", 2.0)          # number of hub orbits = turns wound
ORBIT_FRAMES      = _cfg_i("ORBIT_FRAMES", 1500)
ORBIT_SWEEP_DIR   = _cfg_f("ORBIT_SWEEP_DIR", 0.0)  # ±1 sweep sense about +z; 0 ⇒ auto = OPPOSITE to the wrap fold (the fold went over the top; the supply winds on from below)
WRAP_PITCH        = _cfg_f("WRAP_PITCH", 0.0)       # axial advance per turn (m); 0 = stacked
HANG_RADIUS       = _cfg_f("HANG_RADIUS", -1.0)     # orbit radius (m); <0 ⇒ auto = hub's natural distance to the rod axis at orbit entry

# ---- adhesion-to-rod (tape ↔ rod). Defaults mirror tape-tape unless set ----
STICKY            = _cfg_i("STICKY", 0)             # 0=double-sided (robust), ±1=single

# ---- timeline (dt = 0.01) ----
#   PIN      → hub STC + free-end-row SPC hold the stand-up pose; the rest
#              of the tape settles between the two anchors.
#   APPROACH → hub aim + free-end-row aim translate toward the rod until
#              the sticky face meets the rod surface (IPC band).
#   WRAP     → hub frozen; contact-band rows SPC-frozen; the rows above
#              them rotate rigidly 180° about the rod axis.
#   SQUEEZE  → below the rod, the folded-over strip and the incoming strip
#              are SPC-driven horizontally TOWARD EACH OTHER until their
#              mid-surfaces sit inside the adhesion band → tape–tape bond.
#   SETTLE   → all constraints held; adhesion/bonds strengthen.
#   FREE     → every tape SPC releases (hub stays held): if the bond took,
#              the wrap stays on the rod; if not, it springs open.
#   ORBIT    → the hub sweeps a circle around the rod axis (translation
#              only, orientation pinned): each orbit winds one more turn
#              onto the rod, paying tape off the roll.
#   SETTLE2  → hub held at the final orbit point; the winding relaxes.
PIN_FRAMES        = _cfg_i("PIN_FRAMES", 30)
APPROACH_FRAMES   = _cfg_i("APPROACH_FRAMES", 120)
WRAP_FRAMES       = _cfg_i("WRAP_FRAMES", 240)
SQUEEZE_FRAMES    = _cfg_i("SQUEEZE_FRAMES", 120)
SETTLE_FRAMES     = _cfg_i("SETTLE_FRAMES", 150)
FREE_FRAMES       = _cfg_i("FREE_FRAMES", 150)
ORBIT_SETTLE_FRAMES = _cfg_i("ORBIT_SETTLE_FRAMES", 150)

SPC_STRENGTH      = _cfg_f("SPC_STRENGTH", 1.0e9)
STC_ETA_P         = _cfg_f("STC_ETA_P", 1.0e8)      # hub translation strength
STC_ETA_A         = _cfg_f("STC_ETA_A", 1.0e8)      # hub rotation strength (>0 ⇒ orientation pinned)

_T0  = PIN_FRAMES
_Ta  = _T0 + APPROACH_FRAMES     # approach done → wrap start
_Tw  = _Ta + WRAP_FRAMES         # wrap done → squeeze start
_Tsq = _Tw + SQUEEZE_FRAMES      # squeeze done → settle
_Ts  = _Tsq + SETTLE_FRAMES      # settle done → free
_Tf2 = _Ts + FREE_FRAMES         # free done → orbit start
_To  = _Tf2 + ORBIT_FRAMES       # orbit done → settle2
TOTAL_FRAMES      = _To + ORBIT_SETTLE_FRAMES


def phase_at(f: int) -> str:
    if f < _T0:  return "pin"
    if f < _Ta:  return "approach"
    if f < _Tw:  return "wrap"
    if f < _Tsq: return "squeeze"
    if f < _Ts:  return "settle"
    if f < _Tf2: return "free"
    if f < _To:  return "orbit"
    return "settle2"


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
    sc = trimesh(positions.astype(np.float64), tris)
    label_surface(sc)
    return sc


def _make_rod_abd_sc(R, length, n_sides, center):
    """Closed cylinder trimesh (side + both caps) for a rigid ABD rod, axis
    +z, centered at `center`. Face orientation is validated by signed volume
    (flipped if inward) so ABD mass/volume integration is well-posed."""
    cx, cy, cz = center
    z0, z1 = cz - 0.5 * length, cz + 0.5 * length
    ang = np.arange(n_sides) * (2.0 * np.pi / n_sides)
    ring = np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], axis=1)
    v0 = np.column_stack([ring, np.full(n_sides, z0)])
    v1 = np.column_stack([ring, np.full(n_sides, z1)])
    verts = np.vstack([v0, v1,
                       [[cx, cy, z0]], [[cx, cy, z1]]]).astype(np.float64)
    ic0, ic1 = 2 * n_sides, 2 * n_sides + 1
    tris = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        a, b = i, j                      # back ring (z0)
        c, d = n_sides + i, n_sides + j  # front ring (z1)
        tris += [[a, b, d], [a, d, c]]   # side
        tris += [[b, a, ic0], [c, d, ic1]]  # caps
    tris = np.asarray(tris, dtype=np.int32)
    # signed volume via divergence theorem; flip if inward-facing
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    vol6 = np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum()
    if vol6 < 0.0:
        tris = tris[:, ::-1].copy()
    sc = trimesh(verts, tris)
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


def _stand_upright(tape_pos, hub_T, hub_R_outer, ground_clearance, target_xz,
                   n_width, tip_rows=6):
    """Rotate the whole asset (tape + hub) about the hub axis (+z, through
    the hub center) so the protruding free segment points straight UP, then
    translate so the roll's lowest point sits `ground_clearance` above y=0
    with the hub center at target_xz=(x, z).

    The free-end direction is estimated from the last `tip_rows` row
    centers (xy projection). Returns (tape_new, hub_T_new, Rz) — Rz so the
    caller can rotate the saved velocity snapshot consistently."""
    hub_center = hub_T[:3, 3].copy()
    centers = tape_pos.reshape(-1, n_width, 3).mean(axis=1)
    d = centers[-1] - centers[-1 - max(tip_rows, 1)]
    rot = np.pi / 2 - float(np.arctan2(d[1], d[0]))   # tip direction → +y
    c, s = np.cos(rot), np.sin(rot)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    tape_new = (tape_pos - hub_center) @ Rz.T + hub_center
    hub_T_new = hub_T.copy()
    hub_T_new[:3, :3] = Rz @ hub_T[:3, :3]

    hub_bottom_y = float(hub_center[1]) - hub_R_outer
    min_y = min(float(tape_new[:, 1].min()), hub_bottom_y)
    shift = np.array([target_xz[0] - hub_center[0],
                      ground_clearance - min_y,
                      target_xz[1] - hub_center[2]], dtype=np.float64)
    tape_new = tape_new + shift
    hub_T_new[:3, 3] += shift
    return tape_new, hub_T_new, Rz


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

    # ---- stand the roll upright: free segment pointing straight UP ----
    # Keep the roll OFF the ground: stood on its rolling face, the roll's own
    # weight squeezes the bottom inter-layer gaps down to the thickness bound
    # (DCD assert). Suspended from the STC-held hub (the innermost layer is
    # bonded to the hub), the layers hang in tension instead.
    GROUND_CLEARANCE = _cfg_f("GROUND_CLEARANCE", 0.005)
    n_width = TAPE_NZ + 1
    tape_pos_up, hub_T_up, Rz_up = _stand_upright(
        tape_pos, hub_T, HUB_R_OUTER, GROUND_CLEARANCE, (0.0, ROD_CZ), n_width)

    # ---- rod placement from the stood-up geometry ----
    # Protruding segment: from the roll's top (root) to the tip.
    centers   = tape_pos_up.reshape(-1, n_width, 3).mean(axis=1)
    hub_c_up  = hub_T_up[:3, 3]
    LAYER_THICKNESS = float(params.get("LAYER_THICKNESS", 2.5e-4))
    ASSET_TURNS     = float(params.get("N_TURNS", 2.0))
    R_roll    = HUB_R_OUTER + ASSET_TURNS * LAYER_THICKNESS + TAPE_THICKNESS
    y_tip     = float(centers[-1, 1])
    y_root    = float(hub_c_up[1]) + R_roll
    rod_y     = ROD_Y if ROD_Y >= 0 else (
        y_root + ROD_HEIGHT_FRAC * (y_tip - y_root))
    # Sticky side faces the roll center (the face that was glued to the
    # layer below continues onto the free segment) → the rod goes on the
    # side of the tape face that points toward the hub center.
    x_tape    = float(centers[-1, 0])               # vertical-segment plane
    side_inner = 1.0 if (float(hub_c_up[0]) - x_tape) >= 0.0 else -1.0
    rod_x     = x_tape + side_inner * (ROD_GAP + ROD_R)
    # APPROACH translation: move tape face into the IPC band at the rod
    # surface (stop just inside the band; IPC carries the actual contact).
    band_target = 0.5 * TAPE_THICKNESS + 0.5 * D_HAT
    approach_dx = side_inner * max(ROD_GAP - band_target, 0.0)
    wrap_dir  = WRAP_DIR if WRAP_DIR != 0.0 else -side_inner
    print(f"[rod-wind] stand-up: tip y={y_tip:.4f}, root y={y_root:.4f}, "
          f"rod=({rod_x:.4f}, {rod_y:.4f}), side_inner={side_inner:+.0f}, "
          f"approach dx={approach_dx:+.4f}, wrap_dir={wrap_dir:+.0f}")

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
        # CCD ON by default (SKIP_CCD=0): keep the thickness guard on locked
        # pairs too. `--set SKIP_CCD=-1` restores the auto-skip (faster, but a
        # locked pair's only anti-penetration force is the ABD bond energy).
        config["rcc_bonded_pt_skip_ccd"] = int(_CFG.get("SKIP_CCD", 0))
        config["rcc_bonded_pt_beta_lock_threshold"] = beta_lock_threshold
        config["rcc_bonded_pt_lock_face_interior_only"] = (
            1 if L.cfg_flag(_CFG, "LOCK_FACE_INTERIOR_ONLY", default=False) else 0)
        config["rcc_bonded_pt_energy_model"] = "abd_ortho"
        config["rcc_bonded_pt_kappa"] = float(_CFG.get("RCC_KAPPA", kappa))
        config["rcc_bonded_pt_release_force"] = release_force
        # Phase 7 distance-locked bonding. Precedence: `--set DISTANCE_LOCK=...`
        # wins; else the asset's saved flag (a tape wound in distance-lock mode
        # auto-replays in it). No soft adhesion energy / beta in this mode.
        if L.resolve_flag(_CFG, params, "DISTANCE_LOCK", default=False):
            config["rcc_bonded_pt_distance_lock"] = 1
            _ratio = _CFG.get("DISTANCE_LOCK_RATIO")
            if _ratio is None:
                _ratio = params.get("DISTANCE_LOCK_RATIO", 0.5)
            config["rcc_bonded_pt_distance_lock_ratio"] = float(_ratio)
    L.apply_solver_overrides(config, _CFG, params=params)
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    dsb = DiscreteShellBending()
    spc = SoftPositionConstraint()
    stc = SoftTransformConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    hub_contact     = tabular.default_element()      # hub + ground
    tape_contact    = tabular.create("tape")
    rod_contact     = tabular.create("rod")
    tabular.insert(tape_contact,    tape_contact,    0.5, 1.0e9)
    tabular.insert(tape_contact,    hub_contact,     0.5, 1.0e9)
    tabular.insert(tape_contact,    rod_contact,     0.5, 1.0e9)
    tabular.insert(rod_contact,     hub_contact,     0.5, 1.0e9)
    tabular.insert(rod_contact,     rod_contact,     0.5, 1.0e9)

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
            # Per-pair lock thresholds (-1 = inherit the global config).
            # tape-hub mirrors tape-tape unless explicitly overridden: the
            # innermost layer peeling off the hub is the same event as a
            # layer peeling off the layer below.
            _tt_lk = float(_CFG.get("BONDED_TAPE_LOCK", -1.0))
            _th_lk = float(_CFG.get("BONDED_HUB_LOCK", _tt_lk))
            _tr_lk = float(_CFG.get("BONDED_ROD_LOCK", -1.0))
            # Per-pair release forces. tape-rod bonds must HOLD while the
            # roll peels: the global release force is tuned low so roll
            # layers let go, but with the same value on the rod the winding
            # falls off — default the rod side to 100x the global one.
            # tape-hub mirrors tape-tape.
            _tt_rf = float(_CFG.get("BONDED_TAPE_RELEASE_FORCE", -1.0))
            _th_rf = float(_CFG.get("BONDED_HUB_RELEASE_FORCE", _tt_rf))
            _tr_rf_default = (100.0 * release_force
                              if release_force < 1.0e29 else -1.0)
            _tr_rf = float(_CFG.get("BONDED_ROD_RELEASE_FORCE", _tr_rf_default))
            for (A, B, lk, rf) in [(tape_contact, tape_contact, _tt_lk, _tt_rf),
                                   (tape_contact, hub_contact,  _th_lk, _th_rf),
                                   (tape_contact, rod_contact,  _tr_lk, _tr_rf)]:
                adhesive.set_bonded(tabular, A, B,
                                    lock_threshold=lk, release_strain=-1.0,
                                    release_gap=-1.0, release_slip=-1.0,
                                    release_force=rf)

    # ---- hub (ABD; STC-held for the whole sequence) ----
    hub_sc = L.make_ring_hub(R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
                             height=HUB_HEIGHT, n_radial=48, center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    hub_contact.apply_to(hub_sc)
    stc.apply_to(hub_sc, np.array([STC_ETA_P, STC_ETA_A], dtype=np.float64))
    view(hub_sc.transforms())[0] = _mat4_to_uipc(hub_T_up)
    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- rod: ONE rigid ABD body (default) or a deformable tube shell ----
    if ROD_ABD:
        rod_sc = _make_rod_abd_sc(ROD_R, ROD_LENGTH, ROD_SIDES,
                                  (rod_x, rod_y, ROD_CZ))
        abd.apply_to(rod_sc, ROD_ABD_KAPPA, ROD_DENSITY)
        rod_contact.apply_to(rod_sc)
        # statically fixed: the rod never moves
        view(rod_sc.instances().find(builtin.is_fixed))[0] = 1
    else:
        rod_sc, rod_pin = L.make_rod_tube_shell(
            R=ROD_R, length=ROD_LENGTH, segments=ROD_SEGMENTS, n_sides=ROD_SIDES,
            center=(rod_x, rod_y, ROD_CZ))
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

    # ---- carrier: ABD rod through the hub's center hole ----
    # ORBIT no longer drags the hub by STC. Instead this carrier (radius =
    # half the hub hole) is the STC-driven body: PIN..FREE it shadows the hub
    # center, at ORBIT entry the hub's own STC disengages and the hub rides
    # the carrier through hole contact while the carrier sweeps the circle.
    CARRIER_R   = _cfg_f("CARRIER_R", 0.5 * HUB_R_INNER)
    CARRIER_LEN = _cfg_f("CARRIER_LEN", 4.0 * HUB_HEIGHT)
    # CARRIER_JOINT=1 (default): the hub-carrier bearing is an
    # AffineBodyRevoluteJoint on the carrier axis and the hub-carrier
    # CONTACT pair is disabled — a bearing is a joint, not a line contact
    # (the rigid-rigid contact + friction left the hub's spin a nearly flat
    # energy direction and Newton ground 600+ iterations on it).
    # CARRIER_JOINT=0 restores the contact bearing.
    CARRIER_JOINT          = _cfg_i("CARRIER_JOINT", 1)
    CARRIER_JOINT_STRENGTH = _cfg_f("CARRIER_JOINT_STRENGTH", 100.0)
    carrier_c0  = np.array([float(hub_T_up[0, 3]), float(hub_T_up[1, 3]),
                            ROD_CZ], dtype=np.float64)
    carrier_sc = _make_rod_abd_sc(CARRIER_R, CARRIER_LEN, ROD_SIDES,
                                  tuple(carrier_c0))
    abd.apply_to(carrier_sc, ROD_ABD_KAPPA, ROD_DENSITY)
    carrier_contact = tabular.create("carrier")
    carrier_contact.apply_to(carrier_sc)
    tabular.insert(carrier_contact, hub_contact, 0.5, 1.0e9,
                   enable=(CARRIER_JOINT == 0))
    stc.apply_to(carrier_sc, np.array([STC_ETA_P, STC_ETA_A], dtype=np.float64))
    carrier_obj = scene.objects().create("carrier")
    carrier_geo, _ = carrier_obj.geometries().create(carrier_sc)

    # ---- tape (current = stood-up wound pose; rest = wind's straight strip) ----
    tris = _make_tape_topology_tris(TAPE_NX, TAPE_NZ)
    current_sc = _make_tape_sc(tape_pos_up, tris)
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

    # TAPE_PARTITION>0: tag the tape with a `mesh_part` vertex attribute so the
    # FEM solver uses the MAS (multilevel additive Schwarz) preconditioner
    # instead of block-diagonal Jacobi — domain decomposition that cuts the
    # PCG iteration count on the deformable shell (the dominant solver cost).
    # Attribute-only (no vertex reorder) so the seeded-lock index remap is
    # unaffected. Default off to preserve the diagonal-preconditioner baseline.
    _part = _cfg_i("TAPE_PARTITION", 0)
    if _part > 0:
        mesh_partition(current_sc, _part)

    if tape_vel is not None and tape_vel.shape == tape_pos.shape:
        # stand-upright rotates the asset ⇒ rotate the velocity snapshot too
        tape_vel_up = tape_vel @ Rz_up.T
        if current_sc.vertices().find("velocity") is None:
            current_sc.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        view(current_sc.vertices().find("velocity"))[:] = tape_vel_up.reshape(-1, 3, 1)

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(current_sc, rest_sc)

    # ---- ground ----
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    if CARRIER_JOINT:
        # Revolute bearing on the carrier axis (+z through the hub center).
        # Created LAST so its vertices land after every body the seed-lock
        # remap cares about.
        half = 0.5 * HUB_HEIGHT
        joint_mesh = linemesh(
            np.array([carrier_c0 + [0.0, 0.0, -half],
                      carrier_c0 + [0.0, 0.0, +half]], dtype=np.float64),
            np.array([[0, 1]], dtype=np.int32))
        revolute = AffineBodyRevoluteJoint()
        revolute.apply_to(joint_mesh,
                          [hub_geo], [0],
                          [carrier_geo], [0],
                          [CARRIER_JOINT_STRENGTH])
        joint_obj = scene.objects().create("hub_carrier_bearing")
        joint_obj.geometries().create(joint_mesh)

    # ---- free-end row + wrap targets ----
    def vid(i, j): return i * (TAPE_NZ + 1) + j
    band    = TAPE_THICKNESS + 0.5 * D_HAT
    ds_len  = TAPE_LENGTH / TAPE_NX
    tip_ids = [vid(TAPE_NX, j) for j in range(TAPE_NZ + 1)]
    tip_p0  = tape_pos_up[tip_ids].copy()            # stand-up pose of the free-end row
    rod_axis = np.array([rod_x, rod_y], dtype=np.float64)

    # WRAP state, captured at wrap entry from the LIVE shape:
    #   contact rows — row centers within the IPC band of the rod surface →
    #                  SPC-frozen where they touched;
    #   fold rows    — rows above the contact band (or the last FOLD_ROWS) →
    #                  rotate rigidly about the rod axis by 180°.
    # SQUEEZE state, captured at squeeze entry: below the rod the folded
    # strip and the incoming strip are parallel, ~2·(rod R + band) apart —
    # too far for adhesion. Each side gets a horizontal SPC displacement
    # toward the other until their mid-surfaces sit one tape thickness +
    # half an IPC band apart (inside the adhesion / distance-lock band).
    wrap_state    = {"init": None}
    squeeze_state = {"init": None}
    free_state    = {"order": None}
    # Mid-surface separation target for the squeeze. The tape-tape contact
    # floor is xi = 2*TAPE_THICKNESS (PT_thickness sums BOTH vertices' full
    # thickness attributes), so aim for the adhesion-band CENTER xi + d_hat/2
    # — squeezing to xi itself parks the pair on the thickness assert.
    squeeze_sep_target = 2.0 * TAPE_THICKNESS + 0.5 * D_HAT

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim  = view(geo.vertices().find(builtin.aim_position))
        is_c[:] = 0
        if f >= _Tf2:                     # fully free (orbit and beyond)
            return
        if f < _Ta:                       # PIN + APPROACH: drive the free-end row
            t = 0.0 if f < _T0 else min((f - _T0) / max(APPROACH_FRAMES, 1), 1.0)
            dx = smooth_lerp(0.0, approach_dx, t)
            for jj, k in enumerate(tip_ids):
                is_c[k] = 1
                tgt = tip_p0[jj] + np.array([dx, 0.0, 0.0])
                aim[k] = tgt.reshape(3, 1)
            wrap_state["init"] = None
            squeeze_state["init"] = None
            return
        # WRAP + SQUEEZE + SETTLE
        if wrap_state["init"] is None:
            live = np.asarray(view(geo.positions())).reshape(-1, 3).copy()
            centers_live = live.reshape(-1, TAPE_NZ + 1, 3).mean(axis=1)
            dist = np.hypot(centers_live[:, 0] - rod_axis[0],
                            centers_live[:, 1] - rod_axis[1])
            contact = dist < (ROD_R + band + 0.5 * ds_len)
            contact_rows = np.nonzero(contact)[0]
            if len(contact_rows) == 0:    # fallback: row closest to the rod
                contact_rows = np.array([int(np.argmin(dist))])
            top_contact = int(contact_rows.max())
            first_fold  = (top_contact + 1 if FOLD_ROWS < 0
                           else max(TAPE_NX + 1 - FOLD_ROWS, top_contact + 1))
            fold_rows   = list(range(first_fold, TAPE_NX + 1))
            # Freeze ONLY the top contact row: hard-pinning the whole contact
            # band presses the tape into the rod; one anchored row is enough,
            # the rows between it and the rotating fold rows stay free and
            # bend naturally around the rod surface.
            frozen_rows = [top_contact]
            wrap_state["init"] = (live, frozen_rows, fold_rows)
            print(f"[rod-wind] wrap start: frozen row {frozen_rows}, "
                  f"fold rows {first_fold}..{TAPE_NX} "
                  f"({len(fold_rows)} rows; "
                  f"{first_fold - top_contact - 1} free in between)", flush=True)
        live0, frozen_rows, fold_rows = wrap_state["init"]
        # fold rows: rigid rotation about the rod axis, 0 → 180°
        t = min((f - _Ta) / max(WRAP_FRAMES, 1), 1.0)
        theta = wrap_dir * np.pi * (0.5 - 0.5 * np.cos(np.pi * t))
        c, s = np.cos(theta), np.sin(theta)
        # Enlarged fold radius: push each rotated target OUT by FOLD_CLEAR
        # (ramped with the wrap) so the fold arcs AROUND the rod with a
        # clearance instead of pressing onto its surface.
        clear_t = FOLD_CLEAR * (0.5 - 0.5 * np.cos(np.pi * t))
        fold_tgt = {}
        for r in fold_rows:
            for j in range(TAPE_NZ + 1):
                k = vid(r, j)
                dx0 = live0[k, 0] - rod_axis[0]
                dy0 = live0[k, 1] - rod_axis[1]
                r0  = float(np.hypot(dx0, dy0))
                rs  = (r0 + clear_t) / max(r0, 1e-12)
                dxs, dys = dx0 * rs, dy0 * rs
                fold_tgt[k] = np.array([rod_axis[0] + c * dxs - s * dys,
                                        rod_axis[1] + s * dxs + c * dys,
                                        live0[k, 2]], dtype=np.float64)
        # SQUEEZE: pinch the two strips below the rod toward each other
        sq_dx_fold = sq_dx_in = 0.0
        in_ids = []
        if f >= _Tw:
            if squeeze_state["init"] is None:
                live = np.asarray(view(geo.positions())).reshape(-1, 3).copy()
                # Only vertices ENTIRELY below the rod participate (y under
                # the rod's bottom surface) — squeezing anything level with
                # the rod presses the tape into it. The displacement still
                # TAPERS from 0 at the cut to its full value `taper_len`
                # further down, so the free bend around the rod is not
                # sheared (a uniform pinch dragged the transition row into
                # the rod -> thickness violation).
                y_cut     = rod_axis[1] - (ROD_R + band)
                taper_len = max(3.0 * ds_len, 2.0 * (ROD_R + band))
                def _w(y):
                    return float(np.clip((y_cut - y) / taper_len, 0.0, 1.0))
                # folded-strip vertices below the rod (use the wrap endpoints)
                f_ids = [k for k, tg in fold_tgt.items() if tg[1] < y_cut]
                f_w   = {k: _w(fold_tgt[k][1]) for k in f_ids}
                # incoming rows: below the contact band, vertically overlapping
                # the folded strip
                y_lo = (min(fold_tgt[k][1] for k in f_ids) - 2 * ds_len
                        if f_ids else y_cut)
                ids, ws = [], []
                for r in range(0, min(frozen_rows)):
                    for j in range(TAPE_NZ + 1):
                        k = vid(r, j)
                        if y_lo <= live[k, 1] <= y_cut:
                            ids.append(k)
                            ws.append(_w(live[k, 1]))
                if f_ids and ids:
                    xs_f = np.asarray([fold_tgt[k][0] for k in f_ids])
                    xs_i = live[ids, 0]
                    # Calibrate the pinch on the NEAREST pair of faces, not
                    # the means: the folded strip is an ARC (esp. with
                    # FOLD_CLEAR), and pinching until the MEANS reach the
                    # target drives the arc's bulge through the thickness
                    # floor first.
                    if float(xs_f.mean()) < float(xs_i.mean()):
                        gap = float(xs_i.min()) - float(xs_f.max())
                        sgn = +1.0          # fold moves +x, incoming -x
                    else:
                        gap = float(xs_f.min()) - float(xs_i.max())
                        sgn = -1.0
                    move = max(gap - squeeze_sep_target, 0.0) / 2.0
                    sq_f = sgn * move
                    sq_i = -sgn * move
                else:
                    sq_f = sq_i = 0.0
                squeeze_state["init"] = (live, f_w, ids, ws, sq_f, sq_i)
                print(f"[rod-wind] squeeze start: fold verts {len(f_ids)}, "
                      f"incoming verts {len(ids)}, "
                      f"dx fold={sq_f:+.4f}, dx in={sq_i:+.4f}", flush=True)
            live_sq, f_w, in_ids, in_ws, sq_f, sq_i = squeeze_state["init"]
            tq = min((f - _Tw) / max(SQUEEZE_FRAMES, 1), 1.0)
            sq = 0.5 - 0.5 * np.cos(np.pi * tq)
            sq_dx_fold, sq_dx_in = sq * sq_f, sq * sq_i
            in_pairs = list(zip(in_ids, in_ws))
        else:
            squeeze_state["init"] = None
            f_w = {}
            live_sq, in_pairs, sq_dx_in = None, [], 0.0
        # FREE: STAGED release. Dropping every SPC at once lets the bent
        # section's elastic recoil (violent for stiff tape) slam the two
        # strips together (thickness assert). Release farthest-from-rod
        # first; the fold near the rod goes last, once the rest has settled
        # onto the bonds.
        keep = None
        if f >= _Ts:
            if free_state["order"] is None:
                entries = []
                for r in frozen_rows:
                    for j in range(TAPE_NZ + 1):
                        k = vid(r, j)
                        entries.append((k, live0[k]))
                for k, w in in_pairs:
                    entries.append((k, live_sq[k]))
                for k, tg in fold_tgt.items():
                    entries.append((k, tg))
                d2rod = {k: float(np.hypot(p[0] - rod_axis[0],
                                           p[1] - rod_axis[1]))
                         for k, p in entries}
                free_state["order"] = sorted(d2rod, key=d2rod.get, reverse=True)
            order = free_state["order"]
            tf = min((f - _Ts) / max(FREE_FRAMES, 1), 1.0)
            keep = set(order[int(tf * len(order)):])
        # contact row(s): frozen where they touched the rod
        for r in frozen_rows:
            for j in range(TAPE_NZ + 1):
                k = vid(r, j)
                if keep is not None and k not in keep:
                    continue
                is_c[k] = 1
                aim[k] = live0[k].reshape(3, 1)
        for k, w in in_pairs:
            if keep is not None and k not in keep:
                continue
            is_c[k] = 1
            tgt = live_sq[k] + np.array([sq_dx_in * w, 0.0, 0.0])
            aim[k] = tgt.reshape(3, 1)
        for k, tg in fold_tgt.items():
            if keep is not None and k not in keep:
                continue
            is_c[k] = 1
            tgt = tg + np.array([sq_dx_fold * f_w.get(k, 0.0), 0.0, 0.0])
            aim[k] = tgt.reshape(3, 1)

    scene.animator().insert(tape_obj, animate_tape)

    # ---- hub: STC-held until orbit, then free ----
    # PIN: hold the stand-up pose. APPROACH: translate by the same horizontal
    # vector as the free-end row. WRAP..FREE: frozen at the contact pose.
    # ORBIT and beyond: the hub's STC disengages — the CARRIER (through the
    # hub hole) sweeps the circle and the hub rides it under gravity + tape
    # tension, free to find its own pose (the tape pays off by peeling).
    MIN_ORBIT_R = ROD_R + R_roll + 0.003
    orbit_sweep = ORBIT_SWEEP_DIR if ORBIT_SWEEP_DIR != 0.0 else -wrap_dir

    def animate_hub(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.instances().find(builtin.is_constrained))
        aim  = view(geo.instances().find(builtin.aim_transform))
        if f >= _Tf2:                     # ORBIT and beyond: hub is free
            is_c[0] = 0
            return
        is_c[0] = 1
        t = 0.0 if f < _T0 else min((f - _T0) / max(APPROACH_FRAMES, 1), 1.0)
        dx = smooth_lerp(0.0, approach_dx, t)
        M = hub_T_up.copy()
        M[0, 3] += dx
        aim[0] = _mat4_to_uipc(M)

    scene.animator().insert(hub_obj, animate_hub)

    # ---- carrier: shadows the hub center until orbit, then sweeps ----
    # ORBIT: circle around the rod axis. The radius defaults to the hub's
    # natural entry radius PLUS the hole slack (hole R - carrier R), so the
    # tape tension takes the slack up and the free segment stays taut.
    carrier_state = {"init": None}

    def animate_carrier(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.instances().find(builtin.is_constrained))
        aim  = view(geo.instances().find(builtin.aim_transform))
        is_c[0] = 1
        if f < _Tf2:                      # PIN .. FREE: shadow the hub center
            t = 0.0 if f < _T0 else min((f - _T0) / max(APPROACH_FRAMES, 1), 1.0)
            dx = smooth_lerp(0.0, approach_dx, t)
            M = np.eye(4, dtype=np.float64)
            M[0, 3] = dx                  # mesh is baked at the hub center
            carrier_state["init"] = None
            aim[0] = _mat4_to_uipc(M)
            return
        cur = np.asarray(view(geo.transforms())[0]).reshape(4, 4)
        if carrier_state["init"] is None:
            c0 = carrier_c0 + cur[:3, 3]  # baked center + current translation
            dxr, dyr = float(c0[0] - rod_axis[0]), float(c0[1] - rod_axis[1])
            r_nat = float(np.hypot(dxr, dyr))
            slack = 0.0 if CARRIER_JOINT else max(HUB_R_INNER - CARRIER_R, 0.0)
            r = HANG_RADIUS if HANG_RADIUS > 0 else max(r_nat + slack, MIN_ORBIT_R)
            phi0 = float(np.arctan2(dxr, -dyr))   # point(φ0) = current center
            carrier_state["init"] = (r, phi0, float(c0[2]), cur[:3, :3].copy())
            print(f"[rod-wind] carrier orbit start: r={r*1e3:.1f} mm "
                  f"(natural {r_nat*1e3:.1f} mm + slack {slack*1e3:.1f} mm), "
                  f"φ0={np.degrees(phi0):.0f}°, sweep={orbit_sweep:+.0f}", flush=True)
        r, phi0, z0, R_hold = carrier_state["init"]
        frac  = min((f - _Tf2) / max(ORBIT_FRAMES, 1), 1.0)
        sweep = orbit_sweep * 2.0 * np.pi * N_TURNS * (0.5 - 0.5 * np.cos(np.pi * frac))
        phi   = phi0 + sweep
        c = np.array([rod_axis[0] + r * np.sin(phi),
                      rod_axis[1] - r * np.cos(phi),
                      z0 + WRAP_PITCH * sweep / (2.0 * np.pi)], dtype=np.float64)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_hold
        # mesh is baked at carrier_c0: x = A x̄ + b  ⇒  b = c − A·c0
        M[:3, 3]  = c - R_hold @ carrier_c0
        aim[0] = _mat4_to_uipc(M)

    scene.animator().insert(carrier_obj, animate_carrier)

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
            # Saved topos are GLOBAL vertex indices from the winding scene,
            # whose backend layout is [hub | tape] (ABD bodies are numbered
            # before FEM geometry, regardless of creation order). This scene
            # inserts the rod and carrier blocks between them —
            # [hub | rod | carrier | tape] — so every tape-range index must
            # jump over both. Hub-range indices (tape-hub bonds reference
            # hub triangles/points) stay put.
            hub_nv = len(np.asarray(view(hub_sc.positions())).reshape(-1, 3))
            shift = (len(np.asarray(view(rod_sc.positions())).reshape(-1, 3))
                     + len(np.asarray(view(carrier_sc.positions())).reshape(-1, 3)))
            topos = np.where(topos >= hub_nv, topos + shift, topos)
            bpt.seed_locks(topos, lbetas, beta_lock_threshold)
            print(f"[rod-wind] seeded {len(lbetas)} bonded locks "
                  f"(tape indices shifted +{shift} past rod+carrier).")

    return {"engine": engine, "world": world, "scene": scene,
            "scene_io": SceneIO(scene), "hub_geo": hub_geo,
            "tape_geo": tape_geo, "rod_geo": rod_geo, "params": params}


def run_demo():
    # `--set BONDED=...` wins; else the asset's saved flag (a tape wound WITH
    # bonded — incl. distance-lock mode — auto-enables it); else OFF.
    _asset_params = L.peek_asset_params(ASSET_IN_PATH)
    bonded = L.resolve_flag(_CFG, _asset_params, "BONDED", default=False)
    sim = build_demo(adhesion_on=True, bonded=bonded,
                     beta_lock_threshold=float(L.resolve_param(
                         _CFG, _asset_params, "RCC_BETA_LOCK_THRESHOLD")),
                     kappa=_cfg_f("RCC_KAPPA", 1.0e8),
                     release_force=float(L.resolve_param(
                         _CFG, _asset_params, "RCC_RELEASE_FORCE")) if bonded else 1.0e30)

    record_dir = _CFG.get("RECORD_DIR")
    if record_dir:
        # Bonded-lock count per progress line: the live peel-rate signal
        # (roll locks must DROP through the orbit, or tension accumulates).
        bpt_for_log = sim["world"].features().find(RCCBondedPTStateAccessorFeature)
        def _on_progress(f, t):
            locked = bpt_for_log.locked_pair_count() if bpt_for_log else -1
            print(f"  frame {f}/{t} [{phase_at(f)}] locked={locked}", flush=True)
        L.record_demo_to_pngs(
            sim=sim, total_frames=TOTAL_FRAMES, output_dir=record_dir,
            every_n=_cfg_i("RECORD_EVERY", 10), up_dir="y_up",
            mesh_name="rod_wind", zoom=_cfg_f("RECORD_ZOOM", 1.5),
            on_progress=_on_progress)
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
