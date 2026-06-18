"""
Tape-roll-on-its-side drop demo.

Loads a wound-tape asset (saved by `rcc_adhesive_tape_winding_demo.py`)
and stands it on the ground:

  - the wind demo saves the roll with hub axis along world +z; we
    rotate the whole assembly by R_x(-90°) so the hub axis lands on
    world +y (vertical — like a tape dispenser standing on a desk).
  - then translate it upward so the lowest vertex sits one IPC-band
    offset above the ground (tape thickness + ½ d_hat) — gives the
    barrier something to engage with at frame 0.
  - hub is NOT fixed: it's an ABD body free to move under gravity.
  - the wound layers and the inner-most layer-to-hub bond are held
    together purely by RCC adhesion (β restored from the asset when
    saved, otherwise driven by `initial_beta` and bonding-rate growth).
  - after a brief HOLD phase, an SPC engages on the tape's free-end
    row and lifts it upward by `LIFT_HEIGHT` (10 cm by default —
    clears the roll's diameter) over `PULL_FRAMES`. With strong
    adhesion the whole spool lifts off the ground; with weak adhesion
    the outer layer peels off and the roll stays grounded.
  - ground = libuipc's implicit half-plane at y=0 (the contact engine
    uses it; polyscope shows a matching flat quad for visualization).
  - gravity (0, -9.8, 0).

Adhesion preset comes from `UNWIND_PRESETS` (`rigid`/`strong-bond`
recommended — soft-bond may not survive elastic + gravity loads).

Run:
    python python/examples/rcc_adhesive_tape_drop_demo.py \
        --preset rigid --asset temflex175
"""

from __future__ import annotations

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

from uipc import (
    Logger,
    Matrix4x4,
    Engine,
    World,
    Scene,
    SceneIO,
    Animation,
    view,
    builtin,
)
from uipc.geometry import trimesh, label_surface, mesh_partition, ground
from uipc.core import RCCAdhesionStateAccessorFeature, RCCBondedPTStateAccessorFeature
from uipc.constitution import (
    AffineBodyConstitution,
    NeoHookeanShell,
    DiscreteShellBending,
    SoftPositionConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ---- CLI / preset resolution (re-uses UNWIND_PRESETS — same material
# + adhesion semantics. SPC_STRENGTH is in the preset but unused here.)
_CFG = L.parse_tape_cli(L.UNWIND_PRESETS)
print(f"[drop] preset={_CFG['__preset_name__']}: "
      f"E={_CFG['TAPE_YOUNGS']:.1e} Pa, ν={_CFG['TAPE_POISSON']}, "
      f"ρ={_CFG['TAPE_MASS_DENSITY']} kg/m³, "
      f"Cn={_CFG['ADH_CN']:.1e}, Ct={_CFG['ADH_CT']:.1e}, W={_CFG['ADH_W']}, "
      f"η={_CFG['ADH_ETA']}")

# IPC band fallbacks. build_demo overrides these from the asset unless
# the user `--set`s them on the CLI (same logic as the unwind demo).
D_HAT             = _CFG["D_HAT"]
TAPE_THICKNESS    = _CFG["TAPE_THICKNESS"]

# ---- materials ----
TAPE_YOUNGS       = _CFG["TAPE_YOUNGS"]
TAPE_POISSON      = _CFG["TAPE_POISSON"]
TAPE_MASS_DENSITY = _CFG["TAPE_MASS_DENSITY"]
HUB_KAPPA         = 1.0e8
HUB_MASS_DENSITY  = 1000.0

# ---- adhesion ----
ADH_CN            = _CFG["ADH_CN"]
ADH_CT            = _CFG["ADH_CT"]
ADH_W             = _CFG["ADH_W"]
ADH_ETA           = _CFG["ADH_ETA"]
ADH_BONDING_RATE  = _CFG["ADH_BONDING_RATE"]
ADH_INITIAL_BETA  = _CFG["ADH_INITIAL_BETA"]

# ---- timeline (dt=0.01) ----
# Four phases:
#   HOLD       — assembly settles on ground under gravity alone, free
#                end is unconstrained so its initial pose can relax.
#   PULL       — SPC engages on the tape's free-end row and smoothly
#                ramps it upward by LIFT_HEIGHT. The ramp uses a cosine
#                ease (smooth_lerp) so there's no velocity discontinuity
#                at either end of the pull.
#   TOP        — SPC holds the free end at the top while the rest of
#                the roll either follows (strong adhesion → whole spool
#                lifts off the ground) or peels off (weak adhesion →
#                outer layer detaches).
#   FREEFALL   — SPC is released; the lifted free end + whatever the
#                adhesion drags along falls back under gravity. This is
#                the "does the bonded roll stay together after being
#                dropped from the top?" stress test.
# `--set HALF_LIFT=1`: pull the free end up only HALF as far
# (LIFT_HEIGHT/2, applied to lift_y below) and hold at the top for half as
# long (TOP_FRAMES halved) before releasing — a gentler lift that lets go
# before fully clearing the roll.
HALF_LIFT          = L.cfg_flag(_CFG, "HALF_LIFT", default=False)
LIFT_FRACTION      = 0.5 if HALF_LIFT else 1.0

HOLD_FRAMES        = 30
# Frames over which the free end is lifted to LIFT_HEIGHT. Default 300 (a fast
# pull); override with `--set PULL_FRAMES=600` for a slower lift.
PULL_FRAMES        = int(_CFG.get("PULL_FRAMES", 300))
# Hold at top long enough to see whether the lifted roll stays
# bonded or starts peeling — 6 s @ dt=0.01 (halved with HALF_LIFT).
TOP_FRAMES         = 300 if HALF_LIFT else 600
# Post-release free-fall window. 5 s @ dt=0.01 is enough for a 10 cm
# lift to hit the ground (½·g·t² ≈ 0.49 m after 1 s already).
FREEFALL_FRAMES    = int(_CFG.get("FREEFALL_FRAMES", 500))
TOTAL_FRAMES       = HOLD_FRAMES + PULL_FRAMES + TOP_FRAMES + FREEFALL_FRAMES


def phase_at(f: int) -> str:
    """Frame → phase label. Module-level so headless record's progress
    callback can use it (it would otherwise be a closure local to
    run_demo and unreachable from the helper)."""
    if f < HOLD_FRAMES:                                          return "hold"
    if f < HOLD_FRAMES + PULL_FRAMES:                            return "pull"
    if f < HOLD_FRAMES + PULL_FRAMES + TOP_FRAMES:               return "top"
    return "freefall"

# How high to lift the free end (metres, +y). Must exceed the roll's
# vertical reach (≈ R_outer + N_TURNS·2·t when standing on side, plus
# slack for the tail) so a successful lift visibly clears the ground.
# Default 0.20 m (a high lift that clearly clears the roll); override
# per-run with `--set LIFT_HEIGHT=0.10` for a shorter pull, etc.
LIFT_HEIGHT        = float(_CFG.get("LIFT_HEIGHT", 0.20))

# Initial drop height of the assembly above the IPC active band. The
# `_stand_on_ground` shift puts the lowest geometry vertex at
# `(t + 0.5·d_hat) + DROP_HEIGHT` above y=0. Default 0 starts the roll
# as low as possible — essentially touching the ground, with only the
# `(t + 0.5·d_hat)` IPC band clearance needed to keep the barrier
# well-defined (the barrier is immediately active). Set e.g. `--set
# DROP_HEIGHT=0.01` for a brief free-fall window before contact engages:
# visually clearer "drop onto ground" motion, and a test of the
# barrier's response to incoming velocity.
DROP_HEIGHT        = float(_CFG.get("DROP_HEIGHT", 0.0))

SPC_STRENGTH       = float(_CFG.get("SPC_STRENGTH", 1.0e9))

# FEM rest reference. Default 1 mirrors wind's setup: rest = the same
# straight-tangent strip wind started with, so the wound state at load
# is in the SAME elastic-energy configuration as wind's end state.
# Concretely: at end of wind, elastic (outward) + adhesion (inward) +
# barrier together net to zero — i.e. it's an equilibrium. Setting
# rest = wound (REST_FROM_WIND=0) zeros the elastic component, which
# breaks the balance: adhesion pulls layers inward unopposed and the
# barrier has to compensate via large pressure → visible first-frame
# motion. With REST_FROM_WIND=1 (default), the load is force-balanced
# and the only frame-0 motion is what gravity drives (DROP_HEIGHT).
#
# Use REST_FROM_WIND=0 only for the "tape with memory" interpretation
# — a long-stored roll whose wound shape IS the rest pose. That
# variant won't spring-back on peel either.
REST_FROM_WIND     = int(_CFG.get("REST_FROM_WIND", 1))

ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_winding")
if _CFG.get("__list_assets__"):
    L.list_assets(ASSET_DIR)
ASSET_IN_PATH = L.resolve_asset_path(
    ASSET_DIR, _CFG.get("__asset_arg__"), _CFG["__preset_name__"])
print(f"[drop] load source: {ASSET_IN_PATH}")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _straight_rest_positions(R_anchor, length, width, NX, NZ):
    """The same straight-tangent layout wind originally built its rest
    around. Used as the FEM strain reference when REST_FROM_WIND=1.

    Returned in the wind demo's coord frame (x=R_anchor, long axis y,
    width z). NeoHookean strain is per-element so a global rotation
    between rest and current cancels out — no need to apply the
    `_stand_on_ground` transform here.
    """
    dy = length / NX
    dz = width  / NZ
    n_width = NZ + 1
    verts = np.empty(((NX + 1) * n_width, 3), dtype=np.float64)
    for i in range(NX + 1):
        for j in range(n_width):
            verts[i * n_width + j] = (R_anchor, i * dy, j * dz - 0.5 * width)
    return verts


def _make_tape_topology_tris(NX, NZ):
    """Identical triangle layout / winding to wind & unwind demos so
    `set_sticky_side(-1)` still tags the hub-facing side as sticky."""
    n_width = NZ + 1
    def vid(i, j): return i * n_width + j
    tris = []
    for i in range(NX):
        for j in range(NZ):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    return np.asarray(tris, dtype=np.int32)


def _make_tape_sc(positions, tris):
    sc = trimesh(positions, tris)
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc


def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


# ----------------------------------------------------------------------
# Stand the roll on the ground.
#
# Asset frame: wind demo saves the roll with hub axis along world +z
# (axis horizontal; the round face faces ±z). To make it stand up like
# a tape dispenser on a desk, we apply R_x(-90°) which sends world +z
# to world +y. Under that rotation, vertex (x, y, z) → (x, z, -y).
#
# Then we shift upward in +y so the lowest geometry point (hub or
# tape — hub is usually slightly taller than the tape width) sits
# `ground_clearance` above the ground plane y=0.
# ----------------------------------------------------------------------
_R_X_NEG90 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _stand_on_ground(tape_pos: np.ndarray,
                     hub_T:    np.ndarray,
                     hub_height: float,
                     ground_clearance: float):
    """Return (tape_pos_new, hub_T_new) with the assembly rotated so
    the hub axis stands vertical (+y) and translated so the lowest
    point sits `ground_clearance` above y=0."""
    # rotate tape verts: (x, y, z) → (x, z, -y)
    tape_rot = np.column_stack([
        tape_pos[:, 0],
        tape_pos[:, 2],
        -tape_pos[:, 1],
    ])
    hub_T_rot = _R_X_NEG90 @ hub_T

    # After the rotation the hub's local +y axis ends up on world +y,
    # so its local z-extent [-H/2, H/2] becomes the world y-extent
    # [-H/2, H/2] around the hub's translated center. Use whichever is
    # lower — tape or hub — as the contact reference.
    hub_center_y = float(hub_T_rot[1, 3])
    hub_bottom_y = hub_center_y - 0.5 * hub_height
    min_y = min(float(tape_rot[:, 1].min()), hub_bottom_y)

    shift_y = ground_clearance - min_y
    tape_rot[:, 1] += shift_y
    T_shift = np.eye(4)
    T_shift[1, 3] = shift_y
    return tape_rot, T_shift @ hub_T_rot


# ----------------------------------------------------------------------
def build_demo(adhesion_on: bool = True,
               bonded: bool = False,
               beta_lock_threshold: float = 0.9,
               kappa: float = 1.0e8,
               release_force: float = 1.0e30):
    # Default Warn; bump to e.g. info/debug via `--set LOG_LEVEL=info`.
    L.apply_log_level(_CFG, default="warn")

    if not os.path.isfile(ASSET_IN_PATH):
        raise SystemExit(
            f"Asset not found: {ASSET_IN_PATH}\n"
            f"Run the wind demo first and click 'save asset':\n"
            f"  python/.venv/bin/python python/examples/rcc_adhesive_tape_winding_demo.py")

    hub_T, tape_pos, params, pair_state, tape_vel = L.load_tape_asset(ASSET_IN_PATH)
    HUB_R_OUTER = float(params["HUB_R_OUTER"])
    HUB_R_INNER = float(params["HUB_R_INNER"])
    HUB_HEIGHT  = float(params["HUB_HEIGHT"])
    TAPE_LENGTH = float(params["TAPE_LENGTH"])
    TAPE_WIDTH  = float(params["TAPE_WIDTH"])
    TAPE_NX     = int(params["TAPE_NX"])
    TAPE_NZ     = int(params["TAPE_NZ"])

    saved_preset = params.get("__preset_name__", "(unknown)")
    print(f"loaded asset [preset={saved_preset}]: tape ({TAPE_NX+1}×{TAPE_NZ+1} verts), "
          f"hub R∈[{HUB_R_INNER},{HUB_R_OUTER}], L_tape={TAPE_LENGTH:.3f} m")

    # Asset-loaded values win for IPC numerics, material, and adhesion
    # — same `--set > asset > preset` precedence rule for every
    # physical parameter. Geometry (HUB_*, TAPE_LENGTH/WIDTH/NX/NZ) is
    # always taken from the asset above; β state is restored after
    # world.init below. Shadows the matching module-level constants
    # for the rest of build_demo.
    def _resolve_and_log(key, label, fmt=".4e"):
        before = _CFG[key]
        after  = L.resolve_param(_CFG, params, key)
        if isinstance(before, float) and abs(after - before) > 1e-12:
            print(f"  {label:<18s} ← asset {after:{fmt}}  "
                  f"(preset value {before:{fmt}} overridden)")
        return after

    D_HAT             = _resolve_and_log("D_HAT",             "D_HAT")
    TAPE_THICKNESS    = _resolve_and_log("TAPE_THICKNESS",    "TAPE_THICKNESS")
    TAPE_YOUNGS       = _resolve_and_log("TAPE_YOUNGS",       "TAPE_YOUNGS")
    BENDING_STIFFNESS = _resolve_and_log("BENDING_STIFFNESS", "BENDING_STIFFNESS")
    TAPE_POISSON      = _resolve_and_log("TAPE_POISSON",      "TAPE_POISSON", ".3f")
    TAPE_MASS_DENSITY = _resolve_and_log("TAPE_MASS_DENSITY", "TAPE_MASS_DENSITY", ".1f")
    ADH_CN            = _resolve_and_log("ADH_CN",            "ADH_CN")
    ADH_CT            = _resolve_and_log("ADH_CT",            "ADH_CT")
    ADH_W             = _resolve_and_log("ADH_W",             "ADH_W",   ".3f")
    ADH_ETA           = _resolve_and_log("ADH_ETA",           "ADH_ETA", ".3f")
    ADH_BONDING_RATE  = _resolve_and_log("ADH_BONDING_RATE",  "ADH_BONDING_RATE", ".3f")
    ADH_INITIAL_BETA  = _resolve_and_log("ADH_INITIAL_BETA",  "ADH_INITIAL_BETA", ".3f")

    # ---- stand the roll on the ground (axis +y). Lowest geometry
    # vertex ends up at (TAPE_THICKNESS + 0.5·D_HAT) + DROP_HEIGHT
    # above y=0. The first summand keeps it inside the IPC active band
    # so the barrier is well-defined; DROP_HEIGHT (default 0) adds an
    # optional free-fall margin so gravity has a visible drop phase
    # before ground contact engages.
    GROUND_CLEARANCE = TAPE_THICKNESS + 0.5 * D_HAT + DROP_HEIGHT
    tape_pos_lay, hub_T_lay = _stand_on_ground(
        tape_pos, hub_T, HUB_HEIGHT, GROUND_CLEARANCE)
    hub_bottom_y = float(hub_T_lay[1, 3]) - 0.5 * HUB_HEIGHT
    print(f"[drop] hub bottom y={hub_bottom_y*1e3:.3f} mm, "
          f"tape lowest y={tape_pos_lay[:,1].min()*1e3:.3f} mm "
          f"(clearance = {GROUND_CLEARANCE*1e3:.3f} mm above ground: "
          f"IPC band {(TAPE_THICKNESS + 0.5*D_HAT)*1e3:.3f} + "
          f"drop {DROP_HEIGHT*1e3:.3f})")

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
    # SOFT RCC normal-adhesion energy-minimum offset (applies whether or not
    # bonding is on). Leave at the engine default (0.5 = band center d*=xi+d_hat/2,
    # a gentle spring to a natural gap instead of pulling into the C-IPC barrier
    # wall) unless overridden. `--set RCC_ADHESION_NORMAL_OFFSET_COEFF=0` -> legacy
    # min at d=0; =1 -> band outer edge.
    _off = _CFG.get("RCC_ADHESION_NORMAL_OFFSET_COEFF")
    if _off is not None:
        config["rcc_adhesion_normal_offset_coeff"] = float(_off)
    if bonded:
        # RCC bonded-PT acceleration: stable high-beta face-interior tape
        # contacts are replaced by a stiff ABD virtual tet (point-plane).
        config["rcc_bonded_pt_enabled"] = 1
        # CCD on locked pairs. Default -1 = auto (skip CCD for locked pairs when
        # bonded, since the ABD tet owns them). `--set SKIP_CCD=0` keeps CCD on
        # for locked pairs (the last non-penetration guard; may hit the D=0
        # thickness assert on over-compressed wound layers). `--set SKIP_CCD=1`
        # forces skip.
        config["rcc_bonded_pt_skip_ccd"] = int(_CFG.get("SKIP_CCD", -1))
        config["rcc_bonded_pt_beta_lock_threshold"] = beta_lock_threshold
        # Default: ALL VTs may bond (incl. edge/corner) — maximizes the locked
        # fraction but risks skewed sliver tets. `--set LOCK_FACE_INTERIOR_ONLY=1`
        # restricts to face-interior VTs (sound point-plane tets, fewer locks).
        config["rcc_bonded_pt_lock_face_interior_only"] = (
            1 if L.cfg_flag(_CFG, "LOCK_FACE_INTERIOR_ONLY", default=False) else 0)
        # Virtual-tet constitution: "abd_ortho" (kappa) [default] or
        # "stable_neo_hookean" (Young+Poisson). RUNTIME config (NOT asset-saved),
        # so wind / drop / rod-wind must all pass the same RCC_ENERGY_MODEL.
        _energy_model = str(_CFG.get("RCC_ENERGY_MODEL", "abd_ortho"))
        config["rcc_bonded_pt_energy_model"] = _energy_model
        # Bond stiffness. `--set RCC_KAPPA=1e7` softens the ABD bond (better
        # Hessian conditioning / fewer line-search blowups on thin sliver tets,
        # at the cost of softer bonds). Default = the build_demo kwarg (1e8).
        config["rcc_bonded_pt_kappa"] = float(_CFG.get("RCC_KAPPA", kappa))
        if _energy_model == "stable_neo_hookean":
            config["rcc_bonded_pt_neohookean_young"] = float(
                _CFG.get("RCC_NEOHOOKEAN_YOUNG", 5.0e7))
            config["rcc_bonded_pt_neohookean_poisson"] = float(
                _CFG.get("RCC_NEOHOOKEAN_POISSON", 0.45))
        # Release threshold (1e30 = never release). Per-preset RCC_RELEASE_FORCE
        # energy-matches the non-bonded debonding load; see the note in
        # tape_asset_lib.py above WIND_PRESETS.
        config["rcc_bonded_pt_release_force"] = release_force
        # Phase 7 distance-locked bonding. Precedence: `--set DISTANCE_LOCK=...`
        # wins; else the asset's saved flag (a tape wound in distance-lock mode
        # auto-replays in it). No soft adhesion energy / beta in this mode; the
        # lock gate is the end-of-step distance band d < xi + c*d_hat. Caveat
        # vs beta mode: a force-released bond relocks next step while the pair
        # is still inside the band (no load-based relock suppression).
        if L.resolve_flag(_CFG, params, "DISTANCE_LOCK", default=False):
            config["rcc_bonded_pt_distance_lock"] = 1
            _ratio = _CFG.get("DISTANCE_LOCK_RATIO")
            if _ratio is None:
                _ratio = params.get("DISTANCE_LOCK_RATIO", 0.5)
            config["rcc_bonded_pt_distance_lock_ratio"] = float(_ratio)
    # User-facing solver knobs (e.g. `--set LIN_TOL_RATE=1e-5
    # --set NEWTON_VELOCITY_TOL=0.005`) get translated into the
    # libuipc nested config here, AFTER the demo's own defaults so
    # CLI always wins. Passing `params` makes asset-saved solver
    # knobs (from `wind --set …`) auto-apply too — matches the
    # `--set > asset > preset` rule we use for material/adhesion.
    # See SOLVER_KEYS in tape_asset_lib.
    L.apply_solver_overrides(config, _CFG, params=params)
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    spc = SoftPositionConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    hub_contact    = tabular.default_element()          # used by hub + ground
    tape_contact   = tabular.create("tape")
    tabular.insert(tape_contact, tape_contact, 0.5, 1.0e9)
    tabular.insert(tape_contact, hub_contact,  0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        # disable adhesion on default pairs (hub–hub, hub–ground, …)
        adhesive.default_model(
            tabular,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )
        adhesive.set(
            tabular, tape_contact, tape_contact,
            Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
            bonding_rate=ADH_BONDING_RATE, p0=0.0,
            initial_beta=ADH_INITIAL_BETA,
            enabled=True,
        )
        adhesive.set(
            tabular, tape_contact, hub_contact,
            Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
            bonding_rate=ADH_BONDING_RATE, p0=0.0,
            initial_beta=ADH_INITIAL_BETA,
            enabled=True,
        )
        # PER-PAIR bonded-PT overrides (only meaningful with bonded on). Each
        # value < 0 inherits the global rcc_bonded_pt_* config, so by default
        # this is a no-op. Lets tape-tape and tape-hub lock / release under
        # different conditions, e.g. `--set BONDED_HUB_LOCK=0.1` to bond the
        # tape to the hub more readily than tape-to-tape, or
        # `--set BONDED_TAPE_RELEASE_FORCE=3e-7` to peel inter-layer bonds while
        # the tape-hub bond holds.
        _tt_lk = float(_CFG.get("BONDED_TAPE_LOCK", -1.0))
        _th_lk = float(_CFG.get("BONDED_HUB_LOCK", -1.0))
        _tt_rf = float(_CFG.get("BONDED_TAPE_RELEASE_FORCE", -1.0))
        _th_rf = float(_CFG.get("BONDED_HUB_RELEASE_FORCE", -1.0))
        adhesive.set_bonded(tabular, tape_contact, tape_contact,
                            lock_threshold=_tt_lk, release_strain=-1.0,
                            release_gap=-1.0, release_slip=-1.0, release_force=_tt_rf)
        adhesive.set_bonded(tabular, tape_contact, hub_contact,
                            lock_threshold=_th_lk, release_strain=-1.0,
                            release_gap=-1.0, release_slip=-1.0, release_force=_th_rf)

    # ---- hub (FREE — no is_fixed, no SPC) ----
    hub_sc = L.make_ring_hub(
        R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
        height=HUB_HEIGHT, n_radial=48,
        center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    hub_contact.apply_to(hub_sc)
    view(hub_sc.transforms())[0] = _mat4_to_uipc(hub_T_lay)
    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- tape: REST geometry choice — see REST_FROM_WIND knob above.
    # Default (REST_FROM_WIND=1): rest = wind's original straight-
    # tangent strip → preserves wind's elastic-vs-adhesion force
    # balance → frame 0 is a true equilibrium except for gravity.
    # Legacy (REST_FROM_WIND=0): rest = wound = current → no elastic
    # spring-back when layers debond; but adhesion is no longer
    # balanced at frame 0, causing visible collapse.
    tris = _make_tape_topology_tris(TAPE_NX, TAPE_NZ)
    current_sc = _make_tape_sc(tape_pos_lay, tris)
    if REST_FROM_WIND:
        # Same R_anchor formula wind used → rest matches wind's setup
        # element-by-element. Per-element strain absorbs any global
        # rotation between rest and current's coord frames.
        R_anchor_rest = HUB_R_OUTER + TAPE_THICKNESS + 0.5 * D_HAT
        rest_positions = _straight_rest_positions(
            R_anchor_rest, TAPE_LENGTH, TAPE_WIDTH, TAPE_NX, TAPE_NZ)
        rest_sc = _make_tape_sc(rest_positions, tris)
    else:
        rest_sc = _make_tape_sc(tape_pos_lay, tris)

    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    dsb = DiscreteShellBending()
    nhs.apply_to(current_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    dsb.apply_to(current_sc, BENDING_STIFFNESS)
    nhs.apply_to(rest_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    dsb.apply_to(rest_sc, BENDING_STIFFNESS)
    tape_contact.apply_to(current_sc)
    spc.apply_to(current_sc, SPC_STRENGTH)
    if adhesion_on:
        RCCAdhesive.set_sticky_side(current_sc, -1)

    # Seed FEM velocity from asset (None on legacy assets → v=0 default).
    # Drop applies the same R_x(-90°) rotation that `_stand_on_ground`
    # applied to positions: (vx, vy, vz) → (vx, vz, -vy).
    # The shift in y is a translation, which doesn't affect velocity.
    # libuipc stores Vector3 attributes as (N, 3, 1) so we reshape.
    if tape_vel is not None and tape_vel.shape == tape_pos.shape:
        tape_vel_lay = np.column_stack([tape_vel[:, 0],
                                         tape_vel[:, 2],
                                         -tape_vel[:, 1]])
        existing = current_sc.vertices().find("velocity")
        if existing is None:
            current_sc.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        vel_view = view(current_sc.vertices().find("velocity"))
        vel_view[:] = tape_vel_lay.reshape(-1, 3, 1)
        vmax = float(np.linalg.norm(tape_vel_lay, axis=1).max())
        if vmax > 0:
            print(f"[drop] seeded tape velocity: max={vmax:.3e} m/s")

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(current_sc, rest_sc)

    # ---- ground at y=0 ----
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    # ---- free-end SPC + lift animation ----
    # Tape mesh uses vid(i, j) = i*(NZ+1) + j (matches
    # _make_tape_topology_tris). The free end is the last row of
    # vertices along the tape's length direction — these stuck out
    # tangentially from the wound spool at the end of the wind sim.
    def vid_drop(i, j): return i * (TAPE_NZ + 1) + j
    # Pinch a SINGLE vertex (centre of the tape's free-end row) rather
    # than the whole row. Visually mimics "pick the tape up by one
    # point" — the rest of the free edge is free to flop, which gives
    # a more realistic peel/drop response than yanking on 11 verts in
    # lockstep. Wider TAPE_NZ → still just the middle vert; the asset
    # geometry doesn't need to change.
    free_ids = [vid_drop(TAPE_NX, TAPE_NZ // 2)]

    def smooth_lerp(a: float, b: float, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        s = 0.5 - 0.5 * np.cos(np.pi * t)  # cosine ease (slow start + end)
        return a + (b - a) * s

    anim_state = {
        # UI toggle: when False, animate_tape writes no constraints —
        # the free end is governed entirely by physics. Toggling back
        # to True triggers a one-shot snapshot of the live free-end
        # positions so the SPC trajectory restarts from wherever the
        # tape currently is, not from a stale frozen target.
        "pull_enabled":   True,
        # Captured at the first PULL-phase frame (or after OFF→ON).
        # Each free vertex's aim is `pull_start_pos[jj] + (0, lift_y, 0)`.
        "pull_start_pos": None,
    }

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim  = view(geo.vertices().find(builtin.aim_position))
        is_c[:] = 0

        # Pull disabled by UI → leave free end unconstrained, drop the
        # stale start-pos snapshot so re-enabling captures fresh.
        if not anim_state["pull_enabled"]:
            anim_state["pull_start_pos"] = None
            return
        # HOLD phase: free end free, let gravity settle the assembly.
        if f < HOLD_FRAMES:
            anim_state["pull_start_pos"] = None
            return

        # FREEFALL phase: SPC released, free end drops under gravity.
        # is_c[:] is already cleared at the top of this function, so
        # simply returning leaves every vertex unconstrained — the
        # lifted roll + tail fall together (and whatever's bonded
        # falls with them).
        if f >= HOLD_FRAMES + PULL_FRAMES + TOP_FRAMES:
            anim_state["pull_start_pos"] = None
            return

        # On PULL phase entry (or after OFF→ON edge) snapshot the live
        # free-end positions so the ramp departs from where the tape
        # actually is, with no positional jump.
        if anim_state["pull_start_pos"] is None:
            live = np.asarray(view(geo.positions())).reshape(-1, 3)
            anim_state["pull_start_pos"] = live[free_ids].copy()

        # Compute current lift height. After PULL_FRAMES the ramp
        # saturates at LIFT_HEIGHT and SPC holds the free end there.
        pull_idx = f - HOLD_FRAMES
        t = min(pull_idx / max(PULL_FRAMES, 1), 1.0)
        lift_y = smooth_lerp(0.0, LIFT_HEIGHT * LIFT_FRACTION, t)

        start = anim_state["pull_start_pos"]
        for jj, k in enumerate(free_ids):
            is_c[k] = 1
            aim[k] = np.array([start[jj, 0],
                               start[jj, 1] + lift_y,
                               start[jj, 2]], dtype=np.float64).reshape(3, 1)

    scene.animator().insert(tape_obj, animate_tape)

    world.init(scene)

    # Restore β from the asset if it was saved with one. Must happen AFTER
    # world.init() (the backend feature is only registered then) and BEFORE
    # the first world.advance() (so the next step's Phase B reads the
    # loaded prev-state via match_or_init instead of init_all_new). With
    # the wound β=1 restored on existing layer pairs, the user can use
    # `--set ADH_INITIAL_BETA=0` to keep new tail-flop contacts unbonded.
    if adhesion_on and pair_state is not None:
        keys, betas = pair_state
        acc = world.features().find(RCCAdhesionStateAccessorFeature)
        if acc is not None and len(betas) > 0:
            acc.load_pt_state(keys, betas)
            print(f"[drop] restored β: n={len(betas)} pairs, "
                  f"mean={betas.mean():.3f}, frac>0.9={(betas > 0.9).mean():.2%}")
        else:
            if acc is None:
                print("[drop] WARNING: RCCAdhesionStateAccessorFeature not found "
                      "— β was not restored.")

    # Restore the bonded-PT lock state directly: re-lock the saved bonds against
    # the loaded geometry BEFORE the first advance, so step 1's trajectory
    # filter already compacts them out (no re-form transient / no doubling /
    # no soft-instead-rigid first step). Falls back to the merged-β re-form path
    # for legacy assets without bonded_locked_topos.
    if bonded:
        locked_pairs = L.load_tape_locked_pairs(ASSET_IN_PATH)
        bpt = world.features().find(RCCBondedPTStateAccessorFeature)
        if locked_pairs is not None and bpt is not None:
            topos, lbetas = locked_pairs
            bpt.seed_locks(topos, lbetas, beta_lock_threshold)
            print(f"[drop] seeded {len(lbetas)} bonded locks from asset "
                  f"(threshold={beta_lock_threshold:g}).")

    return {
        "engine": engine, "world": world, "scene": scene,
        "scene_io": SceneIO(scene),
        "hub_geo": hub_geo, "tape_geo": tape_geo,
        "params": params,
        # Animator state — the UI's "pull" button flips
        # state["pull_enabled"] live; animate_tape consumes it on the
        # next frame.
        "anim_state": anim_state,
    }


def _adhesion_pt_counts(sim):
    """(soft, locked): # of soft-adhesion PT pairs (evolving beta, excludes
    locked) and # of bonded/locked PT pairs. Either is None if its accessor is
    absent (e.g. bonded disabled). soft + locked = all PTs in the RCC system."""
    feats = sim["world"].features()
    rcc = feats.find(RCCAdhesionStateAccessorFeature)
    bpt = feats.find(RCCBondedPTStateAccessorFeature)
    soft = int(rcc.pt_pair_count()) if rcc is not None else None
    locked = int(bpt.locked_pair_count()) if bpt is not None else None
    return soft, locked


def _bonded_bonds(sim):
    """(locked_count, (nodes, edges)) for the bonded virtual tets, or (count, None).

    Each locked PT is drawn as the 6 edges of its (point, t0, t1, t2) tet.
    """
    acc = sim["world"].features().find(RCCBondedPTStateAccessorFeature)
    if acc is None:
        return 0, None
    locked = int(acc.locked_pair_count())
    pts = np.asarray(acc.dump_locked_tet_world_positions(), dtype=np.float64)
    if pts.ndim != 3 or pts.shape[0] == 0:
        return locked, None
    nodes = pts.reshape(-1, 3)
    edges = []
    for i in range(pts.shape[0]):
        b = 4 * i
        edges += [[b, b + 1], [b, b + 2], [b, b + 3],
                  [b + 1, b + 2], [b + 2, b + 3], [b + 3, b + 1]]
    return locked, (nodes, np.asarray(edges, dtype=np.int64))


def _print_stage_breakdown(timer_frames, top=16):
    """Aggregate the top-level timer stages (direct children of each frame's
    root node) across the profiled frames and print a per-stage breakdown.
    Timer `duration` is in seconds; printed as ms. The full nested tree is in
    the saved report/."""
    from collections import defaultdict
    tot, cnt = defaultdict(float), defaultdict(int)
    frames = [f for f in (timer_frames or []) if f]

    def stage_children(node):
        # Descend through wrapper nodes (single-child, e.g. root->"Pipeline", or
        # a single dominant child holding ~all the time, e.g. "Simulation") to
        # the first level where the time actually splits across stages.
        cur, hops = node, 0
        while hops < 12:
            kids = cur.get("children") or []
            if not kids:
                return []
            if len(kids) == 1:
                cur, hops = kids[0], hops + 1
                continue
            durs = sorted(((k, float(k.get("duration", 0.0) or 0.0)) for k in kids),
                          key=lambda kd: -kd[1])
            total = sum(d for _, d in durs) or 1.0
            if durs[0][1] / total > 0.9:          # one child dominates -> descend
                cur, hops = durs[0][0], hops + 1
                continue
            return kids
        return cur.get("children") or []

    for fr in frames:
        for r in (fr if isinstance(fr, list) else [fr]):
            for c in stage_children(r):
                tot[c.get("name", "?")] += float(c.get("duration", 0.0) or 0.0)
                cnt[c.get("name", "?")] += int(c.get("count", 0) or 0)
    if not tot:
        print("[profile] (no top-level timer stages captured)")
        return
    nf = max(len(frames), 1)
    grand = sum(tot.values())
    print(f"[profile] per-stage (top level), {nf} frames, "
          f"{grand * 1000:.1f} ms total  ({grand / nf * 1000:.2f} ms/frame):")
    for name, sec in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
        print(f"    {name:40s} {sec * 1000:10.1f} ms  "
              f"{sec / nf * 1000:8.3f} ms/frame  {sec / grand * 100:5.1f}%  "
              f"(n={cnt[name]})")


def run_demo():
    # Bonded-PT acceleration. Precedence: `--set BONDED=...` wins; else the
    # loaded asset's saved BONDED flag (a tape wound WITH bonded auto-enables
    # it on load); else OFF. Defaults OFF because a tightly wound tape can
    # destabilize the stiff ABD bonds at the default kappa (face contacts
    # over-compress to a zero gap -> trajectory-filter thickness assert).
    _asset_params = L.peek_asset_params(ASSET_IN_PATH)
    bonded_on = L.resolve_flag(_CFG, _asset_params, "BONDED", default=False)
    # asset > preset (same precedence as Cn/W/D_HAT) so a bonded asset keeps the
    # release load it was wound with; falls back to the preset for legacy assets
    # that predate RCC_RELEASE_FORCE.
    release_force = float(L.resolve_param(_CFG, _asset_params, "RCC_RELEASE_FORCE"))
    beta_lock = float(L.resolve_param(_CFG, _asset_params, "RCC_BETA_LOCK_THRESHOLD"))
    state = {"adhesion_on": True, "bonded": bonded_on,
             "release_force": release_force, "beta_lock": beta_lock}
    sim = build_demo(state["adhesion_on"], bonded=state["bonded"],
                     beta_lock_threshold=state["beta_lock"], release_force=state["release_force"])

    # Ground quad shared between interactive and headless paths.
    ground_quad_verts = np.array([
        [-0.5, 0.0, -0.5],
        [ 0.5, 0.0, -0.5],
        [ 0.5, 0.0,  0.5],
        [-0.5, 0.0,  0.5],
    ], dtype=np.float64)
    ground_quad_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    # Headless record mode: when RECORD_DIR is set, skip the interactive
    # polyscope GUI entirely and dump a PNG sequence via EGL. Must
    # branch BEFORE any `ps.init()` (which would try a display backend
    # on headless boxes and crash). Combine PNGs to MP4 with the
    # ffmpeg command the helper prints at the end.
    record_dir = _CFG.get("RECORD_DIR")
    if record_dir:
        every_n = int(_CFG.get("RECORD_EVERY", 10))
        # Default zoom < 1 = wider FoV (zoom 0.6 → FoV ≈ 75°). The drop
        # demo's vertical trajectory (HOLD → PULL → TOP → FREEFALL spans
        # `LIFT_HEIGHT + DROP_HEIGHT` ≈ 11 cm) needs a wider view than
        # the wind/sweep defaults; 0.6 keeps the entire fall in frame.
        zoom    = float(_CFG.get("RECORD_ZOOM", 0.6))
        def _setup_extras(ps_mod):
            gm = ps_mod.register_surface_mesh(
                "ground", ground_quad_verts, ground_quad_tris)
            gm.set_color((0.6, 0.6, 0.6))
            gm.set_transparency(0.5)
        def _on_progress(f, tot):
            # Lift verdict probes: locked-bond census + whole-assembly
            # altitude. A held lift keeps locked ~flat and tape_min_y well
            # above ground through TOP; a shedding roll collapses locked
            # and tape_min_y returns to the contact band.
            soft, locked = _adhesion_pt_counts(sim)
            tp = sim["tape_geo"].geometry().positions().view().reshape(-1, 3)
            hub_y = view(sim["hub_geo"].geometry().transforms())[0][1, 3]
            print(f"[record] frame {f}/{tot} ({f/tot*100:.1f}%)  "
                  f"Phase: {phase_at(f - 1)}  locked={locked} soft={soft} "
                  f"tape_min_y={tp[:, 1].min():.4f} hub_y={hub_y:.4f}")
        # Tell the helper to frame for the full vertical trajectory
        # (roll diameter + LIFT_HEIGHT + DROP_HEIGHT) so the lifted
        # tape stays in view at the end of the pull. Without this,
        # at zoom=5 the bbox is computed only from the roll's initial
        # pose and the lifted tape flies off-screen mid-sim.
        # HUB_R_OUTER / HUB_HEIGHT live in the asset params (not at
        # module scope), so read them via sim["params"].
        _hub_r = float(sim["params"]["HUB_R_OUTER"])
        _hub_h = float(sim["params"]["HUB_HEIGHT"])
        full_extent = max(2.0 * _hub_r,
                          LIFT_HEIGHT + DROP_HEIGHT + 2.0 * _hub_h)
        L.record_demo_to_pngs(sim=sim, total_frames=TOTAL_FRAMES,
                              output_dir=record_dir, every_n=every_n,
                              up_dir="y_up", mesh_name="drop_tape",
                              setup_extras_fn=_setup_extras,
                              zoom=zoom, on_progress=_on_progress,
                              bbox_extent_override=full_extent)
        return

    # Headless profiling: `--set PROFILE=1` runs the drop headlessly, collects
    # per-stage timer stats via uipc.profile, prints a breakdown, and saves
    # benchmark.json / timer_frames.json / a report/ under PROFILE_DIR. Knobs:
    # `--set PROFILE_FRAMES=N` (default = TOTAL_FRAMES), `--set PROFILE_WARMUP=N`
    # (frames advanced before stats start), `--set PROFILE_DIR=path`. Like
    # RECORD_DIR, must branch BEFORE ps.init() so it stays headless.
    if L.cfg_flag(_CFG, "PROFILE", default=False):
        from uipc import profile as _uprofile
        n_frames = int(_CFG.get("PROFILE_FRAMES", TOTAL_FRAMES))
        warmup   = int(_CFG.get("PROFILE_WARMUP", 0))
        out_dir  = _CFG.get("PROFILE_DIR") or os.path.join(
            AssetDir.output_path(__file__), "profile_drop")
        name = (os.path.splitext(os.path.basename(ASSET_IN_PATH))[0]
                + ("_bonded" if state["bonded"] else "_nobond"))
        print(f"[profile] {name}: warmup={warmup}, profile={n_frames} frames "
              f"(bonded={state['bonded']}, kappa via --set RCC_KAPPA) -> "
              f"{os.path.join(out_dir, name)}")
        with _uprofile.session(sim["world"], name=name, output_dir=out_dir) as s:
            if warmup > 0:
                s.advance(warmup)
            s.profile(n_frames)
        res = s.result
        print("[profile] " + res["summary"])
        _print_stage_breakdown(res["timer_frames"])
        print(f"[profile] saved -> {os.path.join(out_dir, name)}/ "
              f"(benchmark.json, timer_frames.json, report/)")
        return

    # Interactive path: needs a display.
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    # libuipc's `ground(0.0)` is an implicit half-plane — the contact
    # engine uses it but SceneIO won't return it through
    # `simplicial_surface()`, so we register a flat quad at y=0 just
    # for visualization. Size: 1 m × 1 m centered at origin (much
    # bigger than the roll's footprint).
    ground_mesh = ps.register_surface_mesh(
        "ground", ground_quad_verts, ground_quad_tris)
    ground_mesh.set_color((0.6, 0.6, 0.6))
    ground_mesh.set_transparency(0.5)

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "drop_tape",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(0.3)

    ui = {"run": False, "show_bonds": True}
    bonds_net = [None]

    def update_bonds():
        # Bonded virtual tets as a red curve network; re-register on lock-count
        # change, else move nodes; removed when toggled off or no locks.
        if not ui["show_bonds"]:
            if bonds_net[0] is not None:
                ps.remove_curve_network("bonds")
                bonds_net[0] = None
            return
        _, data = _bonded_bonds(sim)
        if data is None:
            if bonds_net[0] is not None:
                ps.remove_curve_network("bonds")
                bonds_net[0] = None
            return
        nodes, edges = data
        if bonds_net[0] is None or bonds_net[0].n_nodes() != nodes.shape[0]:
            if bonds_net[0] is not None:
                ps.remove_curve_network("bonds")
            net = ps.register_curve_network("bonds", nodes, edges)
            net.set_radius(0.004)
            net.set_color((1.0, 0.15, 0.1))
            bonds_net[0] = net
        else:
            bonds_net[0].update_node_positions(nodes)

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        v = merged.positions().view().reshape(-1, 3)
        t = merged.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != v.shape[0]:
            ps.remove_surface_mesh("drop_tape")
            mesh = ps.register_surface_mesh("drop_tape", v, t)
            mesh.set_edge_width(0.3)
        else:
            mesh.update_vertex_positions(v)
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
        sim = build_demo(state["adhesion_on"], bonded=state["bonded"],
                         beta_lock_threshold=state["beta_lock"], release_force=state["release_force"])
        update_visual()

    # phase_at is now module-level (see above), reused by the
    # interactive on_update text below.

    def on_update():
        if psim.Button("run / pause"):
            ui["run"] = not ui["run"]
        psim.SameLine()
        if psim.Button("step"):
            step_once()
        psim.SameLine()
        if psim.Button(f"adhesion: {'ON' if state['adhesion_on'] else 'OFF'}"):
            state["adhesion_on"] = not state["adhesion_on"]
        psim.SameLine()
        anim = sim["anim_state"]
        if psim.Button(f"pull: {'ON' if anim['pull_enabled'] else 'OFF'}"):
            anim["pull_enabled"] = not anim["pull_enabled"]
            # animate_tape consumes pull_start_pos=None on the next
            # frame as the OFF→ON re-snapshot trigger.
        psim.SameLine()
        if psim.Button("reset"):
            reset()
        if state["bonded"]:
            psim.SameLine()
            if psim.Button(f"bonds: {'ON' if ui['show_bonds'] else 'OFF'}"):
                ui["show_bonds"] = not ui["show_bonds"]
                update_bonds()

        if ui["run"]:
            step_once()

        f = min(sim["world"].frame(), TOTAL_FRAMES)
        psim.Separator()
        psim.Text(f"Frame: {f} / {TOTAL_FRAMES}    Phase: {phase_at(f)}")
        soft_pt, locked_pt = _adhesion_pt_counts(sim)
        if soft_pt is not None:
            if locked_pt:
                psim.Text(f"adhesion PTs: {soft_pt + locked_pt}  "
                          f"({soft_pt} soft + {locked_pt} locked)")
            else:
                psim.Text(f"adhesion PTs: {soft_pt}")
        if state["bonded"]:
            locked, _ = _bonded_bonds(sim)
            psim.Text(f"bonded locks: {locked}  (red tets; toggle above)")

        # Pull progress + lift height in mm
        f1 = max(f - 1, 0)
        if f1 < HOLD_FRAMES:
            t = 0.0
        elif f1 < HOLD_FRAMES + PULL_FRAMES:
            t = (f1 - HOLD_FRAMES) / PULL_FRAMES
        else:
            t = 1.0
        # smooth_lerp progress; raw t is enough for the UI bar
        lift_mm = (0.5 - 0.5 * np.cos(np.pi * float(np.clip(t, 0, 1)))) * LIFT_HEIGHT * 1000
        if f1 >= HOLD_FRAMES + PULL_FRAMES + TOP_FRAMES:
            pull_state = "RELEASED (freefall)"
        elif anim["pull_enabled"]:
            pull_state = "ENABLED"
        else:
            pull_state = "DISABLED (free end)"
        psim.Text(f"Lift: {t*100:.1f}%  ({lift_mm:.1f} mm of {LIFT_HEIGHT*1000:.0f} mm)  SPC: {pull_state}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
