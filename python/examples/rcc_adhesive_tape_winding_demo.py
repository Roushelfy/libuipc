"""
Guided-winding asset demo (Option C).

A straight tape strip is positioned tangent to a FIXED ABD hub:
  - The anchor end (i=0) starts in contact with the hub surface at angle 0.
  - The free end (i=NX) is pinned by `SoftPositionConstraint` and animated
    along a spiral trajectory around the hub axis (+z). As the free end
    orbits + spirals inward, the tape between it and the anchor drapes
    onto the hub and bonds layer-by-layer via v3 single-sided adhesion.

Timeline:
    Phase 0 (preheat) : both ends pinned; adhesion β rises to 1 on anchor.
    Phase 1 (wind)    : anchor SPC released; free-end target follows the
                        winding spiral. Tape physically wraps onto hub.
    Phase 2 (settle)  : free-end stops; system relaxes.

Run:
    python python/examples/rcc_adhesive_tape_winding_demo.py
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
from uipc.geometry import trimesh, label_surface, mesh_partition
from uipc.core import (RCCAdhesionStateAccessorFeature,
                       FiniteElementStateAccessorFeature,
                       RCCBondedPTStateAccessorFeature)
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


# ---- CLI / preset resolution ----
# Pick a parameter bundle with `--preset`; tweak any single key with
# `--set KEY=VALUE`; list all bundles with `--list`. See
# `tape_asset_lib.TAPE_PRESETS` for what each preset contains.
_CFG = L.parse_tape_cli(L.WIND_PRESETS)
print(f"[wind] preset={_CFG['__preset_name__']}: "
      f"E={_CFG['TAPE_YOUNGS']:.1e} Pa, ν={_CFG['TAPE_POISSON']}, "
      f"ρ={_CFG['TAPE_MASS_DENSITY']} kg/m³, t={_CFG['TAPE_THICKNESS']*1e3:.3f} mm, "
      f"d_hat={_CFG['D_HAT']*1e3:.3f} mm, LAYER={_CFG['LAYER_THICKNESS']*1e3:.3f} mm, "
      f"hub R={_CFG['HUB_R_OUTER']*1e3:.1f} mm × W={_CFG['TAPE_WIDTH']*1e3:.1f} mm, "
      f"N_TURNS={_CFG['N_TURNS']}, "
      f"adh: Cn={_CFG['ADH_CN']:.0e} Ct={_CFG['ADH_CT']:.0e} "
      f"r={_CFG['ADH_BONDING_RATE']} β₀={_CFG['ADH_INITIAL_BETA']}")

# ---- hub geometry (from preset) ----
HUB_R_OUTER       = _CFG["HUB_R_OUTER"]
HUB_R_INNER       = _CFG["HUB_R_INNER"]
HUB_HEIGHT        = _CFG["HUB_HEIGHT"]
N_RADIAL          = 48

# ---- tape geometry / discretization ----
# TAPE_NZ comes from the preset (override with --set TAPE_NZ=N).
# TAPE_NX defaults to the square-cell rule (TAPE_LENGTH / TAPE_DS); if
# the preset / CLI supplies TAPE_NX explicitly, that wins.
N_TURNS           = int(_CFG["N_TURNS"])
TAPE_LENGTH       = _CFG["TAPE_LENGTH"]
TAPE_WIDTH        = _CFG["TAPE_WIDTH"]
TAPE_NZ           = int(_CFG.get("TAPE_NZ", 10))
TAPE_DS           = TAPE_WIDTH / TAPE_NZ
_nx_override      = _CFG.get("TAPE_NX")
TAPE_NX           = (int(_nx_override) if _nx_override
                     else max(1, int(round(TAPE_LENGTH / TAPE_DS))))

# ---- IPC contact band (from preset; see preset block for the
# 2t < LAYER < 2t+d_hat constraint chain — LAYER is used by procedural
# demo, not here; wind grows layers implicitly via 2·TAPE_THICKNESS) ----
D_HAT             = _CFG["D_HAT"]
TAPE_THICKNESS    = _CFG["TAPE_THICKNESS"]

# ---- tape material (from preset) ----
TAPE_YOUNGS       = _CFG["TAPE_YOUNGS"]
TAPE_POISSON      = _CFG["TAPE_POISSON"]
TAPE_MASS_DENSITY = _CFG["TAPE_MASS_DENSITY"]
# Shell bending stiffness (DiscreteShellBending κ, Pa) — electrical-tape value;
# see BENDING_STIFFNESS_DEFAULT in tape_asset_lib. `--set BENDING_STIFFNESS=…`.
BENDING_STIFFNESS = float(_CFG.get("BENDING_STIFFNESS", L.BENDING_STIFFNESS_DEFAULT))

# ---- hub material ----
HUB_KAPPA         = 1.0e8
HUB_MASS_DENSITY  = 1000.0

# ---- adhesion (from preset; override with `--set ADH_INITIAL_BETA=...`) ----
# Wind defaults: initial_beta=0, bonding_rate=5. New PT contact pairs start
# unbonded (no β-jump at first contact → smoother sim, no line-search
# blips), and β grows quickly under SPC-driven compression to reach
# ~1 in a few frames of sustained pressure.
ADH_CN            = _CFG["ADH_CN"]
ADH_CT            = _CFG["ADH_CT"]
ADH_W             = _CFG["ADH_W"]
ADH_ETA           = _CFG["ADH_ETA"]
ADH_BONDING_RATE  = _CFG["ADH_BONDING_RATE"]
ADH_INITIAL_BETA  = _CFG["ADH_INITIAL_BETA"]

# ---- SPC ----
# SPC strength rate. Force per vertex ≈ strength * mass * (x - aim).
# Lower values make the pinning "softer" — less inertial shock on the
# inner anchor when the wind starts, less Hessian-conditioning trouble
# during the release ramp. 1e3 still firmly pins for typical tape mass
# (ρ * V_vertex ≈ 1e-6 kg → effective k ≈ 1 N/m per vertex; aggregated
# across the anchor strip it's plenty stiff). Override per-run with
# `--set SPC_STRENGTH=...` if the wind asset slips during winding.
SPC_STRENGTH      = float(_CFG.get("SPC_STRENGTH", 1000))
# Free tail strategy: every vertex whose arc-length from the anchor is
# more than (L_wound + BUFFER_LENGTH) is pinned along the tangent line
# from the wrap-off point. The unpinned BUFFER_LENGTH worth of tape
# right at the wrap-off region is the free lead-in that bends onto the
# hub. Default 0.005 m — a very short free lead-in (≈1/26 turn for this
# hub) that keeps the about-to-wind section short and the wrap tight
# against the hub, removing the loose/wrinkled dangling lead-in that a
# long buffer (the old 0.04 ≈ 0.3 turn) produced. Verified stable on the
# 2-turn e5e7 preset; for many-turn / different-hub winds a slightly
# larger value may be safer — tune with `--set BUFFER_LENGTH=…`.
BUFFER_LENGTH     = float(_CFG.get("BUFFER_LENGTH", 0.005))
# Anchor strip: the first ANCHOR_ROWS mesh rows (at i=0..ANCHOR_ROWS-1)
# are locked at their rest pose for the ENTIRE simulation — mimics a
# tape with a permanent glue-tab. Specified in *cell count* rather than
# meters so the topology footprint stays consistent across NZ values
# (varying NZ changes dy, but the anchor still pins the same number of
# rows). 3 rows ≈ 5–6 mm at NZ=10; the exact metric depends on
# TAPE_DS = TAPE_WIDTH / TAPE_NZ.
ANCHOR_ROWS       = 3

# ---- timeline (dt=0.01) ----
# Free-end tangential speed = r · ω. At start, r ≈ TAPE_LENGTH, so a
# fast schedule launches the free end at ~5 m/s — way past d_hat per
# step. WIND_FRAMES = 4500 (~9 s/turn) keeps it well under 1 m/s and
# gives IPC time to engage each new contact pair (3× slower than the
# previous default; the slower pull also lets β grow more under each
# new layer's compression before the next layer rolls on).
PREHEAT_FRAMES    = 30
WIND_FRAMES       = 1500
# Long settle1 lets β grow under the wound-end SPC pressure — anchor
# rows + tail-tangent rows still pinned, middle wound zone in
# compression. Goal: most PT pairs reach β > 0.9 before we let go.
SETTLE1_FRAMES    = int(_CFG.get("SETTLE1_FRAMES", 1000))
# RELEASE: gradually unpin the SPC, walking from the inner anchor
# row outward through the tail. Each pinned row's release time is
# linear in its index in the ordered list [0, 1, …, ANCHOR_ROWS-1,
# i_pin_start, …, NX]. At the end of this phase no SPC remains; the
# tape is held together purely by RCC adhesion + IPC barrier. Slower
# release (larger RELEASE_FRAMES) gives β time to ramp up before the
# spring force vanishes; too fast → elastic snap-back, tape balls up.
RELEASE_FRAMES    = int(_CFG.get("RELEASE_FRAMES", 2000))
# Final relaxation with no SPC. Asset is meant to be saved at the
# END of this phase — captures the truly self-sustaining wound state
# (which is what downstream demos load). Extend if max|v| at the
# saved-frame snapshot is still > a few mm/s.
SETTLE2_FRAMES    = int(_CFG.get("SETTLE2_FRAMES", 1000))
TOTAL_FRAMES      = (PREHEAT_FRAMES + WIND_FRAMES
                     + SETTLE1_FRAMES + RELEASE_FRAMES + SETTLE2_FRAMES)
THETA_END         = 2.0 * np.pi * N_TURNS

# Frame markers (used by the animator + UI to dispatch).
_WIND_END     = PREHEAT_FRAMES + WIND_FRAMES
_SETTLE1_END  = _WIND_END + SETTLE1_FRAMES
_RELEASE_END  = _SETTLE1_END + RELEASE_FRAMES
# (SETTLE2 ends at TOTAL_FRAMES.)

# Anchor radius: inside the tape-hub IPC active band so adhesion fires.
R_ANCHOR          = HUB_R_OUTER + TAPE_THICKNESS + 0.5 * D_HAT

ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_winding")
if _CFG.get("__list_assets__"):
    L.list_assets(ASSET_DIR)
ASSET_OUT_PATH = L.resolve_asset_path(
    ASSET_DIR, _CFG.get("__asset_arg__"), _CFG["__preset_name__"])
print(f"[wind] save target: {ASSET_OUT_PATH}")


# ----------------------------------------------------------------------
# Trajectory math
# ----------------------------------------------------------------------
def R_eff(theta: float) -> float:
    """Effective wrap radius — grows by 2·TAPE_THICKNESS per turn so each
    new layer settles on top of the previous one."""
    return HUB_R_OUTER + TAPE_THICKNESS * (1.0 + theta / np.pi)


def L_wound(theta: float) -> float:
    """Total tape arc length wound on hub after angle θ (analytical
    integral of R_eff)."""
    return HUB_R_OUTER * theta + TAPE_THICKNESS * (theta + theta * theta / (2.0 * np.pi))


def free_end_center(theta: float) -> np.ndarray:
    """Free-end center position (xy plane, z=0) at wind angle θ.

    Wrap-off at angle θ on hub: P_wrap = R_eff(θ)·(cosθ, sinθ).
    Tangent (CCW): T = (-sinθ, cosθ).
    Free tail length: L_tape - L_wound(θ), extended along T.
    """
    r = R_eff(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    p_wrap = np.array([r * cos_t, r * sin_t, 0.0])
    tangent = np.array([-sin_t, cos_t, 0.0])
    l_free = max(TAPE_LENGTH - L_wound(theta), 0.0)
    return p_wrap + l_free * tangent


def theta_at_frame(f: int) -> float:
    if f < PREHEAT_FRAMES:
        return 0.0
    if f < PREHEAT_FRAMES + WIND_FRAMES:
        t = (f - PREHEAT_FRAMES) / WIND_FRAMES
        return t * THETA_END
    return THETA_END


def _save_asset_impl(sim, state):
    """Module-level save-asset routine. Used by both the interactive
    UI button (via the `save_asset` closure inside `run_demo`) and
    the headless `RECORD_DIR` path (which can't reach in-function
    closures). Mutates `state["saved"] = True` on success."""
    os.makedirs(os.path.dirname(ASSET_OUT_PATH), exist_ok=True)
    hub_geo = sim["hub_geo"].geometry()
    tape_geo = sim["tape_geo"].geometry()
    hub_T = np.array(view(hub_geo.transforms()), copy=True).reshape(4, 4)
    tape_pos = np.array(view(tape_geo.positions()), copy=True).reshape(-1, 3)
    # Live tape velocity via FE state accessor (the SC's velocity
    # attribute is NOT written back by FEM.write_scene, so we have
    # to ask FE directly).
    tape_vel = None
    fe_acc = sim["world"].features().find(FiniteElementStateAccessorFeature)
    if fe_acc is not None:
        state_geo = fe_acc.create_geometry()
        state_geo.vertices().create("position", np.zeros(3, dtype=np.float64))
        state_geo.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        fe_acc.copy_to(state_geo)
        tape_vel = np.array(view(state_geo.vertices().find("velocity")),
                            copy=True).reshape(-1, 3)
    # Full cfg dump (geometry + IPC + material + adhesion + provenance)
    # plus derived TAPE_NX. Asset becomes self-describing.
    params = L.filter_cfg_for_save(_CFG)
    params["TAPE_NX"] = TAPE_NX
    # Persist whether bonded-PT was enabled at wind time so a downstream
    # load (drop/unwind) auto-enables bonded — no `--set BONDED=1` needed.
    params["BONDED"] = 1 if state.get("bonded") else 0
    # Same for distance-lock mode: a tape wound in distance-lock mode replays
    # in it (the saved β snapshot carries sentinel 1.0 per lock, no soft beta).
    _dlock = state.get("bonded") and L.cfg_flag(_CFG, "DISTANCE_LOCK", default=False)
    params["DISTANCE_LOCK"] = 1 if _dlock else 0
    if _dlock:
        params["DISTANCE_LOCK_RATIO"] = float(_CFG.get("DISTANCE_LOCK_RATIO", 0.5))
    # Snapshot the RCC adhesion β state for downstream demos.
    pair_state = None
    if state["adhesion_on"]:
        acc = sim["world"].features().find(RCCAdhesionStateAccessorFeature)
        if acc is not None:
            keys, betas = acc.dump_pt_state()
            pair_state = (keys, betas)
            print(f"  β snapshot: n={len(betas)} pairs, "
                  f"mean={betas.mean():.3f}, "
                  f"frac>0.9={(betas > 0.9).mean():.2%}" if len(betas)
                  else "  β snapshot: 0 pairs (no PT contacts saved)")
    # Snapshot the bonded-PT lock state (topology + β per lock) so a downstream
    # load can re-lock the bonds directly (seed_locks) instead of re-forming
    # them over the first step. Only present when bonded was enabled.
    locked_pairs = None
    if state.get("bonded"):
        bpt = sim["world"].features().find(RCCBondedPTStateAccessorFeature)
        if bpt is not None:
            topos, lbetas = bpt.dump_locked_pairs()
            if len(lbetas) > 0:
                locked_pairs = (topos, lbetas)
                print(f"  bonded-lock snapshot: n={len(lbetas)} locks "
                      f"(mean β={lbetas.mean():.3f})")
    L.save_tape_asset(ASSET_OUT_PATH, hub_T, tape_pos, params,
                      pair_state=pair_state, tape_velocity=tape_vel,
                      locked_pairs=locked_pairs)
    if tape_vel is not None:
        vmax = float(np.linalg.norm(tape_vel, axis=1).max())
        vmean = float(np.linalg.norm(tape_vel, axis=1).mean())
        print(f"  velocity snapshot: max={vmax:.3e} m/s, mean={vmean:.3e} m/s")
    print(f"saved wound-tape asset → {ASSET_OUT_PATH}")
    state["saved"] = True


def _record_save_asset_after_run(sim, state):
    """Thin wrapper called by the headless record path."""
    _save_asset_impl(sim, state)


def phase_at(f: int) -> str:
    if f < PREHEAT_FRAMES:        return "preheat"
    if f < _WIND_END:             return "wind"
    if f < _SETTLE1_END:          return "settle1"
    if f < _RELEASE_END:          return "release"
    return "settle2"


# ----------------------------------------------------------------------
# Build the tangent-oriented tape mesh directly (avoids modifying
# `make_flat_tape`'s xz-plane output in-place).
# ----------------------------------------------------------------------
def _make_tangent_tape(R_anchor, length, width, NX, NZ):
    """Flat shell with long axis +y, width centered on z=0, normal +x.

    Vertices: (R_anchor, i·dy, j·dz - W/2) for i ∈ [0,NX], j ∈ [0,NZ].
    Triangle winding chosen so face normals = +x (radially outward
    from the hub center). Caller does `set_sticky_side(tape, -1)` to
    make the sticky face point inward (toward hub).
    """
    dy = length / NX
    dz = width / NZ
    n_width = NZ + 1
    verts = np.empty(((NX + 1) * n_width, 3), dtype=np.float64)
    for i in range(NX + 1):
        for j in range(n_width):
            verts[i * n_width + j] = (R_anchor, i * dy, j * dz - 0.5 * width)

    def vid(i, j):
        return i * n_width + j

    tris = []
    # Winding analysis (face in yz plane at constant x):
    #   edges (i,j)→(i+1,j) = +y;  (i,j)→(i,j+1) = +z;  cross +y×+z = +x ✓
    # tri 1: (v00, v10, v11). v10-v00 = +y·dy. v11-v00 = (+y·dy, +z·dz).
    #   cross(+y·dy, +y·dy + +z·dz) = +y×+y + +y×+z·dz = 0 + +x·dz ⇒ +x.
    # tri 2: (v00, v11, v01). v11-v00 = (+y·dy, +z·dz). v01-v00 = +z·dz.
    #   cross((+y·dy, +z·dz), +z·dz) = +y×+z·dy·dz + 0 ⇒ +x.
    for i in range(NX):
        for j in range(NZ):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    sc = trimesh(verts, np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc, verts


def vid(i, j):
    return i * (TAPE_NZ + 1) + j


# ----------------------------------------------------------------------
# Build scene
# ----------------------------------------------------------------------
def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


def _hub_axis_to_z() -> Matrix4x4:
    """R_x(π/2) so the hub's natural +y axis maps to +z."""
    Rx = np.eye(4, dtype=np.float64)
    Rx[1, 1] =  0.0;  Rx[1, 2] = -1.0
    Rx[2, 1] =  1.0;  Rx[2, 2] =  0.0
    return _mat4_to_uipc(Rx)


def build_demo(adhesion_on: bool = True,
               bonded: bool = False,
               beta_lock_threshold: float = 0.9,
               kappa: float = 1.0e8):
    # Default Warn; bump to e.g. info/debug via `--set LOG_LEVEL=info`.
    L.apply_log_level(_CFG, default="warn")

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]   # no gravity — pure winding
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = D_HAT
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
    # SOFT RCC normal-adhesion energy-minimum offset (independent of bonding).
    # Leave at the engine default (0.5 = band center) unless overridden. `--set
    # RCC_ADHESION_NORMAL_OFFSET_COEFF=0` -> legacy min at d=0; =1 -> band edge.
    _off = _CFG.get("RCC_ADHESION_NORMAL_OFFSET_COEFF")
    if _off is not None:
        config["rcc_adhesion_normal_offset_coeff"] = float(_off)
    if bonded:
        # RCC bonded-PT acceleration: stable high-beta face-interior tape
        # contacts are replaced by a stiff ABD virtual tet (point-plane).
        config["rcc_bonded_pt_enabled"] = 1
        # CCD on locked pairs during winding. Default -1 = auto (skip CCD for
        # locked pairs when bonded). `--set SKIP_CCD=0` keeps CCD on locked
        # pairs, `=1` forces skip. Set the same way for the matching drop so a
        # config is wound and dropped under identical CCD behaviour.
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
        # Phase 7 distance-locked bonding: `--set DISTANCE_LOCK=1` disables the
        # soft adhesion energy entirely (no beta) and locks by the end-of-step
        # distance band d < xi + c*d_hat instead (c = `--set
        # DISTANCE_LOCK_RATIO=...`, default 0.5, clamped [0,2] by the engine;
        # c > 1 locks beyond the contact band — the trajectory filters
        # extend their candidate range to xi + c*d_hat to match).
        # The preset's Cn/Ct are ignored in this mode (the engine warns once);
        # release gates are unchanged. Recommend LOCK_FACE_INTERIOR_ONLY=1 —
        # without beta's multi-step integration, edge/corner VTs would
        # mass-lock skewed sliver tets on first contact.
        if L.cfg_flag(_CFG, "DISTANCE_LOCK", default=False):
            config["rcc_bonded_pt_distance_lock"] = 1
            config["rcc_bonded_pt_distance_lock_ratio"] = float(
                _CFG.get("DISTANCE_LOCK_RATIO", 0.5))
        # Release thresholds left at their disabled defaults (1e30): once a
        # tape contact bonds it stays bonded (the wound tape does not peel).
        # The preset's RCC_RELEASE_FORCE is NOT applied here on purpose — it is
        # consumed downstream by the drop/unwind demos (the replay/peel), where
        # the bond should release like the non-bonded adhesion would. It is
        # still saved into the asset as a record of the intended release load.
    # User-facing solver knobs (e.g. `--set LIN_TOL_RATE=1e-5
    # --set NEWTON_VELOCITY_TOL=0.005`) get translated into the
    # libuipc nested config here, AFTER the demo's own defaults so
    # CLI always wins. See SOLVER_KEYS in tape_asset_lib.
    L.apply_solver_overrides(config, _CFG)
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    dsb = DiscreteShellBending()
    spc = SoftPositionConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    hub_contact  = tabular.default_element()
    tape_contact = tabular.create("tape")
    tabular.insert(tape_contact, tape_contact, 0.5, 1.0e9)
    tabular.insert(tape_contact, hub_contact,  0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
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

    # ---- hub (fixed in place) ----
    hub_sc = L.make_ring_hub(
        R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
        height=HUB_HEIGHT, n_radial=N_RADIAL,
        center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    hub_contact.apply_to(hub_sc)
    # Rotate the ABD instance so the hub axis aligns with +z (wrap plane = xy).
    view(hub_sc.transforms())[0] = _hub_axis_to_z()
    # Lock the hub via builtin.is_fixed (no animator, no STC needed).
    view(hub_sc.instances().find(builtin.is_fixed))[0] = 1

    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- tape (tangent-oriented straight strip) ----
    tape_sc, tape_verts = _make_tangent_tape(
        R_anchor=R_ANCHOR,
        length=TAPE_LENGTH, width=TAPE_WIDTH,
        NX=TAPE_NX, NZ=TAPE_NZ)
    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    nhs.apply_to(tape_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    # Shell bending resistance (a single triangle shell has none without this).
    dsb.apply_to(tape_sc, BENDING_STIFFNESS)
    tape_contact.apply_to(tape_sc)
    spc.apply_to(tape_sc, SPC_STRENGTH)
    if adhesion_on:
        # tape normal = +x (outward). sticky = -1 → sticky face is -x = inward
        # (toward hub center). Matches the wound-tape v3 gate semantics.
        RCCAdhesive.set_sticky_side(tape_sc, -1)

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(tape_sc)

    # Snapshot rest positions for animator z-lookup.
    rest_positions = tape_verts.copy()
    # Per-row arc length from anchor (i=0): row i is at arc length i·dy
    # along the (initially straight) tape.
    dy_tape = TAPE_LENGTH / TAPE_NX
    arc_at_row = np.arange(TAPE_NX + 1) * dy_tape
    # Anchor strip: rows in [0, ANCHOR_ROWS) — always pinned at rest pose.
    # ANCHOR_ROWS is a module-level cell count (NZ-independent topology).

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim  = view(geo.vertices().find(builtin.aim_position))
        # Per-vertex SPC strength multiplier. SoftPositionConstraint
        # creates this attribute at apply_to time (`strength_ratio`,
        # set to SPC_STRENGTH for every vertex). The cuda backend
        # reads it per-vertex per-step (force ∝ strength · mass), so
        # writing here lets us smoothly fade SPC strength to zero
        # during the RELEASE phase without unpinning any row.
        strength = view(geo.vertices().find("strength_ratio"))
        is_c[:] = 0

        # SETTLE2: all SPCs released — tape is held purely by RCC
        # adhesion + IPC barrier. This is the state captured by
        # save_asset to give downstream demos a self-consistent
        # equilibrium (no SPC-dependence to recover from on load).
        if f >= _RELEASE_END:
            return

        # Tangent geometry from the current θ (constant after wind).
        theta = theta_at_frame(f)
        l_wound = L_wound(theta)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r = R_eff(theta)
        p_wrap = np.array([r * cos_t, r * sin_t])
        tangent = np.array([-sin_t, cos_t])

        # i_pin_start: first row beyond the active-bend window.
        i_pin_start = int(np.ceil((l_wound + BUFFER_LENGTH) / dy_tape))
        i_pin_start = max(ANCHOR_ROWS, min(i_pin_start, TAPE_NX + 1))

        # RELEASE-phase strength multiplier: linearly decays from 1
        # (full SPC_STRENGTH) at _SETTLE1_END to 0 at _RELEASE_END.
        # Before SETTLE1 ends, multiplier stays at 1 (constraints
        # are at full strength during preheat / wind / settle1).
        if f >= _SETTLE1_END:
            t = (f - _SETTLE1_END) / max(RELEASE_FRAMES, 1)
            t = float(np.clip(t, 0.0, 1.0))
            spc_mul = 1.0 - t
        else:
            spc_mul = 1.0
        spc_now = SPC_STRENGTH * spc_mul

        # Helpers: pin one row at either the rest pose (anchor side)
        # or along the tangent line (tail side). Both write the
        # current strength so the release ramp is uniform across all
        # constrained rows.
        def _pin_anchor_row(i: int):
            for j in range(TAPE_NZ + 1):
                k = vid(i, j)
                is_c[k] = 1
                aim[k] = rest_positions[k].reshape(3, 1)
                strength[k] = spc_now

        def _pin_tail_row(i: int):
            s = arc_at_row[i] - l_wound
            x, y = p_wrap + s * tangent
            for j in range(TAPE_NZ + 1):
                k = vid(i, j)
                is_c[k] = 1
                aim[k] = np.array(
                    [x, y, rest_positions[k, 2]],
                    dtype=np.float64,
                ).reshape(3, 1)
                strength[k] = spc_now

        # Pin all originally-constrained rows every frame in the
        # release window. The release schedule is now a global
        # strength ramp (spc_mul above), not per-row removal — every
        # constrained vertex loses strength at the same rate so the
        # system has time to settle into a self-supporting state
        # without redistributing stress through suddenly-freed rows.
        for i in range(ANCHOR_ROWS):
            _pin_anchor_row(i)
        if i_pin_start <= TAPE_NX:
            for i in range(i_pin_start, TAPE_NX + 1):
                _pin_tail_row(i)

    scene.animator().insert(tape_obj, animate_tape)

    world.init(scene)
    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "hub_geo": hub_geo,
        "tape_geo": tape_geo,
    }


# ----------------------------------------------------------------------
# Polyscope viewer
# ----------------------------------------------------------------------
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


def run_demo():
    # Bonded-PT acceleration enabled with beta lock threshold 0.9 (pass
    # `--set BONDED=0` to disable for an A/B comparison).
    state = {"adhesion_on": True, "saved": False,
             "bonded": L.cfg_flag(_CFG, "BONDED", default=True),
             "beta_lock": float(_CFG.get("RCC_BETA_LOCK_THRESHOLD", 0.9))}
    sim = build_demo(state["adhesion_on"], bonded=state["bonded"],
                     beta_lock_threshold=state["beta_lock"])

    # Trajectory sanity check — print free-end positions at sampled angles.
    print(f"tape mesh: NX={TAPE_NX}, NZ={TAPE_NZ}  "
          f"(dy={TAPE_LENGTH/TAPE_NX*1000:.3f} mm, dz={TAPE_WIDTH/TAPE_NZ*1000:.3f} mm)")
    print(f"L_tape={TAPE_LENGTH:.4f} m, "
          f"L_wound(θ_end)={L_wound(THETA_END):.4f} m, "
          f"slack={TAPE_LENGTH - L_wound(THETA_END):.4f} m")
    for k in range(N_TURNS + 1):
        th = k * 2.0 * np.pi
        p = free_end_center(th)
        print(f"  θ={th:6.3f} ({k} turns): R_eff={R_eff(th):.5f}, "
              f"L_free={max(TAPE_LENGTH - L_wound(th), 0.0):.4f}, "
              f"free_end=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    # Headless record mode (EGL). Must branch BEFORE any `ps.init()`
    # because the interactive init tries the display backend and
    # crashes on a headless box. See drop demo for full notes.
    # Auto-saves the asset at the end (the wind sim's whole point).
    record_dir = _CFG.get("RECORD_DIR")
    if record_dir:
        every_n = int(_CFG.get("RECORD_EVERY", 10))
        zoom    = float(_CFG.get("RECORD_ZOOM", 5.0))
        # Periodic max|v| sample — pulled from the FE state accessor.
        # Helps tune RELEASE_FRAMES / SETTLE2_FRAMES by showing the
        # velocity trajectory through release + final settle.
        vel_log_every = int(_CFG.get("VEL_LOG_EVERY", 100))
        fe_acc_for_log = sim["world"].features().find(FiniteElementStateAccessorFeature)
        state_geo_for_log = None
        if fe_acc_for_log is not None:
            state_geo_for_log = fe_acc_for_log.create_geometry()
            state_geo_for_log.vertices().create("position", np.zeros(3, dtype=np.float64))
            state_geo_for_log.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        def _on_progress(f, tot):
            print(f"[record] frame {f}/{tot} ({f/tot*100:.1f}%)  "
                  f"Phase: {phase_at(f - 1)}")
            if (state_geo_for_log is not None
                    and (f % vel_log_every == 0 or f == tot)):
                fe_acc_for_log.copy_to(state_geo_for_log)
                v = np.array(view(state_geo_for_log.vertices().find("velocity")),
                             copy=True).reshape(-1, 3)
                vn = np.linalg.norm(v, axis=1)
                print(f"  [v-trace] frame {f:>5d}  "
                      f"max|v|={vn.max():.3e} m/s  "
                      f"mean|v|={vn.mean():.3e} m/s  "
                      f"phase={phase_at(f - 1)}")
        L.record_demo_to_pngs(sim=sim, total_frames=TOTAL_FRAMES,
                              output_dir=record_dir, every_n=every_n,
                              up_dir="z_up", mesh_name="wound_tape",
                              zoom=zoom, on_progress=_on_progress)
        # Inline the save_asset equivalent (the regular closure is
        # defined later in this function and not in scope here).
        _record_save_asset_after_run(sim, state)
        return

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "wound_tape",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(0.3)

    ui = {"run": False, "show_bonds": True}
    bonds_net = [None]

    def update_bonds():
        # Draw the bonded virtual tets (red curve network). Re-register when
        # the lock count changes (bonds form / release); otherwise just move
        # the nodes. Removed entirely when the toggle is off or no locks.
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
            ps.remove_surface_mesh("wound_tape")
            mesh = ps.register_surface_mesh("wound_tape", v, t)
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

    # save_asset is the closure used by the interactive UI button.
    # It delegates to the module-level _save_asset_impl so the
    # headless record path can reuse the same logic.
    def save_asset():
        _save_asset_impl(sim, state)

    def reset():
        nonlocal sim
        sim = build_demo(state["adhesion_on"], bonded=state["bonded"],
                         beta_lock_threshold=state["beta_lock"])
        update_visual()
        state["saved"] = False

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
        if psim.Button("reset"):
            reset()
        psim.SameLine()
        if psim.Button("save asset"):
            save_asset()
        if state["bonded"]:
            psim.SameLine()
            if psim.Button(f"bonds: {'ON' if ui['show_bonds'] else 'OFF'}"):
                ui["show_bonds"] = not ui["show_bonds"]
                update_bonds()

        if ui["run"]:
            step_once()

        f = min(sim["world"].frame(), TOTAL_FRAMES)
        theta = theta_at_frame(f)
        psim.Separator()
        psim.Text(f"Frame: {f} / {TOTAL_FRAMES}    Phase: {phase_at(f)}")
        psim.Text(f"θ = {theta:.2f} rad   ({theta/(2*np.pi):.2f} turns)")
        psim.Text(f"L_wound = {L_wound(theta):.3f} / {TAPE_LENGTH} m")
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
        # Release-phase progress: SPC strength multiplier on every
        # constrained vertex. 100% during settle1 → 0% at release end.
        if _SETTLE1_END <= f < _RELEASE_END:
            rel_t = (f - _SETTLE1_END) / max(RELEASE_FRAMES, 1)
            psim.Text(f"SPC strength: {(1.0 - rel_t)*100:.1f}% "
                      f"(ramping uniformly to 0)")
        elif f >= _RELEASE_END:
            psim.Text("SPC strength: 0% — adhesion + barrier only")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        if state["saved"]:
            psim.Text(f"asset @ {ASSET_OUT_PATH}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
