"""
Unwind / peel demo (uses Option C's saved asset).

Loads `wound_tape.npz` (saved by `rcc_adhesive_tape_winding_demo.py`),
rebuilds the scene with the hub fixed and the tape in its wound state,
re-establishes the inter-layer / tape-hub adhesion bonds (β=1 at frame 0),
and animates the outermost free end of the tape outward so the tape
gradually peels off.

The tape's REST pose is set to the original straight strip (not the
wound geometry), so debonded sections spring back to straight under
NeoHookeanShell elasticity — that's the "tear off" visual.

Adhesion uses v3 single-sided semantics (sticky face = -n̂ = inward
toward the hub center), same as the asset was built with.

Run:
    python python/examples/rcc_adhesive_tape_unwind_demo.py
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
from uipc.core import RCCAdhesionStateAccessorFeature
from uipc.constitution import (
    AffineBodyConstitution,
    NeoHookeanShell,
    SoftPositionConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ---- CLI / preset resolution ----
# Uses UNWIND_PRESETS (material + adhesion + SPC). Geometry comes from
# the loaded asset's params, NOT from the preset. Pick an asset with
# `--asset NAME` (the wind preset name you used) and a peel physics
# config with `--preset`. Examples:
#   python ... --preset medium-bond --asset temflex175
#   python ... --preset rigid       --asset temflex175 --set ADH_CN=5e2
_CFG = L.parse_tape_cli(L.UNWIND_PRESETS)
print(f"[unwind] preset={_CFG['__preset_name__']}: "
      f"E={_CFG['TAPE_YOUNGS']:.1e} Pa, ν={_CFG['TAPE_POISSON']}, "
      f"ρ={_CFG['TAPE_MASS_DENSITY']} kg/m³, "
      f"Cn={_CFG['ADH_CN']:.1e}, Ct={_CFG['ADH_CT']:.1e}, W={_CFG['ADH_W']}, "
      f"η={_CFG['ADH_ETA']}, SPC={_CFG['SPC_STRENGTH']:.1e}")

# ---- IPC contact band — preset value is only a fallback.
# `build_demo()` overrides D_HAT and TAPE_THICKNESS from the loaded asset
# unless the user passed `--set D_HAT=...` / `--set TAPE_THICKNESS=...`
# on the CLI. The asset's IPC band must match the geometry that the wind
# step laid out (e.g. wind with TAPE_NZ=20 saves a smaller D_HAT — unwind
# has to use the same so the layers don't pop on frame 0).
D_HAT             = _CFG["D_HAT"]
TAPE_THICKNESS    = _CFG["TAPE_THICKNESS"]

# ---- materials (from preset) ----
TAPE_YOUNGS       = _CFG["TAPE_YOUNGS"]
TAPE_POISSON      = _CFG["TAPE_POISSON"]
TAPE_MASS_DENSITY = _CFG["TAPE_MASS_DENSITY"]
HUB_KAPPA         = 1.0e8
HUB_MASS_DENSITY  = 1000.0

# ---- adhesion (from preset) ----
ADH_CN            = _CFG["ADH_CN"]
ADH_CT            = _CFG["ADH_CT"]
ADH_W             = _CFG["ADH_W"]
ADH_ETA           = _CFG["ADH_ETA"]
ADH_BONDING_RATE  = _CFG["ADH_BONDING_RATE"]
ADH_INITIAL_BETA  = _CFG["ADH_INITIAL_BETA"]

# ---- SPC strength (from preset). Formula: stiffness = SPC_STRENGTH ×
# vertex_mass. 1e9 → ~300 N/m per vertex (dominates moderate adhesion).
SPC_STRENGTH      = _CFG["SPC_STRENGTH"]

# ---- pull animation ----
# Free end (i = NX, last row) is dragged perpendicular to its local
# tangent. PULL_DISTANCE is the total path length traversed over
# PULL_FRAMES at constant speed. The pull direction is sampled from
# the live tape geometry every PERP_UPDATE_INTERVAL frames and held
# constant in between (avoids noisy frame-by-frame oscillation).
PULL_DISTANCE         = 0.5      # 50 cm
PULL_FRAMES           = 1500     # 15 s @ dt=0.01 — slow, lets β evolve
PERP_UPDATE_INTERVAL  = 100      # frames between pull-direction refreshes
SETTLE_FRAMES         = 60
# Initial settle phase: no SPC at all. Mirrors wind's `settle2`
# (also fully unconstrained). If wind ended in a self-sustaining
# wound state, this phase should show near-zero displacement —
# acts as a smoke test for "was the saved asset actually at
# equilibrium?". After it, anchor + free-end SPCs engage at the
# *live* positions (not the asset's wound positions) so the
# transition is bump-free even if there was a tiny drift.
INITIAL_SETTLE_FRAMES = 60
HOLD_FRAMES           = 20       # short hold before pulling, lets β rise
TOTAL_FRAMES      = INITIAL_SETTLE_FRAMES + HOLD_FRAMES + PULL_FRAMES + SETTLE_FRAMES

# Frame markers (used by animator + UI to dispatch).
_INIT_END     = INITIAL_SETTLE_FRAMES
_HOLD_END     = _INIT_END + HOLD_FRAMES
_PULL_END     = _HOLD_END + PULL_FRAMES

ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_winding")
if _CFG.get("__list_assets__"):
    L.list_assets(ASSET_DIR)
ASSET_IN_PATH = L.resolve_asset_path(
    ASSET_DIR, _CFG.get("__asset_arg__"), _CFG["__preset_name__"])
print(f"[unwind] load source: {ASSET_IN_PATH}")


# ----------------------------------------------------------------------
# Helpers — identical topology to `rcc_adhesive_tape_winding_demo.py`
# ----------------------------------------------------------------------
def _make_tape_topology_tris(NX, NZ):
    """Triangle indices for a (NX+1)×(NZ+1) grid laid out as `vid(i,j) =
    i·(NZ+1) + j`. Winding matches the wind demo — face normals point
    +x in the demo's frame, so `set_sticky_side(tape, -1)` makes the
    sticky face point toward the hub center."""
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


def _straight_rest_positions(R_anchor, length, width, NX, NZ):
    """Same straight-tangent layout the wind demo started from. Used as
    REST geometry so unwound sections relax to straight."""
    dy = length / NX
    dz = width / NZ
    n_width = NZ + 1
    verts = np.empty(((NX + 1) * n_width, 3), dtype=np.float64)
    for i in range(NX + 1):
        for j in range(n_width):
            verts[i * n_width + j] = (R_anchor, i * dy, j * dz - 0.5 * width)
    return verts


def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


# ----------------------------------------------------------------------
def build_demo(adhesion_on: bool = True):
    Logger.set_level(Logger.Level.Warn)

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
    TAPE_POISSON      = _resolve_and_log("TAPE_POISSON",      "TAPE_POISSON", ".3f")
    TAPE_MASS_DENSITY = _resolve_and_log("TAPE_MASS_DENSITY", "TAPE_MASS_DENSITY", ".1f")
    ADH_CN            = _resolve_and_log("ADH_CN",            "ADH_CN")
    ADH_CT            = _resolve_and_log("ADH_CT",            "ADH_CT")
    ADH_W             = _resolve_and_log("ADH_W",             "ADH_W",   ".3f")
    ADH_ETA           = _resolve_and_log("ADH_ETA",           "ADH_ETA", ".3f")
    ADH_BONDING_RATE  = _resolve_and_log("ADH_BONDING_RATE",  "ADH_BONDING_RATE", ".3f")
    ADH_INITIAL_BETA  = _resolve_and_log("ADH_INITIAL_BETA",  "ADH_INITIAL_BETA", ".3f")

    # Anchor radius identical to the wind demo's so positions match.
    R_ANCHOR = HUB_R_OUTER + TAPE_THICKNESS + 0.5 * D_HAT

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = D_HAT
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
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

    # ---- hub (fixed, transform from asset) ----
    hub_sc = L.make_ring_hub(
        R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
        height=HUB_HEIGHT, n_radial=48,
        center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    hub_contact.apply_to(hub_sc)
    view(hub_sc.transforms())[0] = _mat4_to_uipc(hub_T)
    view(hub_sc.instances().find(builtin.is_fixed))[0] = 1
    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- tape: current = wound (from asset); rest = straight ----
    tris = _make_tape_topology_tris(TAPE_NX, TAPE_NZ)
    rest_positions = _straight_rest_positions(
        R_anchor=R_ANCHOR, length=TAPE_LENGTH, width=TAPE_WIDTH,
        NX=TAPE_NX, NZ=TAPE_NZ)
    current_sc = _make_tape_sc(tape_pos, tris)
    rest_sc    = _make_tape_sc(rest_positions, tris)

    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    # `nhs.apply_to` is what (a) tags constitution_uid, (b) sets thickness,
    # mass_density, and (c) runs compute_vertex_volume → adds `volume`
    # vertex attribute. The CUDA FEM backend reads `volume` from the REST
    # geometry to compute mass (finite_element_method.cu:679), so the
    # rest_sc needs apply_to as well — without it the engine aborts at
    # world.init() on the missing slot.
    nhs.apply_to(current_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    nhs.apply_to(rest_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    # Contact / SPC / sticky_side are dynamic-state attributes — only
    # needed on the current geometry.
    tape_contact.apply_to(current_sc)
    spc.apply_to(current_sc, SPC_STRENGTH)
    if adhesion_on:
        # sticky = inward toward hub center (same gate as wind demo).
        RCCAdhesive.set_sticky_side(current_sc, -1)

    # Seed FEM velocity from asset (None on legacy assets → v=0 default).
    # Unwind shares wind's coordinate frame so velocities go in raw.
    # Must happen BEFORE geometries().create so the FEM ingests them.
    # libuipc stores Vector3 attributes as (N, 3, 1) column-vector
    # tensors, so we reshape the (N, 3) numpy array to match.
    if tape_vel is not None and tape_vel.shape == tape_pos.shape:
        existing = current_sc.vertices().find("velocity")
        if existing is None:
            current_sc.vertices().create("velocity", np.zeros(3, dtype=np.float64))
        vel_view = view(current_sc.vertices().find("velocity"))
        vel_view[:] = tape_vel.reshape(-1, 3, 1)
        vmax = float(np.linalg.norm(tape_vel, axis=1).max())
        if vmax > 0:
            print(f"[unwind] seeded tape velocity: max={vmax:.3e} m/s")

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(current_sc, rest_sc)

    # ---- pin layout ----
    def vid(i, j): return i * (TAPE_NZ + 1) + j
    anchor_ids = [vid(0, j) for j in range(TAPE_NZ + 1)]   # innermost row (glued)
    free_ids   = [vid(TAPE_NX, j) for j in range(TAPE_NZ + 1)]   # outermost row (pulled)

    # Snapshot wound-state positions for the free-end pull trajectory.
    wound_free_pos = tape_pos[free_ids].copy()             # (NZ+1, 3)
    wound_anchor_pos = tape_pos[anchor_ids].copy()         # (NZ+1, 3)
    # Per-vertex z offsets at free end — kept constant during the pull.
    free_z = wound_free_pos[:, 2].copy()
    prev_row_ids = [vid(TAPE_NX - 1, j) for j in range(TAPE_NZ + 1)]

    # Pull is INCREMENTAL: each frame we advance the SPC target by
    # PULL_STEP along the *live* perpendicular-to-tangent direction
    # at the free end. This way the pull stays normal to the tape as
    # the peeled portion rotates.
    PULL_STEP = PULL_DISTANCE / PULL_FRAMES
    state = {
        # Set lazily at the INITIAL_SETTLE → HOLD edge from live
        # geometry, not from the asset's wound positions. This lets
        # any tiny drift during initial-settle propagate forward
        # without a yank.
        "target_xy":      None,
        "last_perp":      np.array([1.0, 0.0]),
        # SPC targets for anchor + free-end. Both snapshotted at the
        # initial-settle boundary so they reflect the actually-
        # equilibrated state (not the asset's saved positions).
        "live_anchor_pos": None,
        "live_free_pos":   None,
        # Free-end SPC toggle. When False, animate_tape skips ALL
        # free-end is_constrained/aim_position writes — the tape's
        # outer row is unconstrained and only governed by physics
        # (elasticity + adhesion + contact). Toggled live from the UI.
        "pull_enabled":     True,
        # OFF→ON edge marker. animate_tape consumes this to snap
        # target_xy back onto the live free-end center (so the pull
        # resumes from wherever the tape currently is, not from where
        # the target was stale-frozen at).
        "pull_needs_reset": False,
    }

    def _compute_live_perp(geo) -> np.ndarray:
        """Perp to local tangent at free end, pointing away from hub axis.
        Falls back to last good perp on degenerate tangent."""
        pos = np.asarray(view(geo.positions())).reshape(-1, 3)
        free_c = pos[free_ids][:, :2].mean(axis=0)
        prev_c = pos[prev_row_ids][:, :2].mean(axis=0)
        tan = free_c - prev_c
        tn = float(np.linalg.norm(tan))
        if tn < 1e-9:
            return state["last_perp"]
        tan /= tn
        perp = np.array([-tan[1], tan[0]])
        r = float(np.linalg.norm(free_c))
        if r > 1e-9:
            radial = free_c / r
            if np.dot(perp, radial) < 0.0:
                perp = -perp
        return perp

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim  = view(geo.vertices().find(builtin.aim_position))
        is_c[:] = 0

        # ───── INITIAL_SETTLE ─────
        # No SPC anywhere — mirrors wind's settle2 end state.
        # If the asset is at true equilibrium, this phase shows
        # near-zero displacement. If not (β too low, IPC drift),
        # the system relaxes here before SPC engages.
        if f < _INIT_END:
            return

        # First time we leave initial-settle: snapshot the LIVE
        # anchor + free-end positions and use those as SPC targets.
        # This avoids a position jump that would happen if we
        # blindly used the asset's `wound_anchor_pos` after the
        # system has already drifted slightly.
        if state["live_anchor_pos"] is None:
            pos = np.asarray(view(geo.positions())).reshape(-1, 3)
            state["live_anchor_pos"] = pos[anchor_ids].copy()
            state["live_free_pos"]   = pos[free_ids].copy()
            state["target_xy"]       = state["live_free_pos"][:, :2].mean(axis=0).copy()

        # ───── HOLD / PULL / SETTLE ─────
        # Anchor: locked at the live snapshot from the boundary,
        # not at the asset's `wound_anchor_pos` (which may have
        # drifted by a sub-mm during initial settle).
        for jj, k in enumerate(anchor_ids):
            is_c[k] = 1
            aim[k] = state["live_anchor_pos"][jj].reshape(3, 1)

        # Free-end SPC is gated by the UI toggle. When disabled,
        # leave free_ids unconstrained and freeze target advancement
        # so re-enabling resumes from the live geometry.
        if not state["pull_enabled"]:
            return

        # OFF→ON edge: snap target back onto the current free-end
        # center so the SPC doesn't yank the tape from wherever the
        # tape drifted to back to a stale target.
        if state["pull_needs_reset"]:
            pos = np.asarray(view(geo.positions())).reshape(-1, 3)
            state["target_xy"] = pos[free_ids][:, :2].mean(axis=0).copy()
            state["pull_needs_reset"] = False

        # Advance the free-end target along the peel direction. The
        # direction is re-sampled from the live geometry every
        # PERP_UPDATE_INTERVAL frames and held constant in between.
        # PULL phase runs from _HOLD_END to _PULL_END (frame indices
        # are global, the +1 shift via `f = info.frame() - 1` is
        # already applied above).
        if _HOLD_END <= f < _PULL_END:
            pull_idx = f - _HOLD_END
            if pull_idx % PERP_UPDATE_INTERVAL == 0:
                state["last_perp"] = _compute_live_perp(geo)
            state["target_xy"] = state["target_xy"] + PULL_STEP * state["last_perp"]

        # SPC at current accumulated target, preserving each vertex's
        # original z so the strip stays flat across the width.
        tx, ty = state["target_xy"]
        for jj, k in enumerate(free_ids):
            is_c[k] = 1
            aim[k] = np.array([tx, ty, free_z[jj]], dtype=np.float64).reshape(3, 1)

    scene.animator().insert(tape_obj, animate_tape)

    world.init(scene)

    # Restore β from the asset if it was saved with one. Must happen AFTER
    # world.init() (the backend feature is only registered then) and BEFORE
    # the first world.advance() (so the very next step's Phase B reads the
    # loaded prev-state via match_or_init instead of init_all_new).
    if adhesion_on and pair_state is not None:
        keys, betas = pair_state
        acc = world.features().find(RCCAdhesionStateAccessorFeature)
        if acc is not None and len(betas) > 0:
            acc.load_pt_state(keys, betas)
            print(f"[unwind] restored β: n={len(betas)} pairs, "
                  f"mean={betas.mean():.3f}, frac>0.9={(betas > 0.9).mean():.2%}")
        else:
            if acc is None:
                print("[unwind] WARNING: RCCAdhesionStateAccessorFeature not found "
                      "— β was not restored.")

    return {
        "engine": engine, "world": world, "scene": scene,
        "scene_io": SceneIO(scene),
        "hub_geo": hub_geo, "tape_geo": tape_geo,
        "params": params,
        # Effective IPC numerics actually used by the running sim
        # (may differ from the module-level preset values when the
        # asset's saved IPC overrides them — see the "← asset" log
        # above). `save_asset` reuses these to keep the snapshot
        # consistent with what was simulated.
        "D_HAT_eff":          D_HAT,
        "TAPE_THICKNESS_eff": TAPE_THICKNESS,
        # Animator state dict — the UI's "pull" toggle button flips
        # state["pull_enabled"] and arms state["pull_needs_reset"]
        # to make the OFF→ON edge resume from the live tape pose.
        "anim_state":         state,
    }


def phase_at(f: int) -> str:
    if f < _INIT_END:  return "initial-settle"
    if f < _HOLD_END:  return "hold"
    if f < _PULL_END:  return "pull"
    return "settle"


def run_demo():
    state = {"adhesion_on": True}
    sim = build_demo(state["adhesion_on"])

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "unwind_tape",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(0.3)

    ui = {"run": False}

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        v = merged.positions().view().reshape(-1, 3)
        t = merged.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != v.shape[0]:
            ps.remove_surface_mesh("unwind_tape")
            mesh = ps.register_surface_mesh("unwind_tape", v, t)
            mesh.set_edge_width(0.3)
        else:
            mesh.update_vertex_positions(v)

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
        sim = build_demo(state["adhesion_on"])
        update_visual()

    def save_asset():
        """Snapshot the paused-frame state as a fresh .npz.

        Captures the current hub transform, current tape positions, the
        full params dict that was loaded with the original asset (so the
        downstream loader gets the same geometry / IPC / material), and
        — when adhesion is on — the live per-pair β snapshot via
        RCCAdhesionStateAccessorFeature. Saved to
        ASSET_DIR/<input_stem>_paused_F<frame>.npz so it doesn't clobber
        the source asset.
        """
        sim["world"].retrieve()
        hub_geo  = sim["hub_geo"].geometry()
        tape_geo = sim["tape_geo"].geometry()
        hub_T    = np.array(view(hub_geo.transforms()), copy=True).reshape(4, 4)
        tape_pos = np.array(view(tape_geo.positions()), copy=True).reshape(-1, 3)

        # Merge order (later wins):
        #   1) source asset's params  — keeps wind-time geometry +
        #      provenance fields that the unwind cfg doesn't carry
        #      (HUB_*, TAPE_LENGTH, TAPE_NX, etc.).
        #   2) the full unwind cfg    — every preset/--set value
        #      actually driving this sim (material, adhesion, IPC,
        #      and `__preset_name__` = unwind preset).
        #   3) effective IPC numerics — what the sim REALLY used,
        #      which can differ from the unwind cfg when the asset
        #      itself overrode them (build_demo's "← asset" logic).
        #   4) provenance fields chaining back to the source asset.
        source_preset = sim["params"].get("__preset_name__", "?")
        out_params = dict(sim["params"])
        out_params.update(L.filter_cfg_for_save(_CFG))
        out_params["TAPE_THICKNESS"]     = sim["TAPE_THICKNESS_eff"]
        out_params["D_HAT"]              = sim["D_HAT_eff"]
        out_params["__source_preset__"]  = source_preset
        out_params["__unwind_preset__"]  = _CFG["__preset_name__"]
        out_params["__unwind_source__"]  = os.path.basename(ASSET_IN_PATH)
        out_params["__unwind_frame__"]   = int(sim["world"].frame())

        pair_state = None
        if state["adhesion_on"]:
            acc = sim["world"].features().find(RCCAdhesionStateAccessorFeature)
            if acc is not None:
                keys, betas = acc.dump_pt_state()
                pair_state = (keys, betas)

        # Output path: <input_stem>_paused_F<frame>.npz in the asset dir.
        stem = os.path.splitext(os.path.basename(ASSET_IN_PATH))[0]
        f    = int(sim["world"].frame())
        out_path = os.path.join(ASSET_DIR, f"{stem}_paused_F{f}.npz")
        os.makedirs(ASSET_DIR, exist_ok=True)
        L.save_tape_asset(out_path, hub_T, tape_pos, out_params,
                          pair_state=pair_state)
        print(f"saved paused-state asset → {out_path}")
        if pair_state is not None:
            keys, betas = pair_state
            if len(betas):
                print(f"  β snapshot: n={len(betas)} pairs, "
                      f"mean={betas.mean():.3f}, "
                      f"frac>0.9={(betas > 0.9).mean():.2%}")
            else:
                print("  β snapshot: 0 PT pairs (no active contacts)")
        else:
            print("  (adhesion off; β not saved)")

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
            was_on = anim["pull_enabled"]
            anim["pull_enabled"] = not was_on
            # OFF→ON: arm a one-shot reset so the SPC target snaps
            # onto wherever the free end currently is, instead of
            # yanking back to the stale target_xy.
            if not was_on:
                anim["pull_needs_reset"] = True
        psim.SameLine()
        if psim.Button("reset"):
            reset()
        psim.SameLine()
        if psim.Button("save asset"):
            save_asset()

        if ui["run"]:
            step_once()

        f = min(sim["world"].frame(), TOTAL_FRAMES)
        psim.Separator()
        psim.Text(f"Frame: {f} / {TOTAL_FRAMES}    Phase: {phase_at(f)}")

        # show free-end pull progress (PULL phase runs _HOLD_END..PULL_END)
        f1 = max(f - 1, 0)
        if f1 < _HOLD_END:
            p = 0.0
        elif f1 < _PULL_END:
            p = (f1 - _HOLD_END) / PULL_FRAMES
        else:
            p = 1.0
        pull_state = "ENABLED" if sim["anim_state"]["pull_enabled"] else "DISABLED (free end)"
        psim.Text(f"Pull: {p*100:.1f}%  ({p*PULL_DISTANCE*1000:.0f} mm of {PULL_DISTANCE*1000:.0f} mm)  SPC: {pull_state}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
