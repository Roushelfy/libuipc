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

# ---- IPC contact band (preset value is a fallback; build_demo prefers
# what's stored in the asset's params dict — see warning logic there) ----
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
HOLD_FRAMES           = 20       # short hold before pulling, lets β rise
TOTAL_FRAMES      = HOLD_FRAMES + PULL_FRAMES + SETTLE_FRAMES

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

    hub_T, tape_pos, params = L.load_tape_asset(ASSET_IN_PATH)
    HUB_R_OUTER = float(params["HUB_R_OUTER"])
    HUB_R_INNER = float(params["HUB_R_INNER"])
    HUB_HEIGHT  = float(params["HUB_HEIGHT"])
    TAPE_LENGTH = float(params["TAPE_LENGTH"])
    TAPE_WIDTH  = float(params["TAPE_WIDTH"])
    TAPE_NX     = int(params["TAPE_NX"])
    TAPE_NZ     = int(params["TAPE_NZ"])

    # Warn if the asset's saved IPC params don't match the current preset.
    # The active barrier band shifts otherwise → init penetration or
    # missed contacts.
    saved_t = params.get("TAPE_THICKNESS", None)
    saved_dhat = params.get("D_HAT", None)
    saved_preset = params.get("__preset_name__", "(unknown)")
    print(f"loaded asset [preset={saved_preset}]: tape ({TAPE_NX+1}×{TAPE_NZ+1} verts), "
          f"hub R∈[{HUB_R_INNER},{HUB_R_OUTER}], L_tape={TAPE_LENGTH:.3f} m")
    if saved_t is not None and abs(saved_t - TAPE_THICKNESS) > 1e-9:
        print(f"  WARNING: TAPE_THICKNESS mismatch — asset={saved_t}, "
              f"current={TAPE_THICKNESS}. IPC band will shift.")
    if saved_dhat is not None and abs(saved_dhat - D_HAT) > 1e-9:
        print(f"  WARNING: D_HAT mismatch — asset={saved_dhat}, current={D_HAT}.")

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
        "target_xy": wound_free_pos[:, :2].mean(axis=0).copy(),
        "last_perp": np.array([1.0, 0.0]),
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

        # Anchor: locked forever at the wound state.
        for jj, k in enumerate(anchor_ids):
            is_c[k] = 1
            aim[k] = wound_anchor_pos[jj].reshape(3, 1)

        # Advance the free-end target along the peel direction. The
        # direction is re-sampled from the live geometry every
        # PERP_UPDATE_INTERVAL frames and held constant in between.
        if HOLD_FRAMES <= f < HOLD_FRAMES + PULL_FRAMES:
            pull_idx = f - HOLD_FRAMES
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
    return {
        "engine": engine, "world": world, "scene": scene,
        "scene_io": SceneIO(scene),
        "hub_geo": hub_geo, "tape_geo": tape_geo,
        "params": params,
    }


def phase_at(f: int) -> str:
    if f < HOLD_FRAMES:
        return "hold"
    if f < HOLD_FRAMES + PULL_FRAMES:
        return "pull"
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

        if ui["run"]:
            step_once()

        f = min(sim["world"].frame(), TOTAL_FRAMES)
        psim.Separator()
        psim.Text(f"Frame: {f} / {TOTAL_FRAMES}    Phase: {phase_at(f)}")

        # show free-end pull progress
        f1 = max(f - 1, 0)
        if f1 < HOLD_FRAMES:
            p = 0.0
        elif f1 < HOLD_FRAMES + PULL_FRAMES:
            p = (f1 - HOLD_FRAMES) / PULL_FRAMES
        else:
            p = 1.0
        psim.Text(f"Pull: {p*100:.1f}%  ({p*PULL_DISTANCE*1000:.0f} mm of {PULL_DISTANCE*1000:.0f} mm)")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
