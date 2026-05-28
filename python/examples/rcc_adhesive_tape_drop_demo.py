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
  - no SPC anchor on the tape, no pull animation. The wound layers and
    the inner-most layer–to–hub bond are held together purely by RCC
    adhesion (β=1 at frame 0 per the preset).
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
    view,
    builtin,
)
from uipc.geometry import trimesh, label_surface, mesh_partition, ground
from uipc.constitution import (
    AffineBodyConstitution,
    NeoHookeanShell,
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

# ---- timeline ----
SETTLE_FRAMES     = 600       # 6 s @ dt=0.01 — long enough to see settle/roll

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

    saved_preset = params.get("__preset_name__", "(unknown)")
    print(f"loaded asset [preset={saved_preset}]: tape ({TAPE_NX+1}×{TAPE_NZ+1} verts), "
          f"hub R∈[{HUB_R_INNER},{HUB_R_OUTER}], L_tape={TAPE_LENGTH:.3f} m")

    # Asset's saved IPC params win unless user --set them on CLI (same
    # rule as the unwind demo).
    explicit = _CFG.get("__explicit__", set())
    asset_t = params.get("TAPE_THICKNESS", None)
    asset_dhat = params.get("D_HAT", None)
    D_HAT_eff = _CFG["D_HAT"]
    TAPE_THICKNESS_eff = _CFG["TAPE_THICKNESS"]
    if asset_dhat is not None and "D_HAT" not in explicit:
        D_HAT_eff = float(asset_dhat)
    if asset_t is not None and "TAPE_THICKNESS" not in explicit:
        TAPE_THICKNESS_eff = float(asset_t)
    if abs(D_HAT_eff - _CFG["D_HAT"]) > 1e-12:
        print(f"  D_HAT          ← asset {D_HAT_eff:.4e}  "
              f"(preset value {_CFG['D_HAT']:.4e} overridden)")
    if abs(TAPE_THICKNESS_eff - _CFG["TAPE_THICKNESS"]) > 1e-12:
        print(f"  TAPE_THICKNESS ← asset {TAPE_THICKNESS_eff:.4e}  "
              f"(preset value {_CFG['TAPE_THICKNESS']:.4e} overridden)")
    D_HAT = D_HAT_eff
    TAPE_THICKNESS = TAPE_THICKNESS_eff

    # ---- stand the roll on the ground (axis +y), lowest point inside
    # the IPC band so the barrier engages on frame 0
    GROUND_CLEARANCE = TAPE_THICKNESS + 0.5 * D_HAT
    tape_pos_lay, hub_T_lay = _stand_on_ground(
        tape_pos, hub_T, HUB_HEIGHT, GROUND_CLEARANCE)
    hub_bottom_y = float(hub_T_lay[1, 3]) - 0.5 * HUB_HEIGHT
    print(f"[drop] hub bottom y={hub_bottom_y*1e3:.3f} mm, "
          f"tape lowest y={tape_pos_lay[:,1].min()*1e3:.3f} mm "
          f"(IPC band offset = {GROUND_CLEARANCE*1e3:.3f} mm above ground)")

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
    nhs = NeoHookeanShell()

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

    # ---- tape: rest = wound = current (no internal elastic energy at
    # frame 0; pure gravity + adhesion demo). Different from unwind,
    # which sets rest = straight to drive the unrolling spring-back.
    tris = _make_tape_topology_tris(TAPE_NX, TAPE_NZ)
    current_sc = _make_tape_sc(tape_pos_lay, tris)
    rest_sc    = _make_tape_sc(tape_pos_lay, tris)

    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    nhs.apply_to(current_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    nhs.apply_to(rest_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    tape_contact.apply_to(current_sc)
    if adhesion_on:
        RCCAdhesive.set_sticky_side(current_sc, -1)

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(current_sc, rest_sc)

    # ---- ground at y=0 ----
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    world.init(scene)
    return {
        "engine": engine, "world": world, "scene": scene,
        "scene_io": SceneIO(scene),
        "hub_geo": hub_geo, "tape_geo": tape_geo,
        "params": params,
    }


def run_demo():
    state = {"adhesion_on": True}
    sim = build_demo(state["adhesion_on"])

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    # libuipc's `ground(0.0)` is an implicit half-plane — the contact
    # engine uses it but SceneIO won't return it through
    # `simplicial_surface()`, so we register a flat quad at y=0 just
    # for visualization. Size: 1 m × 1 m centered at origin (much
    # bigger than the roll's footprint).
    ground_quad_verts = np.array([
        [-0.5, 0.0, -0.5],
        [ 0.5, 0.0, -0.5],
        [ 0.5, 0.0,  0.5],
        [-0.5, 0.0,  0.5],
    ], dtype=np.float64)
    ground_quad_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
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

    ui = {"run": False}

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

    def step_once():
        if sim["world"].frame() >= SETTLE_FRAMES:
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

        f = min(sim["world"].frame(), SETTLE_FRAMES)
        psim.Separator()
        psim.Text(f"Frame: {f} / {SETTLE_FRAMES}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
