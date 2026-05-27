"""
Procedural wound-tape asset demo (Option B).

Builds a pre-wound electrical-tape roll geometrically:
  - ABD ring "hub" via `make_ring_hub`.
  - NeoHookeanShell tape whose vertex positions lie along an Archimedean
    spiral wrapping the hub (via `make_wound_tape`).
  - RCC adhesion sticky-side = -1 (sticky face on -n̂ = inward toward hub
    center): outer turn's sticky face touches the next inner turn's
    non-sticky back, so RCC's OR-semantics keeps adjacent turns bonded.

A short "relax" sim warms up β across all the inter-layer PT pairs; then
the polyscope viewer lets you confirm the roll is stable under gravity
(dropping onto a ground plane).

Saves the wound state at the end as an .npz asset suitable for the
unwind demo.

Run:
    python python/examples/rcc_adhesive_tape_procedural_demo.py
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
from uipc.geometry import ground
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


# ---- asset parameters ----
HUB_R_OUTER       = 0.025      # 25 mm — small enough to look like a tape spool
HUB_R_INNER       = 0.023      # 23 mm
HUB_HEIGHT        = 0.04       # 40 mm — slightly taller than tape width
HUB_CENTER_Y      = 0.20       # spawn ~20 cm above the ground

N_TURNS           = 2
TAPE_WIDTH        = 0.04

# ---- IPC contact thickness band ----
# IPC's barrier active band for a pair (P,T) is on the surface distance d:
#     thickness_P + thickness_T  <  d  <  thickness_P + thickness_T + d_hat
# (Below the lower bound = geometric penetration → world is_valid() = false.)
# Constraints for this demo's three pair types:
#     tape-tape:   2*TAPE_THICKNESS  <  LAYER_THICKNESS  <  2*TAPE_THICKNESS + D_HAT
#     tape-hub:    TAPE_THICKNESS    <  start_gap        <  TAPE_THICKNESS + D_HAT
#     tape-ground: TAPE_THICKNESS    <  gap-to-ground    <  TAPE_THICKNESS + D_HAT
#
# Performance: D_HAT is the IPC barrier's width. Making it TOO small makes
# the barrier extremely stiff and forces Newton + CCD into many sub-iters
# per frame. Rule of thumb: D_HAT ≈ 5–10% of the smallest body's
# characteristic length. For a 25 mm hub, D_HAT = 1–2 mm is the sweet
# spot. To get the *visual* of a thin tape, lower LAYER_THICKNESS (it
# only has to be inside the band), not D_HAT.
D_HAT             = 0.001       # 1 mm — IPC band; do not shrink without lowering dt
TAPE_THICKNESS    = 0.0001      # 0.1 mm — per-vertex shell thickness
LAYER_THICKNESS   = 0.0007      # 0.7 mm: midpoint of (2t=0.2mm, 2t+D_HAT=1.2mm)
NZ                = 6
DS                = 0.005

# ---- tape material (NeoHookeanShell) ----
TAPE_YOUNGS       = 1.0e7      # Pa
TAPE_POISSON      = 0.4
TAPE_MASS_DENSITY = 2.0e2      # kg/m³

# ---- hub material (ABD) ----
HUB_KAPPA         = 1.0e8      # ABD bulk modulus
HUB_MASS_DENSITY  = 1000.0     # kg/m³

# ---- RCC adhesion (applied identically to tape-tape and tape-hub pairs) ----
ADH_CN            = 1.0e4
ADH_CT            = 1.0e5
ADH_W             = 1.0
ADH_ETA           = 2.0
ADH_BONDING_RATE  = 1.0
ADH_INITIAL_BETA  = 1.0

# ---- sim ----
RELAX_FRAMES      = 60         # let β settle, let roll fall onto the ground
TOTAL_FRAMES      = 200

ASSET_OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_procedural", "wound_tape.npz")


def build_demo(adhesion_on: bool = True):
    Logger.set_level(Logger.Level.Warn)

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

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    cube_contact  = tabular.default_element()
    tape_contact  = tabular.create("tape")
    tabular.insert(tape_contact, tape_contact, 0.5, 1.0e9)
    tabular.insert(tape_contact, cube_contact,  0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            tabular,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )
        # Inter-tape (layer-vs-layer) adhesion: this is the bond that holds
        # the roll together.
        adhesive.set(
            tabular, tape_contact, tape_contact,
            Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
            bonding_rate=ADH_BONDING_RATE, p0=0.0, initial_beta=ADH_INITIAL_BETA,
            enabled=True,
        )
        # Tape-hub adhesion: same parameters; bonds the innermost turn to
        # the hub. With sticky-inward the inner turn's sticky face touches
        # the hub's outer cylinder.
        adhesive.set(
            tabular, tape_contact, cube_contact,
            Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
            bonding_rate=ADH_BONDING_RATE, p0=0.0, initial_beta=ADH_INITIAL_BETA,
            enabled=True,
        )

    # ---- ground plane ----
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    # ---- hub ----
    hub_sc = L.make_ring_hub(
        R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
        height=HUB_HEIGHT, n_radial=48,
        center=(0.0, HUB_CENTER_Y, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    cube_contact.apply_to(hub_sc)
    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- tape (spiral-laid positions) ----
    tape_sc, tape_length = L.make_wound_tape(
        hub_R_outer=HUB_R_OUTER, n_turns=N_TURNS,
        tape_width=TAPE_WIDTH, layer_thickness=LAYER_THICKNESS,
        NZ=NZ, ds=DS,
        hub_center=(0.0, HUB_CENTER_Y, 0.0),
        tape_thickness=TAPE_THICKNESS, d_hat=D_HAT)
    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    nhs.apply_to(tape_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    tape_contact.apply_to(tape_sc)
    # if adhesion_on:
    #     # Sticky face = -n̂ → spiral-inward (toward hub center). Outer turn's
    #     # sticky face touches the next inner turn's non-sticky outer face.
    #     RCCAdhesive.set_sticky_side(tape_sc, -1)
    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(tape_sc)

    world.init(scene)
    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "tape_length": tape_length,
        "hub_geo": hub_geo,
        "tape_geo": tape_geo,
    }


def run_demo():
    state = {"adhesion_on": True, "ran_to_end": False}
    sim = build_demo(state["adhesion_on"])
    print(f"tape strip length: {sim['tape_length']:.4f} m, "
          f"{N_TURNS} turns × hub R_outer={HUB_R_OUTER} m.")

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "wound_tape",
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
            ps.remove_surface_mesh("wound_tape")
            mesh = ps.register_surface_mesh("wound_tape", v, t)
            mesh.set_edge_width(0.3)
        else:
            mesh.update_vertex_positions(v)

    def step_once():
        if sim["world"].frame() >= TOTAL_FRAMES:
            ui["run"] = False
            if not state["ran_to_end"]:
                save_asset()
                state["ran_to_end"] = True
            return
        sim["world"].advance()
        if not sim["world"].is_valid():
            ui["run"] = False
            return
        sim["world"].retrieve()
        update_visual()

    def save_asset():
        os.makedirs(os.path.dirname(ASSET_OUT_PATH), exist_ok=True)
        # Live geometry refs captured from build_demo. obj.geometries().create()
        # returns SimplicialComplexSlot wrappers; .geometry() unpacks to the SC.
        hub_geo = sim["hub_geo"].geometry()
        tape_geo = sim["tape_geo"].geometry()
        hub_T = np.array(view(hub_geo.transforms()), copy=True).reshape(4, 4)
        tape_pos = np.array(view(tape_geo.positions()), copy=True).reshape(-1, 3)
        params = dict(
            HUB_R_OUTER=HUB_R_OUTER, HUB_R_INNER=HUB_R_INNER,
            HUB_HEIGHT=HUB_HEIGHT, HUB_CENTER_Y=HUB_CENTER_Y,
            N_TURNS=N_TURNS, TAPE_WIDTH=TAPE_WIDTH,
            LAYER_THICKNESS=LAYER_THICKNESS, NZ=NZ, DS=DS,
        )
        L.save_tape_asset(ASSET_OUT_PATH, hub_T, tape_pos, params)
        print(f"saved wound-tape asset → {ASSET_OUT_PATH}")

    def reset():
        nonlocal sim
        sim = build_demo(state["adhesion_on"])
        update_visual()
        state["ran_to_end"] = False

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

        if ui["run"]:
            step_once()

        frame = min(sim["world"].frame(), TOTAL_FRAMES)
        phase = "relax" if frame < RELAX_FRAMES else "settled"
        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}    Phase: {phase}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        psim.Text(f"Tape length: {sim['tape_length']:.3f} m, turns: {N_TURNS}")
        psim.Text(f"Hub: R={HUB_R_OUTER*1000:.0f}–{HUB_R_OUTER + LAYER_THICKNESS*N_TURNS*1000:.0f} mm")
        if frame >= TOTAL_FRAMES and state["ran_to_end"]:
            psim.Text(f"Asset saved to {ASSET_OUT_PATH}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
