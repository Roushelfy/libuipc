"""
RCC oriented (single-sided) adhesion demo — "tape pick-and-lift".

Layout (sandwich):
- Bottom ABD cube on the ground (fixed in place).
- A NeoHookeanShell cloth ~30% larger than the cube footprint, suspended just
  above the bottom cube's top face. The cloth's **upper face** (+n̂) is marked
  as sticky via `RCCAdhesive.set_sticky_side(cloth, +1)`. The lower face has
  no adhesion.
- Top ABD cube above the cloth, animated with a SoftTransformConstraint:
  presses down onto the cloth, holds, then lifts straight up.

Expected behaviour with v3 oriented adhesion ON (default):
- During press: the cloth's sticky-up face contacts the top cube → adhesion
  bonds. The cloth's non-sticky-down face contacts the bottom cube → no bond.
- During lift: cloth follows the top cube upward. Bottom cube stays put.
- Toggling adhesion OFF: cloth falls back to the bottom cube as the top cube
  lifts away (no bond at all).
- Toggling `oriented` OFF (double-sided fallback): cloth bonds to BOTH cubes
  → the lift drags the bottom cube up too, or the cloth stretches between them.

Run:
    python python/examples/rcc_adhesive_oriented_cloth_demo.py

UI:
    run / pause    — toggle playback
    step           — one frame
    adhesion       — ON / OFF (rebuilds scene)
    oriented       — v3 single-sided ON / v2 double-sided fallback (rebuilds scene)
    reset          — rebuild scene
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

try:
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
    from uipc.geometry import (
        SimplicialComplex,
        SimplicialComplexIO,
        trimesh,
        label_surface,
        label_triangle_orient,
        mesh_partition,
        ground,
    )
    from uipc.constitution import (
        AffineBodyConstitution,
        NeoHookeanShell,
        SoftTransformConstraint,
        ElasticModuli2D,
        RCCAdhesive,
    )
    from uipc.core import RCCBondedPTStateAccessorFeature
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ---- ground / cubes ----
# Everything is anchored to the ground plane at y = GROUND_Y so the top-cube
# trajectory does not depend on where the bottom cube happens to settle.
GROUND_Y            = 0.0
CUBE_SCALE          = 0.3
# Bottom cube starts a hair above the ground; gravity drops it onto the
# plane (the plane is a half-plane contact, no adhesion).
BOTTOM_INITIAL_CENTER_Y = GROUND_Y + 0.5 * CUBE_SCALE + 0.01
# Once it settles its top face sits at ~GROUND_Y + CUBE_SCALE. The top-cube
# press target is defined relative to that ground-anchored top, with a small
# air gap so the cloth (thickness 0) is squeezed between.
PRESS_GAP            = 0.01
TOP_CONTACT_CENTER_Y = GROUND_Y + CUBE_SCALE + PRESS_GAP + 0.5 * CUBE_SCALE
TOP_INITIAL_CENTER_Y = GROUND_Y + CUBE_SCALE + 0.45 + 0.5 * CUBE_SCALE   # ~0.9
TOP_FINAL_CENTER_Y   = TOP_INITIAL_CENTER_Y + 0.30                       # lifted target

# ---- cloth ----
# Cloth is ~33% larger than the cube footprint. Its sticky face is the
# +n̂ (upper) face — set via RCCAdhesive.set_sticky_side(cloth, +1) in
# build_demo. The v3 gate ensures the cloth's lower face never bonds to
# the bottom cube, even with overhang draping around the cube edges.
CLOTH_N    = 14
CLOTH_SIZE = 0.4
CLOTH_SPACING = CLOTH_SIZE / CLOTH_N
# Cloth starts slightly above the bottom cube's settled top face; gravity
# drops it the rest of the way. Adhesion gate on the cloth's lower face
# fails (it's the non-sticky side), so no bond forms with the bottom cube.
CLOTH_Y    = GROUND_Y + CUBE_SCALE + 0.012

# ---- timeline (frames at dt=0.01) ----
PRESS_START  = 5
CONTACT_AT   = 60
HOLD_UNTIL   = 100
LIFT_UNTIL   = 200
TOTAL_FRAMES = 280


def smooth_lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def picker_y(frame: int) -> tuple[float, str]:
    if frame < PRESS_START:
        return TOP_INITIAL_CENTER_Y, "init"
    if frame < CONTACT_AT:
        t = (frame - PRESS_START) / (CONTACT_AT - PRESS_START)
        return smooth_lerp(TOP_INITIAL_CENTER_Y, TOP_CONTACT_CENTER_Y, t), "press"
    if frame < HOLD_UNTIL:
        return TOP_CONTACT_CENTER_Y, "hold"
    if frame < LIFT_UNTIL:
        t = (frame - HOLD_UNTIL) / (LIFT_UNTIL - HOLD_UNTIL)
        return smooth_lerp(TOP_CONTACT_CENTER_Y, TOP_FINAL_CENTER_Y, t), "lift"
    return TOP_FINAL_CENTER_Y, "settled"


def cube_transform(center_y: float) -> Matrix4x4:
    m = Matrix4x4.Identity()
    m[0:3, 3] = np.array([0.0, center_y, 0.0], dtype=np.float64)
    return m


def vid(i: int, j: int) -> int:
    return i * (CLOTH_N + 1) + j


def make_cloth_mesh(base_y: float) -> SimplicialComplex:
    verts = []
    tris = []
    for i in range(CLOTH_N + 1):
        for j in range(CLOTH_N + 1):
            x = i * CLOTH_SPACING - 0.5 * CLOTH_SIZE
            z = j * CLOTH_SPACING - 0.5 * CLOTH_SIZE
            verts.append([x, base_y, z])
    # Triangles wound so that the cross-product (B-A)×(C-A) points in +y, i.e.
    # the cloth's outward normal n̂ points UP. With set_sticky_side(+1) this
    # makes the upper face sticky.
    for i in range(CLOTH_N):
        for j in range(CLOTH_N):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v01, v11])
            tris.append([v00, v11, v10])
    sc = trimesh(np.asarray(verts, dtype=np.float64),
                 np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc


def build_demo(
    adhesion_on: bool,
    oriented: bool,
    bonded: bool = False,
    skip_ccd: bool = False,
    beta_lock_threshold: float = 0.9,
    kappa: float = 1.0e8,
    release_strain: float = 1.0e30,
    release_gap: float = 1.0e30,
    release_slip: float = 1.0e30,
    youngs: float = 1.0e7,
    cloth_cn: float = 1.0e4,
):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = 0.005
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
    if bonded:
        config["rcc_bonded_pt_enabled"] = 1
        config["rcc_bonded_pt_skip_ccd"] = 1 if skip_ccd else 0
        config["rcc_bonded_pt_beta_lock_threshold"] = beta_lock_threshold
        config["rcc_bonded_pt_energy_model"] = "abd_ortho"
        config["rcc_bonded_pt_kappa"] = kappa
        config["rcc_bonded_pt_release_strain"] = release_strain
        config["rcc_bonded_pt_release_gap"] = release_gap
        config["rcc_bonded_pt_release_slip"] = release_slip
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    stc = SoftTransformConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    cube_contact  = tabular.default_element()
    cloth_contact = tabular.create("cloth")
    # cloth-cube is the only adhesive pair; (cube,cube) and (cloth,cloth)
    # have no adhesion — only barrier + friction.
    tabular.insert(cloth_contact, cloth_contact, 0.0, 1.0e9)
    tabular.insert(cloth_contact, cube_contact,  0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            tabular,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )
        adhesive.set(
            tabular, cloth_contact, cube_contact,
            Cn=cloth_cn, Ct=1.0e5, W=1.0, eta=2.0,
            bonding_rate=1.0, p0=0.0, initial_beta=1.0,
            enabled=True,
        )
        adhesive.set(
            tabular, cloth_contact, cloth_contact,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )

    # pre-scale a unit cube down to CUBE_SCALE
    pre = Matrix4x4.Identity()
    pre[0, 0] = CUBE_SCALE
    pre[1, 1] = CUBE_SCALE
    pre[2, 2] = CUBE_SCALE
    io = SimplicialComplexIO(pre)
    cube = io.read(f"{AssetDir.tetmesh_path()}/cube.msh")
    label_surface(cube)
    label_triangle_orient(cube)
    # 2 instances of the same SC: idx 0 = bottom (free; falls onto ground),
    # idx 1 = top (animated picker driven by SoftTransformConstraint).
    cube.instances().resize(2)
    abd.apply_to(cube, 1.0e8)
    cube_contact.apply_to(cube)
    stc.apply_to(cube, np.array([1.0e8, 0.0], dtype=np.float64))

    trans = view(cube.transforms())
    trans[0] = cube_transform(BOTTOM_INITIAL_CENTER_Y)
    trans[1] = cube_transform(TOP_INITIAL_CENTER_Y)

    cube_obj = scene.objects().create("cubes")
    cube_slot = cube_obj.geometries().create(cube)[0]

    # ground at y = 0
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    # ---- cloth ----
    moduli = ElasticModuli2D.youngs_poisson(youngs, 0.4)
    cloth_obj = scene.objects().create("cloth")
    cloth = make_cloth_mesh(CLOTH_Y)
    nhs.apply_to(cloth, moduli)
    cloth_contact.apply_to(cloth)
    if adhesion_on and oriented:
        # Sticky face = +n̂ (top side) because our trimesh winding gives +n̂ in +y.
        RCCAdhesive.set_sticky_side(cloth, +1)
    cloth_obj.geometries().create(cloth)

    def animate_cubes(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform  = view(geo.instances().find(builtin.aim_transform))
        # bottom cube is free (no SPC); only the top cube is animated.
        is_constrained[0] = 0
        is_constrained[1] = 1
        y, _ = picker_y(max(info.frame() - 1, 0))
        aim_transform[1] = cube_transform(y)
        # aim_transform[0] is unread when is_constrained[0] == 0, but keep a
        # sane value to avoid stale data.
        aim_transform[0] = cube_transform(BOTTOM_INITIAL_CENTER_Y)

    scene.animator().insert(cube_obj, animate_cubes)
    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
    }


def _bonded_bonds(sim):
    """(locked_count, (nodes, edges)) for the bonded virtual tets, or (count, None)."""
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
    # Pass --bonded to enable bonded-PT acceleration + pre-CCD skip and draw the
    # bonded virtual tets (red). Default is the plain (non-bonded) demo.
    state = {"adhesion_on": True, "oriented": True, "bonded": ("--bonded" in sys.argv)}
    sim = build_demo(state["adhesion_on"], state["oriented"],
                     bonded=state["bonded"], skip_ccd=state["bonded"])

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "rcc_oriented_cloth",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(0.5)
    ui = {"run": False}

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        verts = merged.positions().view().reshape(-1, 3)
        tris  = merged.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != verts.shape[0]:
            ps.remove_surface_mesh("rcc_oriented_cloth")
            mesh = ps.register_surface_mesh("rcc_oriented_cloth", verts, tris)
            mesh.set_edge_width(0.5)
        else:
            mesh.update_vertex_positions(verts)

        if ps.has_curve_network("oriented_cloth_bonds"):
            ps.remove_curve_network("oriented_cloth_bonds")
        if state["bonded"]:
            _, bonds = _bonded_bonds(sim)
            if bonds is not None:
                net = ps.register_curve_network("oriented_cloth_bonds", bonds[0], bonds[1])
                net.set_radius(0.003)
                net.set_color((1.0, 0.15, 0.1))

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
        sim = build_demo(state["adhesion_on"], state["oriented"],
                         bonded=state["bonded"], skip_ccd=state["bonded"])
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
        if psim.Button(f"oriented: {'ON' if state['oriented'] else 'OFF'}"):
            state["oriented"] = not state["oriented"]
        psim.SameLine()
        if psim.Button(f"bonded: {'ON' if state['bonded'] else 'OFF'}"):
            state["bonded"] = not state["bonded"]
        psim.SameLine()
        if psim.Button("reset"):
            reset()

        if ui["run"]:
            step_once()

        frame = min(sim["world"].frame(), TOTAL_FRAMES)
        target_y, phase = picker_y(frame)

        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase}")
        psim.Text(f"Picker (top cube) target Y: {target_y:+.3f}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        psim.Text(f"Oriented (v3): {'ENABLED' if state['oriented'] else 'DISABLED (double-sided)'}")
        if state["bonded"]:
            locked, _ = _bonded_bonds(sim)
            psim.Text(f"Bonded PT (skip_ccd): ON  locked={locked} (red bonds)")
        else:
            psim.Text("Bonded PT: OFF  (toggle `bonded` + `reset`, or pass --bonded)")
        if frame >= TOTAL_FRAMES:
            psim.Text("Sim done. Hit `reset` to rebuild (toggle modes first if desired).")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
