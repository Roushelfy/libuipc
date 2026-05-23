"""
RCC adhesion cloth-peel demo.

A NeoHookeanShell cloth patch is bonded to the top face of a large fixed ABD
cube, then one edge of the cloth is lifted to peel it off.

Sequence:
    Phase 1 (press): the cloth descends from y_apart onto the cube's top
                     face and stays there for a few frames to let the bond
                     form.
    Phase 2 (peel):  one edge of the cloth ("pull edge") is lifted upward
                     while the opposite edge ("anchor edge") stays clamped
                     against the cube.

With RCC adhesion ON the bond resists the peel — the cloth visibly bends
and stretches before snapping free. With adhesion OFF the cloth lifts away
cleanly.

Run:
    python python/examples/rcc_adhesive_cloth_peel_demo.py
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
    )
    from uipc.constitution import (
        AffineBodyConstitution,
        NeoHookeanShell,
        SoftPositionConstraint,
        ElasticModuli2D,
        RCCAdhesive,
    )
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ---- big cube ----
CUBE_SCALE   = 0.6          # cube edge length
CUBE_CENTER_Y = 0.30        # cube center -> top face at y = 0.60

# ---- cloth ----
N            = 14
CLOTH_SIZE   = 0.5
SPACING      = CLOTH_SIZE / N

# top face of the cube is at y = CUBE_CENTER_Y + CUBE_SCALE / 2
TOP_FACE_Y   = CUBE_CENTER_Y + 0.5 * CUBE_SCALE
Y_BONDED     = TOP_FACE_Y + 0.012   # just above the cube (inside IPC band)
Y_LIFTED     = TOP_FACE_Y + 0.45    # peeled-up height

# ---- timeline (frames at dt=0.01) ----
# The cloth starts at Y_BONDED (already in the IPC active band) so we skip
# the "press" phase entirely — v1 RCC adhesion has trouble with hundreds of
# pairs simultaneously activating at the band boundary, which made the
# implicit solve thrash. Starting bonded sidesteps that transition.
HOLD_UNTIL   = 30     # let the bond settle
PEEL_UNTIL   = 230    # 30..230: lift the pull edge
TOTAL_FRAMES = 280


def vid(i: int, j: int) -> int:
    return i * (N + 1) + j


def smooth_lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def make_cloth_mesh(base_y: float) -> SimplicialComplex:
    verts = []
    tris = []
    for i in range(N + 1):
        for j in range(N + 1):
            x = i * SPACING - 0.5 * CLOTH_SIZE
            z = j * SPACING - 0.5 * CLOTH_SIZE
            verts.append([x, base_y, z])
    for i in range(N):
        for j in range(N):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    sc = trimesh(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
    )
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc


def cube_transform(center_y: float) -> Matrix4x4:
    m = Matrix4x4.Identity()
    m[0:3, 3] = np.array([0.0, center_y, 0.0], dtype=np.float64)
    return m


def timeline(frame: int) -> tuple[float, float, str]:
    """Return (y_anchor, y_pull, phase) for the given frame."""
    if frame < HOLD_UNTIL:
        return Y_BONDED, Y_BONDED, "settle"
    if frame < PEEL_UNTIL:
        t = (frame - HOLD_UNTIL) / max(PEEL_UNTIL - HOLD_UNTIL - 1, 1)
        return Y_BONDED, smooth_lerp(Y_BONDED, Y_LIFTED, t), "peel"
    return Y_BONDED, Y_LIFTED, "lifted"


def build_demo(adhesion_on: bool):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    # v1 adhesion reach is bounded by the IPC active band; widen it slightly.
    config["contact"]["d_hat"] = 0.02
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    spc = SoftPositionConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.0, 1.0e6)
    cube_contact  = tabular.default_element()
    cloth_contact = tabular.create("cloth")
    # Explicitly create the two non-default rows we will configure adhesion on:
    #   - (cloth, cloth): used to disable cloth-self adhesion
    #   - (cloth, cube):  the bond we actually want
    tabular.insert(cloth_contact, cloth_contact, 0.0, 1.0e6)
    tabular.insert(cloth_contact, cube_contact,  0.0, 1.0e6)

    if adhesion_on:
        adhesive = RCCAdhesive()
        # NOTE: each cloth vert above the cube top forms several PT pairs
        # (multiple cube triangles within d_hat), so the effective adhesion
        # per vert is several × Cn. Keep Cn modest.
        # Default row 0 (cube-cube): disabled — only the cloth-cube row matters.
        adhesive.default_model(
            tabular,
            Cn=1.0e4,
            Ct=1.0e5,
            W=1.0,
            eta=2.0,
            bonding_rate=1.0,
            p0=0.0,
            initial_beta=1.0,
            enabled=False,
        )
        # cloth-cube: ENABLED.
        adhesive.set(
            tabular, cloth_contact, cube_contact,
            Cn=1.0e3, Ct=1.0e5, W=1.0, eta=2.0,
            bonding_rate=1.0, p0=0.0, initial_beta=1.0,
            enabled=True,
        )
        # cloth-cloth: DISABLED so a folded cloth doesn't stick to itself.
        adhesive.set(
            tabular, cloth_contact, cloth_contact,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )

    # ---- big fixed ABD cube ----
    pre = Matrix4x4.Identity()
    pre[0, 0] = CUBE_SCALE
    pre[1, 1] = CUBE_SCALE
    pre[2, 2] = CUBE_SCALE
    io = SimplicialComplexIO(pre)
    cube = io.read(f"{AssetDir.tetmesh_path()}/cube.msh")
    label_surface(cube)
    label_triangle_orient(cube)
    abd.apply_to(cube, 1.0e8)
    cube_contact.apply_to(cube)
    view(cube.transforms())[0] = cube_transform(CUBE_CENTER_Y)
    view(cube.instances().find(builtin.is_fixed))[0] = 1

    cube_obj = scene.objects().create("press_cube")
    cube_slot = cube_obj.geometries().create(cube)[0]

    # ---- animated cloth ----
    moduli = ElasticModuli2D.youngs_poisson(1.0e8, 0.49)

    cloth_obj = scene.objects().create("cloth")
    cloth = make_cloth_mesh(Y_BONDED)
    nhs.apply_to(cloth, moduli)
    cloth_contact.apply_to(cloth)
    # SPC stiffness on the animated edges. Too high and the implicit solve
    # blows up when contact pairs enter the IPC active band; too low and
    # the puller can't peel against adhesion.
    spc.apply_to(cloth, 1.0e5)
    cloth_slot = cloth_obj.geometries().create(cloth)[0]

    rest_positions = np.array(view(cloth.positions()), copy=True).reshape(-1, 3)

    # pull edge: i == N    (high-x edge of the cloth is lifted to peel)
    # The opposite edge (i == 0) is now free — only adhesion to the cube
    # keeps it on the surface as the pull edge lifts.
    pull_ids   = [vid(N, j) for j in range(N + 1)]

    def animate_cloth(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.vertices().find(builtin.is_constrained))
        aim_position = view(geo.vertices().find(builtin.aim_position))

        is_constrained[:] = 0
        for k in pull_ids:
            is_constrained[k] = 1

        _, y_pull, _ = timeline(max(info.frame() - 1, 0))

        for k in pull_ids:
            aim_position[k] = np.array(
                [rest_positions[k, 0], y_pull, rest_positions[k, 2]],
                dtype=np.float64,
            ).reshape(3, 1)

    scene.animator().insert(cloth_obj, animate_cloth)

    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "cube_slot": cube_slot,
        "cloth_slot": cloth_slot,
    }


def run_demo():
    state = {"adhesion_on": True}
    sim = build_demo(state["adhesion_on"])

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    verts = surface.positions().view().reshape(-1, 3)
    tris = surface.triangles().topo().view().reshape(-1, 3)
    mesh = ps.register_surface_mesh("rcc_cloth_on_cube", verts, tris)
    mesh.set_edge_width(0.5)

    ui = {"run": False}

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        v = merged.positions().view().reshape(-1, 3)
        t = merged.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != v.shape[0]:
            ps.remove_surface_mesh("rcc_cloth_on_cube")
            mesh = ps.register_surface_mesh("rcc_cloth_on_cube", v, t)
            mesh.set_edge_width(0.5)
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

        frame = min(sim["world"].frame(), TOTAL_FRAMES)
        y_anchor, y_pull, phase = timeline(frame)

        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase}")
        psim.Text(f"Anchor edge Y: {y_anchor:+.3f}")
        psim.Text(f"Pull edge   Y: {y_pull:+.3f}")
        psim.Text(f"Cube top face Y: {TOP_FACE_Y:+.3f}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        if frame >= TOTAL_FRAMES:
            psim.Text("Sim done. Hit `reset` to rebuild (toggle adhesion first if desired).")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
