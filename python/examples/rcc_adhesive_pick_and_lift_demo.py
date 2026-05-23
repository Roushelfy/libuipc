"""
RCC adhesion pick-and-lift demo.

Two ABD cubes on a ground plane:
- bottom (pickee) — free, rests on the ground under gravity.
- top    (picker) — animated via SoftTransformConstraint; presses down on
                    the pickee, holds, then lifts straight up.

With RCC adhesion enabled the pickee follows the picker up; without it the
pickee stays on the ground.

Run:
    python python/examples/rcc_adhesive_pick_and_lift_demo.py

Controls:
    - run / pause: toggle playback
    - step:        advance one frame
    - adhesion:    toggle adhesion ON/OFF (requires a sim reset)
    - reset:       rebuild the scene
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
        ground,
        label_surface,
        label_triangle_orient,
    )
    from uipc.constitution import (
        AffineBodyConstitution,
        SoftTransformConstraint,
        RCCAdhesive,
    )
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# --- timeline (frames, dt=0.01) ---
PRESS_START = 5
CONTACT_AT  = 60
HOLD_UNTIL  = 100
LIFT_UNTIL  = 200
TOTAL_FRAMES = 260

# --- geometry params ---
CUBE_SCALE = 0.3   # cube edge length
TOP_INITIAL_Y = 0.70
TOP_CONTACT_Y = 0.47
TOP_FINAL_Y   = 0.90


def smooth_lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def picker_y(frame: int) -> tuple[float, str]:
    """Return (target_y, phase_label) for the picker on the given frame."""
    if frame < PRESS_START:
        return TOP_INITIAL_Y, "init"
    if frame < CONTACT_AT:
        t = (frame - PRESS_START) / (CONTACT_AT - PRESS_START)
        return smooth_lerp(TOP_INITIAL_Y, TOP_CONTACT_Y, t), "press"
    if frame < HOLD_UNTIL:
        return TOP_CONTACT_Y, "hold"
    if frame < LIFT_UNTIL:
        t = (frame - HOLD_UNTIL) / (LIFT_UNTIL - HOLD_UNTIL)
        return smooth_lerp(TOP_CONTACT_Y, TOP_FINAL_Y, t), "lift"
    return TOP_FINAL_Y, "settled"


def cube_transform(center_y: float) -> Matrix4x4:
    m = Matrix4x4.Identity()
    m[0:3, 3] = np.array([0.0, center_y, 0.0], dtype=np.float64)
    return m


def build_demo(adhesion_on: bool):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    # widen the IPC active band so v1 adhesion has a longer effective reach
    config["contact"]["d_hat"] = 0.02
    # let line search retry instead of aborting on stiff adhesion forces
    config["extras"]["strict_mode"]["enable"] = False
    scene = Scene(config)

    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()

    scene.contact_tabular().default_model(0.5, 1.0e9)
    default_contact = scene.contact_tabular().default_element()

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            scene.contact_tabular(),
            Cn=1.0e6,
            Ct=1.0e6,
            W=1.0,
            eta=2.0,
            bonding_rate=1.0,
            p0=0.0,
            initial_beta=1.0,
            enabled=True,
        )

    # pre-scale the unit cube down to CUBE_SCALE
    pre = Matrix4x4.Identity()
    pre[0, 0] = CUBE_SCALE
    pre[1, 1] = CUBE_SCALE
    pre[2, 2] = CUBE_SCALE
    io = SimplicialComplexIO(pre)
    cube = io.read(f"{AssetDir.tetmesh_path()}/cube.msh")
    label_surface(cube)
    label_triangle_orient(cube)

    # 2 instances: pickee (idx 0) and picker (idx 1)
    cube.instances().resize(2)
    abd.apply_to(cube, 1.0e8)
    default_contact.apply_to(cube)
    # stiff transform constraint so the picker tracks aim_transform tightly
    stc.apply_to(cube, np.array([1.0e8, 0.0], dtype=np.float64))

    trans = view(cube.transforms())
    trans[0] = cube_transform(0.17)             # pickee on the ground
    trans[1] = cube_transform(TOP_INITIAL_Y)     # picker above

    cube_obj = scene.objects().create("cubes")
    cube_slot = cube_obj.geometries().create(cube)[0]

    # ground at y = 0
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    def animate_cubes(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))

        # pickee free, picker animated
        is_constrained[0] = 0
        is_constrained[1] = 1

        y, _ = picker_y(max(info.frame() - 1, 0))
        aim_transform[1] = cube_transform(y)
        # keep pickee's aim identity (unused but avoids stale data)
        aim_transform[0] = cube_transform(0.155)

    scene.animator().insert(cube_obj, animate_cubes)

    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "cube_slot": cube_slot,
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
    mesh = ps.register_surface_mesh(
        "rcc_pick_and_lift",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(1.0)

    ui = {"run": False}

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        verts = merged.positions().view().reshape(-1, 3)
        tris = merged.triangles().topo().view().reshape(-1, 3)
        # topology may change if we rebuilt; re-register if needed
        if mesh.n_vertices() != verts.shape[0]:
            ps.remove_surface_mesh("rcc_pick_and_lift")
            mesh = ps.register_surface_mesh("rcc_pick_and_lift", verts, tris)
            mesh.set_edge_width(1.0)
        else:
            mesh.update_vertex_positions(verts)

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
        target_y, phase = picker_y(frame)

        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase}")
        psim.Text(f"Picker target Y: {target_y:+.3f}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        if frame >= TOTAL_FRAMES:
            psim.Text("Sim done. Hit `reset` to rebuild.")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
