"""
RCC adhesion subdivided-cube lift/release demo.

Two ABD cubes on a ground plane:
- bottom (pickee) -- free, rests on the ground under gravity.
- top    (picker) -- animated via SoftTransformConstraint; presses down on
                    the pickee, holds, lifts, then pulls upward while the
                    bottom cube is pulled downward.

This variant uses a procedurally subdivided tet cube so the contact face has
enough point-triangle samples for a stable adhesion-on/off visual difference.

Run:
    python python/examples/rcc_adhesive_subdivided_cube_lift_release_demo.py

Controls:
    - run / pause: toggle playback
    - step:        advance one frame
    - adhesion:    toggle adhesion ON/OFF (requires reset)
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
        ground,
        label_surface,
        label_triangle_orient,
        tetmesh,
    )
    from uipc.constitution import (
        AffineBodyConstitution,
        SoftTransformConstraint,
        RCCAdhesive,
    )
    from uipc.core import RCCAdhesionStateAccessorFeature
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# --- timeline (frames, dt=0.01) ---
PRESS_START = 5
CONTACT_AT = 60
HOLD_UNTIL = 100
LIFT_UNTIL = 200
PRE_PULL_HOLD_UNTIL = 240
PULL_UNTIL = 360
TOTAL_FRAMES = 400

# --- geometry params ---
GRID_N = 2
CUBE_SCALE = 0.3
BOTTOM_INITIAL_Y = 0.17
BOTTOM_PULL_Y = 0.17
TOP_INITIAL_Y = 0.70
TOP_CONTACT_Y = 0.47
TOP_LIFT_Y = 0.90
TOP_PULL_Y = 1.20
BOTTOM_LIFT_Y = BOTTOM_INITIAL_Y + (TOP_LIFT_Y - TOP_CONTACT_Y)

# --- adhesion params ---
ADHESION_CN = 1.0e3
ADHESION_CT = 1.0e3
ADHESION_W = 3.8
ADHESION_ETA = 0.2


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
        return smooth_lerp(TOP_CONTACT_Y, TOP_LIFT_Y, t), "lift"
    if frame < PRE_PULL_HOLD_UNTIL:
        return TOP_LIFT_Y, "pre-pull hold"
    if frame < PULL_UNTIL:
        t = (frame - PRE_PULL_HOLD_UNTIL) / (PULL_UNTIL - PRE_PULL_HOLD_UNTIL)
        return smooth_lerp(TOP_LIFT_Y, TOP_PULL_Y, t), "pull"
    return TOP_PULL_Y, "settled"


def bottom_pull_y(frame: int, adhesion_on: bool) -> float:
    """Return the downward pull target for the bottom cube."""
    if frame < LIFT_UNTIL:
        return BOTTOM_INITIAL_Y
    start_y = BOTTOM_LIFT_Y if adhesion_on else BOTTOM_INITIAL_Y
    if frame < PRE_PULL_HOLD_UNTIL:
        return start_y
    if frame < PULL_UNTIL:
        t = (frame - PRE_PULL_HOLD_UNTIL) / (PULL_UNTIL - PRE_PULL_HOLD_UNTIL)
        return smooth_lerp(start_y, BOTTOM_PULL_Y, t)
    return BOTTOM_PULL_Y


def cube_transform(center_y: float) -> Matrix4x4:
    m = Matrix4x4.Identity()
    m[0:3, 3] = np.array([0.0, center_y, 0.0], dtype=np.float64)
    return m


def make_subdivided_cube(n: int = GRID_N, scale: float = CUBE_SCALE) -> SimplicialComplex:
    coords = np.linspace(-0.5 * scale, 0.5 * scale, n + 1)

    def vid(i: int, j: int, k: int) -> int:
        return (i * (n + 1) + j) * (n + 1) + k

    verts = []
    for x in coords:
        for y in coords:
            for z in coords:
                verts.append([x, y, z])

    cell_corners = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    local_tets = [
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    ]

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ids = [vid(i + di, j + dj, k + dk) for di, dj, dk in cell_corners]
                for tet in local_tets:
                    tet_ids = [ids[q] for q in tet]
                    a, b, c, d = np.asarray([verts[q] for q in tet_ids])
                    signed_volume = np.dot(b - a, np.cross(c - a, d - a))
                    if signed_volume < 0.0:
                        tet_ids[2], tet_ids[3] = tet_ids[3], tet_ids[2]
                    tets.append(tet_ids)

    cube = tetmesh(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tets, dtype=np.int32),
    )
    label_surface(cube)
    label_triangle_orient(cube)
    return cube


def build_demo(
    adhesion_on: bool,
    bonded: bool = False,
    skip_ccd: bool = False,
    beta_lock_threshold: float = 0.9,
    kappa: float = 1.0e8,
    release_strain: float = 1.0e30,
    release_gap: float = 1.0e30,
    release_slip: float = 1.0e30,
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
    config["contact"]["d_hat"] = 0.02
    config["extras"]["strict_mode"]["enable"] = False
    if bonded:
        # Bonded-PT acceleration: stable high-beta PT pairs become a stiff ABD
        # virtual tet. With skip_ccd, they are also removed from CCD broadphase.
        config["rcc_bonded_pt_enabled"] = 1
        config["rcc_bonded_pt_skip_ccd"] = 1 if skip_ccd else 0
        config["rcc_bonded_pt_beta_lock_threshold"] = beta_lock_threshold
        config["rcc_bonded_pt_energy_model"] = "abd_ortho"
        config["rcc_bonded_pt_kappa"] = kappa
        # Release thresholds (default 1e30 = disabled). Finite values let a
        # locked pair release back to RCC/contact when the bond is stretched
        # (normal gap), distorted (strain), or slid (slip) past the threshold.
        config["rcc_bonded_pt_release_strain"] = release_strain
        config["rcc_bonded_pt_release_gap"] = release_gap
        config["rcc_bonded_pt_release_slip"] = release_slip
    scene = Scene(config)

    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    cube_contact = tabular.default_element()
    ground_contact = tabular.create("ground")
    tabular.insert(cube_contact, ground_contact, 0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            tabular,
            Cn=0.0,
            Ct=0.0,
            W=0.0,
            eta=1.0,
            bonding_rate=0.0,
            p0=0.0,
            initial_beta=0.0,
            enabled=False,
        )
        adhesive.set(
            tabular,
            cube_contact,
            cube_contact,
            Cn=ADHESION_CN,
            Ct=ADHESION_CT,
            W=ADHESION_W,
            eta=ADHESION_ETA,
            bonding_rate=1.0,
            p0=0.0,
            initial_beta=1.0,
            enabled=True,
        )

    cube = make_subdivided_cube()
    cube.instances().resize(2)
    abd.apply_to(cube, 1.0e8)
    cube_contact.apply_to(cube)
    stc.apply_to(cube, np.array([1.0e8, 0.0], dtype=np.float64))

    trans = view(cube.transforms())
    trans[0] = cube_transform(BOTTOM_INITIAL_Y)
    trans[1] = cube_transform(TOP_INITIAL_Y)

    cube_obj = scene.objects().create("subdivided_cubes")
    cube_slot = cube_obj.geometries().create(cube)[0]

    ground_sc = ground(0.0)
    ground_contact.apply_to(ground_sc)
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground_sc)

    def animate_cubes(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))

        frame = max(info.frame() - 1, 0)
        top_y, _ = picker_y(frame)
        aim_transform[1] = cube_transform(top_y)

        if frame < PRE_PULL_HOLD_UNTIL:
            is_constrained[0] = 0
            aim_transform[0] = cube_transform(BOTTOM_INITIAL_Y)
        else:
            is_constrained[0] = 1
            aim_transform[0] = cube_transform(bottom_pull_y(frame, adhesion_on))

        is_constrained[1] = 1

    scene.animator().insert(cube_obj, animate_cubes)

    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "cube_slot": cube_slot,
    }


def cube_height_stats(sim) -> tuple[float, float, float]:
    surface = sim["scene_io"].simplicial_surface()
    verts = surface.positions().view().reshape(-1, 3)
    half = verts.shape[0] // 2
    bottom = verts[:half]
    top = verts[half : 2 * half]
    bottom_y = float(bottom[:, 1].mean())
    top_y = float(top[:, 1].mean())
    return bottom_y, top_y, top_y - bottom_y


def cube_contact_gap_stats(sim) -> dict[str, float | int]:
    geo = sim["cube_slot"].geometry()
    rest = np.asarray(view(geo.positions()), dtype=np.float64).reshape(-1, 3)
    transforms = np.asarray(view(geo.transforms()), dtype=np.float64).reshape(-1, 4, 4)

    def transform_points(instance: int, points: np.ndarray) -> np.ndarray:
        linear = transforms[instance][:3, :3]
        translation = transforms[instance][:3, 3]
        return points @ linear.T + translation

    tol = 1.0e-9
    bottom_top_face = rest[np.abs(rest[:, 1] - rest[:, 1].max()) <= tol]
    top_bottom_face = rest[np.abs(rest[:, 1] - rest[:, 1].min()) <= tol]
    bottom_face = transform_points(0, bottom_top_face)
    top_face = transform_points(1, top_bottom_face)
    gaps = top_face[:, 1] - bottom_face[:, 1]

    return {
        "bottom_count": int(len(bottom_face)),
        "top_count": int(len(top_face)),
        "gap_mean": float(gaps.mean()),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
    }


def adhesion_beta_stats(sim) -> dict[str, float | int] | None:
    acc = sim["world"].features().find(RCCAdhesionStateAccessorFeature)
    if acc is None:
        return None

    _, betas = acc.dump_pt_state()
    betas = np.asarray(betas, dtype=np.float64)
    if len(betas) == 0:
        return {"count": 0, "mean": float("nan"), "min": float("nan"), "frac_09": 0.0}
    return {
        "count": int(len(betas)),
        "mean": float(betas.mean()),
        "min": float(betas.min()),
        "frac_09": float(np.mean(betas > 0.9)),
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
        "rcc_subdivided_cube_lift_release",
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
        if mesh.n_vertices() != verts.shape[0]:
            ps.remove_surface_mesh("rcc_subdivided_cube_lift_release")
            mesh = ps.register_surface_mesh(
                "rcc_subdivided_cube_lift_release", verts, tris
            )
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
        bottom_target_y = bottom_pull_y(frame, state["adhesion_on"])
        bottom_y, top_y, gap_y = cube_height_stats(sim)
        contact_stats = cube_contact_gap_stats(sim)
        beta_stats = adhesion_beta_stats(sim) if state["adhesion_on"] else None

        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase}")
        psim.Text(f"Picker target Y: {target_y:+.3f}")
        psim.Text(f"Bottom target Y: {bottom_target_y:+.3f}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        psim.Text(f"Bottom avg Y: {bottom_y:+.3f}")
        psim.Text(f"Top avg Y: {top_y:+.3f}")
        psim.Text(f"Avg gap Y: {gap_y:+.3f}")
        psim.Text(
            "Contact gap: "
            f"bottom_n={contact_stats['bottom_count']} "
            f"top_n={contact_stats['top_count']} "
            f"mean={contact_stats['gap_mean']:+.4f}"
        )
        if beta_stats is not None:
            psim.Text(
                "PT beta: "
                f"n={beta_stats['count']} "
                f"mean={beta_stats['mean']:.3f} "
                f"min={beta_stats['min']:.3f} "
                f">0.9={100.0 * beta_stats['frac_09']:.1f}%"
            )
        if frame >= TOTAL_FRAMES:
            psim.Text("Sim done. Hit `reset` to rebuild.")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
