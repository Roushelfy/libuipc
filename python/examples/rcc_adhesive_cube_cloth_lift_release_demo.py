"""
RCC adhesion cube-cloth lift/release demo.

A top ABD cube presses down onto a horizontal NeoHookeanShell cloth patch,
holds to bond, lifts the cloth, then pulls apart: the cube target moves upward
while a SoftPositionConstraint target pulls the cloth downward.

Run:
    python python/examples/rcc_adhesive_cube_cloth_lift_release_demo.py

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
        SimplicialComplexIO,
        label_surface,
        label_triangle_orient,
        mesh_partition,
        trimesh,
    )
    from uipc.constitution import (
        AffineBodyConstitution,
        ElasticModuli2D,
        NeoHookeanShell,
        SoftPositionConstraint,
        SoftTransformConstraint,
        RCCAdhesive,
    )
    from uipc.core import RCCAdhesionStateAccessorFeature, RCCBondedPTStateAccessorFeature
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
CUBE_SCALE = 0.3
CLOTH_N = 14
CLOTH_SIZE = 0.44
CLOTH_SPACING = CLOTH_SIZE / CLOTH_N
CLOTH_INITIAL_Y = 0.30
# Pull a cloth CORNER (outside the cube footprint, so it is unbonded) to create a
# peel front at the edge of the bonded patch instead of yanking the center.
PULL_VERTEX_I = 0
PULL_VERTEX_J = 0

PRESS_GAP = 0.010
TOP_CONTACT_Y = CLOTH_INITIAL_Y + 0.5 * CUBE_SCALE + PRESS_GAP
TOP_INITIAL_Y = TOP_CONTACT_Y + 0.40
TOP_LIFT_Y = TOP_CONTACT_Y + 0.36
TOP_PULL_Y = TOP_LIFT_Y + 0.35

CLOTH_LIFT_Y = CLOTH_INITIAL_Y + (TOP_LIFT_Y - TOP_CONTACT_Y)
CLOTH_PULL_Y = CLOTH_INITIAL_Y - 0.16

# --- adhesion / material params ---
ADHESION_CN = 3.0e-1
ADHESION_CT = 3.0
ADHESION_W = 0.5
ADHESION_ETA = 0.5


def smooth_lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def cube_y(frame: int) -> tuple[float, str]:
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


def cloth_pull_y(frame: int, adhesion_on: bool) -> float:
    if frame < LIFT_UNTIL:
        return CLOTH_INITIAL_Y
    start_y = CLOTH_LIFT_Y if adhesion_on else CLOTH_INITIAL_Y
    if frame < PRE_PULL_HOLD_UNTIL:
        return start_y
    if frame < PULL_UNTIL:
        t = (frame - PRE_PULL_HOLD_UNTIL) / (PULL_UNTIL - PRE_PULL_HOLD_UNTIL)
        return smooth_lerp(start_y, CLOTH_PULL_Y, t)
    return CLOTH_PULL_Y


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

    # Wound so the cloth normal points upward. Sticky side +1 therefore bonds
    # the cube pressing from above and ignores the lower cloth face.
    for i in range(CLOTH_N):
        for j in range(CLOTH_N):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v01, v11])
            tris.append([v00, v11, v10])

    cloth = trimesh(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
    )
    label_surface(cloth)
    mesh_partition(cloth, 16)
    return cloth


def build_demo(
    adhesion_on: bool,
    bonded: bool = False,
    skip_ccd: bool = False,
    beta_lock_threshold: float = 0.9,
    kappa: float = 1.0e8,
    release_strain: float = 1.0e30,
    release_gap: float = 1.0e30,
    release_slip: float = 1.0e30,
    release_force: float = 1.0e30,
):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = 0.02
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
        config["rcc_bonded_pt_release_force"] = release_force
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    spc = SoftPositionConstraint()
    stc = SoftTransformConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    cube_contact = tabular.default_element()
    cloth_contact = tabular.create("cloth")
    tabular.insert(cloth_contact, cloth_contact, 0.0, 1.0e9)
    tabular.insert(cloth_contact, cube_contact, 0.5, 1.0e9)

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
            cloth_contact,
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
        adhesive.set(
            tabular,
            cloth_contact,
            cloth_contact,
            Cn=0.0,
            Ct=0.0,
            W=0.0,
            eta=1.0,
            bonding_rate=0.0,
            p0=0.0,
            initial_beta=0.0,
            enabled=False,
        )

    # ---- animated top cube ----
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
    stc.apply_to(cube, np.array([1.0e8, 0.0], dtype=np.float64))
    view(cube.transforms())[0] = cube_transform(TOP_INITIAL_Y)

    cube_obj = scene.objects().create("press_cube")
    cube_obj.geometries().create(cube)

    # ---- cloth patch ----
    cloth = make_cloth_mesh(CLOTH_INITIAL_Y)
    nhs.apply_to(cloth, ElasticModuli2D.youngs_poisson(1.0e7, 0.4))
    cloth_contact.apply_to(cloth)
    spc.apply_to(cloth, 2.0e5)
    if adhesion_on:
        RCCAdhesive.set_sticky_side(cloth, +1)

    rest_positions = np.array(view(cloth.positions()), copy=True).reshape(-1, 3)
    pull_vertex = vid(PULL_VERTEX_I, PULL_VERTEX_J)

    cloth_obj = scene.objects().create("cloth")
    cloth_obj.geometries().create(cloth)

    def animate_cube(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))
        is_constrained[0] = 1
        y, _ = cube_y(max(info.frame() - 1, 0))
        aim_transform[0] = cube_transform(y)

    def animate_cloth(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.vertices().find(builtin.is_constrained))
        aim_position = view(geo.vertices().find(builtin.aim_position))

        frame = max(info.frame() - 1, 0)
        if frame < PRE_PULL_HOLD_UNTIL:
            is_constrained[:] = 0
            return

        target_y = cloth_pull_y(frame, adhesion_on)
        is_constrained[:] = 0
        is_constrained[pull_vertex] = 1
        rest = rest_positions[pull_vertex]
        aim_position[pull_vertex] = np.array(
            [rest[0], target_y, rest[2]],
            dtype=np.float64,
        ).reshape(3, 1)

    scene.animator().insert(cube_obj, animate_cube)
    scene.animator().insert(cloth_obj, animate_cloth)

    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "cube_surface_vertices": 8,
    }


def height_stats(sim) -> tuple[float, float, float]:
    surface = sim["scene_io"].simplicial_surface()
    verts = surface.positions().view().reshape(-1, 3)
    cube_n = sim["cube_surface_vertices"]
    cube = verts[:cube_n]
    cloth = verts[cube_n:]
    cube_y_avg = float(cube[:, 1].mean())
    cloth_y_avg = float(cloth[:, 1].mean())
    return cube_y_avg, cloth_y_avg, cube_y_avg - cloth_y_avg


def bottom_contact_stats(sim) -> dict[str, float | int]:
    surface = sim["scene_io"].simplicial_surface()
    verts = surface.positions().view().reshape(-1, 3)
    cube_n = sim["cube_surface_vertices"]
    cube = verts[:cube_n]
    cloth = verts[cube_n:]

    cube_min = cube.min(axis=0)
    cube_max = cube.max(axis=0)
    in_footprint = (
        (cloth[:, 0] >= cube_min[0])
        & (cloth[:, 0] <= cube_max[0])
        & (cloth[:, 2] >= cube_min[2])
        & (cloth[:, 2] <= cube_max[2])
    )
    covered = cloth[in_footprint]
    if len(covered) == 0:
        return {
            "count": 0,
            "gap_mean": float("nan"),
            "gap_min": float("nan"),
            "gap_max": float("nan"),
        }

    bottom_y = float(cube_min[1])
    gaps = bottom_y - covered[:, 1]
    return {
        "count": int(len(covered)),
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


# Bonded-PT viewer config (kappa lowered for the FEM cloth; finite release
# thresholds so the pull stretches the bond). See scripts/probe_rcc_bonded_pt_demos.py.
def _build(state):
    return build_demo(
        state["adhesion_on"],
        bonded=state["bonded"],
        skip_ccd=state["bonded"],
        beta_lock_threshold=0.85,
        kappa=5.0e7,
        # Geometric release (strain/gap) cannot peel a stiff bond on compliant
        # cloth at this kappa; the force/energy criterion can. It holds through
        # press/hold/lift then peels the corner pull (~all bonds release).
        release_strain=1.0e30,
        release_gap=1.0e30,
        release_force=1.0e-4,
    )


def run_demo():
    # Pass --bonded to enable bonded-PT acceleration + skip_ccd and draw the
    # bonded virtual tets (red). Default is the plain (non-bonded) demo.
    state = {"adhesion_on": True, "bonded": ("--bonded" in sys.argv)}
    sim = _build(state)

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "rcc_cube_cloth_lift_release",
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
            ps.remove_surface_mesh("rcc_cube_cloth_lift_release")
            mesh = ps.register_surface_mesh(
                "rcc_cube_cloth_lift_release", verts, tris
            )
            mesh.set_edge_width(1.0)
        else:
            mesh.update_vertex_positions(verts)

        if ps.has_curve_network("cube_cloth_bonds"):
            ps.remove_curve_network("cube_cloth_bonds")
        if state["bonded"]:
            _, bonds = _bonded_bonds(sim)
            if bonds is not None:
                net = ps.register_curve_network("cube_cloth_bonds", bonds[0], bonds[1])
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
        sim = _build(state)
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
        if psim.Button(f"bonded: {'ON' if state['bonded'] else 'OFF'}"):
            state["bonded"] = not state["bonded"]

        psim.SameLine()
        if psim.Button("reset"):
            reset()

        if ui["run"]:
            step_once()

        frame = min(sim["world"].frame(), TOTAL_FRAMES)
        cube_target_y, phase = cube_y(frame)
        cloth_target_y = cloth_pull_y(frame, state["adhesion_on"])
        cube_y_avg, cloth_y_avg, gap_y = height_stats(sim)
        bottom_stats = bottom_contact_stats(sim)
        beta_stats = adhesion_beta_stats(sim) if state["adhesion_on"] else None

        psim.Separator()
        psim.Text(f"Frame: {frame} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase}")
        psim.Text(f"Cube target Y: {cube_target_y:+.3f}")
        psim.Text(f"Cloth target Y: {cloth_target_y:+.3f}")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        psim.Text(f"Cube avg Y: {cube_y_avg:+.3f}")
        psim.Text(f"Cloth avg Y: {cloth_y_avg:+.3f}")
        psim.Text(f"Avg gap Y: {gap_y:+.3f}")
        psim.Text(
            "Bottom gap: "
            f"n={bottom_stats['count']} "
            f"mean={bottom_stats['gap_mean']:+.4f} "
            f"min={bottom_stats['gap_min']:+.4f} "
            f"max={bottom_stats['gap_max']:+.4f}"
        )
        if beta_stats is not None:
            psim.Text(
                "PT beta: "
                f"n={beta_stats['count']} "
                f"mean={beta_stats['mean']:.3f} "
                f"min={beta_stats['min']:.3f} "
                f">0.9={100.0 * beta_stats['frac_09']:.1f}%"
            )
        if state["bonded"]:
            locked, _ = _bonded_bonds(sim)
            psim.Text(f"Bonded PT (skip_ccd): ON  locked={locked} (red bonds)")
        else:
            psim.Text("Bonded PT: OFF  (toggle `bonded` + `reset`, or pass --bonded)")
        if frame >= TOTAL_FRAMES:
            psim.Text("Sim done. Hit `reset` to rebuild.")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
