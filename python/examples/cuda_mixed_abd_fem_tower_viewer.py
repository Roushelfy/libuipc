"""
cuda_mixed ABD/FEM tower viewer.

The scene stacks alternating ABD and FEM cubes on a ground plane. By default it
uses fused_pcg so the mixed-provider scene is always runnable; pass
``--solver socu_approx`` to exercise the default structured mixed-provider path.

Run:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_abd_fem_tower_viewer.py

Smoke:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_abd_fem_tower_viewer.py --smoke-frames 5

SOCU mixed-provider smoke:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_abd_fem_tower_viewer.py --solver socu_approx --smoke-frames 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from uipc import Engine, Logger, Matrix4x4, Scene, World, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    ground,
    label_surface,
    label_triangle_orient,
)
from uipc.gui import SceneGUI

from cuda_mixed_runtime import init_cuda_mixed_module_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


def process_cube(cube: SimplicialComplex) -> SimplicialComplex:
    label_surface(cube)
    label_triangle_orient(cube)
    return flip_inward_triangles(cube)


def translate_positions(mesh: SimplicialComplex, offset: np.ndarray) -> None:
    positions = view(mesh.positions())
    for i in range(len(positions)):
        positions[i][0] += offset[0]
        positions[i][1] += offset[1]
        positions[i][2] += offset[2]


def set_instance_translation(mesh: SimplicialComplex, offset: np.ndarray) -> None:
    transforms = view(mesh.transforms())
    transform = Matrix4x4.Identity()
    transform[0, 3] = float(offset[0])
    transform[1, 3] = float(offset[1])
    transform[2, 3] = float(offset[2])
    transforms[0] = transform


def configure_solver(config, solver: str, workspace: str) -> None:
    config["linear_system"]["solver"] = solver
    config["linear_system"]["tol_rate"] = 1e-3

    if solver != "socu_approx":
        return

    report_path = os.path.join(workspace, "socu_approx_report.json")
    ordering_path = os.path.join(workspace, "socu_approx_ordering.json")
    config["linear_system"]["socu_approx"]["mode"] = "solve"
    config["linear_system"]["socu_approx"]["ordering_source"] = "init_time"
    config["linear_system"]["socu_approx"]["ordering_orderer"] = "auto_stable"
    config["linear_system"]["socu_approx"]["ordering_block_size"] = "auto"
    config["linear_system"]["socu_approx"]["generated_ordering_report"] = ordering_path
    config["linear_system"]["socu_approx"]["dry_run_report"] = report_path
    config["linear_system"]["socu_approx"]["min_block_utilization"] = 0.0
    config["linear_system"]["socu_approx"]["damping_shift"] = 1.0
    config["linear_system"]["socu_approx"]["max_relative_residual"] = 1e-3
    config["linear_system"]["socu_approx"]["debug_validation"] = 1
    config["linear_system"]["socu_approx"]["report_each_solve"] = 1


def build_scene(backend: str, solver: str, levels: int):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    os.makedirs(workspace, exist_ok=True)
    engine = Engine(backend, workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["contact"]["constitution"] = "ipc"
    config["line_search"]["max_iter"] = 8
    configure_solver(config, solver, workspace)

    scene = Scene(config)
    scene.contact_tabular().default_model(0.45, 1e9)
    default_contact = scene.contact_tabular().default_element()

    abd = AffineBodyConstitution()
    snh = StableNeoHookean()
    scene.constitution_tabular().insert(abd)
    scene.constitution_tabular().insert(snh)

    pre = Matrix4x4.Identity()
    pre[0, 0] = 0.22
    pre[1, 1] = 0.22
    pre[2, 2] = 0.22
    io = SimplicialComplexIO(pre)

    moduli = ElasticModuli.youngs_poisson(8e4, 0.42)
    cube_path = f"{AssetDir.tetmesh_path()}/cube.msh"

    abd_object = scene.objects().create("abd_tower_blocks")
    fem_object = scene.objects().create("fem_tower_blocks")

    for i in range(levels):
        x_offset = 0.035 if i % 2 else -0.035
        offset = np.array([x_offset, 0.16 + i * 0.245, 0.0], dtype=np.float64)
        cube = process_cube(io.read(cube_path))
        default_contact.apply_to(cube)

        if i % 2 == 0:
            abd.apply_to(cube, 1.0e8)
            set_instance_translation(cube, offset)
            abd_object.geometries().create(cube)
        else:
            snh.apply_to(cube, moduli)
            translate_positions(cube, offset)
            fem_object.geometries().create(cube)

    scene.objects().create("ground").geometries().create(ground(0.0))

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("cuda_mixed ABD/FEM tower scene failed to initialize")
    world.retrieve()

    return engine, world, scene


def smoke(world: World, frames: int) -> None:
    for _ in range(frames):
        world.advance()
        if not world.is_valid():
            raise RuntimeError(f"world became invalid at frame {world.frame()}")
        world.retrieve()
    print(f"abd_fem_tower smoke passed: frame={world.frame()}")


def run_viewer(world: World, scene: Scene, solver: str) -> None:
    scene_gui = SceneGUI(scene)
    state = {"run": False}

    ps.init()
    ps.set_ground_plane_height(0.0)
    scene_gui.register()
    scene_gui.set_edge_width(1.0)

    def on_update():
        if psim.Button("run / pause"):
            state["run"] = not state["run"]

        psim.SameLine()
        if psim.Button("step"):
            world.advance()
            world.retrieve()
            scene_gui.update()

        if state["run"]:
            world.advance()
            world.retrieve()
            scene_gui.update()

        psim.Separator()
        psim.Text("ABD/FEM tower")
        psim.Text(f"Solver: {solver}")
        psim.Text(f"Frame: {world.frame()}")

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda_mixed")
    parser.add_argument("--solver", choices=["fused_pcg", "socu_approx"], default="fused_pcg")
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "cuda_mixed":
        init_cuda_mixed_module_dir()

    engine, world, scene = build_scene(args.backend, args.solver, args.levels)
    if args.smoke_frames > 0:
        smoke(world, args.smoke_frames)
        return

    _ = engine
    run_viewer(world, scene, args.solver)


if __name__ == "__main__":
    main()
