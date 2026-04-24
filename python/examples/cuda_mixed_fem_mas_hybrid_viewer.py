"""
Small cuda_mixed FEM/MAS hybrid viewer.

This mirrors apps/tests/sim_case/84_cuda_mixed_fem_mas_hybrid_smoke.cpp with two
elastic cubes and a ground contact plane.

Run:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_fem_mas_hybrid_viewer.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from uipc import Engine, Logger, Matrix4x4, Scene, World, view
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    ground,
    label_surface,
    label_triangle_orient,
    mesh_partition,
)
from uipc.gui import SceneGUI

from cuda_mixed_runtime import init_cuda_mixed_module_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


def process_cube(cube: SimplicialComplex) -> SimplicialComplex:
    label_surface(cube)
    label_triangle_orient(cube)
    return flip_inward_triangles(cube)


def translate_positions(cube: SimplicialComplex, offset: np.ndarray) -> None:
    positions = view(cube.positions())
    for i in range(len(positions)):
        positions[i][0] += offset[0]
        positions[i][1] += offset[1]
        positions[i][2] += offset[2]


def build_scene(backend: str):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine(backend, workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["line_search"]["max_iter"] = 8
    config["linear_system"]["tol_rate"] = 1e-3
    scene = Scene(config)

    snh = StableNeoHookean()
    scene.constitution_tabular().insert(snh)
    scene.contact_tabular().default_model(0.5, 1e9)
    default_element = scene.contact_tabular().default_element()

    pre = Matrix4x4.Identity()
    pre[0, 0] = 0.2
    pre[1, 1] = 0.2
    pre[2, 2] = 0.2
    io = SimplicialComplexIO(pre)

    moduli = ElasticModuli.youngs_poisson(1e5, 0.49)
    obj = scene.objects().create("hybrid")

    cube_a = process_cube(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
    mesh_partition(cube_a, 16)
    snh.apply_to(cube_a, moduli)
    default_element.apply_to(cube_a)
    translate_positions(cube_a, np.array([-0.15, 0.55, 0.0], dtype=np.float64))
    obj.geometries().create(cube_a)

    cube_b = process_cube(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
    snh.apply_to(cube_b, moduli)
    default_element.apply_to(cube_b)
    translate_positions(cube_b, np.array([0.15, 1.05, 0.0], dtype=np.float64))
    obj.geometries().create(cube_b)

    scene.objects().create("ground").geometries().create(ground(0.0))

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("cuda_mixed FEM/MAS hybrid scene failed to initialize")
    world.retrieve()

    return engine, world, scene


def smoke(world: World, frames: int) -> None:
    for _ in range(frames):
        world.advance()
        if not world.is_valid():
            raise RuntimeError(f"world became invalid at frame {world.frame()}")
        world.retrieve()
    print(f"fem_mas_hybrid smoke passed: frame={world.frame()}")


def run_viewer(world: World, scene: Scene) -> None:
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
        psim.Text("FEM/MAS hybrid cubes")
        psim.Text(f"Frame: {world.frame()}")

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda_mixed")
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "cuda_mixed":
        init_cuda_mixed_module_dir()

    engine, world, scene = build_scene(args.backend)
    if args.smoke_frames > 0:
        smoke(world, args.smoke_frames)
        return

    _ = engine
    run_viewer(world, scene)


if __name__ == "__main__":
    main()
