"""
Small cuda_mixed ABD BDF2 viewer.

This mirrors apps/tests/sim_case/80_cuda_mixed_abd_bdf2_smoke.cpp, but keeps the
scene tiny enough to step interactively from Polyscope.

Run:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_abd_bdf2_viewer.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from uipc import Engine, Logger, Matrix4x4, Scene, World, view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh
from uipc.gui import SceneGUI

from cuda_mixed_runtime import init_cuda_mixed_module_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


def build_scene(backend: str):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine(backend, workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["integrator"]["type"] = "bdf2"
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["contact"]["constitution"] = "ipc"
    scene = Scene(config)

    scene.contact_tabular().default_model(0.5, 1e9)
    default_element = scene.contact_tabular().default_element()

    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-np.sqrt(3.0) / 2.0, 0.0, -0.5],
            [np.sqrt(3.0) / 2.0, 0.0, -0.5],
        ],
        dtype=np.float64,
    )
    vertices *= 0.3
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)

    body = tetmesh(vertices, tets)
    label_surface(body)
    label_triangle_orient(body)
    AffineBodyConstitution().apply_to(body, 1e8)
    default_element.apply_to(body)

    transforms = view(body.transforms())
    transform = Matrix4x4.Identity()
    transform[1, 3] = 0.6
    transforms[0] = transform

    scene.objects().create("falling_abd").geometries().create(body)
    scene.objects().create("ground").geometries().create(ground(0.0))

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("cuda_mixed ABD BDF2 scene failed to initialize")
    world.retrieve()

    return engine, world, scene


def smoke(world: World, frames: int) -> None:
    for _ in range(frames):
        world.advance()
        if not world.is_valid():
            raise RuntimeError(f"world became invalid at frame {world.frame()}")
        world.retrieve()
    print(f"abd_bdf2 smoke passed: frame={world.frame()}")


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
        psim.Text("ABD BDF2 falling tetra")
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
