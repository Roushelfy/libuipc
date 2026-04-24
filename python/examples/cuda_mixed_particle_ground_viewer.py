"""
Small cuda_mixed particle-ground viewer.

Run:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_particle_ground_viewer.py

Smoke check:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_particle_ground_viewer.py --smoke-frames 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from uipc import Engine, Logger, Scene, World
from uipc.constitution import Particle
from uipc.geometry import ground, label_surface, pointcloud
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
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["contact"]["constitution"] = "ipc"
    scene = Scene(config)

    scene.contact_tabular().default_model(0.5, 1e9)
    default_element = scene.contact_tabular().default_element()

    particles = np.zeros((10, 3), dtype=np.float64)
    particles[:, 1] = np.arange(10, dtype=np.float64) * 0.05 + 0.2
    particles[:, 0] = np.linspace(-0.18, 0.18, 10)

    mesh = pointcloud(particles)
    label_surface(mesh)
    Particle().apply_to(mesh, 1e3, 0.01)
    default_element.apply_to(mesh)

    scene.objects().create("particles").geometries().create(mesh)
    scene.objects().create("ground").geometries().create(ground(0.0))

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("cuda_mixed particle scene failed to initialize")
    world.retrieve()

    return engine, world, scene


def smoke(world: World, frames: int) -> None:
    for _ in range(frames):
        world.advance()
        if not world.is_valid():
            raise RuntimeError(f"world became invalid at frame {world.frame()}")
        world.retrieve()
    print(f"particle_ground smoke passed: frame={world.frame()}")


def run_viewer(world: World, scene: Scene) -> None:
    scene_gui = SceneGUI(scene)
    state = {"run": False}

    ps.init()
    ps.set_ground_plane_height(0.0)
    scene_gui.register()

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
        psim.Text(f"Backend: cuda_mixed")
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

    # Keep the engine alive while the viewer owns the callback.
    _ = engine
    run_viewer(world, scene)


if __name__ == "__main__":
    main()
