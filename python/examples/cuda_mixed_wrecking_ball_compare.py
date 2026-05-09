"""
Run the apps/examples/wrecking_ball scene through cuda_mixed solver variants.

Example:
    LD_LIBRARY_PATH=build/build_impl_fp64/python/src/uipc/_native:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_fp64/python/src \
    apps/benchmarks/mixed/uipc_assets/.venv/bin/python \
        python/examples/cuda_mixed_wrecking_ball_compare.py --variant regression --frames 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from uipc import Engine, Logger, Matrix4x4, Scene, World, builtin, view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    ground,
    label_surface,
    label_triangle_orient,
)
from uipc.stats import SimulationStats

from cuda_mixed_runtime import init_cuda_mixed_module_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


VARIANTS = {
    "fused_pcg": {"solver": "fused_pcg", "runtime_interval": 0},
    "socu_init": {"solver": "socu_approx", "runtime_interval": 0},
    "socu_rt1": {"solver": "socu_approx", "runtime_interval": 1},
    "socu_rt1_contact_hessian": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "contact_hessian",
    },
    "socu_rt1_contact_hessian_diag": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "contact_hessian",
        "contact_offband_policy": "diag",
    },
    "socu_rt1_contact_hessian_diag_lump": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "contact_hessian",
        "contact_offband_policy": "diag_lump",
    },
    "socu_rt1_full_hessian": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "full_hessian",
    },
    "socu_rt1_full_hessian_diag": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "full_hessian",
        "contact_offband_policy": "diag",
    },
    "socu_rt1_full_hessian_diag_lump": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "full_hessian",
        "contact_offband_policy": "diag_lump",
    },
    "socu_rt1_contact_weight_approx": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "contact_weight_approx",
    },
    "socu_rt1_full_weight_approx": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "full_weight_approx",
    },
    "socu_rt1_full_hessian_cached": {
        "solver": "socu_approx",
        "runtime_interval": 1,
        "runtime_graph_source": "full_hessian_cached",
    },
    "socu_rt5": {"solver": "socu_approx", "runtime_interval": 5},
    "socu_rt10": {"solver": "socu_approx", "runtime_interval": 10},
}

VARIANTS.update(
    {
        "socu_init_topology_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 0,
            "runtime_graph_source": "topology",
            "contact_offband_policy": "diag_lump",
        },
        "socu_init_contact_hessian_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 0,
            "runtime_graph_source": "contact_hessian",
            "contact_offband_policy": "diag_lump",
        },
        "socu_init_full_hessian_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 0,
            "runtime_graph_source": "full_hessian",
            "contact_offband_policy": "diag_lump",
        },
        "socu_rt20_topology_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 20,
            "runtime_graph_source": "topology",
            "contact_offband_policy": "diag_lump",
        },
        "socu_rt25_topology_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 25,
            "runtime_graph_source": "topology",
            "contact_offband_policy": "diag_lump",
        },
        "socu_rt50_topology_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 50,
            "runtime_graph_source": "topology",
            "contact_offband_policy": "diag_lump",
        },
        "socu_rt20_contact_hessian_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 20,
            "runtime_graph_source": "contact_hessian",
            "contact_offband_policy": "diag_lump",
        },
        "socu_rt50_contact_hessian_diag_lump": {
            "solver": "socu_approx",
            "runtime_interval": 50,
            "runtime_graph_source": "contact_hessian",
            "contact_offband_policy": "diag_lump",
        },
    }
)

QUICK_VARIANTS = ["fused_pcg", "socu_init", "socu_rt1", "socu_rt5", "socu_rt10"]

REGRESSION_VARIANTS = [
    "fused_pcg",
    "socu_rt1_full_hessian",
    "socu_rt1_full_hessian_cached",
    "socu_init_full_hessian_diag_lump",
    "socu_rt20_topology_diag_lump",
    "socu_rt50_topology_diag_lump",
]


def set_scene_config_path(config: Any, path: str, value: Any) -> None:
    slot = config.find(path)
    if slot is None:
        config.create(path, value)
    else:
        view(slot)[0] = value


def matrix_from_position_rotation(position: list[float], rotation_deg: list[float]) -> Matrix4x4:
    rx, ry, rz = [math.radians(float(v)) for v in rotation_deg]

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot = rz_mat @ ry_mat @ rx_mat

    transform = Matrix4x4.Identity()
    for r in range(3):
        for c in range(3):
            transform[r, c] = float(rot[r, c])
    transform[0, 3] = float(position[0])
    transform[1, 3] = float(position[1])
    transform[2, 3] = float(position[2])
    return transform


def configure_solver(config: Any, variant: str, workspace: Path) -> None:
    spec = VARIANTS[variant]
    config["linear_system"]["solver"] = spec["solver"]
    config["linear_system"]["tol_rate"] = 1e-3

    if spec["solver"] != "socu_approx":
        return

    socu = config["linear_system"]["socu_approx"]
    socu["ordering_source"] = "init_time"
    socu["ordering_block_size"] = "64"
    socu["damping_shift"] = float(os.environ.get("SOCU_DAMPING_SHIFT", "0.0"))
    socu["native_diag_rhs"] = 1 if os.environ.get("SOCU_NATIVE_DIAG_RHS") == "1" else 0
    socu["debug_compare_native_diag_rhs"] = (
        1 if os.environ.get("SOCU_NATIVE_DIAG_RHS_DIFF") == "1" else 0
    )
    socu["native_chain_base_hessian"] = (
        1 if os.environ.get("SOCU_NATIVE_CHAIN_BASE") == "1" else 0
    )
    socu["debug_compare_native_chain_base_hessian"] = (
        1 if os.environ.get("SOCU_NATIVE_CHAIN_BASE_DIFF") == "1" else 0
    )
    socu["native_contact_hessian"] = (
        1 if os.environ.get("SOCU_NATIVE_CONTACT") == "1" else 0
    )
    socu["debug_compare_native_contact_hessian"] = (
        1 if os.environ.get("SOCU_NATIVE_CONTACT_DIFF") == "1" else 0
    )
    socu["runtime_reorder_frame_interval"] = int(spec["runtime_interval"])
    socu["runtime_reorder_graph_source"] = spec.get(
        "runtime_graph_source",
        "topology",
    )
    socu["contact_offband_policy"] = spec.get("contact_offband_policy", "drop")
    report_counters = os.environ.get("SOCU_REPORT_COUNTERS", "1") != "0"
    socu["debug_validation"] = 1 if report_counters else 0
    socu["debug_timing"] = 1 if report_counters else 0
    if os.environ.get("SOCU_DEBUG_DUMP"):
        socu["debug_dump_problem_file"] = 1
        socu["debug_dump_structured_matrix"] = 1
        socu["debug_compare_full_sparse"] = 1
        config["extras"]["debug"]["dump_linear_system"] = 1
    if os.environ.get("SOCU_DEBUG_RUNTIME_ORDERING"):
        socu["debug_write_runtime_ordering_report"] = 1
    socu["report_each_solve"] = 1 if report_counters else 0
    socu["generated_ordering_report"] = str(workspace / "socu_approx_ordering.json")
    socu["report"] = str(workspace / "socu_approx_report.json")


def build_mesh(
    src: SimplicialComplex,
    desc: dict[str, Any],
    abd: AffineBodyConstitution,
    default_contact: Any,
) -> SimplicialComplex:
    mesh = src.copy()
    abd.apply_to(mesh, 10.0e6)
    label_surface(mesh)
    label_triangle_orient(mesh)
    default_contact.apply_to(mesh)

    position = desc.get("position", [0.0, 0.0, 0.0])
    rotation = desc.get("rotation", [0.0, 0.0, 0.0])
    view(mesh.transforms())[0] = matrix_from_position_rotation(position, rotation)

    fixed_attr = mesh.instances().find(builtin.is_fixed)
    view(fixed_attr)[0] = 1 if bool(desc.get("is_dof_fixed", False)) else 0
    return mesh


def backend_for_variant(requested_backend: str, variant: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if VARIANTS[variant]["solver"] == "socu_approx":
        return "cuda_mixed_socu"
    return "cuda_mixed"


def build_scene(variant: str, backend: str, workspace: Path) -> tuple[Engine, World]:
    init_cuda_mixed_module_dir(backend)
    Logger.set_level(Logger.Level.Info)

    engine = Engine(backend, str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    contact_enabled = os.environ.get("SOCU_CONTACT_ENABLE", "1") != "0"
    config["contact"]["friction"]["enable"] = contact_enabled
    config["contact"]["enable"] = contact_enabled
    config["contact"]["d_hat"] = 0.01
    config["line_search"]["max_iter"] = 8
    config["collision_detection"]["method"] = "stackless_bvh"
    configure_solver(config, variant, workspace)

    scene = Scene(config)
    if VARIANTS[variant]["solver"] == "socu_approx":
        set_scene_config_path(
            scene.config(),
            "linear_system/socu_approx/contact_offband_policy",
            VARIANTS[variant].get("contact_offband_policy", "drop"),
        )
    if VARIANTS[variant]["solver"] == "socu_approx" and os.environ.get("SOCU_DEBUG_DUMP"):
        scene_config = scene.config()
        set_scene_config_path(scene_config, "linear_system/socu_approx/debug_dump_problem_file", 1)
        set_scene_config_path(scene_config,
                              "linear_system/socu_approx/debug_dump_structured_matrix",
                              1)
        set_scene_config_path(scene_config, "linear_system/socu_approx/debug_compare_full_sparse", 1)
        set_scene_config_path(scene_config, "extras/debug/dump_linear_system", 1)
    abd = AffineBodyConstitution()
    scene.constitution_tabular().insert(abd)
    scene.contact_tabular().default_model(0.01, 20.0e9)
    default_contact = scene.contact_tabular().default_element()

    tetmesh_dir = Path(AssetDir.tetmesh_path())
    this_folder = Path(__file__).resolve().parents[2] / "apps" / "examples" / "wrecking_ball"
    with (this_folder / "wrecking_ball.json").open("r", encoding="utf-8") as f:
        scene_desc = json.load(f)

    io = SimplicialComplexIO(Matrix4x4.Identity())
    meshes = {
        "cube.msh": io.read(str(tetmesh_dir / "cube.msh")),
        "ball.msh": io.read(str(tetmesh_dir / "ball.msh")),
        "link.msh": io.read(str(tetmesh_dir / "link.msh")),
    }
    objects = {
        "cube.msh": scene.objects().create("cubes"),
        "ball.msh": scene.objects().create("balls"),
        "link.msh": scene.objects().create("links"),
    }

    for desc in scene_desc:
        mesh_name = desc["mesh"]
        mesh = build_mesh(meshes[mesh_name], desc, abd, default_contact)
        objects[mesh_name].geometries().create(mesh)

    scene.objects().create("ground").geometries().create(ground(-1.0))

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError(f"wrecking_ball failed to initialize for {variant}")
    world.retrieve()
    return engine, world


def output_dir_for(output_root: Path, backend: str, variant: str) -> Path:
    if backend == "cuda_mixed":
        return output_root / variant
    return output_root / backend / variant


def run_one(variant: str, frames: int, output_root: Path, requested_backend: str) -> dict[str, Any]:
    backend = backend_for_variant(requested_backend, variant)
    out_dir = output_dir_for(output_root, backend, variant)
    workspace = out_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    engine, world = build_scene(variant, backend, workspace)
    stats = SimulationStats()

    frame_times: list[float] = []
    for _ in range(frames):
        frame_start = time.perf_counter()
        world.advance()
        if not world.is_valid():
            raise RuntimeError(f"{variant}: world invalid after advance at frame {world.frame()}")
        world.retrieve()
        stats.collect()
        frame_times.append(time.perf_counter() - frame_start)
        print(f"{variant}: frame {world.frame()}", flush=True)

    _ = engine
    result = {
        "backend": backend,
        "requested_backend": requested_backend,
        "variant": variant,
        "runtime_graph_source": VARIANTS[variant].get(
            "runtime_graph_source",
            "topology",
        ),
        "contact_offband_policy": VARIANTS[variant].get(
            "contact_offband_policy",
            "drop",
        ),
        "frames": frames,
        "final_frame": int(world.frame()),
        "wall_time_s": time.perf_counter() - start,
        "mean_frame_ms": 1000.0 * sum(frame_times) / max(len(frame_times), 1),
        "timer_frames": list(getattr(stats, "_frames", [])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_variants(args: argparse.Namespace, variants: list[str]) -> int:
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for variant in variants:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--variant",
            variant,
            "--frames",
            str(args.frames),
            "--output",
            str(output_root),
            "--backend",
            args.backend,
        ]
        print("RUN", " ".join(cmd))
        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True)
        summary.append({"variant": variant, "returncode": completed.returncode})
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all(row["returncode"] == 0 for row in summary) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["quick", "regression", "all", *VARIANTS.keys()],
        default="quick",
        help=(
            "'quick' runs the legacy smoke sweep, 'regression' runs the current "
            "SOCU diag_lump regression sweep, and 'all' runs every defined variant."
        ),
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(Path("output") / "examples" / "cuda_mixed_wrecking_ball_compare"),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda_mixed", "cuda_mixed_socu"],
        default="auto",
        help=(
            "Backend module to load. 'auto' uses cuda_mixed for fused_pcg and "
            "cuda_mixed_socu for SOCU variants."
        ),
    )
    args = parser.parse_args()

    if args.variant == "quick":
        return run_variants(args, QUICK_VARIANTS)
    if args.variant == "regression":
        return run_variants(args, REGRESSION_VARIANTS)
    if args.variant == "all":
        return run_variants(args, list(VARIANTS.keys()))

    result = run_one(args.variant, args.frames, Path(args.output).resolve(), args.backend)
    print(json.dumps({k: v for k, v in result.items() if k != "timer_frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
