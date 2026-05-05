"""
Reproduce socu_approx failures for:
  - hello_affine_body (perf)
  - fem_bouncing_cubes (perf)
  - rigid_ipc_card_house_2 (perf + quality)

Usage (from repo root):
  LEVEL=fp64
  BUILD_DIR=$PWD/build/build_impl_${LEVEL}
  CONFIG=RelWithDebInfo
  MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

  LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
  PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
  python scripts/debug_socu_repro.py [--dump] [--scenario hello_affine_body|fem_bouncing_cubes|rigid_ipc_card_house_2]
"""

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SOCU_BASE_CONFIG = {
    "linear_system/solver": "socu_approx",
    "linear_system/socu_approx/report_each_solve": 1,
    "linear_system/socu_approx/debug_validation": 1,
}

DUMP_CONFIG = {
    "extras/debug/dump_linear_system": 1,
    "linear_system/socu_approx/debug_dump_structured_matrix": 1,
    "linear_system/socu_approx/debug_dump_problem_file": 1,
    "linear_system/socu_approx/debug_compare_full_sparse": 1,
}

SCENARIOS = {
    "hello_affine_body": {"frames": 5, "warmup": 0},
    "fem_bouncing_cubes": {"frames": 5, "warmup": 0},
    "rigid_ipc_card_house_2": {"frames": 5, "warmup": 0},
}


def set_scene_config(scene, path: str, value):
    import uipc
    config = scene.config()
    slot = config.find(path)
    if isinstance(value, bool):
        value = int(value)
    if slot is None:
        config.create(path, value)
    else:
        uipc.view(slot)[0] = value


def run_scenario(name: str, frames: int, enable_dump: bool, output_root: Path):
    import uipc
    import uipc.assets as assets
    from uipc import Engine, Logger, Scene, World

    Logger.set_level(Logger.Level.Warn)

    print(f"\n{'='*60}")
    print(f"Scenario: {name}  frames={frames}  dump={enable_dump}")
    print(f"{'='*60}")

    scene = Scene(Scene.default_config())

    cache_dir = REPO_ROOT / "output" / "assets_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets.load(name, scene, cache_dir=str(cache_dir))

    for k, v in SOCU_BASE_CONFIG.items():
        set_scene_config(scene, k, v)

    if enable_dump:
        for k, v in DUMP_CONFIG.items():
            set_scene_config(scene, k, v)

    workspace = output_root / name
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        engine = Engine("cuda_mixed", str(workspace))
        world = World(engine)
        world.init(scene)

        for i in range(frames):
            print(f"  frame {i+1}/{frames} ...", end="", flush=True)
            world.advance()
            world.retrieve()
            print(" ok")

        print(f"  PASSED")
        return True

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true", help="Enable matrix dumps")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default=None,
                        help="Run only one scenario")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    # Try to locate and configure module_dir from env or default
    import os
    module_dir = os.environ.get("UIPC_MODULE_DIR", "")
    if not module_dir:
        for candidate in [
            REPO_ROOT / "build/build_impl_fp64/RelWithDebInfo/bin",
            REPO_ROOT / "build/build_impl_fp64/Release/bin",
        ]:
            if candidate.exists():
                module_dir = str(candidate)
                break

    import uipc
    if module_dir:
        cfg = uipc.default_config()
        cfg["module_dir"] = module_dir
        uipc.init(cfg)
        print(f"module_dir: {module_dir}")
    else:
        print("[warn] UIPC_MODULE_DIR not set and no default build found")

    output_root = REPO_ROOT / "output" / "debug_socu_repro"
    output_root.mkdir(parents=True, exist_ok=True)

    names = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    results = {}
    for name in names:
        frames = args.frames
        ok = run_scenario(name, frames, args.dump, output_root)
        results[name] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}: {status}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
