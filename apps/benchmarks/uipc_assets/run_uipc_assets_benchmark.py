#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_uipc_assets_benchmark import aggregate_run
from uipc_assets_bench_common import (
    REPO_ID,
    SceneSpec,
    default_manifest_path,
    default_output_root,
    enabled_scenes,
    find_module_dir,
    find_recursive_exact,
    find_recursive_first,
    find_scene,
    load_asset_scene,
    load_manifest,
    make_run_id,
    manifest_hash,
    now_utc_iso,
    ordered_levels,
    prepend_library_path,
    read_json,
    runtime_versions,
    write_json,
)


def log(message: str) -> None:
    print(f"[uipc_assets_bench] {message}")


def ensure_runtime_dependencies(require_uipc: bool = True) -> None:
    missing: List[str] = []
    try:
        importlib.import_module("huggingface_hub")
    except ImportError:
        missing.append("huggingface_hub")

    if require_uipc:
        try:
            importlib.import_module("uipc")
        except ImportError:
            missing.append("uipc/pyuipc")

    if missing:
        hints = []
        if "huggingface_hub" in missing:
            hints.append("python -m pip install huggingface_hub")
        if "uipc/pyuipc" in missing:
            hints.append("build/install pyuipc with UIPC_BUILD_PYBIND=ON")
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + "\n"
            + "\n".join(hints)
        )


def fetch_remote_dataset_info() -> Dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(REPO_ID)
    last_modified = getattr(info, "lastModified", None)
    if last_modified is not None:
        last_modified = str(last_modified)
    return {
        "repo_id": REPO_ID,
        "remote_sha": getattr(info, "sha", None),
        "last_modified": last_modified,
    }


def sync_assets(manifest_specs: List[SceneSpec], cache_dir: Path, state_file: Path) -> Dict[str, Any]:
    dataset_info = fetch_remote_dataset_info()
    remote_sha = dataset_info["remote_sha"]
    current_manifest_hash = manifest_hash(manifest_specs)
    versions = runtime_versions()

    old_state: Dict[str, Any] = {}
    if state_file.exists():
        old_state = read_json(state_file)

    needs_download = (
        not state_file.exists()
        or old_state.get("remote_sha") != remote_sha
        or old_state.get("manifest_hash") != current_manifest_hash
    )

    assets: Dict[str, str] = {}
    if needs_download:
        log("asset cache is missing/stale; syncing manifest assets")
    else:
        log("asset cache is up-to-date; reusing local cache")

    for spec in manifest_specs:
        from uipc_assets_bench_common import snapshot_asset_path

        path = snapshot_asset_path(spec.name, revision=remote_sha, cache_dir=cache_dir)
        assets[spec.name] = str(path)

    state = {
        "repo_id": REPO_ID,
        "remote_sha": remote_sha,
        "last_modified": dataset_info["last_modified"],
        "checked_at": now_utc_iso(),
        "manifest_hash": current_manifest_hash,
        "runtime_versions": versions,
        "assets": assets,
    }
    write_json(state_file, state)
    return state


def normalize_config_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def worker_set_scene_config(scene: Any, path: str, value: Any) -> None:
    import uipc

    value = normalize_config_value(value)
    config = scene.config()
    slot = config.find(path)
    if slot is None:
        config.create(path, value)
        return
    uipc.view(slot)[0] = value


def run_profile_worker(
    *,
    scene_name: str,
    manifest_path: Path,
    mode: str,
    level: str,
    module_dir: Path,
    output_dir: Path,
    revision: str,
    cache_dir: Path,
    dump_surface: bool,
    reference_dir: Optional[Path],
) -> Dict[str, Any]:
    import uipc
    from uipc import Scene
    from uipc.profile import run as profile_run

    specs = load_manifest(manifest_path)
    spec = find_scene(specs, scene_name)

    cfg = uipc.default_config()
    cfg["module_dir"] = str(module_dir)
    uipc.init(cfg)

    scene = Scene(Scene.default_config())
    load_asset_scene(scene_name, scene, revision=revision, cache_dir=cache_dir)

    if spec.contact_enabled is not None:
        worker_set_scene_config(scene, "contact/enable", spec.contact_enabled)
    for key, value in spec.config_overrides.items():
        worker_set_scene_config(scene, key, value)

    worker_set_scene_config(scene, "extras/debug/dump_surface", dump_surface)

    if mode in ("quality_reference", "quality_compare"):
        worker_set_scene_config(scene, "extras/debug/dump_solution_x", 1)
    if mode == "quality_compare":
        if reference_dir is None:
            raise RuntimeError("quality_compare requires --reference_dir")

    frames = spec.frames_perf if mode == "perf" else spec.frames_quality
    result = profile_run(
        scene,
        num_frames=frames,
        name=scene_name,
        output_dir=str(output_dir),
        backend="cuda_mixed",
    )

    workspace = Path(result["workspace"]).resolve()
    benchmark_json = find_recursive_exact(output_dir, "benchmark.json")
    timer_frames_json = find_recursive_exact(output_dir, "timer_frames.json")
    worker_result: Dict[str, Any] = {
        "scene": scene_name,
        "mode": mode,
        "level": level,
        "module_dir": str(module_dir),
        "output_dir": str(output_dir),
        "workspace": str(workspace),
        "frames": frames,
        "benchmark_json": str(benchmark_json) if benchmark_json is not None else None,
        "timer_frames_json": str(timer_frames_json) if timer_frames_json is not None else None,
        "solution_dump_dir": None,
        "reference_dir": None,
    }

    if mode in ("quality_reference", "quality_compare"):
        x_dump = find_recursive_first(workspace, "x.", ".mtx")
        if x_dump is None:
            raise FileNotFoundError(f"Solution x dump not found under {workspace}")
        worker_result["solution_dump_dir"] = str(x_dump.parent)
        if mode == "quality_reference":
            worker_result["reference_dir"] = str(x_dump.parent)

    write_json(output_dir / "worker_result.json", worker_result)
    return worker_result


def run_worker_subprocess(
    *,
    python_exe: str,
    script_path: Path,
    scene_name: str,
    manifest_path: Path,
    mode: str,
    level: str,
    module_dir: Path,
    output_dir: Path,
    revision: str,
    cache_dir: Path,
    dump_surface: bool,
    reference_dir: Optional[Path],
) -> Dict[str, Any]:
    env = prepend_library_path(os.environ.copy(), module_dir)
    cmd = [
        python_exe,
        str(script_path),
        "_worker",
        "--scene",
        scene_name,
        "--manifest",
        str(manifest_path),
        "--mode",
        mode,
        "--level",
        level,
        "--module_dir",
        str(module_dir),
        "--output_dir",
        str(output_dir),
        "--revision",
        revision,
        "--cache_dir",
        str(cache_dir),
        "--dump_surface",
        "ON" if dump_surface else "OFF",
    ]
    if reference_dir is not None:
        cmd.extend(["--reference_dir", str(reference_dir)])
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(SCRIPT_DIR.parents[2]))
    return read_json(output_dir / "worker_result.json")


def snapshot_manifest(run_root: Path, specs: List[SceneSpec], state: Dict[str, Any]) -> None:
    write_json(run_root / "manifest_snapshot.json", [spec.to_json() for spec in specs])
    write_json(run_root / "dataset_state.json", state)


def default_build_dir_for(level: str) -> Path:
    return SCRIPT_DIR.parents[2] / f"build_impl_{level}"


def do_run(args: argparse.Namespace) -> int:
    ensure_runtime_dependencies(require_uipc=True)

    manifest_path = args.manifest.resolve()
    specs = enabled_scenes(load_manifest(manifest_path))
    if args.scenes:
        selected = set(args.scenes)
        specs = [spec for spec in specs if spec.name in selected]
    if not specs:
        raise RuntimeError("No enabled scenes selected from manifest")

    base_root = default_output_root()
    base_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else base_root / "hf_cache"
    state_file = base_root / "dataset_state.json"
    state = sync_assets(specs, cache_dir, state_file)

    run_root = args.run_root.resolve() if args.run_root else base_root / make_run_id("dataset")
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot_manifest(run_root, specs, state)
    manifest_snapshot_path = run_root / "manifest_snapshot.json"

    compare_levels = ordered_levels(args.compare_levels)
    compare_levels = [level for level in compare_levels if level != "fp64"]
    if not compare_levels:
        raise RuntimeError("compare_levels must contain at least one non-fp64 level")

    module_dirs = {"fp64": find_module_dir(args.build_fp64.resolve(), args.config)}
    for level in compare_levels:
        if level == "path1":
            build_dir = args.build_path1.resolve()
        else:
            build_dir = (args.build_root.resolve() / f"build_impl_{level}").resolve()
        module_dirs[level] = find_module_dir(build_dir, args.config)

    for spec in specs:
        fp64_perf_dir = run_root / "fp64" / spec.name / "perf"
        run_worker_subprocess(
            python_exe=sys.executable,
            script_path=Path(__file__).resolve(),
            scene_name=spec.name,
            manifest_path=manifest_snapshot_path,
            mode="perf",
            level="fp64",
            module_dir=module_dirs["fp64"],
            output_dir=fp64_perf_dir,
            revision=state["remote_sha"],
            cache_dir=cache_dir,
            dump_surface=str(args.dump_surface).upper() == "ON",
            reference_dir=None,
        )

        reference_dir: Optional[Path] = None
        if spec.quality_enabled:
            fp64_ref_dir = run_root / "fp64" / spec.name / "quality_reference"
            fp64_ref_result = run_worker_subprocess(
                python_exe=sys.executable,
                script_path=Path(__file__).resolve(),
                scene_name=spec.name,
                manifest_path=manifest_snapshot_path,
                mode="quality_reference",
                level="fp64",
                module_dir=module_dirs["fp64"],
                output_dir=fp64_ref_dir,
                revision=state["remote_sha"],
                cache_dir=cache_dir,
                dump_surface=str(args.dump_surface).upper() == "ON",
                reference_dir=None,
            )
            reference_dir = Path(fp64_ref_result["reference_dir"])

        for level in compare_levels:
            perf_dir = run_root / level / spec.name / "perf"
            run_worker_subprocess(
                python_exe=sys.executable,
                script_path=Path(__file__).resolve(),
                scene_name=spec.name,
                manifest_path=manifest_snapshot_path,
                mode="perf",
                level=level,
                module_dir=module_dirs[level],
                output_dir=perf_dir,
                revision=state["remote_sha"],
                cache_dir=cache_dir,
                dump_surface=str(args.dump_surface).upper() == "ON",
                reference_dir=None,
            )

            if spec.quality_enabled and reference_dir is not None:
                cmp_dir = run_root / level / spec.name / "quality_compare"
                run_worker_subprocess(
                    python_exe=sys.executable,
                    script_path=Path(__file__).resolve(),
                    scene_name=spec.name,
                    manifest_path=manifest_snapshot_path,
                    mode="quality_compare",
                    level=level,
                    module_dir=module_dirs[level],
                    output_dir=cmp_dir,
                    revision=state["remote_sha"],
                    cache_dir=cache_dir,
                    dump_surface=str(args.dump_surface).upper() == "ON",
                    reference_dir=reference_dir,
                )

    aggregate_run(run_root, manifest_snapshot_path, run_root)
    log(f"run finished: {run_root}")
    return 0


def do_sync(args: argparse.Namespace) -> int:
    ensure_runtime_dependencies(require_uipc=True)
    manifest_path = args.manifest.resolve()
    specs = enabled_scenes(load_manifest(manifest_path))
    if args.scenes:
        selected = set(args.scenes)
        specs = [spec for spec in specs if spec.name in selected]
    base_root = default_output_root()
    base_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else base_root / "hf_cache"
    state_file = base_root / "dataset_state.json"
    state = sync_assets(specs, cache_dir, state_file)
    log(f"synced {len(specs)} asset(s) at revision {state['remote_sha']}")
    return 0


def do_report(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_root
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else (run_root / "manifest_snapshot.json")
    )
    if not manifest_path.exists():
        manifest_path = default_manifest_path()
    aggregate_run(run_root, manifest_path, out_dir)
    log(f"report written to {out_dir}")
    return 0


def do_worker(args: argparse.Namespace) -> int:
    ensure_runtime_dependencies(require_uipc=True)
    run_profile_worker(
        scene_name=args.scene,
        manifest_path=args.manifest.resolve(),
        mode=args.mode,
        level=args.level,
        module_dir=args.module_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        revision=args.revision,
        cache_dir=args.cache_dir.resolve(),
        dump_surface=str(args.dump_surface).upper() == "ON",
        reference_dir=args.reference_dir.resolve() if args.reference_dir else None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset-driven benchmark runner for MuGdxy/uipc-assets.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("sync", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=None, help="benchmark manifest")

    p_sync = sub.choices["sync"]
    p_sync.add_argument("--cache_dir", type=Path, default=None, help="HuggingFace asset cache dir")
    p_sync.add_argument("--scenes", nargs="*", default=None, help="optional scene subset")

    p_run = sub.choices["run"]
    p_run.add_argument("--build_fp64", type=Path, default=SCRIPT_DIR.parents[2] / "build_impl_fp64", help="fp64 build dir")
    p_run.add_argument("--build_path1", type=Path, default=SCRIPT_DIR.parents[2] / "build_impl_path1", help="path1 build dir")
    p_run.add_argument("--build_root", type=Path, default=SCRIPT_DIR.parents[2], help="root used to resolve build_impl_<level> for compare levels other than path1")
    p_run.add_argument("--compare_levels", nargs="+", default=["path1"], help="compare levels against fp64, e.g. path1 path2 path3")
    p_run.add_argument("--config", type=str, default="RelWithDebInfo", help="build config containing backend dlls")
    p_run.add_argument("--run_root", type=Path, default=None, help="run output root")
    p_run.add_argument("--cache_dir", type=Path, default=None, help="HuggingFace asset cache dir")
    p_run.add_argument("--scenes", nargs="*", default=None, help="optional scene subset")
    p_run.add_argument("--dump_surface", type=str, choices=("ON", "OFF", "on", "off"), default="OFF", help="enable debug surface obj dumps")

    p_report = sub.choices["report"]
    p_report.add_argument("--run_root", type=Path, required=True, help="existing run root")
    p_report.add_argument("--out_dir", type=Path, default=None, help="summary output dir")

    p_worker = sub.add_parser("_worker")
    p_worker.add_argument("--scene", required=True)
    p_worker.add_argument("--manifest", type=Path, required=True)
    p_worker.add_argument("--mode", choices=("perf", "quality_reference", "quality_compare"), required=True)
    p_worker.add_argument("--level", required=True)
    p_worker.add_argument("--module_dir", type=Path, required=True)
    p_worker.add_argument("--output_dir", type=Path, required=True)
    p_worker.add_argument("--revision", required=True)
    p_worker.add_argument("--cache_dir", type=Path, required=True)
    p_worker.add_argument("--dump_surface", choices=("ON", "OFF", "on", "off"), default="OFF")
    p_worker.add_argument("--reference_dir", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "manifest", None) is None and args.command in {"sync", "run"}:
        args.manifest = default_manifest_path()

    if args.command == "sync":
        return do_sync(args)
    if args.command == "run":
        return do_run(args)
    if args.command == "report":
        return do_report(args)
    if args.command == "_worker":
        return do_worker(args)
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
