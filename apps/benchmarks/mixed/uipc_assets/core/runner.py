from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .artifacts import now_utc_iso, read_json, write_json
from .builds import prepend_library_path, prepend_pythonpath
from .manifest import AssetSpec
from .quality import collect_solution_metrics
from .selection import REPO_ID
from .timers import summarize_iteration_counters, summarize_timer_frames


def ensure_runtime_dependencies(require_uipc: bool = True) -> None:
    missing: List[str] = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    if require_uipc:
        try:
            import uipc  # noqa: F401
        except ImportError:
            missing.append("uipc/pyuipc")
    if missing:
        raise RuntimeError("Missing runtime dependencies: " + ", ".join(missing))


def fetch_remote_dataset_info(revision: str | None = None) -> Dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=revision)
    last_modified = getattr(info, "lastModified", None)
    return {
        "repo_id": REPO_ID,
        "remote_sha": getattr(info, "sha", None),
        "last_modified": None if last_modified is None else str(last_modified),
    }


def runtime_versions() -> Dict[str, Optional[str]]:
    import importlib

    huggingface_hub = importlib.import_module("huggingface_hub")
    try:
        uipc = importlib.import_module("uipc")
        uipc_version = getattr(uipc, "__version__", None)
    except Exception:
        uipc_version = None
    return {
        "uipc": uipc_version,
        "huggingface_hub": getattr(huggingface_hub, "__version__", None),
        "dataset_repo": REPO_ID,
    }


def snapshot_asset_path(name: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    dl = snapshot_download(
        REPO_ID,
        allow_patterns=[f"assets/{name}/**", "assets/_*.py"],
        revision=revision,
        cache_dir=str(cache_dir),
        repo_type="dataset",
    )
    path = Path(dl) / "assets" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Asset '{name}' not found at revision {revision}")
    return path


def load_asset_scene(name: str, scene: Any, revision: str, cache_dir: Path) -> Path:
    import uipc.assets as assets

    asset_dir = snapshot_asset_path(name, revision=revision, cache_dir=cache_dir)
    assets.load(name, scene, revision=revision, cache_dir=str(cache_dir))
    return asset_dir


def sync_assets(specs: Iterable[AssetSpec], cache_dir: Path, state_file: Path, revision: str | None = None) -> Dict[str, Any]:
    dataset_info = fetch_remote_dataset_info(revision=revision)
    remote_sha = revision or dataset_info["remote_sha"]
    assets = {spec.name: str(snapshot_asset_path(spec.name, revision=remote_sha, cache_dir=cache_dir)) for spec in specs}
    state = {
        "repo_id": REPO_ID,
        "remote_sha": remote_sha,
        "last_modified": dataset_info["last_modified"],
        "checked_at": now_utc_iso(),
        "runtime_versions": runtime_versions(),
        "assets": assets,
    }
    write_json(state_file, state)
    return state


def load_timer_frames_json(path: Path, retries: int = 5, delay_s: float = 0.2) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    last_error: json.JSONDecodeError | None = None
    last_text = ""
    for attempt in range(retries):
        last_text = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(last_text)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(delay_s)

    debug_path = path.with_suffix(path.suffix + ".invalid")
    debug_path.write_text(last_text, encoding="utf-8")
    print(
        f"warning: failed to parse timer frames JSON after {retries} attempts: {path}. "
        f"Last error: {last_error}. Saved raw content to {debug_path}. Falling back to empty timer frames.",
        file=sys.stderr,
    )
    return []


def normalize_config_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def worker_set_scene_config(scene: Any, path: str, value: Any) -> None:
    import uipc

    config = scene.config()
    slot = config.find(path)
    value = normalize_config_value(value)
    if slot is None:
        config.create(path, value)
    else:
        uipc.view(slot)[0] = value


def copy_visual_exports(workspace: Path, visual_dir: Path, frames: set[int] | None = None) -> Dict[str, Any]:
    exported = []
    for obj in sorted(workspace.rglob("scene_surface*.obj")):
        stem = obj.stem
        frame_str = stem.removeprefix("scene_surface")
        if not frame_str.isdigit():
            continue
        frame = int(frame_str)
        if frames is not None and frame not in frames:
            continue
        frame_dir = visual_dir / f"frame_{frame:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        dst = frame_dir / obj.name
        shutil.copy2(obj, dst)
        exported.append({"frame": frame, "file": str(dst)})
    manifest = {
        "visual_dir": str(visual_dir),
        "frames_exported": len({row["frame"] for row in exported}),
        "files": exported,
    }
    write_json(visual_dir / "visual_manifest.json", manifest)
    return manifest


def run_profile_worker(
    *,
    asset_spec: AssetSpec,
    mode: str,
    level: str,
    module_dir: Path,
    output_dir: Path,
    revision: str,
    cache_dir: Path,
    dump_surface: bool,
    reference_dir: Optional[Path],
    visual_frames: set[int] | None,
) -> Dict[str, Any]:
    import uipc
    from uipc import Scene
    from uipc.profile import run as profile_run

    cfg = uipc.default_config()
    cfg["module_dir"] = str(module_dir)
    uipc.init(cfg)

    scene = Scene(Scene.default_config())
    load_asset_scene(asset_spec.name, scene, revision=revision, cache_dir=cache_dir)

    if asset_spec.contact_enabled is not None:
        worker_set_scene_config(scene, "contact/enable", asset_spec.contact_enabled)
    for key, value in (asset_spec.config_overrides or {}).items():
        worker_set_scene_config(scene, key, value)
    worker_set_scene_config(scene, "extras/debug/dump_surface", dump_surface)
    if mode == "quality":
        worker_set_scene_config(scene, "extras/debug/dump_solution_x", 1)

    num_frames = asset_spec.frames_perf if mode == "perf" else asset_spec.frames_quality
    result = profile_run(scene, num_frames=num_frames, name=asset_spec.name, output_dir=str(output_dir), backend="cuda_mixed")
    profile_dir = output_dir / asset_spec.name
    benchmark_json = profile_dir / "benchmark.json"
    timer_frames_json = profile_dir / "timer_frames.json"
    if benchmark_json.exists():
        shutil.copy2(benchmark_json, output_dir / "benchmark.json")
        benchmark_json = output_dir / "benchmark.json"
    if timer_frames_json.exists():
        shutil.copy2(timer_frames_json, output_dir / "timer_frames.json")
        timer_frames_json = output_dir / "timer_frames.json"
    timer_frames = load_timer_frames_json(timer_frames_json)
    stage_summary = summarize_timer_frames(timer_frames)
    iteration_summary = summarize_iteration_counters(timer_frames)
    write_json(output_dir / "stage_summary.json", stage_summary)
    write_json(output_dir / "iteration_summary.json", iteration_summary)

    worker_result = {
        "asset": asset_spec.name,
        "mode": mode,
        "level": level,
        "module_dir": str(module_dir),
        "output_dir": str(output_dir),
        "profile_dir": str(profile_dir),
        "workspace": str(Path(result["workspace"]).resolve()),
        "frames": num_frames,
        "benchmark_json": str(benchmark_json) if benchmark_json.exists() else None,
        "timer_frames_json": str(timer_frames_json) if timer_frames_json.exists() else None,
        "stage_summary_json": str(output_dir / "stage_summary.json"),
        "iteration_summary_json": str(output_dir / "iteration_summary.json"),
        "solution_dump_dir": None,
        "visual_manifest": None,
    }

    if dump_surface:
        worker_result["visual_manifest"] = copy_visual_exports(Path(result["workspace"]).resolve(), output_dir.parent / "visual", visual_frames)

    if mode == "quality":
        x_dump = next(Path(result["workspace"]).resolve().rglob("x.*.mtx"), None)
        if x_dump is None:
            raise FileNotFoundError(f"Solution x dump not found under {result['workspace']}")
        dump_dir = output_dir / "x_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        for mtx in x_dump.parent.glob("x.*.mtx"):
            shutil.copy2(mtx, dump_dir / mtx.name)
        worker_result["solution_dump_dir"] = str(dump_dir)
        if reference_dir is not None:
            metrics = collect_solution_metrics(reference_dir, dump_dir)
            metrics["reference_dir"] = str(reference_dir)
            metrics["compare_dir"] = str(dump_dir)
            write_json(output_dir / "quality_metrics.json", metrics)

    write_json(output_dir / "worker_result.json", worker_result)
    return worker_result


def run_worker_subprocess(
    *,
    cli_path: Path,
    python_exe: str,
    asset_spec: AssetSpec,
    mode: str,
    level: str,
    module_dir: Path,
    pyuipc_src_dir: Path,
    output_dir: Path,
    revision: str,
    cache_dir: Path,
    dump_surface: bool,
    reference_dir: Path | None,
    visual_frames: set[int] | None,
    resume: bool = False,
) -> Dict[str, Any]:
    env = prepend_library_path(os.environ.copy(), module_dir)
    env = prepend_pythonpath(env, pyuipc_src_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "asset_spec.json"
    result_path = output_dir / "worker_result.json"
    failure_path = output_dir / "failure.json"
    stdout_log = output_dir / "worker_stdout.log"
    stderr_log = output_dir / "worker_stderr.log"
    if resume and result_path.exists():
        if failure_path.exists():
            failure_path.unlink()
        print(f"[resume] skip asset={asset_spec.name} mode={mode} level={level} -> {result_path}")
        return read_json(result_path)
    write_json(spec_path, asset_spec.to_json())
    cmd = [
        python_exe,
        str(cli_path),
        "_worker",
        "--asset_spec",
        str(spec_path),
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
    if visual_frames:
        cmd.extend(["--visual_frames", ",".join(str(i) for i in sorted(visual_frames))])
    completed = subprocess.run(
        cmd,
        check=False,
        env=env,
        cwd=str(cli_path.parents[4]),
        capture_output=True,
        text=True,
    )
    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for asset={asset_spec.name} mode={mode} level={level} "
            f"(exit={completed.returncode}); logs: stdout={stdout_log}, stderr={stderr_log}"
        )
    if failure_path.exists():
        failure_path.unlink()
    return read_json(result_path)


def parse_visual_frames(value: str | None, frame_range: str | None) -> set[int] | None:
    if value is None and frame_range is None:
        return None
    if value == "all" and frame_range is None:
        return None
    frames: set[int] = set()
    if value and value != "all":
        frames.update(int(item) for item in value.split(",") if item.strip())
    if frame_range:
        start_str, end_str = frame_range.split(":", 1)
        start = int(start_str)
        end = int(end_str)
        frames.update(range(start, end + 1))
    return frames
