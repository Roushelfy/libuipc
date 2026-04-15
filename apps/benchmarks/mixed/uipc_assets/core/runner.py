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

SEARCH_DIRECTION_INVALID_PREFIX = "search_direction_invalid:"
BENCHMARK_MPLBACKEND = "Agg"
BENCHMARK_LINE_SEARCH_FAIL_CONFIG_PATH = (
    "extras/benchmark/fail_after_consecutive_line_search_max_newton_iters"
)
BENCHMARK_LINE_SEARCH_FAIL_THRESHOLD = 10


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


def _local_visual_filename(obj: Path, local_frame: int) -> str:
    frame_str = obj.stem.removeprefix("scene_surface")
    if frame_str.isdigit():
        return f"scene_surface{local_frame:04d}{obj.suffix}"
    return obj.name


def copy_visual_exports(
    workspace: Path,
    visual_dir: Path,
    frames: set[int] | None = None,
    *,
    source_frame_offset: int = 0,
    frame_count: int | None = None,
) -> Dict[str, Any]:
    exported = []
    for obj in sorted(workspace.rglob("scene_surface*.obj")):
        stem = obj.stem
        frame_str = stem.removeprefix("scene_surface")
        if not frame_str.isdigit():
            continue
        source_frame = int(frame_str)
        if source_frame < source_frame_offset:
            continue
        frame = source_frame - source_frame_offset
        if frame_count is not None and frame >= frame_count:
            continue
        if frames is not None and frame not in frames:
            continue
        frame_dir = visual_dir / f"frame_{frame:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        dst = frame_dir / _local_visual_filename(obj, frame)
        shutil.copy2(obj, dst)
        exported.append({"frame": frame, "source_frame": source_frame, "file": str(dst)})
    manifest = {
        "visual_dir": str(visual_dir),
        "frames_exported": len({row["frame"] for row in exported}),
        "files": exported,
    }
    write_json(visual_dir / "visual_manifest.json", manifest)
    return manifest


def extract_search_direction_message(text: str) -> str | None:
    for line in text.splitlines():
        idx = line.find(SEARCH_DIRECTION_INVALID_PREFIX)
        if idx != -1:
            return line[idx:].strip()
    return None


def _assert_worker_world_valid(world: Any, *, stage: str, frame_hint: int) -> None:
    if world.is_valid():
        return
    raise RuntimeError(f"worker simulation became invalid during {stage} at frame {frame_hint}")


def _update_benchmark_metadata(
    benchmark_json: Path,
    *,
    warmup_frames: int,
    warmup_wall_time_s: float,
    end_to_end_wall_time_s: float,
) -> None:
    if not benchmark_json.exists():
        return
    payload = read_json(benchmark_json)
    payload["warmup_frames"] = warmup_frames
    payload["warmup_wall_time_s"] = warmup_wall_time_s
    payload["end_to_end_wall_time_s"] = end_to_end_wall_time_s
    write_json(benchmark_json, payload)


def _save_worker_profile_result(
    *,
    asset_name: str,
    profile_dir: Path,
    workspace: Path,
    stats: Any,
    wall_time: float,
    warmup_frames: int,
    warmup_wall_time_s: float,
    end_to_end_wall_time_s: float,
) -> Dict[str, Any]:
    # Benchmark workers should always render reports headlessly.
    os.environ["MPLBACKEND"] = BENCHMARK_MPLBACKEND
    try:
        import matplotlib

        matplotlib.use(BENCHMARK_MPLBACKEND, force=True)
    except Exception:
        pass
    import uipc.profile as uipc_profile

    num_frames = getattr(stats, "num_frames", len(getattr(stats, "_frames", [])))
    result = {
        "name": asset_name,
        "num_frames": num_frames,
        "wall_time": wall_time,
        "stats": stats,
        "timer_frames": list(getattr(stats, "_frames", [])),
        "summary": (
            f"Scene: {asset_name}  |  Frames: {num_frames}  |  "
            f"Wall time: {wall_time:.3f}s  |  "
            f"Avg: {wall_time / max(num_frames, 1) * 1000:.1f}ms/frame"
        ),
        "workspace": str(workspace.resolve()),
        "steps": ([("warmup", warmup_frames)] if warmup_frames > 0 else []) + [("profile", num_frames)],
    }
    uipc_profile._save_result(result, str(profile_dir))
    _update_benchmark_metadata(
        profile_dir / "benchmark.json",
        warmup_frames=warmup_frames,
        warmup_wall_time_s=warmup_wall_time_s,
        end_to_end_wall_time_s=end_to_end_wall_time_s,
    )
    result["warmup_frames"] = warmup_frames
    result["warmup_wall_time_s"] = warmup_wall_time_s
    result["end_to_end_wall_time_s"] = end_to_end_wall_time_s
    return result


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
    from uipc import Engine, Logger, Scene, World
    from uipc.stats import SimulationStats

    cfg = uipc.default_config()
    cfg["module_dir"] = str(module_dir)
    uipc.init(cfg)
    Logger.set_level(Logger.Level.Warn)

    scene = Scene(Scene.default_config())
    load_asset_scene(asset_spec.name, scene, revision=revision, cache_dir=cache_dir)

    if asset_spec.contact_enabled is not None:
        worker_set_scene_config(scene, "contact/enable", asset_spec.contact_enabled)
    for key, value in (asset_spec.config_overrides or {}).items():
        worker_set_scene_config(scene, key, value)
    worker_set_scene_config(scene, "extras/debug/dump_surface", dump_surface)
    worker_set_scene_config(
        scene,
        BENCHMARK_LINE_SEARCH_FAIL_CONFIG_PATH,
        BENCHMARK_LINE_SEARCH_FAIL_THRESHOLD,
    )
    if mode == "quality":
        worker_set_scene_config(scene, "extras/debug/dump_solution_x", 1)

    num_frames = asset_spec.frames_perf if mode == "perf" else asset_spec.frames_quality
    warmup_frames = asset_spec.perf_warmup_frames if mode == "perf" else 0
    workspace = output_dir / f"workspace_{asset_spec.name}"
    profile_dir = output_dir / asset_spec.name
    workspace.mkdir(parents=True, exist_ok=True)
    engine = Engine("cuda_mixed", str(workspace))
    world = World(engine)
    stats = SimulationStats()
    run_start = time.perf_counter()
    warmup_start: float | None = None
    profile_start: float | None = None
    warmup_wall_time_s = 0.0
    profile_wall_time_s = 0.0
    pending_error: Exception | None = None
    try:
        world.init(scene)
        _assert_worker_world_valid(world, stage="init", frame_hint=0)
        if warmup_frames > 0:
            warmup_start = time.perf_counter()
            for warmup_index in range(warmup_frames):
                frame_hint = warmup_index + 1
                world.advance()
                _assert_worker_world_valid(world, stage="warmup advance", frame_hint=frame_hint)
                world.retrieve()
                _assert_worker_world_valid(world, stage="warmup retrieve", frame_hint=frame_hint)
            warmup_wall_time_s = time.perf_counter() - warmup_start
        profile_start = time.perf_counter()
        for frame_index in range(num_frames):
            frame_hint = warmup_frames + frame_index + 1
            world.advance()
            _assert_worker_world_valid(world, stage="advance", frame_hint=frame_hint)
            world.retrieve()
            _assert_worker_world_valid(world, stage="retrieve", frame_hint=frame_hint)
            stats.collect()
    except Exception as exc:
        if warmup_start is not None and profile_start is None:
            warmup_wall_time_s = time.perf_counter() - warmup_start
        if profile_start is not None:
            profile_wall_time_s = time.perf_counter() - profile_start
        pending_error = exc
    else:
        if profile_start is not None:
            profile_wall_time_s = time.perf_counter() - profile_start
    end_to_end_wall_time_s = time.perf_counter() - run_start

    result = _save_worker_profile_result(
        asset_name=asset_spec.name,
        profile_dir=profile_dir,
        workspace=workspace,
        stats=stats,
        wall_time=profile_wall_time_s,
        warmup_frames=warmup_frames,
        warmup_wall_time_s=warmup_wall_time_s,
        end_to_end_wall_time_s=end_to_end_wall_time_s,
    )
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
    if pending_error is not None:
        raise pending_error

    worker_result = {
        "asset": asset_spec.name,
        "mode": mode,
        "level": level,
        "module_dir": str(module_dir),
        "output_dir": str(output_dir),
        "profile_dir": str(profile_dir),
        "workspace": str(Path(result["workspace"]).resolve()),
        "frames": num_frames,
        "warmup_frames": warmup_frames,
        "warmup_wall_time_s": warmup_wall_time_s,
        "end_to_end_wall_time_s": end_to_end_wall_time_s,
        "benchmark_json": str(benchmark_json) if benchmark_json.exists() else None,
        "timer_frames_json": str(timer_frames_json) if timer_frames_json.exists() else None,
        "stage_summary_json": str(output_dir / "stage_summary.json"),
        "iteration_summary_json": str(output_dir / "iteration_summary.json"),
        "solution_dump_dir": None,
        "visual_manifest": None,
    }

    if dump_surface:
        worker_result["visual_manifest"] = copy_visual_exports(
            Path(result["workspace"]).resolve(),
            output_dir.parent / "visual",
            visual_frames,
            source_frame_offset=warmup_frames,
            frame_count=num_frames,
        )

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
    env["MPLBACKEND"] = BENCHMARK_MPLBACKEND
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
        search_direction_message = extract_search_direction_message(
            (completed.stderr or "") + "\n" + (completed.stdout or "")
        )
        if search_direction_message is not None:
            raise RuntimeError(search_direction_message)
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
