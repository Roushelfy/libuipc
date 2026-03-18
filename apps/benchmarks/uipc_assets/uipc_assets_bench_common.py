#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ID = "MuGdxy/uipc-assets"
DEFAULT_QUALITY_REL_L2_THRESHOLD = 1e-5
DEFAULT_QUALITY_ABS_LINF_THRESHOLD = 5e-4
DEFAULT_PERF_WARNING_PCT = 15.0
BACKEND_DLL_BASENAME = "uipc_backend_cuda_mixed"
LEVEL_ORDER = ("fp64", "path1", "path2", "path3", "path4", "path5", "path6", "path7")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return script_dir().parents[3]


def default_manifest_path() -> Path:
    return script_dir() / "uipc_assets_manifest.json"


def default_output_root() -> Path:
    return repo_root() / "output" / "benchmarks" / "uipc_assets"


def ordered_levels(levels: Iterable[str]) -> List[str]:
    rank = {name: idx for idx, name in enumerate(LEVEL_ORDER)}
    deduped = list(dict.fromkeys(levels))
    return sorted(deduped, key=lambda level: (rank.get(level, len(rank)), level))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SceneSpec:
    name: str
    enabled: bool
    frames_perf: int
    frames_quality: int
    quality_enabled: bool
    contact_enabled: Optional[bool]
    tags: List[str]
    perf_warning_pct: float
    notes: str
    config_overrides: Dict[str, Any]

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "SceneSpec":
        return SceneSpec(
            name=str(obj["name"]),
            enabled=bool(obj.get("enabled", True)),
            frames_perf=int(obj["frames_perf"]),
            frames_quality=int(obj["frames_quality"]),
            quality_enabled=bool(obj.get("quality_enabled", True)),
            contact_enabled=(
                None
                if "contact_enabled" not in obj
                else bool(obj.get("contact_enabled"))
            ),
            tags=list(obj.get("tags", [])),
            perf_warning_pct=float(obj.get("perf_warning_pct", DEFAULT_PERF_WARNING_PCT)),
            notes=str(obj.get("notes", "")),
            config_overrides=dict(obj.get("config_overrides", {})),
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path) -> List[SceneSpec]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return [SceneSpec.from_json(item) for item in raw]


def manifest_hash(specs: Iterable[SceneSpec]) -> str:
    return sha256_json([spec.to_json() for spec in specs])


def enabled_scenes(specs: Iterable[SceneSpec]) -> List[SceneSpec]:
    return [spec for spec in specs if spec.enabled]


def find_scene(specs: Iterable[SceneSpec], name: str) -> SceneSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(f"Scene '{name}' not found in manifest")


def import_uipc_assets() -> Any:
    assets = importlib.import_module("uipc.assets")
    repo_id = getattr(assets, "REPO_ID", None)
    if repo_id != REPO_ID:
        raise RuntimeError(
            f"uipc.assets.REPO_ID mismatch: expected {REPO_ID}, got {repo_id}"
        )
    return assets


def runtime_versions() -> Dict[str, Optional[str]]:
    uipc = importlib.import_module("uipc")
    huggingface_hub = importlib.import_module("huggingface_hub")
    assets = import_uipc_assets()
    return {
        "uipc": getattr(uipc, "__version__", None),
        "huggingface_hub": getattr(huggingface_hub, "__version__", None),
        "uipc_assets_repo_id": getattr(assets, "REPO_ID", None),
    }


def snapshot_asset_path(name: str, revision: str, cache_dir: Path) -> Path:
    assets = import_uipc_assets()
    result = Path(
        assets.asset_path(
            name,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    )
    if not result.is_dir():
        raise FileNotFoundError(f"Asset '{name}' not found in {REPO_ID} at revision {revision}")
    return result


def load_asset_scene(name: str, scene: Any, revision: str, cache_dir: Path) -> Path:
    assets = import_uipc_assets()
    asset_dir = snapshot_asset_path(name, revision=revision, cache_dir=cache_dir)
    assets.load(
        name,
        scene,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    return asset_dir


def dll_name() -> str:
    if os.name == "nt":
        return f"{BACKEND_DLL_BASENAME}.dll"
    if sys.platform == "darwin":
        return f"lib{BACKEND_DLL_BASENAME}.dylib"
    return f"lib{BACKEND_DLL_BASENAME}.so"


def candidate_module_dirs(build_dir: Path, config: str) -> List[Path]:
    return [
        build_dir / config / "bin",
        build_dir / "bin",
        build_dir,
    ]


def find_module_dir(build_dir: Path, config: str) -> Path:
    target = dll_name()
    for candidate in candidate_module_dirs(build_dir, config):
        if (candidate / target).exists():
            return candidate
    for found in build_dir.rglob(target):
        return found.parent
    raise FileNotFoundError(
        f"Cannot find {target} under build dir {build_dir}. "
        f"Please build cuda_mixed first."
    )


def prepend_library_path(env: Dict[str, str], module_dir: Path) -> Dict[str, str]:
    updated = dict(env)
    module_dir_str = str(module_dir)
    if os.name == "nt":
        key = "PATH"
    elif sys.platform == "darwin":
        key = "DYLD_LIBRARY_PATH"
    else:
        key = "LD_LIBRARY_PATH"
    old = updated.get(key, "")
    updated[key] = module_dir_str if not old else module_dir_str + os.pathsep + old
    return updated


def find_recursive_first(root: Path, prefix: str, suffix: str) -> Optional[Path]:
    if not root.exists():
        return None
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith(prefix) and name.endswith(suffix):
            return entry
    return None


def find_recursive_exact(root: Path, filename: str) -> Optional[Path]:
    if not root.exists():
        return None
    for entry in root.rglob(filename):
        if entry.is_file():
            return entry
    return None


def collect_error_metrics(error_jsonl: Path) -> Dict[str, Any]:
    rel_l2_max = 0.0
    abs_linf_max = 0.0
    nan_inf_count = 0
    record_count = 0
    records: List[Dict[str, Any]] = []

    if not error_jsonl.exists():
        raise FileNotFoundError(f"Missing error.jsonl: {error_jsonl}")

    for line in error_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rel_l2 = float(obj.get("rel_l2_x", 0.0))
        abs_linf = float(obj.get("abs_linf_x", 0.0))
        nan_inf = bool(obj.get("nan_inf_flag", False))
        rel_l2_max = max(rel_l2_max, rel_l2)
        abs_linf_max = max(abs_linf_max, abs_linf)
        nan_inf_count += 1 if nan_inf else 0
        record_count += 1
        records.append(obj)

    return {
        "rel_l2_max": rel_l2_max,
        "abs_linf_max": abs_linf_max,
        "nan_inf_count": nan_inf_count,
        "record_count": record_count,
        "records": records,
    }


def load_benchmark_meta(result_path: Path) -> Dict[str, Any]:
    benchmark_json = result_path
    if result_path.is_dir():
        benchmark_json = result_path / "benchmark.json"
        if not benchmark_json.exists():
            found = find_recursive_exact(result_path, "benchmark.json")
            if found is not None:
                benchmark_json = found
    if not benchmark_json.exists():
        raise FileNotFoundError(f"Missing benchmark.json: {benchmark_json}")
    return read_json(benchmark_json)


def load_timer_frames(result_path: Path) -> List[Dict[str, Any]]:
    timer_path = result_path
    if result_path.is_dir():
        timer_path = result_path / "timer_frames.json"
        if not timer_path.exists():
            found = find_recursive_exact(result_path, "timer_frames.json")
            if found is not None:
                timer_path = found
    if not timer_path.exists():
        return []
    data = read_json(timer_path)
    if not isinstance(data, list):
        return []
    return data


def _accumulate_timer(node: Dict[str, Any], totals: Dict[str, Dict[str, float]]) -> None:
    name = node.get("name")
    if name:
        slot = totals.setdefault(name, {"duration": 0.0, "count": 0.0, "hits": 0.0})
        slot["duration"] += float(node.get("duration", 0.0))
        slot["count"] += float(node.get("count", 0.0))
        slot["hits"] += 1.0
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            _accumulate_timer(child, totals)


def timer_hotspots(timer_frames: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    totals: Dict[str, Dict[str, float]] = {}
    for frame in timer_frames:
        if isinstance(frame, dict):
            _accumulate_timer(frame, totals)

    rows: List[Dict[str, Any]] = []
    for name, stat in totals.items():
        if name == "GlobalTimer":
            continue
        avg_ms = 0.0
        if stat["hits"] > 0:
            avg_ms = stat["duration"] / stat["hits"] * 1000.0
        rows.append(
            {
                "name": name,
                "avg_ms": avg_ms,
                "total_ms": stat["duration"] * 1000.0,
                "count_sum": stat["count"],
                "hits": int(stat["hits"]),
            }
        )

    rows.sort(key=lambda item: item["avg_ms"], reverse=True)
    return rows[:top_n]


def avg_ms_per_frame(benchmark_meta: Dict[str, Any]) -> Optional[float]:
    wall = benchmark_meta.get("wall_time")
    frames = benchmark_meta.get("num_frames")
    if wall is None or frames in (None, 0):
        return None
    return float(wall) / float(frames) * 1000.0


def warning_for_quality(metrics: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if metrics["rel_l2_max"] > DEFAULT_QUALITY_REL_L2_THRESHOLD:
        warnings.append(
            f"rel_l2_x.max {metrics['rel_l2_max']:.6e} > {DEFAULT_QUALITY_REL_L2_THRESHOLD:.1e}"
        )
    if metrics["abs_linf_max"] > DEFAULT_QUALITY_ABS_LINF_THRESHOLD:
        warnings.append(
            f"abs_linf_x.max {metrics['abs_linf_max']:.6e} > {DEFAULT_QUALITY_ABS_LINF_THRESHOLD:.1e}"
        )
    if metrics["nan_inf_count"] > 0:
        warnings.append(f"nan_inf_count={metrics['nan_inf_count']}")
    return warnings
