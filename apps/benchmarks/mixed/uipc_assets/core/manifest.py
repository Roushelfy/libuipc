from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PERF_WARNING_PCT = 15.0
DEFAULT_FRAMES_PERF = 100
DEFAULT_FRAMES_QUALITY = 30
DEFAULT_PERF_WARMUP_FRAMES = 0
LEVEL_ORDER = ("fp64", "path1", "path2", "path3", "path4", "path5", "path6", "path7", "path8")


@dataclass(frozen=True)
class AssetSpec:
    name: str
    enabled: bool = True
    scenario: str = ""
    scenario_family: str = ""
    frames_perf: int = DEFAULT_FRAMES_PERF
    perf_warmup_frames: int = DEFAULT_PERF_WARMUP_FRAMES
    frames_quality: int = DEFAULT_FRAMES_QUALITY
    quality_enabled: bool = True
    contact_enabled: Optional[bool] = None
    tags: List[str] | None = None
    perf_warning_pct: float = DEFAULT_PERF_WARNING_PCT
    notes: str = ""
    config_overrides: Dict[str, Any] | None = None

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "AssetSpec":
        return AssetSpec(
            name=str(obj["name"]),
            enabled=bool(obj.get("enabled", True)),
            scenario=str(obj.get("scenario", "")),
            scenario_family=str(obj.get("scenario_family", "")),
            frames_perf=int(obj.get("frames_perf", DEFAULT_FRAMES_PERF)),
            perf_warmup_frames=int(obj.get("perf_warmup_frames", DEFAULT_PERF_WARMUP_FRAMES)),
            frames_quality=int(obj.get("frames_quality", DEFAULT_FRAMES_QUALITY)),
            quality_enabled=bool(obj.get("quality_enabled", True)),
            contact_enabled=(
                None if "contact_enabled" not in obj else bool(obj.get("contact_enabled"))
            ),
            tags=list(obj.get("tags", [])),
            perf_warning_pct=float(obj.get("perf_warning_pct", DEFAULT_PERF_WARNING_PCT)),
            notes=str(obj.get("notes", "")),
            config_overrides=dict(obj.get("config_overrides", {})),
        )

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags or [])
        data["config_overrides"] = dict(self.config_overrides or {})
        return data


def load_manifest(path: Path) -> List[AssetSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return [AssetSpec.from_json(item) for item in raw]


def save_manifest(path: Path, specs: Iterable[AssetSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_json() for spec in specs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def enabled_specs(specs: Iterable[AssetSpec]) -> List[AssetSpec]:
    return [spec for spec in specs if spec.enabled]


def ordered_levels(levels: Iterable[str]) -> List[str]:
    rank = {name: idx for idx, name in enumerate(LEVEL_ORDER)}
    deduped = list(dict.fromkeys(levels))
    return sorted(deduped, key=lambda level: (rank.get(level, len(rank)), level))


def merge_manifest_specs(manifest_specs: Iterable[AssetSpec], discovered_names: Iterable[str]) -> Dict[str, AssetSpec]:
    merged: Dict[str, AssetSpec] = {spec.name: spec for spec in manifest_specs}
    for name in discovered_names:
        merged.setdefault(name, AssetSpec(name=name))
    return merged
