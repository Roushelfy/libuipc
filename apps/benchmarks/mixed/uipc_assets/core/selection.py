from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .artifacts import script_dir
from .manifest import AssetSpec, enabled_specs, load_manifest, merge_manifest_specs

REPO_ID = "MuGdxy/uipc-assets"


def default_manifest_dir() -> Path:
    return script_dir() / "manifests"


def default_manifest_path() -> Path:
    return default_manifest_dir() / "default.json"


def full_manifest_path() -> Path:
    return default_manifest_dir() / "full.json"


def assets_catalog_path() -> Path:
    return default_manifest_dir() / "assets_catalog.json"


def list_remote_assets(revision: str = "main", local_repo: Path | None = None) -> List[str]:
    from huggingface_hub import HfApi, RepoFolder

    if local_repo is not None:
        assets_root = local_repo / "assets"
        if not assets_root.is_dir():
            raise FileNotFoundError(f"Local assets directory not found: {assets_root}")
        return sorted(d.name for d in assets_root.iterdir() if d.is_dir() and (d / "scene.py").exists())

    api = HfApi()
    entries = api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="assets", revision=revision)
    return sorted(
        entry.path.removeprefix("assets/")
        for entry in entries
        if isinstance(entry, RepoFolder)
    )


def load_assets_catalog(path: Path | None = None) -> List[AssetSpec]:
    return load_manifest(path or assets_catalog_path())


def load_assets_catalog_map(path: Path | None = None) -> Dict[str, AssetSpec]:
    return {spec.name: spec for spec in load_assets_catalog(path)}


def validate_catalog_coverage(
    *,
    revision: str = "main",
    local_repo: Path | None = None,
    catalog_path: Path | None = None,
) -> Dict[str, List[str]]:
    remote_assets = list_remote_assets(revision=revision, local_repo=local_repo)
    catalog_names = set(load_assets_catalog_map(catalog_path).keys())
    missing = sorted(name for name in remote_assets if name not in catalog_names)
    extra = sorted(name for name in catalog_names if name not in set(remote_assets))
    return {"remote_assets": remote_assets, "missing": missing, "extra": extra}


def _match_tags(spec: AssetSpec, required_tags: Sequence[str]) -> bool:
    if not required_tags:
        return True
    tags = set(spec.tags or [])
    return all(tag in tags for tag in required_tags)


def _match_scenarios(spec: AssetSpec, required_scenarios: Sequence[str]) -> bool:
    if not required_scenarios:
        return True
    return spec.scenario in set(required_scenarios)


def _match_scenario_families(spec: AssetSpec, required_families: Sequence[str]) -> bool:
    if not required_families:
        return True
    return spec.scenario_family in set(required_families)


def _load_manifest_specs_with_catalog_overrides(
    manifest_path: Path,
    catalog: Dict[str, AssetSpec],
) -> List[AssetSpec]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")

    specs: List[AssetSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entries must be objects: {manifest_path}")
        name = str(item["name"])
        if name not in catalog:
            raise RuntimeError(f"Manifest asset is missing from assets catalog: {name}")
        merged = catalog[name].to_json()
        merged.update(item)
        specs.append(AssetSpec.from_json(merged))
    return specs


def resolve_asset_specs(
    *,
    manifest_paths: Sequence[Path],
    scene_names: Sequence[str],
    tags: Sequence[str],
    scenarios: Sequence[str],
    scenario_families: Sequence[str],
    select_all: bool,
    revision: str = "main",
    local_repo: Path | None = None,
) -> List[AssetSpec]:
    catalog = load_assets_catalog_map()
    coverage = validate_catalog_coverage(revision=revision, local_repo=local_repo)
    if coverage["missing"] or coverage["extra"]:
        problems = []
        if coverage["missing"]:
            problems.append("uncatalogued remote assets: " + ", ".join(coverage["missing"]))
        if coverage["extra"]:
            problems.append("catalog-only assets missing remotely: " + ", ".join(coverage["extra"]))
        raise RuntimeError("; ".join(problems))

    if select_all:
        candidates = enabled_specs(catalog.values())
    else:
        manifest_specs: List[AssetSpec] = []
        for path in manifest_paths:
            manifest_specs.extend(enabled_specs(_load_manifest_specs_with_catalog_overrides(path, catalog)))
        merged = merge_manifest_specs(manifest_specs, [])
        candidates = list(enabled_specs(merged.values()))

    if scene_names:
        names = set(scene_names)
        candidates = [spec for spec in candidates if spec.name in names]
    if tags:
        candidates = [spec for spec in candidates if _match_tags(spec, tags)]
    if scenarios:
        candidates = [spec for spec in candidates if _match_scenarios(spec, scenarios)]
    if scenario_families:
        candidates = [spec for spec in candidates if _match_scenario_families(spec, scenario_families)]

    if not candidates:
        raise RuntimeError("No assets matched the current selection")

    candidates.sort(key=lambda spec: spec.name)
    return candidates


def selection_payload(
    specs: Iterable[AssetSpec],
    *,
    manifest_paths: Sequence[Path],
    scene_names: Sequence[str],
    tags: Sequence[str],
    scenarios: Sequence[str],
    scenario_families: Sequence[str],
    select_all: bool,
) -> Dict[str, object]:
    return {
        "manifests": [str(path) for path in manifest_paths],
        "scene_filters": list(scene_names),
        "tag_filters": list(tags),
        "scenario_filters": list(scenarios),
        "scenario_family_filters": list(scenario_families),
        "all": bool(select_all),
        "assets": [spec.to_json() for spec in specs],
    }
