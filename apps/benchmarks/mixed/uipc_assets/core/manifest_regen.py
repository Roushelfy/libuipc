from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .manifest import AssetSpec, load_manifest, save_manifest

CURATED_MANIFESTS = ("default.json", "core.json", "smoke.json")
RENAMED_SOURCE_MANIFESTS = {
    "rigid_ipc.json": "source_rigid_ipc",
    "rigid_ipc_fracture.json": "rigid_ipc_fracture",
    "rigid_ipc_gear.json": "rigid_ipc_mechanisms_gears",
}
REMOVED_MANIFESTS = ("rigid.json", "rigid_fracture.json", "rigid_gear.json")


@dataclass(frozen=True)
class SceneClassification:
    name: str
    scenario: str
    scenario_family: str
    tags: tuple[str, ...]
    notes: str
    contact_enabled: bool


def _read_scene_text(scene_path: Path) -> str:
    return scene_path.read_bytes().decode("utf-8", errors="ignore")


def _extract_module_docstring(text: str) -> str:
    try:
        module = ast.parse(text)
    except SyntaxError:
        module = None
    if module is not None:
        doc = ast.get_docstring(module, clean=False)
        if doc:
            return doc
    match = re.search(r'^\s*(?P<quote>"""|\'\'\')(?P<body>.*?)(?P=quote)', text, flags=re.S)
    return "" if not match else match.group("body")


def _extract_source_line(text: str) -> str:
    match = re.search(r"^Source:\s*(.+)$", text, flags=re.M)
    if not match:
        raise RuntimeError("scene.py is missing a Source: line")
    return match.group(1).strip()


def _normalize_token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\.[a-z0-9]+$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _normalize_source(source: str) -> str:
    normalized = source.strip().replace("\\", "/")
    normalized = re.sub(r"^adapted from\s+", "", normalized, flags=re.I)
    normalized = normalized.replace("rigid-ipc fixtures/3D/", "rigid-ipc/3D/")
    normalized = normalized.replace("rigid-ipc 3D/", "rigid-ipc/3D/")
    return normalized


def _derive_scenario(flags: Mapping[str, bool]) -> str:
    constituents = [name for name in ("abd", "fem", "particle") if flags[name]]
    if len(constituents) == 1:
        return constituents[0]
    if len(constituents) >= 2:
        return "coupling"
    raise RuntimeError("scene.py does not contain a recognized physical type")


def _derive_source_tag(normalized_source: str) -> str:
    lower = normalized_source.lower()
    if "ipc-sim/rigid-ipc/3d/" in lower:
        return "source_rigid_ipc"
    if "ipc-sim/ipc input/tutorialexamples/" in lower:
        return "source_ipc_tutorial"
    if lower.startswith("libuipc apps/examples/"):
        return "source_libuipc_example"
    if re.match(r"^libuipc test_[^/]+", lower):
        return "source_libuipc_test"
    raise RuntimeError(f"Unsupported Source: {normalized_source}")


def _derive_scenario_family(normalized_source: str) -> str:
    lower = normalized_source.lower()
    if lower.startswith("libuipc apps/examples/"):
        return "libuipc_example"
    if re.match(r"^libuipc test_[^/]+", lower):
        return "libuipc_test"
    if "ipc-sim/ipc input/tutorialexamples/" in lower:
        return "ipc_tutorial"

    prefix = "ipc-sim/rigid-ipc/3d/"
    if prefix not in lower:
        raise RuntimeError(f"Unsupported Source: {normalized_source}")

    suffix = normalized_source[lower.index(prefix) + len(prefix) :]
    raw_parts = [part for part in suffix.split("/") if part]
    if not raw_parts:
        raise RuntimeError(f"Unable to derive scenario_family from Source: {normalized_source}")

    clean_first = _normalize_token(raw_parts[0])
    tokens = [clean_first]
    include_second = (
        len(raw_parts) >= 2
        and Path(raw_parts[1]).suffix == ""
        and clean_first in {"unit_tests", "friction", "mechanisms"}
    )
    if include_second:
        tokens.append(_normalize_token(raw_parts[1]))
    return "rigid_ipc_" + "_".join(tokens)


def classify_scene_file(scene_path: Path) -> SceneClassification:
    text = _read_scene_text(scene_path)
    normalized_source = _normalize_source(_extract_source_line(text))
    flags = {
        "fem": "StableNeoHookean" in text,
        "abd": "AffineBodyConstitution" in text or "AffineBodyShell" in text,
        "particle": bool(re.search(r"\bParticle\b", text)),
    }
    scenario = _derive_scenario(flags)
    scenario_family = _derive_scenario_family(normalized_source)
    source_tag = _derive_source_tag(normalized_source)

    tags: List[str] = [scenario]
    for name in ("abd", "fem", "particle"):
        if flags[name]:
            tags.append(f"contains_{name}")
    tags.append(source_tag)

    contact_enabled = "contact_tabular()" in text
    if contact_enabled:
        tags.append("contact")
    if "scene.animator()" in text:
        tags.append("animated")
    if "RotatingMotor" in text:
        tags.append("rotating_motor")
    tags.append(scenario_family)

    notes = ""
    module_doc = _extract_module_docstring(text)
    if module_doc:
        notes = module_doc.strip().splitlines()[0].strip()

    return SceneClassification(
        name=scene_path.parent.name,
        scenario=scenario,
        scenario_family=scenario_family,
        tags=tuple(dict.fromkeys(tag for tag in tags if tag)),
        notes=notes,
        contact_enabled=contact_enabled,
    )


def classify_assets(assets_root: Path) -> Dict[str, SceneClassification]:
    assets_root = assets_root.resolve()
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Assets root not found: {assets_root}")

    classifications: Dict[str, SceneClassification] = {}
    for asset_dir in sorted(path for path in assets_root.iterdir() if path.is_dir()):
        scene_path = asset_dir / "scene.py"
        if not scene_path.is_file():
            continue
        meta = classify_scene_file(scene_path)
        classifications[meta.name] = meta
    if not classifications:
        raise RuntimeError(f"No assets with scene.py were found under: {assets_root}")
    return classifications


def _runtime_spec_map(manifest_dir: Path) -> Dict[str, AssetSpec]:
    catalog_path = manifest_dir / "assets_catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing manifest catalog: {catalog_path}")
    return {spec.name: spec for spec in load_manifest(catalog_path)}


def _base_order(runtime_specs: Mapping[str, AssetSpec]) -> List[str]:
    return list(runtime_specs.keys())


def _merged_spec(runtime_spec: AssetSpec, scene_meta: SceneClassification) -> AssetSpec:
    return AssetSpec(
        name=runtime_spec.name,
        enabled=runtime_spec.enabled,
        scenario=scene_meta.scenario,
        scenario_family=scene_meta.scenario_family,
        frames_perf=runtime_spec.frames_perf,
        frames_quality=runtime_spec.frames_quality,
        quality_enabled=runtime_spec.quality_enabled,
        contact_enabled=scene_meta.contact_enabled,
        tags=list(scene_meta.tags),
        perf_warning_pct=runtime_spec.perf_warning_pct,
        notes=scene_meta.notes,
        config_overrides=dict(runtime_spec.config_overrides or {}),
    )


def _select_specs(
    names: Sequence[str],
    regenerated: Mapping[str, AssetSpec],
) -> List[AssetSpec]:
    return [regenerated[name] for name in names]


def _load_curated_names(manifest_dir: Path, filename: str) -> List[str]:
    path = manifest_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing curated manifest: {path}")
    return [spec.name for spec in load_manifest(path)]


def build_manifest_payloads(
    manifest_dir: Path,
    assets_root: Path,
) -> Dict[str, List[AssetSpec]]:
    runtime_specs = _runtime_spec_map(manifest_dir)
    classifications = classify_assets(assets_root)

    runtime_names = set(runtime_specs)
    scene_names = set(classifications)
    missing_from_scenes = sorted(runtime_names - scene_names)
    extra_from_scenes = sorted(scene_names - runtime_names)
    if missing_from_scenes or extra_from_scenes:
        problems = []
        if missing_from_scenes:
            problems.append("manifest assets missing scene.py: " + ", ".join(missing_from_scenes))
        if extra_from_scenes:
            problems.append("uncatalogued scene.py assets: " + ", ".join(extra_from_scenes))
        raise RuntimeError("; ".join(problems))

    ordered_names = _base_order(runtime_specs)
    regenerated = {
        name: _merged_spec(runtime_specs[name], classifications[name])
        for name in ordered_names
    }

    payloads: Dict[str, List[AssetSpec]] = {
        "assets_catalog.json": _select_specs(ordered_names, regenerated),
        "full.json": _select_specs(ordered_names, regenerated),
    }

    scenario_to_file = {
        "abd": "abd.json",
        "fem": "fem.json",
        "coupling": "coupling.json",
        "particle": "particle.json",
    }
    for scenario, filename in scenario_to_file.items():
        names = [name for name in ordered_names if regenerated[name].scenario == scenario]
        payloads[filename] = _select_specs(names, regenerated)

    source_tag = RENAMED_SOURCE_MANIFESTS["rigid_ipc.json"]
    payloads["rigid_ipc.json"] = [
        regenerated[name] for name in ordered_names if source_tag in set(regenerated[name].tags or [])
    ]
    for filename, family in RENAMED_SOURCE_MANIFESTS.items():
        if filename == "rigid_ipc.json":
            continue
        payloads[filename] = [
            regenerated[name] for name in ordered_names if regenerated[name].scenario_family == family
        ]

    for filename in CURATED_MANIFESTS:
        curated_names = _load_curated_names(manifest_dir, filename)
        missing = [name for name in curated_names if name not in regenerated]
        if missing:
            raise RuntimeError(f"Curated manifest {filename} contains unknown assets: {', '.join(missing)}")
        payloads[filename] = _select_specs(curated_names, regenerated)

    return payloads


def write_manifest_payloads(
    manifest_dir: Path,
    payloads: Mapping[str, Iterable[AssetSpec]],
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for filename, specs in payloads.items():
        save_manifest(manifest_dir / filename, specs)
    for filename in REMOVED_MANIFESTS:
        path = manifest_dir / filename
        if path.exists():
            path.unlink()


def regenerate_manifest_dir(manifest_dir: Path, assets_root: Path) -> Dict[str, List[AssetSpec]]:
    payloads = build_manifest_payloads(manifest_dir=manifest_dir, assets_root=assets_root)
    write_manifest_payloads(manifest_dir, payloads)
    return payloads

