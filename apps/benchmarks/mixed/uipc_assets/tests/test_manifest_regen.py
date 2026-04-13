from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.manifest import AssetSpec, save_manifest
from uipc_assets.core.manifest_regen import classify_scene_file, regenerate_manifest_dir


def _write_scene(asset_dir: Path, body: str) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "scene.py").write_text(body, encoding="utf-8")


def _seed_manifest_dir(manifest_dir: Path, specs: list[AssetSpec]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest_dir / "assets_catalog.json", specs)
    save_manifest(manifest_dir / "full.json", specs)
    save_manifest(manifest_dir / "default.json", specs[:3])
    save_manifest(manifest_dir / "core.json", specs[1:4])
    save_manifest(manifest_dir / "smoke.json", specs[:2])
    save_manifest(manifest_dir / "abd.json", [])
    save_manifest(manifest_dir / "fem.json", [])
    save_manifest(manifest_dir / "coupling.json", [])
    save_manifest(manifest_dir / "rigid.json", specs)
    save_manifest(manifest_dir / "rigid_fracture.json", [specs[1]])
    save_manifest(manifest_dir / "rigid_gear.json", [specs[2]])


def test_classify_scene_file_examples(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"

    _write_scene(
        assets_root / "abd_fem_tower",
        '''"""ABD-FEM Tower: alternating rigid and soft cubes.

Source: adapted from libuipc test_abd_fem.py
"""
from uipc.constitution import StableNeoHookean, AffineBodyConstitution

def build_scene(scene):
    StableNeoHookean()
    AffineBodyConstitution()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )
    _write_scene(
        assets_root / "particle_rain",
        '''"""Particle Rain: particles falling onto a ground plane.

Source: adapted from libuipc test_particle_ground.py
"""
from uipc.constitution import Particle

def build_scene(scene):
    Particle()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )
    _write_scene(
        assets_root / "rigid_ipc_bat",
        '''"""Rigid IPC Bat: auto-converted from rigid-ipc fixture.

Source: ipc-sim/rigid-ipc fixtures/3D/unit-tests/bat.json
"""
from uipc.constitution import AffineBodyConstitution, RotatingMotor, AffineBodyShell

def build_scene(scene):
    AffineBodyConstitution()
    AffineBodyShell()
    RotatingMotor()
    scene.contact_tabular().default_model(0.5, 1e9)
    scene.animator()
''',
    )
    _write_scene(
        assets_root / "ipc_2cubes_fall_larger_dt",
        '''"""IPC 2 Cubes Fall (larger dt): two cubes falling.

Source: ipc-sim/IPC input/tutorialExamples/2cubesFall_largerDt.txt
"""
from uipc.constitution import StableNeoHookean

def build_scene(scene):
    StableNeoHookean()
    scene.contact_tabular().default_model(0.1, 1e9)
''',
    )

    abd_fem = classify_scene_file(assets_root / "abd_fem_tower" / "scene.py")
    assert abd_fem.scenario == "coupling"
    assert abd_fem.scenario_family == "libuipc_test"
    assert "contains_abd" in abd_fem.tags
    assert "contains_fem" in abd_fem.tags
    assert abd_fem.notes == "ABD-FEM Tower: alternating rigid and soft cubes."

    particle = classify_scene_file(assets_root / "particle_rain" / "scene.py")
    assert particle.scenario == "particle"
    assert particle.scenario_family == "libuipc_test"
    assert "contains_particle" in particle.tags

    rigid_ipc = classify_scene_file(assets_root / "rigid_ipc_bat" / "scene.py")
    assert rigid_ipc.scenario == "abd"
    assert rigid_ipc.scenario_family == "rigid_ipc_unit_tests"
    assert "source_rigid_ipc" in rigid_ipc.tags
    assert "animated" in rigid_ipc.tags
    assert "rotating_motor" in rigid_ipc.tags

    tutorial = classify_scene_file(assets_root / "ipc_2cubes_fall_larger_dt" / "scene.py")
    assert tutorial.scenario == "fem"
    assert tutorial.scenario_family == "ipc_tutorial"
    assert "source_ipc_tutorial" in tutorial.tags


def test_regenerate_manifest_dir_updates_subsets_and_removes_old_aliases(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    manifest_dir = tmp_path / "manifests"

    _write_scene(
        assets_root / "abd_external_force",
        '''"""Affine-body external-force regression scene.

Source: adapted from libuipc test_affine_body_external_force.py
"""
from uipc.constitution import AffineBodyConstitution

def build_scene(scene):
    AffineBodyConstitution()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )
    _write_scene(
        assets_root / "rigid_ipc_fracture_cube",
        '''"""Rigid IPC Cube: auto-converted from rigid-ipc fixture.

Source: ipc-sim/rigid-ipc fixtures/3D/fracture/cube.json
"""
from uipc.constitution import AffineBodyConstitution, AffineBodyShell

def build_scene(scene):
    AffineBodyConstitution()
    AffineBodyShell()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )
    _write_scene(
        assets_root / "rigid_ipc_gear_chain",
        '''"""Rigid IPC Chain: auto-converted from rigid-ipc fixture.

Source: ipc-sim/rigid-ipc 3D/mechanisms/gears/chain.json
"""
from uipc.constitution import AffineBodyConstitution

def build_scene(scene):
    AffineBodyConstitution()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )
    _write_scene(
        assets_root / "abd_fem_tower",
        '''"""ABD-FEM Tower: alternating rigid and soft cubes.

Source: adapted from libuipc test_abd_fem.py
"""
from uipc.constitution import StableNeoHookean, AffineBodyConstitution

def build_scene(scene):
    StableNeoHookean()
    AffineBodyConstitution()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )
    _write_scene(
        assets_root / "ipc_2cubes_fall_larger_dt",
        '''"""IPC 2 Cubes Fall (larger dt): two cubes falling.

Source: ipc-sim/IPC input/tutorialExamples/2cubesFall_largerDt.txt
"""
from uipc.constitution import StableNeoHookean

def build_scene(scene):
    StableNeoHookean()
    scene.contact_tabular().default_model(0.1, 1e9)
''',
    )
    _write_scene(
        assets_root / "particle_rain",
        '''"""Particle Rain: particles falling onto a ground plane.

Source: adapted from libuipc test_particle_ground.py
"""
from uipc.constitution import Particle

def build_scene(scene):
    Particle()
    scene.contact_tabular().default_model(0.5, 1e9)
''',
    )

    specs = [
        AssetSpec(name="abd_external_force", frames_perf=60, frames_quality=25, quality_enabled=True, perf_warning_pct=15.0),
        AssetSpec(name="rigid_ipc_fracture_cube", frames_perf=120, frames_quality=30, quality_enabled=False, perf_warning_pct=25.0),
        AssetSpec(name="rigid_ipc_gear_chain", frames_perf=200, frames_quality=50, quality_enabled=False, perf_warning_pct=25.0),
        AssetSpec(name="abd_fem_tower", frames_perf=60, frames_quality=25, quality_enabled=True, perf_warning_pct=20.0),
        AssetSpec(name="ipc_2cubes_fall_larger_dt", frames_perf=50, frames_quality=20, quality_enabled=True, perf_warning_pct=20.0),
        AssetSpec(name="particle_rain", frames_perf=80, frames_quality=20, quality_enabled=True, perf_warning_pct=15.0),
    ]
    _seed_manifest_dir(manifest_dir, specs)

    payloads = regenerate_manifest_dir(manifest_dir, assets_root)

    assert len(payloads["assets_catalog.json"]) == 6
    assert len(payloads["full.json"]) == 6
    assert [spec.name for spec in payloads["abd.json"]] == [
        "abd_external_force",
        "rigid_ipc_fracture_cube",
        "rigid_ipc_gear_chain",
    ]
    assert [spec.name for spec in payloads["fem.json"]] == ["ipc_2cubes_fall_larger_dt"]
    assert [spec.name for spec in payloads["coupling.json"]] == ["abd_fem_tower"]
    assert [spec.name for spec in payloads["particle.json"]] == ["particle_rain"]

    assert [spec.name for spec in payloads["rigid_ipc.json"]] == [
        "rigid_ipc_fracture_cube",
        "rigid_ipc_gear_chain",
    ]
    assert [spec.name for spec in payloads["rigid_ipc_fracture.json"]] == ["rigid_ipc_fracture_cube"]
    assert [spec.name for spec in payloads["rigid_ipc_gear.json"]] == ["rigid_ipc_gear_chain"]

    assert [spec.name for spec in payloads["default.json"]] == [
        "abd_external_force",
        "rigid_ipc_fracture_cube",
        "rigid_ipc_gear_chain",
    ]
    assert [spec.name for spec in payloads["core.json"]] == [
        "rigid_ipc_fracture_cube",
        "rigid_ipc_gear_chain",
        "abd_fem_tower",
    ]
    assert [spec.name for spec in payloads["smoke.json"]] == [
        "abd_external_force",
        "rigid_ipc_fracture_cube",
    ]

    assert not (manifest_dir / "rigid.json").exists()
    assert not (manifest_dir / "rigid_fracture.json").exists()
    assert not (manifest_dir / "rigid_gear.json").exists()
    assert (manifest_dir / "particle.json").exists()
