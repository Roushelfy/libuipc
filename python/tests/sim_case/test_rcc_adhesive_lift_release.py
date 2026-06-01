from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("polyscope")

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def load_example(filename: str):
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def advance_to(sim, frame: int):
    while sim["world"].frame() < frame:
        sim["world"].advance()
        assert sim["world"].is_valid(), f"world invalid at frame {sim['world'].frame()}"
        sim["world"].retrieve()


@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.example
def test_rcc_subdivided_cube_lift_hold_release():
    demo = load_example("rcc_adhesive_subdivided_cube_lift_release_demo.py")
    sim = demo.build_demo(adhesion_on=True)

    advance_to(sim, demo.PRE_PULL_HOLD_UNTIL)
    bottom_y, _top_y, gap_y = demo.cube_height_stats(sim)
    contact = demo.cube_contact_gap_stats(sim)
    beta = demo.adhesion_beta_stats(sim)

    assert bottom_y > demo.BOTTOM_LIFT_Y - 0.04
    assert gap_y < 0.35
    assert contact["bottom_count"] == (demo.GRID_N + 1) ** 2
    assert contact["top_count"] == (demo.GRID_N + 1) ** 2
    assert 0.0 <= contact["gap_min"] <= contact["gap_max"] <= 0.03
    assert beta is not None
    assert beta["count"] >= 8
    assert beta["min"] >= 0.95
    assert beta["frac_09"] == pytest.approx(1.0)

    advance_to(sim, demo.PULL_UNTIL)
    bottom_y, _top_y, gap_y = demo.cube_height_stats(sim)
    contact = demo.cube_contact_gap_stats(sim)
    beta = demo.adhesion_beta_stats(sim)

    assert bottom_y < demo.BOTTOM_INITIAL_Y + 0.02
    assert gap_y > 0.8
    assert contact["gap_min"] > 0.6
    assert beta is not None
    assert beta["frac_09"] == pytest.approx(0.0)


@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.example
def test_rcc_cube_cloth_lift_hold_release():
    demo = load_example("rcc_adhesive_cube_cloth_lift_release_demo.py")
    sim = demo.build_demo(adhesion_on=True)

    advance_to(sim, demo.PRE_PULL_HOLD_UNTIL)
    _cube_y, cloth_y, gap_y = demo.height_stats(sim)
    bottom = demo.bottom_contact_stats(sim)
    beta = demo.adhesion_beta_stats(sim)

    assert cloth_y > demo.CLOTH_LIFT_Y - 0.04
    assert gap_y < 0.20
    assert bottom["count"] >= 80
    assert abs(bottom["gap_mean"]) <= 0.03
    assert bottom["gap_min"] >= -0.06
    assert bottom["gap_max"] <= 0.06
    assert beta is not None
    assert beta["count"] >= 80
    assert beta["min"] >= 0.99
    assert beta["frac_09"] == pytest.approx(1.0)

    advance_to(sim, demo.PULL_UNTIL)
    _cube_y, cloth_y, gap_y = demo.height_stats(sim)
    bottom = demo.bottom_contact_stats(sim)
    beta = demo.adhesion_beta_stats(sim)

    assert cloth_y < demo.CLOTH_INITIAL_Y
    assert gap_y > 0.8
    assert bottom["gap_mean"] > 0.5
    assert beta is not None
    assert beta["frac_09"] == pytest.approx(0.0)
