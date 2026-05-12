from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_analyzer_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "analyze_socu_native_contact_m56.py"
    spec = importlib.util.spec_from_file_location("socu_native_contact_m56", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(
    path: Path,
    *,
    cold: bool = False,
    cache_hit: bool = False,
    rebuilt: bool = False,
    replay_path: str = "native_plan",
    plan_ms: float = 0.0,
    hessian_ms: float = 0.0,
    executor_ms: float = 0.0,
) -> None:
    payload = {
        "solver": "socu_approx",
        "timing": {
            "native_contact_plan_build_ms": plan_ms,
            "native_contact_hessian_triplet_ms": hessian_ms,
            "native_contact_executor_scatter_ms": executor_ms,
            "native_contact_numeric_ms": hessian_ms + executor_ms,
            "contact_assembly_time_ms": plan_ms + hessian_ms + executor_ms,
        },
        "contact": {
            "native_contact_plan_cold_start": cold,
            "native_contact_plan_cache_hit": cache_hit,
            "native_contact_plan_rebuilt_this_solve": rebuilt,
        },
        "runtime_reorder": {"native_contact_replay_path": replay_path},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.basic
def test_socu_native_contact_m56_analyzer_groups_cold_and_cache_hit(tmp_path: Path):
    analyzer = load_analyzer_module()
    cold_path = tmp_path / "cold.json"
    hit_path = tmp_path / "hit.json"
    ignored_path = tmp_path / "ignored.json"
    write_report(cold_path, cold=True, plan_ms=2.0, hessian_ms=4.0, executor_ms=0.5)
    write_report(hit_path, cache_hit=True, plan_ms=0.0, hessian_ms=3.0, executor_ms=0.25)
    ignored_path.write_text(json.dumps({"solver": "fused_pcg"}), encoding="utf-8")

    loaded = [
        (path, payload)
        for path in analyzer.iter_report_paths([tmp_path])
        if (payload := analyzer.load_socu_report(path)) is not None
    ]
    summary = analyzer.summarize(loaded)

    assert summary["report_count"] == 2
    assert summary["native_plan_report_count"] == 2
    assert summary["cache_state_counts"] == {"cache_hit": 1, "cold_rebuild": 1}
    assert summary["replay_path_counts"] == {"native_plan": 2}

    groups = {
        (group["cache_state"], group["replay_path"]): group
        for group in summary["groups"]
    }
    assert groups[("cold_rebuild", "native_plan")]["fields"][
        "native_contact_plan_build_ms"
    ]["median_ms"] == pytest.approx(2.0)
    assert groups[("cache_hit", "native_plan")]["fields"][
        "native_contact_executor_scatter_ms"
    ]["median_ms"] == pytest.approx(0.25)


@pytest.mark.basic
def test_socu_native_contact_m56_analyzer_requires_native_plan(tmp_path: Path):
    analyzer = load_analyzer_module()
    report_path = tmp_path / "legacy.json"
    write_report(report_path, rebuilt=True, replay_path="legacy_structured")

    assert analyzer.main([str(tmp_path), "--require-native-plan"]) == 3
