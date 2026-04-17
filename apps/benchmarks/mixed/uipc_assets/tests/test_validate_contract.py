from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.commands import validate_contract


def _write_run_root(tmp_path: Path, *, include_pcg_details: bool, rel_l2: float = 1.0e-6) -> Path:
    run_root = tmp_path / "run"
    reports = run_root / "reports"
    deep = reports / "deep_analysis"
    charts = deep / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    levels = validate_contract.DEFAULT_REQUIRED_LEVELS
    asset_levels = {
        level: {"quality_metrics": None if level == "fp64" else {"ok": True}}
        for level in levels
    }
    summary = {
        "run_meta": {"levels": levels},
        "assets": {"representative_asset": {"levels": asset_levels}},
        "quality_rows": [
            {
                "asset": "representative_asset",
                "level": level,
                "rel_l2_max": rel_l2,
                "abs_linf_max": 1.0e-5,
                "nan_inf_count": 0,
            }
            for level in levels
            if level != "fp64"
        ],
        "failures": [],
    }
    (reports / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    analysis_lines = [
        "# Analysis",
        "",
        "## 4. Stage Per-Iter Analysis",
        "",
        "![Overall Solver Stage Per-PCG](charts/overall_solver_stage_per_pcg_heatmap.svg)",
        "",
        "## 5. Iteration Analysis",
        "",
        "![PCG Count Ratio](charts/pcg_count_ratio_heatmap.svg)",
        "",
    ]
    if not include_pcg_details:
        analysis_lines.append("- PCG count ratios were omitted because the stable samples reported zero PCG totals throughout this run.")
    (deep / "analysis.md").write_text("\n".join(analysis_lines), encoding="utf-8")

    (charts / "pcg_count_ratio_heatmap.svg").write_text("<svg/>", encoding="utf-8")
    (charts / "overall_solver_stage_per_pcg_heatmap.svg").write_text("<svg/>", encoding="utf-8")

    iteration_csv = "\n".join(
        [
            "slice_type,slice_name,level,counter,stable_asset_count,support_count,hard_failure_count,advisory_count,blocked_count,mean_fp64_mean_count,mean_level_mean_count,mean_fp64_total_count,mean_level_total_count,median_count_ratio,mean_count_ratio,p90_count_ratio,main_report",
            f"overall,all,path1,pcg_iteration_count,1,{1 if include_pcg_details else 0},0,0,0,10,10,100,100,1,1,1,True",
        ]
    )
    (deep / "iteration_counts_by_slice.csv").write_text(iteration_csv, encoding="utf-8")

    solver_csv = "\n".join(
        [
            "slice_type,slice_name,level,stage,stable_asset_count,support_count,hard_failure_count,advisory_count,blocked_count,mean_fp64_ms_per_pcg,mean_level_ms_per_pcg,median_per_pcg_speedup,mean_per_pcg_speedup,p90_per_pcg_speedup,mean_per_pcg_saved_ms,median_per_pcg_saved_ms,main_report",
            f"overall,all,path1,SpMV,1,{1 if include_pcg_details else 0},0,0,0,1.0,0.8,1.2,1.2,1.2,0.2,0.2,True",
        ]
    )
    (deep / "solver_stage_per_pcg_by_slice.csv").write_text(solver_csv, encoding="utf-8")
    return run_root


def test_validate_contract_passes_when_quality_and_pcg_artifacts_exist(tmp_path: Path) -> None:
    run_root = _write_run_root(tmp_path, include_pcg_details=True)
    args = SimpleNamespace(
        run_root=run_root,
        levels=validate_contract.DEFAULT_REQUIRED_LEVELS,
        max_rel_l2=1.0e-5,
        max_abs_linf=5.0e-4,
    )
    assert validate_contract.run(args) == 0


def test_validate_contract_fails_when_pcg_details_are_missing(tmp_path: Path) -> None:
    run_root = _write_run_root(tmp_path, include_pcg_details=False)
    args = SimpleNamespace(
        run_root=run_root,
        levels=validate_contract.DEFAULT_REQUIRED_LEVELS,
        max_rel_l2=1.0e-5,
        max_abs_linf=5.0e-4,
    )
    assert validate_contract.run(args) == 1
