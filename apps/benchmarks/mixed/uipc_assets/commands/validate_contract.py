from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..core.report_schema import build_summary_payload, collect_report_data, write_report_files


DEFAULT_REQUIRED_LEVELS = ["fp64", "path1", "path2", "path3", "path4", "path5", "path6"]
REQUIRED_ANALYSIS_FILES = (
    "analysis.md",
    "iteration_counts_by_slice.csv",
    "solver_stage_per_pcg_by_slice.csv",
    "charts/pcg_count_ratio_heatmap.svg",
    "charts/overall_solver_stage_per_pcg_heatmap.svg",
)
REQUIRED_ANALYSIS_HEADINGS = (
    "PCG Count Ratio",
    "Solver Stage Per-PCG",
)
OMISSION_MARKERS = (
    "Solver-stage per-PCG heatmaps were omitted",
    "PCG count ratios were omitted",
)


def _load_summary(run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "reports" / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    summary = build_summary_payload(collect_report_data(run_root))
    write_report_files(run_root, summary)
    return summary


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_levels(summary: dict[str, Any], required_levels: list[str]) -> list[str]:
    issues: list[str] = []
    available = list(summary.get("run_meta", {}).get("levels", []))
    missing = [level for level in required_levels if level not in available]
    if missing:
        issues.append(f"missing required levels: {', '.join(missing)}")
    return issues


def _validate_quality(summary: dict[str, Any], required_levels: list[str], max_rel_l2: float, max_abs_linf: float) -> list[str]:
    issues: list[str] = []
    compare_levels = [level for level in required_levels if level != "fp64"]
    asset_levels = summary.get("assets", {})

    for asset, asset_summary in asset_levels.items():
        level_payloads = asset_summary.get("levels", {})
        for level in compare_levels:
            if level not in level_payloads:
                issues.append(f"{asset}: missing level payload for {level}")
                continue
            if not level_payloads[level].get("quality_metrics"):
                issues.append(f"{asset}: missing quality metrics for {level}")

    for row in summary.get("quality_rows", []):
        asset = row.get("asset", "unknown")
        level = row.get("level")
        if level not in compare_levels:
            continue

        rel_l2 = _as_float(row.get("rel_l2_max"))
        abs_linf = _as_float(row.get("abs_linf_max"))
        nan_inf = _as_int(row.get("nan_inf_count"))

        if rel_l2 is None or not rel_l2 < max_rel_l2:
            issues.append(f"{asset}/{level}: rel_l2_max={rel_l2} violates < {max_rel_l2}")
        if abs_linf is None or not abs_linf < max_abs_linf:
            issues.append(f"{asset}/{level}: abs_linf_max={abs_linf} violates < {max_abs_linf}")
        if nan_inf is None or nan_inf != 0:
            issues.append(f"{asset}/{level}: nan_inf_count={nan_inf} violates == 0")

    return issues


def _validate_deep_analysis(run_root: Path) -> list[str]:
    issues: list[str] = []
    deep_dir = run_root / "reports" / "deep_analysis"

    if not deep_dir.exists():
        return [f"missing deep analysis directory: {deep_dir}"]

    for relative_path in REQUIRED_ANALYSIS_FILES:
        path = deep_dir / relative_path
        if not path.exists():
            issues.append(f"missing deep analysis artifact: {path}")

    analysis_path = deep_dir / "analysis.md"
    if analysis_path.exists():
        analysis_text = analysis_path.read_text(encoding="utf-8")
        for heading in REQUIRED_ANALYSIS_HEADINGS:
            if heading not in analysis_text:
                issues.append(f"analysis.md missing heading: {heading}")
        for marker in OMISSION_MARKERS:
            if marker in analysis_text:
                issues.append(f"analysis.md still reports omitted PCG detail: {marker}")

    iteration_path = deep_dir / "iteration_counts_by_slice.csv"
    if iteration_path.exists():
        rows = _read_csv_rows(iteration_path)
        pcg_rows = [row for row in rows if row.get("counter") == "pcg_iteration_count"]
        if not pcg_rows:
            issues.append("iteration_counts_by_slice.csv has no pcg_iteration_count rows")
        elif not any((_as_int(row.get("support_count")) or 0) > 0 for row in pcg_rows):
            issues.append("iteration_counts_by_slice.csv has no pcg_iteration_count rows with support_count > 0")

    solver_path = deep_dir / "solver_stage_per_pcg_by_slice.csv"
    if solver_path.exists():
        rows = _read_csv_rows(solver_path)
        if not rows:
            issues.append("solver_stage_per_pcg_by_slice.csv is empty")
        elif not any((_as_int(row.get("support_count")) or 0) > 0 for row in rows):
            issues.append("solver_stage_per_pcg_by_slice.csv has no rows with support_count > 0")

    return issues


def run(args) -> int:
    run_root = Path(args.run_root).resolve()
    summary = _load_summary(run_root)
    required_levels = list(args.levels or DEFAULT_REQUIRED_LEVELS)

    issues: list[str] = []
    issues.extend(_validate_levels(summary, required_levels))
    issues.extend(_validate_quality(summary, required_levels, args.max_rel_l2, args.max_abs_linf))
    issues.extend(_validate_deep_analysis(run_root))

    if issues:
        print("mixed benchmark contract validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    compare_levels = [level for level in required_levels if level != "fp64"]
    quality_rows = [row for row in summary.get("quality_rows", []) if row.get("level") in compare_levels]
    print(
        "mixed benchmark contract validation passed: "
        f"{len(quality_rows)} quality rows, deep PCG analysis present"
    )
    return 0
