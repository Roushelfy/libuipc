from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .artifacts import read_json, write_json
from .charts import render_chart_artifacts
from .quality import write_quality_csv
from .selection import load_assets_catalog_map
from .timers import CANONICAL_STAGES, summarize_iteration_counters, summarize_timer_frames


def _perf_dir(run_root: Path, asset: str, level: str) -> Path:
    return run_root / "runs" / asset / level / "perf"


def _quality_dir(run_root: Path, asset: str, level: str) -> Path:
    return run_root / "runs" / asset / level / "quality"


def _visual_dir(run_root: Path, asset: str, level: str) -> Path:
    return run_root / "runs" / asset / level / "visual"


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _load_iteration_summary(run_root: Path, asset: str, level: str) -> Dict[str, Any] | None:
    iteration_path = _perf_dir(run_root, asset, level) / "iteration_summary.json"
    if iteration_path.exists():
        return read_json(iteration_path)

    timer_frames_path = _perf_dir(run_root, asset, level) / "timer_frames.json"
    if not timer_frames_path.exists():
        return None

    try:
        timer_frames = read_json(timer_frames_path)
    except Exception:
        return None

    if not isinstance(timer_frames, list):
        return None
    return summarize_iteration_counters(timer_frames)


def _load_stage_summary(run_root: Path, asset: str, level: str) -> Dict[str, Any] | None:
    timer_frames_path = _perf_dir(run_root, asset, level) / "timer_frames.json"
    if timer_frames_path.exists():
        try:
            timer_frames = read_json(timer_frames_path)
        except Exception:
            timer_frames = None
        if isinstance(timer_frames, list):
            return summarize_timer_frames(timer_frames)

    return _load_json_if_exists(_perf_dir(run_root, asset, level) / "stage_summary.json")


def collect_report_data(run_root: Path) -> Dict[str, Any]:
    run_meta = read_json(run_root / "run_meta.json")
    selection = read_json(run_root / "selection.json")
    assets = [row["name"] for row in selection["assets"]]
    levels = list(run_meta["levels"])
    active_failures = []
    for row in run_meta.get("failures", []):
        output_dir = row.get("output_dir")
        if not output_dir:
            continue
        if (Path(output_dir) / "failure.json").exists():
            active_failures.append(row)
    data: Dict[str, Any] = {
        "run_meta": run_meta,
        "selection": selection,
        "assets": {},
        "failures": active_failures,
    }
    for asset in assets:
        data["assets"][asset] = {}
        for level in levels:
            perf = _load_stage_summary(run_root, asset, level)
            iteration = _load_iteration_summary(run_root, asset, level)
            bench = _load_json_if_exists(_perf_dir(run_root, asset, level) / "benchmark.json")
            quality = _load_json_if_exists(_quality_dir(run_root, asset, level) / "quality_metrics.json")
            visual = _load_json_if_exists(_visual_dir(run_root, asset, level) / "visual_manifest.json")
            perf_failure = _load_json_if_exists(_perf_dir(run_root, asset, level) / "failure.json")
            quality_failure = _load_json_if_exists(_quality_dir(run_root, asset, level) / "failure.json")
            data["assets"][asset][level] = {
                "benchmark": bench,
                "stage_summary": perf,
                "iteration_summary": iteration,
                "quality_metrics": quality,
                "visual_manifest": visual,
                "perf_failure": perf_failure,
                "quality_failure": quality_failure,
            }
    return data


def _frame_ms(bench: Dict[str, Any] | None) -> float | None:
    if not bench:
        return None
    wall = bench.get("wall_time")
    frames = bench.get("num_frames")
    if wall is None or not frames:
        return None
    return float(wall) / float(frames) * 1000.0


def build_summary_payload(report_data: Dict[str, Any]) -> Dict[str, Any]:
    run_meta = report_data["run_meta"]
    assets = report_data["assets"]
    catalog = load_assets_catalog_map()
    selection_assets = {
        row["name"]: row for row in report_data.get("selection", {}).get("assets", [])
    }
    levels = run_meta["levels"]
    compare_levels = [level for level in levels if level != "fp64"]
    summary: Dict[str, Any] = {
        "run_meta": run_meta,
        "assets": {},
        "pipeline_overview": [],
        "scenario_overview": [],
        "scenario_family_overview": [],
        "iteration_overview": [],
        "quality_rows": [],
        "visual_rows": [],
        "chart_artifacts": {
            "chart_segments_json": None,
            "asset_charts": [],
            "scenario_charts": [],
        },
        "chart_warnings": [],
        "failures": list(report_data.get("failures", [])),
    }

    for asset, asset_levels in assets.items():
        asset_meta = selection_assets.get(asset, {})
        catalog_meta = catalog.get(asset)
        scenario = asset_meta.get("scenario") or (catalog_meta.scenario if catalog_meta else None)
        scenario_family = asset_meta.get("scenario_family") or (
            catalog_meta.scenario_family if catalog_meta else None
        )
        asset_summary = {"levels": {}, "comparisons": {}}
        fp64_bench = asset_levels.get("fp64", {}).get("benchmark")
        fp64_ms = _frame_ms(fp64_bench)
        fp64_stages = {row["stage"]: row for row in (asset_levels.get("fp64", {}).get("stage_summary") or {}).get("stages", [])}
        fp64_counters = {
            row["counter"]: row
            for row in (asset_levels.get("fp64", {}).get("iteration_summary") or {}).get("counters", [])
        }

        for level, payload in asset_levels.items():
            asset_summary["levels"][level] = {
                "frame_ms": _frame_ms(payload.get("benchmark")),
                "stage_summary": payload.get("stage_summary"),
                "iteration_summary": payload.get("iteration_summary"),
                "quality_metrics": payload.get("quality_metrics"),
                "visual_manifest": payload.get("visual_manifest"),
                "scenario": scenario,
                "scenario_family": scenario_family,
            }
            if payload.get("quality_metrics"):
                metrics = dict(payload["quality_metrics"])
                metrics["asset"] = asset
                metrics["level"] = level
                metrics["scenario"] = scenario
                metrics["scenario_family"] = scenario_family
                summary["quality_rows"].append(metrics)
            if payload.get("visual_manifest"):
                summary["visual_rows"].append(
                    {
                        "asset": asset,
                        "level": level,
                        "scenario": scenario,
                        "scenario_family": scenario_family,
                        "frames_exported": payload["visual_manifest"].get("frames_exported", 0),
                        "visual_dir": payload["visual_manifest"].get("visual_dir"),
                    }
                )

        for level in compare_levels:
            level_payload = asset_levels.get(level, {})
            level_ms = _frame_ms(level_payload.get("benchmark"))
            comparisons = []
            level_stages = {row["stage"]: row for row in (level_payload.get("stage_summary") or {}).get("stages", [])}
            level_counters = {
                row["counter"]: row
                for row in (level_payload.get("iteration_summary") or {}).get("counters", [])
            }
            for stage in CANONICAL_STAGES:
                fp64_stage = fp64_stages.get(stage, {})
                level_stage = level_stages.get(stage, {})
                fp64_stage_ms = fp64_stage.get("mean_ms", 0.0)
                level_stage_ms = level_stage.get("mean_ms", 0.0)
                delta_ms = level_stage_ms - fp64_stage_ms
                delta_pct = None if fp64_stage_ms == 0 else delta_ms / fp64_stage_ms * 100.0
                row = {
                    "stage": stage,
                    "scenario": scenario,
                    "scenario_family": scenario_family,
                    "fp64_ms_per_frame": fp64_stage_ms,
                    "level_ms_per_frame": level_stage_ms,
                    "delta_ms": delta_ms,
                    "delta_pct": delta_pct,
                }
                comparisons.append(row)
                summary["pipeline_overview"].append({"asset": asset, "level": level, **row})
            counter_rows = []
            for counter_key in ["newton_iteration_count", "line_search_iteration_count", "pcg_iteration_count"]:
                fp64_counter = fp64_counters.get(counter_key, {})
                level_counter = level_counters.get(counter_key, {})
                fp64_mean = fp64_counter.get("mean_count", 0.0)
                level_mean = level_counter.get("mean_count", 0.0)
                delta = level_mean - fp64_mean
                delta_pct = None if fp64_mean == 0 else delta / fp64_mean * 100.0
                row = {
                    "counter": counter_key,
                    "scenario": scenario,
                    "scenario_family": scenario_family,
                    "fp64_mean_count": fp64_mean,
                    "level_mean_count": level_mean,
                    "delta_count": delta,
                    "delta_pct": delta_pct,
                }
                counter_rows.append(row)
                summary["iteration_overview"].append({"asset": asset, "level": level, **row})
            asset_summary["comparisons"][level] = {
                "fp64_ms_per_frame": fp64_ms,
                "level_ms_per_frame": level_ms,
                "stages": comparisons,
                "iteration_counters": counter_rows,
            }

        summary["assets"][asset] = asset_summary
    summary["scenario_overview"] = aggregate_pipeline_rows(summary["pipeline_overview"], ["scenario", "level", "stage"])
    summary["scenario_family_overview"] = aggregate_pipeline_rows(
        summary["pipeline_overview"], ["scenario", "scenario_family", "level", "stage"]
    )
    return summary


def aggregate_pipeline_rows(rows: Iterable[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        bucket = buckets.setdefault(
            key,
            {
                **{name: row.get(name) for name in group_keys},
                "asset_count": 0,
                "_assets": set(),
                "_fp64_values": [],
                "_level_values": [],
                "_delta_values": [],
                "_delta_pct_values": [],
            },
        )
        asset_name = row.get("asset")
        if asset_name is not None:
            bucket["_assets"].add(asset_name)
        if row.get("fp64_ms_per_frame") is not None:
            bucket["_fp64_values"].append(float(row["fp64_ms_per_frame"]))
        if row.get("level_ms_per_frame") is not None:
            bucket["_level_values"].append(float(row["level_ms_per_frame"]))
        if row.get("delta_ms") is not None:
            bucket["_delta_values"].append(float(row["delta_ms"]))
        if row.get("delta_pct") is not None:
            bucket["_delta_pct_values"].append(float(row["delta_pct"]))

    aggregated: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        def mean(values: List[float]) -> float | None:
            if not values:
                return None
            return sum(values) / len(values)

        aggregated.append(
            {
                **{name: bucket.get(name) for name in group_keys},
                "asset_count": len(bucket["_assets"]),
                "fp64_ms_per_frame": mean(bucket["_fp64_values"]),
                "level_ms_per_frame": mean(bucket["_level_values"]),
                "delta_ms": mean(bucket["_delta_values"]),
                "delta_pct": mean(bucket["_delta_pct_values"]),
            }
        )
    aggregated.sort(key=lambda row: tuple("" if row.get(name) is None else str(row.get(name)) for name in group_keys))
    return aggregated


def write_report_files(run_root: Path, summary: Dict[str, Any]) -> None:
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    chart_artifacts = render_chart_artifacts(run_root, summary)
    summary["chart_artifacts"] = {
        "chart_segments_json": chart_artifacts.get("chart_segments_json"),
        "asset_charts": chart_artifacts.get("asset_charts", []),
        "scenario_charts": chart_artifacts.get("scenario_charts", []),
    }
    summary["chart_warnings"] = list(chart_artifacts.get("warnings", []))
    write_json(reports_dir / "summary.json", summary)
    (reports_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_perf_by_stage_csv(reports_dir / "perf_by_stage.csv", summary["pipeline_overview"])
    write_perf_by_asset_csv(reports_dir / "perf_by_asset.csv", summary)
    write_perf_by_scenario_csv(reports_dir / "perf_by_scenario.csv", summary["pipeline_overview"])
    write_solver_iters_csv(reports_dir / "solver_iters.csv", summary["iteration_overview"])
    write_quality_csv(reports_dir / "quality.csv", [
        {
            "asset": row["asset"],
            "level": row["level"],
            "rel_l2_max": row.get("rel_l2_max", 0.0),
            "abs_linf_max": row.get("abs_linf_max", 0.0),
            "nan_inf_count": row.get("nan_inf_count", 0),
            "record_count": row.get("record_count", 0),
            "missing_in_compare_count": row.get("missing_in_compare_count", 0),
            "missing_in_reference_count": row.get("missing_in_reference_count", 0),
            "reference_dir": row.get("reference_dir"),
            "compare_dir": row.get("compare_dir"),
        }
        for row in summary["quality_rows"]
    ])
    write_visual_exports_csv(reports_dir / "visual_exports.csv", summary["visual_rows"])


def write_perf_by_stage_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "asset",
                "scenario",
                "scenario_family",
                "level",
                "stage",
                "fp64_ms_per_frame",
                "level_ms_per_frame",
                "delta_ms",
                "delta_pct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_perf_by_scenario_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "scenario_family",
                "asset",
                "level",
                "stage",
                "fp64_ms_per_frame",
                "level_ms_per_frame",
                "delta_ms",
                "delta_pct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_perf_by_asset_csv(path: Path, summary: Dict[str, Any]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "asset",
                "scenario",
                "scenario_family",
                "level",
                "fp64_ms_per_frame",
                "level_ms_per_frame",
                "top_regression_stage",
                "top_regression_delta_pct",
            ],
        )
        writer.writeheader()
        for asset, asset_summary in summary["assets"].items():
            for level, compare in asset_summary.get("comparisons", {}).items():
                stages = list(compare.get("stages", []))
                stages.sort(key=lambda row: row.get("delta_pct") or float("-inf"), reverse=True)
                top = stages[0] if stages else {}
                writer.writerow(
                    {
                        "asset": asset,
                        "scenario": asset_summary["levels"].get(level, {}).get("scenario"),
                        "scenario_family": asset_summary["levels"].get(level, {}).get("scenario_family"),
                        "level": level,
                        "fp64_ms_per_frame": compare.get("fp64_ms_per_frame"),
                        "level_ms_per_frame": compare.get("level_ms_per_frame"),
                        "top_regression_stage": top.get("stage"),
                        "top_regression_delta_pct": top.get("delta_pct"),
                    }
                )


def write_visual_exports_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["asset", "scenario", "scenario_family", "level", "frames_exported", "visual_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_solver_iters_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "asset",
                "scenario",
                "scenario_family",
                "level",
                "counter",
                "fp64_mean_count",
                "level_mean_count",
                "delta_count",
                "delta_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: Dict[str, Any]) -> str:
    def fmt3(value: Any) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines: List[str] = []
    run_meta = summary["run_meta"]
    lines.append("# Mixed UIPC Assets Benchmark")
    lines.append("")
    lines.append(f"- `run_root`: `{run_meta['run_root']}`")
    lines.append(f"- `levels`: `{', '.join(run_meta['levels'])}`")
    lines.append(f"- `asset_count`: `{len(summary['assets'])}`")
    if summary.get("chart_artifacts", {}).get("chart_segments_json"):
        lines.append(
            f"- `chart_segments_json`: `{summary['chart_artifacts']['chart_segments_json']}`"
        )
    lines.append("")
    lines.append("## Chart Warnings")
    lines.append("")
    if not summary.get("chart_warnings"):
        lines.append("- none")
    else:
        for warning in summary["chart_warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Scenario Overview")
    lines.append("")
    lines.append("| Scenario | Level | Stage | Asset Count | FP64 ms/frame | Level ms/frame | Delta % |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in summary.get("scenario_overview", []):
        lines.append(
            f"| {row.get('scenario','n/a')} | {row['level']} | {row['stage']} | {row.get('asset_count', 0)} | "
            f"{fmt3(row.get('fp64_ms_per_frame'))} | {fmt3(row.get('level_ms_per_frame'))} | "
            f"{'n/a' if row['delta_pct'] is None else format(row['delta_pct'], '.2f')} |"
        )
    lines.append("")
    lines.append("## Scenario Family Overview")
    lines.append("")
    lines.append("| Scenario | Family | Level | Stage | Asset Count | FP64 ms/frame | Level ms/frame | Delta % |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for row in summary.get("scenario_family_overview", []):
        lines.append(
            f"| {row.get('scenario','n/a')} | {row.get('scenario_family','n/a')} | {row['level']} | {row['stage']} | "
            f"{row.get('asset_count', 0)} | {fmt3(row.get('fp64_ms_per_frame'))} | "
            f"{fmt3(row.get('level_ms_per_frame'))} | "
            f"{'n/a' if row['delta_pct'] is None else format(row['delta_pct'], '.2f')} |"
        )
    lines.append("")
    lines.append("## Scenario Charts")
    lines.append("")
    scenario_charts = summary.get("chart_artifacts", {}).get("scenario_charts", [])
    if not scenario_charts:
        lines.append("- none")
    else:
        for chart in scenario_charts:
            chart_path = Path(chart["path"])
            rel_path = chart_path.relative_to(Path(summary["run_meta"]["run_root"]) / "reports")
            lines.append(f"### {chart['scenario']}")
            lines.append("")
            lines.append(f"![Scenario {chart['scenario']}]({rel_path.as_posix()})")
            lines.append("")
    lines.append("")
    lines.append("## Asset Charts")
    lines.append("")
    asset_charts = summary.get("chart_artifacts", {}).get("asset_charts", [])
    if not asset_charts:
        lines.append("- none")
    else:
        for chart in asset_charts:
            chart_path = Path(chart["path"])
            rel_path = chart_path.relative_to(Path(summary["run_meta"]["run_root"]) / "reports")
            lines.append(f"### {chart['asset']}")
            lines.append("")
            lines.append(f"![Asset {chart['asset']}]({rel_path.as_posix()})")
            lines.append("")
    lines.append("")
    lines.append("## Pipeline Overview")
    lines.append("")
    lines.append("| Asset | Level | Stage | FP64 ms/frame | Level ms/frame | Delta ms | Delta % |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in summary["pipeline_overview"]:
        lines.append(
            f"| {row['asset']} | {row['level']} | {row['stage']} | "
            f"{fmt3(row.get('fp64_ms_per_frame'))} | {fmt3(row.get('level_ms_per_frame'))} | "
            f"{fmt3(row.get('delta_ms'))} | "
            f"{'n/a' if row['delta_pct'] is None else format(row['delta_pct'], '.2f')} |"
        )
    lines.append("")
    lines.append("## Asset Regressions")
    lines.append("")
    lines.append("| Asset | Level | FP64 ms/frame | Level ms/frame | Top Regression Stage | Top Regression Delta % |")
    lines.append("|---|---|---:|---:|---|---:|")
    for asset, asset_summary in summary["assets"].items():
        for level, compare in asset_summary.get("comparisons", {}).items():
            stages = list(compare.get("stages", []))
            stages.sort(key=lambda row: row.get("delta_pct") or float("-inf"), reverse=True)
            top = stages[0] if stages else {}
            top_pct = top.get("delta_pct")
            lines.append(
                f"| {asset} | {level} | {fmt3(compare.get('fp64_ms_per_frame'))} | "
                f"{fmt3(compare.get('level_ms_per_frame'))} | {top.get('stage', 'n/a')} | "
                f"{'n/a' if top_pct is None else f'{top_pct:.2f}'} |"
            )
    lines.append("")
    lines.append("## Iteration Counters")
    lines.append("")
    lines.append("| Asset | Level | Counter | FP64 Mean | Level Mean | Delta | Delta % |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in summary.get("iteration_overview", []):
        lines.append(
            f"| {row['asset']} | {row['level']} | {row['counter']} | "
            f"{fmt3(row.get('fp64_mean_count'))} | {fmt3(row.get('level_mean_count'))} | "
            f"{fmt3(row.get('delta_count'))} | "
            f"{'n/a' if row['delta_pct'] is None else format(row['delta_pct'], '.2f')} |"
        )
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    if not summary["failures"]:
        lines.append("- none")
    else:
        lines.append("| Asset | Level | Mode | Stage | stderr log |")
        lines.append("|---|---|---|---|---|")
        for row in summary["failures"]:
            lines.append(
                f"| {row.get('asset','n/a')} | {row.get('level','n/a')} | {row.get('mode','n/a')} | "
                f"{row.get('stage','unknown')} | `{row.get('stderr_log','')}` |"
            )
    lines.append("")
    lines.append("## Quality")
    lines.append("")
    if not summary["quality_rows"]:
        lines.append("- none")
    else:
        lines.append("| Asset | Level | rel_l2.max | abs_linf.max | nan_inf |")
        lines.append("|---|---|---:|---:|---:|")
        for row in summary["quality_rows"]:
            lines.append(
                f"| {row['asset']} | {row['level']} | "
                f"{'n/a' if row.get('rel_l2_max') is None else format(row.get('rel_l2_max'), '.6e')} | "
                f"{'n/a' if row.get('abs_linf_max') is None else format(row.get('abs_linf_max'), '.6e')} | "
                f"{row.get('nan_inf_count', 0)} |"
            )
    lines.append("")
    lines.append("## Visual Exports")
    lines.append("")
    if not summary["visual_rows"]:
        lines.append("- none")
    else:
        for row in summary["visual_rows"]:
            lines.append(f"- `{row['asset']}` / `{row['level']}`: {row['frames_exported']} frame(s) -> `{row['visual_dir']}`")
    lines.append("")
    return "\n".join(lines)
