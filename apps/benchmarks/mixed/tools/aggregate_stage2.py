#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from solution_metrics import collect_solution_dir_metrics


UNIT_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
}

KNOWN_SCENARIOS = (
    "wrecking_ball",
    "fem_ground_contact",
    "fem_heavy_nocontact",
    "fem_heavy_ground_contact",
    "fem_gravity",
    "abd_gravity",
)

PERF_SCENARIO_THRESHOLDS = {
    "wrecking_ball": 20.0,
    "fem_heavy_nocontact": 10.0,
    "fem_heavy_ground_contact": 15.0,
}


def to_ms(value: float, unit: str) -> float:
    factor = UNIT_TO_MS.get(unit, 1e-6)
    return float(value) * factor


def load_benchmark_file(path: Path) -> Tuple[Dict[str, float], List[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: Dict[str, float] = {}
    failures: List[dict] = []
    for item in data.get("benchmarks", []):
        name = item.get("name", "")
        if not name:
            continue
        if item.get("error_occurred", False):
            failures.append(
                {
                    "name": name,
                    "error": item.get("error_message", "unknown benchmark error"),
                }
            )
            continue

        if "real_time" not in item:
            continue
        real_time_ms = to_ms(item["real_time"], item.get("time_unit", "ns"))
        if name not in entries:
            entries[name] = real_time_ms
        else:
            entries[name] = min(entries[name], real_time_ms)
    return entries, failures


def compare_path1_vs_fp64(fp64: Dict[str, float], path1: Dict[str, float]) -> List[dict]:
    rows: List[dict] = []
    for name, fp64_ms in fp64.items():
        if not name.startswith("Mixed.Stage2.Perf."):
            continue
        if ".cuda_mixed" not in name:
            continue
        if name not in path1:
            continue
        scenario = infer_scenario_from_path(Path(name))
        path1_ms = path1[name]
        delta_pct = None
        if fp64_ms > 0:
            delta_pct = (path1_ms - fp64_ms) / fp64_ms * 100.0
        threshold = PERF_SCENARIO_THRESHOLDS.get(scenario, 20.0)
        warning = bool(delta_pct is not None and delta_pct > threshold)
        rows.append(
            {
                "name": name,
                "scenario": scenario,
                "fp64_ms": fp64_ms,
                "path1_ms": path1_ms,
                "delta_pct": delta_pct,
                "warning": warning,
                "threshold_pct": threshold,
            }
        )
    rows.sort(key=lambda x: x["name"])
    return rows


def compare_telemetry_overhead(level_name: str, entries: Dict[str, float]) -> List[dict]:
    rows: List[dict] = []
    for on_name, on_ms in entries.items():
        if ".TelemetryOn." not in on_name:
            continue
        off_name = on_name.replace(".TelemetryOn.", ".TelemetryOff.")
        off_ms = entries.get(off_name)
        if off_ms is None:
            continue
        delta_pct = None
        if off_ms > 0:
            delta_pct = (on_ms - off_ms) / off_ms * 100.0
        warning = bool(delta_pct is not None and delta_pct > 25.0)
        rows.append(
            {
                "level": level_name,
                "on_name": on_name,
                "off_name": off_name,
                "off_ms": off_ms,
                "on_ms": on_ms,
                "delta_pct": delta_pct,
                "warning": warning,
                "threshold_pct": 25.0,
            }
        )
    rows.sort(key=lambda x: x["on_name"])
    return rows


def infer_scenario_from_path(path: Path) -> str:
    lower = str(path).lower()
    for scenario in KNOWN_SCENARIOS:
        if scenario in lower:
            return scenario
    return "unknown"


def _collect_solution_dirs_by_tag(workspace_root: Path,
                                  run_mode: str,
                                  workspace_tag: str) -> Tuple[Dict[str, Path], List[str]]:
    dirs: Dict[str, Path] = {}
    issues: List[str] = []

    for sol_file in workspace_root.rglob("x.*.mtx"):
        tag_dir = next((parent for parent in sol_file.parents if parent.name == workspace_tag), None)
        if tag_dir is None:
            continue
        if tag_dir.parent.name.lower() != "telemetryoff":
            continue
        if tag_dir.parent.parent.name.lower() != run_mode.lower():
            continue

        scenario = infer_scenario_from_path(tag_dir)
        prev = dirs.get(scenario)
        if prev is not None and prev != tag_dir:
            issues.append(
                f"duplicate solution dump dirs for scenario={scenario}, mode={run_mode}, tag={workspace_tag}: {prev} vs {tag_dir}"
            )
            continue
        dirs[scenario] = tag_dir

    return dirs, issues


def collect_quality_metrics(workspace_root: Path) -> Tuple[dict, List[dict], List[str]]:
    by_scenario: Dict[str, dict] = {}
    records: List[dict] = []
    issues: List[str] = []

    if not workspace_root.exists():
        return {
            "overall": {
                "rel_l2_max": None,
                "abs_linf_max": None,
                "nan_inf_count": 0,
                "record_count": 0,
            },
            "by_scenario": {},
        }, records, issues

    reference_dirs, ref_issues = _collect_solution_dirs_by_tag(workspace_root,
                                                               "QualityReference",
                                                               "fp64_ref")
    compare_dirs, cmp_issues = _collect_solution_dirs_by_tag(workspace_root,
                                                             "QualityCompare",
                                                             "path1_cmp")
    issues.extend(ref_issues)
    issues.extend(cmp_issues)

    scenarios = sorted(set(reference_dirs) | set(compare_dirs))
    for scenario in scenarios:
        ref_dir = reference_dirs.get(scenario)
        cmp_dir = compare_dirs.get(scenario)
        if ref_dir is None or cmp_dir is None:
            issues.append(
                f"missing solution dump dir for scenario={scenario}: reference={ref_dir} compare={cmp_dir}"
            )
            continue

        try:
            metrics = collect_solution_dir_metrics(ref_dir, cmp_dir)
        except Exception as exc:
            issues.append(
                f"failed to compare solution dumps for scenario={scenario}: {exc}"
            )
            continue

        by_scenario[scenario] = {
            "rel_l2_max": metrics["rel_l2_max"],
            "abs_linf_max": metrics["abs_linf_max"],
            "nan_inf_count": metrics["nan_inf_count"],
            "record_count": metrics["record_count"],
            "missing_in_compare_count": metrics.get("missing_in_compare_count", 0),
            "missing_in_reference_count": metrics.get("missing_in_reference_count", 0),
            "reference_dir": metrics["reference_dir"],
            "compare_dir": metrics["compare_dir"],
        }
        if metrics.get("missing_in_compare_count", 0) or metrics.get("missing_in_reference_count", 0):
            issues.append(
                f"solution dump mismatch for scenario={scenario}: missing_in_compare={metrics.get('missing_in_compare_count', 0)} missing_in_reference={metrics.get('missing_in_reference_count', 0)}"
            )
        for record in metrics["records"]:
            records.append({"scenario": scenario, **record})

    overall = {
        "rel_l2_max": None,
        "abs_linf_max": None,
        "nan_inf_count": 0,
        "record_count": 0,
    }
    if by_scenario:
        overall["rel_l2_max"] = max(v["rel_l2_max"] for v in by_scenario.values())
        overall["abs_linf_max"] = max(v["abs_linf_max"] for v in by_scenario.values())
        overall["nan_inf_count"] = sum(v["nan_inf_count"] for v in by_scenario.values())
        overall["record_count"] = sum(v["record_count"] for v in by_scenario.values())

    return {"overall": overall, "by_scenario": by_scenario}, records, issues


def write_markdown(path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append("# Mixed Stage2 Benchmark Summary")
    lines.append("")
    lines.append("## Performance: Path1 vs FP64")
    lines.append("")
    lines.append("| Case | FP64 (ms) | Path1 (ms) | Delta (%) | Threshold (%) | Warning |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in summary["comparisons"]["path1_vs_fp64"]:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:.2f}"
        lines.append(
            f"| {row['name']} | {row['fp64_ms']:.4f} | {row['path1_ms']:.4f} | {delta} | {row['threshold_pct']:.1f} | {'YES' if row['warning'] else 'NO'} |"
        )

    lines.append("")
    lines.append("## Telemetry Overhead")
    lines.append("")
    lines.append("| Level | Case(On) | Off (ms) | On (ms) | Delta (%) | Warning |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in summary["comparisons"]["telemetry_overhead"]:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:.2f}"
        lines.append(
            f"| {row['level']} | {row['on_name']} | {row['off_ms']:.4f} | {row['on_ms']:.4f} | {delta} | {'YES' if row['warning'] else 'NO'} |"
        )

    lines.append("")
    lines.append("## Quality (Solution Dump Comparison)")
    lines.append("")
    lines.append("| Scenario | rel_l2.max | abs_linf.max | nan_inf_count | records | Warning |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for scenario, stat in sorted(summary["quality"]["by_scenario"].items()):
        warn = (
            (stat["rel_l2_max"] is not None and stat["rel_l2_max"] > 1e-5)
            or (stat["abs_linf_max"] is not None and stat["abs_linf_max"] > 5e-4)
            or (stat["nan_inf_count"] > 0)
        )
        lines.append(
            f"| {scenario} | {stat['rel_l2_max']:.6e} | {stat['abs_linf_max']:.6e} | {stat['nan_inf_count']} | {stat['record_count']} | {'YES' if warn else 'NO'} |"
        )

    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if not summary["warnings"]:
        lines.append("- none")
    else:
        for warn in summary["warnings"]:
            lines.append(f"- {warn}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Stage2 mixed benchmark results.")
    parser.add_argument("--fp64_perf", required=True, type=Path, help="fp64 perf benchmark json")
    parser.add_argument("--path1_perf", required=True, type=Path, help="path1 perf benchmark json")
    parser.add_argument("--fp64_quality_reference", required=True, type=Path, help="fp64 quality reference benchmark json")
    parser.add_argument("--path1_quality_compare", required=True, type=Path, help="path1 quality compare benchmark json")
    parser.add_argument("--workspace_root", required=True, type=Path, help="workspace root for solution dump files")
    parser.add_argument("--out_dir", required=True, type=Path, help="output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fp64_perf_entries, fp64_perf_failures = load_benchmark_file(args.fp64_perf)
    path1_perf_entries, path1_perf_failures = load_benchmark_file(args.path1_perf)
    fp64_ref_entries, fp64_ref_failures = load_benchmark_file(args.fp64_quality_reference)
    path1_cmp_entries, path1_cmp_failures = load_benchmark_file(args.path1_quality_compare)

    path_cmp = compare_path1_vs_fp64(fp64_perf_entries, path1_perf_entries)
    tele_cmp = compare_telemetry_overhead("fp64", fp64_perf_entries)
    tele_cmp.extend(compare_telemetry_overhead("path1", path1_perf_entries))

    quality, raw_quality_records, quality_issues = collect_quality_metrics(args.workspace_root)

    warnings: List[str] = []
    for row in path_cmp:
        if row["warning"]:
            warnings.append(
                f"path1 slower than fp64 by {row['delta_pct']:.2f}% (> {row['threshold_pct']:.1f}%) for {row['name']}"
            )
    for row in tele_cmp:
        if row["warning"]:
            warnings.append(
                f"telemetry overhead {row['delta_pct']:.2f}% (> {row['threshold_pct']:.1f}%) for {row['on_name']}"
            )

    for scenario, stat in quality["by_scenario"].items():
        if stat["rel_l2_max"] > 1e-5:
            warnings.append(
                f"quality rel_l2_x.max {stat['rel_l2_max']:.6e} > 1e-5 in {scenario}"
            )
        if stat["abs_linf_max"] > 5e-4:
            warnings.append(
                f"quality abs_linf_x.max {stat['abs_linf_max']:.6e} > 5e-4 in {scenario}"
            )
        if stat["nan_inf_count"] > 0:
            warnings.append(f"quality nan_inf_count={stat['nan_inf_count']} in {scenario}")
    warnings.extend(quality_issues)

    for fail in fp64_perf_failures:
        warnings.append(f"fp64 perf benchmark failed: {fail['name']} -> {fail['error']}")
    for fail in path1_perf_failures:
        warnings.append(f"path1 perf benchmark failed: {fail['name']} -> {fail['error']}")
    for fail in fp64_ref_failures:
        warnings.append(f"fp64 quality reference benchmark failed: {fail['name']} -> {fail['error']}")
    for fail in path1_cmp_failures:
        warnings.append(f"path1 quality compare benchmark failed: {fail['name']} -> {fail['error']}")

    summary = {
        "inputs": {
            "fp64_perf": str(args.fp64_perf),
            "path1_perf": str(args.path1_perf),
            "fp64_quality_reference": str(args.fp64_quality_reference),
            "path1_quality_compare": str(args.path1_quality_compare),
            "workspace_root": str(args.workspace_root),
        },
        "levels": {
            "fp64_perf": {
                "benchmark_count": len(fp64_perf_entries),
                "times_ms": fp64_perf_entries,
            },
            "path1_perf": {
                "benchmark_count": len(path1_perf_entries),
                "times_ms": path1_perf_entries,
            },
            "fp64_quality_reference": {
                "benchmark_count": len(fp64_ref_entries),
                "times_ms": fp64_ref_entries,
            },
            "path1_quality_compare": {
                "benchmark_count": len(path1_cmp_entries),
                "times_ms": path1_cmp_entries,
            },
        },
        "comparisons": {
            "path1_vs_fp64": path_cmp,
            "telemetry_overhead": tele_cmp,
        },
        "quality": quality,
        "quality_raw_records": raw_quality_records,
        "warnings": warnings,
    }

    summary_json = args.out_dir / "summary_stage2.json"
    summary_md = args.out_dir / "summary_stage2.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary_md, summary)

    print(f"[aggregate_stage2] wrote {summary_json}")
    print(f"[aggregate_stage2] wrote {summary_md}")
    print(f"[aggregate_stage2] warning_count={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
