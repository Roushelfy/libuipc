#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


UNIT_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
}

LEVELS = ("fp64", "path1", "path2", "path3", "path4", "path5", "path6", "path7")
MIXED_COMPARE_LEVELS = ("path1", "path2", "path3", "path4", "path5", "path6", "path7")
KNOWN_SCENARIOS = (
    "wrecking_ball",
    "fem_ground_contact",
    "fem_heavy_nocontact",
    "fem_heavy_ground_contact",
)

PERF_SCENARIO_THRESHOLDS = {
    "wrecking_ball": 20.0,
    "fem_heavy_nocontact": 10.0,
    "fem_heavy_ground_contact": 15.0,
    "fem_ground_contact": 20.0,
}
TELEMETRY_THRESHOLD_PCT = 25.0
QUALITY_REL_L2_WARN = 1e-5
QUALITY_ABS_LINF_WARN = 5e-4


def to_ms(value: float, unit: str) -> float:
    return float(value) * UNIT_TO_MS.get(unit, 1e-6)


def load_gbench(path: Path) -> Tuple[Dict[str, float], List[dict]]:
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
        val_ms = to_ms(item["real_time"], item.get("time_unit", "ns"))
        # gbench may emit aggregates/repetitions; keep the minimum to reduce jitter.
        if name not in entries or val_ms < entries[name]:
            entries[name] = val_ms
    return entries, failures


def infer_scenario(name_or_path: str) -> str:
    lower = name_or_path.lower()
    for s in KNOWN_SCENARIOS:
        if s in lower:
            return s
    return "unknown"


def compare_perf(level: str, fp64_perf: Dict[str, float], level_perf: Dict[str, float]) -> List[dict]:
    rows: List[dict] = []
    for name, fp64_ms in fp64_perf.items():
        if not name.startswith("Mixed.Stage2.Perf.") or ".cuda_mixed" not in name:
            continue
        level_ms = level_perf.get(name)
        if level_ms is None:
            continue
        delta_pct = (level_ms - fp64_ms) / fp64_ms * 100.0 if fp64_ms > 0 else None
        rows.append(
            {
                "level": level,
                "name": name,
                "scenario": infer_scenario(name),
                "fp64_ms": fp64_ms,
                "level_ms": level_ms,
                "delta_pct": delta_pct,
            }
        )
    rows.sort(key=lambda x: x["name"])
    return rows


def telemetry_overhead(level: str, perf_entries: Dict[str, float]) -> List[dict]:
    rows: List[dict] = []
    for on_name, on_ms in perf_entries.items():
        if ".TelemetryOn." not in on_name:
            continue
        off_name = on_name.replace(".TelemetryOn.", ".TelemetryOff.")
        off_ms = perf_entries.get(off_name)
        if off_ms is None:
            continue
        delta_pct = (on_ms - off_ms) / off_ms * 100.0 if off_ms > 0 else None
        rows.append(
            {
                "level": level,
                "on_name": on_name,
                "off_name": off_name,
                "off_ms": off_ms,
                "on_ms": on_ms,
                "delta_pct": delta_pct,
                "scenario": infer_scenario(on_name),
            }
        )
    rows.sort(key=lambda x: x["on_name"])
    return rows


def collect_quality(workspace_root: Path) -> dict:
    result = {
        "overall_by_level": {},
        "by_level": {},
        "raw_records": [],
    }
    if not workspace_root.exists():
        return result

    by_level: Dict[str, Dict[str, dict]] = {}
    raw: List[dict] = []

    for error_file in workspace_root.rglob("error.jsonl"):
        path_str = str(error_file)
        level = "unknown"
        for lv in MIXED_COMPARE_LEVELS:
            token = f"{lv}_cmp"
            if token in path_str:
                level = lv
                break

        scenario = infer_scenario(path_str)
        level_stats = by_level.setdefault(level, {})
        stat = level_stats.setdefault(
            scenario,
            {
                "rel_l2_max": 0.0,
                "abs_linf_max": 0.0,
                "nan_inf_count": 0,
                "record_count": 0,
                "files": [],
            },
        )
        stat["files"].append(path_str)

        for line in error_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rel_l2 = float(obj.get("rel_l2_x", 0.0))
            abs_linf = float(obj.get("abs_linf_x", 0.0))
            nan_inf = bool(obj.get("nan_inf_flag", False))

            stat["rel_l2_max"] = max(stat["rel_l2_max"], rel_l2)
            stat["abs_linf_max"] = max(stat["abs_linf_max"], abs_linf)
            stat["nan_inf_count"] += 1 if nan_inf else 0
            stat["record_count"] += 1

            raw.append(
                {
                    "level": level,
                    "scenario": scenario,
                    "file": path_str,
                    "frame": obj.get("frame"),
                    "newton_iter": obj.get("newton_iter"),
                    "rel_l2_x": rel_l2,
                    "abs_linf_x": abs_linf,
                    "nan_inf_flag": nan_inf,
                }
            )

    overall_by_level: Dict[str, dict] = {}
    for level, scenarios in by_level.items():
        if not scenarios:
            continue
        overall_by_level[level] = {
            "rel_l2_max": max(s["rel_l2_max"] for s in scenarios.values()),
            "abs_linf_max": max(s["abs_linf_max"] for s in scenarios.values()),
            "nan_inf_count": sum(s["nan_inf_count"] for s in scenarios.values()),
            "record_count": sum(s["record_count"] for s in scenarios.values()),
        }

    result["by_level"] = by_level
    result["overall_by_level"] = overall_by_level
    result["raw_records"] = raw
    return result


def summarize_stage1(stage1_root: Path) -> dict:
    summary = {"levels": {}, "warnings": []}
    for level in LEVELS:
        gbench = stage1_root / level / "gbench.json"
        if not gbench.exists():
            continue
        entries, failures = load_gbench(gbench)
        summary["levels"][level] = {
            "benchmark_count": len(entries),
            "times_ms": entries,
            "failures": failures,
        }
        for fail in failures:
            summary["warnings"].append(f"stage1 {level} failed: {fail['name']} -> {fail['error']}")
    return summary


def write_markdown(out_path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append("# Mixed Stage2 All-Paths Audit Summary")
    lines.append("")
    lines.append("## Path Semantics")
    lines.append("")
    lines.append("- `path1 = ALU (ALU=float, Store=double, PCG=double)`")
    lines.append("- `path2 = Store (ALU=double, Store=float, PCG=double)`")
    lines.append("- `path3 = ALU + Store (ALU=float, Store=float, PCG=double)`")
    lines.append("- `path4 = Store + PCG (ALU=double, Store=float, PCG=float)`")
    lines.append("- `path5 = ALU + Store + PCG (ALU=float, Store=float, PCG=float)`")
    lines.append("- `path6 = ALU + Store + PCG (ALU=float, Store=float, PCG=float) + preconditioner_no_double_intermediate (phase-1)`")
    lines.append("- `path7 = ALU + Store + PCG + full_pcg_fp32 (ALU=float, Store=float, PCG=float, Solve=float, Iter=float)`")
    lines.append("")

    lines.append("## Stage1 Smoke")
    lines.append("")
    lines.append("| Level | Benchmarks | Failures |")
    lines.append("|---|---:|---:|")
    for level in LEVELS:
        info = summary["stage1"]["levels"].get(level)
        if not info:
            continue
        lines.append(f"| {level} | {info['benchmark_count']} | {len(info['failures'])} |")
    lines.append("")

    lines.append("## Stage2 Performance vs FP64 (`cuda_mixed`) ")
    lines.append("")
    lines.append("| Level | Case | FP64 (ms) | Level (ms) | Delta (%) |")
    lines.append("|---|---|---:|---:|---:|")
    for level in MIXED_COMPARE_LEVELS:
        for row in summary["stage2"]["comparisons"]["perf_vs_fp64"].get(level, []):
            delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:.2f}"
            lines.append(
                f"| {level} | {row['name']} | {row['fp64_ms']:.3f} | {row['level_ms']:.3f} | {delta} |"
            )
    lines.append("")

    lines.append("## Stage2 Telemetry Overhead")
    lines.append("")
    lines.append("| Level | Case(On) | Off (ms) | On (ms) | Delta (%) |")
    lines.append("|---|---|---:|---:|---:|")
    for level in LEVELS:
        for row in summary["stage2"]["comparisons"]["telemetry_overhead"].get(level, []):
            delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:.2f}"
            lines.append(
                f"| {level} | {row['on_name']} | {row['off_ms']:.3f} | {row['on_ms']:.3f} | {delta} |"
            )
    lines.append("")

    lines.append("## Stage2 Quality (Offline ErrorTracker)")
    lines.append("")
    lines.append("| Level | Scenario | rel_l2.max | abs_linf.max | nan_inf_count | records |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for level in MIXED_COMPARE_LEVELS:
        for scenario, stat in sorted(summary["stage2"]["quality"]["by_level"].get(level, {}).items()):
            lines.append(
                f"| {level} | {scenario} | {stat['rel_l2_max']:.6e} | {stat['abs_linf_max']:.6e} | {stat['nan_inf_count']} | {stat['record_count']} |"
            )
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if not summary["warnings"]:
        lines.append("- none")
    else:
        for w in summary["warnings"]:
            lines.append(f"- {w}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Stage2 benchmark results for fp64/path1/path2/path3/path4/path5/path6/path7")
    parser.add_argument("--run_root", required=True, type=Path, help="benchmark output root (contains stage1/, stage2/, workspaces/)")
    args = parser.parse_args()

    run_root = args.run_root
    stage1_root = run_root / "stage1"
    stage2_root = run_root / "stage2"
    workspace_root = run_root / "workspaces"

    perf_entries: Dict[str, Dict[str, float]] = {}
    perf_failures: Dict[str, List[dict]] = {}
    cmp_entries: Dict[str, Dict[str, float]] = {}
    cmp_failures: Dict[str, List[dict]] = {}
    fp64_ref_entries: Dict[str, float] = {}
    fp64_ref_failures: List[dict] = []

    for level in LEVELS:
        perf_file = stage2_root / level / "perf" / "gbench.json"
        if perf_file.exists():
            perf_entries[level], perf_failures[level] = load_gbench(perf_file)

    ref_file = stage2_root / "fp64" / "quality_reference" / "gbench.json"
    if ref_file.exists():
        fp64_ref_entries, fp64_ref_failures = load_gbench(ref_file)

    for level in MIXED_COMPARE_LEVELS:
        cmp_file = stage2_root / level / "quality_compare" / "gbench.json"
        if cmp_file.exists():
            cmp_entries[level], cmp_failures[level] = load_gbench(cmp_file)

    perf_vs_fp64 = {}
    tele_overhead = {}
    if "fp64" in perf_entries:
        for level in MIXED_COMPARE_LEVELS:
            if level in perf_entries:
                perf_vs_fp64[level] = compare_perf(level, perf_entries["fp64"], perf_entries[level])
        for level, entries in perf_entries.items():
            tele_overhead[level] = telemetry_overhead(level, entries)

    quality = collect_quality(workspace_root)
    stage1_summary = summarize_stage1(stage1_root)

    warnings: List[str] = []
    for level, failures in perf_failures.items():
        for fail in failures:
            warnings.append(f"stage2 perf {level} failed: {fail['name']} -> {fail['error']}")
    for fail in fp64_ref_failures:
        warnings.append(f"stage2 fp64 quality reference failed: {fail['name']} -> {fail['error']}")
    for level, failures in cmp_failures.items():
        for fail in failures:
            warnings.append(f"stage2 quality compare {level} failed: {fail['name']} -> {fail['error']}")

    for level, ov in quality["overall_by_level"].items():
        if ov["nan_inf_count"] > 0:
            warnings.append(f"{level}: nan_inf_count={ov['nan_inf_count']}")

    for level, rows in perf_vs_fp64.items():
        for row in rows:
            if row["delta_pct"] is None:
                continue
            threshold = PERF_SCENARIO_THRESHOLDS.get(row["scenario"], 20.0)
            if row["delta_pct"] > threshold:
                warnings.append(
                    f"{level} slower than fp64 by {row['delta_pct']:.2f}% (> {threshold:.1f}%) for {row['name']}"
                )

    for level, rows in tele_overhead.items():
        for row in rows:
            if row["delta_pct"] is not None and row["delta_pct"] > TELEMETRY_THRESHOLD_PCT:
                warnings.append(
                    f"{level} telemetry overhead {row['delta_pct']:.2f}% (> {TELEMETRY_THRESHOLD_PCT:.1f}%) for {row['on_name']}"
                )

    for level, scenario_map in quality["by_level"].items():
        for scenario, stat in scenario_map.items():
            if stat["abs_linf_max"] > QUALITY_ABS_LINF_WARN:
                warnings.append(
                    f"{level} {scenario}: abs_linf.max {stat['abs_linf_max']:.6e} > {QUALITY_ABS_LINF_WARN:.1e}"
                )
            if stat["rel_l2_max"] > QUALITY_REL_L2_WARN:
                warnings.append(
                    f"{level} {scenario}: rel_l2.max {stat['rel_l2_max']:.6e} > {QUALITY_REL_L2_WARN:.1e}"
                )
            # Detect denominator-collapse style inflation: huge rel_l2 but tiny absolute error.
            if stat["rel_l2_max"] > 1.0 and stat["abs_linf_max"] < 1e-5:
                warnings.append(
                    f"{level} {scenario}: rel_l2 likely inflated by near-zero reference norm (rel_l2={stat['rel_l2_max']:.3e}, abs_linf={stat['abs_linf_max']:.3e})"
                )

    summary = {
        "run_root": str(run_root),
        "stage1": stage1_summary,
        "stage2": {
            "levels": {
                "perf": {k: {"benchmark_count": len(v), "times_ms": v} for k, v in perf_entries.items()},
                "fp64_quality_reference": {
                    "benchmark_count": len(fp64_ref_entries),
                    "times_ms": fp64_ref_entries,
                    "failures": fp64_ref_failures,
                },
                "quality_compare": {k: {"benchmark_count": len(v), "times_ms": v} for k, v in cmp_entries.items()},
            },
            "comparisons": {
                "perf_vs_fp64": perf_vs_fp64,
                "telemetry_overhead": tele_overhead,
            },
            "quality": quality,
        },
        "warnings": warnings + stage1_summary["warnings"],
    }

    out_json = run_root / "summary_all_paths.json"
    out_md = run_root / "summary_all_paths.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_md, summary)
    print(f"[aggregate_stage2_all_paths] wrote {out_json}")
    print(f"[aggregate_stage2_all_paths] wrote {out_md}")
    print(f"[aggregate_stage2_all_paths] warning_count={len(summary['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
