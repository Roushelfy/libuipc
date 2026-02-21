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
            # Failed cases should not participate in timing comparisons.
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
        if not name.startswith("Mixed.Stage1.Advance20F."):
            continue
        if name not in path1:
            continue
        path1_ms = path1[name]
        delta_pct = None
        if fp64_ms > 0:
            delta_pct = (path1_ms - fp64_ms) / fp64_ms * 100.0
        warning = bool(delta_pct is not None and delta_pct > 15.0)
        rows.append(
            {
                "name": name,
                "fp64_ms": fp64_ms,
                "path1_ms": path1_ms,
                "delta_pct": delta_pct,
                "warning": warning,
                "threshold_pct": 15.0,
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


def write_markdown(path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append("# Mixed Stage1 Benchmark Summary")
    lines.append("")
    lines.append("## Path1 vs FP64 (Advance Cases)")
    lines.append("")
    lines.append("| Case | FP64 (ms) | Path1 (ms) | Delta (%) | Warning |")
    lines.append("|---|---:|---:|---:|---|")
    for row in summary["comparisons"]["path1_vs_fp64"]:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:.2f}"
        lines.append(
            f"| {row['name']} | {row['fp64_ms']:.4f} | {row['path1_ms']:.4f} | {delta} | {'YES' if row['warning'] else 'NO'} |"
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
    parser = argparse.ArgumentParser(description="Aggregate Stage1 mixed benchmark results.")
    parser.add_argument("--fp64", required=True, type=Path, help="fp64 benchmark json")
    parser.add_argument("--path1", required=True, type=Path, help="path1 benchmark json")
    parser.add_argument("--cuda_fp64", type=Path, default=None, help="optional cuda baseline json")
    parser.add_argument("--out_dir", required=True, type=Path, help="output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fp64_entries, fp64_failures = load_benchmark_file(args.fp64)
    path1_entries, path1_failures = load_benchmark_file(args.path1)

    cuda_entries = {}
    cuda_failures: List[dict] = []
    if args.cuda_fp64 is not None and args.cuda_fp64.exists():
        cuda_entries, cuda_failures = load_benchmark_file(args.cuda_fp64)

    path_cmp = compare_path1_vs_fp64(fp64_entries, path1_entries)
    tele_cmp = compare_telemetry_overhead("fp64", fp64_entries)
    tele_cmp.extend(compare_telemetry_overhead("path1", path1_entries))

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

    for fail in fp64_failures:
        warnings.append(f"fp64 benchmark failed: {fail['name']} -> {fail['error']}")
    for fail in path1_failures:
        warnings.append(f"path1 benchmark failed: {fail['name']} -> {fail['error']}")
    for fail in cuda_failures:
        warnings.append(f"cuda benchmark failed: {fail['name']} -> {fail['error']}")

    summary = {
        "inputs": {
            "fp64": str(args.fp64),
            "path1": str(args.path1),
            "cuda_fp64": str(args.cuda_fp64) if args.cuda_fp64 else None,
        },
        "levels": {
            "fp64": {
                "benchmark_count": len(fp64_entries),
                "times_ms": fp64_entries,
            },
            "path1": {
                "benchmark_count": len(path1_entries),
                "times_ms": path1_entries,
            },
            "cuda_fp64": {
                "benchmark_count": len(cuda_entries),
                "times_ms": cuda_entries,
            },
        },
        "comparisons": {
            "path1_vs_fp64": path_cmp,
            "telemetry_overhead": tele_cmp,
        },
        "warnings": warnings,
        "reserved": {
            "quality": {
                "rel_l2": None,
                "abs_linf": None,
            },
            "memory": {
                "peak_mb": None,
            },
        },
    }

    summary_json = args.out_dir / "summary.json"
    summary_md = args.out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary_md, summary)

    print(f"[aggregate_stage1] wrote {summary_json}")
    print(f"[aggregate_stage1] wrote {summary_md}")
    print(f"[aggregate_stage1] warning_count={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
