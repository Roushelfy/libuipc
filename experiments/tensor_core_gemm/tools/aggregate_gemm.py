from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from gemm_registry import find_shape, layout_variant_name, physical_dims


MODE_MAP = {
    "Fp64": "fp64_ref_no_tc",
    "Fp32": "fp32_no_tc",
    "Tc32": "tc32_tf32",
}

IMPL_PATH_MAP = {
    0: "baseline",
    1: "tc_blas",
    2: "tc_wmma",
}

TENSOR_CORE_VERIFIED_MAP = {
    0: "no",
    1: "yes",
    2: "blocked_by_permissions",
}

PATTERN = re.compile(
    r"^BM_Gemm(?P<layout>Raw|Padded)(?P<mode>Fp64|Fp32|Tc32)/(?P<m>\d+)/(?P<n>\d+)/(?P<k>\d+)/(?P<batch>\d+)/manual_time$"
)

FIELDNAMES = [
    "name",
    "shape_tag",
    "shape_group",
    "layout_variant",
    "m",
    "n",
    "k",
    "physical_m",
    "physical_n",
    "physical_k",
    "batch",
    "mode",
    "impl_path",
    "tensor_core_requested",
    "tensor_core_verified",
    "profile_status",
    "tensor_pipe_max_inst",
    "tensor_pipe_max_pct",
    "iterations",
    "time_us",
    "speedup_vs_fp32_no_tc",
    "speedup_vs_fp64_ref_no_tc",
    "rel_error",
    "abs_linf",
    "nan_inf_count",
    "logical_gflops",
    "physical_gflops",
    "padding_overhead_ratio",
]


def load_ncu_profile(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: dict[str, dict[str, object]] = {}
    for case in payload.get("cases", []):
        benchmark_name = case.get("benchmark_name")
        if not benchmark_name:
            continue
        out[str(benchmark_name)] = {
            "profile_status": str(case.get("status", "")),
            "tensor_pipe_any_nonzero": bool(case.get("tensor_pipe_any_nonzero", False)),
            "tensor_pipe_max_inst": float(case.get("tensor_pipe_max_inst", 0.0)),
            "tensor_pipe_max_pct": float(case.get("tensor_pipe_max_pct", 0.0)),
        }
    return out


def parse_name(name: str) -> dict[str, object]:
    match = PATTERN.match(name)
    if not match:
        raise ValueError(f"unsupported benchmark name format: {name}")

    groups = match.groupdict()
    m = int(groups["m"])
    n = int(groups["n"])
    k = int(groups["k"])
    layout = groups["layout"]
    shape = find_shape(m, n, k)
    physical_m, physical_n, physical_k = physical_dims(layout, m, n, k)
    return {
        "shape_tag": shape.tag,
        "shape_group": shape.group,
        "layout_variant": layout_variant_name(layout),
        "m": m,
        "n": n,
        "k": k,
        "physical_m": physical_m,
        "physical_n": physical_n,
        "physical_k": physical_k,
        "batch": int(groups["batch"]),
        "mode": MODE_MAP[groups["mode"]],
    }


def parse_impl_path(value: object) -> str:
    try:
        return IMPL_PATH_MAP[int(float(value))]
    except (KeyError, TypeError, ValueError):
        return "baseline"


def parse_tensor_core_requested(value: object) -> str:
    try:
        return "yes" if float(value) != 0.0 else "no"
    except (TypeError, ValueError):
        return "no"


def parse_tensor_core_verified(value: object) -> str:
    try:
        return TENSOR_CORE_VERIFIED_MAP[int(float(value))]
    except (KeyError, TypeError, ValueError):
        return "no"


def convert_to_us(value: float, unit: str) -> float:
    scale = {
        "ns": 1.0e-3,
        "us": 1.0,
        "ms": 1.0e3,
        "s": 1.0e6,
    }.get(unit)
    if scale is None:
        raise ValueError(f"unsupported time unit: {unit}")
    return value * scale


def safe_speedup(baseline_us: float | None, value_us: float) -> float | str:
    if baseline_us is None or baseline_us <= 0.0 or value_us <= 0.0:
        return ""
    return baseline_us / value_us


def safe_gflops(flops: float, time_us: float) -> float | str:
    if time_us <= 0.0:
        return ""
    return flops / (time_us * 1.0e-6) / 1.0e9


def format_float(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return str(value)
    return f"{float(value):.6g}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["shape_tag"]), str(row["layout_variant"])), []).append(row)

    with path.open("w", encoding="utf-8") as f:
        for (shape_tag, layout_variant) in sorted(grouped):
            f.write(f"## {shape_tag} ({layout_variant})\n\n")
            f.write(
                "| mode | impl_path | tc_req | tc_verified | profile_status | tensor_pipe_max_inst | tensor_pipe_max_pct | batch | time_us | speedup_vs_fp32 | speedup_vs_fp64 | rel_error | abs_linf | logical_gflops | physical_gflops | padding_overhead_ratio |\n"
            )
            f.write(
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            )
            for row in grouped[(shape_tag, layout_variant)]:
                f.write(
                    f"| {row['mode']} | {row['impl_path']} | {row['tensor_core_requested']} | "
                    f"{row['tensor_core_verified']} | {row['profile_status']} | "
                    f"{format_float(row['tensor_pipe_max_inst'])} | {format_float(row['tensor_pipe_max_pct'])} | "
                    f"{row['batch']} | {format_float(row['time_us'])} | "
                    f"{format_float(row['speedup_vs_fp32_no_tc'])} | {format_float(row['speedup_vs_fp64_ref_no_tc'])} | "
                    f"{format_float(row['rel_error'])} | {format_float(row['abs_linf'])} | "
                    f"{format_float(row['logical_gflops'])} | {format_float(row['physical_gflops'])} | "
                    f"{format_float(row['padding_overhead_ratio'])} |\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate tensor_core_gemm benchmark output.")
    parser.add_argument("--bench_json", required=True, type=Path)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--ncu_profile", default=None, type=Path)
    args = parser.parse_args()

    with args.bench_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ncu_profile = args.ncu_profile
    if ncu_profile is None:
        inferred_profile = args.run_root / "raw" / "ncu_profile.json"
        ncu_profile = inferred_profile if inferred_profile.exists() else None
    profile_data = load_ncu_profile(ncu_profile)

    tables = args.run_root / "tables"
    per_shape = args.run_root / "per_shape"
    tables.mkdir(parents=True, exist_ok=True)
    per_shape.mkdir(parents=True, exist_ok=True)
    for stale_csv in per_shape.glob("*.csv"):
        stale_csv.unlink()

    rows: list[dict[str, object]] = []
    for item in data.get("benchmarks", []):
        if item.get("run_type") != "iteration":
            continue

        parsed = parse_name(item.get("name", ""))
        time_us = convert_to_us(float(item.get("real_time", 0.0)), item.get("time_unit", "ns"))
        logical_flops = 2.0 * float(parsed["m"]) * float(parsed["n"]) * float(parsed["k"]) * float(parsed["batch"])
        physical_flops = (
            2.0
            * float(parsed["physical_m"])
            * float(parsed["physical_n"])
            * float(parsed["physical_k"])
            * float(parsed["batch"])
        )
        rows.append(
            {
                "name": item.get("name", ""),
                "shape_tag": parsed["shape_tag"],
                "shape_group": parsed["shape_group"],
                "layout_variant": parsed["layout_variant"],
                "m": parsed["m"],
                "n": parsed["n"],
                "k": parsed["k"],
                "physical_m": parsed["physical_m"],
                "physical_n": parsed["physical_n"],
                "physical_k": parsed["physical_k"],
                "batch": parsed["batch"],
                "mode": parsed["mode"],
                "impl_path": parse_impl_path(item.get("impl_path_id")),
                "tensor_core_requested": parse_tensor_core_requested(item.get("tensor_core_requested")),
                "tensor_core_verified": parse_tensor_core_verified(item.get("tensor_core_verified_id")),
                "profile_status": "",
                "tensor_pipe_max_inst": "",
                "tensor_pipe_max_pct": "",
                "iterations": item.get("iterations", 0),
                "time_us": time_us,
                "speedup_vs_fp32_no_tc": "",
                "speedup_vs_fp64_ref_no_tc": "",
                "rel_error": item.get("rel_error", item.get("rel_fro", "")),
                "abs_linf": item.get("abs_linf", ""),
                "nan_inf_count": item.get("nan_inf_count", 0),
                "logical_gflops": safe_gflops(logical_flops, time_us),
                "physical_gflops": safe_gflops(physical_flops, time_us),
                "padding_overhead_ratio": (physical_flops / logical_flops) if logical_flops > 0.0 else "",
            }
        )

    for row in rows:
        profile = profile_data.get(str(row["name"]))
        if not profile:
            continue
        row["profile_status"] = profile["profile_status"]
        row["tensor_pipe_max_inst"] = profile["tensor_pipe_max_inst"]
        row["tensor_pipe_max_pct"] = profile["tensor_pipe_max_pct"]

        status = str(profile["profile_status"])
        if status == "blocked_by_permissions":
            row["tensor_core_verified"] = "blocked_by_permissions"
        elif status == "ok":
            row["tensor_core_verified"] = (
                "yes" if bool(profile["tensor_pipe_any_nonzero"]) else "no"
            )

    rows.sort(
        key=lambda row: (
            str(row["shape_group"]),
            str(row["shape_tag"]),
            str(row["layout_variant"]),
            int(row["batch"]),
            str(row["mode"]),
        )
    )

    groups: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row["shape_tag"]), str(row["layout_variant"]), int(row["batch"]))
        groups.setdefault(key, {})[str(row["mode"])] = float(row["time_us"])

    for row in rows:
        key = (str(row["shape_tag"]), str(row["layout_variant"]), int(row["batch"]))
        baselines = groups[key]
        row["speedup_vs_fp32_no_tc"] = safe_speedup(baselines.get("fp32_no_tc"), float(row["time_us"]))
        row["speedup_vs_fp64_ref_no_tc"] = safe_speedup(
            baselines.get("fp64_ref_no_tc"), float(row["time_us"])
        )

    summary_csv = tables / "summary.csv"
    write_csv(summary_csv, rows)

    for shape_tag in sorted({str(row["shape_tag"]) for row in rows}):
        write_csv(per_shape / f"{shape_tag}.csv", [row for row in rows if row["shape_tag"] == shape_tag])

    write_markdown(tables / "summary.md", rows)


if __name__ == "__main__":
    main()
