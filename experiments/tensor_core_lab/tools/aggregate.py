from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


MODE_MAP = {
    "Fp64": "fp64_ref_no_tc",
    "Fp32": "fp32_no_tc",
    "Tc32": "tc32_tf32",
}

PATTERNS = (
    (re.compile(r"^BM_Abd12Assemble(?P<mode>Fp64|Fp32|Tc32)/(?P<batch>\d+)/(?P<cond_exp>\d+)/manual_time$"), "abd12_assemble"),
    (re.compile(r"^BM_Fem12(?P<mode>Fp64|Fp32|Tc32)/(?P<batch>\d+)/(?P<cond_exp>\d+)/manual_time$"), "fem12_local"),
    (re.compile(r"^BM_Joint24(?P<mode>Fp64|Fp32|Tc32)/(?P<batch>\d+)/(?P<cond_exp>\d+)/manual_time$"), "joint24_local"),
    (re.compile(r"^BM_Abd12(?P<op>Factor|Inverse|Solve)(?P<mode>Fp64|Fp32|Tc32)/(?P<batch>\d+)/(?P<cond_exp>\d+)/manual_time$"), "abd12"),
    (re.compile(r"^BM_Mas48(?P<op>Factor|Inverse|Solve)(?P<mode>Fp64|Fp32|Tc32)/(?P<batch>\d+)/(?P<cond_exp>\d+)/manual_time$"), "mas48"),
)

FIELDNAMES = [
    "name",
    "op_family",
    "mode",
    "impl_path",
    "tensor_core_requested",
    "tensor_core_verified",
    "batch",
    "condition_exp",
    "condition_scale",
    "iterations",
    "time_us",
    "speedup_vs_fp32_no_tc",
    "speedup_vs_fp64_ref_no_tc",
    "rel_error",
    "abs_linf",
    "nan_inf_count",
    "symmetry_error",
]

OP_MAP = {
    "Factor": "factorize",
    "Inverse": "inverse",
    "Solve": "solve",
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


def parse_name(name: str) -> dict[str, object]:
    for pattern, family in PATTERNS:
        match = pattern.match(name)
        if not match:
            continue

        groups = match.groupdict()
        op = groups.get("op")
        op_family = family if op is None else f"{family}_{OP_MAP[op]}"
        cond_exp = int(groups["cond_exp"])
        return {
            "op_family": op_family,
            "mode": MODE_MAP[groups["mode"]],
            "batch": int(groups["batch"]),
            "condition_exp": cond_exp,
            "condition_scale": 10.0**cond_exp,
        }

    raise ValueError(f"unsupported benchmark name format: {name}")


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


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


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["op_family"]), []).append(row)

    with path.open("w", encoding="utf-8") as f:
        for op_family in sorted(grouped):
            f.write(f"## {op_family}\n\n")
            f.write(
                "| mode | impl_path | tc_req | tc_verified | batch | cond | time_us | speedup_vs_fp32 | speedup_vs_fp64 | rel_error | abs_linf | nan_inf_count | symmetry_error |\n"
            )
            f.write(
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            )
            for row in grouped[op_family]:
                f.write(
                    f"| {row['mode']} | {row['impl_path']} | {row['tensor_core_requested']} | "
                    f"{row['tensor_core_verified']} | {row['batch']} | {format_float(row['condition_scale'])} | "
                    f"{format_float(row['time_us'])} | {format_float(row['speedup_vs_fp32_no_tc'])} | "
                    f"{format_float(row['speedup_vs_fp64_ref_no_tc'])} | {format_float(row['rel_error'])} | "
                    f"{format_float(row['abs_linf'])} | {format_float(row['nan_inf_count'])} | "
                    f"{format_float(row['symmetry_error'])} |\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate tensor_core_lab benchmark output.")
    parser.add_argument("--bench_json", required=True, type=Path)
    parser.add_argument("--run_root", required=True, type=Path)
    args = parser.parse_args()

    with args.bench_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tables = args.run_root / "tables"
    per_op = args.run_root / "per_op"
    tables.mkdir(parents=True, exist_ok=True)
    per_op.mkdir(parents=True, exist_ok=True)
    for stale_csv in per_op.glob("*.csv"):
        stale_csv.unlink()

    rows = []
    for item in data.get("benchmarks", []):
        if item.get("run_type") != "iteration":
            continue

        parsed = parse_name(item.get("name", ""))
        rows.append(
            {
                "name": item.get("name", ""),
                "op_family": parsed["op_family"],
                "mode": parsed["mode"],
                "impl_path": parse_impl_path(item.get("impl_path_id")),
                "tensor_core_requested": parse_tensor_core_requested(item.get("tensor_core_requested")),
                "tensor_core_verified": parse_tensor_core_verified(
                    item.get("tensor_core_verified_id")
                ),
                "batch": parsed["batch"],
                "condition_exp": parsed["condition_exp"],
                "condition_scale": parsed["condition_scale"],
                "iterations": item.get("iterations", 0),
                "time_us": convert_to_us(float(item.get("real_time", 0.0)), item.get("time_unit", "ns")),
                "speedup_vs_fp32_no_tc": "",
                "speedup_vs_fp64_ref_no_tc": "",
                "rel_error": item.get("rel_error", item.get("rel_fro", "")),
                "abs_linf": item.get("abs_linf", ""),
                "nan_inf_count": item.get("nan_inf_count", 0),
                "symmetry_error": item.get("symmetry_error", ""),
            }
        )

    rows.sort(key=lambda row: (str(row["op_family"]), int(row["batch"]), int(row["condition_exp"]), str(row["mode"])))

    groups: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row["op_family"]), int(row["batch"]), int(row["condition_exp"]))
        groups.setdefault(key, {})[str(row["mode"])] = float(row["time_us"])

    for row in rows:
        key = (str(row["op_family"]), int(row["batch"]), int(row["condition_exp"]))
        baselines = groups[key]
        row["speedup_vs_fp32_no_tc"] = safe_speedup(baselines.get("fp32_no_tc"), float(row["time_us"]))
        row["speedup_vs_fp64_ref_no_tc"] = safe_speedup(
            baselines.get("fp64_ref_no_tc"), float(row["time_us"])
        )

    csv_path = tables / "summary.csv"
    write_csv(csv_path, rows)

    for op_family in sorted({str(row["op_family"]) for row in rows}):
        write_csv(per_op / f"{op_family}.csv", [row for row in rows if row["op_family"] == op_family])

    md_path = tables / "summary.md"
    write_markdown(md_path, rows)


if __name__ == "__main__":
    main()
