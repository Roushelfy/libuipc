from __future__ import annotations

import argparse
import csv
from pathlib import Path


def format_float(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit markdown summary from tensor_core_lab csv output.")
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(
            "| op_family | mode | impl_path | tc_req | tc_verified | batch | cond | time_us | speedup_vs_fp32 | speedup_vs_fp64 | rel_error | abs_linf | nan_inf_count | symmetry_error |\n"
        )
        f.write(
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            f.write(
                f"| {row.get('op_family', '')} | {row.get('mode', '')} | {row.get('impl_path', '')} | "
                f"{row.get('tensor_core_requested', '')} | {row.get('tensor_core_verified', '')} | {row.get('batch', '')} | "
                f"{format_float(row.get('condition_scale', ''))} | {format_float(row.get('time_us', ''))} | "
                f"{format_float(row.get('speedup_vs_fp32_no_tc', ''))} | "
                f"{format_float(row.get('speedup_vs_fp64_ref_no_tc', ''))} | "
                f"{format_float(row.get('rel_error', ''))} | {format_float(row.get('abs_linf', ''))} | "
                f"{format_float(row.get('nan_inf_count', ''))} | {format_float(row.get('symmetry_error', ''))} |\n"
            )


if __name__ == "__main__":
    main()
