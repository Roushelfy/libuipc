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
    parser = argparse.ArgumentParser(description="Emit markdown summary from tensor_core_gemm csv output.")
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(
            "| shape_tag | group | layout | mode | impl_path | tc_req | tc_verified | profile_status | tensor_pipe_max_inst | tensor_pipe_max_pct | batch | time_us | speedup_vs_fp32 | speedup_vs_fp64 | rel_error | abs_linf | logical_gflops | physical_gflops | padding_overhead_ratio |\n"
        )
        f.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            f.write(
                f"| {row.get('shape_tag', '')} | {row.get('shape_group', '')} | {row.get('layout_variant', '')} | "
                f"{row.get('mode', '')} | {row.get('impl_path', '')} | {row.get('tensor_core_requested', '')} | "
                f"{row.get('tensor_core_verified', '')} | {row.get('profile_status', '')} | "
                f"{format_float(row.get('tensor_pipe_max_inst', ''))} | {format_float(row.get('tensor_pipe_max_pct', ''))} | "
                f"{row.get('batch', '')} | "
                f"{format_float(row.get('time_us', ''))} | {format_float(row.get('speedup_vs_fp32_no_tc', ''))} | "
                f"{format_float(row.get('speedup_vs_fp64_ref_no_tc', ''))} | {format_float(row.get('rel_error', ''))} | "
                f"{format_float(row.get('abs_linf', ''))} | {format_float(row.get('logical_gflops', ''))} | "
                f"{format_float(row.get('physical_gflops', ''))} | {format_float(row.get('padding_overhead_ratio', ''))} |\n"
            )


if __name__ == "__main__":
    main()
