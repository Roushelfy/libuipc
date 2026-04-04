from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from gemm_registry import ALL_SHAPES, REPRESENTATIVE_PROFILE_SHAPES, benchmark_name, smoke_batches

METRICS = [
    "smsp__inst_executed_pipe_tensor.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
]

TENSOR_INST_RE = re.compile(
    r"smsp__inst_executed_pipe_tensor\.sum\s+inst\s+([0-9,]+(?:\.[0-9]+)?)"
)
TENSOR_PCT_RE = re.compile(
    r"sm__pipe_tensor_cycles_active\.avg\.pct_of_peak_sustained_active\s+%\s+([0-9]+(?:\.[0-9]+)?)"
)


def find_bench_exe(build_dir: Path, config: str) -> Path:
    exe_name = "tensor_core_gemm_bench.exe" if os.name == "nt" else "tensor_core_gemm_bench"
    candidates = [
        build_dir / config / "bin" / exe_name,
        build_dir / "bin" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find {exe_name} under {build_dir}")


def find_ncu() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.bat"),
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.exe"),
        Path("ncu"),
    ]
    for candidate in candidates:
        if candidate.exists() or str(candidate) == "ncu":
            return candidate
    return None


def profile_case(ncu_path: Path, bench_exe: Path, benchmark_name_value: str, benchmark_min_time: str) -> dict[str, object]:
    command = [
        str(ncu_path),
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--metrics",
        ",".join(METRICS),
        str(bench_exe),
        f"--benchmark_filter=^{benchmark_name_value}$",
        f"--benchmark_min_time={benchmark_min_time}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = stdout + "\n" + stderr
    tensor_inst_values = [float(value.replace(",", "")) for value in TENSOR_INST_RE.findall(stdout)]
    tensor_pct_values = [float(value) for value in TENSOR_PCT_RE.findall(stdout)]
    max_inst = max(tensor_inst_values) if tensor_inst_values else 0.0
    max_pct = max(tensor_pct_values) if tensor_pct_values else 0.0
    tensor_pipe_any_nonzero = any(value != 0.0 for value in tensor_inst_values) or any(
        value != 0.0 for value in tensor_pct_values
    )
    if "ERR_NVGPUCTRPERM" in combined:
        return {
            "status": "blocked_by_permissions",
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "tensor_pipe_any_nonzero": False,
            "tensor_pipe_max_inst": 0.0,
            "tensor_pipe_max_pct": 0.0,
        }
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "tensor_pipe_any_nonzero": tensor_pipe_any_nonzero,
        "tensor_pipe_max_inst": max_inst,
        "tensor_pipe_max_pct": max_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile representative tensor_core_gemm cases with Nsight Compute.")
    parser.add_argument("--build_dir", required=True, type=Path)
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--output",
        default=Path("output") / "tensor_core_gemm" / "manual" / "raw" / "ncu_profile.json",
        type=Path,
    )
    parser.add_argument("--benchmark_min_time", default="0.05s")
    args = parser.parse_args()

    bench_exe = find_bench_exe(args.build_dir.resolve(), args.config)
    ncu_path = find_ncu()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if ncu_path is None:
        payload = {"status": "ncu_not_found", "cases": []}
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return

    shape_map = {(shape.m, shape.n, shape.k): shape for shape in ALL_SHAPES}
    cases: list[dict[str, object]] = []
    for tag, (layout, m, n, k) in REPRESENTATIVE_PROFILE_SHAPES.items():
        shape = shape_map[(m, n, k)]
        batch = smoke_batches(shape)[0]
        for mode in ("Fp32", "Tc32"):
            name = benchmark_name(layout, mode, shape, batch)
            result = profile_case(ncu_path, bench_exe, name, args.benchmark_min_time)
            cases.append(
                {
                    "shape_tag": tag,
                    "layout": layout.lower(),
                    "mode": mode,
                    "benchmark_name": name,
                    **result,
                }
            )

    overall_status = "ok"
    case_statuses = {case["status"] for case in cases}
    if case_statuses == {"blocked_by_permissions"}:
        overall_status = "blocked_by_permissions"
    elif "failed" in case_statuses:
        overall_status = "failed"

    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"status": overall_status, "cases": cases}, f, indent=2)


if __name__ == "__main__":
    main()
