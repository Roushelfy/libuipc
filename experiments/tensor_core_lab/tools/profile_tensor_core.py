from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


METRICS = (
    "smsp__inst_executed_pipe_tensor.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
)

APPROVED_CASES = (
    ("fem12_local", "fp32_no_tc", "BM_Fem12Fp32/4096/2/manual_time"),
    ("fem12_local", "tc32_tf32", "BM_Fem12Tc32/4096/2/manual_time"),
    ("joint24_local", "fp32_no_tc", "BM_Joint24Fp32/2048/2/manual_time"),
    ("joint24_local", "tc32_tf32", "BM_Joint24Tc32/2048/2/manual_time"),
    ("abd12_factorize", "fp32_no_tc", "BM_Abd12FactorFp32/4096/2/manual_time"),
    ("abd12_factorize", "tc32_tf32", "BM_Abd12FactorTc32/4096/2/manual_time"),
    ("mas48_factorize", "fp32_no_tc", "BM_Mas48FactorFp32/256/2/manual_time"),
    ("mas48_factorize", "tc32_tf32", "BM_Mas48FactorTc32/256/2/manual_time"),
)


def find_bench_exe(build_dir: Path, config: str) -> Path:
    exe_name = "tensor_core_lab_bench.exe" if os.name == "nt" else "tensor_core_lab_bench"
    candidates = [
        build_dir / config / "bin" / exe_name,
        build_dir / "bin" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find {exe_name} under {build_dir}")


def find_ncu() -> Path | None:
    for candidate in ("ncu", "ncu.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)

    windows_candidates = [
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.1\ncu.exe"),
    ]
    for candidate in windows_candidates:
        if candidate.exists():
            return candidate
    return None


def metric_value_from_output(text: str, metric_name: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if metric_name not in line:
            continue

        csv_parts = [part.strip().strip('"') for part in raw_line.split(",")]
        if len(csv_parts) >= 2 and csv_parts[0] == metric_name:
            return csv_parts[1]

        parts = line.split()
        if metric_name in parts:
            idx = parts.index(metric_name)
            if idx + 1 < len(parts):
                return parts[idx + 1]

        if line.startswith(metric_name):
            return line[len(metric_name) :].strip(" ,:")
    return ""


def profile_case(
    ncu_path: Path, bench_exe: Path, benchmark_name: str, benchmark_min_time: str
) -> dict[str, object]:
    command = [
        str(ncu_path),
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(METRICS),
        str(bench_exe),
        f"--benchmark_filter=^{benchmark_name}$",
        f"--benchmark_min_time={benchmark_min_time}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")

    if "ERR_NVGPUCTRPERM" in combined_output:
        return {
            "status": "blocked_by_permissions",
            "return_code": completed.returncode,
            "metrics": {},
            "stderr_excerpt": combined_output.strip(),
        }

    metric_values = {metric: metric_value_from_output(combined_output, metric) for metric in METRICS}
    status = "ok" if completed.returncode == 0 else "failed"
    if status == "ok" and not any(metric_values.values()):
        status = "no_metrics_detected"

    return {
        "status": status,
        "return_code": completed.returncode,
        "metrics": metric_values,
        "stderr_excerpt": combined_output.strip() if status != "ok" else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile tensor_core_lab representative cases with Nsight Compute."
    )
    parser.add_argument("--build_dir", required=True, type=Path)
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--output",
        default=Path("output") / "tensor_core_lab" / "ncu_profile.json",
        type=Path,
    )
    parser.add_argument("--benchmark_min_time", default="0.05s")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    bench_exe = find_bench_exe(build_dir, args.config)
    ncu_path = find_ncu()

    output = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "build_dir": str(build_dir),
        "config": args.config,
        "bench_executable": str(bench_exe),
        "metrics": list(METRICS),
        "cases": [],
    }

    if ncu_path is None:
        output["status"] = "ncu_not_found"
        output["message"] = (
            "Nsight Compute executable was not found on PATH or in the default Windows install locations."
        )
    else:
        output["status"] = "ok"
        output["ncu_executable"] = str(ncu_path)
        for op_family, mode, benchmark_name in APPROVED_CASES:
            case_result = profile_case(ncu_path, bench_exe, benchmark_name, args.benchmark_min_time)
            output["cases"].append(
                {
                    "op_family": op_family,
                    "mode": mode,
                    "benchmark_name": benchmark_name,
                    **case_result,
                }
            )
        if any(case["status"] == "blocked_by_permissions" for case in output["cases"]):
            output["status"] = "blocked_by_permissions"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
