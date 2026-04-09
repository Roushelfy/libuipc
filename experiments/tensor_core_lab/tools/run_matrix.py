from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROFILE_FILTERS = {
    "smoke": r"^.*/2/manual_time$",
    "full": r".*",
}


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


def find_test_exe(build_dir: Path, config: str) -> Path:
    exe_name = "tensor_core_lab_test.exe" if os.name == "nt" else "tensor_core_lab_test"
    candidates = [
        build_dir / config / "bin" / exe_name,
        build_dir / "bin" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find {exe_name} under {build_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tensor_core_lab benchmark matrix.")
    parser.add_argument("--build_dir", required=True, type=Path)
    parser.add_argument("--config", default="Release")
    parser.add_argument("--run_root", default=None, type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILE_FILTERS), default="smoke")
    parser.add_argument("--benchmark_filter", default=None)
    parser.add_argument("--skip_tests", action="store_true")
    args = parser.parse_args()

    root = (
        args.run_root.resolve()
        if args.run_root
        else Path("output") / "tensor_core_lab" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    raw_dir = root / "raw"
    tables_dir = root / "tables"
    per_op_dir = root / "per_op"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_op_dir.mkdir(parents=True, exist_ok=True)

    build_dir = args.build_dir.resolve()
    bench_exe = find_bench_exe(build_dir, args.config)
    test_exe = find_test_exe(build_dir, args.config)
    out_json = raw_dir / "bench.json"
    test_log = raw_dir / "test.log"
    benchmark_filter = args.benchmark_filter or PROFILE_FILTERS[args.profile]

    if not args.skip_tests:
        with test_log.open("w", encoding="utf-8") as f:
            subprocess.run([str(test_exe)], check=True, stdout=f, stderr=subprocess.STDOUT)

    bench_cmd = [
        str(bench_exe),
        f"--benchmark_filter={benchmark_filter}",
        f"--benchmark_out={out_json}",
        "--benchmark_out_format=json",
    ]
    bench_result = subprocess.run(bench_cmd, check=False)
    if bench_result.returncode != 0:
        raise RuntimeError(f"benchmark process failed with exit code {bench_result.returncode}")
    if not out_json.exists() or out_json.stat().st_size == 0:
        raise RuntimeError("benchmark did not produce a valid bench.json; check the benchmark filter")

    try:
        with out_json.open("r", encoding="utf-8") as f:
            bench_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError("bench.json is not valid JSON; benchmark output is incomplete") from exc

    meta = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "build_dir": str(build_dir),
        "config": args.config,
        "profile": args.profile,
        "benchmark_filter": benchmark_filter,
        "skip_tests": args.skip_tests,
        "test_executable": str(test_exe),
        "bench_executable": str(bench_exe),
        "bench_context": bench_data.get("context", {}),
    }
    with (root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    aggregate = Path(__file__).resolve().parent / "aggregate.py"
    subprocess.run(
        [sys.executable, str(aggregate), "--bench_json", str(out_json), "--run_root", str(root)],
        check=True,
    )


if __name__ == "__main__":
    main()
