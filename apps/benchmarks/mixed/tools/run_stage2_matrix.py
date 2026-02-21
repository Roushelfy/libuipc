#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def run_cmd(cmd, cwd=None, env=None):
    printable = " ".join(str(x) for x in cmd)
    print(f"[run_stage2_matrix] $ {printable}")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def read_cmake_cache(cache_file: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not cache_file.exists():
        return values

    for line in cache_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        if "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _type = key_type.split(":", 1)
        values[key] = value
    return values


def needs_configure(build_dir: Path, expected: Dict[str, str], force_configure: bool) -> bool:
    if force_configure:
        return True
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.exists():
        return True
    cache = read_cmake_cache(cache_file)
    for k, v in expected.items():
        if cache.get(k) != v:
            return True
    return False


def find_benchmark_exe(build_dir: Path, config: str) -> Path:
    exe_name = "uipc_benchmark_mixed_stage2.exe" if os.name == "nt" else "uipc_benchmark_mixed_stage2"
    candidates = [
        build_dir / config / "bin" / exe_name,
        build_dir / "bin" / exe_name,
        build_dir / exe_name,
    ]
    for c in candidates:
        if c.exists():
            return c

    for found in build_dir.rglob(exe_name):
        return found

    raise FileNotFoundError(f"Cannot find benchmark executable {exe_name} under {build_dir}")


def configure_and_build(source_dir: Path,
                        build_dir: Path,
                        level: str,
                        config: str,
                        jobs: int,
                        enable_cuda: bool,
                        generator: Optional[str],
                        force_configure: bool):
    expected = {
        "CMAKE_BUILD_TYPE": config,
        "UIPC_BUILD_BENCHMARKS": "ON",
        "UIPC_BUILD_TESTS": "OFF",
        "UIPC_BUILD_EXAMPLES": "OFF",
        "UIPC_BUILD_GUI": "OFF",
        "UIPC_WITH_CUDA_BACKEND": "ON" if enable_cuda else "OFF",
        "UIPC_WITH_CUDA_MIXED_BACKEND": "ON",
        "UIPC_CUDA_MIXED_PRECISION_LEVEL": level,
    }

    if needs_configure(build_dir, expected, force_configure):
        cmake_configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={config}",
            "-DUIPC_BUILD_BENCHMARKS=ON",
            "-DUIPC_BUILD_TESTS=OFF",
            "-DUIPC_BUILD_EXAMPLES=OFF",
            "-DUIPC_BUILD_GUI=OFF",
            f"-DUIPC_WITH_CUDA_BACKEND={'ON' if enable_cuda else 'OFF'}",
            "-DUIPC_WITH_CUDA_MIXED_BACKEND=ON",
            f"-DUIPC_CUDA_MIXED_PRECISION_LEVEL={level}",
        ]
        if generator:
            cmake_configure.extend(["-G", generator])
        run_cmd(cmake_configure)
    else:
        print(f"[run_stage2_matrix] reuse configured build dir: {build_dir}")

    cmake_build = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        config,
        "--target",
        "mixed_stage2",
        "--parallel",
        str(jobs),
    ]
    run_cmd(cmake_build)


def run_benchmark(exe: Path, out_json: Path, benchmark_filter: str, env: dict):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        f"--benchmark_filter={benchmark_filter}",
        f"--benchmark_out={out_json}",
        "--benchmark_out_format=json",
    ]
    run_cmd(cmd, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mixed Stage2 benchmark matrix.")
    parser.add_argument("--source_dir", type=Path, default=None, help="repository root")
    parser.add_argument("--build_root", type=Path, default=None, help="deprecated root for build directories")
    parser.add_argument("--build_fp64", type=Path, default=None, help="reused fp64 build dir (default: build_impl_fp64)")
    parser.add_argument("--build_path1", type=Path, default=None, help="reused path1 build dir (default: build_impl_path1)")
    parser.add_argument("--run_root", type=Path, default=None, help="output directory for benchmark json/report")
    parser.add_argument("--config", type=str, default="Release", help="build config")
    parser.add_argument("--jobs", type=int, default=8, help="parallel build jobs")
    parser.add_argument("--generator", type=str, default=None, help="optional CMake generator")
    parser.add_argument("--force_configure", action="store_true", help="force running cmake configure even if cache matches")
    parser.add_argument("--with_cuda_backend",
                        type=str,
                        choices=("ON", "OFF", "on", "off"),
                        default="OFF",
                        help="whether to compile with UIPC_WITH_CUDA_BACKEND (default: OFF)")
    parser.add_argument("--enable_cuda_baseline",
                        action="store_true",
                        help="register/run optional cuda perf baseline (requires cuda backend)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    source_dir = args.source_dir.resolve() if args.source_dir else script_dir.parents[3]
    run_root = args.run_root.resolve() if args.run_root else source_dir / "output" / "benchmarks" / "mixed_stage2"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.build_root:
        build_root = args.build_root.resolve()
        fp64_build = args.build_fp64.resolve() if args.build_fp64 else build_root / "mixed_fp64"
        path1_build = args.build_path1.resolve() if args.build_path1 else build_root / "mixed_path1"
    else:
        fp64_build = args.build_fp64.resolve() if args.build_fp64 else source_dir / "build_impl_fp64"
        path1_build = args.build_path1.resolve() if args.build_path1 else source_dir / "build_impl_path1"

    workspace_root = run_root / "workspaces"
    enable_cuda_backend = str(args.with_cuda_backend).upper() == "ON" or args.enable_cuda_baseline

    # fp64 build and runs
    configure_and_build(source_dir,
                        fp64_build,
                        level="fp64",
                        config=args.config,
                        jobs=args.jobs,
                        enable_cuda=enable_cuda_backend,
                        generator=args.generator,
                        force_configure=args.force_configure)
    fp64_exe = find_benchmark_exe(fp64_build, args.config)

    fp64_perf_env = os.environ.copy()
    fp64_perf_env["UIPC_BENCH_WORKSPACE_ROOT"] = str(workspace_root)
    fp64_perf_env["UIPC_BENCH_WORKSPACE_TAG"] = "fp64_perf"
    if args.enable_cuda_baseline:
        fp64_perf_env["UIPC_BENCH_ENABLE_CUDA_BASELINE"] = "1"
    else:
        fp64_perf_env.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)

    run_benchmark(fp64_exe,
                  run_root / "fp64" / "perf" / "gbench.json",
                  benchmark_filter="^Mixed\\.Stage2\\.Perf\\..*",
                  env=fp64_perf_env)

    fp64_ref_env = os.environ.copy()
    fp64_ref_env["UIPC_BENCH_WORKSPACE_ROOT"] = str(workspace_root)
    fp64_ref_env["UIPC_BENCH_WORKSPACE_TAG"] = "fp64_ref"
    fp64_ref_env.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)

    run_benchmark(fp64_exe,
                  run_root / "fp64" / "quality_reference" / "gbench.json",
                  benchmark_filter="^Mixed\\.Stage2\\.Quality\\.Reference(20|30)F\\..*",
                  env=fp64_ref_env)

    # path1 build and runs
    configure_and_build(source_dir,
                        path1_build,
                        level="path1",
                        config=args.config,
                        jobs=args.jobs,
                        enable_cuda=enable_cuda_backend,
                        generator=args.generator,
                        force_configure=args.force_configure)
    path1_exe = find_benchmark_exe(path1_build, args.config)

    path1_perf_env = os.environ.copy()
    path1_perf_env["UIPC_BENCH_WORKSPACE_ROOT"] = str(workspace_root)
    path1_perf_env["UIPC_BENCH_WORKSPACE_TAG"] = "path1_perf"
    if args.enable_cuda_baseline:
        path1_perf_env["UIPC_BENCH_ENABLE_CUDA_BASELINE"] = "1"
    else:
        path1_perf_env.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)

    run_benchmark(path1_exe,
                  run_root / "path1" / "perf" / "gbench.json",
                  benchmark_filter="^Mixed\\.Stage2\\.Perf\\..*",
                  env=path1_perf_env)

    path1_cmp_env = os.environ.copy()
    path1_cmp_env["UIPC_BENCH_WORKSPACE_ROOT"] = str(workspace_root)
    path1_cmp_env["UIPC_BENCH_WORKSPACE_TAG"] = "path1_cmp"
    path1_cmp_env["UIPC_BENCH_ERROR_REFERENCE_ROOT"] = str(workspace_root / "stage2" / "cuda_mixed")
    path1_cmp_env.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)

    run_benchmark(path1_exe,
                  run_root / "path1" / "quality_compare" / "gbench.json",
                  benchmark_filter="^Mixed\\.Stage2\\.Quality\\.Compare(20|30)F\\..*",
                  env=path1_cmp_env)

    aggregate_script = script_dir / "aggregate_stage2.py"
    aggregate_cmd = [
        sys.executable,
        str(aggregate_script),
        "--fp64_perf",
        str(run_root / "fp64" / "perf" / "gbench.json"),
        "--path1_perf",
        str(run_root / "path1" / "perf" / "gbench.json"),
        "--fp64_quality_reference",
        str(run_root / "fp64" / "quality_reference" / "gbench.json"),
        "--path1_quality_compare",
        str(run_root / "path1" / "quality_compare" / "gbench.json"),
        "--workspace_root",
        str(workspace_root),
        "--out_dir",
        str(run_root),
    ]
    run_cmd(aggregate_cmd)

    print(f"[run_stage2_matrix] done. report dir: {run_root}")
    print(f"[run_stage2_matrix] build dirs: fp64={fp64_build}, path1={path1_build}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
