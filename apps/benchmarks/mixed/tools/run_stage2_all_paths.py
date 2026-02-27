#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


ALL_LEVELS = ("fp64", "path1", "path2", "path3", "path4")
COMPARE_LEVELS = ("path1", "path2", "path3", "path4")


def run_cmd(cmd, cwd=None, env=None):
    printable = " ".join(str(x) for x in cmd)
    print(f"[run_stage2_all_paths] $ {printable}")
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
        print(f"[run_stage2_all_paths] reuse configured build dir: {build_dir}")

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


def base_env(workspace_root: Path, frame_scale: int, dump_surface: bool) -> dict:
    env = os.environ.copy()
    env["UIPC_BENCH_WORKSPACE_ROOT"] = str(workspace_root)
    env["UIPC_BENCH_STAGE2_FRAME_SCALE"] = str(frame_scale)
    if dump_surface:
        env["UIPC_BENCH_DUMP_SURFACE"] = "1"
    else:
        env.pop("UIPC_BENCH_DUMP_SURFACE", None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage2 benchmark across fp64/path1/path2/path3/path4.")
    parser.add_argument("--source_dir", type=Path, default=None, help="repository root")
    parser.add_argument("--build_root", type=Path, default=None, help="deprecated root for build directories")
    parser.add_argument("--build_fp64", type=Path, default=None, help="reused fp64 build dir (default: build_impl_fp64)")
    parser.add_argument("--build_path1", type=Path, default=None, help="reused path1 build dir (default: build_impl_path1)")
    parser.add_argument("--build_path2", type=Path, default=None, help="reused path2 build dir (default: build_impl_path2)")
    parser.add_argument("--build_path3", type=Path, default=None, help="reused path3 build dir (default: build_impl_path3)")
    parser.add_argument("--build_path4", type=Path, default=None, help="reused path4 build dir (default: build_impl_path4)")
    parser.add_argument("--run_root", type=Path, default=None, help="output directory for benchmark json/report")
    parser.add_argument("--config", type=str, default="Release", help="build config")
    parser.add_argument("--jobs", type=int, default=8, help="parallel build jobs")
    parser.add_argument("--generator", type=str, default=None, help="optional CMake generator")
    parser.add_argument("--force_configure", action="store_true", help="force running cmake configure")
    parser.add_argument("--with_cuda_backend",
                        type=str,
                        choices=("ON", "OFF", "on", "off"),
                        default="OFF",
                        help="whether to compile with UIPC_WITH_CUDA_BACKEND (default: OFF)")
    parser.add_argument("--enable_cuda_baseline",
                        action="store_true",
                        help="register/run optional cuda perf baseline (requires cuda backend)")
    parser.add_argument("--frame_scale", type=int, default=1, help="set UIPC_BENCH_STAGE2_FRAME_SCALE")
    parser.add_argument("--dump_surface",
                        type=str,
                        choices=("ON", "OFF", "on", "off"),
                        default="ON",
                        help="set UIPC_BENCH_DUMP_SURFACE (default: ON)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    source_dir = args.source_dir.resolve() if args.source_dir else script_dir.parents[3]
    run_root = args.run_root.resolve() if args.run_root else source_dir / "output" / "benchmarks" / "mixed_stage2_all_paths"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.build_root:
        build_root = args.build_root.resolve()
        build_dirs = {
            "fp64": args.build_fp64.resolve() if args.build_fp64 else build_root / "mixed_fp64",
            "path1": args.build_path1.resolve() if args.build_path1 else build_root / "mixed_path1",
            "path2": args.build_path2.resolve() if args.build_path2 else build_root / "mixed_path2",
            "path3": args.build_path3.resolve() if args.build_path3 else build_root / "mixed_path3",
            "path4": args.build_path4.resolve() if args.build_path4 else build_root / "mixed_path4",
        }
    else:
        build_dirs = {
            "fp64": args.build_fp64.resolve() if args.build_fp64 else source_dir / "build_impl_fp64",
            "path1": args.build_path1.resolve() if args.build_path1 else source_dir / "build_impl_path1",
            "path2": args.build_path2.resolve() if args.build_path2 else source_dir / "build_impl_path2",
            "path3": args.build_path3.resolve() if args.build_path3 else source_dir / "build_impl_path3",
            "path4": args.build_path4.resolve() if args.build_path4 else source_dir / "build_impl_path4",
        }

    workspace_root = run_root / "workspaces"
    enable_cuda_backend = str(args.with_cuda_backend).upper() == "ON" or args.enable_cuda_baseline
    dump_surface = str(args.dump_surface).upper() == "ON"

    # Build + locate exe for all levels first.
    exes = {}
    for level in ALL_LEVELS:
        configure_and_build(source_dir,
                            build_dirs[level],
                            level=level,
                            config=args.config,
                            jobs=args.jobs,
                            enable_cuda=enable_cuda_backend,
                            generator=args.generator,
                            force_configure=args.force_configure)
        exes[level] = find_benchmark_exe(build_dirs[level], args.config)

    # fp64 perf + reference
    env_fp64_perf = base_env(workspace_root, args.frame_scale, dump_surface)
    env_fp64_perf["UIPC_BENCH_WORKSPACE_TAG"] = "fp64_perf"
    if args.enable_cuda_baseline:
        env_fp64_perf["UIPC_BENCH_ENABLE_CUDA_BASELINE"] = "1"
    else:
        env_fp64_perf.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)

    run_benchmark(exes["fp64"],
                  run_root / "stage2" / "fp64" / "perf" / "gbench.json",
                  benchmark_filter="^Mixed\\.Stage2\\.Perf\\..*",
                  env=env_fp64_perf)

    env_fp64_ref = base_env(workspace_root, args.frame_scale, dump_surface)
    env_fp64_ref["UIPC_BENCH_WORKSPACE_TAG"] = "fp64_ref"
    env_fp64_ref.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)
    env_fp64_ref.pop("UIPC_BENCH_ERROR_REFERENCE_ROOT", None)

    run_benchmark(exes["fp64"],
                  run_root / "stage2" / "fp64" / "quality_reference" / "gbench.json",
                  benchmark_filter="^Mixed\\.Stage2\\.Quality\\.Reference[0-9]+F\\..*",
                  env=env_fp64_ref)

    # path1..path4 perf + compare
    reference_root = workspace_root / "stage2" / "cuda_mixed"
    for level in COMPARE_LEVELS:
        env_perf = base_env(workspace_root, args.frame_scale, dump_surface)
        env_perf["UIPC_BENCH_WORKSPACE_TAG"] = f"{level}_perf"
        if args.enable_cuda_baseline:
            env_perf["UIPC_BENCH_ENABLE_CUDA_BASELINE"] = "1"
        else:
            env_perf.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)
        env_perf.pop("UIPC_BENCH_ERROR_REFERENCE_ROOT", None)

        run_benchmark(exes[level],
                      run_root / "stage2" / level / "perf" / "gbench.json",
                      benchmark_filter="^Mixed\\.Stage2\\.Perf\\..*",
                      env=env_perf)

        env_cmp = base_env(workspace_root, args.frame_scale, dump_surface)
        env_cmp["UIPC_BENCH_WORKSPACE_TAG"] = f"{level}_cmp"
        env_cmp["UIPC_BENCH_ERROR_REFERENCE_ROOT"] = str(reference_root)
        env_cmp.pop("UIPC_BENCH_ENABLE_CUDA_BASELINE", None)

        run_benchmark(exes[level],
                      run_root / "stage2" / level / "quality_compare" / "gbench.json",
                      benchmark_filter="^Mixed\\.Stage2\\.Quality\\.Compare[0-9]+F\\..*",
                      env=env_cmp)

    aggregate_script = script_dir / "aggregate_stage2_all_paths.py"
    run_cmd([sys.executable, str(aggregate_script), "--run_root", str(run_root)])

    print(f"[run_stage2_all_paths] done. report dir: {run_root}")
    print("[run_stage2_all_paths] build dirs:")
    for level in ALL_LEVELS:
        print(f"  - {level}: {build_dirs[level]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
