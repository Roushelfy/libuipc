#!/usr/bin/env python3
"""Run SOCU native contact validation gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_BINARY_NAME = "uipc_test_backend_cuda_mixed_socu"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def find_test_binary(build: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        raise FileNotFoundError(explicit)

    candidates = [
        build / "bin" / TEST_BINARY_NAME,
        build / "Release" / "bin" / TEST_BINARY_NAME,
        build / "Debug" / "bin" / TEST_BINARY_NAME,
        build / "RelWithDebInfo" / "bin" / TEST_BINARY_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = sorted(build.rglob(TEST_BINARY_NAME)) if build.exists() else []
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"could not find {TEST_BINARY_NAME} under {build}; pass --test-binary"
    )


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["SOCU_NATIVE_CONTACT_BUILD_DIR"] = str(args.build.resolve())
    return env


def cmake_cache_value(build: Path, key: str) -> str | None:
    cache = build / "CMakeCache.txt"
    if not cache.exists():
        return None
    prefix = f"{key}:"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            _, value = line.split("=", 1)
            return value.strip()
    return None


def configured_python(args: argparse.Namespace) -> str:
    if args.python is not None:
        return str(args.python.resolve())
    cache_python = cmake_cache_value(args.build, "UIPC_PYTHON_EXECUTABLE_PATH")
    if cache_python:
        return cache_python
    return sys.executable


def prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    value = str(path.resolve())
    current = env.get(key)
    env[key] = value if not current else f"{value}{os.pathsep}{current}"


def add_build_python_paths(env: dict[str, str], build: Path) -> None:
    build = build.resolve()
    python_src = build / "python" / "src"
    native_dir = python_src / "uipc" / "_native"
    if python_src.exists():
        prepend_env_path(env, "PYTHONPATH", python_src)
    if native_dir.exists():
        prepend_env_path(env, "LD_LIBRARY_PATH", native_dir)

    config = cmake_cache_value(build, "CMAKE_BUILD_TYPE") or "RelWithDebInfo"
    for candidate in (
        build / config / "bin",
        build / "bin",
        build / "RelWithDebInfo" / "bin",
        build / "Release" / "bin",
        build / "Debug" / "bin",
    ):
        if candidate.exists():
            prepend_env_path(env, "LD_LIBRARY_PATH", candidate)


def run_contract(args: argparse.Namespace) -> None:
    binary = find_test_binary(args.build, args.test_binary)
    run([str(binary), "[cuda_mixed_socu][contract]"], env=build_env(args))


def run_source_scan(args: argparse.Namespace) -> None:
    binary = find_test_binary(args.build, args.test_binary)
    env = build_env(args)
    filters = [
        "cuda_mixed_socu_contact_assembly_plan_source_scan",
        "cuda_mixed_socu_contact_executor_source_isolation",
        "cuda_mixed_socu_contact_program_writer_source_isolation",
    ]
    for test_filter in filters:
        run([str(binary), test_filter], env=env)


def run_report(args: argparse.Namespace) -> None:
    if args.reports is None:
        raise ValueError("--reports is required for --mode report and --mode all")
    cmd = [
        configured_python(args),
        str(ROOT / "scripts/analyze_socu_native_contact_reports.py"),
        str(args.reports),
        "--require-native-plan",
        "--format",
        "markdown",
    ]
    if args.report_evaluator:
        cmd.extend(["--require-evaluator", args.report_evaluator])
    if args.require_no_triplets:
        cmd.append("--require-no-triplets")
    if args.require_direct_compare_zero:
        cmd.append("--require-direct-compare-zero")
    if args.require_no_direct_fallbacks:
        cmd.append("--require-no-direct-fallbacks")
    if args.min_samples is not None:
        cmd.extend(["--min-samples", str(args.min_samples)])
    run(cmd)


def run_scene(args: argparse.Namespace) -> None:
    output = args.output
    if output is None:
        raise ValueError("--output is required for --mode scene")
    env = build_env(args)
    add_build_python_paths(env, args.build)
    env.update(
        {
            "SOCU_NATIVE_CONTACT_PLAN": "1",
            "SOCU_NATIVE_CONTACT_PLAN_EXECUTOR": "1",
            "SOCU_NATIVE_CONTACT_EVALUATOR": args.scene_evaluator,
            "SOCU_NATIVE_CONTACT_SIDE_COVERAGE_MODE": args.scene_side_coverage,
            "SOCU_REPORT_COUNTERS": "1",
        }
    )
    python = configured_python(args)
    run(
        [
            python,
            str(ROOT / "python/examples/cuda_mixed_wrecking_ball_compare.py"),
            "--variant",
            args.scene_variant,
            "--frames",
            str(args.frames),
            "--output",
            str(output),
            "--backend",
            "cuda_mixed_socu",
        ],
        env=env,
    )

    report_cmd = [
        python,
        str(ROOT / "scripts/analyze_socu_native_contact_reports.py"),
        str(output),
        "--require-native-plan",
        "--require-evaluator",
        args.scene_evaluator,
        "--format",
        "markdown",
    ]
    if args.scene_evaluator == "direct":
        report_cmd.append("--require-no-triplets")
        report_cmd.append("--require-no-direct-fallbacks")
    if args.scene_evaluator == "direct_compare":
        report_cmd.append("--require-direct-compare-zero")
    run(report_cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SOCU native contact docs/gates validation."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "contract", "source-scan", "report", "scene", "build-graph"),
        default="all",
    )
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument(
        "--python",
        type=Path,
        help=(
            "Python executable for report and scene gates; defaults to "
            "UIPC_PYTHON_EXECUTABLE_PATH from CMakeCache.txt."
        ),
    )
    parser.add_argument("--test-binary", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument(
        "--report-evaluator",
        choices=("direct", "direct_compare", "triplet_compat"),
    )
    parser.add_argument("--require-no-triplets", action="store_true")
    parser.add_argument("--require-direct-compare-zero", action="store_true")
    parser.add_argument("--require-no-direct-fallbacks", action="store_true")
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scene-evaluator",
        choices=("direct", "direct_compare", "triplet_compat"),
        default="direct",
    )
    parser.add_argument(
        "--scene-side-coverage",
        choices=("global", "demand_filled", "active_set_temporary"),
        default="global",
    )
    parser.add_argument("--scene-variant", default="socu_rt50_topology_diag_lump")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        if args.mode in ("all", "contract"):
            run_contract(args)
        if args.mode in ("all", "source-scan"):
            run_source_scan(args)
        if args.mode in ("all", "report"):
            run_report(args)
        if args.mode == "scene":
            run_scene(args)
        if args.mode == "build-graph":
            raise ValueError("build-graph gate is planned but not implemented yet")
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
