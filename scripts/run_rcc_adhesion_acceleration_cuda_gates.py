#!/usr/bin/env python3
"""Run local CUDA gates for RCC adhesion acceleration.

This gate requires a configured CUDA build tree. It intentionally includes the
small bunny GPU sanity check because oversized bonded-PT device payloads can
surface later as BVH/radix-sort CUDA failures.
"""

from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def require_exe(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"missing executable: {path}\n"
            "Configure/build the CUDA tree first, or pass --build-dir."
        )
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RCC adhesion acceleration CUDA validation gates."
    )
    parser.add_argument(
        "--build-dir",
        default="build/cuda_mixed_fused_pcg",
        help="Configured CUDA build directory relative to the repo root.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Parallel build jobs used before running gates.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the CMake build step and run existing test binaries.",
    )
    args = parser.parse_args()

    build_dir = ROOT / args.build_dir
    if not args.no_build:
        run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "uipc_test_core",
                "uipc_test_backend_cuda",
                f"-j{args.jobs}",
            ]
        )

    bin_dir = build_dir / "RelWithDebInfo" / "bin"
    core_test = require_exe(bin_dir / "uipc_test_core")
    backend_test = require_exe(bin_dir / "uipc_test_backend_cuda")

    run([core_test, "[rcc_bonded_pt]", "-r", "compact"])
    run([backend_test, "[rcc_bonded_pt]", "-r", "compact"])
    run([backend_test, "gpu_sanity_check", "-c", "bunny", "-r", "compact"])

    print("RCC adhesion acceleration CUDA gates passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
