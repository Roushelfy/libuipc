#!/usr/bin/env python3
"""Run portable RCC adhesion acceleration docs/source gates.

CUDA scene gates are wired into pytest and sim_case, but stay as explicit local
commands because they need a built CUDA test tree.
"""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> int:
    py = sys.executable
    run([py, "scripts/run_rcc_adhesion_acceleration_gates.py"])
    run(
        [
            py,
            "-m",
            "py_compile",
            "scripts/run_rcc_adhesion_acceleration_gates.py",
            "scripts/run_rcc_adhesion_acceleration_all_gates.py",
            "scripts/run_rcc_adhesion_acceleration_cuda_gates.py",
            "scripts/build_docs.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("DISABLE_MKDOCS_2_WARNING", "true")
    run([py, "scripts/build_docs.py", "-o", "/tmp/libuipc-docs-check"], env=env)
    print("Portable RCC adhesion acceleration docs/source gates passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
