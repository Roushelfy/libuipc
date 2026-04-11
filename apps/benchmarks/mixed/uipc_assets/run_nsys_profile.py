#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CLI = Path(__file__).resolve().parent / "cli.py"
MANIFEST = Path(__file__).resolve().parent / "manifests" / "full.json"
OUT_ROOT = REPO_ROOT / "output" / "benchmarks" / "mixed" / "uipc_assets_nsys"

NSYS = Path("nsys")


def run_one(level: str, module_dir: Path, scene: str) -> None:
    out_dir = OUT_ROOT / level / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    report = OUT_ROOT / "nsys" / f"{level}_{scene}"
    report.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    env[path_key] = str(module_dir) + os.pathsep + env.get(path_key, "")

    cmd = [
        str(NSYS),
        "profile",
        "--trace=cuda,nvtx",
        "--cuda-memory-usage=true",
        f"--output={report}",
        "--force-overwrite=true",
        sys.executable,
        str(CLI),
        "run",
        "--manifest",
        str(MANIFEST),
        "--scene",
        scene,
        "--levels",
        "fp64",
        level,
        "--build",
        f"fp64={REPO_ROOT / 'build' / 'build_impl_fp64'}",
        "--build",
        f"{level}={REPO_ROOT / 'build' / f'build_impl_{level}'}",
        "--run_root",
        str(out_dir / "run"),
        "--quality",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    levels = {
        "path1": REPO_ROOT / "build" / "build_impl_path1" / "RelWithDebInfo" / "bin",
        "path7": REPO_ROOT / "build" / "build_impl_path7" / "RelWithDebInfo" / "bin",
        "path8": REPO_ROOT / "build" / "build_impl_path8" / "RelWithDebInfo" / "bin",
    }
    scenes = ["rigid_ipc_wrecking_ball"]
    for level, module_dir in levels.items():
        if not module_dir.exists():
            print(f"[skip] missing module_dir: {module_dir}")
            continue
        for scene in scenes:
            run_one(level, module_dir, scene)


if __name__ == "__main__":
    main()
