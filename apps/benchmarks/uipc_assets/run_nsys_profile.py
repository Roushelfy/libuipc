#!/usr/bin/env python3
"""Run nsys profile for each scene x level combination (fp64, path7)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parent / "run_uipc_assets_benchmark.py"
MANIFEST = Path(__file__).resolve().parent / "uipc_assets_manifest_rigid_extra.json"

NSYS = Path("C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe")

REVISION = "7355b9030fbabdcfcbf304dc9988f76b8fc397cc"
CACHE_DIR = Path("C:/Users/danie/.cache/huggingface/hub")

LEVELS = {
    "fp64":  REPO_ROOT / "build_impl_fp64"  / "RelWithDebInfo" / "bin",
    "path7": REPO_ROOT / "build_impl_path7" / "RelWithDebInfo" / "bin",
}

PYTHON_PER_LEVEL = {
    "fp64":  REPO_ROOT / ".venv"       / "Scripts" / "python.exe",
    "path7": REPO_ROOT / ".venv_path7" / "Scripts" / "python.exe",
}

SCENES_ALL = ["rigid_ipc_arch_101", "rigid_ipc_cone_pile", "rigid_ipc_gear_chain", "rigid_ipc_fracture_wall"]

OUT_ROOT = REPO_ROOT / "output" / "nsight_profile"


def run_one(level: str, module_dir: Path, scene: str) -> None:
    worker_out = OUT_ROOT / level / scene
    worker_out.mkdir(parents=True, exist_ok=True)
    nsys_out = OUT_ROOT / "nsys" / f"{level}_{scene}"
    nsys_out.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    env[path_key] = str(module_dir) + os.pathsep + env.get(path_key, "")

    cmd = [
        str(NSYS), "profile",
        "--trace=cuda,nvtx",
        "--cuda-memory-usage=true",
        f"--output={nsys_out}",
        "--force-overwrite=true",
        str(PYTHON_PER_LEVEL.get(level, sys.executable)), str(SCRIPT),
        "_worker",
        "--scene", scene,
        "--manifest", str(MANIFEST),
        "--mode", "perf",
        "--level", level,
        "--module_dir", str(module_dir),
        "--output_dir", str(worker_out),
        "--revision", REVISION,
        "--cache_dir", str(CACHE_DIR),
        "--dump_surface", "OFF",
    ]

    print(f"\n[nsys] {level}/{scene}")
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print(f"[nsys] WARNING: worker exited with code {r.returncode} (0x{r.returncode & 0xFFFFFFFF:08X})")
    print(f"[nsys] done -> {nsys_out}.nsys-rep")


def main() -> None:
    for level, module_dir in LEVELS.items():
        if not module_dir.exists():
            print(f"[skip] module_dir not found: {module_dir}")
            continue
        for scene in SCENES_ALL:
            run_one(level, module_dir, scene)


if __name__ == "__main__":
    main()
