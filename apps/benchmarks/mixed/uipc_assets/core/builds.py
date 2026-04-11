from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

from .artifacts import repo_root
from .manifest import ordered_levels


BACKEND_DLL_BASENAME = "uipc_backend_cuda_mixed"


def dll_name() -> str:
    if os.name == "nt":
        return f"{BACKEND_DLL_BASENAME}.dll"
    if os.name == "darwin":
        return f"lib{BACKEND_DLL_BASENAME}.dylib"
    return f"lib{BACKEND_DLL_BASENAME}.so"


def candidate_module_dirs(build_dir: Path, config: str) -> list[Path]:
    return [
        build_dir / config / "bin",
        build_dir / "bin",
        build_dir,
    ]


def find_module_dir(build_dir: Path, config: str) -> Path:
    target = dll_name()
    for candidate in candidate_module_dirs(build_dir, config):
        if (candidate / target).exists():
            return candidate.resolve()
    for found in build_dir.rglob(target):
        return found.parent.resolve()
    raise FileNotFoundError(f"Cannot find {target} under build dir {build_dir}")


def parse_build_args(values: Iterable[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LEVEL=PATH, got: {value}")
        level, raw_path = value.split("=", 1)
        result[level.strip()] = Path(raw_path).expanduser().resolve()
    return {level: result[level] for level in ordered_levels(result.keys())}


def default_build_dir_for(level: str) -> Path:
    return (repo_root() / "build" / f"build_impl_{level}").resolve()


def python_src_dir(build_dir: Path) -> Path:
    return build_dir / "python" / "src"


def validate_python_src(build_dir: Path) -> Path:
    py_src = python_src_dir(build_dir)
    uipc_pkg = py_src / "uipc"
    native_pkg = uipc_pkg / "_native"
    if not uipc_pkg.is_dir() or not native_pkg.is_dir():
        raise FileNotFoundError(
            "Missing Python bindings under "
            f"{py_src}. Expected {uipc_pkg} and {native_pkg}. "
            "Build this level with UIPC_BUILD_PYBIND=ON."
        )
    return py_src.resolve()


def resolve_builds(levels: Iterable[str], build_overrides: Dict[str, Path], config: str) -> Dict[str, Dict[str, str]]:
    resolved: Dict[str, Dict[str, str]] = {}
    for level in ordered_levels(levels):
        build_dir = build_overrides.get(level, default_build_dir_for(level))
        module_dir = find_module_dir(build_dir, config)
        pyuipc_src_dir = validate_python_src(build_dir)
        resolved[level] = {
            "build_dir": str(build_dir),
            "module_dir": str(module_dir),
            "pyuipc_src_dir": str(pyuipc_src_dir),
        }
    return resolved


def prepend_library_path(env: dict[str, str], module_dir: Path) -> dict[str, str]:
    updated = dict(env)
    module_dir_str = str(module_dir)
    if os.name == "nt":
        key = "PATH"
    elif os.name == "darwin":
        key = "DYLD_LIBRARY_PATH"
    else:
        key = "LD_LIBRARY_PATH"
    old = updated.get(key, "")
    updated[key] = module_dir_str if not old else module_dir_str + os.pathsep + old
    return updated


def prepend_pythonpath(env: dict[str, str], python_src: Path) -> dict[str, str]:
    updated = dict(env)
    python_src_str = str(python_src)
    old = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = python_src_str if not old else python_src_str + os.pathsep + old
    return updated
