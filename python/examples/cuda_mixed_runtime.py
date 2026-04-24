"""Runtime helpers for cuda_mixed examples launched from build trees."""

from __future__ import annotations

import os
from pathlib import Path

import uipc


def _backend_library_name() -> str:
    if os.name == "nt":
        return "uipc_backend_cuda_mixed.dll"
    if os.name == "darwin":
        return "libuipc_backend_cuda_mixed.dylib"
    return "libuipc_backend_cuda_mixed.so"


def _runtime_library_path_key() -> str:
    if os.name == "nt":
        return "PATH"
    if os.name == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def _path_is_on_runtime_library_path(path: Path) -> bool:
    runtime_path = os.environ.get(_runtime_library_path_key(), "")
    return str(path.resolve()) in {
        str(Path(item).expanduser().resolve())
        for item in runtime_path.split(os.pathsep)
        if item
    }


def _candidate_module_dirs() -> list[tuple[Path, bool]]:
    candidates: list[tuple[Path, bool]] = []

    env_module_dir = os.environ.get("UIPC_MODULE_DIR")
    if env_module_dir:
        candidates.append((Path(env_module_dir).expanduser(), True))

    package_file = getattr(uipc, "__file__", None)
    if package_file is not None:
        package_path = Path(package_file).resolve()
        lib_name = _backend_library_name()
        configs = [
            os.environ.get("UIPC_CONFIG"),
            "Release",
            "RelWithDebInfo",
            "Debug",
            "MinSizeRel",
        ]
        for parent in package_path.parents:
            found_package_candidate = False
            for config in configs:
                if not config:
                    continue
                config_bin = parent / config / "bin"
                if (config_bin / lib_name).exists():
                    candidates.append((config_bin, False))
                    found_package_candidate = True
                    break
            if found_package_candidate:
                break

    return candidates


def init_cuda_mixed_module_dir() -> Path | None:
    """Prefer the backend library in build/<level>/Release/bin over stale _native copies."""
    lib_name = _backend_library_name()
    for module_dir, explicit in _candidate_module_dirs():
        module_dir = module_dir.resolve()
        if (module_dir / lib_name).exists():
            if not explicit and not _path_is_on_runtime_library_path(module_dir):
                continue
            cfg = uipc.default_config()
            cfg["module_dir"] = str(module_dir)
            uipc.init(cfg)
            return module_dir
    return None
