"""Runtime helpers for cuda_mixed examples launched from build trees."""

from __future__ import annotations

import os
from pathlib import Path

import uipc


def _backend_library_name(backend_name: str = "cuda_mixed") -> str:
    if os.name == "nt":
        return f"uipc_backend_{backend_name}.dll"
    if os.name == "darwin":
        return f"libuipc_backend_{backend_name}.dylib"
    return f"libuipc_backend_{backend_name}.so"


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


def _candidate_module_dirs(backend_name: str = "cuda_mixed") -> list[tuple[Path, bool]]:
    candidates: list[tuple[Path, bool]] = []

    env_module_dir = os.environ.get("UIPC_MODULE_DIR")
    if env_module_dir:
        candidates.append((Path(env_module_dir).expanduser(), True))

    package_file = getattr(uipc, "__file__", None)
    if package_file is not None:
        package_path = Path(package_file).resolve()
        lib_name = _backend_library_name(backend_name)
        configs = [
            os.environ.get("UIPC_CONFIG"),
            "Release",
            "RelWithDebInfo",
            "Debug",
            "MinSizeRel",
        ]
        for parent in package_path.parents:
            for config in configs:
                if not config:
                    continue
                config_bin = parent / config / "bin"
                if (config_bin / lib_name).exists():
                    candidates.append((config_bin, False))

    return candidates


def init_cuda_mixed_module_dir(backend_name: str = "cuda_mixed") -> Path | None:
    """Prefer the backend library in build/<level>/Release/bin over stale _native copies."""
    lib_name = _backend_library_name(backend_name)
    for module_dir, explicit in _candidate_module_dirs(backend_name):
        module_dir = module_dir.resolve()
        if (module_dir / lib_name).exists():
            if not explicit and not _path_is_on_runtime_library_path(module_dir):
                continue
            cfg = uipc.default_config()
            cfg["module_dir"] = str(module_dir)
            uipc.init(cfg)
            return module_dir
    return None
