# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibUIPC is a cross-platform C++20 library implementing Unified Incremental Potential Contact for GPU-accelerated physics simulation. It simulates rigid bodies, soft bodies, cloth, and threads with penetration-free frictional contact. Both C++ and Python APIs are provided.

## Build Commands

### Prerequisites
- CMake >= 3.26
- Python >= 3.11
- CUDA >= 12.4
- Vcpkg with `CMAKE_TOOLCHAIN_FILE` environment variable set

### Configure and Build
```bash
# Using presets (recommended)
cmake --preset release
cmake --build --preset release -j8

# Or manually
mkdir build && cd build
cmake -S .. -DUIPC_BUILD_PYBIND=ON
cmake --build . --config Release -j8
```

### Key CMake Options
- `UIPC_BUILD_PYBIND` - Build Python bindings (OFF by default)
- `UIPC_BUILD_TESTS` - Build test suite (ON by default)
- `UIPC_BUILD_EXAMPLES` - Build examples (ON by default)
- `UIPC_WITH_CUDA_BACKEND` - Enable CUDA backend (auto, disabled on macOS)
- `UIPC_WITH_CUDA_MIXED_BACKEND` - Enable the separate `cuda_mixed` backend
- `UIPC_CUDA_MIXED_PRECISION_LEVEL` - Mixed precision build level for `cuda_mixed` (`fp64|path1|path2|path3|path4|path5|path6|path7`)
- `UIPC_WITH_NVTX` - Enable optional NVTX markers in `cuda_mixed`

### Run Tests
Tests are Catch2 executables built to `build/Release/bin/`:
```bash
./build/Release/bin/uipc_test_<name>
```

### Build Documentation

The docs site uses MkDocs with `docs/nav.md` as the navigation source via `literate-nav`.

If MkDocs is not installed in the active environment, prefer an ephemeral `uv` run instead of changing repo files:

```bash
uv run --with mkdocs-material --with mkdocs-video --with mkdocs-literate-nav mkdocs build -f mkdocs.yaml
```

### Install Python Package
```bash
cd build/python
pip install .
python ../python/uipc_info.py  # verify installation
```

## Architecture

### Three-Tier Design
1. **Engine** - Simulation algorithm running on a backend (`"cuda"` or `"none"`)
2. **World** - Manages simulation lifecycle (`init()`, `advance()`, `retrieve()`)
3. **Scene** - Data structure containing simulation state (Objects, Geometries, Constitutions, Contacts, Animator)

### Reporter-Manager-Receiver (RMR) Pattern
The codebase uses Data-Oriented Programming with an ECS-inspired RMR pattern for cache-friendly data flow between components. See `docs/development/index.md`.

### Source Layout
- `src/core/` - Main simulation engine, compiled into `libuipc_core` shared library
- `src/geometry/` - Geometry processing (SimplicialComplex, BVH, distance, intersection)
- `src/constitution/` - Material models (AffineBody, NeoHookean, springs, constraints)
- `src/backends/` - Backend implementations loaded as dynamic modules
  - `cuda/` - GPU backend with CUDA kernels
  - `cuda_mixed/` - Separate experimental mixed-precision CUDA backend
  - `none/` - CPU reference implementation
- `src/pybind/` - Python bindings via pybind11
- `src/io/` - File I/O (obj, gltf, serialization)

### Key Classes
- **SimplicialComplex** - Core geometry type (vertices, edges, triangles, tetrahedra)
- **Constitution** - Material models applied via `apply_to(mesh, properties)`
- **Contact Model** - Pairwise contact parameters stored in tabular form

### Backend Architecture
Backends are MODULE libraries dynamically loaded at runtime. They implement a visitor pattern for scene traversal and provide device-specific optimizations.

## Documentation Conventions

- The published docs live under `docs/`.
- Update `docs/nav.md` whenever adding a new page that should appear in the site nav.
- The existing docs are primarily English; keep new docs in English unless the task explicitly asks otherwise.
- Prefer documenting stable interfaces and workflows over hard-coding transient benchmark numbers.
- For mixed-precision docs, treat code under `src/backends/cuda_mixed/` as the source of truth. Use `claude_plan.md` only for rationale or history when it still matches code.

## Mixed Precision Notes

- `cuda_mixed` is a separate backend loaded with `Engine("cuda_mixed")`; it is not a runtime mode inside `cuda`.
- Mixed precision is compile-time only in the current implementation. Do not describe runtime precision switching or auto-fallback unless code actually supports it.
- The main mixed-precision docs live under `docs/development/backend_cuda/mixed_precision/`.
- `src/backends/cuda_mixed/mixed_precision/PRECISION_SCOPE.md` is an implementation-side reference, but the site docs should be written in normal docs style rather than copied verbatim.
- Mixed-precision validation entrypoints are under `apps/benchmarks/mixed/` and `apps/benchmarks/uipc_assets/`.
- `extras/debug/dump_solution_x` is part of the default scene config and is used for mixed-precision quality comparison workflows.

## Testing Structure

Tests are organized under `apps/tests/`:
- `geometry/` - Geometry processing tests
- `core/` - Engine, scene, world tests
- `common/` - Utility tests
- `backends/cuda/` - CUDA backend tests
- `sim_case/` - Full simulation scenarios

Test executables are named `uipc_test_<name>` via the `uipc_add_test()` CMake function.

## Python API Structure

Modules mirror C++ namespaces:
- `uipc.core` - Engine, World, Scene
- `uipc.geometry` - Geometry operations
- `uipc.constitution` - Material models
- `uipc.unit` - Physical units (GPa, MPa, etc.)
