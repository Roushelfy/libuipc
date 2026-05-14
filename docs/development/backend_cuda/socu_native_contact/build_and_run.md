# SOCU Native Contact Build And Run

This page records the default compile and execution commands for the SOCU native
contact gates. Prefer `uv` for Python entrypoints.

## One-Time Toolchain Setup

This branch expects vcpkg under `~/work` unless `CMAKE_TOOLCHAIN_FILE` is set to
another local vcpkg checkout:

```bash
cd /home/zhaofeng/work
git clone https://github.com/microsoft/vcpkg.git vcpkg
/home/zhaofeng/work/vcpkg/bootstrap-vcpkg.sh
```

The SOCU native solver is a git submodule pinned by the parent repository. Use
the SSH remote and initialize it before configuring with
`UIPC_WITH_SOCU_NATIVE=AUTO` or `ON`:

```bash
cd /home/zhaofeng/work/libuipc
git submodule sync external/socu-native-cuda
git submodule update --init external/socu-native-cuda
```

If this clone still has an old HTTPS or local-path submodule URL in `.git/config`,
reset it to SSH first:

```bash
cd /home/zhaofeng/work/libuipc
git config submodule.external/socu-native-cuda.url git@github.com:Roushelfy/socu-native-cuda.git
git -C external/socu-native-cuda remote set-url origin git@github.com:Roushelfy/socu-native-cuda.git
```

Use the repo-local Python uv environment for pybind builds and Python scene
gates. Do not rely on `/usr/bin/python3` for pybind dependencies on systems
where pip is externally managed.

```bash
cd /home/zhaofeng/work/libuipc
uv sync --project python --extra dev
uv pip install --python python/.venv/bin/python pybind11
```

On the verified local machine, CUDA is available through
`/usr/local/cuda-12.8/bin/nvcc`. If `nvcc` is not on `PATH`, pass
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc` explicitly as shown
below.

## Configure And Build

Default development build:

```bash
BUILD_DIR=build/socu_native_contact
VCPKG_ROOT=/home/zhaofeng/work/vcpkg
CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc
PYTHON_EXE=$PWD/python/.venv/bin/python

cmake -S . -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_CUDA_COMPILER=${CUDA_NVCC} \
  -DUIPC_PYTHON_EXECUTABLE_PATH=${PYTHON_EXE} \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DUIPC_BUILD_TESTS=ON \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_BENCHMARKS=OFF \
  -DUIPC_BUILD_GUI=OFF \
  -DUIPC_BUILD_PYBIND=ON \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_SOCU_BACKEND=ON \
  -DUIPC_WITH_SOCU_NATIVE=AUTO \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=fp64 \
  -DUIPC_CUDA_MIXED_WRECKING_BALL_MINIMAL_BUILD=ON \
  -DUIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF \
  -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF

cmake --build ${BUILD_DIR} \
  --target uipc_test_backend_cuda_mixed_socu \
  --parallel 8
```

Native-only build graph investigation:

```bash
BUILD_DIR=build/socu_native_contact_native_only
VCPKG_ROOT=/home/zhaofeng/work/vcpkg
CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc
PYTHON_EXE=$PWD/python/.venv/bin/python

cmake -S . -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_CUDA_COMPILER=${CUDA_NVCC} \
  -DUIPC_PYTHON_EXECUTABLE_PATH=${PYTHON_EXE} \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DUIPC_BUILD_TESTS=ON \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_BENCHMARKS=OFF \
  -DUIPC_BUILD_GUI=OFF \
  -DUIPC_BUILD_PYBIND=ON \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_SOCU_BACKEND=ON \
  -DUIPC_WITH_SOCU_NATIVE=AUTO \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=fp64 \
  -DUIPC_CUDA_MIXED_WRECKING_BALL_MINIMAL_BUILD=ON \
  -DUIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF \
  -DUIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON

cmake --build ${BUILD_DIR} \
  --target uipc_test_backend_cuda_mixed_socu \
  --parallel 8
```

Use `UIPC_WITH_SOCU_NATIVE=ON` instead of `AUTO` when a missing
`external/socu-native-cuda` integration should fail configure immediately.

## C++ Gates

Run the Catch binary directly when you know its path:

```bash
BUILD_DIR=build/socu_native_contact
CONFIG=RelWithDebInfo
SOCU_NATIVE_CONTACT_BUILD_DIR=${BUILD_DIR} \
  ${BUILD_DIR}/${CONFIG}/bin/uipc_test_backend_cuda_mixed_socu \
  "[cuda_mixed_socu][contract]"
```

Depending on the generator/config, the binary may live under
`${BUILD_DIR}/RelWithDebInfo/bin`, `${BUILD_DIR}/Release/bin`, or
`${BUILD_DIR}/Debug/bin`. The gate runner searches these paths automatically.

Preferred gate-runner commands:

```bash
BUILD_DIR=build/socu_native_contact

uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build ${BUILD_DIR} \
  --mode contract

uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build ${BUILD_DIR} \
  --mode source-scan
```

## Report Analysis

The report analyzer uses only the Python standard library. Prefer `uv
run --no-project`:

```bash
uv run --no-project python scripts/analyze_socu_native_contact_reports.py \
  output/examples/socu_native_contact_direct \
  --require-native-plan \
  --require-evaluator direct \
  --require-no-triplets \
  --require-no-direct-fallbacks \
  --format markdown
```

The gate runner can wrap the same check:

```bash
uv run --no-project python scripts/run_socu_native_contact_gates.py \
  --build build/socu_native_contact \
  --mode report \
  --reports output/examples/socu_native_contact_direct \
  --report-evaluator direct \
  --require-no-triplets \
  --require-no-direct-fallbacks
```

## Python Scene Gates

Scene gates need Python package dependencies and the freshly built pyuipc module.
Use the `python/` uv project for dependencies, then point `PYTHONPATH` and
`LD_LIBRARY_PATH` at the build output.

```bash
BUILD_DIR=$PWD/build/socu_native_contact
CONFIG=RelWithDebInfo
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
uv run --project python python scripts/run_socu_native_contact_gates.py \
  --build ${BUILD_DIR} \
  --mode scene \
  --scene-evaluator direct \
  --scene-side-coverage global \
  --output output/examples/socu_native_contact_direct
```

If the backend libraries are under `${BUILD_DIR}/bin` instead of a config
subdirectory, set `MODULE_DIR=${BUILD_DIR}/bin`.

Direct example invocation without the gate runner:

```bash
BUILD_DIR=$PWD/build/socu_native_contact
CONFIG=RelWithDebInfo
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

SOCU_NATIVE_CONTACT_PLAN=1 \
SOCU_NATIVE_CONTACT_PLAN_EXECUTOR=1 \
SOCU_NATIVE_CONTACT_EVALUATOR=direct \
SOCU_NATIVE_CONTACT_SIDE_COVERAGE_MODE=global \
SOCU_REPORT_COUNTERS=1 \
LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
uv run --project python python python/examples/cuda_mixed_wrecking_ball_compare.py \
  --variant socu_rt50_topology_diag_lump \
  --frames 20 \
  --output output/examples/socu_native_contact_direct \
  --backend cuda_mixed_socu
```

## Command Policy

- Use `uv run --no-project python ...` for standard-library scripts in
  `scripts/`.
- Run `uv sync --project python --extra dev` before pybind builds or Python
  scene gates, and pass `-DUIPC_PYTHON_EXECUTABLE_PATH=$PWD/python/.venv/bin/python`
  to CMake.
- Use `uv run --project python python ...` for Python examples or viewers that
  need `numpy`, `polyscope`, `huggingface_hub`, or `pyuipc`.
- Use direct binary execution for C++ Catch tests when the test path is known.
- Record the exact configure command, build directory, binary path, GPU, driver,
  and report directory before accepting benchmark claims.
