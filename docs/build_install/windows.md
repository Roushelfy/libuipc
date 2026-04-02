# Build on Windows

## Prerequisites

The following dependencies are required to build the project.

| Name                                                | Version      | Usage           | Import         |
| --------------------------------------------------- | ------------ | --------------- | -------------- |
| [CMake](https://cmake.org/download/)                | >=3.26       | build system    | system install |
| [XMake](https://xmake.io/)                          | >=3.0.5      | build system    | system install |
| [Python](https://www.python.org/downloads/)         | >=3.11       | build system    | system install |
| [Cuda](https://developer.nvidia.com/cuda-downloads) | >=12.4       | GPU programming | system install |
| [Vcpkg](https://github.com/microsoft/vcpkg)         | >=2025.7.25  | package manager | git clone      |

## Install Vcpkg

If you haven't installed Vcpkg, you can clone the repository with the following command:

```shell
mkdir ~/Toolchain
cd ~/Toolchain
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
```

The simplest way to let CMake detect Vcpkg is to set the **System Environment Variable** `CMAKE_TOOLCHAIN_FILE` to `~/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake`

You can set the environment variable in the PowerShell:

```shell
# In PowerShell: Permanently set the environment variable
[System.Environment]::SetEnvironmentVariable("CMAKE_TOOLCHAIN_FILE", "~/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake", "User")
```

## Build Libuipc

Clone the repository with the following command:

```shell
git clone https://github.com/spiriMirror/libuipc.git
```

### CMake-GUI

On Windows, you can use the `CMake-GUI` to **configure** the project and **generate** the Visual Studio solution file with only a few clicks.

- Toggling the `UIPC_BUILD_PYBIND` option to `ON` to enable the Python binding.

### CMake-CLI

Or, you can use the following commands to build the project.

```shell
cd libuipc; mkdir build; cd build
cmake -S .. -DUIPC_BUILD_PYBIND=1
cmake --build . --config <Release/RelWithDebInfo> -j8
```

!!!NOTE
    Use multi-thread to speed up the build process as possible, becasue the NVCC compiler will take a lot of time.

### Build `cuda_mixed` with CMake

Use the `cuda_mixed` backend when you want a compile-time mixed-precision build.

```shell
cd libuipc; mkdir build_mixed; cd build_mixed
cmake -S .. ^
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON ^
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=path3 ^
  -DUIPC_WITH_NVTX=OFF
cmake --build . --config Release -j8
```

Valid precision levels are `fp64`, `path1`, `path2`, `path3`, `path4`, `path5`, `path6`, and `path7`.

If you also want optional side-by-side benchmark comparisons against the main CUDA backend, add `-DUIPC_WITH_CUDA_BACKEND=ON` during configure.

At runtime, load the mixed backend explicitly:

```cpp
Engine engine{"cuda_mixed"};
World  world{engine};
```

Precision selection is compile-time only in the current implementation.

## Build Libuipc with XMake

If you prefer XMake over CMake, use the following commands in PowerShell or `cmd`.

```shell
cd libuipc
xmake f -c
xmake build -j8
```

Enable Python bindings with the following configuration.

```shell
cd libuipc
xmake f --pybind=true --python_system=true --python_version=3.11.x -c
xmake build -j8
```

If you are building against another Python installation, replace `3.11.x` with the version you want XMake to resolve.

The build outputs are placed under `build/`, and the staged Python package is generated in `build/.xpack/pyuipc`.

## Run Project

Just run the executable files in `build/<Release/RelWithDebInfo>/bin` folder.

## Install Pyuipc 

With `UIPC_BUILD_PYBIND` option set to `ON`, the Python binding will be **built** and **installed** in the specified Python environment.

If some **errors** occur during the installation, you can try to **manually** install the Python binding.

```shell
cd build/python
pip install .
```

## Conda Environment (Alternative)

Create and activate a conda environment with the following command:

```shell
conda env create -f conda/env.yaml
conda activate uipc_env
```

Setup the `CMAKE_TOOLCHAIN_FILE` environment variable in the conda environment:

```shell
conda env config vars set CMAKE_TOOLCHAIN_FILE=~/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Then, you can build the project with the same commands as above in the conda environment.

## Check Installation

You can run the `uipc_info.py` to check if the `Pyuipc` is installed correctly.

```shell
cd libuipc/python
python uipc_info.py
```

More samples are at [Pyuipc Samples](https://github.com/spiriMirror/libuipc-samples).

## Install in Any Python Venv

If you want to install the Pyuipc to any Python Venv (like [uv](https://docs.astral.sh/uv/)) after build, you can use the following command:

```shell
cmake -S .. -DUIPC_BUILD_PYBIND=1 -DUIPC_PYTHON_EXECUTABLE_PATH=<YOUR_PYTHON_EXECUTABLE_PATH>
cmake --build . --config <Release/RelWithDebInfo> -j8
```
