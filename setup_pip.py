#!/usr/bin/env python3
"""
LibUIPC pip-installable setup
This creates a pip-installable version using scikit-build-core
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform


def check_dependencies():
    """Check if required dependencies are available"""
    required = {
        "cmake": "CMake ≥3.26 is required",
        "git": "Git is required",
    }
    
    missing = []
    for cmd, desc in required.items():
        if not shutil.which(cmd):
            missing.append((cmd, desc))
    
    return missing

def setup_vcpkg():
    """Setup vcpkg if needed"""
    vcpkg_dir = Path.home() / "Toolchain" / "vcpkg"
    
    if not vcpkg_dir.exists():
        print("Setting up vcpkg...")
        toolchain_dir = vcpkg_dir.parent
        toolchain_dir.mkdir(parents=True, exist_ok=True)
        
        subprocess.run([
            "git", "clone", 
            "https://github.com/microsoft/vcpkg.git", 
            str(vcpkg_dir)
        ], check=True)
        
        # Bootstrap
        if platform.system().lower() == "windows":
            bootstrap_script = vcpkg_dir / "bootstrap-vcpkg.bat"
        else:
            bootstrap_script = vcpkg_dir / "bootstrap-vcpkg.sh"
            os.chmod(bootstrap_script, 0o755)
            
        subprocess.run([str(bootstrap_script)], cwd=vcpkg_dir, check=True)
    
    return vcpkg_dir

def create_pip_pyproject():
    """Create a pip-installable pyproject.toml"""
    content = '''[build-system]
requires = [
    "scikit-build-core[pyproject]>=0.8.0",
    "pybind11>=2.10.0",
    "cmake>=3.26.0",
    "ninja",
]
build-backend = "scikit_build_core.build"

[project]
name = "libuipc"
version = "0.9.0"
description = "Unified Incremental Potential Contact (IPC) library"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Zihang Zhu", email = "paradise_craftsman@hotmail.com"},
]
maintainers = [
    {name = "Zihang Zhu", email = "paradise_craftsman@hotmail.com"},
    {name = "Lu Xinyu", email = "lxy819469559@gmail.com"}
]
keywords = ["ipc", "computer graphics", "physics simulation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Physics",
]

requires-python = ">=3.10"
dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
gui = ["polyscope"]
dev = ["pytest", "pytest-cov"]
test = ["pytest"]

[project.urls]
Homepage = "https://spirimirror.github.io/libuipc-doc/"
Documentation = "https://spirimirror.github.io/libuipc-doc/"
Repository = "https://github.com/spiriMirror/libuipc"
Issues = "https://github.com/spiriMirror/libuipc/issues"

[tool.scikit-build]
# Specify CMake args
cmake.args = [
    "-DUIPC_BUILD_PYBIND=ON",
    "-DUIPC_BUILD_EXAMPLES=OFF", 
    "-DUIPC_BUILD_TESTS=OFF",
    "-DUIPC_BUILD_BENCHMARKS=OFF",
    "-DUIPC_BUILD_GUI=OFF",
]

cmake.build-type = "Release"
cmake.verbose = true

# Install directory for the CMake project
install.components = ["python"]

# Override wheel directory
wheel.install-dir = "libuipc"

[tool.scikit-build.cmake.define]
# Set vcpkg toolchain if available
CMAKE_TOOLCHAIN_FILE = {env="CMAKE_TOOLCHAIN_FILE", default=""}

[tool.pytest.ini_options]
addopts = ["-v", "-s"]
testpaths = ["python/tests"]
markers = [
    "basic: mark a test as basic",
    "gui: mark a test as requiring GUI", 
    "typing: mark a test as requiring type checking",
    "example: mark a test as an example",
]

[tool.cibuildwheel]
# Build wheels for common platforms
build = ["cp310-*", "cp311-*", "cp312-*"]
skip = ["*-win32", "*-manylinux_i686", "*-musllinux_*"]

# Set environment variables for building
[tool.cibuildwheel.environment]
CMAKE_TOOLCHAIN_FILE = "~/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"

[tool.cibuildwheel.linux]
before-all = [
    "yum update -y",
    "yum install -y cuda-toolkit-12-4",
    "mkdir -p ~/Toolchain",
    "cd ~/Toolchain && git clone https://github.com/microsoft/vcpkg.git && cd vcpkg && ./bootstrap-vcpkg.sh",
]

[tool.cibuildwheel.windows] 
before-all = [
    "mkdir C:\\Toolchain",
    "cd C:\\Toolchain && git clone https://github.com/microsoft/vcpkg.git && cd vcpkg && .\\bootstrap-vcpkg.bat",
]
environment = {CMAKE_TOOLCHAIN_FILE = "C:/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"}

[tool.cibuildwheel.macos]
before-all = [
    "mkdir -p ~/Toolchain", 
    "cd ~/Toolchain && git clone https://github.com/microsoft/vcpkg.git && cd vcpkg && ./bootstrap-vcpkg.sh",
]
'''

    return content

def create_cmake_wrapper():
    """Create a CMakeLists.txt wrapper for pip installation"""
    content = '''cmake_minimum_required(VERSION 3.26)

# Set up project
project(libuipc_pip)

# Use existing CMakeLists.txt but override some options
set(UIPC_BUILD_PYBIND ON CACHE BOOL "Build Python bindings" FORCE)
set(UIPC_BUILD_EXAMPLES OFF CACHE BOOL "Skip examples" FORCE)
set(UIPC_BUILD_TESTS OFF CACHE BOOL "Skip tests" FORCE)
set(UIPC_BUILD_BENCHMARKS OFF CACHE BOOL "Skip benchmarks" FORCE)
set(UIPC_BUILD_GUI OFF CACHE BOOL "Skip GUI" FORCE)

# Set vcpkg toolchain if available and not already set
if(NOT CMAKE_TOOLCHAIN_FILE)
    if(EXISTS "$ENV{HOME}/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{HOME}/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake")
    elseif(EXISTS "C:/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake") 
        set(CMAKE_TOOLCHAIN_FILE "C:/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake")
    endif()
endif()

# Include the main CMakeLists.txt from the project root
include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")

# Install Python package
if(TARGET pyuipc)
    # Install the pybind module
    install(TARGETS pyuipc 
            COMPONENT python
            DESTINATION libuipc)
    
    # Install Python source files
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/src/"
            COMPONENT python
            DESTINATION libuipc
            FILES_MATCHING PATTERN "*.py")
            
    # Install type stubs if they exist
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/typings/"
            COMPONENT python  
            DESTINATION libuipc
            FILES_MATCHING PATTERN "*.pyi"
            OPTIONAL)
endif()
'''
    
    return content

def main():
    """Main setup function"""
    print("🚀 Setting up LibUIPC for pip installation...")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("❌ Missing dependencies:")
        for cmd, desc in missing:
            print(f"  - {cmd}: {desc}")
        return False
    
    # Setup vcpkg
    vcpkg_dir = setup_vcpkg()
    
    # Set environment variable for CMake toolchain
    toolchain_file = vcpkg_dir / "scripts/buildsystems/vcpkg.cmake"
    os.environ["CMAKE_TOOLCHAIN_FILE"] = str(toolchain_file)
    
    # Create pip-friendly files
    print("📦 Creating pip-installable package files...")
    
    # Create pyproject.toml for pip
    pip_pyproject = create_pip_pyproject()
    with open("pyproject_pip.toml", "w") as f:
        f.write(pip_pyproject)
    print("  ✅ Created pyproject_pip.toml")
    
    # Create CMakeLists wrapper
    cmake_wrapper = create_cmake_wrapper()
    with open("CMakeLists_pip.txt", "w") as f:
        f.write(cmake_wrapper)
    print("  ✅ Created CMakeLists_pip.txt")
    
    # Create simple setup instructions
    instructions = """
# LibUIPC pip installation setup

## Option 1: Install directly from source (recommended)
```bash
# Install dependencies first
pip install scikit-build-core[pyproject] pybind11 cmake ninja

# Install the package
pip install . -v
```

## Option 2: Use the pip-optimized configuration
```bash
# Use the pip-optimized files
cp pyproject_pip.toml pyproject.toml
cp CMakeLists_pip.txt CMakeLists.txt

# Install
pip install . -v
```

## Option 3: Build wheel for distribution
```bash
pip install build cibuildwheel
python -m build --wheel
```

## Requirements
- CMake ≥ 3.26
- CUDA ≥ 12.4 (for GPU support)
- Python ≥ 3.10
- vcpkg (automatically set up)

## Notes
- The installation will automatically set up vcpkg
- First installation may take 30+ minutes due to dependency compilation
- Subsequent installs will be faster due to vcpkg caching
"""
    
    with open("PIP_INSTALL.md", "w") as f:
        f.write(instructions)
    print("  ✅ Created PIP_INSTALL.md with instructions")
    
    print("\n🎉 Setup completed!")
    print("\nTo install with pip:")
    print("  pip install . -v")
    print("\nSee PIP_INSTALL.md for detailed instructions.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)