#!/usr/bin/env python3
"""
LibUIPC Auto-Installation Script
Automatically installs LibUIPC on Windows and Linux platforms with pybind support.
"""

import os
import sys
import platform
import subprocess
import shutil
import argparse
from pathlib import Path
import tempfile
import urllib.request
import tarfile
import zipfile

class LibUIPC_Installer:
    def __init__(self, use_conda=True, toolchain_dir=None, build_dir=None, jobs=None):
        self.platform_system = platform.system().lower()
        self.use_conda = use_conda
        self.toolchain_dir = Path(toolchain_dir) if toolchain_dir else Path.home() / "Toolchain"
        self.build_dir = Path(build_dir) if build_dir else Path("CMakeBuild") 
        self.jobs = jobs if jobs else os.cpu_count()
        self.vcpkg_dir = self.toolchain_dir / "vcpkg"
        
        print(f"🚀 LibUIPC Auto-Installer")
        print(f"Platform: {self.platform_system}")
        print(f"Use Conda: {self.use_conda}")
        print(f"Toolchain Dir: {self.toolchain_dir}")
        print(f"Build Dir: {self.build_dir}")

    def run_command(self, cmd, cwd=None, check=True):
        """Execute shell command safely"""
        print(f"🔧 Running: {cmd}")
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        
        if result.stdout:
            print(f"  📤 {result.stdout.strip()}")
        if result.stderr:
            print(f"  ❌ {result.stderr.strip()}")
            
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
            
        return result

    def check_command_exists(self, cmd):
        """Check if a command exists in PATH"""
        return shutil.which(cmd) is not None

    def check_dependencies(self):
        """Check required dependencies"""
        print("\n📋 Checking dependencies...")
        
        required = {
            "git": "Git is required for cloning repositories",
            "cmake": "CMake ≥3.26 is required"
        }
        
        missing = []
        for cmd, desc in required.items():
            if not self.check_command_exists(cmd):
                missing.append((cmd, desc))
                print(f"  ❌ Missing: {cmd} - {desc}")
            else:
                print(f"  ✅ Found: {cmd}")
        
        # Check CMake version
        if self.check_command_exists("cmake"):
            result = self.run_command("cmake --version", check=False)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"  ℹ️  {version_line}")
        
        # Check Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info < (3, 10):
            missing.append(("python", f"Python ≥3.10 required (current: {python_version})"))
        else:
            print(f"  ✅ Python {python_version}")
            
        return missing

    def install_vcpkg(self):
        """Install or update vcpkg"""
        print(f"\n🛠️  Setting up vcpkg at {self.vcpkg_dir}")
        
        self.toolchain_dir.mkdir(parents=True, exist_ok=True)
        
        if self.vcpkg_dir.exists():
            print("  ℹ️  vcpkg directory exists, updating...")
            self.run_command("git pull", cwd=self.vcpkg_dir)
        else:
            print("  📥 Cloning vcpkg...")
            self.run_command(f"git clone https://github.com/microsoft/vcpkg.git {self.vcpkg_dir}")
        
        # Bootstrap vcpkg
        if self.platform_system == "windows":
            bootstrap_cmd = str(self.vcpkg_dir / "bootstrap-vcpkg.bat")
        else:
            bootstrap_cmd = str(self.vcpkg_dir / "bootstrap-vcpkg.sh")
            # Make executable
            os.chmod(bootstrap_cmd, 0o755)
            
        print("  🔧 Bootstrapping vcpkg...")
        self.run_command(bootstrap_cmd, cwd=self.vcpkg_dir)

    def setup_conda_env(self):
        """Setup conda environment"""
        if not self.use_conda:
            return None
            
        print("\n🐍 Setting up Conda environment...")
        
        if not self.check_command_exists("conda"):
            print("  ❌ Conda not found, skipping conda setup")
            return None
            
        # Check if environment exists
        result = self.run_command("conda env list", check=False)
        env_exists = "uipc_env" in result.stdout
        
        env_yaml = Path("conda/env.yaml")
        if env_yaml.exists():
            if env_exists:
                print("  ✅ uipc_env already exists, updating...")
                self.run_command("conda env update -f conda/env.yaml")
            else:
                print("  📦 Creating uipc_env from conda/env.yaml...")
                self.run_command("conda env create -f conda/env.yaml")
        else:
            print("  ⚠️  conda/env.yaml not found, creating minimal environment...")
            if not env_exists:
                self.run_command("conda create -n uipc_env python=3.11 cmake cuda-toolkit=12.4 -y")
        
        return "uipc_env"

    def get_cmake_toolchain_path(self):
        """Get CMake toolchain file path"""
        toolchain_file = self.vcpkg_dir / "scripts/buildsystems/vcpkg.cmake"
        return str(toolchain_file)

    def configure_cmake(self, python_executable=None):
        """Configure CMake build"""
        print(f"\n🏗️  Configuring CMake build...")
        
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        cmake_args = [
            "cmake",
            "-S", ".",
            "-B", str(self.build_dir),
            "-DUIPC_BUILD_PYBIND=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_TOOLCHAIN_FILE={self.get_cmake_toolchain_path()}"
        ]
        
        if python_executable:
            cmake_args.extend([f"-DUIPC_PYTHON_EXECUTABLE_PATH={python_executable}"])
            
        if self.platform_system == "windows":
            cmake_args.extend(["-A", "x64"])
            
        self.run_command(cmake_args)

    def build_project(self):
        """Build the project"""
        print(f"\n🔨 Building project with {self.jobs} jobs...")
        
        build_args = [
            "cmake", 
            "--build", str(self.build_dir),
            "--config", "Release",
            "--parallel", str(self.jobs)
        ]
        
        self.run_command(build_args)

    def install_python_package(self):
        """Install Python package"""
        print("\n📦 Installing Python package...")
        
        python_build_dir = self.build_dir / "python"
        if not python_build_dir.exists():
            raise RuntimeError(f"Python build directory not found: {python_build_dir}")
        
        # Install using pip
        self.run_command(["pip", "install", "."], cwd=python_build_dir)

    def verify_installation(self):
        """Verify the installation"""
        print("\n✅ Verifying installation...")
        
        try:
            # Test import
            result = self.run_command([sys.executable, "-c", "import uipc; print('✅ uipc import successful')"], check=False)
            if result.returncode == 0:
                print("  ✅ Python package import successful")
            else:
                print("  ❌ Python package import failed")
                return False
                
            # Run info script if exists
            info_script = Path("python/uipc_info.py")
            if info_script.exists():
                print("  🔍 Running uipc_info.py...")
                self.run_command([sys.executable, str(info_script)], check=False)
                
            return True
        except Exception as e:
            print(f"  ❌ Verification failed: {e}")
            return False

    def install(self):
        """Main installation process"""
        try:
            # Check if we're in the right directory
            if not Path("CMakeLists.txt").exists():
                raise RuntimeError("❌ Not in LibUIPC root directory. Please run from the project root.")
            
            # Check dependencies
            missing = self.check_dependencies()
            if missing:
                print("\n❌ Missing dependencies:")
                for cmd, desc in missing:
                    print(f"  - {cmd}: {desc}")
                print("\nPlease install missing dependencies and try again.")
                return False
            
            # Setup vcpkg
            self.install_vcpkg()
            
            # Setup conda environment
            conda_env = self.setup_conda_env()
            
            # Get Python executable
            if conda_env and self.use_conda:
                # Try to get conda env python path
                result = self.run_command(f"conda activate {conda_env} && python -c \"import sys; print(sys.executable)\"", check=False)
                python_executable = result.stdout.strip() if result.returncode == 0 else None
            else:
                python_executable = sys.executable
                
            print(f"  🐍 Using Python: {python_executable}")
            
            # Configure CMake
            self.configure_cmake(python_executable)
            
            # Build project
            self.build_project()
            
            # Install Python package
            self.install_python_package()
            
            # Verify installation
            if self.verify_installation():
                print("\n🎉 Installation completed successfully!")
                print("\nTo use LibUIPC:")
                print("  import uipc")
                return True
            else:
                print("\n⚠️  Installation completed but verification failed.")
                return False
                
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Auto-install LibUIPC with pybind support")
    parser.add_argument("--no-conda", action="store_true", help="Don't use conda environment")
    parser.add_argument("--toolchain-dir", help="Custom toolchain directory (default: ~/Toolchain)")
    parser.add_argument("--build-dir", help="Custom build directory (default: CMakeBuild)")
    parser.add_argument("--jobs", "-j", type=int, help="Number of parallel build jobs")
    
    args = parser.parse_args()
    
    installer = LibUIPC_Installer(
        use_conda=not args.no_conda,
        toolchain_dir=args.toolchain_dir,
        build_dir=args.build_dir,
        jobs=args.jobs
    )
    
    success = installer.install()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()