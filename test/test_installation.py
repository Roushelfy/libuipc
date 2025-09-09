#!/usr/bin/env python3
"""
LibUIPC Installation Test Suite
Tests the installation process in clean environments
"""

import os
import sys
import subprocess
import time
import tempfile
import shutil
from pathlib import Path

class InstallationTester:
    def __init__(self):
        self.start_time = time.time()
        self.test_results = []
        
    def log(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, cmd, timeout=3600, check=True):
        """Run command with timeout and logging"""
        self.log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=isinstance(cmd, str)
            )
            
            if result.stdout:
                self.log(f"STDOUT: {result.stdout.strip()}")
            if result.stderr:
                self.log(f"STDERR: {result.stderr.strip()}")
                
            if check and result.returncode != 0:
                raise RuntimeError(f"Command failed with code {result.returncode}")
                
            return result
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout}s", "ERROR")
            raise
            
    def test_dependencies(self):
        """Test if required dependencies are available"""
        self.log("Testing system dependencies...")
        
        deps = {
            "git": "git --version",
            "cmake": "cmake --version", 
            "python3": "python3 --version",
            "conda": "conda --version"
        }
        
        results = {}
        for name, cmd in deps.items():
            try:
                result = self.run_command(cmd, check=False)
                results[name] = result.returncode == 0
                if results[name]:
                    self.log(f"✅ {name} available")
                else:
                    self.log(f"❌ {name} not available", "WARNING")
            except Exception as e:
                results[name] = False
                self.log(f"❌ {name} check failed: {e}", "ERROR")
                
        return results
        
    def test_auto_install(self, use_conda=True):
        """Test the auto_install.py script"""
        self.log("Testing auto_install.py...")
        
        try:
            # Clone the repository first
            if not Path("libuipc").exists():
                self.log("Cloning LibUIPC repository (cuda-13.0-support branch)...")
                self.run_command([
                    "git", "clone", "--depth", "1", 
                    "--branch", "cuda-13.0-support",
                    "https://github.com/Roushelfy/libuipc.git"
                ])
            
            os.chdir("libuipc")
            
            # Copy our installation files
            shutil.copy("../auto_install.py", ".")
            shutil.copy("../install.sh", ".")
            
            # Run installation
            cmd = ["python3", "auto_install.py"]
            if not use_conda:
                cmd.append("--no-conda")
            cmd.extend(["--jobs", "2"])  # Limit jobs for container
            
            self.log("Starting installation process...")
            result = self.run_command(cmd, timeout=7200)  # 2 hour timeout
            
            return result.returncode == 0
            
        except Exception as e:
            self.log(f"Auto install failed: {e}", "ERROR")
            return False
        finally:
            os.chdir("..")
            
    def test_pip_install(self):
        """Test pip installation method"""
        self.log("Testing pip installation...")
        
        try:
            if not Path("libuipc").exists():
                self.log("Cloning LibUIPC repository (cuda-13.0-support branch)...")
                self.run_command([
                    "git", "clone", "--depth", "1",
                    "--branch", "cuda-13.0-support", 
                    "https://github.com/Roushelfy/libuipc.git"
                ])
            
            os.chdir("libuipc")
            
            # Copy pip setup
            shutil.copy("../setup_pip.py", ".")
            
            # Setup pip environment
            self.log("Setting up pip installation...")
            self.run_command(["python3", "setup_pip.py"])
            
            # Install dependencies
            self.log("Installing build dependencies...")
            self.run_command([
                "pip", "install", 
                "scikit-build-core[pyproject]", 
                "pybind11", "cmake", "ninja"
            ])
            
            # Install package
            self.log("Installing LibUIPC via pip...")
            self.run_command(["pip", "install", ".", "-v"], timeout=7200)
            
            return True
            
        except Exception as e:
            self.log(f"Pip install failed: {e}", "ERROR")
            return False
        finally:
            os.chdir("..")
            
    def test_import(self):
        """Test if the package can be imported"""
        self.log("Testing package import...")
        
        try:
            result = self.run_command([
                "python3", "-c", 
                "import uipc; print(f'✅ LibUIPC version: {getattr(uipc, \"__version__\", \"unknown\")}'); "
                "scene = uipc.Scene(); print('✅ Basic functionality works')"
            ])
            return result.returncode == 0
        except Exception as e:
            self.log(f"Import test failed: {e}", "ERROR")
            return False
            
    def run_basic_test(self):
        """Run basic functionality test"""
        self.log("Running basic functionality test...")
        
        test_code = '''
import uipc
import numpy as np

# Create a scene
scene = uipc.Scene()
print("✅ Scene created")

# Test basic geometry creation  
try:
    from uipc.geometry import ground
    g = ground()
    print("✅ Ground geometry created")
except ImportError:
    print("⚠️  Ground geometry not available")

print("✅ Basic test completed successfully")
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                test_file = f.name
                
            result = self.run_command(["python3", test_file])
            os.unlink(test_file)
            
            return result.returncode == 0
        except Exception as e:
            self.log(f"Basic test failed: {e}", "ERROR")
            return False
            
    def run_full_test_suite(self, method="auto"):
        """Run full test suite"""
        self.log(f"🚀 Starting LibUIPC installation test suite (method: {method})")
        
        # Test 1: Dependencies
        deps = self.test_dependencies()
        self.test_results.append(("dependencies", all(deps.values())))
        
        # Test 2: Installation
        if method == "auto":
            install_success = self.test_auto_install()
        elif method == "pip":
            install_success = self.test_pip_install()
        else:
            self.log(f"Unknown installation method: {method}", "ERROR")
            install_success = False
            
        self.test_results.append(("installation", install_success))
        
        if not install_success:
            self.log("Installation failed, skipping further tests", "ERROR")
            return False
            
        # Test 3: Import
        import_success = self.test_import()
        self.test_results.append(("import", import_success))
        
        # Test 4: Basic functionality
        if import_success:
            basic_success = self.run_basic_test()
            self.test_results.append(("basic_test", basic_success))
        else:
            basic_success = False
            self.test_results.append(("basic_test", False))
            
        # Report results
        self.report_results()
        
        return all(result for _, result in self.test_results)
        
    def report_results(self):
        """Report test results"""
        elapsed = time.time() - self.start_time
        
        self.log("=" * 60)
        self.log("🏁 TEST RESULTS")
        self.log("=" * 60)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            self.log(f"{test_name.upper()}: {status}")
            
        overall = all(result for _, result in self.test_results)
        overall_status = "✅ SUCCESS" if overall else "❌ FAILURE"
        
        self.log("=" * 60)
        self.log(f"OVERALL: {overall_status}")
        self.log(f"TIME: {elapsed:.1f}s")
        self.log("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LibUIPC installation")
    parser.add_argument("--method", choices=["auto", "pip"], default="auto",
                       help="Installation method to test")
    parser.add_argument("--no-conda", action="store_true",
                       help="Don't use conda (for auto method)")
    
    args = parser.parse_args()
    
    tester = InstallationTester()
    success = tester.run_full_test_suite(method=args.method)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()