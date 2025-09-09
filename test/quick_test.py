#!/usr/bin/env python3
"""
Quick installation test - runs locally without Docker
Use this to test in your current environment
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

def run_command(cmd, check=True):
    print(f"🔧 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))
    
    if result.stdout:
        print(f"📤 {result.stdout.strip()}")
    if result.stderr and result.returncode != 0:
        print(f"❌ {result.stderr.strip()}")
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    return result

def test_dependencies():
    """Quick dependency check"""
    print("📋 Checking dependencies...")
    
    deps = ["git", "cmake", "python3"]
    all_good = True
    
    for dep in deps:
        try:
            result = run_command([dep, "--version"], check=False)
            if result.returncode == 0:
                print(f"✅ {dep}: available")
            else:
                print(f"❌ {dep}: not found")
                all_good = False
        except FileNotFoundError:
            print(f"❌ {dep}: not found")
            all_good = False
    
    return all_good

def test_installation():
    """Test the installation scripts exist and are valid"""
    print("📦 Checking installation files...")
    
    files = ["auto_install.py", "setup_pip.py", "install.sh"]
    all_good = True
    
    for file in files:
        if Path(file).exists():
            print(f"✅ {file}: found")
        else:
            print(f"❌ {file}: missing")
            all_good = False
    
    return all_good

def test_dry_run():
    """Test installation in dry-run mode"""
    print("🧪 Running installation dry-run test...")
    
    # Create a simple test to check if our script would work
    test_code = '''
import sys
import os
sys.path.insert(0, ".")

def main():
    # Test if we can import our installer
    try:
        from auto_install import LibUIPC_Installer
        installer = LibUIPC_Installer(use_conda=False)
        print("✅ Installer class loaded successfully")
        
        # Test dependency checking
        missing = installer.check_dependencies()
        if missing:
            print(f"⚠️  Missing dependencies: {[dep for dep, _ in missing]}")
        else:
            print("✅ All dependencies satisfied")
            
        print("✅ Dry-run test passed")
        return True
    except Exception as e:
        print(f"❌ Dry-run test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
            
        result = run_command([sys.executable, test_file], check=False)
        os.unlink(test_file)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Dry-run failed: {e}")
        return False

def test_import_current():
    """Test if uipc is already installed and working"""
    print("🔍 Testing current uipc installation...")
    
    try:
        result = run_command([
            sys.executable, "-c",
            "import uipc; print(f'✅ Current uipc version: {getattr(uipc, \"__version__\", \"unknown\")}');"
            "scene = uipc.Scene(); print('✅ Basic functionality works')"
        ], check=False)
        
        if result.returncode == 0:
            print("✅ uipc is already installed and working")
            return True
        else:
            print("ℹ️  uipc not installed or not working")
            return False
    except Exception:
        print("ℹ️  uipc not installed")
        return False

def main():
    start_time = time.time()
    
    print("🚀 LibUIPC Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Installation Files", test_installation), 
        ("Dry Run", test_dry_run),
        ("Current Import", test_import_current),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 QUICK TEST RESULTS")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    elapsed = time.time() - start_time
    overall = all(result for _, result in results)
    overall_status = "✅ SUCCESS" if overall else "❌ FAILURE" 
    
    print("=" * 50)
    print(f"OVERALL: {overall_status}")
    print(f"TIME: {elapsed:.1f}s")
    print("=" * 50)
    
    if overall:
        print("\n🎉 Your environment looks ready for LibUIPC installation!")
        print("\nTo proceed with installation:")
        print("  ./install.sh                    # Auto install")
        print("  python3 auto_install.py        # Manual install")
        print("  pip install . -v               # Pip install")
    else:
        print("\n⚠️  Some issues were found. Please check the results above.")
    
    return overall

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)