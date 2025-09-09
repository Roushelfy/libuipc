# LibUIPC 自动安装指南

本项目提供了多种便捷的安装方式，支持 Windows 和 Linux 平台。

## 🚀 快速安装（推荐）

### Linux/macOS
```bash
# 克隆仓库
git clone https://github.com/spiriMirror/libuipc.git
cd libuipc

# 一键安装
./install.sh
```

### Windows
```cmd
# 克隆仓库
git clone https://github.com/spiriMirror/libuipc.git
cd libuipc

# 一键安装
install.bat
```

## 📦 pip 安装（实验性）

```bash
# 安装构建依赖
pip install scikit-build-core[pyproject] pybind11 cmake ninja

# 从源码安装
pip install . -v
```

## 🛠️ 高级安装选项

### 使用 Python 脚本直接安装
```bash
python3 auto_install.py [选项]
```

#### 可用选项：
- `--no-conda`: 不使用 conda 环境
- `--toolchain-dir PATH`: 自定义工具链目录 (默认: ~/Toolchain)
- `--build-dir PATH`: 自定义构建目录 (默认: CMakeBuild)  
- `--jobs N`: 并行构建任务数 (默认: CPU核心数)

### 设置 pip 安装
```bash
# 生成 pip 安装配置
python3 setup_pip.py

# 使用生成的配置安装
pip install . -v
```

## 📋 系统要求

### 必需依赖
- **CMake** ≥ 3.26
- **Python** ≥ 3.10  
- **CUDA** ≥ 12.4 (用于 GPU 支持)
- **Git** (用于克隆依赖)

### 自动安装的依赖
- **vcpkg** (包管理器，自动设置)
- 各种 C++ 库 (通过 vcpkg 自动安装)

## 🔧 安装过程说明

自动安装脚本会执行以下步骤：

1. **检查系统依赖** - 验证 CMake、Python、Git 等
2. **设置 vcpkg** - 自动下载并配置 vcpkg 包管理器
3. **创建 conda 环境** - 设置隔离的 Python 环境（可选）
4. **配置 CMake** - 使用适当的参数配置构建
5. **编译项目** - 并行编译 C++ 代码和 Python 绑定
6. **安装 Python 包** - 将编译好的包安装到 Python 环境
7. **验证安装** - 测试导入和基本功能

## ⚡ 性能提示

- **首次安装**: 可能需要 30-60 分钟，因为需要编译大量依赖
- **后续安装**: 由于 vcpkg 缓存，会显著加快
- **使用 SSD**: 强烈建议在 SSD 上进行编译
- **充足内存**: 建议至少 8GB RAM 用于并行编译

## 🐛 常见问题

### 1. libstdc++ 版本不兼容
```bash
# 更新 conda 环境中的 libstdc++
conda install -c conda-forge libstdcxx-ng

# 或使用系统库
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 2. CUDA 版本不匹配
- 确保安装 CUDA ≥ 12.4
- 检查驱动版本 ≥ 550.54.14

### 3. CMake 版本过低
```bash
# Ubuntu/Debian
sudo apt remove cmake
pip install cmake

# 或从官网下载最新版本
```

### 4. 内存不足
```bash
# 减少并行任务数
python3 auto_install.py --jobs 2
```

### 5. vcpkg 下载失败
```bash
# 手动设置 vcpkg
mkdir ~/Toolchain
cd ~/Toolchain  
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # Linux/macOS
# 或 ./bootstrap-vcpkg.bat  # Windows
```

## ✅ 验证安装

安装完成后，运行以下命令验证：

```python
import uipc
print("✅ LibUIPC 安装成功!")

# 检查版本信息
print(f"版本: {uipc.__version__}")

# 运行基本测试
scene = uipc.Scene()
print("✅ 基本功能正常!")
```

或运行测试脚本：
```bash
cd python
python uipc_info.py
```

## 🎯 开发者模式

对于开发者，建议使用以下配置：

```bash
# 启用开发模式和测试
python3 auto_install.py --no-conda
cd CMakeBuild
cmake -DUIPC_DEV_MODE=ON -DUIPC_BUILD_TESTS=ON ..
make -j$(nproc)
```

## 📞 获取帮助

- **文档**: https://spirimirror.github.io/libuipc-doc/
- **问题报告**: https://github.com/spiriMirror/libuipc/issues
- **讨论**: https://github.com/spiriMirror/libuipc/discussions

---

**注意**: 首次安装可能需要较长时间，请耐心等待。建议在稳定的网络环境下进行安装。