# Mixed Backend Path 运行手册

这页记录如何正确运行 `cuda_mixed` 后端的 `fp64` 和 `path1` 到 `path6`，重点避免 Python 绑定目录、backend 动态库目录、以及 benchmark worker 环境混用。

## 一句话原则

`cuda_mixed` 的 path 是编译期选择，不是运行期配置。运行某个 path 时，必须同时使用同一个 build 目录里的 Python 绑定和 backend 动态库。

| 运行时输入 | 正确来源 |
|---|---|
| Python package | `build/build_impl_<level>/python/src` |
| Backend module dir | `build/build_impl_<level>/<config>/bin` |
| Backend name | `cuda_mixed` |
| Precision level | CMake configure 时的 `UIPC_CUDA_MIXED_PRECISION_LEVEL` |

这里的 `<config>` 是实际构建配置，例如 `Release` 或 `RelWithDebInfo`。`uipc_assets` CLI 默认传 `--config RelWithDebInfo`，但如果对应目录不存在，会继续在 build 目录下查找 backend `.so`。手动运行时请自己确认 module dir 真实存在。

不要只设置 `PYTHONPATH=build/build_impl_<level>/python/src` 就直接跑。`uipc` Python package 里通常还有一份 `uipc/_native` 动态库拷贝，它可能比 `<config>/bin` 旧。如果只依赖默认 import 行为，可能会加载旧 backend，出现看起来像 `fp64/path1` 数值坏掉的假象。

## Level 矩阵

仓库约定每个 level 一个 build 目录：

| Level | Build dir |
|---|---|
| `fp64` | `build/build_impl_fp64` |
| `path1` | `build/build_impl_path1` |
| `path2` | `build/build_impl_path2` |
| `path3` | `build/build_impl_path3` |
| `path4` | `build/build_impl_path4` |
| `path5` | `build/build_impl_path5` |
| `path6` | `build/build_impl_path6` |

精度含义见 [Mixed Precision Overview](index.md) 和 [Mixed Precision Scope](precision_scope.md)。运行时不能把一个 build 目录从 `path6` 切成 `fp64`；需要换 build 目录。

## 推荐构建方式

如果要准备完整矩阵，优先用脚本生成默认 build 目录：

```powershell
powershell -File scripts/setup_mixed_uipc_assets_builds.ps1
```

这个脚本会串行准备 `fp64` 到 `path6`。单个 level 内部仍然使用 `cmake --build --parallel` 并行编译。内存紧张时不要同时编多个 level；一轮两个 level 已经比较保守。

单个 level 的手动 CMake 模板：

```shell
LEVEL=path6
BUILD_DIR=build/build_impl_${LEVEL}

cmake -S . -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DUIPC_BUILD_BENCHMARKS=ON \
  -DUIPC_BUILD_TESTS=ON \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_GUI=OFF \
  -DUIPC_BUILD_PYBIND=ON \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=${LEVEL}

cmake --build ${BUILD_DIR} --config RelWithDebInfo --parallel 8
```

如果要构建 `fp64`，把 `LEVEL=fp64` 即可。

## Python 脚本和 Viewer

手动跑 Python 脚本时，必须同时设置 `PYTHONPATH` 和动态库搜索路径。Linux 模板如下：

```shell
LEVEL=path6
CONFIG=RelWithDebInfo
BUILD_DIR=$PWD/build/build_impl_${LEVEL}
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
python/.venv/bin/python3 python/examples/cuda_mixed_abd_bdf2_viewer.py --smoke-frames 3
```

如果你的当前 build 目录实际是 `Release/bin`，把 `CONFIG=RelWithDebInfo` 改成 `CONFIG=Release`。

如果不确定当前 build 目录里实际使用哪个 config，可以先查：

```shell
LEVEL=path6
find build/build_impl_${LEVEL} -name libuipc_backend_cuda_mixed.so -printf '%h\n' | sort
```

交互 viewer 只需要去掉 `--smoke-frames`：

```shell
LEVEL=path6
CONFIG=RelWithDebInfo
BUILD_DIR=$PWD/build/build_impl_${LEVEL}
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
python/.venv/bin/python3 python/examples/cuda_mixed_abd_bdf2_viewer.py
```

Windows PowerShell 对应写法：

```powershell
$Level = "path6"
$Config = "RelWithDebInfo"
$BuildDir = (Resolve-Path "build/build_impl_$Level").Path
$ModuleDir = "$BuildDir/$Config/bin"
$env:PATH = "$ModuleDir;$env:PATH"
$env:PYTHONPATH = "$BuildDir/python/src;$env:PYTHONPATH"
python\.venv\Scripts\python.exe python\examples\cuda_mixed_abd_bdf2_viewer.py --smoke-frames 3
```

macOS 对应把 `LD_LIBRARY_PATH` 换成 `DYLD_LIBRARY_PATH`。

### 自己写 Python 场景时

如果脚本里自己创建 `Engine("cuda_mixed", workspace)`，建议在创建 engine 之前显式初始化 module dir：

```python
from pathlib import Path
import uipc

level = "path6"
config = "RelWithDebInfo"
module_dir = Path("build") / f"build_impl_{level}" / config / "bin"

cfg = uipc.default_config()
cfg["module_dir"] = str(module_dir.resolve())
uipc.init(cfg)

from uipc import Engine, World

engine = Engine("cuda_mixed", "output/my_workspace")
world = World(engine)
```

即使显式调用了 `uipc.init(cfg)`，仍然建议 shell 环境里把同一个 module dir 放进 `LD_LIBRARY_PATH`，因为 backend `.so` 还依赖同目录下的其他 libuipc 动态库。

### Python example helper

`python/examples/cuda_mixed_runtime.py` 提供了一个轻量 helper。当前 `cuda_mixed_*_viewer.py` 和 joint limit GUI demo 在 `backend=cuda_mixed` 时会尝试使用它。

这个 helper 只会在以下几种情况下切到 build 下的 module dir：

| 条件 | 行为 |
|---|---|
| 设置了 `UIPC_MODULE_DIR` | 使用这个目录作为 module dir |
| `UIPC_CONFIG` 指向的 `<config>/bin` 已在动态库搜索路径中 | 优先使用这个 config |
| 常见 `<config>/bin` 已在 `LD_LIBRARY_PATH` / `PATH` / `DYLD_LIBRARY_PATH` 中 | 使用当前 build tree 的 module dir |

如果没有满足这些条件，helper 不会强行切换，避免出现“找到了 backend `.so`，但其依赖库不在动态库搜索路径中”的加载失败。

## uipc-assets Benchmark

`apps/benchmarks/mixed/uipc_assets` 是最不容易踩坑的入口。它会为每个 worker 自动注入：

| 环境 / 配置 | 来源 |
|---|---|
| `PYTHONPATH` | 当前 level 的 `build/build_impl_<level>/python/src` |
| `LD_LIBRARY_PATH` | 当前 level 的 module dir，默认优先 `build/build_impl_<level>/<config>/bin` |
| `uipc.init()["module_dir"]` | 同一个 module dir |

推荐完整矩阵命令：

```shell
python apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/representative.json \
  --levels fp64 path1 path2 path3 path4 path5 path6 \
  --build fp64=build/build_impl_fp64 \
  --build path1=build/build_impl_path1 \
  --build path2=build/build_impl_path2 \
  --build path3=build/build_impl_path3 \
  --build path4=build/build_impl_path4 \
  --build path5=build/build_impl_path5 \
  --build path6=build/build_impl_path6 \
  --run_root output/benchmarks/mixed/uipc_assets/mixed_manual_run
```

如果使用默认 build 目录，`--build ...` 可以省略。benchmark 会通过 `apps/benchmarks/mixed/uipc_assets/core/builds.py` 查找 module dir，优先顺序是：

| 优先级 | 候选目录 |
|---|---|
| 1 | `<build_dir>/<config>/bin`，CLI 默认 `config=RelWithDebInfo` |
| 2 | `<build_dir>/bin` |
| 3 | `<build_dir>` |
| 4 | 在 build dir 下递归查找 `libuipc_backend_cuda_mixed.so` |

因此 benchmark 正常情况下会避开 Python `_native` 里的旧 backend 拷贝。

## C++ Tests 和本地可执行文件

CTest 或直接运行 C++ 可执行文件时，也要让当前 build 的 module dir 在动态库搜索路径里。

构建并运行某个 test target：

```shell
LEVEL=path6
CONFIG=RelWithDebInfo
BUILD_DIR=$PWD/build/build_impl_${LEVEL}
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

cmake --build ${BUILD_DIR} --target uipc_test_sim_case --config RelWithDebInfo --parallel 8

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
ctest --test-dir ${BUILD_DIR} -C RelWithDebInfo -R cuda_mixed --output-on-failure
```

直接运行某个 build 产物时：

```shell
LEVEL=path6
CONFIG=RelWithDebInfo
BUILD_DIR=$PWD/build/build_impl_${LEVEL}
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
${MODULE_DIR}/<executable>
```

如果一个 test 运行的是 `Engine{"cuda_mixed"}`，它使用的是当前进程能找到的 `cuda_mixed` backend module。不要把 `path6` 的可执行文件和 `fp64` 的 module dir 混在一起。

## 快速 smoke 矩阵

以下命令可以快速确认 `fp64` 到 `path6` 的 Python smoke 是否都能加载正确 backend。每个脚本只跑 3 帧。

```shell
for LEVEL in fp64 path1 path2 path3 path4 path5 path6; do
  CONFIG=RelWithDebInfo
  BUILD_DIR=$PWD/build/build_impl_${LEVEL}
  MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin
  echo "== ${LEVEL} =="
  LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
  PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
  python/.venv/bin/python3 python/examples/cuda_mixed_particle_ground_viewer.py --smoke-frames 3

  LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
  PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
  python/.venv/bin/python3 python/examples/cuda_mixed_abd_bdf2_viewer.py --smoke-frames 3

  LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
  PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
  python/.venv/bin/python3 python/examples/cuda_mixed_fem_mas_hybrid_viewer.py --smoke-frames 3
done
```

需要 joint limit viewer 时再加：

```shell
LEVEL=path6
CONFIG=RelWithDebInfo
BUILD_DIR=$PWD/build/build_impl_${LEVEL}
MODULE_DIR=${BUILD_DIR}/${CONFIG}/bin

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
python/.venv/bin/python3 python/examples/cuda_mixed_prismatic_joint_limit_viewer.py --smoke-frames 3

LD_LIBRARY_PATH=${MODULE_DIR}:${LD_LIBRARY_PATH} \
PYTHONPATH=${BUILD_DIR}/python/src:${PYTHONPATH} \
python/.venv/bin/python3 python/examples/cuda_mixed_revolute_joint_limit_viewer.py --smoke-frames 3
```

## 常见坑和排查

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `fp64/path1` 报 FusedPCG NaN，但 `path2+` 正常 | 可能加载了 Python `_native` 里的旧 backend | 同时设置 `LD_LIBRARY_PATH=<module_dir>` 和 `PYTHONPATH=<build>/python/src`，或使用 benchmark CLI |
| `RuntimeError: Could not load library ".../libuipc_backend_cuda_mixed.so"` | 找到了 backend `.so`，但同目录依赖库不在动态库搜索路径里 | 把同一个 module dir 放进 `LD_LIBRARY_PATH` / `PATH` / `DYLD_LIBRARY_PATH` |
| `ModuleNotFoundError: No module named 'uipc'` | 没有设置当前 level 的 `python/src` | 设置 `PYTHONPATH=<build>/python/src` |
| `ModuleNotFoundError` 出现在 example wrapper 之间 | 从不包含 `python/examples` 的入口绕过了脚本正常路径 | 直接运行脚本文件，或把 `python/examples` 加入 `PYTHONPATH` |
| benchmark 里质量对比缺 `fp64` reference | `run` 的 `--levels` 没包含 `fp64` | `uipc_assets run` 必须包含 `fp64` |
| 结果像是 path 被切换但数值不对 | 混用了不同 level 的 `PYTHONPATH` 和 module dir | 保证两者来自同一个 `build/build_impl_<level>` |

快速查看一个 build 目录里是否有多份 backend `.so`：

```shell
LEVEL=fp64
find build/build_impl_${LEVEL} -name libuipc_backend_cuda_mixed.so -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

如果同时看到 `python/src/uipc/_native/libuipc_backend_cuda_mixed.so` 和 `<config>/bin/libuipc_backend_cuda_mixed.so`，手动运行时优先使用 `<config>/bin`。`_native` 只应当视为 Python binding 包内的默认拷贝，不应当作为 mixed path benchmark 和调试的权威 backend。

## 不要这样做

- 不要只设置 `PYTHONPATH` 后直接相信当前加载的一定是最新 backend。
- 不要用 `path6` 的 `python/src` 配 `fp64` 的 module dir。
- 不要在 Scene config 里试图把 `path6` 改成 `fp64`；precision level 不是运行期配置。
- 不要手工复制某个 `.so` 覆盖另一个 build 目录，除非你同时清楚其依赖库、CMake cache 和 Python binding 是否一致。
- 不要把 benchmark 的失败结论建立在手动环境未确认的 smoke 上；先用 benchmark CLI 或本页的 direct+LD 模板复现。
