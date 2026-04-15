# Mixed UIPC Assets Benchmark

入口：

```bash
python apps/benchmarks/mixed/uipc_assets/cli.py <subcommand> ...
```

先准备 mixed benchmark 默认依赖的 build 目录：

```powershell
powershell -File scripts/setup_mixed_uipc_assets_builds.ps1
```

它会串行准备：

- `build/build_impl_fp64`
- `build/build_impl_path1`
- `build/build_impl_path2`
- `build/build_impl_path3`
- `build/build_impl_path4`
- `build/build_impl_path5`
- `build/build_impl_path6`

单个 level 内部使用 `cmake --build --parallel` 并行编译。默认配置为 `RelWithDebInfo`，并且每个 level 都会带上 `UIPC_BUILD_PYBIND=ON`，这样 `uipc_assets` 才能直接使用对应的 `python/src` 绑定目录。

`-Parallel` 默认不是直接吃满逻辑核数：脚本默认限制为 8 个 worker。需要更激进时再手工传 `-Parallel`。

常用子命令：

- `sync`: 预拉 `MuGdxy/uipc-assets` 的全部资产到本地 cache
- `list`: 列出可用资产与元信息
- `resolve`: 解析当前筛选条件后的最终资产集合
- `run`: 运行 `fp64` 与各 `path` 的 perf / quality / timer 采集
- `compare`: 对已有 run 重新计算质量指标
- `report`: 生成 Markdown / JSON / CSV 报告
- `export`: 导出逐帧 OBJ 序列

示例：

```bash
python apps/benchmarks/mixed/uipc_assets/cli.py sync

python apps/benchmarks/mixed/uipc_assets/cli.py list --scenario abd

python apps/benchmarks/mixed/uipc_assets/cli.py resolve \
  --tag source_rigid_ipc \
  --scenario_family rigid_ipc_mechanisms_gears

python apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/representative.json \
  --levels fp64 path1 path5 path6

python apps/benchmarks/mixed/uipc_assets/cli.py export \
  --run_root output/benchmarks/mixed/uipc_assets/<run_id> \
  --scenario coupling \
  --levels fp64 path6 \
  --frames 0,1,5,10
```

正式分类使用两层：

- `scenario`: `abd`, `fem`, `coupling`, `particle`
- `scenario_family`: 由 `scene.py` 顶部 `Source:` 归一化生成，例如 `libuipc_test`, `libuipc_example`, `ipc_tutorial`, `rigid_ipc_chain`, `rigid_ipc_friction_turntable`, `rigid_ipc_mechanisms_gears`, `rigid_ipc_unit_tests_rotation`

`scenario` 现在表示代码中的实际物理类型，而不是旧的主题桶：

- `abd`: `scene.py` 只含 `AffineBodyConstitution` / `AffineBodyShell`
- `fem`: `scene.py` 只含 `StableNeoHookean`
- `coupling`: `scene.py` 同时含多类 constitution
- `particle`: `scene.py` 使用 `Particle`

原来的 `rigid` 主题集合已经迁移到 source-based manifests / tags：

- `--tag source_rigid_ipc`
- `--tag source_ipc_tutorial`
- `--manifest apps/benchmarks/mixed/uipc_assets/manifests/representative.json`
- `--manifest apps/benchmarks/mixed/uipc_assets/manifests/rigid_ipc.json`
- `--manifest apps/benchmarks/mixed/uipc_assets/manifests/rigid_ipc_fracture.json`
- `--manifest apps/benchmarks/mixed/uipc_assets/manifests/rigid_ipc_gear.json`

需要按 `scene.py` 重新生成 manifests 时，使用：

```powershell
python scripts/rebuild_mixed_uipc_assets_manifests.py `
  --assets-root output/benchmarks/mixed/uipc_assets/hf_cache/datasets--MuGdxy--uipc-assets/snapshots/<sha>/assets
```

筛选规则：

- `--scene`
- `--tag`
- `--scenario`
- `--scenario_family`
- `--manifest`
- `--all`

这些选择器统一取交集。

manifest 里的运行相关字段现在分成两段：

- `frames_perf`: 真正进入 perf 报表的 profile 帧数
- `perf_warmup_frames`: 仅用于 perf 的预热帧数，默认 `0`

worker 会先推进 `perf_warmup_frames`，再统计 `frames_perf`。现有 `ms/frame`、`perf_by_asset.csv`、`perf_by_stage.csv` 只按 profile 段计算。`--visual_export --frames ...` 也继续使用 profile-local 帧号。

共享 Python 环境固定在：

```text
apps/benchmarks/mixed/uipc_assets/.venv
```

路径相关的 `pyuipc` 不安装进这个 `.venv`。每个 level 的 Python 绑定固定来自：

```text
build/build_impl_<level>/python/src
```

也就是说，只要使用 `scripts/setup_mixed_uipc_assets_builds.ps1` 的默认输出目录，`run` 和 `export` 都不需要额外传 `--build`。

CLI 会在 worker 子进程里按当前 level 自动注入对应的 `PYTHONPATH`，用户不需要手工切换。

默认输出在：

```text
output/benchmarks/mixed/uipc_assets/<run_id>/
```

报告会固定生成：

- `reports/summary.md`
- `reports/summary.json`
- `reports/chart_segments.json`
- `reports/charts/*.svg`
- `reports/perf_by_stage.csv`
- `reports/perf_by_asset.csv`
- `reports/perf_by_scenario.csv`
- `reports/quality.csv`
- `reports/visual_exports.csv`

`report` 会自动补一组阶段堆叠柱状图：

- `asset_<asset>.svg`: 对比同一 asset 在各 level 下的总 `Pipeline` 时间与阶段占比
- `scenario_<scenario>.svg`: 对比同一 scenario 下各 level 的平均总时间与阶段占比

这些图使用非重叠的 coarse pipeline phases，而不是直接堆叠 `perf_by_stage.csv` 里的 canonical stage。
