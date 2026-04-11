# Mixed UIPC Assets Benchmark

入口：

```bash
python apps/benchmarks/mixed/uipc_assets/cli.py <subcommand> ...
```

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

python apps/benchmarks/mixed/uipc_assets/cli.py list --scenario rigid

python apps/benchmarks/mixed/uipc_assets/cli.py resolve --scenario rigid --scenario_family gear

python apps/benchmarks/mixed/uipc_assets/cli.py run \
  --scenario rigid \
  --levels fp64 path1 path7 path8

python apps/benchmarks/mixed/uipc_assets/cli.py export \
  --run_root output/benchmarks/mixed/uipc_assets/<run_id> \
  --scenario coupling \
  --levels fp64 path8 \
  --frames 0,1,5,10
```

正式分类使用两层：

- `scenario`: `rigid`, `fem`, `abd`, `coupling`, `particle`, `tutorial`
- `scenario_family`: 例如 `contact_basic`, `stacking`, `fracture`, `gear`, `turntable`, `tunnel`, `pendulum`, `erleben`, `kinematic`, `coupling`, `fem_basic`, `abd_basic`, `particle`, `example`

筛选规则：

- `--scene`
- `--tag`
- `--scenario`
- `--scenario_family`
- `--manifest`
- `--all`

这些选择器统一取交集。

共享 Python 环境固定在：

```text
apps/benchmarks/mixed/uipc_assets/.venv
```

路径相关的 `pyuipc` 不安装进这个 `.venv`。每个 level 的 Python 绑定固定来自：

```text
build/build_impl_<level>/python/src
```

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
