# libuipc CUDA 后端混合精度——最终实施计划（V3.1）

本版为最小修补版：修正文档冲突，不改变既定路线。

## Context

**目标**：在 `cuda_mixed` 后端引入混合精度（Mixed-Precision）计算，提供四个编译期精度档位（`fp64 / path1 / path2 / path3`）。以全 FP64（`fp64`）为 Ground Truth 基准，按路径分层控制 ALU / Hessian 存储 / PCG 辅助向量精度，并配套 Telemetry 系统（Timer + PCG 统计 + ErrorTracker + NVTX）。

**重要约束**：
- 不修改 `src/backends/cuda` 的任何文件
- 不改 xmake / Python binding
- 首版不做运行时精度切换，仅编译期
- `Engine("cuda")` 与 `Engine("none")` 行为不受影响

---

## V1→V3 关键勘误（基于代码核实）

以下三个 V1 阻断性错误已经 **代码文件直接验证**：

| # | V1 错误 | 核实文件:行 | 正确做法 |
|---|---------|-----------|---------|
| 1 | 提出 `cuda_mixed::SimEngine` 继承 `cuda::SimEngine` | `cuda/sim_engine.h:29` — `class SimEngine final` | `cuda_mixed` 已有自己的独立 `SimEngine final`，直接在其内部改造，无需继承 |
| 2 | 示例代码用 `info.iter_count()` 读取迭代次数 | `cuda_mixed/linear_system/global_linear_system.h:236` — 只有 setter | 需新增 `SizeT iter_count() const { return m_iter_count; }` getter |
| 3 | 在 `do_init` 注入 `Timer::set_sync_func` | `cuda_mixed/engine/sim_engine.cu:51` — 构造函数已调用 | 已存在，不要重复设置 |
| 4 | Phase 0 验收要求"bit-for-bit/误差=0" | SpMV 的 `atomic_add` 路径不保证确定性 | 改为相对 L2 `≤ 1e-12` |
| 5 | Path2/Path3 接口改造被低估 | `iterative_solver.h:28` spmv 硬编码 `Float` | 需显式将 `spmv/global_linear_system` 接口类型化 |

---

## V3 用户审核修正（7 条，基于 2026-02-20 审核）

| 优先级 | # | 问题 | 修正 |
|--------|---|------|------|
| HIGH | 1 | v3.md 风险表"超阈值自动降级到 path2"违背"首版仅编译期切换"约束 | 改为"记录告警日志，建议用户改用 `path2` 重新运行（无运行时降级）" |
| HIGH | 2 | 验证方案引用 `apps/tests/sim_case/`，但该目录测试硬编码 `Engine("cuda")` | 需新建 `cuda_mixed` 专用测试入口（`Engine("cuda_mixed")`）或参数化方案 |
| HIGH | 3 | Step 5 接口类型化表不完整：缺 `global_linear_system.h:33-36`（公共类型别名）和 `iterative_solver.h:30-32`（`apply_preconditioner`/`accuracy_statisfied`）| 已在本计划 Step 5 表中补全 |
| MEDIUM | 4 | "5 个编译矩阵"只列了 4 个命令，缺 dual-backend | 已在构建矩阵中补充第 5 条命令 |
| MEDIUM | 5 | CMake 无非法 level 校验，拼写错误静默 fallback 到 FP64 | 已在 CMake 段添加 `FATAL_ERROR` 校验 |
| MEDIUM | 6 | NVTX 代码示例用原始 `nvtxRangePushA/Pop`，与"复用 `muda::RangeName`"文字矛盾 | mixed_plan_v3.md §5.5 改用 `muda::RangeName` 包装宏 |
| MEDIUM | 7 | Path-3 代码注释 `alpha:float` 与标量全程 double 策略矛盾；代码含 `alpha_f`（float 变量名）| 改为 `alpha:double`，代码改为 `x(i) += alpha * static_cast<double>(p(i))` |

---

## 冻结决策（来自 V2 审核）

1. **切换方式**：编译期，`UIPC_CUDA_MIXED_PRECISION_LEVEL=fp64|path1|path2|path3`
2. **Telemetry**：首版全量实现，默认全部关闭（通过运行时 JSON config 开启）
3. **Path2/3 实现方式**：在 `cuda_mixed` 现有类内做类型化改造，不新建并行系统树
4. **验收口径**：分级容差阈值（`fp64≤1e-12 / path1≤1e-6 / path2≤1e-4 / path3≤1e-3`）
5. **`x`（解向量）**：永远保持 `double`，不受精度档位影响
6. **路径定义修订（2026-02-23）**：`path2/path3` 的 ALU 域回到 `double`；`path1` 才使用 `float` ALU。`path2/3` 重点改造存储域与 PCG 辅助向量域。

---

## 精度组件地图（27 项，扩展版）

> 说明：原版 19 项用于路线抽象，本版扩展到 27 项用于实施追踪。  
> 状态定义：`✅` 已落地，`⚠️` 半落地/存在桥接缺口，`❌` 未落地。

| # | 组件 | 状态 | 关键文件（cuda_mixed 内） | 精度档位 |
|---|------|------|------------------------|---------|
| 1 | Contact normal ALU-gradient | ⚠️ | `contact_system/contact_models/ipc_simplex_normal_contact.cu` | Path-1 |
| 2 | Contact normal ALU-hessian | ⚠️ | `contact_system/contact_models/ipc_simplex_normal_contact.cu` | Path-1 |
| 3 | Contact friction/half-plane ALU | ⚠️ | `contact_system/contact_models/ipc_simplex_frictional_contact.cu` | Path-1 |
| 4 | FEM SNH `F` ALU | ⚠️ | `finite_element/constitutions/stable_neo_hookean_3d.cu` | Path-1 |
| 5 | FEM SNH `G` ALU | ⚠️ | `finite_element/constitutions/stable_neo_hookean_3d.cu` | Path-1 |
| 6 | FEM SNH `H` ALU | ⚠️ | `finite_element/constitutions/stable_neo_hookean_3d.cu` | Path-1 |
| 7 | ABD OrthoPotential ALU | ✅ | `affine_body/constitutions/ortho_potential.cu` | Path-1 |
| 8 | ABD ARAP ALU | ✅ | `affine_body/constitutions/arap.cu` | Path-1 |
| 9 | ABD RevoluteJoint ALU | ✅ | `affine_body/constitutions/affine_body_revolute_joint.cu` | Path-1 |
| 10 | ABD PrismaticJoint ALU | ✅ | `affine_body/constitutions/affine_body_prismatic_joint.cu` | Path-1 |
| 11 | ABD RevoluteJointLimit ALU | ✅ | `affine_body/constitutions/affine_body_revolute_joint_limit.cu` | Path-1 |
| 12 | ABD PrismaticJointLimit ALU | ✅ | `affine_body/constitutions/affine_body_prismatic_joint_limit.cu` | Path-1 |
| 13 | ABD BDF1 kinetic ALU | ✅ | `affine_body/bdf/affine_body_bdf1_kinetic.cu` | Path-1 |
| 14 | ABD SoftTransformConstraint ALU | ✅ | `affine_body/constraints/soft_transform_constraint.cu` | Path-1 |
| 15 | ABD ExternalArticulationConstraint ALU | ✅ | `affine_body/constraints/external_articulation_constraint.cu` | Path-1 |
| 16 | ABDJacobi `JT_H_J` ALU 域 | ⚠️（工具层仍为 Float，主路径已绕开） | `affine_body/abd_jacobi_matrix.h`, `affine_body/abd_jacobi_matrix.cu` | Path-1 |
| 17 | ABDJacobiStack mat-vec/to_mat ALU 域 | ⚠️（类型层待模板化） | `affine_body/details/abd_jacobi_matrix.inl` | Path-1 |
| 18 | ABD linear subsystem: kinetic+shape 聚合 | ✅ | `affine_body/abd_linear_subsystem.cu` | Path-1 |
| 19 | ABD linear subsystem: reporter 聚合 | ✅ | `affine_body/abd_linear_subsystem.cu` | Path-1 |
| 20 | ABD linear subsystem: dytopo 聚合 | ✅ | `affine_body/abd_linear_subsystem.cu` | Path-1 |
| 21 | ABD-FEM coupling 聚合 | ✅ | `coupling_system/abd_fem_linear_subsystem.cu` | Path-1 |
| 22 | Reporter/assembler 局部缓冲（ABD/FEM/inter） | ❌ | `affine_body/abd_linear_subsystem.h`, `finite_element/fem_linear_subsystem.h` | Path-2 |
| 23 | 全局 Triplet Hessian（`A_triplet`） | ❌ | `linear_system/global_linear_system.h` | Path-2 |
| 24 | 全局 BCOO Hessian（`A_bcoo`） | ❌ | `linear_system/global_linear_system.h` | Path-2 |
| 25 | 全局梯度向量 `b` | ❌ | `linear_system/global_linear_system.h` | Path-2 |
| 26 | PCG 辅助向量（`r/z/p/Ap`） | ❌ | `linear_system/linear_pcg.h` | Path-3 |
| 27 | `x` + 收敛标量 + SpMV 域契约 | ✅（契约冻结） | `linear_system/global_linear_system.h`, `linear_system/spmv.h` | 固定 double + Path-2/3 扩展 |

### 与原 19 项映射关系

1. 原 1-6 映射到新 1-6（Contact/FEM ALU）。
2. 原 7-9 拆分到新 18-22（聚合层与缓冲层解耦记录）。
3. 原 10-12 映射到新 23-25（全局 `A/b` 存储层）。
4. 原 13-16 映射到新 26（PCG 辅助向量层）。
5. 原 17-19 映射到新 27（解向量与收敛标量契约 + SpMV 扩展域）。

### ABD Path1 补齐结论（代码核实）

#### 已补齐的 7 个关键文件

1. `src/backends/cuda_mixed/affine_body/constitutions/affine_body_revolute_joint.cu`
2. `src/backends/cuda_mixed/affine_body/constitutions/affine_body_prismatic_joint.cu`
3. `src/backends/cuda_mixed/affine_body/constitutions/affine_body_revolute_joint_limit.cu`
4. `src/backends/cuda_mixed/affine_body/constitutions/affine_body_prismatic_joint_limit.cu`
5. `src/backends/cuda_mixed/affine_body/bdf/affine_body_bdf1_kinetic.cu`
6. `src/backends/cuda_mixed/affine_body/constraints/soft_transform_constraint.cu`
7. `src/backends/cuda_mixed/affine_body/constraints/external_articulation_constraint.cu`

#### 残留技术债（需继续追踪）

1. `ABDJacobi::JT_H_J` / `ABDJacobiStack` 工具层仍是 `Float` 域定义，尚未模板化。
2. `abd_linear_subsystem.cu` 中原 `JT_H_J(...).cast<Alu>()` 后 cast 路径已移除，不再是当前主路径瓶颈。

#### 对 benchmark 解释的影响

当前 Stage2 已能解释 Path1 在 ABD/FEM 主链路下的收益；  
但由于尚无 articulation 专项场景，joint/constraint 子路径的性能贡献仍需单独量化。

---

## 精度策略架构（模块一，V3 修订版）

### 新增目录结构

```
src/backends/cuda_mixed/
├── mixed_precision/          ← 新建
│   ├── build_level.h         ← 编译宏→enum 映射
│   ├── policy.h              ← PrecisionPolicy<Level> 类型中心
│   └── cast.h                ← safe_cast / downcast_gradient / downcast_hessian
└── telemetry/                ← 新建
    ├── nvtx_scope.h          ← NVTX 宏（UIPC_WITH_NVTX=ON 才激活）
    ├── error_tracker.h       ← rel_l2 + abs_linf + nan_inf_flag + JSONL Dump
    └── telemetry_config.h    ← 运行时 JSON config 结构体
```

### `build_level.h` 骨架

```cpp
// src/backends/cuda_mixed/mixed_precision/build_level.h
#pragma once
namespace uipc::backend::cuda_mixed {
enum class MixedPrecisionLevel { FP64, Path1, Path2, Path3 };

#if defined(UIPC_MIXED_LEVEL_PATH3)
  inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path3;
#elif defined(UIPC_MIXED_LEVEL_PATH2)
  inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path2;
#elif defined(UIPC_MIXED_LEVEL_PATH1)
  inline constexpr auto kBuildLevel = MixedPrecisionLevel::Path1;
#else
  inline constexpr auto kBuildLevel = MixedPrecisionLevel::FP64;  // 默认
#endif
}
```

### `policy.h` 骨架

```cpp
// src/backends/cuda_mixed/mixed_precision/policy.h
#pragma once
#include <mixed_precision/build_level.h>
namespace uipc::backend::cuda_mixed {
template <MixedPrecisionLevel L>
struct PrecisionPolicy {
    // ALU（仅 Path1 使用 float；Path2/Path3 回到 double）
    using AluScalar   = std::conditional_t<L == MixedPrecisionLevel::Path1, float, double>;
    using AluMat3x3   = Eigen::Matrix<AluScalar, 3, 3>;
    using AluVec12    = Eigen::Vector<AluScalar, 12>;
    using AluMat12x12 = Eigen::Matrix<AluScalar, 12, 12>;

    // 存储（Path2/Path3 改为 float）
    using StoreScalar = std::conditional_t<
        L == MixedPrecisionLevel::FP64 || L == MixedPrecisionLevel::Path1,
        double, float>;

    // PCG 辅助（Path3+开始改为 float）
    using PcgAuxScalar = std::conditional_t<L == MixedPrecisionLevel::Path3, float, double>;

    // 解向量永远 double
    using SolveScalar = double;
};
using ActivePolicy = PrecisionPolicy<kBuildLevel>;
}
```

### `cast.h` 骨架（插入点 B/C/E 使用）

```cpp
// src/backends/cuda_mixed/mixed_precision/cast.h
template <typename To, typename FromMatrix>
MUDA_INLINE MUDA_GENERIC auto downcast_gradient(const FromMatrix& G) noexcept {
    if constexpr (std::is_same_v<To, typename FromMatrix::Scalar>) return G;
    else return Eigen::Vector<To, FromMatrix::RowsAtCompileTime>{
        G.template cast<To>()};
}
template <typename To, typename FromMatrix>
MUDA_INLINE MUDA_GENERIC auto downcast_hessian(const FromMatrix& H) noexcept {
    if constexpr (std::is_same_v<To, typename FromMatrix::Scalar>) return H;
    else return Eigen::Matrix<To,
        FromMatrix::RowsAtCompileTime, FromMatrix::ColsAtCompileTime>{
        H.template cast<To>()};
}
```

**窄化安全约束（必须保留）：**
- `downcast_gradient/downcast_hessian` 在 `#ifndef NDEBUG` 下执行 `allFinite()` 断言，拒绝 NaN/Inf 输入。
- 对 `float` 窄化增加 range assert（基于 `std::numeric_limits<float>::max()`），防止越界静默溢出到 Inf。
- 目标是阻断 IPC Barrier 极值在插入点 B/E downcast 时悄悄污染系统状态。

---

## Telemetry 设计（模块二，V3 修订版）

### 关键修正
- `Timer::set_sync_func([] { muda::wait_device(); })` **已在 `engine/sim_engine.cu:51`（构造函数）中设置**，不要重复
- `SolvingInfo::iter_count()` 只有 setter，**需新增 getter** 供 PCG 统计使用

### 运行时配置键（通过 Scene config JSON 控制）

| Key | 默认 | 说明 |
|-----|------|------|
| `extras/telemetry/enable` | `0` | 总开关 |
| `extras/telemetry/timer/enable` | `0` | Timer 计时块 |
| `extras/telemetry/nvtx/enable` | `0` | NVTX 标记（需 `UIPC_WITH_NVTX=ON`）|
| `extras/telemetry/pcg/enable` | `0` | PCG 收敛统计 |
| `extras/telemetry/pcg/sample_every_iter` | `10` | 采样节流 |
| `extras/telemetry/error_tracker/enable` | `0` | 误差追踪 |
| `extras/telemetry/error_tracker/mode` | `"offline"` | offline/online |
| `extras/telemetry/error_tracker/reference_dir` | `""` | FP64 基准 dump 目录 |

### PCG 统计采样（不新建派生类，直接在 `LinearPCG` 中内嵌）

```cpp
// 在 LinearPCG 内部新增，不继承 MixedLinearPCG
struct PcgSample { SizeT iter; double norm_r, rz_ratio, alpha, beta; };
std::vector<PcgSample> m_pcg_history;  // 受 sample_every_iter 节流

// do_solve() 末尾：无侵入记录
m_iter_count_history.push_back(info.iter_count());  // 需先添加 getter
```

### ErrorTracker 误差计算（L2 + L∞）

```cpp
// telemetry/error_tracker.h
template <typename ScalarA, typename ScalarB>
struct ErrorMetrics {
    double rel_l2;
    double abs_linf;
    bool   nan_inf_flag;
};
ErrorMetrics compute_error_metrics(
    muda::CDenseVectorView<ScalarA> v_test,
    muda::CDenseVectorView<ScalarB> v_ref,
    double eps = 1e-15);
// rel_l2: sqrt(sum_sq_diff)/max(sqrt(sum_sq_ref), eps)
// abs_linf: max_i(abs(v_test(i)-v_ref(i)))
// nan_inf_flag: rel_l2/abs_linf 是否为有限数
```

说明：`L∞` 用于捕捉单点大误差（例如局部穿透）而不被整体 `L2` 均值掩盖。

---

## 分步实施 Roadmap（模块三，V3 修订版）

### Milestone 1：`fp64 + path1 + 全量 Telemetry`

#### Step 1：构建层接线（CMake）

修改文件：
- `CMakeLists.txt` — 新增 `UIPC_CUDA_MIXED_PRECISION_LEVEL`（默认 `fp64`）和 `UIPC_WITH_NVTX`（默认 `OFF`）
- `src/backends/cuda_mixed/CMakeLists.txt` — 将 level 宏注入到 target compile definitions
- `external/CMakeLists.txt` — 将 `UIPC_WITH_NVTX` 转发到 muda

```cmake
# CMakeLists.txt 新增
set(UIPC_CUDA_MIXED_PRECISION_LEVEL "fp64" CACHE STRING "fp64|path1|path2|path3")
# Level 合法性校验（防止拼写错误静默 fallback 到 FP64）
if(NOT UIPC_CUDA_MIXED_PRECISION_LEVEL MATCHES "^(fp64|path1|path2|path3)$")
    message(FATAL_ERROR
        "UIPC_CUDA_MIXED_PRECISION_LEVEL='${UIPC_CUDA_MIXED_PRECISION_LEVEL}' 非法，"
        "合法值：fp64 path1 path2 path3")
endif()
option(UIPC_WITH_NVTX "Enable NVTX markers" OFF)

# cuda_mixed/CMakeLists.txt 新增
string(TOUPPER ${UIPC_CUDA_MIXED_PRECISION_LEVEL} _LEVEL_UPPER)
target_compile_definitions(cuda_mixed PRIVATE
    UIPC_MIXED_LEVEL_${_LEVEL_UPPER}=1)
```

验收：五个编译矩阵（fp64、path1、path2、path3、dual-backend）均通过

#### Step 2：精度策略头落地

新建文件（不修改任何现有代码）：
- `src/backends/cuda_mixed/mixed_precision/build_level.h`
- `src/backends/cuda_mixed/mixed_precision/policy.h`
- `src/backends/cuda_mixed/mixed_precision/cast.h`

验收：`fp64 / path1` 编译通过，默认 `fp64` 行为不变

#### Step 3：Path-1 核心 Kernel 改造（插入点 B/E）

**改造原则**：使用 `ActivePolicy::AluScalar` 替换局部变量类型，写入前用 `downcast_gradient/hessian<StoreScalar>` 截断。

修改文件（`cuda_mixed` 内，不动 `cuda`）：
1. `finite_element/constitutions/stable_neo_hookean_3d.cu` — NeoHookean ALU 改为 `AluScalar`
2. `contact_system/contact_models/ipc_simplex_normal_contact.cu` — 接触 Kernel
3. `contact_system/contact_models/ipc_simplex_frictional_contact.cu`
4. `contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu`
5. `contact_system/contact_models/ipc_vertex_half_plane_frictional_contact.cu`

关键代码模式（以 NeoHookean 为例）：
```cpp
// 改前：Matrix3x3 F = fem::F(...)  [double]
// 改后（path1）：
using Alu = ActivePolicy::AluScalar;
using Store = ActivePolicy::StoreScalar;
Eigen::Matrix<Alu,3,3> F = fem::F(
    x0.cast<Alu>(), x1.cast<Alu>(), x2.cast<Alu>(), x3.cast<Alu>(), Dm_inv);
// ... ALU 计算 ...
make_spd(H_alu);  // SPD 投影保持在 Alu 精度
// 写入前截断
auto G_store = downcast_gradient<Store>(G_alu);
auto H_store = downcast_hessian<Store>(H_alu);
DVA.segment<4>(I*4).write(tet, G_store);
TMA.half_block<4>(I*HalfHessianSize).write(tet, H_store);
```

验收标准（Path-1）：

| 指标 | 目标 |
|------|------|
| Reporter 组装耗时 | 降低 15-25% |
| 单步 Ap 相对 L2 误差 | `< 1e-5` vs fp64 |
| PCG 迭代次数变化 | `± 5%` |
| 最终位移误差 | `< 1e-6` vs fp64 |
| 穿透检测 | 无新增穿透 |

#### Step 3.5：Reporter / Manager / Consumer 最小覆盖清单（Path1）

| 类别 | 最小覆盖文件组 | 说明 |
|------|----------------|------|
| Reporter | `contact_system/contact_models/*`、`finite_element/constitutions/*`、`affine_body/constitutions/*`、`inter_primitive_effect_system/constitutions/*` | 覆盖接触、FEM、ABD、跨原语能量生产路径 |
| Manager | `utils/matrix_assembler.h`、`finite_element/fem_linear_subsystem.*`、`affine_body/abd_linear_subsystem.*`、`linear_system/global_linear_system.*` | 覆盖写入聚合、子系统缓冲、全局线性系统汇聚 |
| Consumer | `linear_system/spmv.*`、`linear_system/iterative_solver.*`、`linear_system/linear_pcg.*`、`linear_system/*preconditioner*` | 覆盖 SpMV、求解器基类、PCG 与预条件器消费链 |

#### Step 4：Telemetry 全量首版落地

**修改文件**（在现有类内嵌入，不新建派生类）：
- `linear_system/global_linear_system.h:236` — 新增 `SizeT iter_count() const { return m_iter_count; }`
- `linear_system/linear_pcg.h` — 新增 `PcgSample` 结构和 `m_pcg_history` 成员
- `linear_system/linear_pcg.cu` — 采样点和收敛统计

**新建文件**：
- `telemetry/nvtx_scope.h` — 复用 `muda::RangeName`，条件编译
- `telemetry/error_tracker.h` — `rel_l2 + abs_linf + nan_inf_flag` 误差计算（GPU kernel）+ offline/online 两种模式
- `telemetry/telemetry_config.h` — 运行时配置结构体

验收：
1. 默认关闭无额外开销
2. 开启后 `Timer::report_as_json()` 包含 SpMV、PCG 各分量耗时
3. `error_tracker` 能输出 JSONL 误差文件

---

### Milestone 1.5：`ABD Path1 Completion`（补全子阶段，已完成主体）

> 状态：Subtask B/C 已完成，Subtask A 剩余工具层模板化，Subtask D 待补 articulation 专项场景。

#### Subtask A：`ABDJacobi` 模板化（剩余）

修改文件：
1. `src/backends/cuda_mixed/affine_body/abd_jacobi_matrix.h`
2. `src/backends/cuda_mixed/affine_body/abd_jacobi_matrix.cu`
3. `src/backends/cuda_mixed/affine_body/details/abd_jacobi_matrix.inl`

目标：
1. `ABDJacobi::JT_H_J` 支持模板标量 `T`（不再硬编码 `Float`）。
2. `ABDJacobiStack` 的 `Vector/Matrix` 接口改为模板标量 `T`。
3. 说明：`abd_linear_subsystem.cu` 中 `JT_H_J(...).cast<Alu>()` 后 cast 路径已完成清理。

#### Subtask B：4 个 joint constitution 的 Path1 ALU 化（已完成）

修改文件：
1. `affine_body/constitutions/affine_body_revolute_joint.cu`
2. `affine_body/constitutions/affine_body_prismatic_joint.cu`
3. `affine_body/constitutions/affine_body_revolute_joint_limit.cu`
4. `affine_body/constitutions/affine_body_prismatic_joint_limit.cu`

目标：
1. 局部 ALU 全部切到 `ActivePolicy::AluScalar`。
2. 写回前统一 `downcast_gradient/hessian<StoreScalar>`。

#### Subtask C：ABD kinetic + constraints 的 Path1 ALU 化（已完成）

修改文件：
1. `affine_body/bdf/affine_body_bdf1_kinetic.cu`
2. `affine_body/constraints/soft_transform_constraint.cu`
3. `affine_body/constraints/external_articulation_constraint.cu`

目标：
1. kinetic/constraint 的 G/H 计算域切到 `AluScalar`。
2. 保持 `x` 与收敛标量域契约不变（double）。

#### Subtask D：新增 ABD articulation benchmark 场景（待执行）

目标：
1. 增加能覆盖 joint/constraint 路径的 Stage2 场景。
2. 将其纳入 Path1 vs fp64 的性能与误差验收，避免仅靠 `wrecking_ball` 评估。

---

### Milestone 2：`path2 + path3`（接口类型化改造）

#### Step 5：线性系统类型化基础（Path-2 前提）

这是 **V1 低估的核心工程任务**。以下接口仍硬编码 `Float`，必须先类型化：

| 文件 | 当前绑定 | 修改方案 |
|------|---------|---------|
| `linear_system/global_linear_system.h:33-36` | `TripletMatrixView`, `CBCOOMatrixView`, `DenseVectorView`, `CDenseVectorView` 四个公共类型别名均绑定 `<Float,3>` | 随 `StoreScalar` 参数化；外部子系统通过别名引用，无需感知底层精度 |
| `linear_system/iterative_solver.h:28` | `void spmv(Float a, CDenseVectorView<Float> x, ...)` | 标量 `a/b` 固定 `double`；向量参数改为 `CDenseVectorView<PcgAuxScalar>` / `DenseVectorView<PcgAuxScalar>`；`x` 保持 `double` |
| `linear_system/iterative_solver.h:30-32` | `apply_preconditioner(DenseVectorView<Float> z, CDenseVectorView<Float> r)` / `accuracy_statisfied(DenseVectorView<Float> r)` | 接口向量类型统一为 `PcgAuxScalar`（Path3=`float`，其余=`double`）；收敛标量统计保持 `double` |
| `linear_system/spmv.h:30-44` | 仅 `A<Float>, x<Float>, y<Float>` 主签名 | 新增两组重载：Path2 `A<float>, x<double>, y<double>`；Path3 `A<float>, x<float>, y<float>` |
| `linear_system/global_linear_system.h:307-312` | `DeviceTripletMatrix<Float,3>` / `DeviceBCOOMatrix<Float,3>` / `DeviceDenseVector<Float> b` | 用 `ActivePolicy::TripletMatrix3` / `ActivePolicy::BCOOMatrix3` / `ActivePolicy::GradientVec` 替换（`x` 保持 `DeviceDenseVector<double>`）|

修改文件清单（Step 5-7）：
1. `linear_system/global_linear_system.h` + `.cu`
2. `linear_system/iterative_solver.h` + `.cu`
3. `linear_system/spmv.h` + `.cu`
4. `linear_system/linear_pcg.h` + `.cu`
5. `linear_system/global_preconditioner.h` + `.cu`（接口复核；通常无需类型改动）
6. `linear_system/local_preconditioner.h` + `.cu`（接口复核；通常无需类型改动）
7. `affine_body/abd_diag_preconditioner.cu`、`finite_element/fem_diag_preconditioner.cu`（`PcgAuxScalar` 适配复核）
8. `utils/matrix_assembler.h` — `write()` 中加 `downcast_hessian<StoreT>` （插入点 C）
9. `algorithm/matrix_converter.h` + inl — `ge2sym/convert` 的 `float` 特化
10. `affine_body/abd_linear_subsystem.h`
11. `finite_element/fem_linear_subsystem.h`
12. `inter_primitive_effect_system/inter_primitive_constitution_manager.h` + `.cu`
13. `contact_system/*contact*.h/.cu`（reporter/holder 的 `Doublet/Triplet<Float>` 视图与缓存）
14. `coupling_system/*dytopo*_receiver.h`（若参与全局 Hessian 汇入）

> 2026-02-23 路径修订说明：Path2/Path3 的 ALU 域保持 `double`。  
> 因此 Step 5-6 的目标是“**存储域 + SpMV + 接口类型化**”，不是延续 Path1 的 ALU 降精。

#### Step 5.1：Path2 设计完善性补充（2026-02-23 审查）

**已具备（代码层已存在）**
1. `GlobalLinearSystem` 公共 view alias 已引入 `StoreScalar/PcgAuxScalar/SolveScalar`。
2. `IterativeSolver` 接口已切到 `double` 标量 + `PcgAuxScalar` 向量。
3. `Spmv` 已有 Path2/Path3 混合精度重载（`A<float>` + `x/y` 不同精度组合）。
4. `GlobalLinearSystem::Impl::spmv()` 已按 `store_is_fp32 / pcg_is_fp32` 做编译期 dispatch。
5. `ApplyPreconditionerInfo` 与 `ABD/FEM` diag preconditioner 已按 `PcgAuxScalar` 做输出适配。

**仍需补全（Path2 真正落地的主要缺口）**
1. Producer/Reporter 缓冲仍大量硬编码 `Float`（尤其 `contact_system/*contact*.h/.cu`、`inter_primitive_effect_system/*`、`abd/fem_linear_subsystem.h`）。
2. Path2 文件清单此前低估了 contact / inter / coupling 的 buffer owner 覆盖面。
3. 需要明确 Path2 验收口径为“存储/SpMV 优化路径”；不要求在所有场景上优于 Path1（因为 Path2/3 ALU 已回 `double`）。

#### Step 5.2：Path2 实施准备（可直接开工）

**实施顺序（建议冻结）**
1. `Producer/Reporter` 类型化：先改 `abd/fem/contact/inter_primitive` 的 `Doublet/Triplet<Float>` 缓冲与 view 接口为 `StoreScalar`。
2. `Assembler/Converter` 对齐：确认 `matrix_assembler.h` 的 `downcast_*<StoreT>` 路径覆盖全部写入点；补齐 `MatrixConverter<StoreScalar,3>` 的 `float` 路径回归。
3. `GlobalLinearSystem` 收口：确保 `triplet_A/bcoo_A/b/debug_A` 与 `converter` 全部使用 `StoreScalar`，`x` 保持 `SolveScalar=double`。
4. `SpMV + Dispatch` 验证：只保留一条 Path2 运行路径（`A<float>, x<double>, y<double>`）并用编译期分支命中。
5. `Preconditioner` 回归：检查 `ABD/FEM diag preconditioner` 的 `do_assemble/do_apply` 在 Path2 (`PcgAuxScalar=double`) 下无额外窄化。
6. `Benchmark + ErrorTracker`：先跑 `Stage2` 的 `fp64/path2`，再补 `path2 vs path1` 对比（性能解释用），误差仍以 `fp64` 为基准。

**Path2 实施前静态检查（建议每轮执行）**
1. `rg -n \"(Doublet|Triplet)VectorView<Float>|TripletMatrixView<Float>\" src/backends/cuda_mixed/affine_body src/backends/cuda_mixed/finite_element src/backends/cuda_mixed/contact_system src/backends/cuda_mixed/inter_primitive_effect_system src/backends/cuda_mixed/coupling_system`
2. `rg -n \"Device(TripletMatrix|BCOOMatrix|DenseVector)<Float\" src/backends/cuda_mixed/linear_system src/backends/cuda_mixed/* -g\"*.h\"`
3. `rg -n \"rbk_sym_spmv\\(double.*CBCOOMatrixView<float, 3>\" src/backends/cuda_mixed/linear_system/spmv.h src/backends/cuda_mixed/linear_system/spmv.cu`

#### Step 6：混合精度 SpMV（Path-2 核心，插入点 G）

> Path2 的主要收益来源是 `A/b` 存储与 SpMV 带宽，不包含 ALU 降精收益（ALU 域保持 `double`）。

```cpp
// linear_system/spmv.cu — 新增重载
void Spmv::rbk_sym_spmv(
    double a,
    muda::CBCOOMatrixView<float, 3>  A,   // float 矩阵（带宽减半）
    muda::CDenseVectorView<double>   x,   // double 搜索方向
    double b,
    muda::DenseVectorView<double>    y)   // double 输出
{
    // 核心：block.cast<double>() * vx_double → 双精度 MAC
    // 复用 rbk_sym 的 WarpReduce 框架，仅改矩阵读取精度
}
```

验收标准（Path-2）：

| 指标 | 目标 |
|------|------|
| SpMV 单次耗时 | 降低 30-50% |
| Hessian 显存占用 | 减少 50% |
| Ap 相对 L2 误差 | `< 1e-4` vs fp64 |
| 最终位移误差 | `< 1e-4` vs fp64 |
| PCG 迭代次数增加 | `< 20%` |

#### Step 7：PCG 辅助向量降精度（Path-3）

> Path3 在 Path2 基础上仅继续降低 PCG 辅助向量精度；ALU 域仍保持 `double`。

修改文件：
- `linear_system/linear_pcg.h` — `r, z, p, Ap` 类型改为 `ActivePolicy::PcgAuxScalar`
- `linear_system/linear_pcg.cu` — `update_xr` 中 `x` 更新改为 double 累加
- 相关 preconditioner 文件

关键：**解向量 `x` 的更新使用 double 精度累加；alpha/beta/rz 标量全程保持 double**：
```cpp
// x += alpha * p  其中 x:double, alpha:double（标量全程保持 double）, p:float（Path-3）
x(i) += alpha * static_cast<double>(p(i));  // alpha 已为 double，p 从 float 升精度
```

验收标准（Path-3）：

| 指标 | 目标 |
|------|------|
| PCG 总耗时 | 降低 40-60% |
| 全流程显存节省 | ~55% |
| 最终位移误差 | `< 1e-3`（场景相关） |
| 无 NaN/Inf | 通过现有 `check_rz_nan_inf()` 监控 |
| 回退机制 | 通过 `m_iter_history` 监控；记录告警日志，建议改用 `path2` 重新运行（不做运行时自动降级，符合"首版仅编译期切换"约束）|

---

## 文件改动汇总

### 新建文件（`cuda_mixed` 内，cuda 零修改）

```
src/backends/cuda_mixed/
├── mixed_precision/
│   ├── build_level.h          (M1 Step 2)
│   ├── policy.h               (M1 Step 2)
│   └── cast.h                 (M1 Step 2)
└── telemetry/
    ├── nvtx_scope.h           (M1 Step 4)
    ├── error_tracker.h        (M1 Step 4)
    └── telemetry_config.h     (M1 Step 4)
```

### 修改文件（`cuda_mixed` 内）

| 文件 | Milestone | 说明 |
|------|-----------|------|
| `CMakeLists.txt` | M1 Step 1 | 注入精度宏 |
| `engine/sim_engine.cu` | — | 不需要改（Timer 同步已存在）|
| `finite_element/constitutions/stable_neo_hookean_3d.cu` | M1 Step 3 | Path-1 ALU |
| `contact_system/contact_models/ipc_simplex_normal_contact.cu` | M1 Step 3 | Path-1 ALU |
| （其余 4 个 contact kernel） | M1 Step 3 | Path-1 ALU |
| `linear_system/global_linear_system.h` | M1 Step 4 / M2 Step 5 | 新增 getter；Path-2 类型化 |
| `linear_system/linear_pcg.h / .cu` | M1 Step 4 / M2 Step 7 | 统计嵌入；Path-3 |
| `linear_system/iterative_solver.h` | M2 Step 5 | spmv 接口类型化 |
| `linear_system/spmv.h / .cu` | M2 Step 6 | 混合精度 SpMV 重载 |
| `utils/matrix_assembler.h` | M2 Step 5 | 插入点 C downcast |
| `algorithm/matrix_converter.h` | M2 Step 5 | float 特化 |
| `affine_body/abd_linear_subsystem.h` | M2 Step 5 | StoreScalar 替换 |
| `finite_element/fem_linear_subsystem.h` | M2 Step 5 | StoreScalar 替换 |

---

## 验证方案

**构建矩阵（Step 1 验收）：**
```bash
# 矩阵 1-4：单独 cuda_mixed，每个 level 各跑一次
cmake --preset release \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=<fp64|path1|path2|path3> \
  -DUIPC_BUILD_TESTS=OFF
cmake --build --preset release -j8

# 矩阵 5：dual-backend（cuda + cuda_mixed 同时，验证互不污染）
cmake --preset release \
  -DUIPC_WITH_CUDA_BACKEND=ON \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=fp64 \
  -DUIPC_BUILD_TESTS=OFF
cmake --build --preset release -j8
```

**数值验收**（⚠️ 现有 legacy 测试存在 `Engine("cuda")` 硬编码，不能直接复用）：

需为 `cuda_mixed` 新建测试入口：
```cpp
// apps/tests/mixed_precision/<case_name>.cpp（新建）
// 使用 Engine("cuda_mixed")
Engine e{"cuda_mixed"};
```
或提供 CMake 参数化方案，允许同一 `.cpp` 以不同 backend 运行。

| 档位 | 基准 | 验收阈值（最终位移相对 L2） |
|------|------|--------------------------|
| `fp64` | `cuda` 后端 | `≤ 1e-12` |
| `path1` | `fp64` | `≤ 1e-6` |
| `path2` | `fp64` | `≤ 1e-4` |
| `path3` | `fp64` | `≤ 1e-3` |

**ErrorTracker 开启后输出验证（L2 + L∞）：**
```python
import json, numpy as np
records = [json.loads(l) for l in open("debug/telemetry/error.jsonl")]
assert all(r["rel_err_x"] < 1e-6 for r in records)  # path1 场景
assert all(r["abs_linf_x"] < 1e-4 for r in records)  # 单点误差约束
assert all(not r["nan_inf_flag"] for r in records)
```

## 基础类型定义

**文件：** `include/uipc/common/type_define.h`

| 别名 | 实际类型 | 说明 |
|------|----------|------|
| `Float` | `double` (64-bit) | 所有浮点计算的基础类型 |
| `IndexT` | `int32_t` | 索引类型 |
| `Matrix3x3` | `Eigen::Matrix<double, 3, 3>` | 3x3 Hessian 块 |
| `Matrix12x12` | `Eigen::Matrix<double, 12, 12>` | ABD 12x12 Hessian 块 |
| `Vector3` | `Eigen::Vector<double, 3>` | 3D 向量 |
| `Vector12` | `Eigen::Vector<double, 12>` | 12D 梯度向量 |

**说明（避免误导）：** 将公共 `using Float = double;` 改为 `float` 的全局开关方案不用于本计划。本计划采用 `cuda_mixed` 内编译期策略与局部类型化，不改公共 `Float` 定义。

---

## 维度一：Manager（全局内存大管家）的精度边界

### 1.1 GlobalLinearSystem — 顶层管理者

**文件：**
- `src/backends/cuda_mixed/linear_system/global_linear_system.h`
- `src/backends/cuda_mixed/linear_system/global_linear_system.cu`

**`GlobalLinearSystem::Impl` 中的核心 Buffer（h 文件约 307-312 行）：**

| 成员变量 | 类型 | 底层数据类型 | 说明 |
|----------|------|-------------|------|
| `x` | 目标契约：`muda::DeviceDenseVector<double>` | `double*` | **解向量**（PCG 输出，保持 double；若当前仍为 `Float` 别名，按 Path2/Path3 迁移） |
| `b` | `muda::DeviceDenseVector<Float>` | `double*` | **右侧向量**（梯度向量 -g） |
| `triplet_A` | `muda::DeviceTripletMatrix<Float, 3>` | `Matrix3x3*` (double×9) | 组装阶段 Hessian（COO 格式） |
| `bcoo_A` | `muda::DeviceBCOOMatrix<Float, 3>` | `Matrix3x3*` (double×9) | 求解阶段 Hessian（BCOO 格式） |

**Buffer 分配时机（`global_linear_system.cu` 约 269-292 行）：**
```cpp
void GlobalLinearSystem::Impl::_update_subsystem_extent() {
    if(x.capacity() < total_dof) {
        x.reserve(total_dof * reserve_ratio);  // 预留 1.1 倍
        b.reserve(total_dof * reserve_ratio);
    }
    if(triplet_A.triplet_capacity() < total_triplet) {
        triplet_A.reserve_triplets(total_triplet * reserve_ratio);
        bcoo_A.reserve_triplets(total_triplet * reserve_ratio);
    }
}
```

**格式转换管线（从 Triplet 到 BCOO）：**
```
Reporter 写入 → triplet_A (Float)
    → converter.ge2sym()     // 对称化，在 Float 精度下
    → converter.convert()    // Triplet→BCOO，在 Float 精度下
    → bcoo_A (Float)         // 传给 SpMV
```
**精度边界标注：** 如果要让 Hessian 以 `float` 存储但计算以 `double` 进行，`triplet_A` 和 `bcoo_A` 的模板参数 `<Float, 3>` 就是修改点。

### 1.2 FEM 线性子系统

**文件：** `src/backends/cuda_mixed/finite_element/fem_linear_subsystem.h`（约 137-140 行）

| 成员变量 | 类型 | 说明 |
|----------|------|------|
| `kinetic_gradients` | `muda::DeviceDoubletVector<Float, 3>` | 动能梯度（3D） |
| `reporter_gradients` | `muda::DeviceDoubletVector<Float, 3>` | Reporter 汇聚梯度（3D） |

### 1.3 ABD 线性子系统（刚体）

**文件：** `src/backends/cuda_mixed/affine_body/abd_linear_subsystem.h`（约 110-127 行）

| 成员变量 | 类型 | 元素大小 | 说明 |
|----------|------|----------|------|
| `reporter_hessians` | `muda::DeviceTripletMatrix<Float, 12, 12>` | 1152 bytes/块 | ABD Reporter Hessian |
| `reporter_gradients` | `muda::DeviceDoubletVector<Float, 12>` | 96 bytes/段 | ABD Reporter 梯度 |
| `body_id_to_shape_hessian` | `muda::DeviceBuffer<Matrix12x12>` | 1152 bytes/刚体 | 形状能量 Hessian |
| `body_id_to_shape_gradient` | `muda::DeviceBuffer<Vector12>` | 96 bytes/刚体 | 形状能量梯度 |
| `body_id_to_kinetic_hessian` | `muda::DeviceBuffer<Matrix12x12>` | 1152 bytes/刚体 | 动能 Hessian |
| `body_id_to_kinetic_gradient` | `muda::DeviceBuffer<Vector12>` | 96 bytes/刚体 | 动能梯度 |
| `diag_hessian` | `muda::DeviceBuffer<Matrix12x12>` | 1152 bytes/刚体 | 对角预处理 Hessian |

---

## 维度二：Reporter（数据生产者）的精度边界

### 2.1 接触障碍（Contact Barrier）

#### 障碍核函数（纯数学，模板化）

**文件：** `src/backends/cuda_mixed/contact_system/contact_models/sym/codim_ipc_contact.inl`

```cpp
// 当前是全模板函数，类型 T 由调用者决定（目前 T = Float = double）
template <typename T>
__host__ __device__ void KappaBarrier(T& R, const T& kappa, const T& D, const T& dHat, const T& xi);
// 第 6 行 — 能量
template <typename T>
__host__ __device__ void dKappaBarrierdD(...);    // 第 38 行 — 一阶导
template <typename T>
__host__ __device__ void ddKappaBarrierddD(...);  // 第 73 行 — 二阶导
```

**精度截断插入点 A（最细粒度）：** 这里是 ALU 计算层，可以用 `float` 替换 `T`，让物理上不需要高精度的障碍梯度以单精度计算。

#### PT/EE/PE/PP Simplex 接触 Kernel

**文件：** `src/backends/cuda_mixed/contact_system/contact_models/ipc_simplex_normal_contact.cu`

**数据流（以 Point-Triangle 为例）：**

```
输入：Vector3 P, T0, T1, T2 (Float=double, 全局显存读)
     Float kappa, d_hat, thickness (Float=double, 全局显存读)

↓ [局部 ALU 计算，寄存器中为 double]
  Vector12 G;            // 12维梯度，寄存器/局部
  Matrix12x12 H;         // 12×12 Hessian，寄存器/局部
  PT_barrier_gradient_hessian(G, H, ...)  // 第 354 行

↓ [精度截断插入点 B — 在 SPD 投影前/后]
  make_spd(H);           // SPD 投影（保持在 double 中）

↓ [写回全局显存]
  DoubletVectorAssembler::write(tet, G)   // 第 363-369 行
  TripletMatrixAssembler::write(tet, H)   // 第 373-379 行

输出：写入 reporter_gradients (Float=double, 全局显存)
      写入 reporter_hessians  (Float=double, 全局显存)
```

**精度截断插入点 B（推荐）：**
在 `make_spd(H)` 之后，调用 `DVA.write()` / `TMA.write()` 之前，可以做 downcast：
```cpp
// H = H.cast<float>().cast<double>(); // 截断 Hessian 精度
// G = G.cast<float>().cast<double>(); // 截断梯度精度
```

#### 写入接口层（最统一的截断位置）

**文件：** `src/backends/cuda_mixed/utils/matrix_assembler.h`

**梯度写入（第 122-146 行）：**
```cpp
// DoubletVectorAssembler::ProxyRange::write()
MUDA_GENERIC void write(const Eigen::Vector<IndexT, N>& indices,
                        const SegmentVector& value) {
    for(IndexT ii = 0; ii < N; ++ii) {
        ElementVector G = value.template segment<SegmentDim>(ii * SegmentDim);
        // ← 精度截断插入点 C（最通用）：可在此处对 G downcast
        m_assembler.m_doublet(offset++).write(indices(ii), G);
    }
}
```

**Hessian 写入（第 341-357 行）：**
```cpp
// TripletMatrixAssembler::ProxyRangeHalf::write()
MUDA_GENERIC void write(const Eigen::Vector<IndexT, N>& indices,
                        const BlockMatrix& value) {
    for(IndexT ii = 0; ii < N; ++ii)
        for(IndexT jj = ii; jj < N; ++jj) {
            ElementMatrix H = value.template block<BlockDim, BlockDim>(...);
            // ← 精度截断插入点 C（最通用）：可在此处对 H downcast
            m_assembler.m_triplet(offset++).write(indices(L), indices(R), H);
        }
}
```

### 2.2 弹性本构（Elasticity）

#### Stable NeoHookean 3D

**文件：** `src/backends/cuda_mixed/finite_element/constitutions/stable_neo_hookean_3d.cu`

**能量计算 Kernel（第 81-119 行）：**
```
输入：Matrix3x3 Dm_inv (参考形变梯度逆，double)
     Vector3 x0,x1,x2,x3 (当前位置，double)
     Float mu, lambda (材料参数，double)

↓ 局部 ALU 计算（double 寄存器）
  Matrix3x3 F = fem::F(x0,x1,x2,x3, Dm_inv)   // 形变梯度
  Float E;  SNH::E(E, mu, lambda, F)            // 能量标量

↓ 精度截断插入点 D（能量值）
  energies(I) = E;    // 第 118 行，写全局显存
```

**梯度与 Hessian 计算 Kernel（第 122-182 行）：**
```
↓ 局部 ALU 计算（double 寄存器）
  Matrix3x3 dEdF;
  SNH::dEdVecF(dEdF, mu, lambda, F)             // ∂E/∂F（3×3）
  Matrix9x12 dFdx = fem::dFdx(Dm_inv)           // ∂F/∂x（9×12）
  Vector12 G = dFdx.T * VecdEdF                  // 第 166 行，链式法则

  Matrix9x9 ddEddF;
  SNH::ddEddVecF(ddEddF, mu, lambda, F)          // ∂²E/∂F²（9×9）
  make_spd(ddEddF)                               // SPD 投影（double）
  Matrix12x12 H = dFdx.T * ddEddF * dFdx        // 第 178 行

↓ 精度截断插入点 E（梯度/Hessian 写入前）
  DVA.segment<4>(I*4).write(tet, G)             // 第 168 行
  TMA.half_block<4>(I*HalfHessianSize).write(tet, H) // 第 180 行
```

**能量内核文件：** `src/backends/cuda_mixed/finite_element/constitutions/detail/stable_neo_hookean_3d.inl`
- `SNH::E()` — 第 2 行（全模板，`T` 目前为 `double`）
- `SNH::dEdVecF()` — 第 13 行
- `SNH::ddEddVecF()` — 第 35 行

**精度截断插入点 F（最细粒度）：** `.inl` 模板函数可以用 `float` 特化，让 F 的计算以单精度进行。

---

## 维度三：PCG 求解器（数据消费者）的精度边界

### 3.1 求解器文件位置

| 文件 | 说明 |
|------|------|
| `src/backends/cuda_mixed/linear_system/linear_pcg.h` | PCG 类定义，辅助向量声明 |
| `src/backends/cuda_mixed/linear_system/linear_pcg.cu` | PCG 迭代实现 |
| `src/backends/cuda_mixed/linear_system/spmv.h` | SpMV 函数签名 |
| `src/backends/cuda_mixed/linear_system/spmv.cu` | SpMV 实现（Reduce-By-Key） |
| `src/backends/cuda_mixed/linear_system/iterative_solver.h/.cu` | 求解器基类 |

### 3.2 PCG 辅助向量（全部为 `Float = double`）

**定义于 `linear_pcg.h` 第 16-30 行：**

| 向量 | 类型 | 分配位置 | 数学含义 |
|------|------|----------|----------|
| `x` | 目标契约：`muda::DeviceDenseVector<double>` | 外部传入（GlobalLinearSystem::Impl::x） | **解向量**，最终输出（保持 double；若当前仍为 `Float` 别名，按 Path2/Path3 迁移） |
| `b` | `muda::CDenseVectorView<Float>` | 外部传入（GlobalLinearSystem::Impl::b） | 右侧（-梯度） |
| `r` | `muda::DeviceDenseVector<Float>` | `do_solve()`，约第 49 行 | **残差向量** r = b - Ax |
| `z` | `muda::DeviceDenseVector<Float>` | `do_solve()`，约第 49 行 | **预处理残差** z = P⁻¹r |
| `p` | `muda::DeviceDenseVector<Float>` | `do_solve()`，约第 50 行 | **搜索方向** |
| `Ap` | `muda::DeviceDenseVector<Float>` | `do_solve()`，约第 52 行 | **SpMV 结果** Ap |

### 3.3 SpMV 核心函数

**文件：** `src/backends/cuda_mixed/linear_system/spmv.h`（第 14-42 行）

```cpp
// 当前签名（全部为 double 精度）：
void rbk_sym_spmv(Float                           a,
                  muda::CBCOOMatrixView<Float, 3> A,   // ← double 矩阵
                  muda::CDenseVectorView<Float>   x,   // ← double 向量
                  Float                           b,
                  muda::DenseVectorView<Float>    y);  // ← double 输出
```

**调用链：** `LinearPCG::spmv()` → `IterativeSolver::spmv()` → `GlobalLinearSystem::Impl::spmv()` → `Spmver.rbk_sym_spmv()`

**实际调用（`global_linear_system.cu` 约 470-480 行）：**
```cpp
spmver.rbk_sym_spmv(a, bcoo_A.cview(), x, b, y);
```

**精度截断插入点 G：** SpMV 是混合精度的核心改造目标。可以：
- 将 `A` 的模板参数从 `<Float, 3>` 改为 `<float, 3>`（单精度矩阵）
- 保持 `x` 和 `y` 为 `double`（输入/输出为双精度）
- 这就是标准的"FP32 SpMV + FP64 accumulation"混合精度方案

### 3.4 向量更新（AXPY 操作）

**文件：** `src/backends/cuda_mixed/linear_system/linear_pcg.cu`

**融合 x 和 r 更新（第 150-152 行）：**
```cpp
// update_xr() — CUDA 并行 for
x(i) += alpha * p(i);   // x = x + α·p
r(i) -= alpha * Ap(i);  // r = r - α·Ap
```
**类型：** alpha (`Float`), x/p/r/Ap (`Float*`)，当前全为 `double`。

**搜索方向更新（第 164 行）：**
```cpp
p(i) = z(i) + beta * p(i);  // p = z + β·p
```
**类型：** beta (`Float`), p/z (`Float*`), 当前全为 `double`。

**点积/归约（第 193, 210, 223 行）：**
```cpp
rz    = ctx().dot(r.cview(), z.cview());    // r^T·z
alpha = rz / ctx().dot(p.cview(), Ap.cview()); // p^T·Ap
```
**类型：** 结果为 `Float` (`double`)。Dot 使用 muda LinearSystemContext 内置实现。

### 3.5 收敛参数

| 参数 | 值 | 文件位置 |
|------|-----|---------|
| `global_tol_rate` | `1e-4` | `linear_pcg.cu` 约第 33 行 |
| `max_iter_ratio` | `2` | `linear_pcg.cu` 约第 34 行（最大迭代 = 2×N） |

---

## 维度四：最小精度组件清单

| # | 组件名称 | 当前 C++ 类型 | 当前精度 | 数据量级 | 所在文件 | 可改为 FP32？ | 改造备注 |
|---|----------|-------------|---------|---------|----------|-------------|---------|
| 1 | **接触障碍核函数内部变量** | `Float` (local var) | double | 寄存器 | `sym/codim_ipc_contact.inl` | ✅ 高收益 | 改模板参数 `T=float`；精度损失小但加速显著 |
| 2 | **PT/EE 局部梯度 G（寄存器）** | `Vector12` (local) | double | 12 doubles/thread | `ipc_simplex_normal_contact.cu` | ✅ | 插入点 B，在 write() 前 downcast |
| 3 | **PT/EE 局部 Hessian H（寄存器）** | `Matrix12x12` (local) | double | 144 doubles/thread | `ipc_simplex_normal_contact.cu` | ✅ | SPD 投影后 downcast（插入点 B） |
| 4 | **SNH 形变梯度 F（寄存器）** | `Matrix3x3` (local) | double | 9 doubles/thread | `stable_neo_hookean_3d.cu` | ✅ | 改模板参数，最细粒度（插入点 F） |
| 5 | **SNH 弹性局部梯度 G（寄存器）** | `Vector12` (local) | double | 12 doubles/thread | `stable_neo_hookean_3d.cu` | ✅ | 插入点 E |
| 6 | **SNH 弹性局部 Hessian H（寄存器）** | `Matrix12x12` (local) | double | 144 doubles/thread | `stable_neo_hookean_3d.cu` | ✅ | make_spd 后 downcast（插入点 E） |
| 7 | **ABD Reporter Hessian 缓存** | `DeviceTripletMatrix<Float,12>` | double | N_bodies²×144 bytes | `abd_linear_subsystem.h` | ⚠️ 中等风险 | 改模板参数；转换层需新增 cast |
| 8 | **ABD Reporter 梯度缓存** | `DeviceDoubletVector<Float,12>` | double | N_bodies×96 bytes | `abd_linear_subsystem.h` | ⚠️ 中等风险 | 同上 |
| 9 | **FEM Reporter 梯度缓存** | `DeviceDoubletVector<Float,3>` | double | N_verts×24 bytes | `fem_linear_subsystem.h` | ⚠️ 中等风险 | 改模板参数 |
| 10 | **全局 Triplet Hessian (组装阶段)** | `DeviceTripletMatrix<Float,3>` | double | N_triplets×72 bytes | `global_linear_system.h` | ⚠️ 核心风险 | 改精度影响所有 Reporter 写入接口 |
| 11 | **全局 BCOO Hessian (求解阶段)** | `DeviceBCOOMatrix<Float,3>` | double | N_triplets×72 bytes | `global_linear_system.h` | ✅ **最高收益** | 此处改为 FP32 = SpMV 以 FP32 运行 |
| 12 | **全局梯度向量 b** | `DeviceDenseVector<Float>` | double | N_dof×8 bytes | `global_linear_system.h` | ✅ | 改为 float 降低带宽 |
| 13 | **PCG 残差向量 r** | `DeviceDenseVector<Float>` | double | N_dof×8 bytes | `linear_pcg.h` | ⚠️ | 精度影响收敛行为，需谨慎 |
| 14 | **PCG 预处理残差向量 z** | `DeviceDenseVector<Float>` | double | N_dof×8 bytes | `linear_pcg.h` | ⚠️ | 同上 |
| 15 | **PCG 搜索方向向量 p** | `DeviceDenseVector<Float>` | double | N_dof×8 bytes | `linear_pcg.h` | ⚠️ | 同上 |
| 16 | **PCG SpMV 结果向量 Ap** | `DeviceDenseVector<Float>` | double | N_dof×8 bytes | `linear_pcg.h` | ✅ | 中间计算，可降精度 |
| 17 | **PCG 解向量 x** | 目标契约：`DeviceDenseVector<double>` | double | N_dof×8 bytes | `global_linear_system.h` | ⚠️ 高风险 | 保持 double；若当前仍为 `Float` 别名，按 Path2/Path3 迁移 |
| 18 | **PCG 标量 rz, alpha, beta** | `Float` (scalar) | double | 3 scalars | `linear_pcg.cu` | ✅ | 标量，影响微小，可保持 double |
| 19 | **SpMV 矩阵 bcoo_A** | `CBCOOMatrixView<Float,3>` | double | = 组件 #11 | `spmv.h` | ✅ **关键改造点** | FP32 SpMV 的核心 |

---

## 关键数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│ Reporter 层（数据生产）                                           │
│                                                                 │
│  StableNeoHookean3D::do_compute_gradient_hessian()              │
│  IPCSimplexNormalContact::do_assemble()                         │
│                           ↓ ALU 计算（寄存器，当前 double）        │
│                   [插入点 B/E：downcast 发生处]                   │
│                           ↓                                     │
│           DoubletVectorAssembler::write()   [插入点 C]           │
│           TripletMatrixAssembler::write()   [插入点 C]           │
│                           ↓ 写全局显存                            │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Manager 层（全局 Buffer）                                         │
│                                                                 │
│  reporter_hessians: DeviceTripletMatrix<Float,12>               │
│  reporter_gradients: DeviceDoubletVector<Float,12>              │
│           ↓ 汇聚（atomic_add）                                   │
│  triplet_A: DeviceTripletMatrix<Float,3>   ← 全局 Hessian        │
│  b:         DeviceDenseVector<Float>       ← 全局梯度            │
│           ↓ 格式转换                                             │
│  bcoo_A: DeviceBCOOMatrix<Float,3>         ← 求解用 Hessian      │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PCG 求解器层（数据消费）                                          │
│                                                                 │
│  输入：bcoo_A (Float), b (Float)                                 │
│  辅助：r, z, p, Ap (DeviceDenseVector<Float>)                    │
│                                                                 │
│  每次迭代：                                                       │
│    Ap = A·p      (SpMV，插入点 G：FP32 矩阵 × FP64 向量)         │
│    α  = rz/(p·Ap) (标量除法，double)                             │
│    x += α·p      (AXPY，double)                                  │
│    r -= α·Ap     (AXPY，double)                                  │
│    z = P⁻¹·r    (预处理器，double)                               │
│    β  = rz_new/rz (标量除法，double)                             │
│    p = z + β·p   (AXPY，double)                                  │
│                                                                 │
│  输出：x (DeviceDenseVector<double>)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 推荐的混合精度改造路径（优先级排序）

**路径 1（最低风险，最易实现）：** Kernel 内部计算改 FP32
- 对象：组件 #1, #2, #3, #4, #5, #6（寄存器局部变量）
- 方法：`.inl` 模板函数用 `float` 特化，或在 kernel 中用 `__float2double()` 做边界转换
- 风险：低（仅影响局部计算精度，全局存储仍 double）

**路径 2（中等收益）：** 全局 Hessian 存储改 FP32
- 对象：组件 #10, #11, #19（triplet_A, bcoo_A）
- 方法：将 `DeviceTripletMatrix<Float,3>` 改为 `DeviceTripletMatrix<float,3>`
- ALU 域保持 FP64（路径修订：不继承 Path1 的 ALU 降精）
- SpMV 进入混合精度重载（典型 `A<float> * x<double> -> y<double>`）
- 风险：中等（影响所有写入接口的类型兼容性，需在 Assembler::write() 中插入 cast）

**路径 3（最大收益，最高风险）：** PCG 辅助向量部分改 FP32
- 对象：组件 #13-16（`r/z/p/Ap`）
- 方法：PCG 辅助向量以 FP32 存储/计算，`x += alpha * p` 时 `p` 升精到 double 累加
- ALU 域保持 FP64（路径修订）
- 风险：需要评估收敛稳定性

---

## 执行进度（2026-02-21）

### 本轮已完成

1. Stage1/Stage2 benchmark 双层套件已落地并可执行：
   - `uipc_benchmark_mixed_stage1`
   - `uipc_benchmark_mixed_stage2`
2. Stage2 heavy 场景已接入并跑通：
   - `wrecking_ball`（性能 + 质量）
   - `fem_ground_contact`（质量哨兵）
   - `fem_heavy_nocontact`（~57k tet）
   - `fem_heavy_ground_contact`（~57k tet）
3. 离线误差链路已打通：
   - `fp64` 先生成 reference
   - `path1` 使用 offline ErrorTracker 对比并输出 JSONL
4. Path1 扩展改造已完成主体补齐：
   - 已覆盖：`ortho_potential`、`arap`、`abd_linear_subsystem`、`abd_fem_linear_subsystem`
   - 新增覆盖：4 个 joint constitution、`affine_body_bdf1_kinetic`、`soft_transform_constraint`、`external_articulation_constraint`
   - 剩余项：`ABDJacobi` / `ABDJacobiStack` 工具层模板化与 articulation 专项 benchmark
5. benchmark 网格导出链路已打通并收敛为“每帧一个 OBJ”：
   - 场景配置接线 `extras/debug/dump_surface`
   - `dump_surface` 从 line-search 中间态移到帧末（单帧单文件）
   - 导出文件名统一为 `scene_surface{frame}.obj`
6. FEM heavy 场景实例变换已预烘焙到顶点，消除运行时 non-identity transform 警告
7. strict mode 相关默认键已补齐（避免 benchmark 配置键缺失报错）：
   - `extras/debug/dump_solution_x`
   - `extras/telemetry/enable`
   - `extras/telemetry/timer/enable`
   - `extras/telemetry/timer/report_every_frame`
   - `extras/telemetry/nvtx/enable`
   - `extras/telemetry/pcg/enable`
   - `extras/telemetry/pcg/sample_every_iter`
   - `extras/telemetry/error_tracker/enable`
   - `extras/telemetry/error_tracker/mode`
   - `extras/telemetry/error_tracker/reference_dir`

### Stage2 结果快照（`output/benchmarks/stage2_full_20260221_204850/summary_stage2.json`）

1. 告警状态：`warning_count=6`（当前仍为 warning-only，不阻塞）
2. 性能总览（Path1 vs fp64，`cuda_mixed`）：
   - 平均提速：`19.52%`
   - 最佳案例：`wrecking_ball + TelemetryOn`，`27.06%`
   - 最慢改进案例：`fem_heavy_ground_contact + TelemetryOff`，`9.57%`
3. 误差总览（overall）：
   - `rel_l2_max = 2.04305e+08`
   - `abs_linf_max = 2.03560e-03`
   - `nan_inf_count = 0`
   - `record_count = 476`

### 最新运行产物（本地可复现）

1. Stage1：
   - `output/benchmarks/mixed_stage1/summary.json`
   - `output/benchmarks/mixed_stage1/summary.md`
2. Stage2：
   - `output/benchmarks/stage2_full_20260221_204850/summary_stage2.json`
   - `output/benchmarks/stage2_full_20260221_204850/summary_stage2.md`
3. OBJ 导出示例：
   - `output/benchmarks/mixed_stage2/workspaces/stage2/cuda_mixed/wrecking_ball/Perf/TelemetryOff/fp64_perf/debug/cuda_mixed/engine/sim_engine.cu/scene_surface1.obj`
   - 单 case 验证中 `80` 帧对应 `80` 个 `scene_surface*.obj`

---

## 下一阶段准备：M1.6 Path1 质量收敛（保性能优先）

### 优先顺序（冻结）

1. 排查 `wrecking_ball` 的离线误差链路（reference 对齐、坐标系/归一化、采样点一致性）。
2. 增加 articulation 专项 benchmark 场景并纳入 Stage2 同口径验收。
3. 对 `ABDJacobi` / `ABDJacobiStack` 做模板化，去掉工具层 `Float` 绑定。
4. 继续推进 Contact/FEM 剩余 Path1 ALU 点位，目标是在维持当前提速下压低 `rel_l2/abs_linf`。

### 目标文件（本轮关注）

1. `src/backends/cuda_mixed/affine_body/abd_jacobi_matrix.h`
2. `src/backends/cuda_mixed/affine_body/abd_jacobi_matrix.cu`
3. `src/backends/cuda_mixed/affine_body/details/abd_jacobi_matrix.inl`
4. `src/backends/cuda_mixed/linear_system/global_linear_system.cu`（error tracker 输出/对齐）
5. `apps/benchmarks/mixed/mixed_stage2_benchmark.cpp`（articulation 场景接入）
6. `apps/benchmarks/mixed/mixed_scene_builders.cpp`（场景构造与 reference/compare 参数）

### 下一轮验收口径

1. 稳定性检查：`nan_inf_count = 0`
2. 性能检查：`wrecking_ball` 与 heavy FEM 维持 Path1 正收益
3. 误差检查：`rel_l2/abs_linf` 相对当前快照明显下降，warning 数量收敛

---

## 附录提示（非验收条款）

改造完成后，可通过以下方式验证：
1. 运行 `apps/tests/mixed_precision/`（或参数化后的 mixed 专用入口）对比 FP32/FP64 结果差异
2. 运行 `apps/benchmarks/mixed/` 对比性能与 telemetry 开销
3. 使用 CUDA 的 `cuBLAS` 混合精度 GEMM 接口验证 SpMV 数值一致性
4. 监控 PCG 收敛迭代次数（次数增多表明精度损失过大）
5. 对比关键量：接触距离、能量、位移误差（L2 + L∞）

