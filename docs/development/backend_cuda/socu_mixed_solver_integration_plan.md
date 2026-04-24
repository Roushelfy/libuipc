# `cuda_mixed` 接入 `socu_native` 作为备选 linear solver 的方案

## 结论先行

如果目标是把 `socu_native` 以“优雅且高性能”的方式接进 `libuipc` 的 `cuda_mixed` 后端，**最好的做法不是把它硬塞进现有 `IterativeSolver` 边界，也不是在每次 Newton 迭代后把全局 `BCOO` 再转成 `socu` 格式**。

当前 `cuda_mixed` 的线性系统边界是：

- 子系统先分别装配局部 Hessian / gradient
- `GlobalLinearSystem` 汇总成全局 `TripletMatrix<StoreScalar,3>`
- 再转换成对称 `DeviceBCOOMatrix<StoreScalar,3>`
- 顶层 solver 通过 `IterativeSolver` 接口做 `spmv + preconditioner + convergence`

而 `socu_native` 的契约是：

- 输入就是**已经成形的 block tridiagonal** `diag/off_diag/rhs`
- 每个 block 的尺寸 `n` 固定
- `factor` / `solve` / `factor_and_solve` 都在这个结构化格式上完成

因此，**推荐路线**应当只分成两种：

1. **Direct solver（近似）**
   用 `StructuredChainProvider` 直接装配一个结构化 surrogate system `\hat H`，再由 `socu` 直接解 `\hat H p = -g`。如果某个 scene family 恰好满足严格链式结构，那么它会自然退化成“exact structured direct solve”；但对一般 mixed backend，全局真系统并不可能完整装进 `socu` 的链式格式，所以主语义应当明确是**近似 direct solver / direction solver**。

2. **`socu` as preconditioner**
   保留当前全局 sparse `A x = b` 与外层 PCG，只把 preconditioner 换成 `socu` 去反解一个结构化近似系统 `M z = r`。

这两条路线共享同一组基础设施：

1. 先在 `cuda_mixed` 里把 solver 抽象从 `IterativeSolver` 提升成更通用的 `LinearSolver`
2. 再引入一个新的 `StructuredChainProvider` 接口，让满足结构约束的 subsystem 直接装配结构化 `diag/off_diag/rhs`
3. 第一版只支持**严格受限的 structured scene family**，不要试图立刻覆盖通用 `FEM + ABD + contact + coupling` 全局系统

这条路的核心优点是：

- 不破坏当前 `fused_pcg` 主路径
- 不把近似 direct solver 假扮成 iterative solver
- 不在热路径做昂贵的 `BCOO -> socu` 二次重排
- 可以把 gate / fail-fast 语义做得很清楚

---

## 我从现有代码里确认到的事实

### `cuda_mixed` 当前 linear solve 的真实边界

以下文件基本把边界说死了：

- `src/backends/cuda_mixed/linear_system/global_linear_system.h`
- `src/backends/cuda_mixed/linear_system/global_linear_system.cu`
- `src/backends/cuda_mixed/linear_system/iterative_solver.h`
- `src/backends/cuda_mixed/linear_system/iterative_solver.cu`
- `src/backends/cuda_mixed/linear_system/linear_fused_pcg.cu`
- `src/backends/cuda_mixed/linear_system/linear_pcg.cu`

当前形态不是“solver 直接拥有结构化矩阵”，而是：

1. `DiagLinearSubsystem` / `OffDiagLinearSubsystem` 报 extent
2. `GlobalLinearSystem` 统一分配：
   - `triplet_A`
   - `bcoo_A`
   - `b`
   - `x`
3. 子系统向 `TripletMatrixView<StoreScalar,3>` 写入 3x3 block triplets
4. `GlobalLinearSystem::Impl::build_linear_system()` 内做：
   - `_assemble_linear_system()`
   - `converter.ge2sym(triplet_A)`
   - `converter.convert(triplet_A, bcoo_A)`
   - `_assemble_preconditioner()`
5. 顶层 `solve_linear_system()` 调用注册进来的 `IterativeSolver`

这意味着当前 solver API 的默认假设是：

- 输入是**全局稀疏矩阵**
- 算法是**iterative**
- preconditioner 是 solver 协议的一部分

### mixed backend 现在已有的 solver / preconditioner 协作方式

当前默认配置来自 `src/core/core/scene_default_config.cpp`：

- `linear_system/solver = "fused_pcg"`
- 备选还有 `"linear_pcg"`

`SimSystemCollection::build_systems()` 会捕获 `SimSystemException` 并移除无效 system；当前 `LinearFusedPCG` 已经在用 `"unused"` 这套 gate 语义。这个机制适合表达“某个 solver 没被当前配置选中”，因此可以用于注册：

- `socu_approx`

但用户显式选择 `socu_approx` 后的 gate fail 不能靠 `SimSystemException` 静默移除，后文会单独要求 selected-solver validation 和 fatal error。

同时，`cuda_mixed` 线性系统还有清晰的 preconditioner 分层：

- `LocalPreconditioner`
- `GlobalPreconditioner`
- `ABDDiagPreconditioner`
- `FEMDiagPreconditioner`
- `FEMMASPreconditioner`

这说明 `libuipc` 现有代码天然支持“局部结构化快路径”，只是当前没有 direct solver 的公共抽象。

### 当前全局矩阵不是 `socu` 自然适配的对象

从 mixed backend 当前 subsystem 结构看：

- `ABDLinearSubsystem` 内部主块是 `12x12`
- `FEMLinearSubsystem` 内部主块是 `3x3`
- `ABDFEMLinearSubsystem` 是显式 off-diagonal coupling subsystem

而且 `GlobalLinearSystem` 最终统一落在 `BCOO<StoreScalar,3>` 上。

这说明：

1. **全局系统不是现成的 block tridiagonal**
2. **block size 也不是天然对齐 `socu_native` 当前最强的 `n=32/64` 曲面**
3. `ABD + FEM + coupling + contact` 组合时，结构很容易超出带宽 1

所以，如果先把全局 `BCOO` 装好，再在热路径里扫描、排序、压缩成 `socu` 的 `diag/off_diag`，基本可以判定不是高性能方案。

### `socu_native` 这边当前暴露的契约

来自：

- `include/socu_native/common.h`
- `include/socu_native/solver.h`
- 本 repo 的 `CMakeLists.txt`

可以直接依赖的点：

- 现成 CMake target：`socu_native`
- 推荐 embedding API：
  - `query_solver_capability<T>(shape, op, plan_options)`
  - `create_solver_plan<T>(shape, plan_options)`
  - `describe_problem_layout(shape)`
  - `factor_inplace_async<T>(plan, diag, off_diag, launch_options)`
  - `solve_inplace_async<T>(plan, diag, off_diag, rhs, launch_options)`
  - `factor_and_solve_inplace_async<T>(plan, diag, off_diag, rhs, launch_options)`
- 执行期 stream 通过 `LaunchOptions.stream` 传入
- `SolverPlan` 是长期持有的 runtime / graph / backend 状态边界
- `PerfBackend::Auto / Native / CublasLt / MathDx`

但这里也有两个很重要的边界：

1. `socu_native` 期望的是**结构化 `diag/off_diag/rhs`**
2. 目前 API 的 `diag/off_diag/rhs` 是**同一模板类型 `T`**
3. `diag/off_diag/rhs` 都是 in-place 契约，`factor` 会改写 factor storage，`solve` 会改写 RHS 为解
4. `off_diag` 的真实分配长度应来自 `describe_problem_layout()`，不能只按 `(horizon - 1) * n * n` 猜

这和 `cuda_mixed` 里的

- `StoreScalar`
- `SolveScalar`
- `PcgAuxScalar`

三域分离，不是完全同构的。

### `socu_native` embedding API 的硬约束

`libuipc` 新接入代码应把 plan-based async API 当作唯一推荐契约。旧的同步 wrapper 可以继续作为兼容层存在，但不要作为 mixed backend 的主路径。

推荐生命周期：

1. scene build / solver build 时构造 `socu_native::ProblemShape`
2. 调用 `query_solver_capability<T>()`
3. capability supported 后调用 `describe_problem_layout()`
4. 按 layout 分配 `diag/off_diag/rhs` sidecar buffer
5. 调用 `create_solver_plan<T>()` 并长期持有
6. 每次 Newton solve 只填充 buffers，并调用 `*_inplace_async()`，stream 通过 `LaunchOptions` 传入
7. shape、dtype、backend policy、graph policy 或 block size 改变时销毁并重建 plan

也就是说，`SocuApproxSolver` 不应该自己维护 `SolverGraphCache` 指针，也不应该每次 solve 构造短生命周期 plan。`SolverPlan` 已经是 `socu_native` 对宿主暴露的唯一持久执行对象。

第一版 `socu_approx` 的 performance path plan policy 建议固定为：

- `SolverBackend::NativePerf`
- `PerfBackend::MathDx`
- `MathMode::Auto`
- `GraphMode::Off`

`n=12` correctness smoke 不走这条 performance policy；它使用 `NativeProof`、CPU reference 或 standalone dense reference。
如果 `query_solver_capability()` 对 `PerfBackend::MathDx` 返回 unsupported，显式 `socu_approx` 应 fail-fast，而不是自动降级到 `Native` 或 `Auto`。

这里还需要一层 `socu_approx` 自己的 MathDx runtime preflight。`query_solver_capability()` 主要回答“这个 shape / dtype / op / graph policy 在接口层是否支持”，不应被理解为“manifest、runtime artifacts 和当前 device arch 一定已经可用”。因此第一版 gate 必须在创建 `SolverPlan` 前额外确认：

- `socu_native` 编译时启用了 `SOCU_NATIVE_ENABLE_MATHDX`
- `socu_native` 暴露的默认 manifest path 存在
- manifest 中有当前 `dtype + n + rhs + op=factor_and_solve` 所需的 MathDx runtime bundle
- bundle 中的 `.lto` / `.fatbin.lto` / unified `cusolverdx` artifact 路径存在
- 当前 CUDA device arch 能被 manifest 中的 artifact 覆盖
- runtime cache 目录可写，或者已有 cubin cache 可复用

首次运行时 lazy link 生成 cubin 是允许的；preflight 的目标不是强制 cubin 预先存在，而是避免 gate report 先写 `supported=true`、第一次 solve 才因为 manifest 或 artifact 缺失而炸掉。

等真实 frame correctness 和 wall time 稳定后，再考虑让 `GraphMode` 或其他 `PerfBackend` 变成内部实验配置。

---

## 为什么“直接把 `socu` 当成新 solver 塞进去”不是最优解

### 1. `IterativeSolver` 语义不对

`IterativeSolver` 目前自带：

- `spmv()`
- `apply_preconditioner()`
- `accuracy_statisfied()`
- `ctx()`

这套接口明显是给 Krylov / PCG 类 solver 设计的。
如果把 direct solver 也继承自它，代码能写，但语义会变差，而且后续维护会很别扭。

### 2. `BCOO -> socu` 的热路径转换会吞掉收益

当前 `GlobalLinearSystem` 每次 Newton 迭代都会重新：

- 统计 extent
- 装配 triplets
- `ge2sym`
- `triplet -> bcoo`

如果在这个后面再加一次：

- 遍历 `BCOO`
- 识别链结构
- 重新打包 `diag/off_diag`

那么即使 `socu` 求解本体快，前后处理也很可能把收益吃掉。

### 3. mixed precision 契约不完全匹配

`cuda_mixed` 的精度政策里，很多路径是：

- `StoreScalar = float`
- `SolveScalar = double`

例如 `path2/3/4`。

而 `socu_native<T>` 当前要求 factor storage 和 rhs / solution 同类型。
如果不额外做桥接，最自然的首批支持面只能是：

- `fp64`
- `path1`
- `path5`
- `path6`

如果要支持 `path2/3/4`，要么：

- 临时 upcast `A` 到 double
- 要么先用 float 求解，再 cast 回 double

这两者都不该在第一版里悄悄发生。

### 4. 当前 mixed backend 的“高性能”主路径不是全局 direct solve，而是迭代 + preconditioner

特别是 `FEMMASPreconditioner` 已经是一个非常强的结构化局部求解器方向。
如果直接把 `socu` 当全局 direct solver 推进去，会和现有设计的优势面发生正面冲突，而不是复用它。

---

## 推荐架构

## 1. 先把 solver 抽象从 `IterativeSolver` 提升成 `LinearSolver`

### 新增抽象

建议新增：

- `src/backends/cuda_mixed/linear_system/linear_solver.h`
- `src/backends/cuda_mixed/linear_system/linear_solver.cu`

形态建议：

```cpp
class LinearSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo {};

    struct AssemblyRequirements
    {
        bool needs_dof_extent       = true;
        bool needs_gradient_b       = true;
        bool needs_full_sparse_A    = true;
        bool needs_structured_chain = false;
        bool needs_preconditioner   = true;
    };

    virtual AssemblyRequirements assembly_requirements() const = 0;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;

  private:
    friend class GlobalLinearSystem;
    GlobalLinearSystem* m_system = nullptr;
    virtual void do_build() final override;
    void solve(GlobalLinearSystem::SolvingInfo& info);
};
```

然后：

- `IterativeSolver : public LinearSolver`
- `LinearPCG` / `LinearFusedPCG` 继续继承 `IterativeSolver`
- 新的 `SocuApproxSolver` 直接继承 `LinearSolver`

### 为什么这一步值得做

因为这会把 “linear solver” 和 “iterative linear solver” 正式分开。
这不只是代码洁癖，而是能避免后面所有关于 `spmv/preconditioner/convergence` 的歧义。

这里需要把语义说得更准确一点：

- 对 `LinearPCG` / `LinearFusedPCG` 来说，`LinearSolver` 仍然是“解全局 sparse 线性系统”的边界。
- 对 `SocuApproxSolver` 来说，`LinearSolver` 更像“为当前 Newton 外层提供方向”的边界。它可能解的是：
  - 严格 structured scene 下的真实链式系统；
  - 或更常见地，解一个结构化 surrogate system `\hat H p = -g`。

也就是说，**`LinearSolver` 这一层仍然值得做**，只是文档不应再默认把它理解成“所有实现都在解同一个真系统”。它更准确地表达的是：

- “当前 outer solve 阶段的方向求解组件”

而 `IterativeSolver` 则是其中一个更窄的子类，专门覆盖：

- `spmv`
- `preconditioner`
- `convergence`

这套 Krylov / PCG 协议。

### 需要改的现有点

- `GlobalLinearSystem` 把 `SimSystemSlot<IterativeSolver>` 改成 `SimSystemSlot<LinearSolver>`
- `add_solver()` 的签名改成收 `LinearSolver*`
- `solve_linear_system()` 不需要知道 solver 是 iterative 还是 approximate direct
- `GlobalLinearSystem::solve()` 不能再无条件先构建完整 sparse system，而要按 solver 的 `assembly_requirements()` 选择 build path

这一步改动面不大，但会把后面的接入变得干净很多。

### P0：必须拆出 build mode，否则 `socu_approx` 省不掉 full sparse build

当前 `GlobalLinearSystem::solve()` 的实际行为是：

1. `_update_subsystem_extent()`
2. `_assemble_linear_system()`
3. `converter.ge2sym(triplet_A)`
4. `converter.convert(triplet_A, bcoo_A)`
5. `_assemble_preconditioner()`
6. `solve_linear_system()`

这对 `fused_pcg` 是正确的，但对 `socu_approx` 会吞掉主要收益。
如果 `socu_approx` 仍然先做 full triplet assembly 和 `triplet -> bcoo`，那它只是在 solve 阶段换了方向来源，前置 sparse build 成本仍然存在。

因此 `LinearSolver` 抽象必须同时带上 assembly requirement。建议第一版明确三种 build mode：

| solver | needs_gradient_b | needs_full_sparse_A | needs_structured_chain | needs_preconditioner |
|---|---:|---:|---:|---:|
| `LinearFusedPCG` | yes | yes | no | yes |
| `LinearPCG` | yes | yes | no | yes |
| `SocuApproxSolver` | yes | no | yes | no |
| future `SocuPreconditioner` under PCG | yes | yes | yes | yes |

对应到 `GlobalLinearSystem` 的 build flow：

1. 先解析当前被选中的 `LinearSolver`，并验证 exactly one solver。
2. 调用 `_update_subsystem_extent()`，因为无论哪种 solver 都需要 DoF extent。
3. 如果 `needs_gradient_b`，只装配 RHS / gradient buffer。
4. 如果 `needs_full_sparse_A`，才装配 triplets、执行 `ge2sym` 和 `triplet -> bcoo`。
5. 如果 `needs_preconditioner`，才走 `_assemble_preconditioner()`。
6. 如果 `needs_structured_chain`，调用 `StructuredChainProvider` / `StructuredAssemblySink` 装配 `diag/off_diag/rhs` sidecar。

为了做到第 3 步，现有 `_assemble_linear_system()` 需要被拆开。第一版可以先拆成：

- `_assemble_gradient_vector()`
- `_assemble_sparse_hessian_triplets()`
- `_assemble_structured_chain()`

如果某些 subsystem 还没有 gradient-only path，允许第一步先在 dry-run 中保留旧 assembly，但文档和 milestone 必须把它标为过渡状态。真正性能 benchmark 必须在 `socu_approx` 不构建 full sparse `A` 后才算有效。

### P0：build mode 还必须传播到 dytopo / contact 阶段

`GlobalLinearSystem` 内部跳过 full sparse `A` 还不够。当前 outer Newton loop 在调用 `m_global_linear_system->solve()` 之前，已经执行了 dynamic topology effect 的 gradient / Hessian 组装：

1. build collision pairs
2. compute dytopo/contact gradient and Hessian
3. solve global linear system
4. line search

因此 selected `LinearSolver` 的 assembly requirements 必须在 Newton iteration 更早的位置就被解析出来，并传给 dytopo/contact reporter：

- `fused_pcg` / `linear_pcg`：contact / dytopo 继续生成完整 gradient + Hessian，后续进入 sparse `A`
- `socu_approx`：contact / dytopo 不生成全局 sparse Hessian；只生成 gradient，并把 near-band Hessian contribution 写给 structured sink
- future `socu` preconditioner：contact / dytopo 仍生成完整 sparse Hessian，同时可额外生成 structured preconditioner contribution

现有 `GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo` 已经有 `gradient_only` 和 `component_flags` 的概念。第一版可以把 `LinearSolver::assembly_requirements()` 转成一个更高层的 Newton assembly mode：

```cpp
enum class NewtonAssemblyMode
{
    FullSparse,                         // fused_pcg / linear_pcg
    GradientOnly,                       // gradient-only diagnostics or existing gradient-only paths
    GradientStructuredHessian,          // socu_approx
    FullSparsePlusStructuredPreconditioner,
};
```

`GradientStructuredHessian` 不能复用现有 `gradient_only=true` 语义。现有 `gradient_only` 会把 hessian count 直接置 0；而 `socu_approx` 需要的是“全局 sparse Hessian count 为 0，但 near-band Hessian contribution 仍写入 structured sink”。因此需要新增一条明确接口：

```cpp
class StructuredDyTopoEffectSink
{
  public:
    void write_gradient(/* global dof ids, gradient */);
    void write_hessian_block(/* global dof ids i/j, 3x3 block, weight/norm metadata */);
    void mark_off_band_drop(/* global dof ids i/j, norm metadata */);
};
```

`StructuredDyTopoEffectSink` 不应成为第二套独立装配语义。它应该被实现成 `StructuredAssemblySink` 面向 contact / dytopo reporter 的 specialization / adapter：底层仍写同一个 structured contribution buffer 和同一个 quality stats，只是额外暴露 `mark_off_band_drop()` 与 norm metadata。FEM、ABD、coupling、contact、dytopo 的 off-band drop 都应进入同一份 quality report；不能只统计 contact 的 off-band ratio。

第一版推荐让 contact / dytopo reporter 在 `GradientStructuredHessian` 模式下直接写这个 adapter，而不是先写 triplet 再由 manager 转发。这样能避免又生成一遍 sparse hessian。manager 的职责是：

- 根据 selected solver 设置 mode
- 给 reporter 提供 `StructuredDyTopoEffectSink`
- 汇总 near/off-band contribution report
- 不分配 `collected_dytopo_effect_hessian` / `sorted_dytopo_effect_hessian`

然后在 `advance_ipc` 的 Newton loop 中，先向 `GlobalLinearSystem` 查询当前 selected solver 的 requirements，再用该 mode 调用 dytopo/contact manager。否则即使 `GlobalLinearSystem` 跳过 `triplet -> bcoo`，contact / dytopo Hessian 的前置构建成本仍会存在。

显式 gate 也应放在这里：

- 用户选择 `socu_approx`，但当前 subsystem 不支持 gradient-only 或 structured-chain assembly，应 fatal。
- 用户选择 `fused_pcg`，则继续完整 sparse build。
- 用户未显式选择 `socu_approx`，不得因为 `socu` gate 失败影响默认 PCG。

---

## 2. 引入 `StructuredChainProvider`，禁止从 `BCOO` 逆推结构

### 设计目标

不要让 `SocuApproxSolver` 读 `bcoo_A` 再“猜”它是不是链。
应该反过来，让**天然知道自己结构的 subsystem**直接提供：

- 链长度 `N`
- block size `n`
- `diag`
- `off_diag`
- `rhs`
- solution scatter / gather mapping

### 建议新增接口

建议新增：

- `src/backends/cuda_mixed/linear_system/structured_chain_provider.h`

接口建议包含：

```cpp
struct StructuredChainShape
{
    int horizon    = 0;
    int block_size = 0;  // 32 or 64 for performance path
    int nrhs       = 1;
    bool symmetric_positive_definite = false;
};

struct StructuredDofSlot
{
    int old_dof        = -1;
    int chain_dof      = -1;
    int block          = -1;
    int lane           = -1;
    bool is_padding    = false;
    bool scatter_write = true;
};

struct StructuredContributionStats
{
    size_t near_band_pair_count        = 0;
    size_t off_band_pair_count         = 0;
    size_t near_band_block_terms       = 0;
    size_t off_band_block_terms        = 0;
    double near_band_weighted_norm     = 0.0;
    double off_band_weighted_drop_norm = 0.0;
};

struct StructuredQualityReport
{
    double block_utilization = 0.0;
    double near_band_ratio   = 0.0;
    double off_band_ratio    = 0.0;
    int    max_block_distance = 0;
    StructuredContributionStats contact_stats;
};

class StructuredChainProvider
{
  public:
    virtual ~StructuredChainProvider() = default;
    virtual bool is_available() const = 0;
    virtual StructuredChainShape shape() const = 0;
    virtual span<const StructuredDofSlot> dof_slots() const = 0;
    virtual StructuredQualityReport quality_report() const = 0;
    virtual void assemble_chain(/* StructuredAssemblySink& sink */) = 0;
    virtual void scatter_solution(/* global x/displacement */) = 0;
};
```

这份接口草案必须表达清楚以下事实：

- `old_dof -> chain_dof` 和 `chain_dof -> block/lane` 是 build 阶段预计算结果，不允许在 assembly kernel 里搜索或哈希。
- `block_size` 第一版性能路径只允许 `32` 或 `64`。其他尺寸只能作为 correctness smoke，不能拿来声明性能收益。
- `nrhs` 第一版只需要 `1`，但接口应保留字段，方便以后对齐 `socu_native` 的 `{1,2,4}` 支持面。
- provider 必须声明自己是否覆盖完整 DoF；未覆盖 DoF 的行为不能靠 scatter 时静默忽略。
- 多 chain 的第一版策略应保守：每条 chain 独立 pack 成连续 block segment，不为了填满 block 跨 chain merge。
- padding slot 必须显式存在，不能靠 “lane >= valid_count” 这种隐含约定散落在 kernel 里。
- `quality_report()` 必须能说明为什么当前 ordering 可以或不可以进入 `socu_approx`。

### Padding 规则

`n=32` 下 FEM 最常见的 packing 会类似：

- 10 个 3D vertex = 30 DoF
- 2 个 padding lane

padding 的第一版规则必须写死：

- padding lane 的 `rhs = 0`
- padding lane 不参与 scatter
- padding lane 对真实 DoF 的 off-diagonal contribution 必须为 0
- padding lane 的 diagonal 建议写成 `1` 或一个与 damping 一致的正数，保证 block SPD
- padding lane 不计入 physical residual / descent check

`n=64` 也遵循同样规则。

### 与 `socu_native::ProblemBufferLayout` 的关系

`StructuredChainProvider` 输出的是逻辑 block chain。
真正 device buffer 分配必须以 `socu_native::describe_problem_layout(shape)` 为准：

- `diag_element_count` 决定 `diag` 分配
- `off_diag_element_count` 决定 `off_diag` 分配，包括 recursive scratch
- `rhs_element_count` 决定 `rhs` 分配
- `off_diag_levels` 用于调试和 report，不要求 provider 自己复刻 recursive layout

换句话说，provider 不应该直接暴露或手写 `socu` 的 recursive off-diag layout。provider 只负责逻辑链；adapter 负责按 `ProblemBufferLayout` pack。

### Logical block layout contract

adapter 写入 `socu_native` buffer 时必须固定以下契约，不能让不同 provider 自己猜：

- dense block 内部固定为 row-major：`A[row * n + col]`
- RHS 固定为 row-major `n x nrhs`：`rhs[row * nrhs + col]`
- `diag[i]` 是完整 `n x n` image，不是 lower-only 或 upper-only
- `off_diag[i]` 的逻辑含义必须固定为相邻 block 的某一侧，例如 `H(i, i+1)`；另一侧由 SPD 对称性隐含，不能在不同 provider 里有时写 upper、有时写 lower
- padding lane 的 diagonal / offdiag / rhs 写法由上面的 padding 规则统一决定
- provider 只提交 logical block contribution；最终累加、对称化、damping 和 pack 顺序由 structured adapter 统一处理

这条 contract 要配 synthetic test：构造一个已知 `diag/off_diag/rhs` 的小 block-tridiagonal 系统，pack 后用 `socu_native` 解，再和 CPU dense reference 比较。这个 test 是防止 row-major、offdiag 方向和 full-image contract 回归的第一道闸门。

### 关键原则

`StructuredChainProvider` 是**补充路径**，不是替代当前 triplet 装配路径。
也就是说：

- 默认路径仍然可以继续走 `triplet_A -> bcoo_A -> fused_pcg`
- provider 只在 `socu` gate 通过时使用

这样就不会破坏现有 mixed backend 的正确性面。

---

## 3. 第一版只支持“严格受限的 structured scene family”

这是整个接入里最关键的策略选择。

### 推荐的第一版支持面：`ABD` correctness smoke + `32/64` superblock performance path

原因：

1. `ABDLinearSubsystem` 天然工作在 `12x12` block 上，适合做最小 correctness smoke
2. `socu_native` 的高性能 native-perf 曲面主要应盯 `n=32/64`
3. `ABDFEMLinearSubsystem` 是单独的 off-diagonal subsystem，容易 gate 掉
4. 对“纯 articulated chain / pure ABD”类场景，结构判定相对直接

这里必须避免一个路线冲突：`n=12` 不能被当成第一版性能目标。
如果使用 `n=12`，它更像 proof / correctness path，用来验证：

- provider 语义正确
- gather / scatter 正确
- surrogate sign convention 正确
- `socu_native` API 生命周期正确

真正要追性能时，应把 ABD / FEM / contact graph 的 DoF pack 到 `n=32` 或 `n=64` dense superblock：

- Phase A：`ABD n=12` correctness smoke，不声明性能收益
- Phase B：`ABD/FEM` pack 到 `n=32/64`，才进入 performance benchmark

Phase A 不应使用 `SolverBackend::NativePerf` 作为性能路径。它可以使用 `NativeProof`、CPU reference 或 standalone dense reference 来验证 provider / gather / scatter / sign convention。
Phase B 才进入 `NativePerf + n=32/64` 的 steady-state benchmark。

对 ABD 来说，`n=32/64` packing 可以先用保守规则：

- 不跨 chain merge
- 只把连续 chain 区段放进同一个 superblock
- 空 lane 按 padding 规则填充
- utilization 进入 quality report

### 第一版 gate 建议

只有同时满足以下条件才允许 `socu` 直连：

- 只有一个 diag subsystem 提供 structured chain
- 没有有效的 `OffDiagLinearSubsystem`
- 没有把带宽打破到 1 以外的 contact / dytopo / reporter coupling
- performance path 的 block size 必须是 `32` 或 `64`
- `StoreScalar == SolveScalar`

### 为什么不建议先从 FEM 通吃开始

当前 `FEMLinearSubsystem` 的全局装配和 `FEMMASPreconditioner` 更像：

- 一般稀疏 3x3 block 系统
- 借助 partition 做局部 dense cluster solve

它不是天然的 block tridiagonal 问题。
如果第一版就从 FEM 通吃开始，十有八九会变成：

- 结构判定复杂
- 重排开销大
- 很快碰到 contact / partition / mixed precision 的边界

这不适合作为第一版落点。

### Phase 2 再考虑的支持面

等 `ABD` 这条路打通后，再看两类扩展：

1. **FEM chain / rod / 线性拓扑场景**
   需要单独的 provider 和稳定的 chain ordering

2. **`socu` 作为 preconditioner 的第二条路线**
   这条路对一般场景的适配面可能更大，但建议放在 Direct solver（近似）主线稳定之后再推进

### 如果将来要把 FEM 纳入 structured path，建议先引入 assembly sink 抽象

这里最重要的设计选择是：**不要把每个 `do_compute_gradient_hessian()` 都重写成直接写 `socu` buffer。**

更稳、更优雅的路线是：

1. 保持 constitution 的数学主体不变
2. 把末尾的 `Assembler::write(...)` 收敛成统一的 sink 接口
3. 第一阶段只引入零行为变化的 `TripletAssemblySink`
4. 第二阶段再在严格 gate 下引入 `StructuredAssemblySink`

这样可以把“接口抽象”和“结构化求解路径”拆成两步，先证明旧 triplet 路径不掉速，再讨论如何把特定 FEM scene 接到 `socu`。

#### 推荐新增的接口文件

建议新增：

- `src/backends/cuda_mixed/utils/assembly_sink.h`

该文件只负责定义装配 sink，不承载 scene gate、ordering、provider 选择或 solver dispatch。

#### `TripletAssemblySink` / `StructuredAssemblySink` 草案接口

```cpp
template <typename StoreT, int BlockDim>
struct TripletAssemblySink
{
    muda::DoubletVectorView<StoreT, BlockDim> gradients;
    muda::TripletMatrixView<StoreT, BlockDim> hessians;
    bool gradient_only = false;

    template <int StencilSize, typename GVec>
    MUDA_DEVICE __forceinline__
    void write_gradient(IndexT event_offset,
                        const Eigen::Vector<IndexT, StencilSize>& indices,
                        const GVec& G) const;

    template <int StencilSize, typename HMat>
    MUDA_DEVICE __forceinline__
    void write_hessian_half(IndexT event_offset,
                            const Eigen::Vector<IndexT, StencilSize>& indices,
                            const HMat& H) const;
};

template <typename StoreT, int BlockDim>
struct StructuredAssemblySink
{
    bool gradient_only = false;

    // 这些映射必须在 subsystem/build 阶段预计算，
    // 不能在 constitution kernel 内临时推断。
    muda::CBufferView<IndexT> element_to_chain_slot;
    muda::CBufferView<IndexT> element_to_block_ids;

    // 这里建议先写“逻辑链式 block buffer”，
    // 不要让 constitution 直接知道 socu 的内部 scratch layout。
    muda::BufferView<StoreT> diag_blocks;
    muda::BufferView<StoreT> offdiag_blocks;
    muda::BufferView<StoreT> rhs_blocks;

    template <int StencilSize, typename GVec>
    MUDA_DEVICE __forceinline__
    void write_gradient(IndexT event_offset,
                        const Eigen::Vector<IndexT, StencilSize>& indices,
                        const GVec& G) const;

    template <int StencilSize, typename HMat>
    MUDA_DEVICE __forceinline__
    void write_hessian_half(IndexT event_offset,
                            const Eigen::Vector<IndexT, StencilSize>& indices,
                            const HMat& H) const;
};
```

这里的关键点不是“长得像不像统一接口”，而是语义边界：

- `TripletAssemblySink` 内部仍然应该调用今天的 `DoubletVectorAssembler` / `TripletMatrixAssembler`
- `StructuredAssemblySink` 也只负责把局部贡献写到**逻辑上的 structured block buffer**
- 真正把这些 block buffer 映射到 `socu_native` 最终 `diag/off_diag/rhs` 或其 adapter-owned sidecar buffer 的动作，应继续留在 provider / adapter 层，而不是下沉到 constitution
- contact / dytopo 的 `StructuredDyTopoEffectSink` 是 `StructuredAssemblySink` 的 specialization，不是平行体系；所有 provider 的 off-band drop、weighted block contribution 和 dropped norm 都必须汇总到同一份 quality stats

#### `ComputeGradientHessianInfo` 的推荐改法

如果要让 FEM 逐步迁移到 sink 抽象，推荐先改这些位置：

- `src/backends/cuda_mixed/finite_element/finite_element_elastics.h`
- `src/backends/cuda_mixed/finite_element/fem_linear_subsystem.h`
- `src/backends/cuda_mixed/finite_element/fem_3d_constitution.h`
- `src/backends/cuda_mixed/finite_element/codim_2d_constitution.h`
- 后续如要扩展，再补 `codim_1d` / `codim_0d` 对应基类

改法建议是：

- `ComputeGradientHessianInfo` 不再直接暴露 `gradients()` / `hessians()`
- 改为暴露一个 `sink()` 访问器
- constitution 内的数学主体保持不变，只在最后几行装配时改为：

```cpp
auto sink = info.sink();
sink.template write_gradient<StencilSize>(I * StencilSize, indices, G_store);

if(!info.gradient_only())
    sink.template write_hessian_half<StencilSize>(I * HalfHessianSize,
                                                  indices,
                                                  H_store);
```

这样做的好处是：

- constitution 的能量/梯度/Hessian 推导不需要重写
- triplet 路径可以先做到“只换接口，不换行为”
- 以后若某一类 FEM scene 被 gate 成 structured family，也只需要替换 sink 类型，而不需要把每个 constitution 再改一遍

#### 推荐的分阶段落地顺序

**Phase 2A：先做 zero-cost `TripletAssemblySink`**

- 目标是把 writer 抽象出来，但默认代码生成尽量贴近今天的 triplet assembler
- 这一阶段不要引入 `StructuredAssemblySink`
- 也不要改 scene gate、solver 选择或 buffer ownership

**Phase 2B：验证旧路径零回退**

- 跑 mixed backend 现有 FEM benchmark
- 对照改造前后的 Newton 总时间、assemble 时间、linear solve 时间
- 如果出现可见 regression，先停下来检查 codegen / NCU，再决定是否继续

**Phase 2C：只在严格 gate 下加 `StructuredAssemblySink`**

- 只对已经满足 chain ordering、block size、bandwidth 限制的 scene family 开启
- 映射表必须由 subsystem/provider 预先准备好
- constitution 内不允许做结构发现、reordering 或 chain search

#### Performance guardrail：这些实现细节很容易把原版弄慢

下面这些是建议直接写进 implementation checklist 的硬 guardrail：

1. **禁止 device-side `virtual` / type erasure / `std::function`**
   sink 必须是编译期静态类型，不能把写路径做成运行时多态。

2. **sink 必须是薄包装，且尽量保持 POD / trivially copyable**
   它只应持有 view、pointer、flag 等轻量成员，不应持有堆对象、容器或隐藏状态。

3. **`write_gradient()` / `write_hessian_half()` 必须 `MUDA_DEVICE __forceinline__`**
   这部分是热路径，必须让编译器有机会把它摊平回接近今天的 assembler 调用。

4. **默认 triplet 路径禁止增加 staging buffer、额外 copy 或双写**
   `TripletAssemblySink` 不是新的中间层存储，它只是对现有 writer 的零行为变化包装。

5. **禁止在 constitution kernel 内写 `if(path_kind == ...)` 这种运行时分支**
   热路径里不要做“triplet vs structured”的 runtime switch；路径选择应在 launch/type 层完成。

6. **禁止把 global-id 到 chain-id 的搜索、哈希、排序、邻接遍历塞进 assembly kernel**
   `element -> chain slot`、`vertex -> block row` 这类映射必须在 subsystem/build 阶段预计算。

7. **禁止把 fixed-size stencil 写法退化成 fully dynamic generic loop**
   现有 FEM constitution 大量依赖固定 stencil size 和 block size，抽象时不能把它改成动态维度慢路径。

8. **legacy triplet 路径必须保持与当前相同的写序、triplet ordering 和 atomic 语义**
   否则即使数值仍对，也可能伤到写放大、coalescing 或后续 sparse conversion 的局部性。

9. **`StructuredAssemblySink` 不应直接暴露 `socu` 的内部 scratch / recursive layout**
   constitution 最多只写逻辑 structured blocks；solver-specific packing 继续留在 adapter 层。

10. **任何 sink 抽象引入后，都必须先在旧 FEM 场景上做 no-regression benchmark**
    先证明 triplet 路径没被抽象本身拖慢，再让 structured path 进入下一阶段。

这套 guardrail 的目的很简单：把“接口演进”限制成**零额外工作量的代码生成变化**，不要让它在还没带来 `socu` 收益之前，先把 mixed backend 现有 FEM 路径拖慢。

---

## 4. Direct solver（近似）路线的产品语义

上面两条主线里，“Direct solver（近似）”在第一版产品面只保留一个显式实验配置：

- `linear_system/solver = "socu_approx"`

它的语义固定为：

- **让 `socu` 充当方向求解器**
- 方向来自结构化 surrogate system `\hat H`
- `\hat H` 由 init-time ordering 后的块三对角结构装配而来
- contact 第一版只吸收 near-band contribution
- gate 不通过或方向无效时，`socu_approx` 直接 fail-fast
- 不新增 `socu_direct`
- 不新增 `socu_hybrid`
- 不做隐式 hybrid fallback

这意味着 `socu_approx` 不是“完整真系统 direct solver”。只有在极少数 exact-chain scene 上，它才会自然退化成真系统 direct solve；对一般 `cuda_mixed` 场景，它应被理解为**近似 direct solver / direction solver**。

### 为什么第一版不做 `socu_direct` / `socu_hybrid`

第一版不建议保留两个 product mode，原因是它们很容易把三个不同概念混在一起：

1. **真系统 direct solve**
   这要求完整 Hessian 结构能被无损装进 `socu` 链式格式。一般 `FEM + ABD + contact + coupling` 场景不满足。

2. **近似 direct solve**
   这正是 `socu_approx` 要做的事：直接解一个受控构造的 `\hat H p = -g`。

3. **algorithmic hybrid**
   如果需要 full sparse system 和 structured approximation 协同，最干净的定义是“外层 PCG 解完整系统，`socu` 作为 preconditioner”，而不是“direct solver 内部偷偷 fallback”。

所以第一版的配置面应保持简单：

- 默认仍是 `fused_pcg`
- 显式实验才启用 `socu_approx`
- `socu_approx` 失败就报错，让研发者看见 gate reason

这里要特别注意 `SimSystemException` 的语义。当前 `SimSystemCollection` 会捕获 `SimSystemException` 并把 system 标成 invalid；这适合表达“这个 solver 没被配置选中，所以 unused”。
但如果用户显式配置了：

- `linear_system/solver = "socu_approx"`

那么 `socu_approx` 的 gate fail 不能只抛 `SimSystemException` 后被静默移除。否则 `GlobalLinearSystem` 可能最后没有任何 solver，却只是跳过 solve。第一版必须补一个 selected-solver validation：

- 未选中的 solver 可以用 `SimSystemException("unused")`
- 被选中的 solver gate fail 必须 fatal
- `GlobalLinearSystem` build 后必须验证 exactly one selected `LinearSolver`
- 如果 selected solver 数量为 0 或大于 1，直接报配置错误

### 第一版关于 contact 的默认策略

contact 在这里不应被当成“再多写一点 mapping 就能塞进 chain”的问题，而应被视为：

- active set 每轮变化
- primitive 类型混合（PT / EE / PE / PP）
- 耦合范围天然非局部
- 运行时 contact pair 可能让原本漂亮的 ordering 变差

因此第一版建议把行为写死：

- `abs(block_i - block_j) <= 1` 的 contact 视为 near-band，可以进入 `\hat H`
- 其余 contact 视为 off-band，只进入 quality report，不进入 `\hat H`
- 不在 Newton iteration 内重排 permutation
- frame boundary 可选做重排实验，但必须单独统计 reorder cost
- 若 near-band ratio 太低，或方向检查失败，`socu_approx` 直接 fail-fast

这样产品语义最清楚，也不会把一个还没定义清楚的算法级 hybrid 混进用户接口。

### 与 preconditioner 路线的关系

在本计划里，“只分两种路线”的意思是：

1. **Direct solver（近似）**
   `socu_approx` 直接解 `\hat H p = -g`，主 solve 路径不再有 PCG。

2. **`socu` as preconditioner**
   外层仍由 PCG 解完整 sparse system，`socu` 只负责 `M z = r`。

这两条路线可以共享 ordering、structured assembly、contact classification 和 quality report，但求解语义必须分开。

### 默认值保持不变

仍然保留：

- `fused_pcg` 作为默认值

这样不会扰动当前 mixed backend 已验证的行为。

---

## 5. `socu` 运行时应该怎么嵌进 `libuipc`

## 5.1 `SolverPlan` ownership

`SocuApproxSolver` 应长期持有一个 `socu_native::SolverPlan*`，而不是长期持有旧的 `SolverGraphCache`，也不是每次 solve 临时调用同步 wrapper。

推荐成员：

- `socu_native::ProblemShape shape`
- `socu_native::SolverPlanOptions plan_options`
- `socu_native::SolverPlan* plan`
- `socu_native::ProblemBufferLayout buffer_layout`
- `StructuredChainShape chain_shape`
- `StructuredQualityReport latest_quality_report`

plan 重建条件：

- `horizon` 改变
- `block_size` 改变
- `nrhs` 改变
- `StoreScalar / SolveScalar` 改变
- `PerfBackend / GraphMode / MathMode` 改变
- device 改变
- ordering 从 `32` 切到 `64` 或反向切换

第一版建议 `GraphMode::Off`，所以不要在 `libuipc` 侧手写 graph cache 生命周期。后续若启用 graph，也仍应通过 `SolverPlanOptions.graph_mode` 表达，而不是让 mixed backend 直接管理 `SolverGraphCache`。

## 5.2 Buffer ownership 与 in-place 生命周期

建议 `SocuApproxSolver` 自己维护一套 sidecar device buffers：

- `diag`
- `off_diag`
- `rhs`
- `permutation / gather-scatter map`
- quality / report counters

不要直接复用 `triplet_A` / `bcoo_A` 的存储。

原因很简单：

- `socu` 需要的是 dense block chain layout
- `off_diag` 真实长度包含 recursive scratch，必须由 `describe_problem_layout()` 决定
- `diag/off_diag/rhs` 都会被 `socu_native` in-place 改写
- `GlobalLinearSystem` 需要继续保留通用 sparse layout 作为默认 `fused_pcg` 路径

所有权第一版固定为：

- `SocuApproxSolver` 拥有 `SolverPlan` 和 `diag/off_diag/rhs` device buffers
- `StructuredChainProvider` 只拥有 ordering metadata、DoF map、quality report 和 assembly entrypoint
- `GlobalLinearSystem` 只负责按 selected solver 的 requirements 调度 build，并把 current `b/x` view 暴露给 solver
- `StructuredAssemblySink` / adapter 在 `SocuApproxSolver` 的 buffer 上 pack，不把 buffer ownership 下放到 subsystem

这样 `do_solve(GlobalLinearSystem::SolvingInfo&)` 仍然只需要通过 `info.x()` 写回方向；structured sidecar 不需要塞进现有 `SolvingInfo`。如果后续希望让 `GlobalLinearSystem` 统一管理 sidecar，也必须作为第二版重构单独评估。

Direct route 与 preconditioner route 的生命周期不同，文档必须分开：

1. **`socu_approx` direct route**
   每个 Newton direction 重新 pack `diag/off_diag/rhs`，调用 `factor_and_solve_inplace_async()`，得到方向后这些 factorized buffers 可以丢弃或下次覆盖。它不需要在 PCG 内重复 apply。

2. **future `socu` preconditioner route**
   每个 Newton sparse system / preconditioner build 阶段 pack 并 `factor_inplace_async()` 一次。PCG 每次 apply 时只把 residual copy / gather 到 `rhs_tmp`，调用 `solve_inplace_async()`，再 scatter 回 `z`。不能每次 PCG apply 都重新 factor。

这个区别很重要：direct route 优化的是“快速产生方向”，preconditioner route 优化的是“多次快速 apply `M^{-1}`”。

## 5.3 Stream ownership

`socu_native` 的 async API 通过 `LaunchOptions.stream` 接收 stream。
在 `libuipc` 里应当遵循：

- **使用 mixed backend 当前活跃 stream**
- `libuipc` 的 `SocuApproxSolver` 不创建自己的私有 stream
- 不在 async solve 内部做 host-side stream sync
- 若 resolved backend 是 MathDx，允许 `socu_native` runtime 内部创建 stream / event，但必须通过 caller stream 建立正确依赖

这样能保证：

- 和现有 `muda`/backend 调度一致
- profile 更清楚
- 避免 host-side 隐式同步
- 保留 MathDx backend 的最高性能路径

## 5.4 `PerfBackend` 策略

libuipc 不应该理解 `MathDx` 内部 JIT / runtime 细节，但 `socu_approx` 的性能路径应显式要求 MathDx，而不是用 `Auto` 模糊决定。
建议 `libuipc` 第一版不暴露用户级 `socu/perf_backend` knob，而是在 `SocuApproxSolver` performance gate 内部固定：

- `SolverBackend::NativePerf`
- `PerfBackend::MathDx`
- `MathMode::Auto`
- `GraphMode::Off`

如果 capability query 返回 unsupported，`socu_approx` 应报告 `socu_mathdx_unsupported`，并把 `SolverCapability.reason` 放入 gate report。第一版 performance path 不自动降级到 `Native`；`PerfBackend::Native` 只用于 debug / proof / 手动对照，不作为 `socu_approx` 成功路径。

capability query 通过后，还必须跑 `SocuMathDxPreflight`：

```cpp
struct SocuMathDxPreflight
{
    bool ok = false;
    bool build_enabled = false;
    bool manifest_found = false;
    bool runtime_bundle_found = false;
    bool runtime_artifacts_found = false;
    bool device_arch_supported = false;
    bool runtime_cache_available = false;
    std::string manifest_path;
    std::string runtime_cache_path;
    std::string reason;
};
```

这一步不要求提前完成所有 lazy JIT / nvJitLink，也不要求 cubin cache 已存在；它只验证“第一次 solve 有合理机会成功 link”。如果 preflight 不通过，`socu_approx` 应报告 `socu_mathdx_runtime_artifact_unavailable`，并 fail-fast。

## 5.5 第二条路线：让 `socu` 作为 structured preconditioner

如果我们真的要让一个 direct solver 和一个 iterative solver **同时参与同一次求解**，最自然的形态是：

- 外层保留当前 Krylov / PCG
- 全局矩阵仍然按现有 mixed backend 路径装成 sparse `A`
- `SpMV(A, p)` 仍然照常计算
- 只把 preconditioner 从“现有局部/全局近似逆”替换成“`socu` 解一个结构化近似系统 `M z = r`”

也就是说，目标系统仍然是：

- `A x = b`

但 preconditioner 变成：

- `M^{-1} r`，其中 `M` 是由 `StructuredChainProvider` 直接装配出来的 structured approximation

### 这意味着什么

这意味着：

- **原有 `SpMV` 还是要算**
- `socu` 不是取代全局矩阵乘法
- `socu` 提供的是“更强、更接近真实系统的 preconditioner”
- 希望得到的收益是：**更少的 PCG 迭代数**，而不是“把 Krylov 外层整个删掉”

这类 preconditioner 路线的直觉可以写成：

1. 用现有 sparse 流程组出完整 `A`
2. 用 provider 另外组一个结构化 `M`
3. PCG 迭代里继续算 `Ap = A p`
4. 每次需要 `z = P^{-1} r` 时，用 `socu` 去解 `M z = r`

### 推荐的 `M` 语义

`M` 不应来自：

- 从 `BCOO` 逆推出 chain
- 对全局稀疏矩阵做临时重排

`M` 应该来自：

- `StructuredChainProvider` 原生装配的 structured view
- 或某个 diag subsystem 的本地 structured approximation

这样做的原因很简单：
preconditioner 想要的是**稳定、可重用、结构清晰**，而不是每轮从全局 sparse 图里猜结构。

### 第一版 preconditioner 的推荐落点

如果要做算法级 hybrid，我建议先选下面这条最稳的路线：

- 外层：`LinearFusedPCG`
- 结构化主干：`socu`
- 非结构化 contact / coupling / off-band 项：继续留在全局 `SpMV(A, p)` 里

这条路线的好处是：

- contact 不需要被硬塞进 structured solver
- `socu` 只负责它擅长的 block chain 近似逆
- mixed backend 现有 sparse 体系不用推倒重来

## 5.6 与 MAS preconditioner 的关系：推荐“分工组合”，不推荐“同一 slice 重叠”

当前 mixed backend 的 preconditioner 体系本来就分成：

- `GlobalPreconditioner`
- `LocalPreconditioner`

因此把 `socu` 和 MAS 结合起来时，最自然的方式不是“二选一”，而是**按负责的子空间分工**。

### 最推荐的组合方式

最稳的第一版组合是：

- `ABD` 或其他满足 structured gate 的 diag subsystem
  -> 用 `SocuLocalPreconditioner`

- `FEM` 非 structured / partitioned 区域
  -> 继续用 `FEMMASPreconditioner`

- 全局 contact / coupling / off-diagonal
  -> 继续由外层 `SpMV` 和 PCG 处理

这样做的好处是：

- `socu` 和 MAS 不争抢同一段 DoF
- 每个 preconditioner 只在自己最擅长的局部空间上工作
- 外层 Krylov 自动把它们粘成一个全局迭代过程

### 什么时候考虑 `SocuGlobalPreconditioner`

只有在下面这种场景里，才建议把 `socu` 做成 `GlobalPreconditioner`：

- 整个 complement 本身就是一个清晰、稳定的 structured chain
- 并且我们愿意让它覆盖几乎全部有效 DoF

这时 `socu` 更接近“全局 structured inverse”，MAS 往往应该关闭或降级成 fallback。

### 不推荐的组合方式

第一版不建议：

- 一个 `SocuGlobalPreconditioner` 先写完整 `z`
- 然后 `FEMMASPreconditioner` 再在同一段 FEM slice 上改写 `z`

原因不是做不到，而是语义太容易变脏：

- 当前框架里 global preconditioner 先执行，local preconditioner 后执行
- 如果两者覆盖同一段 DoF，local path 实际上会把 global 结果覆盖掉
- 这会让“这个 preconditioner 到底是什么”变得很难解释和调试

因此第一版应明确要求：

- **同一段 DoF 只由一个 preconditioner 负责**

### 能否借 MAS 的思路

可以，而且我认为这是很好的方向，但要借的是**思路**，不是强行把实现缝在一起。

MAS 给我们的启发主要有三点：

1. **用结构化局部近似替代全局精确逆**
   这和 `socu` 作为 preconditioner 的核心思想一致。

2. **允许“主路径 + fallback”并存**
   MAS 对 unpartitioned 顶点有 diagonal fallback；`socu` 也可以对不满足 structured gate 的部分明确不接管。

3. **可选地吸收一部分 coupling 信息，但必须受控**
   MAS 已经有 `contact_aware` 这种思路；`socu` 未来也可以考虑吸收一部分仍保持带宽 1 的耦合，但第一版不要把 active contact 直接塞进 chain。

所以更推荐的方向是：

- `socu` 学 MAS 的“结构化近似逆 + 明确 fallback”思路
- 而不是一上来把 MAS engine 和 `socu` runtime 做成层层嵌套的巨型 preconditioner

## 5.7 Direct solver（近似）主线：`socu_approx`

在这份计划里，`socu_approx` 是 **“Direct solver（近似）”这条主线的唯一第一版实现形态**。

原因很直接：

- 对一般 `cuda_mixed` 全局系统，我们做不到把完整真系统无损装进 `socu` 的链式格式；
- 因此只要不是极少数 exact-chain scene，所谓 “`socu` as direct solver” 实际上都应理解成：
  - **用 `socu` 直接解一个结构化 surrogate system**
  - **把它产出的结果当作当前 Newton step 的方向**

这条路线的核心不是：

- 精确求解完整 Newton 线性系统

而是：

- 用一个便宜得多、但结构化得多的 surrogate Hessian `\hat H`
- 直接求一个 quasi-Newton / surrogate-Newton 方向
- 再用现有真实能量 line search 做 globalization

也就是说，方向由：

- `\hat H p = -g`

给出，而不是由完整全局 Hessian `H` 的 sparse solve 给出。

因此，文档后面凡是提到 “Direct solver（近似）”，默认都可以等价理解为：

- `socu_approx`

而“exact structured direct solve”只应被视为这条主线在少数严格 gate scene 上的特例。

### 它与 preconditioner 路线的区别

这条路和 “`socu` 作为 structured preconditioner” 必须明确区分：

- **preconditioner 路线**
  外层仍是 `LinearFusedPCG`，因此每轮仍要算完整 `SpMV(A, p)`

- **`socu_approx` 路线**
  外层不再依赖 PCG，直接用 `socu` 解 surrogate system，因此主 solve 路径里不再有 Krylov 迭代和每轮 `SpMV`

如果这条路成立，它节省的不只是 preconditioner 误差，还可能直接省掉：

- 一部分 full sparse Hessian solve 成本
- BCOO convert + Krylov 迭代成本
- 每轮 `SpMV` 和 dot-product 同步

### 什么时候这条路有价值

当下面这些条件基本成立时，这条路就很值得单独评估：

1. `n=64` superblock 级别的 chain ordering 是稳定的
2. 大部分 `ABD` / `FEM` 贡献能落在 block diagonal 或相邻副对角
3. 只有一部分 contact / coupling 超出带宽 1
4. 我们愿意接受：
   - 单个方向不再是精确 Newton 方向
   - 但单次方向计算明显更快

换句话说，这条路优化的目标不再是：

- “每次线性 solve 尽量精确”

而是：

- “每个 Newton / frame 的总 wall time 最低”

### 推荐的第一版定义

建议把它定义为一个独立的 solver 模式，例如：

- `linear_system/solver = "socu_approx"`

其工作流可以明确写成：

1. 组真实梯度 `g`
2. `assembly_requirements()` 选择 gradient + structured-chain build mode，不构建完整 sparse `A`
3. 由 `StructuredChainProvider` 直接装配一个 `n=32` 或 `n=64` 的 surrogate Hessian `\hat H`
4. 对 `\hat H` 做 damping / diagonal shift，保证可分解
5. 用 `socu` 解 `\hat H p = -g`
6. 按带阈值的方向规则检查 finite、范数、残差和下降性
7. 用现有真实能量 line search 对 `p` 做 backtracking
8. 若 gate 不通过、方向失效或 line search 连续异常，则 `socu_approx` fail-fast，不做隐式 fallback

### surrogate Hessian `\hat H` 的来源

第一版应明确规定：

- `\hat H` 必须由 `StructuredChainProvider` 原生装配
- 不允许从 `BCOO` 逆推出 chain
- 不允许通过全局 sparse 矩阵临时重排来“猜”出 surrogate system

这样 `\hat H` 的语义才会稳定：

- 它是一个**有意识构造的 banded surrogate**
- 不是从全局系统里抢救出来的一块偶然子图

### 对 contact / coupling 的第一版 inclusion rule

第一版建议非常克制：

- 只吸收那些在 ordering 后自然落入当前 block 或相邻 block 的 contact / coupling
- 所有超出带宽 1 的 contact / coupling 直接从 `\hat H` 里丢掉

不要为了“尽量拟合完整 Hessian”而把 surrogate path 搞脏。
如果这条路后续有效，再单独评估是否扩大带宽或吸收更多近邻 coupling。

### 为什么它仍然可能工作

因为当前 mixed backend 的 globalization 已经是：

- 用真实能量做 line search
- 而不是假设“线性方向必须是精确 Newton 方向”

因此 `socu_approx` 的成功标准不是：

- `p` 必须等于全局 Newton 方向

而是：

- `p` 在绝大多数迭代里是下降方向
- 真实能量 line search 能稳定接受它
- 总 frame 时间比现有 Newton + sparse solve 更低

### 第一版 guardrail

建议把第一版 guardrail 直接写死：

1. 只在 `contact-light` 或“大部分 contact 可落入带宽 1” 的场景上启用
2. Level 1 只允许 single-provider：`ABD-only + near-band contact` 或 `FEM-only + near-band contact` 进入 surrogate Hessian
3. 对 `\hat H` 固定加 damping，保证 SPD / 可因式分解
4. 每次 solve 后按阈值检查方向，而不是只检查 `g^T p < 0`
5. line search 若连续 hit max-iter，则视为 `search_direction_invalid`
6. 出现方向失效时，`socu_approx` fail-fast，并把 gate / direction reason 写入 report
7. 不在 `socu_approx` 内部自动回退到 `fused_pcg`；若用户需要稳态生产路径，应继续使用默认 `fused_pcg`

### 与当前 `advance_ipc` 外层流程的对接点

当前 `cuda_mixed` 的 IPC 外层 loop 大致是：

1. 构建/更新 collision pairs
2. 组真实 gradient + Hessian
3. `m_global_linear_system->solve()`
4. 收集位移方向
5. 用真实总能量做 line search

第一版 `socu_approx` 建议不是重写整套 globalization，而是**只替换第 3 步的方向求解器**：

1. collision / active set / dytopo 逻辑保持不变
2. 真实梯度 `g` 仍然由现有 subsystem / reporter 路径提供
3. `GlobalLinearSystem` 和 dytopo/contact manager 都按 `SocuApproxSolver::assembly_requirements()` 跳过 full sparse `A` build
4. 改为装配 surrogate Hessian `\hat H`
5. 用 `socu` 解出方向 `p`
6. 复用现有 `collect_vertex_displacements()`
7. 复用现有 line search / CCD / CFL / tolerance 逻辑

这意味着：

- 现有 line searcher 不需要推倒重来
- 现有 `search_direction_invalid` 语义也可以直接复用
- 这条路更像“替换 direction source”，不是“替换 Newton 外层”

### `socu_approx` 第一版实现草图

第一版建议把它做成一个**独立的 direction solver**，而不是硬塞进当前 `IterativeSolver` 接口。

原因很简单：

- `IterativeSolver` 当前假设自己服务的是完整 sparse `A x = b`
- `socu_approx` 想服务的是 surrogate system `\hat H p = -g`
- 两者的数据来源和语义不同

因此建议引入单独对象，例如：

- `SocuApproxSolver`

它的输入应是：

- 真实梯度 `g`
- `StructuredChainProvider` 提供的 chain ordering / block metadata
- surrogate Hessian `\hat H`
- damping 参数
- gate / quality threshold 参数

它的输出应是：

- 一个全局方向向量 `p`
- 一个状态码：
  - `ok`
  - `not_descending`
  - `socu_runtime_error`
  - `gate_failed`
  - `residual_too_large`
  - `off_band_ratio_too_high`

注意：`factor_failed` 第一版不能作为精确状态码。当前 `socu_native` async embedding API 返回 `void`，没有 device-side factor status buffer；某些 proof path 内部即使有 Cholesky bool，也没有作为 public async API 返回。因此第一版只能通过：

- `socu_native` host-side exception / launch error
- solution finite check
- descent check
- surrogate residual check

来后验判断 `socu` 解是否可用。只有当 `socu_native` 增加显式 status buffer / async status API 后，`factor_failed` 才能成为真实可区分的状态。

`line_search_rejected` 不应是 `SocuApproxSolver::solve()` 的直接返回状态，因为 line search 发生在 `m_global_linear_system->solve()` 之后。若需要让 solver 知道后续 line search 结果，应在 `LinearSolver` 上新增 outer-loop feedback hook：

```cpp
struct LineSearchFeedback
{
    bool accepted = false;
    int  iteration_count = 0;
    bool hit_max_iter = false;
    double accepted_alpha = 0.0;
};

class LinearSolver
{
  protected:
    virtual void do_notify_line_search_result(const LineSearchFeedback&) {}
};
```

`advance_ipc` 在 line search 完成后调用这个 hook。`SocuApproxSolver` 可以据此统计连续失败次数，但不能在 solve 返回时预言 line search 会不会接受。

### 第一版默认阈值

第一版建议给 gate / report 设定可调默认值，避免“字段有了但测试不知道怎么判”：

```cpp
struct SocuApproxThresholds
{
    double min_block_utilization = 0.65;
    double min_near_band_ratio = 0.90;
    double max_off_band_drop_norm_ratio = 0.05;
    double residual_tol = 1e-3;
    double descent_eta = 1e-8;
    double p_min_abs_float32 = 1e-10;
    double p_min_abs_float64 = 1e-14;
    double p_min_rel = 1e-12;
};
```

第一版 fail/report-only 分层：

- gate fail：
  - block size 不是 `32` / `64`
  - MathDx capability unsupported
  - MathDx runtime preflight failed，例如 manifest、runtime artifact、device arch 或 cache dir 不可用
  - `StoreScalar != SolveScalar`
  - structured provider 缺失
  - block utilization 低于 `min_block_utilization`
  - near-band ratio 低于 `min_near_band_ratio`
  - off-band dropped norm ratio 高于 `max_off_band_drop_norm_ratio`
- solve fail：
  - `p` 非 finite
  - `||p|| < max(p_min_abs(dtype), p_min_rel * max(1, ||rhs||))`
  - descent check 不通过
  - surrogate residual 高于 `residual_tol`
- report-only：
  - ordering time
  - pack time
  - line-search reject streak
  - off-band pair count
  - block utilization 接近阈值但仍通过

### surrogate solve 的推荐数据流

第一版推荐按下面的数据流实现：

1. **组真实梯度 `g`**
   可复用现有 gradient-only 组装语义，不要求完整 Hessian 同时存在

2. **组 structured chain metadata**
   来自 `StructuredChainProvider`

3. **装配 surrogate Hessian `\hat H`**
   只写：
   - diagonal blocks
   - first upper / lower off-diagonal blocks
   - 可选的 near-band contact

4. **对 `\hat H` 做 damping**
   - 固定 shift
   - 或按场景配置给 `diag += \lambda I`

5. **调用 `socu` 解 `\hat H p = -g`**

6. **做方向有效性检查**
   - 检查所有元素 finite
   - 检查 `||p||` 不是 0 / inf / nan
   - 检查 `g^T p < -eta * ||g|| * ||p||`
   - 检查 `||\hat H p - rhs|| / max(1, ||rhs||) < residual_tol`
   - 若失败，则不进入 line search，直接 fail-fast

7. **把 `p` 写回现有全局 displacement buffer**

8. **复用现有 line search**

### RHS 符号与方向检查

当前 `GlobalLinearSystem` 汇总的是 `b`，`LinearFusedPCG` 解的是：

- `A x = b`

而近似 direct 章节习惯写成：

- `\hat H p = -g`

因此第一版必须在 `SocuApproxSolver` 里明确符号约定：

- 若 existing subsystem 写入的 `b` 已经等价于 `-g`，则 `rhs = b`
- 若 structured provider 输出的是真实 gradient `g`，则 `rhs = -g`
- report 中必须写出本次使用的是 `rhs_is_negative_gradient` 还是 `rhs_is_global_b`

方向检查建议使用阈值版本：

```text
finite(p)
||p||_2 > max(p_min_abs(dtype), p_min_rel * max(1, ||rhs||_2))
g^T p < -eta * ||g||_2 * ||p||_2
||Hhat p - rhs||_2 / max(1, ||rhs||_2) < residual_tol
```

其中：

- `eta` 初值可取 `1e-8` 到 `1e-6`
- `p_min_abs` 应按 dtype 取值；建议初值为 float32 `1e-10`、float64 `1e-14`
- `p_min_rel` 初值可取 `1e-12`，用来避免小尺度 / 大尺度问题上 absolute-only 阈值失真
- `residual_tol` 初值可比 PCG 线性残差更宽松，因为这里解的是 surrogate system
- padding lane 不参与 `g^T p`、`||p||` 或 residual 统计

这样能避免 `g^T p` 只是一个接近 0 的数值噪声负值，却被误判为有效下降方向。

### `n=32` / `n=64` superblock 的第一版建议

`n=32` / `n=64` 在这里应被理解为：

- **32x32 或 64x64 dense superblock**

而不是：

- “FEM 原本 3x3 / ABD 原本 12x12，所以天然等于 64”

因此第一版必须显式做两层映射：

1. 全局 DoF ordering -> chain ordering
2. chain ordering -> `32` 或 `64` sized superblock packing

建议第一版先选最保守的 packing 规则：

- 一个 superblock 内只放连续 chain 区段
- 不跨 chain merge
- 不为了追求 occupancy 把不相邻段塞到同一个 dense block 里

这样虽然可能牺牲一点 block 利用率，但会大幅降低：

- gather/scatter 复杂度
- contact inclusion 复杂度
- debug 难度

### 什么时候不该继续走 `socu_approx`

第一版建议把这些情况直接视为 fail-fast 信号：

1. surrogate chain gate 不通过
2. `socu` factor/solve 失败
3. finite / norm / descent / residual 检查不通过
4. outer-loop feedback 显示 line search 连续 hit max-iter
5. surrogate path 连续多轮触发 `search_direction_invalid`

对应行为建议是：

- `socu_approx` 本次 solve 报错并记录原因
- 默认 `fused_pcg` 路径完全不受影响
- 若需要生产级稳定性，用户应继续选择 `fused_pcg`，而不是期待 `socu_approx` 内部自动回退

### Gate 分层

为避免 “ABD-only smoke” 和 “FEM/ABD performance path” 混在一起，第一版 gate 应拆成三层：

1. **Level 0：standalone / proof smoke**
   - 可用 `n=12`
   - 可用 `NativeProof` 或 CPU dense reference
   - 只验证 sign、provider、pack、scatter
   - 不进入 frame wall-time 性能结论

2. **Level 1：single-provider performance gate**
   - 只允许一个 structured provider
   - block size 必须是 `32` 或 `64`
   - 不允许跨 provider coupling
   - near-band ratio / utilization 达标后，才允许跑 `socu_approx` frame benchmark

3. **Level 2：multi-provider FEM/ABD experimental gate**
   - 允许多个 provider 贡献同一个 structured chain 或多个 chain segment
   - 必须定义 chain segment stitching、coverage mask 和 off-provider coupling 处理
   - 这不是第一版默认 gate，必须在 Level 1 稳定后再开

文档后面的 “真实 FEM/ABD surrogate solve” 指的是 Level 2 experimental gate，不应被理解为 Milestone 4/5 就要支持通用 FEM+ABD。

### 评测指标不应只看单次 solve 时间

这条路线要看的第一优先指标不是：

- 单次 linear solve 更快了多少

而是：

- 每 frame 总 wall time
- 每个 frame 的 Newton iteration 数
- 每个 Newton iteration 的 line search iteration 数
- `search_direction_invalid` 触发频率

如果这条路导致：

- 单次方向很快
- 但 line search 经常爆掉
- 或 Newton iteration 明显上升

那它就不是真正更快。

### 这条路线的定位

这条路的定位应当是：

- **Direct solver（近似）这条主线的核心算法形态**

而不是：

- `socu_direct` / `socu_hybrid` 这类 product alias
- 也不是当前 `GlobalPreconditioner` 框架下的一种 preconditioner 变种

如果后续实验表明它在 wall time 上明显占优，再考虑是否把它提升成更正式的产品 solver 名称。第一版先不要扩大配置面，避免在算法还没有稳定时制造语义债。

### 第一版固定 knobs

第一版建议只开放最小必要面：

- `linear_system/solver = "socu_approx"`
- block size 优先 `32`，不达标再尝试 `64`
- runtime contact 只吸收 near-band contribution
- `PerfBackend::MathDx`
- `GraphMode::Off`
- `StoreScalar == SolveScalar`

等 correctness 和基本 benchmark 稳了，再开放更多 knobs。

### `socu_approx` gate / feedback / report schema

第一版不要只打散乱 log。建议定义一个结构化 report，至少包含：

```cpp
struct SocuApproxGateReport
{
    bool explicitly_selected = false;
    bool supported = false;
    std::string fatal_reason;

    int block_size = 0;
    int horizon = 0;
    int nrhs = 1;
    std::string dtype;

    bool needs_full_sparse_A = false;
    bool needs_gradient_b = true;
    bool needs_structured_chain = true;

    double block_utilization = 0.0;
    double near_band_ratio = 0.0;
    double off_band_ratio = 0.0;
    double off_band_drop_norm = 0.0;
    double off_band_drop_norm_ratio = 0.0;

    std::string rhs_sign_convention;
    std::string socu_capability_reason;

    SocuApproxThresholds thresholds_used;
    std::string resolved_backend;
    std::string resolved_perf_backend;
    std::string resolved_math_mode;
    std::string resolved_graph_mode;

    bool mathdx_build_enabled = false;
    bool mathdx_manifest_found = false;
    bool mathdx_runtime_bundle_found = false;
    bool mathdx_runtime_artifact_ok = false;
    bool mathdx_device_arch_supported = false;
    bool mathdx_runtime_cache_available = false;
    std::string mathdx_manifest_path;
    std::string mathdx_runtime_cache_path;
    std::string mathdx_preflight_reason;
};

struct SocuApproxSolveReport
{
    bool ok = false;
    std::string reason;

    double pack_ms = 0.0;
    double solve_ms = 0.0;
    double scatter_ms = 0.0;

    double g_norm = 0.0;
    double p_norm = 0.0;
    double g_dot_p = 0.0;
    double descent_margin = 0.0;
    double surrogate_relative_residual = 0.0;

    int line_search_reject_streak = 0; // updated by feedback hook
};
```

gate report 在 build / frame boundary 更新；solve report 在每次 direction solve 后更新；line search feedback 在 `advance_ipc` 完成 line search 后回填。这样失败原因能被测试读取，而不是依赖解析字符串。report 必须写入本次实际使用的阈值和 resolved runtime policy；否则同一个 scene 在不同 build 或不同 GPU 上失败时，很难区分是 gate 配置变化、MathDx artifact 缺失，还是数值方向真的坏了。

## 5.8 Direct Solver（近似）细分实施计划

这一节把 `socu_approx` 拆成可单独测试、可单独撤回的小步。原则是先证明 ordering 和数据布局真的能把主要 Hessian contribution 压到块三对角附近，再把 `socu` 接进真实 solve。不要一开始就把 ordering、assembly、solver、contact、line search 全部绑在一起调。

### 总体语义

第一版只新增一个显式实验 solver：

- `linear_system/solver = "socu_approx"`

它的语义固定为：

- init-time ordering 生成 chain ordering
- 优先尝试 `32x32` block tridiagonal surrogate
- `32` 不达标时再尝试 `64x64`
- structured provider 只装配 `diag`、first offdiag 和 RHS
- runtime contact 只吸收 near-band contribution
- off-band contact 只进入 quality report
- `socu_native` 直接解 `\hat H p = -g`
- 默认 `fused_pcg` 不变
- `socu_approx` gate 或方向失败时 fail-fast，不做隐式 hybrid fallback

这里的 `\hat H` 不是完整 Hessian，而是为快速方向求解有意识构造的 structured surrogate。成功标准也不是“方向等于 PCG 解”，而是“方向通常下降、line search 能接受、整帧 wall time 更低”。

### 现有 ordering 能力的定位

当前 `libuipc` 已有两类相关能力，但它们都不是 `socu_approx` 需要的全局 DoF bandwidth ordering：

- `src/geometry/mesh_partition.cpp` 已经使用 METIS 做 mesh partition，并写入 `mesh_part`
- `FEMMASPreconditioner` / `MASPreconditionerEngine` 会基于 partition 和 hierarchy 做 MAS 侧 reorder / mapping

这些能力可以参考，但不能直接当作 `socu` chain ordering 使用。`socu_approx` 需要的是：给定 FEM/ABD/contact graph，找一个全局 atom order，让主要 Hessian edge 尽量落入当前 block 或相邻 block。这个目标更接近 bandwidth / profile minimization，而不是传统 domain decomposition 或 MAS hierarchy。

### Milestone 0：Standalone Ordering Lab

新增离线工具：

- `apps/socu_ordering_bench` 或 `tools/socu_ordering_bench`

这个工具不接 `cuda_mixed` runtime，不调用 `socu_native`，只回答一个问题：给定一个 mesh / scene graph，能不能找到足够好的 chain ordering。

输入建议支持：

- 单个 surface / tet mesh
- ABD chain / affine body graph 的简化输入
- 可选 contact pair 文件
- 可选 synthetic scene preset，例如 rod、cloth grid、tet block、random contact

核心输出：

- `old_to_chain`
- `chain_to_old`
- `atom_to_block`
- `block_to_atom_range`
- `block_size = 32 或 64`
- block utilization
- near-band edge count
- off-band edge count
- near-band ratio
- off-band ratio
- max block distance
- ordering time

第一版 atom graph 建议这样定义：

1. FEM vertex 是 graph node，边来自 edge / tri / tet connectivity。
2. ABD 可先以 body 或 body-local dof group 作为 node，边来自 ABD chain / joint / coupling。
3. 如果 FEM 和 ABD 都在同一 scene 中，先允许它们各自生成 chain segment，再在更高层 quotient graph 上排序 segment。
4. contact pair 在 Milestone 0 可以只作为可选 edge，用来评估 ordering 对 contact 的敏感性，不作为必须吸收的结构。

排序策略建议分三层：

1. **Micro-partition**
   用 METIS 把 atom graph 划成目标大小接近 `32` 或 `64` DoF 的小 partition。partition 内部目标是高局部性和高 utilization。

2. **Quotient graph**
   把每个 partition 压成一个 quotient node，partition 间 edge weight 由原 graph 跨 partition edge 累加得到。

3. **Quotient ordering**
   对 quotient graph 做 bandwidth-oriented ordering。第一版可以实现 Reverse Cuthill-McKee 作为确定性 baseline，再提供 METIS recursive bisection / nested-dissection order 作为候选。最终不是盲信某个算法，而是用同一个 scorer 选择 near-band ratio 更好的结果。

Scorer 建议统一成：

```text
score = w_near * near_band_ratio
      - w_off * off_band_ratio
      - w_far * normalized_max_block_distance
      + w_util * block_utilization
      - w_time * normalized_ordering_time
```

第一版的目标不是一步找到最优 ordering，而是建立可重复 benchmark，避免后面凭视觉判断“排序好像更好了”。

测试点：

- 1D rod / chain：off-band ratio 应接近 0。
- cloth grid：ordering 后 off-band ratio 应显著低于原始顶点顺序。
- tet block：ordering 后 max block distance 和 off-band ratio 应低于原始顺序。
- `32` 不达标时自动尝试 `64`，并报告为什么 `32` 失败。
- 同一输入多次运行 ordering 结果稳定，或者明确报告 nondeterministic seed。
- 输出 CSV / JSON report，方便后续做 mesh corpus 对比。

### Milestone 1：Init-Time Reorder Lab

新增 opt-in standalone geometry reorder 工具。它只在离线实验中使用，不直接改变 `cuda_mixed` runtime。

这一步要区分两种 reorder：

1. **solver-owned mirror reorder**
   原 mesh 不变，只在 solver 侧维护 `old_to_chain / chain_to_old`，装配和 scatter 时通过映射访问。

2. **physical geometry reorder**
   直接重排 vertex attributes，并重写 edge / tri / tet topology index，让后续运行时内存访问更连续。

当前 `AttributeCollection::reorder()` 可以重排 attributes，`Simplices<N>::do_reorder()` 可以重排 simplex array，但 physical vertex reorder 还必须显式重写 topology 里的 vertex id。也就是说，不能只 reorder attributes；必须把 edge / tri / tet 的 `topo` index 全部从 old id 改成 new id。

新增 metadata 建议：

- `socu/original_vertex_id`
- `socu/chain_id`
- `socu/block_id`
- `socu/block_local_id`

测试点：

- reorder 前后 vertex count 不变。
- edge / tri / tet count 不变。
- topology index 全部合法。
- 每个 simplex 的几何 invariant 保持：
  - edge length
  - triangle area
  - tet volume
- solver-owned mirror reorder 和 physical reorder 得到同样的 block classification。
- physical reorder 后，按 chain 顺序读取 vertex attributes 的 memory access 更连续。
- original id 可完整反查，debug / IO 不丢身份信息。

性能判断：

- init-time physical reorder 理论上可以降低后续 mapping 成本和提升内存连续性。
- 但它会改变 geometry storage 的全局顺序，所以第一版只放在 standalone lab。
- 真正接 runtime 前，必须确认 renderer、collision、dytopo、attribute transfer 是否依赖原始顺序语义。

### Milestone 2：Runtime Contact Simulation Lab

在 standalone 工具里注入 simulated contact pairs，先不碰真实 collision pipeline。

contact classification 固定为：

- `chain_i == chain_j && abs(block_i - block_j) <= 1`：near-band，可吸收进 `\hat H`
- 其他：off-band，只统计，不进入 `\hat H`

这里的分类单位不能只停留在 contact pair。PT / EE / PE / PP contact 会展开成多个 Hessian block contribution，同一个 primitive 里可能一部分 block pair 是 near-band，另一部分是 off-band。
因此 report 至少要同时按两层统计：

- contact pair count
- expanded Hessian block contribution count / weight

输入模式建议支持：

- `near_band`：只生成相邻 block contact，用来验证 pack / classify 正路径
- `mixed`：按比例混合 near-band 与 off-band contact
- `adversarial`：优先生成远距离 contact，模拟最坏情况
- `from_file`：读取真实或录制的 contact pair 文件

输出 report：

- active contact count
- near-band contact count
- off-band contact count
- near-band block contribution count
- off-band block contribution count
- near-band ratio
- off-band ratio
- weighted near-band contribution norm
- weighted off-band dropped norm
- contact classify time
- estimated absorbed Hessian contribution count
- estimated dropped contribution count

测试点：

- near-band contact 被正确归类并计入 absorbed。
- adversarial contact 显著提高 off-band ratio。
- 少量高 stiffness off-band contact 能通过 weighted norm 被放大，而不是被 pair count 掩盖。
- classify time 单独统计，不能混进 ordering time。
- frame-boundary reorder time 单独统计。
- 不允许 Newton iteration 内重排；同一 frame 的 permutation 必须固定。

### Milestone 3：只加 `LinearSolver` 抽象，不改算法

这是最重要的 no-regression checkpoint。只做架构抽象，不新增 `socu` solver，不改默认配置。

新增：

- `src/backends/cuda_mixed/linear_system/linear_solver.h`
- `src/backends/cuda_mixed/linear_system/linear_solver.cu`

建议形态：

```cpp
class LinearSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo {};

    struct AssemblyRequirements
    {
        bool needs_dof_extent       = true;
        bool needs_gradient_b       = true;
        bool needs_full_sparse_A    = true;
        bool needs_structured_chain = false;
        bool needs_preconditioner   = true;
    };

    virtual AssemblyRequirements assembly_requirements() const = 0;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;

  private:
    friend class GlobalLinearSystem;
    GlobalLinearSystem* m_system = nullptr;
    virtual void do_build() final override;
    void solve(GlobalLinearSystem::SolvingInfo& info);
};
```

然后：

- `IterativeSolver : public LinearSolver`
- `LinearFusedPCG` 继续继承 `IterativeSolver`
- `LinearPCG` 继续继承 `IterativeSolver`
- `GlobalLinearSystem` 从 `SimSystemSlot<IterativeSolver>` 改成 `SimSystemSlot<LinearSolver>`
- `add_solver()` 改成接收 `LinearSolver*`
- `solve_linear_system()` 不关心 solver 是 iterative 还是 approximate direct
- `assembly_requirements()` 先对 PCG 返回 full sparse requirements

测试点：

- 默认 `fused_pcg` 场景结果逐点一致，或保持当前误差量级。
- `linear_pcg` 仍可通过配置选择。
- PCG iteration count 不变。
- `GlobalLinearSystem` build / solve 调用顺序不变。
- selected solver exactly one；`fused_pcg` 和 `linear_pcg` 的 unused path 仍用 `SimSystemException`。
- frame wall time 无明显回退。
- 现有 smoke / ctest 全过。
- 这一步的 commit 应该不包含 `socu_native` 依赖。

### Milestone 4：Skeleton `SocuApproxSolver`，只做 gate 与空方向验证

新增：

- `SocuApproxSolver : public LinearSolver`
- 唯一显式配置值 `linear_system/solver = "socu_approx"`

第一版 skeleton 不调用 `socu_native`，只验证配置、依赖和 ordering report 是否存在，然后输出 gate reason。

gate reason 建议结构化：

- `socu_disabled`
- `ordering_missing`
- `unsupported_precision_contract`
- `unsupported_block_size`
- `socu_mathdx_unsupported`
- `ordering_quality_too_low`
- `contact_off_band_ratio_too_high`
- `structured_provider_missing`

测试点：

- 默认 `fused_pcg` 不受影响。
- `socu_approx` 未满足 gate 时清晰报错。
- 如果用户显式选择 `socu_approx`，gate fail 必须 fatal，不能被 `SimSystemException` 静默吞掉。
- 无 `socu_direct` / `socu_hybrid` 配置面。
- build 系统能在未启用 `socu_native` 时编译 stub；未选择 `socu_approx` 时不影响默认 PCG，显式选择时 fatal。
- 启用 `socu_native` 但缺少 ordering report 时，错误指向 ordering gate，而不是崩在 solve 阶段。

### Milestone 5：Structured Assembly Dry Run

新增 `StructuredChainProvider` 的 ordering-aware path，但仍然不调用 `socu`。

dry-run pack 内容：

- `diag`
- first offdiag
- `rhs`
- near-band contact contribution
- quality report

off-band contact 行为：

- 不写入 `\hat H`
- 只写入 report
- 统计 dropped contribution

建议接口分层：

- subsystem 只负责把局部 contribution 写给 `StructuredAssemblySink`
- `StructuredChainProvider` 负责把 chain metadata、block offset、contact classification 交给 sink
- solver adapter 负责把 logical structured blocks pack 成 `socu_native` 的 `diag/off_diag/rhs`

测试点：

- block index / offset 正确。
- `32` 和 `64` layout 均可生成。
- near-band contact 写入目标 diagonal 或 first offdiag block。
- off-band contact 不写入。
- dry-run pack 与 standalone simulator 的 near/off-band 统计一致。
- 对同一输入，triplet legacy path 仍可独立运行并保持 no-regression。
- dry-run pack time 单独统计，不能和 solve time 混在一起。

### Milestone 6：Synthetic `socu` Solve Smoke

暂不接真实 Hessian，先用 synthetic SPD block-tridiagonal system 打通 `socu_native` 调用。

输入：

- synthetic `diag/off_diag`
- synthetic gradient `g`
- `n = 32`
- `n = 64`
- `StoreScalar == SolveScalar`

求解：

- `\hat H p = -g`

调用路径：

1. `query_solver_capability<T>()`
2. `describe_problem_layout()`
3. `create_solver_plan<T>()`
4. 填充 `diag/off_diag/rhs`
5. `factor_and_solve_inplace_async<T>(plan, diag, off_diag, rhs, {stream})`
6. host 侧只在测试边界同步

测试点：

- residual 有限。
- `g^T p < -eta * ||g|| * ||p||`。
- `||Hhat p - rhs|| / max(1, ||rhs||)` 低于 smoke 阈值。
- `n=32` / `n=64` 都能跑。
- capability query resolved perf backend 为 `MathDx`。
- solve stream 与 mixed backend 当前 stream 一致。
- repeated runs 无资源泄漏。
- plan / runtime cache 只初始化一次，steady-state 不重复构建重资源。
- `socu_native` error 能转换成 `SocuApproxSolver` 的结构化 reason。

### Milestone 7：真实 FEM/ABD Surrogate Solve

开始接真实 gradient `g` 和 near-band Hessian contribution。

数据流：

1. 使用 ordering-aware provider 生成 chain metadata。
2. 使用 structured sink 装配 `\hat H`。
3. 对 `\hat H` 加 damping：
   - 固定 `diag += lambda I`
   - 或根据 block diagonal scale 自适应 shift
4. 用 `socu_native` 解 `\hat H p = -g`。
5. 检查 finite、范数、descent threshold 和 surrogate residual。
6. 把 `p` scatter 回 `GlobalLinearSystem::x` 或现有 displacement buffer。
7. 复用现有 line search / CCD / CFL / tolerance。

测试点：

- `g^T p < -eta * ||g|| * ||p||`。
- `||Hhat p - rhs|| / max(1, ||rhs||)` 在设定阈值内。
- line search 能接受。
- Newton iteration count 被记录。
- line search iteration count 被记录。
- 与 `fused_pcg` 比较 frame wall time。
- 记录 `ordering_time / pack_time / socu_factor_time / socu_solve_time / scatter_time`。
- direction invalid 时 `socu_approx` fail-fast，不自动回退。
- contact-light 场景先过，再看更复杂场景。

### Milestone 8：真实 Runtime Contact Handling

在真实 frame 中统计 active contact 的 near/off-band ratio。

第一版行为：

- 只吸收 near-band contact。
- off-band contact 不进入 `\hat H`。
- off-band contact 只影响 quality report 和 gate。
- quality report 同时按 contact pair、expanded block contribution、weighted norm 三种尺度统计。
- 不在 Newton iteration 内改变 permutation。
- 可在 frame boundary 做可选 reorder 实验，但必须记录成本。

测试点：

- contact-light 场景能稳定运行。
- adversarial contact 场景触发 gate 或 direction invalid。
- frame report 包含 active contact near/off-band 统计。
- 连续失败时，本 frame 的 `socu_approx` 直接判定不可用。
- 默认 `fused_pcg` 场景不受影响。

### Cross-Milestone Test Plan

每个 milestone 都必须单独可测、可单独撤回。

关键 checkpoint：

1. **Milestone 0 到 2**
   只验证 ordering / reorder / contact classification，不碰 `cuda_mixed` runtime，不调用 `socu`。

2. **Milestone 3**
   只加 `LinearSolver` 抽象，原 PCG 行为必须稳定。这是进入 solver 接线前的 no-regression checkpoint。

3. **Milestone 4**
   `socu_approx` 配置面出现，但只做 gate，不求解。这里验证错误信息和 build optionality。

4. **Milestone 5**
   证明 pack / classify / block layout 正确。Milestone 5 前不允许调用 `socu`。

5. **Milestone 6**
   只用 synthetic SPD 系统验证 `socu_native` 调用链和 stream / cache 生命周期。

6. **Milestone 7 到 8**
   才看真实 simulation wall time，不用单次 `socu` solve time 代替整体收益。

核心接受指标：

- no-regression：默认 `fused_pcg` 结果、iteration count、wall time 不明显回退。
- ordering quality：`32` 优先，失败再 `64`，并有明确 scorer。
- correctness：topology / reorder invariant、finite residual、下降方向、line search 接受。
- performance：ordering time、classify time、pack time、solve time、scatter time、line search 次数、frame wall time 全部记录。
- semantics：无 `socu_direct` / `socu_hybrid`，无隐式 fallback。

### 需要明确不做的事

第一版不做：

- runtime Newton iteration 内重排
- off-band contact 进入 `\hat H`
- `socu_approx` 内部自动 fallback 到 `fused_pcg`
- 从 `BCOO` 热路径反推 `socu` 结构
- mixed-precision bridge
- 把 MAS preconditioner 和 `socu_approx` 混成一个求解器

这几个“不做”很重要。它们能让第一版实验边界保持干净，也能让失败结果有解释力。

---

## 6. mixed precision 契约建议

这是接入时最容易被忽略、但最该提前写死的一条。

### 第一版强约束

建议第一版只支持：

- `StoreScalar == SolveScalar`

也就是优先支持：

- `fp64`
- `path1`
- `path5`
- `path6`

而对：

- `path2`
- `path3`
- `path4`

先明确报 `unsupported precision contract`

### 为什么

因为当前 `socu_native<T>` 的 factor storage / rhs / solution 是同类型的。
如果在 `path2-4` 偷偷做 cast，最后会很难分辨：

- 是 solver 精度问题
- 还是 mixed backend contract 变化
- 还是 `socu` 与现有 `fused_pcg` 的误差容忍不同

先把支持面收窄，后面再扩，会更稳。

### 第二版可选扩展

如果后面确实要支持 `path2-4`，有两个方向：

1. 在 `libuipc` 里做显式 bridge：
   - `StoreScalar -> temp SolveScalar`
   - solve
   - 写回 `x`

2. 给 `socu_native` 增加 mixed-type API

这两条都不适合作为第一版目标。

---

## 7. Build / dependency 接法

`socu_native` 当前已经是可被 `add_subdirectory()` 使用的 CMake library target：

- target 名：`socu_native`
- 依赖：
  - `CUDA::cudart`
  - `CUDA::cuda_driver`
  - `CUDA::cublas`
  - `CUDA::cublasLt`
  - `CUDA::cusolver`
  - `Eigen3::Eigen`

### 推荐做法

在 `libuipc` 的 mixed backend 侧新增：

- `UIPC_CUDA_MIXED_ENABLE_SOCU=AUTO|ON|OFF`
- `UIPC_SOCU_NATIVE_SOURCE_DIR=/path/to/socu-native-cuda`（可选）

短期优先顺序建议：

1. 用户提供 `UIPC_SOCU_NATIVE_SOURCE_DIR` 时，优先 `add_subdirectory()`
2. `UIPC_CUDA_MIXED_ENABLE_SOCU=ON` 但 source dir 无效，则配置失败
3. `AUTO` 且没有 source dir，则不编真实 `socu_approx`，但仍编译一个不依赖 `socu_native` header 的 registry / stub
4. 只有在 `socu_native` repo 补齐 install/export config 后，再把 `find_package(socu_native CONFIG QUIET)` 放到优先路径

stub 规则：

- `linear_system/solver != "socu_approx"` 时，stub 走 unused path
- `linear_system/solver == "socu_approx"` 但 `socu_native` 未启用时，stub 必须 fatal，报 `socu_native_disabled_or_not_found`
- `GlobalLinearSystem` 仍然要验证 configured solver 已注册且 exactly one selected

这样即使 `AUTO` 模式下没编进真实 `socu`，用户显式选择 `socu_approx` 也不会变成“没有 solver 但 solve 静默跳过”。

MathDx performance path 还需要 build/runtime artifact preflight。短期 `add_subdirectory()` 路径下，`libuipc` 不应假设 `query_solver_capability()` 已经覆盖所有 artifact 状态；应在 `SocuApproxSolver::do_build()` 里调用一个薄 preflight helper，并把结果写入 `SocuApproxGateReport`：

- `SOCU_NATIVE_ENABLE_MATHDX` 是否开启
- `socu_native` 默认 manifest path 是否非空且存在
- manifest 是否包含当前 `dtype + n + rhs + op=factor_and_solve` 的 runtime backend bundle
- bundle 引用的 MathDx `.lto`、wrapper 所需 `.fatbin.lto` / unified `cusolverdx` artifact 是否存在
- 当前 CUDA device arch 是否匹配 artifact
- runtime cubin cache 目录是否可创建 / 可写

preflight 失败时，显式 `socu_approx` 必须 fatal，建议 reason 使用 `socu_mathdx_runtime_artifact_unavailable`。首次 solve 时仍允许 lazy link / cache miss；preflight 只负责把 manifest 或 artifact 缺失提前暴露。

长期 `find_package` 要求 `socu_native` 先补齐：

- `install(TARGETS socu_native EXPORT ...)`
- `socu_nativeConfig.cmake`
- CUDA arch / toolkit requirement 透传
- MathDx enable / manifest / runtime artifact path 透传
- Eigen 依赖处理，避免 `libuipc` 和 `socu_native` 各自 FetchContent 出两份不一致的 Eigen
- `SOCU_NATIVE_ENABLE_MATHDX` 与 libuipc build option 的关系

### 不建议的做法

- 直接把 `socu_native` 源文件复制进 `libuipc`
- 在 `libuipc` 里重复维护 `MathDx` runtime 逻辑
- 让 `libuipc` 直接依赖 `socu_native_cli`

---

## 8. 具体落地步骤

实际落地按 `5.8` 的 Milestone 0 到 Milestone 8 推进。这里给出工程执行顺序，避免实现时把实验工具、抽象层和真实 solver 混在同一个大 patch 里。

### Step 0：离线排序实验先行

先实现 `socu_ordering_bench`，只处理 mesh / graph / contact pair 输入，不链接 `cuda_mixed` runtime。

完成条件：

- 能输出 `old_to_chain / chain_to_old / atom_to_block`
- 能对 `32` 和 `64` 分别评分
- 能报告 block utilization、near/off-band ratio、ordering time
- 能在 rod / cloth / tet mesh 上证明排序后优于原始顺序

### Step 1：standalone reorder 与 contact simulation

在离线工具里补两件事：

- opt-in physical geometry reorder，重排 vertex attributes 并重写 topology index
- simulated contact classifier，支持 `near_band / mixed / adversarial / from_file`

完成条件：

- topology / area / volume invariant 全过
- physical reorder 与 solver-owned mirror reorder 的 block classification 一致
- contact classify time、near/off-band ratio 可单独输出
- frame-boundary reorder cost 可模拟，不进入 Newton iteration

### Step 2：只加 `LinearSolver` 抽象

这是第一个进入 `cuda_mixed` 的 patch，但它不能包含 `socu` 求解逻辑。

完成条件：

- `IterativeSolver : public LinearSolver`
- `LinearFusedPCG` / `LinearPCG` 行为不变
- `GlobalLinearSystem` 持有 `LinearSolver`
- `LinearSolver::assembly_requirements()` 存在，PCG 默认要求 full sparse `A` + preconditioner
- `advance_ipc` / dytopo-contact assembly 能读取 selected solver 的 Newton assembly mode，但对 PCG 行为保持等价
- 默认 `fused_pcg` 结果、iteration count、wall time 不明显回退

### Step 3：接入 skeleton `SocuApproxSolver`

新增唯一实验配置：

- `linear_system/solver = "socu_approx"`

此时只做 gate，不调用 `socu_native`。

完成条件：

- 默认 `fused_pcg` 不受影响
- `socu_approx` gate fail 时 reason 清晰
- `socu_approx` 的 requirements 为 gradient + structured-chain，不要求 full sparse `A`
- 未启用 `socu_native` 时 stub 可编译；未选择 `socu_approx` 时 unused，显式选择时 fatal
- 不出现 `socu_direct` / `socu_hybrid` 配置面

### Step 4：structured dry-run pack

新增 `StructuredChainProvider` ordering-aware path，只做 dry-run pack。

完成条件：

- `diag/off_diag/rhs` block index 和 offset 正确
- `32` 和 `64` 都能生成 layout
- near-band contact 被写入，off-band contact 只报告
- dry-run 统计与 standalone simulator 一致
- legacy triplet path 无回归

### Step 5：synthetic `socu` solve

用 synthetic SPD block-tridiagonal system 打通 `socu_native` 调用。

完成条件：

- `n=32` / `n=64` 都能跑
- residual 有限
- `g^T p < -eta * ||g|| * ||p||`
- stream 与 mixed backend 一致
- repeated runs 不泄漏、不重复构建重资源

### Step 6：真实 surrogate solve

接入真实 gradient 和 near-band Hessian contribution，生成 `\hat H` 并求方向。

完成条件：

- `\hat H` 有 damping
- `socu` 解出的方向写回当前 displacement buffer
- line search 能接受
- 记录 ordering / pack / solve / scatter / line-search 时间
- direction invalid 时 fail-fast，不自动回退

### Step 7：真实 runtime contact handling

真实 frame 中统计 active contact near/off-band ratio。第一版只吸收 near-band contact。

完成条件：

- contact-light 场景稳定运行
- adversarial contact 场景触发 gate 或 direction invalid
- frame report 包含 contact 质量统计
- 同一 Newton iteration 内 permutation 固定

如果只看 solver 核心时间，不看整步时间，很容易高估 approximate direct solver 的收益。最终判据必须是整帧 wall time、Newton iteration 数、line search iteration 数和 failure rate。

---

## 9. 文件落点建议

### 一定会碰到的文件

| 作用 | 文件 |
|---|---|
| solver 配置默认值 | `src/core/core/scene_default_config.cpp` |
| mixed backend 依赖接线 | `src/backends/cuda_mixed/CMakeLists.txt` |
| 全局 solver 持有者 | `src/backends/cuda_mixed/linear_system/global_linear_system.h` |
| 全局 solver 调用点 | `src/backends/cuda_mixed/linear_system/global_linear_system.cu` |
| build mode / assembly requirement | `src/backends/cuda_mixed/linear_system/global_linear_system.h/.cu` |
| Newton assembly mode 传播 | `src/backends/cuda_mixed/engine/advance_ipc.cu` 与 `src/backends/cuda_mixed/dytopo_effect_system/global_dytopo_effect_manager.*` |
| 现有 iterative solver 抽象 | `src/backends/cuda_mixed/linear_system/iterative_solver.h` |
| 现有 iterative solver 实现 | `src/backends/cuda_mixed/linear_system/iterative_solver.cu` |

### 新增文件建议

| 作用 | 文件 |
|---|---|
| 通用 linear solver 抽象 | `src/backends/cuda_mixed/linear_system/linear_solver.h/.cu` |
| structured chain provider 抽象 | `src/backends/cuda_mixed/linear_system/structured_chain_provider.h` |
| `socu_approx` solver 本体 | `src/backends/cuda_mixed/linear_system/socu_approx_solver.h/.cu` |
| `socu` 配置 / gate 工具 | `src/backends/cuda_mixed/linear_system/socu_solver_common.h/.cu` |
| `socu` quality / gate report schema | `src/backends/cuda_mixed/linear_system/socu_approx_report.h` |
| 未启用 `socu_native` 时的显式失败 stub | `src/backends/cuda_mixed/linear_system/socu_approx_solver_stub.cu` |
| line-search feedback hook | `src/backends/cuda_mixed/linear_system/linear_solver.h/.cu` 与 `src/backends/cuda_mixed/engine/advance_ipc.cu` |
| ordering lab 工具 | `tools/socu_ordering_bench/` 或 `apps/socu_ordering_bench/` |
| standalone reorder / contact simulation | `tools/socu_ordering_bench/` 内的子命令 |

### 第一版 provider 候选

| 作用 | 文件 |
|---|---|
| `ABD` structured provider 扩展点 | `src/backends/cuda_mixed/affine_body/abd_linear_subsystem.h/.cu` |

### 第二版 provider 候选

| 作用 | 文件 |
|---|---|
| `FEM` chain provider | `src/backends/cuda_mixed/finite_element/fem_linear_subsystem.h/.cu` |

---

## 10. 我不建议第一版做的事

这些都很诱人，但不建议第一版碰：

1. **从 `BCOO` 热路径反推 `socu` 结构**
2. **同时支持所有 mixed precision path**
3. **一上来就把 `FEM + ABD + coupling + contact` 通吃**
4. **把 `socu` 直接接成 MAS 的替代品**
5. **让 `libuipc` 理解 `MathDx` 的内部 runtime 细节**

这些事不是永远不能做，而是第一版做了，成功概率会明显下降。

---

## 最终建议

如果目标真的是“把 `socu_native` 接进 `cuda_mixed`，并且只保留两种实用路线：Direct solver（近似）和 `socu` as preconditioner”，那我建议按下面的优先级推进：

1. **先做架构正确**
   先把 `LinearSolver` 抽象立住，不要让近似 direct solver 伪装成 iterative solver；同时明确这层接口负责的是“当前 outer solve 的方向求解边界”。

2. **先做结构正确**
   用 `StructuredChainProvider` 让 subsystem 原生装配 `socu` 格式，不要从全局 `BCOO` 回推。

3. **先做 ordering 正确**
   先用 standalone lab 证明 mesh / ABD / contact graph 在 `32` 或 `64` block 下确实能形成足够好的 near-band 结构，再接真实 solver。

4. **先做配置语义正确**
   第一版只新增 `linear_system/solver = "socu_approx"`。默认 `fused_pcg` 不变，`socu_approx` gate 或方向失败就 fail-fast，不做隐式 hybrid fallback。

5. **先做真实整帧收益验证**
   不用单次 `socu` solve time 代替结论。必须同时记录 ordering、pack、solve、scatter、Newton iteration、line search iteration 和 frame wall time。

6. **最后再做第二条路线**
   等 `socu_approx` 这条链路稳了，再考虑 `socu` as preconditioner，以及更广的 FEM chain 和 mixed precision bridge。

这条路线的核心不是“最短”，而是**最不容易把现有 mixed backend 弄乱，同时又最有机会真拿到性能收益**。
