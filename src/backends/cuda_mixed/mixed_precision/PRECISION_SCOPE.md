# PRECISION_SCOPE — cuda_mixed 后端精度范围文档

> 版本：V3.1 | 对应计划：`claude_plan.md`
> 约束：`src/backends/cuda` 不可修改；编译期精度切换；运行期不切换（V1）

---

## 一、精度维度定义

精度由 5 个独立维度控制，均通过 `ActivePolicy`（`mixed_precision/policy.h`）暴露：

| 类型别名 | 含义 | fp32 路径 |
|---|---|---|
| `AluScalar` | 计算/ALU 精度（梯度、Hessian 内核计算） | Path1/3/5/6/7 |
| `StoreScalar` | 存储精度（Hessian Triplet/BCOO、梯度向量） | Path2/3/4/5/6/7 |
| `PcgAuxScalar` | PCG 辅助向量（r/z/p/Ap） | Path4/5/6/7 |
| `SolveScalar` | 求解向量 x | Path7 only |
| `PcgIterScalar` | PCG 迭代标量（rz, alpha, beta） | Path7 only |

附加编译期 flag：

| Flag | 含义 | 激活路径 |
|---|---|---|
| `preconditioner_no_double_intermediate` | 预条件子无 double 中间量 | Path6/7 |
| `full_pcg_fp32` | 完整 PCG 全 fp32（含 x） | Path7 only |

---

## 二、路径矩阵

| Path | AluScalar | StoreScalar | PcgAuxScalar | SolveScalar | PcgIterScalar |
|------|-----------|-------------|--------------|-------------|---------------|
| fp64 | double | double | double | double | double |
| path1 | **float** | double | double | double | double |
| path2 | double | **float** | double | double | double |
| path3 | **float** | **float** | double | double | double |
| path4 | double | **float** | **float** | double | double |
| path5 | **float** | **float** | **float** | double | double |
| path6 | **float** | **float** | **float** | double | double |
| path7 | **float** | **float** | **float** | **float** | **float** |

> path6 与 path5 类型相同，但额外启用 `preconditioner_no_double_intermediate`

---

## 三、精度组件范围表

### AluScalar 组件（path1 起激活）

| # | 组件 | 文件 | 状态 |
|---|------|------|------|
| 1 | Contact 法向 ALU 梯度 | `contact_system/contact_models/ipc_simplex_normal_contact.cu` | ⚠️ |
| 2 | Contact 法向 ALU Hessian | `contact_system/contact_models/ipc_simplex_normal_contact.cu` | ⚠️ |
| 3 | Contact 摩擦/半平面 ALU | `contact_system/contact_models/ipc_simplex_frictional_contact.cu` | ⚠️ |
| 4 | FEM SNH 变形梯度 F ALU | `finite_element/constitutions/stable_neo_hookean_3d.cu` | ⚠️ |
| 5 | FEM SNH 能量梯度 G ALU | `finite_element/constitutions/stable_neo_hookean_3d.cu` | ⚠️ |
| 6 | FEM SNH Hessian H ALU | `finite_element/constitutions/stable_neo_hookean_3d.cu` | ⚠️ |
| 7 | ABD OrthoPotential ALU | `affine_body/constitutions/ortho_potential.cu` | ✅ |
| 8 | ABD ARAP ALU | `affine_body/constitutions/arap.cu` | ✅ |
| 9 | ABD RevoluteJoint ALU | `affine_body/constitutions/affine_body_revolute_joint.cu` | ✅ |
| 10 | ABD PrismaticJoint ALU | `affine_body/constitutions/affine_body_prismatic_joint.cu` | ✅ |
| 11 | ABD RevoluteJointLimit ALU | `affine_body/constitutions/affine_body_revolute_joint_limit.cu` | ✅ |
| 12 | ABD PrismaticJointLimit ALU | `affine_body/constitutions/affine_body_prismatic_joint_limit.cu` | ✅ |
| 13 | ABD BDF1 动能 ALU | `affine_body/bdf/affine_body_bdf1_kinetic.cu` | ✅ |
| 14 | ABD SoftTransformConstraint ALU | `affine_body/constraints/soft_transform_constraint.cu` | ✅ |
| 15 | ABD ExternalArticulationConstraint ALU | `affine_body/constraints/external_articulation_constraint.cu` | ✅ |
| 16 | ABDJacobi J^T H J ALU domain | `affine_body/abd_jacobi_matrix.h/.cu` | ⚠️ |
| 17 | ABDJacobiStack mat-vec / to_mat ALU | `affine_body/details/abd_jacobi_matrix.inl` | ⚠️ |
| 18 | ABD 线性子系统 kinetic+shape | `affine_body/abd_linear_subsystem.cu` | ✅ |
| 19 | ABD 线性子系统 Reporter | `affine_body/abd_linear_subsystem.cu` | ✅ |
| 20 | ABD 线性子系统 DyTopo | `affine_body/abd_linear_subsystem.cu` | ✅ |
| 21 | ABD-FEM 耦合 | `coupling_system/abd_fem_linear_subsystem.cu` | ✅ |

### StoreScalar 组件（path2 起激活）

| # | 组件 | 文件 | 状态 |
|---|------|------|------|
| 22 | Reporter/Assembler 局部缓冲区 | `affine_body/abd_linear_subsystem.h`<br>`finite_element/fem_linear_subsystem.h` | ❌ |
| 23 | Global Triplet Hessian A_triplet | `linear_system/global_linear_system.h` | ❌ |
| 24 | Global BCOO Hessian A_bcoo | `linear_system/global_linear_system.h` | ❌ |
| 25 | Global 梯度向量 b | `linear_system/global_linear_system.h` | ❌ |

### PcgAuxScalar 组件（path4 起激活）

| # | 组件 | 文件 | 状态 |
|---|------|------|------|
| 26 | PCG 辅助向量 r/z/p/Ap | `linear_system/linear_pcg.h` | ❌ |

### 固定 double（不受 path 影响）

| # | 组件 | 文件 | 说明 |
|---|------|------|------|
| 27 | 求解向量 x + 收敛标量 + SpMV 接口 | `linear_system/global_linear_system.h`<br>`linear_system/spmv.h` | 仅 path7 可降为 float（SolveScalar） |

---

## 四、插入点速查

修改内核时对照此表确认在哪个层次操作：

| 插入点 | 含义 | 适用维度 |
|--------|------|---------|
| A | IPC barrier 核心计算 | AluScalar |
| B | Contact 法向梯度/Hessian 写入前 | AluScalar |
| C | Matrix assembler 写入接口（最通用） | StoreScalar |
| D | FEM 能量标量 | AluScalar |
| E | FEM 梯度/Hessian 写入前 | AluScalar |
| F | FEM 内核模板特化（.inl 文件） | AluScalar |
| G | SpMV（混合精度矩阵-向量乘） | StoreScalar × PcgAuxScalar |

---

## 五、新功能合并检查清单

当 `main` 分支（`cuda` 后端）新增功能后，rebase 到 `mipc` 前后按以下步骤检查：

### Step 1 — 扫描新增的计算代码

```bash
# rebase 后查看 main 新带来的 cuda 后端改动
git diff ORIG_HEAD..HEAD -- src/backends/cuda/ \
  | grep '^+' \
  | grep -vE '^(\+\+\+|---)'
```

### Step 2 — 判断新功能属于哪类计算

- **新 constitution / constraint**（新的 `.cu` 能量/梯度/Hessian 计算）
  → 检查是否需要 `AluScalar` 适配（参考第三节 #7–#21 的现有模式）

- **新 Reporter / Assembler 缓冲区**
  → 检查是否需要 `StoreScalar` 适配（参考 #22–#25）

- **新线性子系统或 PCG 变体**
  → 检查 PCG 辅助向量是否需要 `PcgAuxScalar`（参考 #26）

- **纯非计算性变动**（IO、调度、geometry、UI）
  → 通常无需精度适配，可跳过

### Step 3 — 更新本文档

在第三节对应维度的表格中添加新行，状态标为 `❌`（待实现）或直接实现后标为 `✅`。

### Step 4 — 运行 quality benchmark 验证

```bash
python apps/benchmarks/uipc_assets/run_uipc_assets_benchmark.py run \
  --compare_levels path1 path2 path3 path4 path5 path6 path7 \
  --build_fp64 build_impl_fp64 --build_path1 build_impl_path1 --build_root . \
  --cache_dir <hf_cache_dir> \
  --run_root output/benchmarks/uipc_assets/<run_id>
```

在输出的 `summary.md` 中确认各 path 的误差在阈值内：
- `rel_l2_x.max < 1e-5`
- `abs_linf_x.max < 5e-4`
- `nan_inf_count = 0`

---

## 六、状态说明

| 符号 | 含义 |
|------|------|
| ✅ | 已实现，通过 quality benchmark 验证 |
| ⚠️ | 部分实现（存在 bridge gap 或接口未完整覆盖） |
| ❌ | 尚未实现 |
