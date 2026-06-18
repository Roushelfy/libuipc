# RCC bonded-PT virtual-tet energy models

The RCC bonded point-triangle system replaces each locked PT pair with a **virtual
tetrahedron** (point + 3 triangle vertices) carrying an elastic energy. The energy that
"glues" the tet is a **pluggable constitution**, selected per scene by the config key
`rcc_bonded_pt_energy_model`.

## Models

| `rcc_bonded_pt_energy_model` | material config | formulation |
|---|---|---|
| `abd_ortho` (default) | `rcc_bonded_pt_kappa` | ABD orthogonal potential `ψ = κ‖FFᵀ−I‖²`, evaluated via `q = abd_q_from_F(F)` and `sym::abd_ortho_potential` |
| `stable_neo_hookean` | `rcc_bonded_pt_neohookean_young`, `rcc_bonded_pt_neohookean_poisson` | Stable Neo-Hookean (Smith et al. 2018), F-based; the closed form of `sym::stable_neo_hookean_3d` reimplemented inline (see note below) |

Neo-Hookean Lamé parameters are derived from Young's modulus `E` + Poisson ratio `ν`:

```
mu     = E / (2 (1+ν))            # shear / second Lamé
lambda = E ν / ((1+ν)(1−2ν))      # first Lamé
```

Defaults when unset: `E = 5e7`, `ν = 0.45` (a representative electrical-tape modulus). The
energy is scaled by `rest_volume · dt²` (semi-implicit incremental-potential convention),
identical to the abd_ortho path.

## Architecture — compile-time energy-policy dispatch

The virtual tet's per-element math splits into:

- **Material-specific** `F → (E, dEdVecF[9], ddEddVecF[9×9])`, captured in an **energy policy**
  struct (`AbdOrthoEnergy`, `StableNeoHookeanEnergy` in
  `src/backends/cuda/contact_system/rcc_bonded_pt_virtual_tet_reporter.cu`). Each returns the
  gradient/Hessian in the **column vec(F) convention** that `fem::dFdx` consumes (abd_ortho:
  `abd_row_*_to_column_*`; neo-hookean: `flatten`, identical to `stable_neo_hookean_3d.cu`).

> **Why the Neo-Hookean policy reimplements the closed form instead of calling
> `sym::stable_neo_hookean_3d`:** that header's `.inl` writes `g1.block<3,1>(...)` /
> `HJ.block<3,3>(...)` on a *template-dependent* matrix (`Matrix<T,…>`) without the `.template`
> disambiguator. GCC tolerates this in the existing non-templated constitution TU, but when the
> header is pulled into this file's **policy-templated** context the same lines fail to parse
> (`block` is seen as a non-template → `block < 3`). The reporter's policy methods are **concrete**
> (`Float`), so the inlined copy needs no `.template` and compiles cleanly. The CPU oracle keeps its
> own independent copy; the finite-difference test guards both against the actual energy, so the
> three implementations (reporter, oracle, `sym::`) are cross-checked rather than trusted.
- **Shared geometry/assembly** `F = fem::F(...)`, `dFdxᵀ · make_spd(H9×9) · dFdx → 12×12`, the
  bare-`Float` energy fast path, and the triplet scatter — templated on the policy:
  `eval_virtual_tet_energy<Policy>` and `eval_virtual_tet<MODE, Policy>`.

`Impl::compute_energy` / `compute_dense_energy_gradient_hessian` / `assemble` switch on the model
**once at the C++ launch level** and invoke the policy-templated `run_*` kernel launcher. The CUDA
kernel itself is therefore **branch-free** and gets model-specific register allocation.

### Performance notes

- The bare-`Float` energy path (no `Matrix12x12` local → no spill, the 76e80918 occupancy win) is
  preserved for **both** models.
- The gradient+Hessian kernel hits the 255-register ceiling regardless of model — this is universal
  across *all* libuipc FEM constitutions (`fem::dFdx` triple product + 9×9 `make_spd` eigensolve +
  12×12), verified by `cuobjdump`. So Neo-Hookean is **not** expected to change occupancy. Its
  per-element flop profile differs (no `abd_q_from_F`; adds `J=det(F)`, `log(Ic+1)`), which is what
  the A/B comparison measures.
- Compile-time dispatch means each model is a separate kernel instantiation; the `if`/`switch` is
  amortized once per launch, never per thread.

## Verification (CPU oracle + GPU + finite differences)

- CPU reference oracle: `build_rcc_bonded_pt_virtual_tet_oracle`
  (`src/core/core/rcc_bonded_pt_oracle.cpp`) reimplements each model's `E/dEdF/ddEddF` on the host
  **independently** (not calling the cuda `sym::` functions), so a GPU-vs-oracle test is a real
  cross-check of two implementations.
- Tests (`apps/tests/backends/cuda/rcc_bonded_pt_virtual_tet_reporter.cu`):
  - `[neohookean][cuda]` — GPU reporter == oracle (energy/grad/Hessian, rel-err < 1e-10).
  - `[neohookean][fd]` — oracle grad/Hessian == central finite differences of the energy
    (implementation-independent guard that the closed form is the correct derivative).
  - The existing `[abd_oracle]` test is a regression guard for the abd_ortho path.

## How to add a third model

1. Add a value to `RCCBondedPTVirtualTetEnergyModel` (`include/uipc/core/rcc_bonded_pt_oracle.h`)
   and any new material fields to `RCCBondedPTVirtualTetInput`.
2. Add a CPU branch to `build_rcc_bonded_pt_virtual_tet_oracle` (independent reference).
3. Add an energy-policy struct (`Mat`, `inactive`, `E`, `dEdVecF`, `ddEddVecF`) in the reporter `.cu`.
4. Wire the model + material into `do_build` (config parse) and the `Impl` method `switch`es; add
   members + a `set_material_*` setter to the `.h`.
5. Add a GPU-vs-oracle + finite-difference test case.

## Running the A/B (rod-wind)

```bash
# Neo-Hookean (Young/Poisson)
rcc_adhesive_tape_rod_wind_demo.py --asset <wound.npz> \
  --set RCC_ENERGY_MODEL=stable_neo_hookean \
  --set RCC_NEOHOOKEAN_YOUNG=5e7 --set RCC_NEOHOOKEAN_POISSON=0.45 ...
# vs the default abd_ortho (RCC_KAPPA=...)
```

Compare wall time (timer / `ncu` on `assemble` + `compute_energy`), stability (frames / INVALID),
and physical effect (rendered video).
