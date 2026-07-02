# Tape-Cutting Mechanism — Feasibility & Design

Design analysis for cutting a simulated tape (a `NeoHookeanShell` FEM strip) — e.g. a
tape dispenser's blade severing the roll from the laid seam in the carton tape-seal demo
([`python/examples/carton_tape_seal_v2_demo.py`](../../python/examples/carton_tape_seal_v2_demo.py)).

**Bottom line:** the CUDA backend cannot change mesh topology at runtime, so a cut cannot
delete/split triangles mid-simulation. The practical mechanism is **topology-free**:
pre-split the tape at authoring time and **release the RCC bonds** that hold the halves
together, triggered by the blade. This reuses the existing bonded-PT machinery and needs
no backend change for a working version.

## 1. Hard constraint: topology is frozen at `world.init()`

The backend snapshots vertex/triangle connectivity into fixed-size device buffers exactly
once at init and every per-frame path assumes it never changes:

- The FEM build reads `triangles().topo()` once and bakes each triangle's global vertex
  ids with a fixed per-geometry `vertex_offset` —
  `finite_element_method.cu:593-630`. This is the same fixed global-index remap the RCC
  lock-restore relies on.
- The global vertex/surface buffers are sized once from each reporter's `report_count()`
  in `GlobalVertexManager::init` / `GlobalSimplicialSurfaceManager::init`
  (`global_vertex_manager.cu:28-90`, `global_simplicial_surface_manager.cu:63-125`),
  invoked once from `SimEngine::init_scene`.
- The only functions that could resize those buffers for a new topology —
  `GlobalVertexManager::rebuild` and `GlobalSimplicialSurfaceManager::rebuild` — are both
  `UIPC_ASSERT(false, "Not implemented yet")` stubs (`global_vertex_manager.cu:103-106`,
  `global_simplicial_surface_manager.cu:162-165`).
- The per-frame "rebuild the vertex and surface info" step is a commented-out TODO
  (`advance_ipc.cu:280-282`); no FEM/ABD/global-geometry system registers an
  `on_rebuild_scene` action; and FEM write-back writes **only positions**, with the
  explicit comment *"Now there is no topology modification, so no need to write back"*
  (`finite_element_method.cu:1039-1066`).
- There is **no** fracture / tearing / remeshing / cut code anywhere in `src/` or
  `include/`. (The `dytopo_effect_system` is unrelated — it adds extra energy terms
  (point-picker, vertex-stitch, RCC bonds) on top of a *fixed* mesh, not re-triangulation.)

> The frontend `SimplicialComplex` *is* fully mutable (resize/reorder/rewrite `topo`,
> attribute create) — but only **before** `world.init()`. Authoring-time mesh edits are
> fine; runtime edits do not propagate to the device.

**Consequence:** the cut must be topology-free — pre-split the mesh at authoring time and
change *bond state* at runtime, since vertex/simplex count and connectivity are immutable
after init.

## 2. The runtime cut lever: per-pair `bonded_release_force`

RCC bonds form and release **automatically every frame** by geometric criteria:

- **Form** (`ipc_simplex_rcc_adhesive_contact.cu:1337-1576`,
  `rcc_bonded_pt_system.cu:483-522`): a point-triangle pair locks when it is in-band
  (`gap < ξ + ratio·d_hat`), projects inside the face (face-interior gate), and its
  lock-β ≥ `beta_lock_threshold` (0.9 in the demos).
- **Release** (`rcc_bonded_pt_system.cu:279-306`): each frame the policy recomputes strain
  / normal-gap / slip / restoring-force for every carried lock and releases it (under
  **tension only** — a load-bearing pressed bond is kept) when any exceeds the pair's
  threshold. Thresholds are looked up **per contact-element-pair** from the adhesive
  tabular (`rcc_bonded_pt_system.cu:208-268`), falling back to the global
  `rcc_bonded_pt_release_*` config for `-1` pairs.

Bonds are defined per **contact-element-pair** via
`RCCAdhesive.set_bonded(tabular, L, R, lock_threshold, release_strain, release_gap,
release_slip, release_force, distance_lock, distance_lock_ratio)`
(`rcc_adhesive.cpp:186-237`; pybind `rcc_adhesive.cpp:92-124`). `distance_lock > 0` gives a
rigid tet lock (no β energy); `= 0` gives the soft β-spring adhesion.

**The lever:** `event_rebuild_scene()` fires unconditionally at the start of every frame,
and `IPCSimplexRCCAdhesiveContact` re-reads the contact tabular and re-uploads the
`N×N` adhesive coefficient table to the device **every frame**
(`ipc_simplex_rcc_adhesive_contact.cu:172-173, 331-348`). So from Python, between
`advance()` calls, **lowering a pair's `bonded_release_force` toward 0 makes its locks tear
under the seam tension the next frame** — that is the "cut". Cut granularity therefore
equals **contact-element granularity**.

**Not available:** a per-bond runtime unlock. The state-accessor features
(`RCCBondedPTStateAccessorFeature.seed_locks`, `RCCAdhesionStateAccessorFeature.load_pt_state`)
are **save/restore only**, documented for use *after `world.init` and before the first
`world.advance`* (`state_accessor_feature.cpp:157-331`); they overwrite the whole bridge
and cannot target a subset mid-sim.

## 3. Detecting where the blade cuts

No per-contact query is exposed to Python, but an animator callback already reads live
vertex positions each frame (`geo.positions().view()` — the pattern used by `animate_tape`
/ `_drive_end` in the carton demo). So the simplest trigger is: **in the animator, read
tape vertex positions and select the seam(s) within a thin slab around the blade's
cutting line**, then lower those seams' `release_force` (option 3a) — or drive a physical
blade into the seam so the geometric strain/gap/force criteria fire on their own
(option 3b, no Python release logic).

## 4. Prior art

| Approach | Idea | Fit here |
|---|---|---|
| **Virtual Node Algorithm** (Molino, Bao, Sifakis, Fedkiw 2004; arbitrary tet cutting, Sifakis 2007) | Duplicate the elements the cut crosses; insert *virtual nodes* at edge–cut intersections. | Exact + arbitrary, but **changes topology at runtime** → incompatible with this backend without implementing the rebuild path. |
| **IPC / C-IPC** (Li et al. 2020) | Barrier method over a *fixed* contact-primitive set with thickness/strain barriers. | Assumes fixed topology; combining runtime cutting with the barrier/CCD state is largely unpublished. Reinforces staying topology-free. |
| **Cohesive-zone / pre-fractured bonds** ("score and snap") | Breakable bonds/springs along predefined seams. | **Direct match** — libuipc's bonded-PT *is* a breakable-bond system. |

Links: [Virtual Node Algorithm](https://www.researchgate.net/publication/220184275_A_virtual_node_algorithm_for_changing_mesh_topology_during_simulation),
[GPU virtual node (OpenCL)](https://www.researchgate.net/publication/270826004_CPU-GPU_mixed_implementation_of_virtual_node_method_for_real-time_interactive_cutting_of_deformable_objects_using_OpenCL),
[Arbitrary tet cutting](https://www.researchgate.net/publication/220789197_Arbitrary_cutting_of_deformable_tetrahedralized_objects),
[C-IPC](https://ipc-sim.github.io/C-IPC/).

## 5. Cutting at an *arbitrary* position — the core trade-off

A single pre-scored row cuts only at that row. "Cut anywhere" means **pre-score at many
rows** (a perforation grid); resolution = perforation spacing. But a bond cannot perfectly
reproduce the continuous-FEM coupling it replaces:

- **Membrane** (in-plane stretch/tension) — a bond restores this well (pin the coincident
  rows → they resist separation).
- **Bending** (the `DiscreteShellBending` dihedral across the removed shared edge) — a bond
  does **not** reproduce it, and the two bond types err in opposite directions:
  - **Rigid distance-lock** = a stiff weld → fine perforation makes the tape **too stiff**
    at every seam (can't unwind/conform).
  - **Soft β-adhesion** = compliant/tunable, but **weaker** → may creep apart under load
    before you cut.

So there is an unavoidable **resolution ↔ fidelity ↔ cost** trade-off: finer perforation
→ closer to arbitrary, but more seam deviation from the continuous tape and more locks.

**Holding "as one tape" until cut:** give every perforation a *high* hold parameter (high
`release_force`, or high β) so nothing releases under normal loads; the blade *lowers* only
the crossed seams' `release_force` so they tear locally. The on/off is robust; the fidelity
of "one tape" is the approximation above.

## 6. Options, ranked

1. **Moderate perforation + tuned soft bonds** (simple, approximate; no backend change).
   Pre-split every few rows, soft-adhesion bonds tuned so membrane stiffness ≈ the tape's,
   high `release_force`, blade drops it to cut. Cut resolution = a few rows; membrane
   preserved; bending across seams approximate. Good for visuals.
2. **Runtime topology change (Virtual Node Algorithm)** — the *only* route to exact
   mechanics + truly arbitrary cuts. Requires implementing the stubbed
   `GlobalVertexManager::rebuild` / `GlobalSimplicialSurfaceManager::rebuild`, wiring
   FEM/ABD/contact `on_rebuild_scene`, and re-initializing IPC's CCD/barrier state.
   **Major backend feature (weeks), not "simple".**
3. **Demo-pragmatic: cut at the roll.** A real dispenser cuts at the pay-out point, not at
   an arbitrary point on the laid strip. Pre-score a few rows near the roll's finish and
   release on a scripted frame (or by the blade). Minimal; keeps the rest of the tape
   continuous FEM. **Recommended for finishing the carton seal.**

## 7. Recommended path

- **Carton seal demo:** option 3 — 2–3 pre-scored rows at the pay-out region, cut on a
  scripted frame or by a driven blade body. No fidelity trade-off elsewhere.
- **General reusable "cut anywhere":** option 1, and — to get *fine per-row* control
  without an `O(rows²)` contact tabular — add one small backend method,
  `RCCBondedPTStateAccessorFeature.release_keys(keys)`, that flags matching locks as
  force-released. The release-flag plumbing (`m_released_*` buffers,
  `RCCBondedPTReleaseContext`) already exists; it only lacks a Python-facing per-key entry
  point (~a small addition). Then one element carries all seams and the blade releases
  exactly the locks it crosses by key.
- **Only pursue option 2** if exact mechanics at arbitrary cut resolution is a hard
  requirement.

**Authoring note:** the pre-split is done at load time (before `world.init`), where the
frontend `SimplicialComplex` is mutable — duplicate the cut rows, rewrite the triangle
indices, register the strips as contact elements, and bond them. No backend change is
needed for authoring; only *blade-triggered per-lock release* wants `release_keys`.
