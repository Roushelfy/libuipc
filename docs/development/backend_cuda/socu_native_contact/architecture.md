# SOCU Native Contact Architecture

## Motivation

Legacy structured contact assembly is correct, but it spends hot-path time
rediscovering symbolic facts that usually do not change inside an ordinary
Newton solve. The native redesign separates stable symbolic mapping from numeric
Hessian evaluation and native matrix writes.

## Core Data Flow

```text
reporters
  -> source views
  -> symbolic side/program plan
  -> evaluator
  -> executor
  -> native matrix
```

This flow has three operating shapes:

```text
Correctness Bridge:
reporters -> source views -> symbolic side/program plan
          -> triplet_compat evaluator -> executor -> native matrix

Current Performance Target:
reporters -> source views -> cached symbolic side/program plan
          -> direct evaluator -> executor -> native matrix

Peak Performance Target:
reporters -> unified source views -> cached symbolic side/program plan
          -> fused direct eval+scatter kernels
          -> hot-block reduction / cache-hot replay
          -> native matrix / socu_native solve
```

## Component Responsibilities

| Component | Owns | Must Not Own |
| --- | --- | --- |
| Reporters | Active contacts, model parameters, stencil arrays, legacy triplet oracle | SOCU native matrix layout, block/lane writes, native cache keys |
| Source views | `source_id`, model, family, stencil view, scene/contact data handles | Reordering policy, native block classification, matrix writes |
| Symbolic side plan | Global vertex to native block/lane/writable/projection mapping | Contact Hessian values, per-solve numeric work |
| Symbolic program plan | Contact programs, source/local ids, write policy, buckets, hot-block metadata | IPC formula evaluation, structured sink fallback |
| Evaluator | Local contact Hessian values for one program | Matrix storage policy, source ordering mutation |
| Executor | Preclassified native matrix writes and hot-block reduction | Reporter traversal, scalar pair classification, old target rebuild |
| Native matrix | SOCU solver storage consumed by `socu_native` | Contact topology discovery or reporter ownership |

## Reporters

Reporters own the physical contact model and the current active contact lists.
Examples include simplex normal, simplex frictional, vertex-half-plane normal,
and vertex-half-plane frictional contact reporters.

They should expose source data and remain available as a correctness oracle in
compatibility modes. They should not know how SOCU native storage is organized.

## Source Views

Source views are the only thin interface between reporters and the native
contact pipeline. A source view assigns a dense `source_id` to each contact
model/family stream, such as `SimplexNormal/PT`, `SimplexNormal/EE`, or
`VertexHalfPlaneFrictional/PH`.

The same `source_id + local_contact_id` pair must identify the same contact in
all consumers:

- symbolic contact program builder;
- contact topology stamp;
- direct evaluator source table;
- `triplet_compat` source table;
- `direct_compare` oracle source table.

Until a single shared source enumeration helper exists, every repeated
enumeration must be covered by source-scan and contract tests.

## Symbolic Side Plan

The side plan caches stable native mapping facts:

- global vertex to native block and lane;
- writable and fixed state;
- ABD/FEM ownership and projection data;
- native descriptor and ordering epochs.

The side plan changes when ordering, descriptors, fixed mapping, or projection
data changes. It should not rebuild merely because Hessian values changed.

ABD side lanes have a non-negotiable projection contract. For an ABD local lane
`q` in `[0, 12)`, the side plan must store:

- `component = q < 3 ? q : (q - 3) / 3`;
- `weight = q < 3 ? 1 : x_bar((q - 3) % 3)`.

This is the native symbolic form of the legacy `J^T H J` projection. A builder
that stores `component = q` and `weight = 1` for all twelve ABD lanes is not
accepted, even if hand-written writer fixtures still pass.

## Symbolic Program Plan

The program plan maps active contacts to executable native write programs. It
owns:

- `source_id` and source-local contact ids;
- side ids for each stencil vertex;
- exact, diag, diag-lump, drop, and skipped policies;
- task buckets and hot-block metadata;
- source-to-program lookup tables.

The program plan changes when active contact topology, offband policy, source
layout, hot-block strategy, or the side-plan key changes.

## Evaluator Paths

`triplet_compat` first asks reporters to generate legacy Hessian triplets, then
uses those triplets as an evaluator source. It is a correctness bridge and
fallback, not the target performance path.

`direct` computes local IPC contact Hessians directly from source views and
scene data. This is the current performance target because production replay can
avoid reporter triplet materialization.

`direct_compare` runs direct evaluation and a triplet reference in the same
scene to measure mismatch counts and absolute error. It is a diagnostic gate,
not a production performance path.

`hybrid` is reserved for an explicit fallback policy: direct-capable programs
use direct evaluation, unsupported programs fall back to a counted compatibility
path, and reports update unsupported/fallback counters. Until that behavior is
implemented and tested, `hybrid` must not be treated as a performance-equivalent
alias for `direct`.

## Executor

The executor receives a program plan, an evaluator, and the native matrix view.
It performs only preclassified native writes. It must not call scalar
classification, `old_to_chain`, legacy target-table builders, or structured
contact sink adapters.

Executor buckets are the main control surface:

- exact native block writes;
- diagonal block writes;
- scalar diagonal compatibility writes;
- diag-lump writes;
- drop/no-op programs;
- hot-block owner-reduce or cached microblock handling.

## Peak Performance Shape

The current `direct evaluator + executor` split is the right high-performance
architecture because it removes legacy triplet materialization from the
production path. It is not the final performance ceiling.

The peak target is fused per-family or per-bucket direct eval+scatter kernels:

- evaluate the local Hessian and immediately scatter it while values are still
  in registers or shared memory;
- avoid a dense per-program Hessian global-memory buffer on the production
  path;
- use owner-reduce or cached microblock strategies for hot native blocks;
- keep cache-hot steady state to numeric replay only;
- optionally capture stable replay with CUDA Graph once symbolic shape and
  launch parameters are stable.

The fused form is future work. M6.7 and M8 should first make the current direct
path well-isolated, measurable, and safe to cut over.
