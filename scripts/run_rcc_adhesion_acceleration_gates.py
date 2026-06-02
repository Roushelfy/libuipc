#!/usr/bin/env python3
"""Always-runnable source/doc gates for RCC adhesion acceleration."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_text(rel_path: str) -> str:
    path = ROOT / rel_path
    if not path.exists():
        raise AssertionError(f"missing file: {rel_path}")
    return path.read_text(encoding="utf-8")


def expect_contains(rel_path: str, needle: str, reason: str) -> None:
    text = read_text(rel_path)
    if needle not in text:
        raise AssertionError(f"{rel_path}: missing {needle!r} ({reason})")


def expect_regex(rel_path: str, pattern: str, reason: str) -> None:
    text = read_text(rel_path)
    if re.search(pattern, text, flags=re.MULTILINE) is None:
        raise AssertionError(f"{rel_path}: missing /{pattern}/ ({reason})")


def expect_not_contains(rel_path: str, needle: str, reason: str) -> None:
    text = read_text(rel_path)
    if needle in text:
        raise AssertionError(f"{rel_path}: contains forbidden {needle!r} ({reason})")


def expect_device_entry_stays_small() -> None:
    rel_path = "src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.h"
    text = read_text(rel_path)
    match = re.search(
        r"struct\s+RCCBondedPTDeviceEntry\s*\{(?P<body>.*?)\};",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{rel_path}: missing RCCBondedPTDeviceEntry")
    body = match.group("body")
    for forbidden in ["Matrix3x3", "Dm_inv", "rest_volume"]:
        if forbidden in body:
            raise AssertionError(
                f"{rel_path}: RCCBondedPTDeviceEntry contains {forbidden!r}; "
                "keep radix-sort values small and store rest-shape payloads in SoA buffers"
            )


def main() -> int:
    checks = [
        (
            "docs/roadmap.md",
            [
                ("Core Principle", "roadmap declares the governing tradeoff"),
                ("Current focus: **the benchmark", "roadmap identifies current gating milestone"),
                ("Phase 3: Bonded Virtual-Tet Reporter (Complete For ABD Ortho Scope)", "roadmap records completed ABD reporter scope"),
                ("Phase 4: Release, Fallback, And Scene Gate (Backend Implemented; Downstream Of Pre-CCD Filter)", "roadmap tracks release/lifecycle work downstream of the pre-CCD filter"),
                ("Blockers", "roadmap names current blockers"),
                ("Validation Gates", "roadmap lists runnable current gates"),
                ("Planned Gates", "roadmap separates future gates from current proof"),
                ("Playbook Compliance", "roadmap audits minimum standards"),
                ("Next Safe Task", "roadmap gives handoff-ready next task"),
            ],
        ),
        (
            "docs/architecture.md",
            [
                ("Motivation", "architecture explains why the subsystem exists"),
                ("Core Thesis", "architecture ties design to measurable objective"),
                ("Baseline Data Flow", "architecture ties into current lifecycle"),
                ("Target Data Flow", "architecture names the target data path"),
                ("Project Structure", "architecture names repo locations"),
                ("Ownership And Boundaries", "architecture names ownership boundaries"),
                ("Runtime Data Model", "architecture distinguishes key and topology data"),
            ],
        ),
        (
            "docs/conventions.md",
            [
                ("Hot-Path Rules", "conventions protect performance"),
                ("Data Layout Rules", "conventions define runtime buffers"),
                ("Naming Rules", "conventions define stable prefixes"),
                ("Validation Rules", "conventions define test standards"),
                ("Test Matrix", "conventions map invariants to gates"),
                ("Benchmark Protocol", "conventions define benchmark metadata"),
            ],
        ),
        (
            "docs/rcc_adhesion_acceleration.md",
            [
                ("Scope", "subsystem doc defines scope"),
                ("Algorithm Summary", "subsystem doc gives executable shape"),
                ("Lifecycle State Machine", "subsystem doc defines lock/release states"),
                ("Lock Gate", "subsystem doc defines lock policy"),
                ("Release Gate", "subsystem doc defines release policy"),
                ("Oracles", "subsystem doc defines numeric proof path"),
                ("Reports", "subsystem doc defines observable fields"),
            ],
        ),
        (
            "docs/development/rcc_adhesion_acceleration_journal.md",
            [
                ("2026-05-30", "journal has a dated entry"),
                ("2026-06-02 Strain Release And Beta Carry", "journal records the current release/beta-carry slice"),
                ("2026-06-02 Sticky And Policy Release", "journal records the sticky/policy release slice"),
                ("Source Observations", "journal records observed repository state"),
                ("Decisions", "journal records decisions"),
                ("Commands", "journal records command results"),
            ],
        ),
    ]

    for rel_path, needles in checks:
        for needle, reason in needles:
            expect_contains(rel_path, needle, reason)
        expect_not_contains(rel_path, "<new test>", "no template placeholders in committed docs")
        expect_not_contains(rel_path, "<scene command>", "no template placeholders in committed docs")
        expect_not_contains(rel_path, "<sandbox>", "no template placeholders in committed docs")

    for rel_path in [
        "docs/development/rcc_adhesion_acceleration.md",
        "docs/development/rcc_adhesion_acceleration_architecture.md",
        "docs/development/rcc_adhesion_acceleration_conventions.md",
    ]:
        if (ROOT / rel_path).exists():
            raise AssertionError(f"{rel_path}: obsolete parallel doc should not exist")

    expect_contains("docs/nav.md", "roadmap.md", "roadmap is linked from nav")
    expect_contains("docs/nav.md", "architecture.md", "architecture is linked from nav")
    expect_contains("docs/nav.md", "conventions.md", "conventions are linked from nav")
    expect_contains("docs/nav.md", "rcc_adhesion_acceleration.md", "subsystem doc is linked from nav")
    expect_contains("docs/nav.md", "rcc_adhesion_acceleration_journal.md", "journal is linked from nav")
    expect_contains(
        "scripts/run_rcc_adhesion_acceleration_all_gates.py",
        "scripts/run_rcc_adhesion_acceleration_gates.py",
        "all-gates entry runs source/doc gate",
    )
    expect_contains(
        "scripts/run_rcc_adhesion_acceleration_all_gates.py",
        "scripts/build_docs.py",
        "all-gates entry runs docs build now that doxygen is available",
    )
    expect_contains(
        "scripts/run_rcc_adhesion_acceleration_all_gates.py",
        "scripts/run_rcc_adhesion_acceleration_cuda_gates.py",
        "all-gates entry syntax-checks the local CUDA gate runner",
    )
    expect_contains(
        "scripts/run_rcc_adhesion_acceleration_cuda_gates.py",
        "gpu_sanity_check",
        "local CUDA gates include the GPU sanity regression test",
    )
    expect_contains(
        "scripts/run_rcc_adhesion_acceleration_cuda_gates.py",
        "bunny",
        "local CUDA gates include the bunny BVH regression case",
    )
    expect_contains(
        "docs/roadmap.md",
        "pt_lift_release",
        "planned scene gate names the PT-rich lifecycle scenario",
    )
    expect_not_contains(
        "docs/roadmap.md",
        "two_cube_lift_release",
        "planned scene gate should not be tied to the failed cube-cube fixture",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "rcc_adhesion_cloth_peel",
        "subsystem doc points at a PT-rich existing seed for the scene gate",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Legacy RCC baseline",
        "scene gate separates adhesion-off checks from bonded-vs-legacy comparison",
    )
    expect_contains(
        "docs/architecture.md",
        "ABD-style",
        "architecture requires ABD-style bonded virtual-tet energy",
    )
    expect_contains(
        "docs/architecture.md",
        "same step by high-kappa ABD-style",
        "architecture makes ABD energy a same-step replacement contract",
    )
    expect_contains(
        "docs/architecture.md",
        "Release And Beta Carry Contract",
        "architecture defines release/beta handoff contract",
    )
    expect_contains(
        "docs/roadmap.md",
        "rcc_bonded_pt_kappa",
        "roadmap tracks production bonded stiffness configuration",
    )
    expect_contains(
        "docs/roadmap.md",
        "kappa` must be positive",
        "roadmap rejects zero-stiffness bonded production mode",
    )
    expect_not_contains(
        "docs/roadmap.md",
        "production ABD material keys are still planned",
        "roadmap must not present implemented ABD config as future work",
    )
    expect_contains(
        "docs/conventions.md",
        "Stable Neo-Hookean prototype",
        "conventions distinguish prototype SNH from production ABD energy",
    )
    expect_contains(
        "docs/conventions.md",
        "ABD/SVTS boundary",
        "conventions lock down the SVTS rest-shape versus ABD energy boundary",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "abd_ortho",
        "subsystem doc names the target production energy model",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Hard production contract",
        "subsystem doc makes ABD energy a production precondition",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "rcc_bonded_pt_kappa >= 1e8",
        "subsystem doc records the initial high-stiffness gate value",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Current Implementation Snapshot",
        "subsystem doc has a concise current handoff table",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Release context is not a lock gate",
        "subsystem doc separates release context from lock-gate coverage",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Release ordering matters",
        "subsystem doc defines release before bonded assembly",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Current implementation status",
        "subsystem doc records release reason implementation status",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "released key/topology/beta/age/flag snapshots stay aligned",
        "subsystem doc requires full released snapshot alignment",
    )
    expect_contains(
        "docs/roadmap.md",
        "Scene-level release diagnostics are incomplete",
        "roadmap tracks release reporting as the next scene-facing gap",
    )
    expect_contains(
        "docs/roadmap.md",
        "Lock-gate parity contract",
        "roadmap keeps lock-gate parity as planned work",
    )
    expect_contains(
        "docs/roadmap.md",
        "Scene diagnostics/accessor contract",
        "roadmap keeps scene diagnostics as planned work",
    )
    expect_contains(
        "docs/conventions.md",
        "Lock-gate status and release-gate status are not conflated",
        "conventions prevent release fixtures from being treated as lock proof",
    )
    expect_contains(
        "docs/architecture.md",
        "release evaluator may consume overlapping data",
        "architecture separates classifier and release evaluator contracts",
    )
    expect_contains(
        "docs/architecture.md",
        "scene-accessible released snapshots still planned",
        "architecture records the missing release diagnostic layer",
    )
    expect_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_energy_model",
        "default config exposes production bonded energy model",
    )
    expect_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_kappa",
        "default config exposes production bonded stiffness",
    )
    expect_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_release_strain",
        "default config exposes strain-release threshold",
    )
    expect_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_release_gap",
        "default config exposes normal-gap release threshold",
    )
    expect_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_release_slip",
        "default config exposes tangential-slip release threshold",
    )
    expect_not_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_mu",
        "retired SNH bonded reporter config must not be in defaults",
    )
    expect_not_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_lambda",
        "retired SNH bonded reporter config must not be in defaults",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_virtual_tet_reporter.cu",
        "ortho_potential_function",
        "production bonded reporter uses ABD OrthoPotential",
    )
    expect_not_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_virtual_tet_reporter.cu",
        "soft_vertex_triangle_stitch_function",
        "production bonded reporter must not use the SVTS SNH function",
    )
    expect_contains(
        "docs/conventions.md",
        "adhesion-off baseline",
        "scene gate requires a baseline that prevents false positives",
    )
    expect_contains(
        "docs/conventions.md",
        "E/G/H oracle success does not by itself prove non-penetration",
        "conventions require scene-level no-penetration observation",
    )

    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "m_beta_PT",
        "current PT beta buffer anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "m_prev_keys_PT",
        "current sorted PT key snapshot anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "_evolve_beta_step_at_end",
        "current beta Phase A hook anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "RCCBetaEvolutionTimeIntegrator",
        "current end-of-step integrator anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "RCCBondedPTBetaCarryScratch",
        "RCC integration owns beta-carry scratch storage",
    )
    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "merge_released_beta",
        "RCC integration merges released beta back into PT persistence",
    )
    expect_contains(
        "src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu",
        "RCCBondedPTReleaseContext",
        "RCC integration routes sticky/policy release context into bonded owner",
    )
    expect_contains(
        "include/uipc/core/rcc_bonded_pt_state.h",
        "set_counters",
        "host state counter restore anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.h",
        "muda::DeviceBuffer<U64>",
        "CUDA locked-key bridge anchor",
    )
    expect_device_entry_stays_small()
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.cu",
        "copy_span_to_device",
        "CUDA state upload anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.h",
        "released_age()",
        "CUDA owner exposes released age snapshot for alignment tests",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.h",
        "m_released_age",
        "CUDA owner stores released age alongside key/topology/beta/flags",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "RCCBondedPTReleaseStrain",
        "CUDA owner evaluates strain release reason",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "RCCBondedPTReleaseGap",
        "CUDA owner evaluates normal-gap release reason",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "RCCBondedPTReleaseSlip",
        "CUDA owner evaluates tangential-slip release reason",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "RCCBondedPTReleaseStickySide",
        "CUDA owner evaluates sticky-side release reason",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "RCCBondedPTReleasePolicy",
        "CUDA owner evaluates disabled-policy release reason",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "rcc_bonded_pt_release_strain",
        "CUDA owner reads strain-release config",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "rcc_bonded_pt_release_gap",
        "CUDA owner reads normal-gap release config",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "rcc_bonded_pt_release_slip",
        "CUDA owner reads tangential-slip release config",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_system.cu",
        "m_released_age",
        "CUDA owner fills released age snapshot",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_beta_carry.h",
        "RCCBondedPTBetaCarryScratch",
        "beta-carry scratch helper is declared",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_beta_carry.cu",
        "merge_released_beta",
        "beta-carry helper implements released beta merge",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_state_bridge.cu",
        "[rcc_bonded_pt][backend_state][cuda]",
        "CUDA state bridge fixture anchor",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_system.cu",
        "[rcc_bonded_pt][release][cuda]",
        "backend fixture covers strain release lifecycle",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_system.cu",
        "[rcc_bonded_pt][release][gap][cuda]",
        "backend fixture covers normal-gap release lifecycle",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_system.cu",
        "[rcc_bonded_pt][release][slip][cuda]",
        "backend fixture covers tangential-slip release lifecycle",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_system.cu",
        "[rcc_bonded_pt][release][sticky][cuda]",
        "backend fixture covers sticky-side release lifecycle",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_system.cu",
        "[rcc_bonded_pt][release][policy][cuda]",
        "backend fixture covers disabled-policy release lifecycle",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_system.cu",
        "[rcc_bonded_pt][release][beta_carry][cuda]",
        "backend fixture covers released beta carry merge",
    )
    expect_contains(
        "src/backends/cuda/collision_detection/simplex_trajectory_filter.cu",
        "record_friction_candidates",
        "friction candidate copy anchor",
    )
    expect_contains(
        "src/backends/cuda/collision_detection/simplex_trajectory_filter.h",
        "friction_PTs()",
        "friction PT view anchor",
    )

    for rel_path in [
        "src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu",
        "src/backends/cuda/collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.cu",
        "src/backends/cuda/collision_detection/filters/info_stackless_bvh_v0_simplex_trajectory_filter.cu",
        "src/backends/cuda/collision_detection/filters/lbvh_simplex_trajectory_filter.cu",
    ]:
        expect_contains(
            rel_path,
            "point_triangle_ccd_broadphase",
            "PT CCD broadphase integration point",
        )
        expect_contains(
            rel_path,
            "rcc_bonded_pt_candidate_is_locked",
            "PT broadphase predicate skips locked pairs before CCD",
        )

    expect_contains(
        "src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu",
        "SoftVertexTriangleStitch",
        "rest-shape reference implementation",
    )
    expect_contains(
        "src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu",
        "min_separate_distance",
        "rest-shape thickness anchor",
    )
    expect_regex(
        "src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu",
        r"Dm\.inverse\(\)",
        "rest-shape inverse anchor",
    )
    expect_contains(
        "src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu",
        "make_spd",
        "SPD projection anchor",
    )
    expect_contains(
        "src/backends/cuda/inter_primitive_effect_system/inter_primitive_constitution_manager.h",
        "EnergyComponentFlags::Complement",
        "complement energy ownership anchor",
    )

    # Re-sequencing and review-hardening anchors (P0-A/P0-B/P1-C/P1-D).
    expect_contains(
        "docs/roadmap.md",
        "Pre-CCD Filter Implemented",
        "roadmap marks the pre-CCD filter as implemented and default-off",
    )
    expect_contains(
        "docs/roadmap.md",
        "zero CCD savings",
        "roadmap states plainly that there are no CCD savings while skip is default-off",
    )
    expect_contains(
        "docs/architecture.md",
        "CCD Removal Precondition",
        "architecture defines the precondition for removing locked PTs from CCD",
    )
    expect_contains(
        "docs/architecture.md",
        "shape-preservation term, not a non-penetration barrier",
        "architecture states the reflection-invariance / no-barrier invariant",
    )
    expect_contains(
        "docs/rcc_adhesion_acceleration.md",
        "Beta While Locked",
        "subsystem doc documents the beta-freeze-while-locked approximation",
    )
    expect_contains(
        "docs/conventions.md",
        "steady-state early-out",
        "conventions require a producer steady-state early-out",
    )
    expect_contains(
        "docs/conventions.md",
        "negative threshold sentinel",
        "conventions require a negative disabled-release sentinel, not 1e30",
    )
    expect_contains(
        "docs/conventions.md",
        "rcc_bonded_pt_producer_ms",
        "benchmark protocol times the end-of-step producer",
    )

    # Pre-CCD filter implementation anchors (default-off skip mechanism).
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_lookup.h",
        "rcc_bonded_pt_candidate_is_locked",
        "shared pre-CCD broadphase membership helper exists",
    )
    expect_contains(
        "src/core/core/scene_default_config.cpp",
        "rcc_bonded_pt_skip_ccd",
        "default config exposes the default-off pre-CCD skip flag",
    )
    expect_contains(
        "src/backends/cuda/collision_detection/simplex_trajectory_filter.h",
        "rcc_bonded_pt_skip_ccd",
        "simplex filter carries the pre-CCD skip flag",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_lookup.cu",
        "[rcc_bonded_pt][filter][ccd]",
        "pre-CCD filter membership fixture exists",
    )
    expect_contains(
        "apps/tests/sim_case/rcc_adhesion_lift_release_gate.cpp",
        "[rcc_bonded_pt][scene][pt_lift_release]",
        "bonded-mode no-penetration scene gate exists",
    )
    expect_contains(
        "apps/tests/sim_case/rcc_adhesion_lift_release_gate.cpp",
        "rcc_bonded_pt_skip_ccd",
        "scene gate enables the pre-CCD skip it is validating",
    )

    print("RCC adhesion acceleration source/doc gate passed.")
    print("Checked playbook skeleton, nav links, all-gates entry, and current source anchors.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
