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


def main() -> int:
    checks = [
        (
            "docs/roadmap.md",
            [
                ("Core Principle", "roadmap declares the governing tradeoff"),
                ("Phase 1: State Contract, CPU Oracle, And CUDA State Bridge (Current)", "roadmap identifies current phase"),
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
        "docs/conventions.md",
        "adhesion-off baseline",
        "scene gate requires a baseline that prevents false positives",
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
        "include/uipc/core/rcc_bonded_pt_state.h",
        "set_counters",
        "host state counter restore anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.h",
        "muda::DeviceBuffer<U64>",
        "CUDA locked-key bridge anchor",
    )
    expect_contains(
        "src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.cu",
        "copy_span_to_device",
        "CUDA state upload anchor",
    )
    expect_contains(
        "apps/tests/backends/cuda/rcc_bonded_pt_state_bridge.cu",
        "[rcc_bonded_pt][backend_state][cuda]",
        "CUDA state bridge fixture anchor",
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
        "src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu",
        "SoftVertexTriangleStitch",
        "virtual tet reference implementation",
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

    print("RCC adhesion acceleration source/doc gate passed.")
    print("Checked playbook skeleton, nav links, all-gates entry, and current source anchors.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
