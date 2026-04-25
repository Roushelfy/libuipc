from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.timers import CANONICAL_STAGES, load_frame_counter_values, load_frame_stage_values


def _frame_with_counter(name: str, count: int) -> dict:
    return {
        "name": "Pipeline",
        "duration": 0.001,
        "count": 1,
        "children": [
            {
                "name": "Solve Global Linear System",
                "duration": 0.001,
                "count": 1,
                "children": [
                    {
                        "name": name,
                        "duration": 0.0,
                        "count": count,
                        "children": [],
                    }
                ],
            }
        ],
    }


def test_load_frame_counter_values_prefers_explicit_pcg_counter() -> None:
    frame = _frame_with_counter("PCG Iteration Count", 17)
    frame["children"][0]["children"].append(
        {
            "name": "PCG Iteration",
            "duration": 0.0,
            "count": 3,
            "children": [],
        }
    )

    values = load_frame_counter_values(frame)

    assert values["pcg_iteration_count"] == 17


def test_load_frame_counter_values_falls_back_to_legacy_pcg_timer() -> None:
    frame = _frame_with_counter("PCG Iteration", 11)

    values = load_frame_counter_values(frame)

    assert values["pcg_iteration_count"] == 11


def test_canonical_stages_use_unclassified_names_instead_of_other() -> None:
    assert "Assemble Other" not in CANONICAL_STAGES
    assert "Energy Reporter: Other" not in CANONICAL_STAGES
    assert "Assemble Unclassified DyTopo" in CANONICAL_STAGES
    assert "Assemble Unclassified Linear Subsystem" in CANONICAL_STAGES
    assert "Energy Reporter: Unclassified" in CANONICAL_STAGES


def test_load_frame_stage_values_maps_legacy_other_by_context() -> None:
    frame = {
        "name": "Pipeline",
        "duration": 0.003,
        "children": [
            {
                "name": "Assemble Dytopo Effect",
                "duration": 0.001,
                "children": [
                    {"name": "Assemble Other", "duration": 0.001, "children": []},
                ],
            },
            {
                "name": "Assemble Linear System",
                "duration": 0.001,
                "children": [
                    {"name": "Assemble Other", "duration": 0.001, "children": []},
                ],
            },
            {
                "name": "Compute Energy",
                "duration": 0.001,
                "children": [
                    {"name": "Energy Reporter: Other", "duration": 0.001, "children": []},
                ],
            },
        ],
    }

    values = load_frame_stage_values(frame)

    assert values["Assemble Unclassified DyTopo"] == 1.0
    assert values["Assemble Unclassified Linear Subsystem"] == 1.0
    assert values["Energy Reporter: Unclassified"] == 1.0
