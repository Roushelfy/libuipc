from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.timers import load_frame_counter_values


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
