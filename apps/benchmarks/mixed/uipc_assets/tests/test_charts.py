from __future__ import annotations

import math
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.charts import CHART_PHASES, extract_chart_frame_segments


def _timer(name: str, ms: float, children: list[dict] | None = None) -> dict:
    return {
        "name": name,
        "duration": ms / 1000.0,
        "count": 1,
        "children": children or [],
    }


def test_chart_segments_are_exhaustive_without_other_phase() -> None:
    frame = _timer(
        "Pipeline",
        10.0,
        [
            _timer("Rebuild Scene", 1.0),
            _timer(
                "Simulation",
                9.0,
                [
                    _timer("Detect DCD Candidates", 2.0),
                    _timer(
                        "Compute DyTopo Effect",
                        3.0,
                        [
                            _timer("Assemble Dytopo Effect", 2.0),
                            _timer("Convert Dytopo Matrix", 0.5),
                            _timer("Distribute Dytopo Effect", 0.5),
                        ],
                    ),
                    _timer(
                        "Solve Global Linear System",
                        2.0,
                        [
                            _timer("Build Linear System", 0.8),
                            _timer("Solve Linear System", 1.0),
                        ],
                    ),
                    _timer(
                        "Line Search",
                        1.5,
                        [
                            _timer("Detect Trajectory Candidates", 0.2),
                            _timer("Filter CCD TOI", 0.1),
                            _timer("Compute Energy", 0.6),
                        ],
                    ),
                    _timer("Update Velocity", 0.5),
                ],
            ),
        ],
    )

    segments, warnings = extract_chart_frame_segments(frame)

    assert segments is not None
    assert warnings == []
    assert "Other" not in CHART_PHASES
    assert "Other" not in segments["phases"]
    assert math.isclose(sum(segments["phases"].values()), 10.0)
    assert math.isclose(segments["phases"]["DyTopo Effect"], 3.0)
    assert math.isclose(segments["phases"]["Contact Detection"], 2.3)
    assert math.isclose(segments["phases"]["Linear Build"], 0.8)
    assert math.isclose(segments["phases"]["Linear Solve"], 1.2)
    assert math.isclose(segments["phases"]["Line Search"], 1.2)


def test_chart_segments_map_old_uninstrumented_work_to_bookkeeping() -> None:
    frame = _timer(
        "Pipeline",
        10.0,
        [
            _timer("Rebuild Scene", 1.0),
            _timer(
                "Simulation",
                9.0,
                [
                    _timer("Newton Iteration", 6.0),
                    _timer("Update Velocity", 1.0),
                ],
            ),
        ],
    )

    segments, warnings = extract_chart_frame_segments(frame)

    assert segments is not None
    assert warnings == []
    assert "Other" not in segments["phases"]
    assert math.isclose(sum(segments["phases"].values()), 10.0)
    assert math.isclose(segments["phases"]["Frame Setup"], 2.0)
    assert math.isclose(segments["phases"]["Bookkeeping"], 6.0)


def test_chart_segments_cap_overlapping_children_to_parent_duration() -> None:
    frame = _timer(
        "Pipeline",
        10.0,
        [
            _timer("Rebuild Scene", 8.0),
            _timer("Update Velocity", 8.0),
        ],
    )

    segments, warnings = extract_chart_frame_segments(frame)

    assert segments is not None
    assert warnings == []
    assert math.isclose(sum(segments["phases"].values()), 10.0)
    assert math.isclose(segments["phases"]["Rebuild Scene"], 5.0)
    assert math.isclose(segments["phases"]["Update Velocity"], 5.0)
