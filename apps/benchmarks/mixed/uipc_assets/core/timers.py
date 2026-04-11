from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List


CANONICAL_STAGES = [
    "Pipeline",
    "Rebuild Scene",
    "Simulation",
    "Predict Motion",
    "Detect DCD Candidates",
    "Detect Trajectory Candidates",
    "Filter CCD TOI",
    "Build Linear System",
    "Assemble Linear System",
    "Assemble ABD",
    "Assemble FEM",
    "Assemble Contact",
    "Assemble Other",
    "Convert Matrix",
    "Assemble Preconditioner",
    "Solve Global Linear System",
    "FusedPCG",
    "SpMV",
    "Apply Preconditioner",
    "Line Search",
    "Compute Energy",
    "Energy Reporter: ABD",
    "Energy Reporter: FEM",
    "Energy Reporter: Contact",
    "Energy Reporter: Other",
    "Update Velocity",
]

ITERATION_COUNTERS = {
    "newton_iteration_count": "Newton Iteration",
    "line_search_iteration_count": "Line Search Iteration",
    "pcg_iteration_count": "PCG Iteration",
}


def flatten_timer_tree(node: Dict[str, Any], out: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    rows = out if out is not None else []
    rows.append(node)
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            flatten_timer_tree(child, rows)
    return rows


def load_frame_stage_values(frame: Dict[str, Any], stages: Iterable[str] = CANONICAL_STAGES) -> Dict[str, float]:
    wanted = set(stages)
    values: Dict[str, float] = {stage: 0.0 for stage in wanted}
    for row in flatten_timer_tree(frame):
        name = row.get("name")
        if name in wanted:
            values[name] += float(row.get("duration", 0.0)) * 1000.0
    return values


def load_frame_counter_values(
    frame: Dict[str, Any],
    counters: Dict[str, str] = ITERATION_COUNTERS,
) -> Dict[str, int]:
    values: Dict[str, int] = {key: 0 for key in counters}
    wanted = {name: key for key, name in counters.items()}
    for row in flatten_timer_tree(frame):
        name = row.get("name")
        key = wanted.get(name)
        if key is not None:
            values[key] += int(row.get("count", 0))
    return values


def summarize_timer_frames(timer_frames: List[Dict[str, Any]], stages: Iterable[str] = CANONICAL_STAGES) -> Dict[str, Any]:
    per_frame = [load_frame_stage_values(frame, stages) for frame in timer_frames if isinstance(frame, dict)]
    rows: List[Dict[str, Any]] = []
    for stage in stages:
        samples = [frame[stage] for frame in per_frame]
        total_ms = sum(samples)
        mean_ms = statistics.fmean(samples) if samples else 0.0
        median_ms = statistics.median(samples) if samples else 0.0
        min_ms = min(samples) if samples else 0.0
        max_ms = max(samples) if samples else 0.0
        rows.append(
            {
                "stage": stage,
                "frame_count": len(samples),
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "total_ms": total_ms,
            }
        )

    total_pipeline_mean = next((row["mean_ms"] for row in rows if row["stage"] == "Pipeline"), 0.0)
    for row in rows:
        row["pct_of_frame"] = 0.0 if total_pipeline_mean == 0.0 else row["mean_ms"] / total_pipeline_mean * 100.0

    return {
        "frame_count": len(per_frame),
        "stages": rows,
    }


def summarize_iteration_counters(
    timer_frames: List[Dict[str, Any]],
    counters: Dict[str, str] = ITERATION_COUNTERS,
) -> Dict[str, Any]:
    per_frame = [load_frame_counter_values(frame, counters) for frame in timer_frames if isinstance(frame, dict)]
    rows: List[Dict[str, Any]] = []
    for key, timer_name in counters.items():
        samples = [int(frame[key]) for frame in per_frame]
        total = sum(samples)
        mean = statistics.fmean(samples) if samples else 0.0
        median = statistics.median(samples) if samples else 0.0
        min_count = min(samples) if samples else 0
        max_count = max(samples) if samples else 0
        rows.append(
            {
                "counter": key,
                "timer_name": timer_name,
                "frame_count": len(samples),
                "mean_count": mean,
                "median_count": median,
                "min_count": min_count,
                "max_count": max_count,
                "total_count": total,
            }
        )

    return {
        "frame_count": len(per_frame),
        "counters": rows,
    }


def stage_hotspots(stage_summary: Dict[str, Any], top_n: int = 8) -> List[Dict[str, Any]]:
    rows = list(stage_summary.get("stages", []))
    rows.sort(key=lambda item: item.get("mean_ms", 0.0), reverse=True)
    return rows[:top_n]
