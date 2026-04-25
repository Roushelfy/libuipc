from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .artifacts import write_json
from .runner import load_timer_frames_json


CHART_PHASES = [
    "Rebuild Scene",
    "Frame Setup",
    "Predict Motion",
    "Adaptive Parameters",
    "Contact Detection",
    "DyTopo Effect",
    "Linear Build",
    "Linear Solve",
    "Line Search",
    "AL Active Set",
    "Update Velocity",
    "Bookkeeping",
]

PHASE_COLORS = {
    "Rebuild Scene": "#4C78A8",
    "Frame Setup": "#F58518",
    "Predict Motion": "#54A24B",
    "Adaptive Parameters": "#EECA3B",
    "Contact Detection": "#E45756",
    "DyTopo Effect": "#72B7B2",
    "Linear Build": "#B279A2",
    "Linear Solve": "#FF9DA6",
    "Line Search": "#9D755D",
    "AL Active Set": "#A0CBE8",
    "Update Velocity": "#8CD17D",
    "Bookkeeping": "#BAB0AC",
}

DIRECT_PHASE_TIMERS = {
    "Rebuild Scene": "Rebuild Scene",
    "Frame Setup": "Frame Setup",
    "Record Friction Candidates": "Frame Setup",
    "Clear External Forces": "Frame Setup",
    "Step Animation": "Frame Setup",
    "Compute External Force Accelerations": "Frame Setup",
    "Predict Motion": "Predict Motion",
    "Compute Adaptive Parameters": "Adaptive Parameters",
    "Detect DCD Candidates": "Contact Detection",
    "Detect Trajectory Candidates": "Contact Detection",
    "Filter Contact Candidates": "Contact Detection",
    "Filter CCD TOI": "Contact Detection",
    "Compute CFL Condition": "Contact Detection",
    "Compute DyTopo Effect": "DyTopo Effect",
    "Build Linear System": "Linear Build",
    "Solve Linear System": "Linear Solve",
    "FusedPCG": "Linear Solve",
    "PCG": "Linear Solve",
    "SpMV": "Linear Solve",
    "Apply Preconditioner": "Linear Solve",
    "Collect Vertex Displacements": "Linear Solve",
    "Compute Energy": "Line Search",
    "Energy Reporter: ABD": "Line Search",
    "Energy Reporter: FEM": "Line Search",
    "Energy Reporter: Contact": "Line Search",
    "Energy Reporter: Unclassified": "Line Search",
    "Energy Reporter: Other": "Line Search",
    "Linearize Contact Constraints": "AL Active Set",
    "Recover to Non-Penetrating Positions": "AL Active Set",
    "Advance Non-Penetrating Positions": "AL Active Set",
    "Update Velocity": "Update Velocity",
}

CONTAINER_RESIDUAL_PHASES = {
    "Pipeline": "Bookkeeping",
    "Simulation": "Frame Setup",
    "Newton Iteration": "Bookkeeping",
    "Solve Global Linear System": "Linear Solve",
    "Line Search": "Line Search",
    "Line Search Iteration": "Line Search",
}


def _iter_nodes(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield node
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            yield from _iter_nodes(child)


def _find_first(node: Dict[str, Any], name: str) -> Dict[str, Any] | None:
    for row in _iter_nodes(node):
        if row.get("name") == name:
            return row
    return None


def _duration_ms(node: Dict[str, Any] | None) -> float:
    if not isinstance(node, dict):
        return 0.0
    return float(node.get("duration", 0.0)) * 1000.0


def _clamp_nonnegative(value: float, *, eps: float = 1e-9) -> float:
    if value < 0.0 and abs(value) <= eps:
        return 0.0
    if value < 0.0:
        return 0.0
    return value


def _add_phase(phases: Dict[str, float], phase: str, value: float) -> None:
    phases[phase] = phases.get(phase, 0.0) + _clamp_nonnegative(value)


def _merge_phases(dst: Dict[str, float], src: Dict[str, float], scale: float = 1.0) -> None:
    for phase, value in src.items():
        _add_phase(dst, phase, value * scale)


def _allocate_chart_node(node: Dict[str, Any], warnings: List[str]) -> Dict[str, float]:
    name = node.get("name")
    duration_ms = _clamp_nonnegative(_duration_ms(node))
    children = [child for child in node.get("children", []) or [] if isinstance(child, dict)]

    phase = DIRECT_PHASE_TIMERS.get(str(name))
    if phase is not None:
        phases = {chart_phase: 0.0 for chart_phase in CHART_PHASES}
        _add_phase(phases, phase, duration_ms)
        return phases

    phases = {phase_name: 0.0 for phase_name in CHART_PHASES}
    child_phases = [_allocate_chart_node(child, warnings) for child in children]
    child_ms = sum(sum(child.values()) for child in child_phases)

    if child_ms > duration_ms + 1e-9:
        scale = 0.0 if child_ms <= 0.0 else duration_ms / child_ms
        for child in child_phases:
            _merge_phases(phases, child, scale)
        return phases

    for child in child_phases:
        _merge_phases(phases, child)

    residual_ms = duration_ms - child_ms
    if name in CONTAINER_RESIDUAL_PHASES:
        _add_phase(phases, CONTAINER_RESIDUAL_PHASES[str(name)], residual_ms)
    else:
        _add_phase(phases, "Bookkeeping", residual_ms)
    return phases


def extract_chart_frame_segments(frame: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, List[str]]:
    warnings: List[str] = []
    pipeline = _find_first(frame, "Pipeline")
    if pipeline is None:
        return None, ["missing Pipeline timer"]

    total_ms = _clamp_nonnegative(_duration_ms(pipeline))
    phases = _allocate_chart_node(pipeline, warnings)

    assigned_ms = sum(phases.values())
    delta_ms = total_ms - assigned_ms
    if delta_ms > 1e-6:
        _add_phase(phases, "Bookkeeping", delta_ms)
    elif delta_ms < -1e-3:
        warnings.append(
            f"assigned chart phases exceed Pipeline total by {-delta_ms:.6f} ms"
        )

    return {
        "total_ms_per_frame": total_ms,
        "phases": phases,
    }, warnings


def summarize_chart_timer_frames(timer_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    warnings: List[str] = []
    valid_segments: List[Dict[str, Any]] = []

    for frame_index, frame in enumerate(timer_frames):
        if not isinstance(frame, dict):
            warnings.append(f"frame {frame_index}: invalid timer frame root")
            continue
        segments, frame_warnings = extract_chart_frame_segments(frame)
        for warning in frame_warnings:
            warnings.append(f"frame {frame_index}: {warning}")
        if segments is not None:
            valid_segments.append(segments)

    if not valid_segments:
        return {
            "frame_count": len(timer_frames),
            "valid_frame_count": 0,
            "total_ms_per_frame": None,
            "phases": {phase: 0.0 for phase in CHART_PHASES},
            "warnings": warnings,
        }

    total_ms = sum(row["total_ms_per_frame"] for row in valid_segments) / len(valid_segments)
    phases = {
        phase: sum(row["phases"][phase] for row in valid_segments) / len(valid_segments)
        for phase in CHART_PHASES
    }
    return {
        "frame_count": len(timer_frames),
        "valid_frame_count": len(valid_segments),
        "total_ms_per_frame": total_ms,
        "phases": phases,
        "warnings": warnings,
    }


def _status_for_level(run_root: Path, asset: str, level: str) -> Tuple[str, Dict[str, Any] | None]:
    failure_path = run_root / "runs" / asset / level / "perf" / "failure.json"
    if failure_path.exists():
        import json

        return "failed", json.loads(failure_path.read_text(encoding="utf-8"))
    timer_path = run_root / "runs" / asset / level / "perf" / "timer_frames.json"
    if not timer_path.exists():
        return "missing", None
    return "ok", None


def build_chart_segments(run_root: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    levels = list(summary["run_meta"]["levels"])
    asset_charts: List[Dict[str, Any]] = []
    scenario_buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    warnings: List[str] = []

    for asset, asset_summary in summary.get("assets", {}).items():
        level_rows: List[Dict[str, Any]] = []
        first_level_payload = next(iter(asset_summary.get("levels", {}).values()), {})
        scenario = first_level_payload.get("scenario")
        scenario_family = first_level_payload.get("scenario_family")

        for level in levels:
            perf_dir = run_root / "runs" / asset / level / "perf"
            timer_path = perf_dir / "timer_frames.json"
            status, failure = _status_for_level(run_root, asset, level)
            row: Dict[str, Any] = {
                "level": level,
                "status": status,
                "total_ms_per_frame": None,
                "phases": {phase: 0.0 for phase in CHART_PHASES},
                "warning_count": 0,
            }

            if failure is not None:
                row["failure"] = failure

            if status == "ok":
                timer_frames = load_timer_frames_json(timer_path)
                segment_summary = summarize_chart_timer_frames(timer_frames)
                if segment_summary["valid_frame_count"] == 0 or segment_summary["total_ms_per_frame"] is None:
                    row["status"] = "missing"
                    row["warning_count"] = len(segment_summary["warnings"])
                    if segment_summary["warnings"]:
                        warnings.extend(
                            f"{asset}/{level}: {warning}" for warning in segment_summary["warnings"]
                        )
                else:
                    row["total_ms_per_frame"] = segment_summary["total_ms_per_frame"]
                    row["phases"] = dict(segment_summary["phases"])
                    row["warning_count"] = len(segment_summary["warnings"])
                    if segment_summary["warnings"]:
                        warnings.extend(
                            f"{asset}/{level}: {warning}" for warning in segment_summary["warnings"]
                        )
                    if scenario:
                        scenario_buckets.setdefault(scenario, {}).setdefault(level, []).append(row)

            level_rows.append(row)

        asset_charts.append(
            {
                "asset": asset,
                "scenario": scenario,
                "scenario_family": scenario_family,
                "levels": level_rows,
            }
        )

    scenario_charts: List[Dict[str, Any]] = []
    for scenario in sorted(scenario_buckets):
        levels_rows: List[Dict[str, Any]] = []
        for level in levels:
            samples = scenario_buckets[scenario].get(level, [])
            if not samples:
                levels_rows.append(
                    {
                        "level": level,
                        "status": "missing",
                        "asset_count": 0,
                        "total_ms_per_frame": None,
                        "phases": {phase: 0.0 for phase in CHART_PHASES},
                    }
                )
                continue

            total_ms = sum(sample["total_ms_per_frame"] or 0.0 for sample in samples) / len(samples)
            phases = {
                phase: sum(sample["phases"].get(phase, 0.0) for sample in samples) / len(samples)
                for phase in CHART_PHASES
            }
            levels_rows.append(
                {
                    "level": level,
                    "status": "ok",
                    "asset_count": len(samples),
                    "total_ms_per_frame": total_ms,
                    "phases": phases,
                }
            )

        scenario_charts.append(
            {
                "scenario": scenario,
                "levels": levels_rows,
            }
        )

    return {
        "phases": list(CHART_PHASES),
        "asset_charts": asset_charts,
        "scenario_charts": scenario_charts,
        "warnings": warnings,
    }


def render_chart_artifacts(run_root: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    reports_dir = run_root / "reports"
    charts_dir = reports_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_segments = build_chart_segments(run_root, summary)
    chart_segments_path = reports_dir / "chart_segments.json"
    write_json(chart_segments_path, chart_segments)

    warnings = list(chart_segments.get("warnings", []))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        warnings.append(
            "matplotlib is not available in the current interpreter; skipped SVG chart generation"
        )
        return {
            "chart_segments_json": str(chart_segments_path),
            "asset_charts": [],
            "scenario_charts": [],
            "warnings": warnings,
        }

    def plot_chart(title: str, level_rows: List[Dict[str, Any]], output_path: Path) -> None:
        x = list(range(len(level_rows)))
        labels = [row["level"] for row in level_rows]
        fig_width = max(7.0, 0.9 * len(level_rows) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, 4.8))

        bottoms = [0.0 for _ in level_rows]
        max_total = max(
            (row["total_ms_per_frame"] or 0.0) for row in level_rows if row.get("status") == "ok"
        ) if any(row.get("status") == "ok" for row in level_rows) else 0.0

        for phase in CHART_PHASES:
            heights = [
                row["phases"].get(phase, 0.0) if row.get("status") == "ok" else 0.0
                for row in level_rows
            ]
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                width=0.72,
                label=phase,
                color=PHASE_COLORS[phase],
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

        label_y = max_total * 0.03 if max_total > 0.0 else 0.5
        for idx, row in enumerate(level_rows):
            status = row.get("status")
            total = row.get("total_ms_per_frame")
            if status == "ok" and total is not None:
                ax.text(idx, total + label_y, f"{total:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(
                    idx,
                    label_y,
                    status or "missing",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                    color="#666666",
                )

        ax.set_title(title)
        ax.set_ylabel("ms / frame")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto"))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, format="svg")
        plt.close(fig)

    asset_artifacts: List[Dict[str, Any]] = []
    for chart in chart_segments["asset_charts"]:
        output_path = charts_dir / f"asset_{chart['asset']}.svg"
        plot_chart(f"Asset: {chart['asset']}", chart["levels"], output_path)
        asset_artifacts.append(
            {
                "asset": chart["asset"],
                "scenario": chart.get("scenario"),
                "scenario_family": chart.get("scenario_family"),
                "path": str(output_path),
                "levels": chart["levels"],
            }
        )

    scenario_artifacts: List[Dict[str, Any]] = []
    for chart in chart_segments["scenario_charts"]:
        output_path = charts_dir / f"scenario_{chart['scenario']}.svg"
        plot_chart(f"Scenario: {chart['scenario']}", chart["levels"], output_path)
        scenario_artifacts.append(
            {
                "scenario": chart["scenario"],
                "path": str(output_path),
                "levels": chart["levels"],
            }
        )

    return {
        "chart_segments_json": str(chart_segments_path),
        "asset_charts": asset_artifacts,
        "scenario_charts": scenario_artifacts,
        "warnings": warnings,
    }
