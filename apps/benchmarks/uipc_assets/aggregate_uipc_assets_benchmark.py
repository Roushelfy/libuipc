#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uipc_assets_bench_common import (
    DEFAULT_PERF_WARNING_PCT,
    REPO_ID,
    avg_ms_per_frame,
    collect_error_metrics,
    default_manifest_path,
    enabled_scenes,
    load_benchmark_meta,
    load_manifest,
    load_timer_frames,
    manifest_hash,
    ordered_levels,
    read_json,
    timer_hotspots,
    warning_for_quality,
    write_json,
)


def _mode_dir(run_root: Path, level: str, scene_name: str, mode: str) -> Path:
    return run_root / level / scene_name / mode


def _load_worker_result(result_dir: Path) -> Dict[str, Any]:
    result_file = result_dir / "worker_result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing worker_result.json: {result_file}")
    return read_json(result_file)


def aggregate_run(run_root: Path, manifest_path: Path, out_dir: Path) -> Dict[str, Any]:
    specs = enabled_scenes(load_manifest(manifest_path))
    manifest_digest = manifest_hash(specs)
    dataset_state = {}
    state_file = run_root / "dataset_state.json"
    if state_file.exists():
        dataset_state = read_json(state_file)

    summary: Dict[str, Any] = {
        "repo_id": REPO_ID,
        "remote_sha": dataset_state.get("remote_sha"),
        "runtime_versions": dataset_state.get("runtime_versions", {}),
        "manifest_hash": manifest_digest,
        "run_root": str(run_root),
        "scene_count": len(specs),
        "levels": [],
        "scenes": {},
        "warnings": [],
    }

    detected_levels = ordered_levels(
        entry.name
        for entry in run_root.iterdir()
        if entry.is_dir() and entry.name != "__pycache__"
    )
    compare_levels = [level for level in detected_levels if level != "fp64"]
    summary["levels"] = ["fp64", *compare_levels] if "fp64" in detected_levels else compare_levels

    for spec in specs:
        scene_summary: Dict[str, Any] = {
            "manifest": spec.to_json(),
            "levels": {},
            "comparison": {},
            "warnings": [],
        }
        try:
            for level in summary["levels"]:
                result_dir = _mode_dir(run_root, level, spec.name, "perf")
                worker = _load_worker_result(result_dir)
                benchmark = load_benchmark_meta(Path(worker["benchmark_json"]) if worker.get("benchmark_json") else result_dir)
                timers = load_timer_frames(Path(worker["timer_frames_json"]) if worker.get("timer_frames_json") else result_dir)
                scene_summary["levels"][level] = {
                    "worker": worker,
                    "benchmark": benchmark,
                    "avg_ms_per_frame": avg_ms_per_frame(benchmark),
                    "timer_hotspots": timer_hotspots(timers),
                }

            if spec.quality_enabled:
                ref_dir = _mode_dir(run_root, "fp64", spec.name, "quality_reference")
                scene_summary["levels"]["fp64"]["quality_reference"] = _load_worker_result(ref_dir)
                for level in compare_levels:
                    cmp_dir = _mode_dir(run_root, level, spec.name, "quality_compare")
                    cmp_worker = _load_worker_result(cmp_dir)
                    error_file = Path(cmp_worker["error_jsonl"])
                    quality = collect_error_metrics(error_file)
                    scene_summary["levels"][level]["quality_compare"] = cmp_worker
                    scene_summary["comparison"].setdefault(level, {})
                    scene_summary["comparison"][level]["quality"] = {
                        "rel_l2_max": quality["rel_l2_max"],
                        "abs_linf_max": quality["abs_linf_max"],
                        "nan_inf_count": quality["nan_inf_count"],
                        "record_count": quality["record_count"],
                        "error_jsonl": str(error_file),
                    }
                    for warning in warning_for_quality(quality):
                        scene_summary["warnings"].append(f"{level}: {warning}")

            fp64_ms = scene_summary["levels"]["fp64"]["avg_ms_per_frame"]
            perf_threshold = spec.perf_warning_pct or DEFAULT_PERF_WARNING_PCT
            for level in compare_levels:
                level_ms = scene_summary["levels"][level]["avg_ms_per_frame"]
                delta_pct = None
                if fp64_ms not in (None, 0):
                    delta_pct = (level_ms - fp64_ms) / fp64_ms * 100.0
                scene_summary["comparison"].setdefault(level, {})
                scene_summary["comparison"][level]["performance"] = {
                    "fp64_ms_per_frame": fp64_ms,
                    "level_ms_per_frame": level_ms,
                    "delta_pct": delta_pct,
                }
                if delta_pct is not None and delta_pct > perf_threshold:
                    scene_summary["warnings"].append(
                        f"{level}: slower than fp64 by {delta_pct:.2f}% (> {perf_threshold:.1f}%)"
                    )
        except Exception as e:
            scene_summary["error"] = str(e)
            scene_summary["warnings"].append(f"scene aggregation failed: {e}")

        summary["scenes"][spec.name] = scene_summary
        for warning in scene_summary["warnings"]:
            summary["warnings"].append(f"{spec.name}: {warning}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# UIPC Assets Benchmark Summary")
    lines.append("")
    lines.append(f"- `repo_id`: `{summary.get('repo_id')}`")
    lines.append(f"- `remote_sha`: `{summary.get('remote_sha')}`")
    runtime_versions = summary.get("runtime_versions", {})
    if runtime_versions:
        lines.append(f"- `uipc`: `{runtime_versions.get('uipc')}`")
        lines.append(f"- `huggingface_hub`: `{runtime_versions.get('huggingface_hub')}`")
    lines.append(f"- `manifest_hash`: `{summary.get('manifest_hash')}`")
    lines.append(f"- `scene_count`: `{summary.get('scene_count')}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Scene | Level | Tags | FP64 ms/frame | Level ms/frame | Delta % | rel_l2.max | abs_linf.max | nan_inf | Warnings |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    levels = summary.get("levels", [])
    compare_levels = [level for level in levels if level != "fp64"]
    for scene_name, scene in summary["scenes"].items():
        for level in compare_levels:
            perf = scene.get("comparison", {}).get(level, {}).get("performance", {})
            quality = scene.get("comparison", {}).get(level, {}).get("quality", {})
            delta = perf.get("delta_pct")
            level_warnings = [w for w in scene["warnings"] if w.startswith(f"{level}:")]
            lines.append(
                "| {scene} | {level} | {tags} | {fp64} | {lvl_ms} | {delta} | {rel_l2} | {abs_linf} | {nan_inf} | {warn} |".format(
                    scene=scene_name,
                    level=level,
                    tags=", ".join(scene["manifest"]["tags"]),
                    fp64="n/a" if perf.get("fp64_ms_per_frame") is None else f"{perf['fp64_ms_per_frame']:.3f}",
                    lvl_ms="n/a" if perf.get("level_ms_per_frame") is None else f"{perf['level_ms_per_frame']:.3f}",
                    delta="n/a" if delta is None else f"{delta:.2f}",
                    rel_l2="n/a" if not quality else f"{quality['rel_l2_max']:.6e}",
                    abs_linf="n/a" if not quality else f"{quality['abs_linf_max']:.6e}",
                    nan_inf="n/a" if not quality else str(quality["nan_inf_count"]),
                    warn="; ".join(level_warnings) if level_warnings else "none",
                )
            )

    lines.append("")
    lines.append("## Timer Hotspots")
    lines.append("")
    for scene_name, scene in summary["scenes"].items():
        lines.append(f"### `{scene_name}`")
        for level in summary.get("levels", []):
            lines.append(f"- `{level}`")
            hotspots = scene.get("levels", {}).get(level, {}).get("timer_hotspots", [])
            if not hotspots:
                lines.append("  - none")
                continue
            for row in hotspots:
                lines.append(f"  - `{row['name']}`: {row['avg_ms']:.3f} ms avg")
        lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if not summary["warnings"]:
        lines.append("- none")
    else:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate dataset-driven uipc-assets benchmark results.")
    parser.add_argument("--run_root", required=True, type=Path, help="run root produced by run_uipc_assets_benchmark.py")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest file (default: run_root/manifest_snapshot.json)")
    parser.add_argument("--out_dir", type=Path, default=None, help="summary output dir (default: run_root)")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve() if args.out_dir else args.run_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else (args.run_root.resolve() / "manifest_snapshot.json")
    if not manifest_path.exists():
        manifest_path = default_manifest_path()
    aggregate_run(args.run_root.resolve(), manifest_path.resolve(), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
