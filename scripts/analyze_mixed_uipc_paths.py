from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT / "apps" / "benchmarks" / "mixed"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.artifacts import read_json, write_json
from uipc_assets.core.manifest import LEVEL_ORDER, load_manifest
from uipc_assets.core.report_schema import build_summary_payload, collect_report_data


TAG_SLICES = ("animated", "rotating_motor", "contains_fem", "contains_particle")
SCENARIO_ORDER = ("abd", "fem", "coupling", "particle")
BUCKET_ORDER = ("low", "mid", "high")
ADVISORY_BUCKET_ORDER = ("Line Search",)
ITERATION_COUNTER_ORDER = (
    "newton_iteration_count",
    "pcg_iteration_count",
    "line_search_iteration_count",
)
SOLVER_STAGES = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze mixed UIPC path benchmark runs and generate deep-dive artifacts."
    )
    parser.add_argument("--run-root", required=True, help="Primary run root.")
    parser.add_argument(
        "--supplemental-run-root",
        default="",
        help="Optional supplemental run root used only for appendix charts.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to <run-root>/reports/deep_analysis.",
    )
    parser.add_argument(
        "--min-support-for-claim",
        type=int,
        default=3,
        help="Minimum common stable support for dominance or recommendation claims.",
    )
    parser.add_argument(
        "--line-search-policy",
        choices=("advisory", "failure"),
        default="advisory",
        help="Treat line-search failures as advisory-only or hard failures.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(max(q, 0.0), 1.0)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def _fmt_float(value: float | None, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.1f}%"


def _unique_join(values: Iterable[Any]) -> str:
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in ordered:
            continue
        ordered.append(text)
    return ";".join(ordered)


def _ordered_levels(levels: Iterable[str]) -> List[str]:
    rank = {name: idx for idx, name in enumerate(LEVEL_ORDER)}
    return sorted(dict.fromkeys(levels), key=lambda name: (rank.get(name, len(rank)), name))


def _read_json_safe(path: Path) -> Dict[str, Any] | List[Any] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _frame_ms_from_benchmark(bench: Dict[str, Any] | None) -> float | None:
    if not isinstance(bench, dict):
        return None
    wall = _safe_float(bench.get("wall_time"))
    frames = _safe_float(bench.get("num_frames"))
    if wall is None or frames is None or frames <= 0.0:
        return None
    return wall / frames * 1000.0


def _build_summary_from_disk(run_root: Path) -> Dict[str, Any]:
    run_meta = read_json(run_root / "run_meta.json")
    selection = read_json(run_root / "selection.json")
    assets = [row["name"] for row in selection.get("assets", [])]
    levels = _ordered_levels(run_meta["levels"])
    compare_levels = [level for level in levels if level != "fp64"]
    summary: Dict[str, Any] = {
        "run_meta": run_meta,
        "assets": {},
        "pipeline_overview": [],
        "scenario_overview": [],
        "scenario_family_overview": [],
        "iteration_overview": [],
        "quality_rows": [],
        "visual_rows": [],
        "chart_artifacts": {},
        "chart_warnings": [],
        "failures": [],
    }
    for asset in assets:
        asset_levels: Dict[str, Dict[str, Any]] = {}
        fp64_stage_summary = _read_json_safe(run_root / "runs" / asset / "fp64" / "perf" / "stage_summary.json")
        fp64_bench = _read_json_safe(run_root / "runs" / asset / "fp64" / "perf" / "benchmark.json")
        fp64_iter = _read_json_safe(run_root / "runs" / asset / "fp64" / "perf" / "iteration_summary.json")
        fp64_frame_ms = _frame_ms_from_benchmark(fp64_bench if isinstance(fp64_bench, dict) else None)
        fp64_stage_rows = {
            row.get("stage"): row
            for row in (fp64_stage_summary or {}).get("stages", [])
            if isinstance(row, dict)
        }
        fp64_iter_rows = {
            row.get("counter"): row
            for row in (fp64_iter or {}).get("counters", [])
            if isinstance(row, dict)
        }
        for level in levels:
            perf_dir = run_root / "runs" / asset / level / "perf"
            quality_dir = run_root / "runs" / asset / level / "quality"
            bench = _read_json_safe(perf_dir / "benchmark.json")
            stage_summary = _read_json_safe(perf_dir / "stage_summary.json")
            iteration_summary = _read_json_safe(perf_dir / "iteration_summary.json")
            quality_metrics = _read_json_safe(quality_dir / "quality_metrics.json")
            asset_levels[level] = {
                "frame_ms": _frame_ms_from_benchmark(bench if isinstance(bench, dict) else None),
                "stage_summary": stage_summary,
                "iteration_summary": iteration_summary,
                "quality_metrics": quality_metrics,
                "visual_manifest": None,
            }
            if level != "fp64" and isinstance(quality_metrics, dict):
                summary["quality_rows"].append(
                    {
                        "asset": asset,
                        "level": level,
                        **quality_metrics,
                    }
                )
        summary["assets"][asset] = {"levels": asset_levels, "comparisons": {}}
        for level in compare_levels:
            level_stage_summary = _read_json_safe(
                run_root / "runs" / asset / level / "perf" / "stage_summary.json"
            )
            level_stage_rows = {
                row.get("stage"): row
                for row in (level_stage_summary or {}).get("stages", [])
                if isinstance(row, dict)
            }
            for stage in sorted(set(fp64_stage_rows) | set(level_stage_rows)):
                fp64_stage_ms = _safe_float((fp64_stage_rows.get(stage) or {}).get("mean_ms")) or 0.0
                level_stage_ms = _safe_float((level_stage_rows.get(stage) or {}).get("mean_ms")) or 0.0
                delta_ms = level_stage_ms - fp64_stage_ms
                delta_pct = None if fp64_stage_ms == 0.0 else delta_ms / fp64_stage_ms * 100.0
                summary["pipeline_overview"].append(
                    {
                        "asset": asset,
                        "level": level,
                        "stage": stage,
                        "fp64_ms_per_frame": fp64_stage_ms,
                        "level_ms_per_frame": level_stage_ms,
                        "delta_ms": delta_ms,
                        "delta_pct": delta_pct,
                    }
                )
            level_iter_summary = _read_json_safe(
                run_root / "runs" / asset / level / "perf" / "iteration_summary.json"
            )
            level_iter_rows = {
                row.get("counter"): row
                for row in (level_iter_summary or {}).get("counters", [])
                if isinstance(row, dict)
            }
            for counter in sorted(set(fp64_iter_rows) | set(level_iter_rows)):
                fp64_mean = _safe_float((fp64_iter_rows.get(counter) or {}).get("mean_count")) or 0.0
                level_mean = _safe_float((level_iter_rows.get(counter) or {}).get("mean_count")) or 0.0
                delta_count = level_mean - fp64_mean
                delta_pct = None if fp64_mean == 0.0 else delta_count / fp64_mean * 100.0
                summary["iteration_overview"].append(
                    {
                        "asset": asset,
                        "level": level,
                        "counter": counter,
                        "fp64_mean_count": fp64_mean,
                        "level_mean_count": level_mean,
                        "delta_count": delta_count,
                        "delta_pct": delta_pct,
                    }
                )
    return summary


def _load_summary(run_root: Path) -> Dict[str, Any]:
    summary_path = run_root / "reports" / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    try:
        return build_summary_payload(collect_report_data(run_root))
    except Exception:
        return _build_summary_from_disk(run_root)


def _load_full_catalog() -> Dict[str, Dict[str, Any]]:
    manifest_path = (
        REPO_ROOT / "apps" / "benchmarks" / "mixed" / "uipc_assets" / "manifests" / "full.json"
    )
    rows = load_manifest(manifest_path)
    catalog: Dict[str, Dict[str, Any]] = {}
    for spec in rows:
        catalog[spec.name] = spec.to_json()
    return catalog


def _load_asset_metadata(run_root: Path, full_catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    snapshot_path = run_root / "assets_snapshot.json"
    selection_path = run_root / "selection.json"
    if snapshot_path.exists():
        source_rows = read_json(snapshot_path)
    elif selection_path.exists():
        source_rows = read_json(selection_path).get("assets", [])
    else:
        source_rows = []
    merged: Dict[str, Dict[str, Any]] = {}
    for row in source_rows:
        name = row["name"]
        merged_row = dict(row)
        full_row = full_catalog.get(name)
        if full_row:
            merged_row["scenario"] = full_row.get("scenario", "")
            merged_row["scenario_family"] = full_row.get("scenario_family", "")
            merged_row["tags"] = list(full_row.get("tags", []))
            merged_row["notes"] = full_row.get("notes", merged_row.get("notes", ""))
            merged_row["contact_enabled"] = full_row.get(
                "contact_enabled", merged_row.get("contact_enabled")
            )
        merged_row.setdefault("enabled", True)
        merged_row.setdefault("quality_enabled", True)
        merged_row.setdefault("frames_perf", 0)
        merged_row.setdefault("frames_quality", 0)
        merged_row.setdefault("scenario", "")
        merged_row.setdefault("scenario_family", "")
        merged_row.setdefault("tags", [])
        merged_row.setdefault("notes", "")
        merged_row.setdefault("config_overrides", {})
        merged[name] = merged_row
    return merged


def _failure_bucket(
    stage: str | None, error: str | None = None, reason_code: str | None = None
) -> str:
    stage_l = (stage or "").strip().lower()
    error_l = (error or "").strip().lower()
    reason_l = (reason_code or "").strip().lower()
    if "search direction" in stage_l or "search_direction_invalid" in error_l:
        return "search direction"
    if "line search" in stage_l:
        return "Line Search"
    if reason_l:
        return f"worker:{_sanitize_token(reason_l).strip('_') or 'unknown'}"
    if stage_l:
        return f"worker:{_sanitize_token(stage_l).strip('_') or 'unknown'}"
    return "worker:unknown"


def _scan_run_status(
    run_root: Path, asset_names: Iterable[str], levels: Sequence[str]
) -> tuple[Dict[tuple[str, str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    status: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    for asset in asset_names:
        for level in levels:
            for mode in ("perf", "quality"):
                output_dir = run_root / "runs" / asset / level / mode
                result_path = output_dir / "worker_result.json"
                failure_path = output_dir / "failure.json"
                worker_result = read_json(result_path) if result_path.exists() else None
                failure = read_json(failure_path) if failure_path.exists() else None
                if failure:
                    failure = dict(failure)
                    failure["failure_bucket"] = _failure_bucket(
                        failure.get("stage"), failure.get("error"), failure.get("reason_code")
                    )
                    failures.append(failure)
                status[(asset, level, mode)] = {
                    "worker_result": worker_result,
                    "failure": failure,
                    "exists": output_dir.exists(),
                }
    return status, failures


def _normalize_summary_rows(
    rows: Iterable[Dict[str, Any]], asset_meta: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        normalized_row = dict(row)
        name = row.get("asset")
        meta = asset_meta.get(name or "", {})
        if meta:
            normalized_row["scenario"] = meta.get("scenario", normalized_row.get("scenario"))
            normalized_row["scenario_family"] = meta.get(
                "scenario_family", normalized_row.get("scenario_family")
            )
            normalized_row["tags"] = list(meta.get("tags", []))
        else:
            normalized_row.setdefault("tags", [])
        normalized.append(normalized_row)
    return normalized


def _index_run_data(run_data: Dict[str, Any]) -> None:
    summary = run_data["summary"]
    asset_meta = run_data["asset_meta"]
    stage_rows = _normalize_summary_rows(summary.get("pipeline_overview", []), asset_meta)
    iter_rows = _normalize_summary_rows(summary.get("iteration_overview", []), asset_meta)
    quality_rows = _normalize_summary_rows(summary.get("quality_rows", []), asset_meta)
    run_data["stage_rows"] = stage_rows
    run_data["iter_rows"] = iter_rows
    run_data["quality_rows"] = quality_rows

    stage_index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    pipeline_index: Dict[tuple[str, str], Dict[str, Any]] = {}
    fp64_pipeline_ms: Dict[str, float] = {}
    for row in stage_rows:
        asset = row["asset"]
        level = row["level"]
        stage = row["stage"]
        stage_index[(asset, level, stage)] = row
        if stage == "Pipeline":
            pipeline_index[(asset, level)] = row
            fp64_value = _safe_float(row.get("fp64_ms_per_frame"))
            if fp64_value is not None:
                fp64_pipeline_ms[asset] = fp64_value

    iter_index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    fp64_counter_index: Dict[tuple[str, str], float] = {}
    for row in iter_rows:
        asset = row["asset"]
        level = row["level"]
        counter = row["counter"]
        iter_index[(asset, level, counter)] = row
        fp64_value = _safe_float(row.get("fp64_mean_count"))
        if fp64_value is not None:
            fp64_counter_index[(asset, counter)] = fp64_value

    quality_index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in quality_rows:
        quality_index[(row["asset"], row["level"])] = row

    stage_summary_index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    iteration_summary_index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    fp64_frame_ms: Dict[str, float | None] = {}
    for asset, asset_summary in summary.get("assets", {}).items():
        levels_payload = (asset_summary.get("levels") or {})
        for level, payload in levels_payload.items():
            stage_summary = payload.get("stage_summary") or {}
            for row in stage_summary.get("stages", []):
                if isinstance(row, dict):
                    stage_summary_index[(asset, level, row.get("stage"))] = row
            iteration_summary = payload.get("iteration_summary") or {}
            for row in iteration_summary.get("counters", []):
                if isinstance(row, dict):
                    iteration_summary_index[(asset, level, row.get("counter"))] = row
        fp64_frame_ms[asset] = _safe_float(
            (levels_payload.get("fp64") or {}).get("frame_ms")
        )
        if asset not in fp64_pipeline_ms and fp64_frame_ms[asset] is not None:
            fp64_pipeline_ms[asset] = fp64_frame_ms[asset]

    run_data["stage_index"] = stage_index
    run_data["pipeline_index"] = pipeline_index
    run_data["fp64_pipeline_ms"] = fp64_pipeline_ms
    run_data["iter_index"] = iter_index
    run_data["fp64_counter_index"] = fp64_counter_index
    run_data["quality_index"] = quality_index
    run_data["fp64_frame_ms"] = fp64_frame_ms
    run_data["stage_summary_index"] = stage_summary_index
    run_data["iteration_summary_index"] = iteration_summary_index


def _load_run(run_root: Path, full_catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary = _load_summary(run_root)
    levels = _ordered_levels(summary["run_meta"]["levels"])
    compare_levels = [level for level in levels if level != "fp64"]
    asset_meta = _load_asset_metadata(run_root, full_catalog)
    if not asset_meta:
        for asset in summary.get("assets", {}):
            full_row = full_catalog.get(asset, {"name": asset})
            asset_meta[asset] = dict(full_row)
            asset_meta[asset].setdefault("quality_enabled", True)
            asset_meta[asset].setdefault("tags", [])
    status, failure_rows = _scan_run_status(run_root, asset_meta.keys(), levels)
    run_data = {
        "run_root": run_root,
        "summary": summary,
        "levels": levels,
        "compare_levels": compare_levels,
        "asset_meta": asset_meta,
        "status": status,
        "failures": failure_rows,
    }
    _index_run_data(run_data)
    return run_data


def _assign_terciles(values_by_asset: Dict[str, float]) -> Dict[str, str]:
    items = sorted(values_by_asset.items(), key=lambda item: (item[1], item[0]))
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: "mid"}
    buckets: Dict[str, str] = {}
    total = len(items)
    for idx, (asset, _value) in enumerate(items):
        bucket_idx = min(2, int(idx * 3 / total))
        buckets[asset] = BUCKET_ORDER[bucket_idx]
    return buckets


def _derive_asset_buckets(primary_run: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, str]]:
    fp64_cost_values: Dict[str, float] = {}
    fp64_newton_values: Dict[str, float] = {}
    for asset in primary_run["asset_meta"]:
        cost_value = _safe_float(primary_run["fp64_pipeline_ms"].get(asset))
        if cost_value is not None:
            fp64_cost_values[asset] = cost_value
        newton_value = _safe_float(
            primary_run["fp64_counter_index"].get((asset, "newton_iteration_count"))
        )
        if newton_value is not None:
            fp64_newton_values[asset] = newton_value
    return _assign_terciles(fp64_cost_values), _assign_terciles(fp64_newton_values)


def _build_asset_slice_membership(
    asset_meta: Dict[str, Dict[str, Any]],
    fp64_cost_bucket: Dict[str, str],
    fp64_newton_bucket: Dict[str, str],
) -> Dict[tuple[str, str], List[str]]:
    membership: Dict[tuple[str, str], List[str]] = defaultdict(list)
    for asset, meta in asset_meta.items():
        membership[("overall", "all")].append(asset)
        scenario = meta.get("scenario", "")
        family = meta.get("scenario_family", "")
        if scenario:
            membership[("scenario", scenario)].append(asset)
        if family:
            membership[("scenario_family", family)].append(asset)
        tags = set(meta.get("tags") or [])
        for tag in TAG_SLICES:
            if tag in tags:
                membership[("tag", tag)].append(asset)
        if asset in fp64_cost_bucket:
            membership[("fp64_cost_bucket", fp64_cost_bucket[asset])].append(asset)
        if asset in fp64_newton_bucket:
            membership[("fp64_newton_bucket", fp64_newton_bucket[asset])].append(asset)
    return membership


def _line_search_is_advisory(run_data: Dict[str, Any]) -> bool:
    return run_data.get("line_search_policy", "advisory") == "advisory"


def _build_outcome(
    status: str,
    bucket: str | None,
    details: Dict[str, Any] | None,
    *,
    stage: str = "",
    reason_code: str = "",
    reason: str = "",
    exit_code: int | None = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "bucket": bucket,
        "details": details,
        "stage": stage,
        "reason_code": reason_code,
        "reason": reason,
        "exit_code": exit_code,
    }


def _outcome_from_failure(status: str, failure: Dict[str, Any]) -> Dict[str, Any]:
    return _build_outcome(
        status,
        failure.get("failure_bucket") or "worker:unknown",
        failure,
        stage=str(failure.get("stage", "")),
        reason_code=str(failure.get("reason_code", "")),
        reason=str(failure.get("reason", "")),
        exit_code=_safe_int(failure.get("exit_code")),
    )


def _mode_outcome(run_data: Dict[str, Any], asset: str, level: str, mode: str) -> Dict[str, Any]:
    payload = run_data["status"][(asset, level, mode)]
    worker_result = payload.get("worker_result")
    failure = payload.get("failure")
    if worker_result is not None:
        return _build_outcome("ok", None, worker_result)
    if failure is not None:
        bucket = failure.get("failure_bucket") or "worker:unknown"
        if bucket == "Line Search" and _line_search_is_advisory(run_data):
            return _outcome_from_failure("advisory", failure)
        return _outcome_from_failure("failed", failure)
    return _build_outcome(
        "missing",
        "worker:missing_result",
        None,
        stage="worker",
        reason_code="missing_result",
        reason=f"{mode} worker result missing",
    )


def _quality_outcome(run_data: Dict[str, Any], asset: str, level: str) -> Dict[str, Any]:
    meta = run_data["asset_meta"][asset]
    if not meta.get("quality_enabled", True):
        return _build_outcome("not_required", None, None)
    outcome = _mode_outcome(run_data, asset, level, "quality")
    if level != "fp64" and outcome["status"] == "missing":
        fp64_quality = _mode_outcome(run_data, asset, "fp64", "quality")
        if fp64_quality["status"] != "ok":
            return _build_outcome(
                "blocked",
                "baseline quality",
                fp64_quality,
                stage="baseline quality",
                reason_code="baseline_quality_blocked",
                reason="fp64 quality did not complete successfully",
            )
    if outcome["status"] != "ok":
        return outcome
    if level == "fp64":
        return outcome
    metrics = run_data["quality_index"].get((asset, level))
    if metrics is None:
        return _build_outcome(
            "failed",
            "worker:missing_quality_metrics",
            None,
            stage="worker",
            reason_code="missing_quality_metrics",
            reason="quality metrics missing",
        )
    nan_inf_count = _safe_int(metrics.get("nan_inf_count")) or 0
    if nan_inf_count > 0:
        return _build_outcome(
            "failed",
            "worker:quality_metrics_nan_inf",
            metrics,
            stage="worker",
            reason_code="quality_metrics_nan_inf",
            reason="quality metrics contain NaN or Inf",
        )
    return _build_outcome("ok", None, metrics)


def _asset_level_status(run_data: Dict[str, Any], asset: str, level: str) -> Dict[str, Any]:
    cache = run_data.setdefault("asset_level_status_cache", {})
    key = (asset, level)
    if key in cache:
        return cache[key]

    perf = _mode_outcome(run_data, asset, level, "perf")
    quality = _quality_outcome(run_data, asset, level)
    hard_failure_buckets: List[str] = []
    advisory_buckets: List[str] = []
    blocked_buckets: List[str] = []
    hard_failures: List[Dict[str, Any]] = []
    advisories: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for mode, outcome in (("perf", perf), ("quality", quality)):
        event = {
            "mode": mode,
            "bucket": outcome["bucket"],
            "stage": outcome.get("stage", ""),
            "reason_code": outcome.get("reason_code", ""),
            "reason": outcome.get("reason", ""),
            "exit_code": outcome.get("exit_code"),
        }
        status = outcome["status"]
        if status == "advisory":
            advisory_buckets.append(outcome["bucket"] or "Line Search")
            advisories.append(event)
        elif status == "blocked":
            blocked_buckets.append(outcome["bucket"] or "baseline quality")
            blocked.append(event)
        elif status not in ("ok", "not_required"):
            hard_failure_buckets.append(outcome["bucket"] or "worker:unknown")
            hard_failures.append(event)

    result = {
        "perf": perf,
        "quality": quality,
        "strict_stable": perf["status"] == "ok" and quality["status"] in ("ok", "not_required"),
        "advisory_buckets": list(dict.fromkeys(advisory_buckets)),
        "blocked_buckets": list(dict.fromkeys(blocked_buckets)),
        "hard_failure_buckets": list(dict.fromkeys(hard_failure_buckets)),
        "advisories": advisories,
        "blocked": blocked,
        "hard_failures": hard_failures,
        "has_advisory": bool(advisory_buckets),
        "has_blocked": bool(blocked_buckets),
        "has_hard_failure": bool(hard_failure_buckets),
        "hard_denominator_eligible": not advisory_buckets and not blocked_buckets,
    }
    cache[key] = result
    return result


def _strict_stable(run_data: Dict[str, Any], asset: str, level: str) -> bool:
    return bool(_asset_level_status(run_data, asset, level)["strict_stable"])


def _collect_mode_outcomes(run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    cache = run_data.setdefault("mode_outcomes_cache", [])
    if cache:
        return cache
    rows: List[Dict[str, Any]] = []
    for asset in sorted(run_data["asset_meta"]):
        for level in run_data["levels"]:
            for mode, outcome in (
                ("perf", _mode_outcome(run_data, asset, level, "perf")),
                ("quality", _quality_outcome(run_data, asset, level)),
            ):
                rows.append(
                    {
                        "asset": asset,
                        "level": level,
                        "mode": mode,
                        "status": outcome["status"],
                        "failure_bucket": outcome["bucket"],
                        "stage": outcome.get("stage", ""),
                        "reason_code": outcome.get("reason_code", ""),
                        "reason": outcome.get("reason", ""),
                        "exit_code": outcome.get("exit_code"),
                    }
                )
    run_data["mode_outcomes_cache"] = rows
    return rows


def _hard_failures(run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        row
        for row in _collect_mode_outcomes(run_data)
        if row["status"] not in ("ok", "not_required", "blocked", "advisory")
    ]


def _advisory_failures(run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not _line_search_is_advisory(run_data):
        return []
    return [row for row in _collect_mode_outcomes(run_data) if row["status"] == "advisory"]


def _total_speedup(run_data: Dict[str, Any], asset: str, level: str) -> float | None:
    pipeline_row = run_data["pipeline_index"].get((asset, level))
    if not pipeline_row:
        return None
    fp64_ms = _safe_float(pipeline_row.get("fp64_ms_per_frame"))
    level_ms = _safe_float(pipeline_row.get("level_ms_per_frame"))
    if fp64_ms is None or level_ms is None or level_ms <= 0.0:
        return None
    return fp64_ms / level_ms


def _stage_summary_row(
    run_data: Dict[str, Any], asset: str, level: str, stage: str
) -> Dict[str, Any] | None:
    return run_data["stage_summary_index"].get((asset, level, stage))


def _iteration_summary_row(
    run_data: Dict[str, Any], asset: str, level: str, counter: str
) -> Dict[str, Any] | None:
    return run_data["iteration_summary_index"].get((asset, level, counter))


def _metric_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _stage_speedup_and_saved(
    run_data: Dict[str, Any], asset: str, level: str, stage: str
) -> tuple[float | None, float | None]:
    row = run_data["stage_index"].get((asset, level, stage))
    if not row:
        return None, None
    fp64_ms = _safe_float(row.get("fp64_ms_per_frame"))
    level_ms = _safe_float(row.get("level_ms_per_frame"))
    if fp64_ms is None or level_ms is None:
        return None, None
    return _metric_ratio(fp64_ms, level_ms), fp64_ms - level_ms


def _stage_total_values(
    run_data: Dict[str, Any], asset: str, level: str, stage: str
) -> tuple[float | None, float | None]:
    fp64_row = _stage_summary_row(run_data, asset, "fp64", stage)
    level_row = _stage_summary_row(run_data, asset, level, stage)
    return _safe_float((fp64_row or {}).get("total_ms")), _safe_float((level_row or {}).get("total_ms"))


def _iteration_count_values(
    run_data: Dict[str, Any], asset: str, level: str, counter: str
) -> tuple[float | None, float | None, float | None, float | None]:
    fp64_row = _iteration_summary_row(run_data, asset, "fp64", counter)
    level_row = _iteration_summary_row(run_data, asset, level, counter)
    return (
        _safe_float((fp64_row or {}).get("mean_count")),
        _safe_float((level_row or {}).get("mean_count")),
        _safe_float((fp64_row or {}).get("total_count")),
        _safe_float((level_row or {}).get("total_count")),
    )


def _stage_total_speedup_and_saved(
    run_data: Dict[str, Any], asset: str, level: str, stage: str
) -> tuple[float | None, float | None]:
    fp64_total_ms, level_total_ms = _stage_total_values(run_data, asset, level, stage)
    if fp64_total_ms is None or level_total_ms is None:
        return None, None
    return _metric_ratio(fp64_total_ms, level_total_ms), fp64_total_ms - level_total_ms


def _stage_total_per_newton(
    run_data: Dict[str, Any], asset: str, level: str, stage: str
) -> tuple[float | None, float | None]:
    fp64_total_ms, level_total_ms = _stage_total_values(run_data, asset, level, stage)
    _fp64_mean, _level_mean, fp64_total, level_total = _iteration_count_values(
        run_data, asset, level, "newton_iteration_count"
    )
    return _metric_ratio(fp64_total_ms, fp64_total), _metric_ratio(level_total_ms, level_total)


def _stage_total_per_pcg(
    run_data: Dict[str, Any], asset: str, level: str, stage: str
) -> tuple[float | None, float | None]:
    fp64_total_ms, level_total_ms = _stage_total_values(run_data, asset, level, stage)
    _fp64_mean, _level_mean, fp64_total, level_total = _iteration_count_values(
        run_data, asset, level, "pcg_iteration_count"
    )
    return _metric_ratio(fp64_total_ms, fp64_total), _metric_ratio(level_total_ms, level_total)


def _list_top_stages(stage_saved: Dict[str, List[float]], positive: bool) -> str:
    scored: List[tuple[str, float]] = []
    for stage, values in stage_saved.items():
        avg_saved = _mean(values)
        if avg_saved is None:
            continue
        if positive and avg_saved > 0.0:
            scored.append((stage, avg_saved))
        if not positive and avg_saved < 0.0:
            scored.append((stage, avg_saved))
    if positive:
        scored.sort(key=lambda item: (-item[1], item[0]))
    else:
        scored.sort(key=lambda item: (item[1], item[0]))
    return ", ".join(f"{stage} ({value:.3f} ms)" for stage, value in scored[:3]) or "-"


def _main_report_lookup(rows: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str, str], bool]:
    return {
        (row["slice_type"], row["slice_name"], row["level"]): bool(row.get("main_report", True))
        for row in rows
    }


def _slice_assets_and_counts(
    run_data: Dict[str, Any], expected_assets: Sequence[str], level: str
) -> tuple[List[str], List[str], List[str], List[str]]:
    stable_assets: List[str] = []
    advisory_assets: List[str] = []
    blocked_assets: List[str] = []
    hard_failure_assets: List[str] = []
    for asset in expected_assets:
        status = _asset_level_status(run_data, asset, level)
        if status["strict_stable"]:
            stable_assets.append(asset)
        if status["has_advisory"]:
            advisory_assets.append(asset)
        elif status["has_blocked"]:
            blocked_assets.append(asset)
        elif status["has_hard_failure"] or not status["strict_stable"]:
            hard_failure_assets.append(asset)
    return stable_assets, advisory_assets, blocked_assets, hard_failure_assets


def _build_path_totals_by_slice(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    min_support_for_claim: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    stage_names = sorted({key[2] for key in run_data["stage_index"]})
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        expected_count = len(expected_assets)
        for level in compare_levels:
            stable_assets, advisory_assets, blocked_assets, hard_failure_assets = _slice_assets_and_counts(
                run_data, expected_assets, level
            )
            speedups = [
                speedup
                for asset in stable_assets
                if (speedup := _total_speedup(run_data, asset, level)) is not None
            ]
            stage_saved: Dict[str, List[float]] = defaultdict(list)
            for asset in stable_assets:
                for stage_name in stage_names:
                    _stage_speedup, saved_ms = _stage_speedup_and_saved(
                        run_data, asset, level, stage_name
                    )
                    if saved_ms is not None:
                        stage_saved[stage_name].append(saved_ms)
            hard_denominator_count = max(0, expected_count - len(advisory_assets) - len(blocked_assets))
            row = {
                "slice_type": slice_type,
                "slice_name": slice_name,
                "level": level,
                "expected_count": expected_count,
                "hard_denominator_count": hard_denominator_count,
                "stable_asset_count": len(stable_assets),
                "support_count": len(speedups),
                "failure_count": len(hard_failure_assets),
                "hard_failure_count": len(hard_failure_assets),
                "advisory_count": len(advisory_assets),
                "blocked_count": len(blocked_assets),
                "failure_rate": _pct(len(hard_failure_assets), hard_denominator_count),
                "median_speedup": _median(speedups),
                "mean_speedup": _mean(speedups),
                "p90_speedup": _quantile(speedups, 0.9),
                "min_speedup": min(speedups) if speedups else None,
                "max_speedup": max(speedups) if speedups else None,
                "mean_stage_saved_ms": _mean(
                    [value for values in stage_saved.values() for value in values]
                ),
                "top_positive_stages": _list_top_stages(stage_saved, positive=True),
                "top_negative_stages": _list_top_stages(stage_saved, positive=False),
                "claim_eligible": len(speedups) >= min_support_for_claim,
                "main_report": True,
            }
            rows.append(row)

    family_support: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row["slice_type"] != "scenario_family":
            continue
        family_support[row["slice_name"]] = max(
            family_support[row["slice_name"]], int(row["stable_asset_count"])
        )
    for row in rows:
        if row["slice_type"] == "scenario_family":
            row["main_report"] = family_support[row["slice_name"]] >= 2
    return rows


def _build_stage_speedups_by_slice(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    path_totals_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    main_report_lookup = _main_report_lookup(path_totals_rows)
    stage_names = sorted({key[2] for key in run_data["stage_index"]})
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        expected_count = len(expected_assets)
        for level in compare_levels:
            stable_assets, advisory_assets, blocked_assets, hard_failure_assets = _slice_assets_and_counts(
                run_data, expected_assets, level
            )
            for stage in stage_names:
                speedups: List[float] = []
                saved_values: List[float] = []
                for asset in stable_assets:
                    speedup, saved_ms = _stage_speedup_and_saved(run_data, asset, level, stage)
                    if saved_ms is not None:
                        saved_values.append(saved_ms)
                    if speedup is not None:
                        speedups.append(speedup)
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_name": slice_name,
                        "level": level,
                        "stage": stage,
                        "expected_count": expected_count,
                        "hard_denominator_count": max(
                            0, expected_count - len(advisory_assets) - len(blocked_assets)
                        ),
                        "stable_asset_count": len(stable_assets),
                        "support_count": len(speedups),
                        "hard_failure_count": len(hard_failure_assets),
                        "advisory_count": len(advisory_assets),
                        "blocked_count": len(blocked_assets),
                        "failure_rate": _pct(
                            len(hard_failure_assets),
                            max(0, expected_count - len(advisory_assets) - len(blocked_assets)),
                        ),
                        "mean_stage_speedup": _mean(speedups),
                        "median_stage_speedup": _median(speedups),
                        "p90_stage_speedup": _quantile(speedups, 0.9),
                        "mean_stage_saved_ms": _mean(saved_values),
                        "median_stage_saved_ms": _median(saved_values),
                        "main_report": main_report_lookup.get((slice_type, slice_name, level), True),
                    }
                )
    return rows


def _build_iteration_counts_by_slice(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    path_totals_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    main_report_lookup = _main_report_lookup(path_totals_rows)
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        for level in compare_levels:
            stable_assets, advisory_assets, blocked_assets, hard_failure_assets = _slice_assets_and_counts(
                run_data, expected_assets, level
            )
            for counter in ITERATION_COUNTER_ORDER:
                fp64_means: List[float] = []
                level_means: List[float] = []
                fp64_totals: List[float] = []
                level_totals: List[float] = []
                count_ratios: List[float] = []
                for asset in stable_assets:
                    fp64_mean, level_mean, fp64_total, level_total = _iteration_count_values(
                        run_data, asset, level, counter
                    )
                    if fp64_mean is not None:
                        fp64_means.append(fp64_mean)
                    if level_mean is not None:
                        level_means.append(level_mean)
                    if fp64_total is not None:
                        fp64_totals.append(fp64_total)
                    if level_total is not None:
                        level_totals.append(level_total)
                    ratio = _metric_ratio(level_total, fp64_total)
                    if ratio is not None:
                        count_ratios.append(ratio)
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_name": slice_name,
                        "level": level,
                        "counter": counter,
                        "stable_asset_count": len(stable_assets),
                        "support_count": len(count_ratios),
                        "hard_failure_count": len(hard_failure_assets),
                        "advisory_count": len(advisory_assets),
                        "blocked_count": len(blocked_assets),
                        "mean_fp64_mean_count": _mean(fp64_means),
                        "mean_level_mean_count": _mean(level_means),
                        "mean_fp64_total_count": _mean(fp64_totals),
                        "mean_level_total_count": _mean(level_totals),
                        "median_count_ratio": _median(count_ratios),
                        "mean_count_ratio": _mean(count_ratios),
                        "p90_count_ratio": _quantile(count_ratios, 0.9),
                        "main_report": main_report_lookup.get((slice_type, slice_name, level), True),
                    }
                )
    return rows


def _build_stage_total_time_by_slice(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    path_totals_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    main_report_lookup = _main_report_lookup(path_totals_rows)
    stage_names = sorted({key[2] for key in run_data["stage_summary_index"]})
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        for level in compare_levels:
            stable_assets, advisory_assets, blocked_assets, hard_failure_assets = _slice_assets_and_counts(
                run_data, expected_assets, level
            )
            for stage in stage_names:
                fp64_totals: List[float] = []
                level_totals: List[float] = []
                speedups: List[float] = []
                saved_values: List[float] = []
                for asset in stable_assets:
                    fp64_total_ms, level_total_ms = _stage_total_values(run_data, asset, level, stage)
                    if fp64_total_ms is not None:
                        fp64_totals.append(fp64_total_ms)
                    if level_total_ms is not None:
                        level_totals.append(level_total_ms)
                    speedup, saved_ms = _stage_total_speedup_and_saved(run_data, asset, level, stage)
                    if speedup is not None:
                        speedups.append(speedup)
                    if saved_ms is not None:
                        saved_values.append(saved_ms)
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_name": slice_name,
                        "level": level,
                        "stage": stage,
                        "stable_asset_count": len(stable_assets),
                        "support_count": len(speedups),
                        "hard_failure_count": len(hard_failure_assets),
                        "advisory_count": len(advisory_assets),
                        "blocked_count": len(blocked_assets),
                        "mean_fp64_total_ms": _mean(fp64_totals),
                        "mean_level_total_ms": _mean(level_totals),
                        "median_total_speedup": _median(speedups),
                        "mean_total_speedup": _mean(speedups),
                        "p90_total_speedup": _quantile(speedups, 0.9),
                        "mean_total_saved_ms": _mean(saved_values),
                        "median_total_saved_ms": _median(saved_values),
                        "main_report": main_report_lookup.get((slice_type, slice_name, level), True),
                    }
                )
    return rows


def _build_stage_per_newton_by_slice(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    path_totals_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    main_report_lookup = _main_report_lookup(path_totals_rows)
    stage_names = sorted({key[2] for key in run_data["stage_summary_index"]})
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        for level in compare_levels:
            stable_assets, advisory_assets, blocked_assets, hard_failure_assets = _slice_assets_and_counts(
                run_data, expected_assets, level
            )
            for stage in stage_names:
                fp64_values: List[float] = []
                level_values: List[float] = []
                speedups: List[float] = []
                saved_values: List[float] = []
                for asset in stable_assets:
                    fp64_value, level_value = _stage_total_per_newton(run_data, asset, level, stage)
                    if fp64_value is not None:
                        fp64_values.append(fp64_value)
                    if level_value is not None:
                        level_values.append(level_value)
                    speedup = _metric_ratio(fp64_value, level_value)
                    if speedup is not None:
                        speedups.append(speedup)
                    if fp64_value is not None and level_value is not None:
                        saved_values.append(fp64_value - level_value)
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_name": slice_name,
                        "level": level,
                        "stage": stage,
                        "stable_asset_count": len(stable_assets),
                        "support_count": len(speedups),
                        "hard_failure_count": len(hard_failure_assets),
                        "advisory_count": len(advisory_assets),
                        "blocked_count": len(blocked_assets),
                        "mean_fp64_ms_per_newton": _mean(fp64_values),
                        "mean_level_ms_per_newton": _mean(level_values),
                        "median_per_newton_speedup": _median(speedups),
                        "mean_per_newton_speedup": _mean(speedups),
                        "p90_per_newton_speedup": _quantile(speedups, 0.9),
                        "mean_per_newton_saved_ms": _mean(saved_values),
                        "median_per_newton_saved_ms": _median(saved_values),
                        "main_report": main_report_lookup.get((slice_type, slice_name, level), True),
                    }
                )
    return rows


def _build_solver_stage_per_pcg_by_slice(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    path_totals_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    main_report_lookup = _main_report_lookup(path_totals_rows)
    stage_names = sorted(
        stage for stage in {key[2] for key in run_data["stage_summary_index"]} if stage in SOLVER_STAGES
    )
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        for level in compare_levels:
            stable_assets, advisory_assets, blocked_assets, hard_failure_assets = _slice_assets_and_counts(
                run_data, expected_assets, level
            )
            for stage in stage_names:
                fp64_values: List[float] = []
                level_values: List[float] = []
                speedups: List[float] = []
                saved_values: List[float] = []
                for asset in stable_assets:
                    fp64_value, level_value = _stage_total_per_pcg(run_data, asset, level, stage)
                    if fp64_value is not None:
                        fp64_values.append(fp64_value)
                    if level_value is not None:
                        level_values.append(level_value)
                    speedup = _metric_ratio(fp64_value, level_value)
                    if speedup is not None:
                        speedups.append(speedup)
                    if fp64_value is not None and level_value is not None:
                        saved_values.append(fp64_value - level_value)
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_name": slice_name,
                        "level": level,
                        "stage": stage,
                        "stable_asset_count": len(stable_assets),
                        "support_count": len(speedups),
                        "hard_failure_count": len(hard_failure_assets),
                        "advisory_count": len(advisory_assets),
                        "blocked_count": len(blocked_assets),
                        "mean_fp64_ms_per_pcg": _mean(fp64_values),
                        "mean_level_ms_per_pcg": _mean(level_values),
                        "median_per_pcg_speedup": _median(speedups),
                        "mean_per_pcg_speedup": _mean(speedups),
                        "p90_per_pcg_speedup": _quantile(speedups, 0.9),
                        "mean_per_pcg_saved_ms": _mean(saved_values),
                        "median_per_pcg_saved_ms": _median(saved_values),
                        "main_report": main_report_lookup.get((slice_type, slice_name, level), True),
                    }
                )
    return rows


def _build_stage_leaderboard_by_slice(
    stage_total_rows: Sequence[Dict[str, Any]],
    stage_per_newton_rows: Sequence[Dict[str, Any]],
    solver_stage_per_pcg_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    metric_specs = [
        ("total_speedup", stage_total_rows, "median_total_speedup"),
        ("total_saved_ms", stage_total_rows, "mean_total_saved_ms"),
        ("per_newton_speedup", stage_per_newton_rows, "median_per_newton_speedup"),
        ("per_pcg_speedup", solver_stage_per_pcg_rows, "median_per_pcg_speedup"),
    ]
    rows: List[Dict[str, Any]] = []
    for metric_name, source_rows, value_key in metric_specs:
        grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            if _safe_float(row.get(value_key)) is None or int(row.get("support_count", 0)) <= 0:
                continue
            grouped[(row["slice_type"], row["slice_name"], row["stage"])].append(row)
        for (slice_type, slice_name, stage), group in grouped.items():
            ordered = sorted(
                group,
                key=lambda item: (_safe_float(item.get(value_key)) or -math.inf, item["level"]),
            )
            best = ordered[-1]
            worst = ordered[0]
            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_name": slice_name,
                    "stage": stage,
                    "metric": metric_name,
                    "best_level": best["level"],
                    "best_value": _safe_float(best.get(value_key)),
                    "best_support_count": best["support_count"],
                    "worst_level": worst["level"],
                    "worst_value": _safe_float(worst.get(value_key)),
                    "worst_support_count": worst["support_count"],
                    "main_report": bool(best.get("main_report", True)),
                }
            )
    return rows


def _build_line_search_advisories(run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for failure in _advisory_failures(run_data):
        rows.append(
            {
                "asset": failure.get("asset", ""),
                "level": failure.get("level", ""),
                "mode": failure.get("mode", ""),
                "stage": failure.get("stage", ""),
                "reason_code": failure.get("reason_code", ""),
                "reason": failure.get("reason", ""),
                "failure_bucket": failure.get("failure_bucket", ""),
                "output_dir": failure.get("output_dir", ""),
            }
        )
    rows.sort(key=lambda row: (row["asset"], row["level"], row["mode"]))
    return rows


def _build_failure_reason_breakdown(run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for row in _hard_failures(run_data):
        key = (
            row["level"],
            row["failure_bucket"] or "worker:unknown",
            row.get("stage", ""),
            row.get("reason_code", ""),
            row.get("reason", ""),
        )
        payload = grouped.setdefault(
            key,
            {
                "level": row["level"],
                "failure_bucket": row["failure_bucket"] or "worker:unknown",
                "failure_stage": row.get("stage", ""),
                "reason_code": row.get("reason_code", ""),
                "reason": row.get("reason", ""),
                "count": 0,
                "modes": [],
                "sample_assets": [],
                "sample_exit_codes": [],
            },
        )
        payload["count"] += 1
        if row["mode"] not in payload["modes"]:
            payload["modes"].append(row["mode"])
        if row["asset"] not in payload["sample_assets"] and len(payload["sample_assets"]) < 6:
            payload["sample_assets"].append(row["asset"])
        exit_code = row.get("exit_code")
        if exit_code is not None and exit_code not in payload["sample_exit_codes"] and len(payload["sample_exit_codes"]) < 6:
            payload["sample_exit_codes"].append(exit_code)
    rows = []
    for payload in grouped.values():
        rows.append(
            {
                "level": payload["level"],
                "failure_bucket": payload["failure_bucket"],
                "failure_stage": payload["failure_stage"],
                "reason_code": payload["reason_code"],
                "reason": payload["reason"],
                "count": payload["count"],
                "modes": ",".join(payload["modes"]),
                "sample_assets": ",".join(payload["sample_assets"]),
                "sample_exit_codes": ",".join(str(code) for code in payload["sample_exit_codes"]),
            }
        )
    rows.sort(key=lambda row: (row["level"], -int(row["count"]), row["failure_bucket"], row["reason_code"]))
    return rows


def _build_asset_stability_matrix(
    run_data: Dict[str, Any],
    fp64_cost_bucket: Dict[str, str],
    fp64_newton_bucket: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    levels = run_data["levels"]
    for asset, meta in sorted(run_data["asset_meta"].items()):
        for level in levels:
            status = _asset_level_status(run_data, asset, level)
            quality_metrics = run_data["quality_index"].get((asset, level), {})
            fp64_newton_mean, level_newton_mean, fp64_newton_total, level_newton_total = _iteration_count_values(
                run_data, asset, level, "newton_iteration_count"
            )
            _fp64_pcg_mean, level_pcg_mean, _fp64_pcg_total, level_pcg_total = _iteration_count_values(
                run_data, asset, level, "pcg_iteration_count"
            )
            _fp64_ls_mean, level_line_search_mean, _fp64_ls_total, level_line_search_total = _iteration_count_values(
                run_data, asset, level, "line_search_iteration_count"
            )
            line_search_per_newton = None
            if (
                level_newton_mean is not None
                and level_newton_mean > 0.0
                and level_line_search_mean is not None
            ):
                line_search_per_newton = level_line_search_mean / level_newton_mean
            hard_failures = status["hard_failures"]
            rows.append(
                {
                    "asset": asset,
                    "scenario": meta.get("scenario", ""),
                    "scenario_family": meta.get("scenario_family", ""),
                    "tags": ",".join(meta.get("tags", [])),
                    "level": level,
                    "quality_enabled": bool(meta.get("quality_enabled", True)),
                    "perf_status": status["perf"]["status"],
                    "quality_status": status["quality"]["status"],
                    "strict_stable": status["strict_stable"],
                    "failure_bucket": _unique_join(entry["bucket"] for entry in hard_failures),
                    "hard_failure_bucket": _unique_join(entry["bucket"] for entry in hard_failures),
                    "advisory_bucket": _unique_join(status["advisory_buckets"]),
                    "blocked_bucket": _unique_join(status["blocked_buckets"]),
                    "failure_stage": _unique_join(entry["stage"] for entry in hard_failures),
                    "reason_code": _unique_join(entry["reason_code"] for entry in hard_failures),
                    "reason": _unique_join(entry["reason"] for entry in hard_failures),
                    "exit_code": _unique_join(entry["exit_code"] for entry in hard_failures),
                    "total_speedup": _total_speedup(run_data, asset, level) if level != "fp64" else None,
                    "pipeline_ms": _safe_float(
                        ((run_data["summary"].get("assets") or {}).get(asset, {}).get("levels") or {})
                        .get(level, {})
                        .get("frame_ms")
                    ),
                    "fp64_pipeline_ms": _safe_float(run_data["fp64_pipeline_ms"].get(asset)),
                    "rel_l2_max": _safe_float(quality_metrics.get("rel_l2_max")),
                    "abs_linf_max": _safe_float(quality_metrics.get("abs_linf_max")),
                    "nan_inf_count": _safe_int(quality_metrics.get("nan_inf_count")),
                    "record_count": _safe_int(quality_metrics.get("record_count")),
                    "newton_mean_count": level_newton_mean,
                    "newton_total_count": level_newton_total,
                    "pcg_mean_count": level_pcg_mean,
                    "pcg_total_count": level_pcg_total,
                    "line_search_mean_count": level_line_search_mean,
                    "line_search_total_count": level_line_search_total,
                    "line_search_per_newton": line_search_per_newton,
                    "fp64_newton_mean_count": fp64_newton_mean,
                    "fp64_newton_total_count": fp64_newton_total,
                    "fp64_cost_bucket": fp64_cost_bucket.get(asset, ""),
                    "fp64_newton_bucket": fp64_newton_bucket.get(asset, ""),
                }
            )
    return rows


def _build_path_dominance(
    run_data: Dict[str, Any],
    membership: Dict[tuple[str, str], List[str]],
    path_totals_rows: List[Dict[str, Any]],
    min_support_for_claim: int,
) -> List[Dict[str, Any]]:
    totals_lookup = {
        (row["slice_type"], row["slice_name"], row["level"]): row for row in path_totals_rows
    }
    rows: List[Dict[str, Any]] = []
    compare_levels = run_data["compare_levels"]
    for (slice_type, slice_name), assets in membership.items():
        expected_assets = sorted(dict.fromkeys(assets))
        for dominant in compare_levels:
            for dominated in compare_levels:
                if dominant == dominated:
                    continue
                common_assets: List[str] = []
                dominant_speedups: List[float] = []
                dominated_speedups: List[float] = []
                all_ge = True
                any_gt = False
                for asset in expected_assets:
                    if not _strict_stable(run_data, asset, dominant):
                        continue
                    if not _strict_stable(run_data, asset, dominated):
                        continue
                    dominant_speedup = _total_speedup(run_data, asset, dominant)
                    dominated_speedup = _total_speedup(run_data, asset, dominated)
                    if dominant_speedup is None or dominated_speedup is None:
                        continue
                    common_assets.append(asset)
                    dominant_speedups.append(dominant_speedup)
                    dominated_speedups.append(dominated_speedup)
                    if dominant_speedup + 1e-12 < dominated_speedup:
                        all_ge = False
                    if dominant_speedup > dominated_speedup + 1e-12:
                        any_gt = True
                if len(common_assets) < min_support_for_claim:
                    continue
                dominant_totals = totals_lookup.get((slice_type, slice_name, dominant))
                dominated_totals = totals_lookup.get((slice_type, slice_name, dominated))
                dominant_failure_rate = (
                    _safe_float(dominant_totals.get("failure_rate")) if dominant_totals else None
                )
                dominated_failure_rate = (
                    _safe_float(dominated_totals.get("failure_rate")) if dominated_totals else None
                )
                if dominant_failure_rate is None or dominated_failure_rate is None:
                    continue
                if dominant_failure_rate > dominated_failure_rate + 1e-12:
                    continue
                if not all_ge or not any_gt:
                    continue
                speedup_deltas = [a - b for a, b in zip(dominant_speedups, dominated_speedups)]
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice_name": slice_name,
                        "dominant_path": dominant,
                        "dominated_path": dominated,
                        "common_support": len(common_assets),
                        "dominant_failure_rate": dominant_failure_rate,
                        "dominated_failure_rate": dominated_failure_rate,
                        "median_speedup_delta": _median(speedup_deltas),
                        "min_speedup_delta": min(speedup_deltas),
                        "dominant_mean_speedup": _mean(dominant_speedups),
                        "dominated_mean_speedup": _mean(dominated_speedups),
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_slice_rows(
    rows: Sequence[Dict[str, Any]], slice_type: str, *, main_only: bool = True
) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if row["slice_type"] == slice_type]
    if main_only:
        filtered = [row for row in filtered if row.get("main_report", True)]
    return filtered


def _row_order(slice_type: str, names: Iterable[str]) -> List[str]:
    items = list(dict.fromkeys(names))
    if slice_type == "scenario":
        order = [name for name in SCENARIO_ORDER if name in items]
        return order + sorted(name for name in items if name not in order)
    if slice_type == "tag":
        order = [name for name in TAG_SLICES if name in items]
        return order + sorted(name for name in items if name not in order)
    if slice_type in ("fp64_cost_bucket", "fp64_newton_bucket"):
        order = [name for name in BUCKET_ORDER if name in items]
        return order + sorted(name for name in items if name not in order)
    return sorted(items)


def _sanitize_token(token: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in token)


def _init_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except Exception as exc:  # pragma: no cover - hard fail path
        raise RuntimeError(
            "matplotlib with Agg backend is required for analyze_mixed_uipc_paths.py"
        ) from exc
    return plt, LinearSegmentedColormap, TwoSlopeNorm


def _save_tradeoff_chart(
    chart_path: Path,
    overall_rows: Sequence[Dict[str, Any]],
    levels: Sequence[str],
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs = []
    ys = []
    sizes = []
    labels = []
    for level in levels:
        row = next(row for row in overall_rows if row["level"] == level)
        xs.append(float(row["failure_rate"]))
        ys.append(float(row["median_speedup"]) if row["median_speedup"] is not None else 0.0)
        sizes.append(100.0 + 25.0 * row["support_count"])
        labels.append(level)
    ax.scatter(xs, ys, s=sizes, c=range(len(xs)), cmap="viridis", alpha=0.85, edgecolors="black")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Failure rate")
    ax.set_ylabel("Median speedup vs fp64")
    ax.set_title("Overall Path Tradeoff")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, format="svg")
    plt.close(fig)


def _save_heatmap(
    chart_path: Path,
    rows: Sequence[Dict[str, Any]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    value_key: str,
    title: str,
    plt: Any,
    LinearSegmentedColormap: Any,
    TwoSlopeNorm: Any,
    annotate_kind: str = "speedup",
    higher_is_better: bool = True,
) -> None:
    if not row_labels or not col_labels:
        return
    values = [[math.nan for _ in col_labels] for _ in row_labels]
    annotations = [["" for _ in col_labels] for _ in row_labels]
    row_rank = {label: idx for idx, label in enumerate(row_labels)}
    col_rank = {label: idx for idx, label in enumerate(col_labels)}
    raw_values: List[float] = []
    for row in rows:
        row_label = row["slice_name"]
        col_label = row["level"]
        if row_label not in row_rank or col_label not in col_rank:
            continue
        value = _safe_float(row.get(value_key))
        i = row_rank[row_label]
        j = col_rank[col_label]
        if value is not None:
            values[i][j] = value
            raw_values.append(value)
        support = row.get("support_count", 0)
        failure_rate = row.get("failure_rate")
        if annotate_kind == "speedup":
            annotations[i][j] = (
                f"{_fmt_float(value, 2)}\n"
                f"n={support}\n"
                f"f={_fmt_pct(_safe_float(failure_rate))}"
            )
        elif annotate_kind == "ratio":
            annotations[i][j] = f"{_fmt_float(value, 2)}\n" f"n={support}"
        else:
            annotations[i][j] = f"{_fmt_float(value, 2)}"

    if not raw_values:
        return
    colors = ["#b2182b", "#f7f7f7", "#1a9850"] if higher_is_better else ["#1a9850", "#f7f7f7", "#b2182b"]
    cmap = LinearSegmentedColormap.from_list("tradeoff", colors)
    cmap.set_bad("#d9d9d9")
    vmin = min(raw_values)
    vmax = max(raw_values)
    if annotate_kind in {"speedup", "ratio"}:
        vmin = min(vmin, 0.7)
        vmax = max(vmax, 1.3)
        norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    else:
        bound = max(abs(vmin), abs(vmax), 1e-6)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)

    fig_w = max(8.0, len(col_labels) * 1.05 + 2.0)
    fig_h = max(4.0, len(row_labels) * 0.7 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if annotations[i][j]:
                ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(chart_path, format="svg")
    plt.close(fig)


def _save_stage_saved_chart(
    chart_path: Path,
    rows: Sequence[Dict[str, Any]],
    level_order: Sequence[str],
    title: str,
    plt: Any,
    LinearSegmentedColormap: Any,
    TwoSlopeNorm: Any,
) -> None:
    stage_names = sorted(
        {
            row["stage"]
            for row in rows
            if _safe_float(row.get("mean_stage_saved_ms")) is not None
        }
    )
    if not stage_names:
        return
    _save_heatmap(
        chart_path=chart_path,
        rows=[dict(row, slice_name=row["stage"]) for row in rows],
        row_labels=stage_names,
        col_labels=list(level_order),
        value_key="mean_stage_saved_ms",
        title=title,
        plt=plt,
        LinearSegmentedColormap=LinearSegmentedColormap,
        TwoSlopeNorm=TwoSlopeNorm,
        annotate_kind="saved_ms",
    )


def _save_failure_breakdown_chart(
    chart_path: Path, failures: Sequence[Dict[str, Any]], levels: Sequence[str], plt: Any
) -> None:
    total_by_bucket: Dict[str, int] = defaultdict(int)
    for failure in failures:
        bucket = failure.get("failure_bucket") or "worker:unknown"
        total_by_bucket[bucket] += 1
    if not total_by_bucket:
        return
    ordered_buckets: List[str] = []
    if "search direction" in total_by_bucket:
        ordered_buckets.append("search direction")
    worker_buckets = [
        bucket for bucket in total_by_bucket if bucket != "search direction"
    ]
    worker_buckets.sort(key=lambda bucket: (-total_by_bucket[bucket], bucket))
    keep_buckets = ordered_buckets + worker_buckets[:5]
    if len(total_by_bucket) > len(keep_buckets):
        keep_buckets.append("other")

    counts = {level: {bucket: 0 for bucket in keep_buckets} for level in levels}
    for failure in failures:
        level = failure.get("level")
        if level not in counts:
            continue
        bucket = failure.get("failure_bucket") or "worker:unknown"
        if bucket not in keep_buckets:
            bucket = "other"
        counts[level][bucket] += 1
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bottoms = [0] * len(levels)
    palette = [
        "#b2182b",
        "#2166ac",
        "#4d9221",
        "#984ea3",
        "#ff7f00",
        "#a6761d",
        "#666666",
    ]
    colors = {bucket: palette[idx % len(palette)] for idx, bucket in enumerate(keep_buckets)}
    for bucket in keep_buckets:
        heights = [counts[level][bucket] for level in levels]
        ax.bar(levels, heights, bottom=bottoms, label=bucket, color=colors[bucket])
        bottoms = [base + height for base, height in zip(bottoms, heights)]
    ax.set_ylabel("Failure count")
    ax.set_title("Detailed Hard Failure Breakdown by Path")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(chart_path, format="svg")
    plt.close(fig)


def _pipeline_total_ms(run_data: Dict[str, Any], asset: str, level: str) -> float | None:
    return _safe_float(run_data["stage_summary_index"].get((asset, level, "Pipeline"), {}).get("total_ms"))


def _asset_total_speedup_stats(run_data: Dict[str, Any], asset: str) -> tuple[float | None, float | None]:
    values = [
        _total_speedup(run_data, asset, level)
        for level in run_data["compare_levels"]
        if _strict_stable(run_data, asset, level)
    ]
    values = [value for value in values if value is not None]
    return _median(values), _mean(values)


def _asset_ratio_delta(
    run_data: Dict[str, Any], asset: str, counter: str
) -> float | None:
    values: List[float] = []
    for level in run_data["compare_levels"]:
        if not _strict_stable(run_data, asset, level):
            continue
        _fp64_mean, _level_mean, fp64_total, level_total = _iteration_count_values(
            run_data, asset, level, counter
        )
        ratio = _metric_ratio(level_total, fp64_total)
        if ratio is not None:
            values.append(abs(1.0 - ratio))
    return max(values) if values else None


def _select_standard_report_assets(
    run_data: Dict[str, Any], asset_stability_rows: Sequence[Dict[str, Any]]
) -> List[str]:
    compare_levels = run_data["compare_levels"]
    stable_count_by_asset: Dict[str, int] = defaultdict(int)
    for row in asset_stability_rows:
        if row["level"] == "fp64":
            continue
        if row["strict_stable"]:
            stable_count_by_asset[row["asset"]] += 1
    strict_assets = sorted(
        asset for asset, count in stable_count_by_asset.items() if count == len(compare_levels)
    )
    selected: List[str] = []

    def add_asset(asset: str | None) -> None:
        if asset and asset not in selected:
            selected.append(asset)

    scenarios = {asset: run_data["asset_meta"][asset].get("scenario", "") for asset in strict_assets}
    for scenario in SCENARIO_ORDER:
        scenario_assets = [asset for asset in strict_assets if scenarios.get(asset) == scenario]
        if not scenario_assets:
            continue
        add_asset(
            max(
                scenario_assets,
                key=lambda asset: (_pipeline_total_ms(run_data, asset, "fp64") or -math.inf, asset),
            )
        )

    speedup_rank = sorted(
        (
            (asset, _asset_total_speedup_stats(run_data, asset)[0])
            for asset in strict_assets
        ),
        key=lambda item: (item[1] if item[1] is not None else -math.inf, item[0]),
    )
    if speedup_rank:
        add_asset(speedup_rank[-1][0])
        add_asset(speedup_rank[0][0])

    for counter in ("newton_iteration_count", "pcg_iteration_count"):
        ranked = sorted(
            (
                (asset, _asset_ratio_delta(run_data, asset, counter))
                for asset in strict_assets
            ),
            key=lambda item: (item[1] if item[1] is not None else -math.inf, item[0]),
        )
        if ranked and ranked[-1][1] is not None:
            add_asset(ranked[-1][0])

    search_direction = next(
        (failure.get("asset") for failure in _hard_failures(run_data) if failure.get("failure_bucket") == "search direction"),
        None,
    )
    add_asset(search_direction)

    worker_failure = next(
        (
            failure.get("asset")
            for failure in _hard_failures(run_data)
            if str(failure.get("failure_bucket", "")).startswith("worker:")
        ),
        None,
    )
    add_asset(worker_failure)
    return selected[:8]


def _copy_standard_report_charts(
    out_dir: Path,
    run_data: Dict[str, Any],
    chart_entries: List[Dict[str, Any]],
    asset_stability_rows: Sequence[Dict[str, Any]],
) -> None:
    source_dir = run_data["run_root"] / "reports" / "charts"
    if not source_dir.exists():
        return
    target_dir = out_dir / "charts" / "standard_report"
    target_dir.mkdir(parents=True, exist_ok=True)
    scenario_files = sorted(source_dir.glob("scenario_*.svg"))
    asset_names = _select_standard_report_assets(run_data, asset_stability_rows)
    asset_files = [source_dir / f"asset_{asset}.svg" for asset in asset_names]
    for source in scenario_files + asset_files:
        if not source.exists():
            continue
        relative_name = f"standard_report/{source.name}"
        target = out_dir / "charts" / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        chart_entries.append(
            {
                "filename": relative_name,
                "title": source.stem.replace("_", " "),
                "chart_type": "standard_report",
                "source_run": "primary",
            }
        )


def _save_dominance_matrix(
    chart_path: Path,
    dominance_rows: Sequence[Dict[str, Any]],
    levels: Sequence[str],
    plt: Any,
    LinearSegmentedColormap: Any,
    TwoSlopeNorm: Any,
) -> None:
    matrix = [[0 for _ in levels] for _ in levels]
    rank = {level: idx for idx, level in enumerate(levels)}
    for row in dominance_rows:
        i = rank[row["dominant_path"]]
        j = rank[row["dominated_path"]]
        matrix[i][j] += 1
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    cmap = LinearSegmentedColormap.from_list("dominance", ["#f7f7f7", "#2166ac"])
    bound = max(max(max(row) for row in matrix), 1)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=bound)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    ax.set_title("Dominance Matrix (strict-dominance slice count)")
    for i in range(len(levels)):
        for j in range(len(levels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(chart_path, format="svg")
    plt.close(fig)


def _build_chart_artifacts(
    out_dir: Path,
    run_data: Dict[str, Any],
    path_totals_rows: List[Dict[str, Any]],
    stage_speedup_rows: List[Dict[str, Any]],
    iteration_rows: List[Dict[str, Any]],
    stage_total_rows: List[Dict[str, Any]],
    stage_per_newton_rows: List[Dict[str, Any]],
    solver_stage_per_pcg_rows: List[Dict[str, Any]],
    dominance_rows: List[Dict[str, Any]],
    asset_stability_rows: List[Dict[str, Any]],
    supplemental_rows: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    plt, LinearSegmentedColormap, TwoSlopeNorm = _init_matplotlib()
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_entries: List[Dict[str, Any]] = []
    level_order = list(run_data["compare_levels"])

    def register(filename: str, title: str, chart_type: str, **extra: Any) -> Path:
        chart_entries.append({"filename": filename, "title": title, "chart_type": chart_type, **extra})
        path = charts_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    overall_rows = [
        row
        for row in path_totals_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    tradeoff_path = register(
        "overall_path_tradeoff.svg",
        "Overall Path Tradeoff",
        "tradeoff",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_rows),
    )
    _save_tradeoff_chart(tradeoff_path, overall_rows, level_order, plt)

    scenario_rows = _ordered_slice_rows(path_totals_rows, "scenario")
    scenario_heatmap_path = register(
        "scenario_total_speedup_heatmap.svg",
        "Scenario Total Speedup Heatmap",
        "heatmap",
        slice_type="scenario",
        source_run="primary",
        support_count=sum(row["support_count"] for row in scenario_rows),
    )
    _save_heatmap(
        scenario_heatmap_path,
        scenario_rows,
        _row_order("scenario", [row["slice_name"] for row in scenario_rows]),
        level_order,
        "median_speedup",
        "Scenario Total Speedup",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    family_rows = _ordered_slice_rows(path_totals_rows, "scenario_family")
    family_heatmap_path = register(
        "scenario_family_total_speedup_heatmap.svg",
        "Scenario Family Total Speedup Heatmap",
        "heatmap",
        slice_type="scenario_family",
        source_run="primary",
        support_count=sum(row["support_count"] for row in family_rows),
    )
    _save_heatmap(
        family_heatmap_path,
        family_rows,
        _row_order("scenario_family", [row["slice_name"] for row in family_rows]),
        level_order,
        "median_speedup",
        "Scenario Family Total Speedup",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    tag_rows = _ordered_slice_rows(path_totals_rows, "tag")
    tag_heatmap_path = register(
        "tag_total_speedup_heatmap.svg",
        "Tag Total Speedup Heatmap",
        "heatmap",
        slice_type="tag",
        source_run="primary",
        support_count=sum(row["support_count"] for row in tag_rows),
    )
    _save_heatmap(
        tag_heatmap_path,
        tag_rows,
        _row_order("tag", [row["slice_name"] for row in tag_rows]),
        level_order,
        "median_speedup",
        "Tag Total Speedup",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    cost_rows = _ordered_slice_rows(path_totals_rows, "fp64_cost_bucket")
    cost_heatmap_path = register(
        "fp64_cost_bucket_total_speedup_heatmap.svg",
        "fp64 Cost Bucket Total Speedup Heatmap",
        "heatmap",
        slice_type="fp64_cost_bucket",
        source_run="primary",
        support_count=sum(row["support_count"] for row in cost_rows),
    )
    _save_heatmap(
        cost_heatmap_path,
        cost_rows,
        _row_order("fp64_cost_bucket", [row["slice_name"] for row in cost_rows]),
        level_order,
        "median_speedup",
        "fp64 Cost Bucket Total Speedup",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    newton_rows = _ordered_slice_rows(path_totals_rows, "fp64_newton_bucket")
    newton_heatmap_path = register(
        "fp64_newton_bucket_total_speedup_heatmap.svg",
        "fp64 Newton Bucket Total Speedup Heatmap",
        "heatmap",
        slice_type="fp64_newton_bucket",
        source_run="primary",
        support_count=sum(row["support_count"] for row in newton_rows),
    )
    _save_heatmap(
        newton_heatmap_path,
        newton_rows,
        _row_order("fp64_newton_bucket", [row["slice_name"] for row in newton_rows]),
        level_order,
        "median_speedup",
        "fp64 Newton Bucket Total Speedup",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    overall_newton_ratio_rows = [
        row
        for row in iteration_rows
        if row["slice_type"] == "overall"
        and row["slice_name"] == "all"
        and row["counter"] == "newton_iteration_count"
    ]
    newton_ratio_path = register(
        "newton_count_ratio_heatmap.svg",
        "Newton Count Ratio Heatmap",
        "heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_newton_ratio_rows),
    )
    _save_heatmap(
        newton_ratio_path,
        overall_newton_ratio_rows,
        ["all"],
        level_order,
        "median_count_ratio",
        "Newton Count Ratio (level / fp64)",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
        annotate_kind="ratio",
        higher_is_better=False,
    )

    overall_pcg_ratio_rows = [
        row
        for row in iteration_rows
        if row["slice_type"] == "overall"
        and row["slice_name"] == "all"
        and row["counter"] == "pcg_iteration_count"
    ]
    pcg_ratio_path = register(
        "pcg_count_ratio_heatmap.svg",
        "PCG Count Ratio Heatmap",
        "heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_pcg_ratio_rows),
    )
    _save_heatmap(
        pcg_ratio_path,
        overall_pcg_ratio_rows,
        ["all"],
        level_order,
        "median_count_ratio",
        "PCG Count Ratio (level / fp64)",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
        annotate_kind="ratio",
        higher_is_better=False,
    )

    overall_stage_rows = [
        row
        for row in stage_speedup_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_stage_path = register(
        "stage_saved_ms_overall.svg",
        "Overall Stage Saved Milliseconds",
        "stage_saved_heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_stage_rows),
    )
    _save_stage_saved_chart(
        overall_stage_path,
        overall_stage_rows,
        level_order,
        "Overall Stage Saved ms",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    overall_stage_total_rows = [
        row
        for row in stage_total_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_stage_total_speedup_path = register(
        "overall_stage_total_speedup_heatmap.svg",
        "Overall Stage Total Speedup Heatmap",
        "heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_stage_total_rows),
    )
    _save_heatmap(
        overall_stage_total_speedup_path,
        [dict(row, slice_name=row["stage"]) for row in overall_stage_total_rows],
        sorted({row["stage"] for row in overall_stage_total_rows}),
        level_order,
        "median_total_speedup",
        "Overall Stage Total Speedup",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    overall_stage_total_saved_path = register(
        "overall_stage_total_saved_ms_heatmap.svg",
        "Overall Stage Total Saved Milliseconds Heatmap",
        "heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_stage_total_rows),
    )
    _save_heatmap(
        overall_stage_total_saved_path,
        [dict(row, slice_name=row["stage"]) for row in overall_stage_total_rows],
        sorted({row["stage"] for row in overall_stage_total_rows}),
        level_order,
        "mean_total_saved_ms",
        "Overall Stage Total Saved ms",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
        annotate_kind="saved_ms",
    )

    overall_stage_per_newton_rows = [
        row
        for row in stage_per_newton_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_stage_per_newton_path = register(
        "overall_stage_per_newton_heatmap.svg",
        "Overall Stage Per-Newton Heatmap",
        "heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_stage_per_newton_rows),
    )
    _save_heatmap(
        overall_stage_per_newton_path,
        [dict(row, slice_name=row["stage"]) for row in overall_stage_per_newton_rows],
        sorted({row["stage"] for row in overall_stage_per_newton_rows}),
        level_order,
        "median_per_newton_speedup",
        "Overall Stage ms per Newton",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    overall_solver_stage_per_pcg_rows = [
        row
        for row in solver_stage_per_pcg_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_solver_stage_per_pcg_path = register(
        "overall_solver_stage_per_pcg_heatmap.svg",
        "Overall Solver Stage Per-PCG Heatmap",
        "heatmap",
        slice_type="overall",
        slice_name="all",
        source_run="primary",
        support_count=sum(row["support_count"] for row in overall_solver_stage_per_pcg_rows),
    )
    _save_heatmap(
        overall_solver_stage_per_pcg_path,
        [dict(row, slice_name=row["stage"]) for row in overall_solver_stage_per_pcg_rows],
        sorted({row["stage"] for row in overall_solver_stage_per_pcg_rows}),
        level_order,
        "median_per_pcg_speedup",
        "Overall Solver Stage ms per PCG",
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    for scenario in _row_order("scenario", [row["slice_name"] for row in scenario_rows]):
        scenario_stage_rows = [
            row
            for row in stage_speedup_rows
            if row["slice_type"] == "scenario"
            and row["slice_name"] == scenario
            and row["main_report"]
        ]
        if not scenario_stage_rows:
            continue
        if max(row["stable_asset_count"] for row in scenario_stage_rows) < 2:
            continue
        filename = f"scenario_stage_saved_ms_{_sanitize_token(scenario)}.svg"
        scenario_stage_path = register(
            filename,
            f"Scenario Stage Saved ms: {scenario}",
            "stage_saved_heatmap",
            slice_type="scenario",
            slice_name=scenario,
            source_run="primary",
            support_count=sum(row["support_count"] for row in scenario_stage_rows),
        )
        _save_stage_saved_chart(
            scenario_stage_path,
            scenario_stage_rows,
            level_order,
            f"Stage Saved ms by Path: {scenario}",
            plt,
            LinearSegmentedColormap,
            TwoSlopeNorm,
        )

        scenario_stage_total_rows = [
            row
            for row in stage_total_rows
            if row["slice_type"] == "scenario"
            and row["slice_name"] == scenario
            and row["main_report"]
        ]
        if scenario_stage_total_rows:
            filename = f"scenario_stage_total_speedup_{_sanitize_token(scenario)}.svg"
            scenario_total_path = register(
                filename,
                f"Scenario Stage Total Speedup: {scenario}",
                "heatmap",
                slice_type="scenario",
                slice_name=scenario,
                source_run="primary",
                support_count=sum(row["support_count"] for row in scenario_stage_total_rows),
            )
            _save_heatmap(
                scenario_total_path,
                [dict(row, slice_name=row["stage"]) for row in scenario_stage_total_rows],
                sorted({row["stage"] for row in scenario_stage_total_rows}),
                level_order,
                "median_total_speedup",
                f"Stage Total Speedup by Path: {scenario}",
                plt,
                LinearSegmentedColormap,
                TwoSlopeNorm,
            )

        scenario_stage_per_newton_rows = [
            row
            for row in stage_per_newton_rows
            if row["slice_type"] == "scenario"
            and row["slice_name"] == scenario
            and row["main_report"]
        ]
        if scenario_stage_per_newton_rows:
            filename = f"scenario_stage_per_newton_{_sanitize_token(scenario)}.svg"
            scenario_per_newton_path = register(
                filename,
                f"Scenario Stage Per-Newton: {scenario}",
                "heatmap",
                slice_type="scenario",
                slice_name=scenario,
                source_run="primary",
                support_count=sum(row["support_count"] for row in scenario_stage_per_newton_rows),
            )
            _save_heatmap(
                scenario_per_newton_path,
                [dict(row, slice_name=row["stage"]) for row in scenario_stage_per_newton_rows],
                sorted({row["stage"] for row in scenario_stage_per_newton_rows}),
                level_order,
                "median_per_newton_speedup",
                f"Stage ms per Newton by Path: {scenario}",
                plt,
                LinearSegmentedColormap,
                TwoSlopeNorm,
            )

        scenario_solver_stage_per_pcg_rows = [
            row
            for row in solver_stage_per_pcg_rows
            if row["slice_type"] == "scenario"
            and row["slice_name"] == scenario
            and row["main_report"]
        ]
        if scenario_solver_stage_per_pcg_rows:
            filename = f"scenario_solver_stage_per_pcg_{_sanitize_token(scenario)}.svg"
            scenario_per_pcg_path = register(
                filename,
                f"Scenario Solver Stage Per-PCG: {scenario}",
                "heatmap",
                slice_type="scenario",
                slice_name=scenario,
                source_run="primary",
                support_count=sum(row["support_count"] for row in scenario_solver_stage_per_pcg_rows),
            )
            _save_heatmap(
                scenario_per_pcg_path,
                [dict(row, slice_name=row["stage"]) for row in scenario_solver_stage_per_pcg_rows],
                sorted({row["stage"] for row in scenario_solver_stage_per_pcg_rows}),
                level_order,
                "median_per_pcg_speedup",
                f"Solver Stage ms per PCG by Path: {scenario}",
                plt,
                LinearSegmentedColormap,
                TwoSlopeNorm,
            )

    failure_breakdown_path = register(
        "path_failure_breakdown_detailed.svg",
        "Detailed Path Failure Breakdown",
        "stacked_bar",
        slice_type="overall",
        source_run="primary",
        support_count=len(_hard_failures(run_data)),
    )
    _save_failure_breakdown_chart(failure_breakdown_path, _hard_failures(run_data), level_order, plt)

    dominance_matrix_path = register(
        "path_dominance_matrix.svg",
        "Path Dominance Matrix",
        "matrix",
        slice_type="overall",
        source_run="primary",
        support_count=len(dominance_rows),
    )
    _save_dominance_matrix(
        dominance_matrix_path,
        dominance_rows,
        level_order,
        plt,
        LinearSegmentedColormap,
        TwoSlopeNorm,
    )

    if supplemental_rows:
        sup_rows = [row for row in supplemental_rows if row["support_count"] > 0]
        if sup_rows:
            supplemental_path = register(
                "supplemental_family_total_speedup_heatmap.svg",
                "Supplemental Family Total Speedup Heatmap",
                "heatmap",
                slice_type="scenario_family",
                source_run="supplemental",
                support_count=sum(row["support_count"] for row in sup_rows),
            )
            _save_heatmap(
                supplemental_path,
                sup_rows,
                _row_order("scenario_family", [row["slice_name"] for row in sup_rows]),
                level_order,
                "median_speedup",
                "Supplemental Family Total Speedup",
                plt,
                LinearSegmentedColormap,
                TwoSlopeNorm,
            )

    _copy_standard_report_charts(out_dir, run_data, chart_entries, asset_stability_rows)

    return [entry for entry in chart_entries if (charts_dir / entry["filename"]).exists()]


def _best_rows_by_slice(
    path_totals_rows: Sequence[Dict[str, Any]], slice_type: str, min_support_for_claim: int
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in path_totals_rows:
        if row["slice_type"] != slice_type:
            continue
        if row["support_count"] < min_support_for_claim:
            continue
        grouped[row["slice_name"]].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                -(row["median_speedup"] or 0.0),
                row["failure_rate"],
                -(row["support_count"] or 0),
                row["level"],
            )
        )
    return grouped


def _find_asset_failure_mentions(
    run_data: Dict[str, Any],
    assets: Sequence[str],
    *,
    failure_bucket: str | None = None,
) -> List[str]:
    lines: List[str] = []
    relevant = [failure for failure in run_data["failures"] if failure.get("asset") in assets]
    if failure_bucket is not None:
        relevant = [failure for failure in relevant if failure.get("failure_bucket") == failure_bucket]
    relevant.sort(
        key=lambda row: (
            row.get("asset", ""),
            row.get("level", ""),
            row.get("mode", ""),
            row.get("failure_bucket", ""),
        )
    )
    for failure in relevant:
        lines.append(
            f"- `{failure['asset']}` `{failure['level']}` `{failure['mode']}`: "
            f"{failure['failure_bucket']} ({failure.get('stage', '')})"
        )
    return lines


def _build_analysis_markdown(
    run_data: Dict[str, Any],
    path_totals_rows: List[Dict[str, Any]],
    stage_speedup_rows: List[Dict[str, Any]],
    iteration_rows: List[Dict[str, Any]],
    stage_total_rows: List[Dict[str, Any]],
    stage_per_newton_rows: List[Dict[str, Any]],
    solver_stage_per_pcg_rows: List[Dict[str, Any]],
    stage_leaderboard_rows: List[Dict[str, Any]],
    dominance_rows: List[Dict[str, Any]],
    asset_stability_rows: List[Dict[str, Any]],
    failure_reason_rows: List[Dict[str, Any]],
    advisory_rows: List[Dict[str, Any]],
    chart_entries: List[Dict[str, Any]],
    supplemental_rows: List[Dict[str, Any]] | None,
    min_support_for_claim: int,
) -> str:
    overall_rows = sorted(
        [
            row
            for row in path_totals_rows
            if row["slice_type"] == "overall" and row["slice_name"] == "all"
        ],
        key=lambda row: (-(row["median_speedup"] or 0.0), row["failure_rate"]),
    )
    compare_levels = run_data["compare_levels"]
    stable_count_by_asset = defaultdict(int)
    for row in asset_stability_rows:
        if row["level"] == "fp64":
            continue
        if row["strict_stable"]:
            stable_count_by_asset[row["asset"]] += 1
    complete_stable_assets = sorted(
        asset for asset, count in stable_count_by_asset.items() if count == len(compare_levels)
    )
    near_stable_assets = sorted(
        asset for asset, count in stable_count_by_asset.items() if count == len(compare_levels) - 1
    )
    hard_failures = _hard_failures(run_data)
    failure_counts = defaultdict(int)
    for failure in hard_failures:
        failure_counts[failure["failure_bucket"]] += 1
    advisory_counts = defaultdict(int)
    for failure in advisory_rows:
        advisory_counts[failure["failure_bucket"]] += 1
    blocked_quality_rows = [
        row
        for row in asset_stability_rows
        if row["level"] != "fp64" and row.get("blocked_bucket")
    ]
    blocked_quality_assets = sorted({row["asset"] for row in blocked_quality_rows})

    scenario_best = _best_rows_by_slice(path_totals_rows, "scenario", min_support_for_claim)
    family_best = _best_rows_by_slice(path_totals_rows, "scenario_family", min_support_for_claim)
    tag_best = _best_rows_by_slice(path_totals_rows, "tag", min_support_for_claim)
    cost_best = _best_rows_by_slice(path_totals_rows, "fp64_cost_bucket", min_support_for_claim)
    newton_best = _best_rows_by_slice(path_totals_rows, "fp64_newton_bucket", min_support_for_claim)
    overall_iteration_rows = [
        row
        for row in iteration_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_stage_total_rows = [
        row
        for row in stage_total_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_stage_per_newton_rows = [
        row
        for row in stage_per_newton_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_solver_stage_per_pcg_rows = [
        row
        for row in solver_stage_per_pcg_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    overall_stage_leaderboards = [
        row
        for row in stage_leaderboard_rows
        if row["slice_type"] == "overall"
        and row["slice_name"] == "all"
        and row.get("main_report", True)
    ]

    overall_stage_rows = [
        row
        for row in stage_speedup_rows
        if row["slice_type"] == "overall" and row["slice_name"] == "all"
    ]
    per_level_stage_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in overall_stage_rows:
        if row["support_count"] > 0:
            per_level_stage_rows[row["level"]].append(row)
    stage_lines: List[str] = []
    for level in compare_levels:
        rows = per_level_stage_rows.get(level, [])
        positive = sorted(
            [
                (row["stage"], row["mean_stage_saved_ms"])
                for row in rows
                if (row["mean_stage_saved_ms"] or 0.0) > 0.0
            ],
            key=lambda item: (-item[1], item[0]),
        )[:3]
        negative = sorted(
            [
                (row["stage"], row["mean_stage_saved_ms"])
                for row in rows
                if (row["mean_stage_saved_ms"] or 0.0) < 0.0
            ],
            key=lambda item: (item[1], item[0]),
        )[:3]
        positive_text = ", ".join(f"{stage} ({value:.3f} ms)" for stage, value in positive) or "-"
        negative_text = ", ".join(f"{stage} ({value:.3f} ms)" for stage, value in negative) or "-"
        stage_lines.append(f"- `{level}` gains: {positive_text}; regressions: {negative_text}")

    iteration_lines: List[str] = []
    for counter in ("newton_iteration_count", "pcg_iteration_count"):
        counter_rows = [row for row in overall_iteration_rows if row["counter"] == counter]
        counter_rows = [row for row in counter_rows if row["support_count"] > 0]
        counter_rows.sort(
            key=lambda row: (
                _safe_float(row.get("median_count_ratio")) if _safe_float(row.get("median_count_ratio")) is not None else math.inf,
                row["level"],
            )
        )
        if not counter_rows:
            continue
        best = counter_rows[0]
        worst = counter_rows[-1]
        iteration_lines.append(
            f"- `{counter}`: lowest ratio is `{best['level']}` at {_fmt_float(best['median_count_ratio'], 3)}x; "
            f"highest is `{worst['level']}` at {_fmt_float(worst['median_count_ratio'], 3)}x."
        )
    top_failure_reason_lines = [
        f"- `{row['level']}` `{row['failure_bucket']}`: count={row['count']}, "
        f"stage=`{row['failure_stage'] or '-'}`, sample_assets=`{row['sample_assets'] or '-'}`"
        for row in failure_reason_rows[:12]
    ]

    stage_leaderboard_lines: List[str] = []
    for metric_name in ("total_speedup", "per_newton_speedup", "per_pcg_speedup"):
        metric_rows = [row for row in overall_stage_leaderboards if row["metric"] == metric_name]
        metric_rows = sorted(metric_rows, key=lambda row: row["stage"])
        label = {
            "total_speedup": "Stage total-time winners",
            "per_newton_speedup": "Stage per-Newton winners",
            "per_pcg_speedup": "Solver-stage per-PCG winners",
        }[metric_name]
        if metric_rows:
            stage_leaderboard_lines.append(label + ":")
            for row in metric_rows[:12]:
                stage_leaderboard_lines.append(
                    f"- `{row['stage']}`: best `{row['best_level']}` ({_fmt_float(row['best_value'], 3)}), "
                    f"worst `{row['worst_level']}` ({_fmt_float(row['worst_value'], 3)})"
                )

    dominance_lines = [
        f"- `{row['dominant_path']}` > `{row['dominated_path']}` on "
        f"`{row['slice_type']}={row['slice_name']}` "
        f"(common={row['common_support']}, failure {row['dominant_failure_rate']:.1%} <= {row['dominated_failure_rate']:.1%})"
        for row in sorted(
            dominance_rows,
            key=lambda row: (
                row["slice_type"],
                row["slice_name"],
                row["dominant_path"],
                row["dominated_path"],
            ),
        )
    ]
    if not dominance_lines:
        dominance_lines = ["- No slice met the strict-dominance rule."]

    top_failure_assets = sorted(
        (
            (
                asset,
                sum(1 for failure in hard_failures if failure["asset"] == asset),
            )
            for asset in {failure["asset"] for failure in hard_failures}
        ),
        key=lambda item: (-item[1], item[0]),
    )[:8]
    failure_asset_lines = [f"- `{asset}`: {count} failure events" for asset, count in top_failure_assets]

    quality_risk_rows = sorted(
        [
            row
            for row in asset_stability_rows
            if row["level"] != "fp64" and row["strict_stable"] and row["rel_l2_max"] is not None
        ],
        key=lambda row: (
            -(row["rel_l2_max"] or 0.0),
            -(row["abs_linf_max"] or 0.0),
            row["asset"],
            row["level"],
        ),
    )[:8]
    quality_risk_lines = [
        f"- `{row['asset']}` `{row['level']}`: rel_l2_max={row['rel_l2_max']:.3e}, "
        f"abs_linf_max={row['abs_linf_max']:.3e}, nan_inf={row['nan_inf_count']}"
        for row in quality_risk_rows
    ]

    chart_lookup = {entry["filename"]: entry for entry in chart_entries}

    def image(filename: str, alt: str) -> str:
        if filename not in chart_lookup:
            return ""
        return f"![{alt}](charts/{filename})"

    lines: List[str] = []
    lines.append("# Mixed UIPC Path Deep-Dive Analysis")
    lines.append("")
    lines.append(f"Primary run: `{run_data['run_root']}`")
    lines.append("")
    lines.append("## 1. Overall")
    lines.append("")
    lines.append(
        f"- Worker results: `{sum(1 for payload in run_data['status'].values() if payload['worker_result'] is not None)}`"
    )
    lines.append(f"- Hard failures: `{len(hard_failures)}`")
    lines.append(f"- Line-search advisories: `{len(advisory_rows)}`")
    lines.append(
        f"- Quality tasks skipped after fp64 quality failure: `{len(blocked_quality_rows)}` across `{len(blocked_quality_assets)}` assets"
    )
    lines.append(f"- Complete stable assets: `{len(complete_stable_assets)}`")
    lines.append(f"- Near-stable assets: `{len(near_stable_assets)}`")
    if failure_counts:
        lines.append(
            "- Hard failure buckets: "
            + ", ".join(
                f"`{bucket}`={count}"
                for bucket, count in sorted(
                    failure_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:8]
            )
        )
    else:
        lines.append("- Hard failure buckets: none")
    if advisory_rows:
        lines.append(
            "- Advisory buckets: "
            + ", ".join(
                f"`{bucket}`={advisory_counts.get(bucket, 0)}" for bucket in ADVISORY_BUCKET_ORDER
            )
        )
    if blocked_quality_assets:
        lines.append(
            "- Baseline-quality-blocked assets: "
            + ", ".join(f"`{asset}`" for asset in blocked_quality_assets)
        )
    lines.append("")
    lines.append(image("overall_path_tradeoff.svg", "Overall Path Tradeoff"))
    lines.append("")
    lines.append("| Path | Median speedup | Mean speedup | P90 speedup | Support | Failure rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in overall_rows:
        lines.append(
            f"| `{row['level']}` | {_fmt_float(row['median_speedup'], 3)} | "
            f"{_fmt_float(row['mean_speedup'], 3)} | {_fmt_float(row['p90_speedup'], 3)} | "
            f"{row['support_count']} | {_fmt_pct(row['failure_rate'])} |"
        )
    lines.append("")

    lines.append("## 2. Scenario Recommendations")
    lines.append("")
    lines.append(image("scenario_total_speedup_heatmap.svg", "Scenario Heatmap"))
    lines.append("")
    for scenario in _row_order("scenario", scenario_best.keys()):
        best_rows = scenario_best.get(scenario, [])
        if not best_rows:
            continue
        best = best_rows[0]
        runner_up = best_rows[1] if len(best_rows) > 1 else None
        sentence = (
            f"- `{scenario}`: best median path is `{best['level']}` "
            f"({_fmt_float(best['median_speedup'], 3)}x, support={best['support_count']}, "
            f"failure={_fmt_pct(best['failure_rate'])})"
        )
        if runner_up:
            sentence += (
                f"; next is `{runner_up['level']}` "
                f"({_fmt_float(runner_up['median_speedup'], 3)}x, failure={_fmt_pct(runner_up['failure_rate'])})"
            )
        sentence += "."
        lines.append(sentence)
    lines.append("")
    lines.append(image("scenario_family_total_speedup_heatmap.svg", "Scenario Family Heatmap"))
    lines.append("")
    for family in _row_order("scenario_family", family_best.keys()):
        best_rows = family_best.get(family, [])
        if not best_rows:
            continue
        best = best_rows[0]
        lines.append(
            f"- `{family}`: `{best['level']}` leads at "
            f"{_fmt_float(best['median_speedup'], 3)}x with support={best['support_count']} "
            f"and failure={_fmt_pct(best['failure_rate'])}."
        )
    lines.append("")
    lines.append(image("tag_total_speedup_heatmap.svg", "Tag Heatmap"))
    lines.append("")
    for tag in _row_order("tag", tag_best.keys()):
        best_rows = tag_best.get(tag, [])
        if not best_rows:
            continue
        best = best_rows[0]
        lines.append(
            f"- `{tag}`: `{best['level']}` is the strongest stable median path "
            f"({_fmt_float(best['median_speedup'], 3)}x, failure={_fmt_pct(best['failure_rate'])})."
        )
    lines.append("")
    lines.append(image("fp64_cost_bucket_total_speedup_heatmap.svg", "fp64 Cost Bucket Heatmap"))
    lines.append("")
    for bucket in _row_order("fp64_cost_bucket", cost_best.keys()):
        best_rows = cost_best.get(bucket, [])
        if not best_rows:
            continue
        best = best_rows[0]
        lines.append(
            f"- `fp64_cost_bucket={bucket}`: `{best['level']}` gives "
            f"{_fmt_float(best['median_speedup'], 3)}x median speedup."
        )
    lines.append("")
    lines.append(image("fp64_newton_bucket_total_speedup_heatmap.svg", "fp64 Newton Bucket Heatmap"))
    lines.append("")
    for bucket in _row_order("fp64_newton_bucket", newton_best.keys()):
        best_rows = newton_best.get(bucket, [])
        if not best_rows:
            continue
        best = best_rows[0]
        lines.append(
            f"- `fp64_newton_bucket={bucket}`: `{best['level']}` gives "
            f"{_fmt_float(best['median_speedup'], 3)}x median speedup."
        )
    lines.append("")

    lines.append("## 3. Stage Total-Time Analysis")
    lines.append("")
    lines.append(image("overall_stage_total_speedup_heatmap.svg", "Overall Stage Total Speedup"))
    lines.append("")
    lines.append(image("overall_stage_total_saved_ms_heatmap.svg", "Overall Stage Total Saved ms"))
    lines.append("")
    lines.append(image("stage_saved_ms_overall.svg", "Overall Stage Saved Milliseconds"))
    lines.append("")
    lines.extend(stage_lines)
    lines.append("")
    for scenario in _row_order("scenario", scenario_best.keys()):
        for filename, alt in [
            (f"scenario_stage_saved_ms_{_sanitize_token(scenario)}.svg", f"Stage Saved for {scenario}"),
            (f"scenario_stage_total_speedup_{_sanitize_token(scenario)}.svg", f"Stage Total Speedup for {scenario}"),
        ]:
            if filename not in chart_lookup:
                continue
            lines.append(f"### `{scenario}`")
            lines.append("")
            lines.append(image(filename, alt))
            lines.append("")
    lines.extend(stage_leaderboard_lines or ["- No stage-level leaderboard entries were available."])
    lines.append("")

    lines.append("## 4. Stage Per-Iter Analysis")
    lines.append("")
    lines.append(image("overall_stage_per_newton_heatmap.svg", "Overall Stage Per-Newton"))
    lines.append("")
    lines.append(image("overall_solver_stage_per_pcg_heatmap.svg", "Overall Solver Stage Per-PCG"))
    lines.append("")
    if "overall_solver_stage_per_pcg_heatmap.svg" not in chart_lookup:
        lines.append(
            "- Solver-stage per-PCG heatmaps were omitted because the stable samples carried zero PCG iteration counts in this run."
        )
        lines.append("")
    for scenario in _row_order("scenario", scenario_best.keys()):
        for filename, alt in [
            (f"scenario_stage_per_newton_{_sanitize_token(scenario)}.svg", f"Stage Per-Newton for {scenario}"),
            (f"scenario_solver_stage_per_pcg_{_sanitize_token(scenario)}.svg", f"Solver Stage Per-PCG for {scenario}"),
        ]:
            if filename not in chart_lookup:
                continue
            lines.append(f"### `{scenario}`")
            lines.append("")
            lines.append(image(filename, alt))
            lines.append("")

    lines.append("## 5. Iteration Analysis")
    lines.append("")
    lines.append(image("newton_count_ratio_heatmap.svg", "Newton Count Ratio"))
    lines.append("")
    lines.append(image("pcg_count_ratio_heatmap.svg", "PCG Count Ratio"))
    lines.append("")
    if "pcg_count_ratio_heatmap.svg" not in chart_lookup:
        lines.append(
            "- PCG count ratios were omitted because the stable samples reported zero PCG totals throughout this run."
        )
        lines.append("")
    lines.extend(iteration_lines or ["- No iteration-count comparisons had stable support."])
    lines.append("")

    lines.append("## 6. Strict Dominance")
    lines.append("")
    lines.append(image("path_dominance_matrix.svg", "Path Dominance Matrix"))
    lines.append("")
    lines.extend(dominance_lines)
    lines.append("")

    lines.append("## 7. Failure And Advisory Profile")
    lines.append("")
    lines.append(image("path_failure_breakdown_detailed.svg", "Detailed Path Failure Breakdown"))
    lines.append("")
    lines.append("Top detailed hard-failure reasons:")
    lines.append("")
    lines.extend(top_failure_reason_lines or ["- None"])
    lines.append("")
    lines.extend(failure_asset_lines)
    lines.append("")
    lines.append("Known search-direction failures:")
    lines.append("")
    lines.extend(
        _find_asset_failure_mentions(
            run_data,
            ["rigid_ipc_double_pendulum", "rigid_ipc_punching_press_loose", "rigid_ipc_cube_on_edge"],
            failure_bucket="search direction",
        )
    )
    lines.append("")
    lines.append("Baseline-unstable reference asset:")
    lines.append("")
    lines.extend(
        _find_asset_failure_mentions(
            run_data,
            ["rigid_ipc_minsep_packing"],
        )
    )
    lines.append("")
    lines.append("Highest fidelity drift among strict-stable samples:")
    lines.append("")
    lines.extend(quality_risk_lines or ["- None"])
    lines.append("")
    if advisory_rows:
        lines.append("Line-search advisory events:")
        lines.append("")
        for row in advisory_rows[:20]:
            lines.append(
                f"- `{row['asset']}` `{row['level']}` `{row['mode']}`: {row['reason_code']} ({row['stage']})"
            )
        lines.append("")

    lines.append("## 8. Representative Standard Charts")
    lines.append("")
    standard_chart_filenames = [
        entry["filename"]
        for entry in chart_entries
        if entry.get("chart_type") == "standard_report"
    ]
    if standard_chart_filenames:
        for filename in standard_chart_filenames:
            lines.append(image(filename, filename))
            lines.append("")
    else:
        lines.append("No standard report charts were copied into this analysis directory.")
        lines.append("")

    lines.append("## 9. Supplemental Appendix")
    lines.append("")
    if supplemental_rows:
        lines.append(
            "The supplemental 98-asset run is used only as trend reference for under-covered families. "
            "It does not participate in primary recommendations, best-path claims, or strict dominance."
        )
        lines.append("")
        lines.append(
            image(
                "supplemental_family_total_speedup_heatmap.svg",
                "Supplemental Family Heatmap",
            )
        )
        lines.append("")
        lines.append("| Family | Path | Median speedup | Support | Failure rate |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in sorted(
            supplemental_rows,
            key=lambda item: (
                item["slice_name"],
                -(item["median_speedup"] or 0.0),
                item["failure_rate"],
                item["level"],
            ),
        )[:20]:
            lines.append(
                f"| `{row['slice_name']}` | `{row['level']}` | {_fmt_float(row['median_speedup'], 3)} | "
                f"{row['support_count']} | {_fmt_pct(row['failure_rate'])} |"
            )
    else:
        lines.append("No supplemental run was provided for this analysis.")
    lines.append("")
    return "\n".join(lines)


def _supplemental_family_rows(
    primary_rows: Sequence[Dict[str, Any]],
    supplemental_run: Dict[str, Any] | None,
    min_support_for_claim: int,
) -> List[Dict[str, Any]]:
    if supplemental_run is None:
        return []
    primary_family_support: Dict[str, int] = defaultdict(int)
    for row in primary_rows:
        if row["slice_type"] != "scenario_family":
            continue
        primary_family_support[row["slice_name"]] = max(
            primary_family_support[row["slice_name"]], int(row["stable_asset_count"])
        )
    fp64_cost_bucket, fp64_newton_bucket = _derive_asset_buckets(supplemental_run)
    membership = _build_asset_slice_membership(
        supplemental_run["asset_meta"], fp64_cost_bucket, fp64_newton_bucket
    )
    supplemental_rows = _build_path_totals_by_slice(
        supplemental_run, membership, min_support_for_claim
    )
    filtered = []
    for row in supplemental_rows:
        if row["slice_type"] != "scenario_family":
            continue
        if primary_family_support.get(row["slice_name"], 0) >= 2:
            continue
        filtered.append(row)
    return filtered


def _assert_pcg_counts_present(run_data: Dict[str, Any]) -> None:
    has_pcg = False
    for asset in run_data["asset_meta"]:
        for level in run_data["levels"]:
            row = run_data["iteration_summary_index"].get((asset, level, "pcg_iteration_count"))
            if not row:
                continue
            if (_safe_float(row.get("total_count")) or 0.0) > 0.0:
                has_pcg = True
                break
        if has_pcg:
            break
    if not has_pcg:
        raise RuntimeError(
            "No non-zero pcg_iteration_count totals were found in this run. "
            "The build roots likely do not include the latest stable PCG counter export."
        )


def _assert_basic_counts(
    run_data: Dict[str, Any],
    asset_stability_rows: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    worker_results = sum(
        1 for payload in run_data["status"].values() if payload.get("worker_result") is not None
    )
    hard_failure_count = len(_hard_failures(run_data))
    advisory_count = len(_advisory_failures(run_data))
    stable_by_asset = defaultdict(int)
    compare_levels = run_data["compare_levels"]
    for row in asset_stability_rows:
        if row["level"] == "fp64":
            continue
        if row["strict_stable"]:
            stable_by_asset[row["asset"]] += 1
    complete_stable_assets = sum(
        1 for _asset, count in stable_by_asset.items() if count == len(compare_levels)
    )
    near_stable_assets = sum(
        1 for _asset, count in stable_by_asset.items() if count == len(compare_levels) - 1
    )
    return {
        "worker_results": worker_results,
        "hard_failures": hard_failure_count,
        "advisories": advisory_count,
        "complete_stable_assets": complete_stable_assets,
        "near_stable_assets": near_stable_assets,
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_root / "reports" / "deep_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_catalog = _load_full_catalog()
    primary_run = _load_run(run_root, full_catalog)
    primary_run["line_search_policy"] = args.line_search_policy
    supplemental_run = None
    if args.supplemental_run_root:
        supplemental_run = _load_run(Path(args.supplemental_run_root).resolve(), full_catalog)
        supplemental_run["line_search_policy"] = args.line_search_policy

    fp64_cost_bucket, fp64_newton_bucket = _derive_asset_buckets(primary_run)
    membership = _build_asset_slice_membership(
        primary_run["asset_meta"], fp64_cost_bucket, fp64_newton_bucket
    )
    path_totals_rows = _build_path_totals_by_slice(
        primary_run, membership, args.min_support_for_claim
    )
    stage_speedup_rows = _build_stage_speedups_by_slice(primary_run, membership, path_totals_rows)
    iteration_rows = _build_iteration_counts_by_slice(primary_run, membership, path_totals_rows)
    stage_total_rows = _build_stage_total_time_by_slice(primary_run, membership, path_totals_rows)
    stage_per_newton_rows = _build_stage_per_newton_by_slice(primary_run, membership, path_totals_rows)
    solver_stage_per_pcg_rows = _build_solver_stage_per_pcg_by_slice(
        primary_run, membership, path_totals_rows
    )
    stage_leaderboard_rows = _build_stage_leaderboard_by_slice(
        stage_total_rows, stage_per_newton_rows, solver_stage_per_pcg_rows
    )
    asset_stability_rows = _build_asset_stability_matrix(
        primary_run, fp64_cost_bucket, fp64_newton_bucket
    )
    failure_reason_rows = _build_failure_reason_breakdown(primary_run)
    advisory_rows = _build_line_search_advisories(primary_run)
    dominance_rows = _build_path_dominance(
        primary_run, membership, path_totals_rows, args.min_support_for_claim
    )
    supplemental_rows = _supplemental_family_rows(
        path_totals_rows, supplemental_run, args.min_support_for_claim
    )
    _assert_pcg_counts_present(primary_run)

    _write_csv(out_dir / "path_totals_by_slice.csv", path_totals_rows)
    _write_csv(out_dir / "stage_speedups_by_slice.csv", stage_speedup_rows)
    _write_csv(out_dir / "iteration_counts_by_slice.csv", iteration_rows)
    _write_csv(out_dir / "stage_total_time_by_slice.csv", stage_total_rows)
    _write_csv(out_dir / "stage_per_newton_by_slice.csv", stage_per_newton_rows)
    _write_csv(out_dir / "solver_stage_per_pcg_by_slice.csv", solver_stage_per_pcg_rows)
    _write_csv(out_dir / "stage_leaderboard_by_slice.csv", stage_leaderboard_rows)
    _write_csv(out_dir / "failure_reason_breakdown.csv", failure_reason_rows)
    _write_csv(out_dir / "line_search_advisories.csv", advisory_rows)
    _write_csv(out_dir / "path_dominance.csv", dominance_rows)
    _write_csv(out_dir / "asset_stability_matrix.csv", asset_stability_rows)

    chart_entries = _build_chart_artifacts(
        out_dir,
        primary_run,
        path_totals_rows,
        stage_speedup_rows,
        iteration_rows,
        stage_total_rows,
        stage_per_newton_rows,
        solver_stage_per_pcg_rows,
        dominance_rows,
        asset_stability_rows,
        supplemental_rows,
    )
    write_json(out_dir / "charts_manifest.json", chart_entries)

    analysis_md = _build_analysis_markdown(
        primary_run,
        path_totals_rows,
        stage_speedup_rows,
        iteration_rows,
        stage_total_rows,
        stage_per_newton_rows,
        solver_stage_per_pcg_rows,
        stage_leaderboard_rows,
        dominance_rows,
        asset_stability_rows,
        failure_reason_rows,
        advisory_rows,
        chart_entries,
        supplemental_rows,
        args.min_support_for_claim,
    )
    (out_dir / "analysis.md").write_text(analysis_md, encoding="utf-8")

    counts = _assert_basic_counts(primary_run, asset_stability_rows)
    print(f"Primary run: {run_root}")
    print(f"Output dir: {out_dir}")
    print(f"worker_results={counts['worker_results']}")
    print(f"hard_failures={counts['hard_failures']}")
    print(f"advisories={counts['advisories']}")
    print(f"complete_stable_assets={counts['complete_stable_assets']}")
    print(f"near_stable_assets={counts['near_stable_assets']}")
    print(f"charts={len(chart_entries)}")


if __name__ == "__main__":
    main()
