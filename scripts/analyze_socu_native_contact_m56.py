#!/usr/bin/env python3
"""Summarize SOCU native contact M5.6 timing reports.

The script consumes one or more socu_approx report JSON files, or directories
containing such files, and groups them by cache state and replay path. It is
intended for cold/cache-hit timing tables before M6 hot-reduce work starts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


TIMING_FIELDS = (
    "native_contact_plan_build_ms",
    "native_contact_side_plan_build_ms",
    "native_contact_program_plan_build_ms",
    "native_contact_hessian_triplet_ms",
    "native_contact_executor_scatter_ms",
    "native_contact_numeric_ms",
    "native_contact_hot_reduce_ms",
    "contact_assembly_time_ms",
)


def iter_report_paths(paths: list[Path]) -> list[Path]:
    reports: list[Path] = []
    for path in paths:
        if path.is_dir():
            reports.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            reports.append(path)
        else:
            raise FileNotFoundError(path)
    return reports


def load_socu_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("solver") != "socu_approx":
        return None
    if "timing" not in payload or "contact" not in payload:
        return None
    return payload


def cache_state(contact: dict[str, Any]) -> str:
    if contact.get("native_contact_plan_cold_start", False):
        return "cold_rebuild"
    if contact.get("native_contact_plan_cache_hit", False):
        return "cache_hit"
    if contact.get("native_contact_plan_rebuilt_this_solve", False):
        return "partial_rebuild"
    return "off_or_unknown"


def replay_path(payload: dict[str, Any]) -> str:
    runtime = payload.get("runtime_reorder", {})
    return str(runtime.get("native_contact_replay_path", "off"))


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    cache_state_counts: dict[str, int] = {}
    replay_path_counts: dict[str, int] = {}
    for path, payload in reports:
        timing = payload.get("timing", {})
        contact = payload.get("contact", {})
        state = cache_state(contact)
        replay = replay_path(payload)
        cache_state_counts[state] = cache_state_counts.get(state, 0) + 1
        replay_path_counts[replay] = replay_path_counts.get(replay, 0) + 1
        key = f"{state}|{replay}"
        group = groups.setdefault(
            key,
            {
                "cache_state": state,
                "replay_path": replay,
                "sample_count": 0,
                "reports": [],
                "fields": {field: [] for field in TIMING_FIELDS},
            },
        )
        group["sample_count"] += 1
        group["reports"].append(str(path))
        for field in TIMING_FIELDS:
            value = timing.get(field, 0.0)
            if isinstance(value, (int, float)):
                group["fields"][field].append(float(value))

    output_groups: list[dict[str, Any]] = []
    for group in groups.values():
        fields = {
            field: {
                "median_ms": median(values),
                "mean_ms": mean(values),
                "sample_count": len(values),
            }
            for field, values in group["fields"].items()
        }
        output_groups.append(
            {
                "cache_state": group["cache_state"],
                "replay_path": group["replay_path"],
                "sample_count": group["sample_count"],
                "fields": fields,
                "reports": group["reports"],
            }
        )

    output_groups.sort(key=lambda item: (item["cache_state"], item["replay_path"]))
    return {
        "report_count": len(reports),
        "native_plan_report_count": replay_path_counts.get("native_plan", 0),
        "cache_state_counts": dict(sorted(cache_state_counts.items())),
        "replay_path_counts": dict(sorted(replay_path_counts.items())),
        "groups": output_groups,
    }


def print_markdown(summary: dict[str, Any]) -> None:
    print("| cache state | replay path | count | plan build | Hessian triplet | executor scatter | native numeric | contact assembly |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in summary["groups"]:
        fields = group["fields"]
        def ms(field: str) -> str:
            return f"{fields[field]['median_ms']:.6g}"

        print(
            "| {cache} | {replay} | {count} | {plan} | {hess} | {exec} | {num} | {contact} |".format(
                cache=group["cache_state"],
                replay=group["replay_path"],
                count=group["sample_count"],
                plan=ms("native_contact_plan_build_ms"),
                hess=ms("native_contact_hessian_triplet_ms"),
                exec=ms("native_contact_executor_scatter_ms"),
                num=ms("native_contact_numeric_ms"),
                contact=ms("contact_assembly_time_ms"),
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize SOCU native contact M5.6 cold/cache-hit timing reports."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="report JSON files or directories")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--require-native-plan",
        action="store_true",
        help="fail if no report uses native_contact_replay_path=native_plan",
    )
    args = parser.parse_args(argv)

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in iter_report_paths(args.paths):
        payload = load_socu_report(path)
        if payload is not None:
            loaded.append((path, payload))

    if not loaded:
        print("no socu_approx reports found", file=sys.stderr)
        return 2

    if args.require_native_plan and not any(
        replay_path(payload) == "native_plan" for _, payload in loaded
    ):
        print("no native_plan replay reports found", file=sys.stderr)
        return 3

    summary = summarize(loaded)
    if args.format == "markdown":
        print_markdown(summary)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
