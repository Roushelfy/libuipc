#!/usr/bin/env python3
"""Analyze SOCU native contact reports.

The script consumes one or more socu_approx report JSON files, or directories
containing such files. It is intentionally milestone-neutral: use it for M6.7+
direct, direct_compare, triplet_compat, cache-hit, and benchmark tables.
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
    "native_contact_direct_eval_ms",
    "native_contact_hessian_triplet_ms",
    "native_contact_direct_compare_ms",
    "native_contact_executor_scatter_ms",
    "native_contact_hot_reduce_ms",
    "native_contact_numeric_ms",
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


def replay_path(payload: dict[str, Any]) -> str:
    runtime = payload.get("runtime_reorder", {})
    return str(runtime.get("native_contact_replay_path", "off"))


def evaluator_path(payload: dict[str, Any]) -> str:
    contact = payload.get("contact", {})
    return str(contact.get("native_contact_evaluator_path", "unknown"))


def cache_state(payload: dict[str, Any]) -> str:
    contact = payload.get("contact", {})
    if contact.get("native_contact_plan_cold_start", False):
        return "cold_rebuild"
    if contact.get("native_contact_plan_cache_hit", False):
        return "cache_hit"
    if contact.get("native_contact_plan_rebuilt_this_solve", False):
        return "partial_rebuild"
    return "off_or_unknown"


def number(section: dict[str, Any], field: str) -> float:
    value = section.get(field, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    evaluator_counts: dict[str, int] = {}
    replay_counts: dict[str, int] = {}
    cache_counts: dict[str, int] = {}
    for path, payload in reports:
        evaluator = evaluator_path(payload)
        replay = replay_path(payload)
        cache = cache_state(payload)
        evaluator_counts[evaluator] = evaluator_counts.get(evaluator, 0) + 1
        replay_counts[replay] = replay_counts.get(replay, 0) + 1
        cache_counts[cache] = cache_counts.get(cache, 0) + 1
        key = f"{cache}|{replay}|{evaluator}"
        group = groups.setdefault(
            key,
            {
                "cache_state": cache,
                "replay_path": replay,
                "evaluator_path": evaluator,
                "sample_count": 0,
                "reports": [],
                "fields": {field: [] for field in TIMING_FIELDS},
                "direct_compare_mismatch_count": [],
                "direct_unsupported_program_count": [],
                "direct_fallback_program_count": [],
            },
        )
        group["sample_count"] += 1
        group["reports"].append(str(path))
        timing = payload.get("timing", {})
        contact = payload.get("contact", {})
        for field in TIMING_FIELDS:
            group["fields"][field].append(number(timing, field))
        group["direct_compare_mismatch_count"].append(
            number(contact, "native_contact_direct_compare_mismatch_count")
        )
        group["direct_unsupported_program_count"].append(
            number(contact, "native_contact_direct_unsupported_program_count")
        )
        group["direct_fallback_program_count"].append(
            number(contact, "native_contact_direct_fallback_program_count")
        )

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
        mismatches = group["direct_compare_mismatch_count"]
        unsupported = group["direct_unsupported_program_count"]
        fallback = group["direct_fallback_program_count"]
        output_groups.append(
            {
                "cache_state": group["cache_state"],
                "replay_path": group["replay_path"],
                "evaluator_path": group["evaluator_path"],
                "sample_count": group["sample_count"],
                "fields": fields,
                "direct_compare_mismatch_count_sum": int(sum(mismatches)),
                "direct_unsupported_program_count_sum": int(sum(unsupported)),
                "direct_fallback_program_count_sum": int(sum(fallback)),
                "reports": group["reports"],
            }
        )

    output_groups.sort(
        key=lambda item: (
            item["cache_state"],
            item["replay_path"],
            item["evaluator_path"],
        )
    )
    return {
        "report_count": len(reports),
        "native_plan_report_count": replay_counts.get("native_plan", 0),
        "evaluator_path_counts": dict(sorted(evaluator_counts.items())),
        "replay_path_counts": dict(sorted(replay_counts.items())),
        "cache_state_counts": dict(sorted(cache_counts.items())),
        "groups": output_groups,
    }


def print_markdown(summary: dict[str, Any]) -> None:
    print(
        "| cache state | replay path | evaluator | count | plan build | direct eval | Hessian triplet | executor scatter | hot reduce | contact assembly | compare mismatches | direct unsupported | direct fallback |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in summary["groups"]:
        fields = group["fields"]

        def ms(field: str) -> str:
            return f"{fields[field]['median_ms']:.6g}"

        print(
            "| {cache} | {replay} | {evaluator} | {count} | {plan} | {direct} | {triplet} | {scatter} | {hot} | {contact} | {mismatch} | {unsupported} | {fallback} |".format(
                cache=group["cache_state"],
                replay=group["replay_path"],
                evaluator=group["evaluator_path"],
                count=group["sample_count"],
                plan=ms("native_contact_plan_build_ms"),
                direct=ms("native_contact_direct_eval_ms"),
                triplet=ms("native_contact_hessian_triplet_ms"),
                scatter=ms("native_contact_executor_scatter_ms"),
                hot=ms("native_contact_hot_reduce_ms"),
                contact=ms("contact_assembly_time_ms"),
                mismatch=group["direct_compare_mismatch_count_sum"],
                unsupported=group["direct_unsupported_program_count_sum"],
                fallback=group["direct_fallback_program_count_sum"],
            )
        )


def validate(
    reports: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace
) -> int:
    if args.min_samples is not None and len(reports) < args.min_samples:
        print(
            f"expected at least {args.min_samples} reports, found {len(reports)}",
            file=sys.stderr,
        )
        return 4

    if args.require_native_plan:
        offenders = [
            str(path)
            for path, payload in reports
            if replay_path(payload) != "native_plan"
        ]
        if offenders:
            print(
                "reports with unexpected replay path:\n" + "\n".join(offenders),
                file=sys.stderr,
            )
            return 5

    if args.require_evaluator:
        mismatched = [
            str(path)
            for path, payload in reports
            if evaluator_path(payload) != args.require_evaluator
        ]
        if mismatched:
            print(
                "reports with unexpected evaluator path:\n"
                + "\n".join(mismatched),
                file=sys.stderr,
            )
            return 6

    if args.require_no_triplets:
        offenders = []
        for path, payload in reports:
            timing = payload.get("timing", {})
            if number(timing, "native_contact_hessian_triplet_ms") != 0.0:
                offenders.append(str(path))
        if offenders:
            print(
                "reports have native_contact_hessian_triplet_ms > 0:\n"
                + "\n".join(offenders),
                file=sys.stderr,
            )
            return 7

    if args.require_direct_compare_zero:
        offenders = []
        for path, payload in reports:
            contact = payload.get("contact", {})
            if number(contact, "native_contact_direct_compare_mismatch_count") != 0.0:
                offenders.append(str(path))
        if offenders:
            print(
                "reports have direct compare mismatches:\n" + "\n".join(offenders),
                file=sys.stderr,
            )
            return 8

    if args.require_no_direct_fallbacks:
        offenders = []
        for path, payload in reports:
            contact = payload.get("contact", {})
            unsupported = number(
                contact, "native_contact_direct_unsupported_program_count"
            )
            fallback = number(contact, "native_contact_direct_fallback_program_count")
            if unsupported != 0.0 or fallback != 0.0:
                offenders.append(str(path))
        if offenders:
            print(
                "reports have direct unsupported or fallback programs:\n"
                + "\n".join(offenders),
                file=sys.stderr,
            )
            return 9

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze SOCU native contact socu_approx reports."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="report files or directories")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--require-native-plan", action="store_true")
    parser.add_argument(
        "--require-evaluator",
        choices=("direct", "direct_compare", "triplet_compat"),
    )
    parser.add_argument("--require-no-triplets", action="store_true")
    parser.add_argument("--require-direct-compare-zero", action="store_true")
    parser.add_argument("--require-no-direct-fallbacks", action="store_true")
    parser.add_argument("--min-samples", type=int)
    args = parser.parse_args(argv)

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in iter_report_paths(args.paths):
        payload = load_socu_report(path)
        if payload is not None:
            loaded.append((path, payload))

    if not loaded:
        print("no socu_approx reports found", file=sys.stderr)
        return 2

    validation_status = validate(loaded, args)
    if validation_status != 0:
        return validation_status

    summary = summarize(loaded)
    if args.format == "markdown":
        print_markdown(summary)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
