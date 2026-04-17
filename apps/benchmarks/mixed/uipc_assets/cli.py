#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = SCRIPT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.commands import compare_quality, export_visuals, list_assets, render_report, resolve_selection, run_suite, sync_dataset, validate_contract
from uipc_assets.core.manifest import AssetSpec
from uipc_assets.core.runner import ensure_runtime_dependencies, parse_visual_frames, run_profile_worker
from uipc_assets.core.selection import default_manifest_path, full_manifest_path


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, action="append", default=[], help="manifest path(s)")
    parser.add_argument("--scene", nargs="*", default=None, help="explicit scene name subset")
    parser.add_argument("--tag", nargs="*", default=None, help="tag filters (AND semantics)")
    parser.add_argument("--scenario", nargs="*", default=None, help="scenario class filters")
    parser.add_argument("--scenario_family", nargs="*", default=None, help="scenario family filters")
    parser.add_argument("--all", action="store_true", help="select all discovered assets")
    parser.add_argument("--revision", default="main", help="dataset revision")
    parser.add_argument("--local_repo", type=Path, default=None, help="optional local uipc-assets repo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed-precision asset benchmark toolkit for uipc-assets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list selected assets")
    _add_selection_args(p_list)

    p_resolve = sub.add_parser("resolve", help="print resolved asset selection")
    _add_selection_args(p_resolve)

    p_run = sub.add_parser("run", help="run perf/quality/timer collection")
    _add_selection_args(p_run)
    p_run.add_argument("--levels", nargs="+", required=True, help="levels to run, must include fp64")
    p_run.add_argument("--build", action="append", default=[], help="LEVEL=BUILD_DIR (default: build/build_impl_<level>)")
    p_run.add_argument("--config", default="RelWithDebInfo")
    p_run.add_argument("--run_root", type=Path, default=None)
    p_run.add_argument("--cache_dir", type=Path, default=None)
    p_run.add_argument("--dry-run", dest="dry_run", action="store_true")
    p_run.add_argument("--resume", action="store_true", help="skip tasks with existing worker_result.json and continue the same run_root")
    p_run.add_argument("--perf", action="store_true", default=True)
    p_run.add_argument("--quality", action="store_true", default=True)
    p_run.add_argument("--timers", action="store_true", default=True)
    p_run.add_argument("--visual_export", action="store_true", help="also capture visual OBJ output during run")
    p_run.add_argument("--frames", default=None, help="visual export frames, e.g. all or 0,1,2")
    p_run.add_argument("--frame_range", default=None, help="visual export inclusive frame range start:end")

    p_sync = sub.add_parser("sync", help="prefetch all dataset assets into the local cache")
    p_sync.add_argument("--revision", default="main", help="dataset revision")
    p_sync.add_argument("--cache_dir", type=Path, default=None)
    p_sync.add_argument("--local_repo", type=Path, default=None, help="optional local uipc-assets repo")
    p_sync.add_argument("--output", type=Path, default=None, help="optional dataset_state.json output path")

    p_compare = sub.add_parser("compare", help="recompute quality metrics from an existing run")
    p_compare.add_argument("--run_root", type=Path, required=True)

    p_report = sub.add_parser("report", help="render reports from an existing run")
    p_report.add_argument("--run_root", type=Path, required=True)

    p_validate = sub.add_parser("validate", help="validate representative benchmark contract outputs")
    p_validate.add_argument("--run_root", type=Path, required=True)
    p_validate.add_argument("--levels", nargs="*", default=None, help="required levels (default: fp64 path1..path6)")
    p_validate.add_argument("--max_rel_l2", type=float, default=1.0e-5, help="strict upper bound for rel_l2_max")
    p_validate.add_argument("--max_abs_linf", type=float, default=5.0e-4, help="strict upper bound for abs_linf_max")

    p_export = sub.add_parser("export", help="export visual OBJ sequences for selected assets")
    p_export.add_argument("--run_root", type=Path, required=True)
    p_export.add_argument("--build", action="append", default=[], help="LEVEL=BUILD_DIR (default: build/build_impl_<level>)")
    p_export.add_argument("--config", default="RelWithDebInfo")
    p_export.add_argument("--scene", nargs="*", default=None)
    p_export.add_argument("--scenario", nargs="*", default=None)
    p_export.add_argument("--scenario_family", nargs="*", default=None)
    p_export.add_argument("--levels", nargs="*", default=None)
    p_export.add_argument("--frames", default=None)
    p_export.add_argument("--frame_range", default=None)

    p_worker = sub.add_parser("_worker")
    p_worker.add_argument("--asset_spec", type=Path, required=True)
    p_worker.add_argument("--mode", choices=("perf", "quality"), required=True)
    p_worker.add_argument("--level", required=True)
    p_worker.add_argument("--module_dir", type=Path, required=True)
    p_worker.add_argument("--output_dir", type=Path, required=True)
    p_worker.add_argument("--revision", required=True)
    p_worker.add_argument("--cache_dir", type=Path, required=True)
    p_worker.add_argument("--dump_surface", choices=("ON", "OFF", "on", "off"), default="OFF")
    p_worker.add_argument("--reference_dir", type=Path, default=None)
    p_worker.add_argument("--visual_frames", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "manifest", None) == [] and args.command in {"list", "resolve", "run"}:
        has_explicit_selector = any(
            [
                bool(getattr(args, "all", False)),
                bool(getattr(args, "scene", None)),
                bool(getattr(args, "tag", None)),
                bool(getattr(args, "scenario", None)),
                bool(getattr(args, "scenario_family", None)),
            ]
        )
        args.manifest = [full_manifest_path() if has_explicit_selector else default_manifest_path()]
    if args.command == "list":
        return list_assets.run(args)
    if args.command == "resolve":
        return resolve_selection.run(args)
    if args.command == "run":
        from uipc_assets.core.builds import parse_build_args

        args.build = parse_build_args(args.build)
        return run_suite.run(args)
    if args.command == "sync":
        return sync_dataset.run(args)
    if args.command == "compare":
        return compare_quality.run(args)
    if args.command == "report":
        return render_report.run(args)
    if args.command == "validate":
        return validate_contract.run(args)
    if args.command == "export":
        from uipc_assets.core.builds import parse_build_args

        args.build = parse_build_args(args.build)
        return export_visuals.run(args)
    if args.command == "_worker":
        ensure_runtime_dependencies(require_uipc=True)
        asset_spec = AssetSpec.from_json(__import__("json").loads(args.asset_spec.read_text(encoding="utf-8")))
        frames = parse_visual_frames(args.visual_frames, None)
        run_profile_worker(
            asset_spec=asset_spec,
            mode=args.mode,
            level=args.level,
            module_dir=args.module_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            revision=args.revision,
            cache_dir=args.cache_dir.resolve(),
            dump_surface=str(args.dump_surface).upper() == "ON",
            reference_dir=args.reference_dir.resolve() if args.reference_dir else None,
            visual_frames=frames,
        )
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
