from __future__ import annotations

import json
import sys
from pathlib import Path

from ..core.artifacts import default_output_root, make_run_id, write_json
from ..core.builds import resolve_builds
from ..core.manifest import AssetSpec, ordered_levels, save_manifest
from ..core.report_schema import build_summary_payload, collect_report_data, write_report_files
from ..core.runner import parse_visual_frames, run_worker_subprocess, sync_assets
from ..core.selection import resolve_asset_specs, selection_payload


def _detect_failure_stage(output_dir: Path) -> str:
    stderr_log = output_dir / "worker_stderr.log"
    if not stderr_log.exists():
        return "unknown"
    text = stderr_log.read_text(encoding="utf-8", errors="replace")
    checks = [
        ("FusedPCG", "FusedPCG"),
        ("Line Search", "Line Search"),
        ("quality_metrics", "quality compare"),
        ("JSONDecodeError", "timer parse"),
        ("dump_solution_x", "solution dump"),
    ]
    for needle, stage in checks:
        if needle in text:
            return stage
    return "worker"


def _record_failure(failures: list[dict], *, asset: str, level: str, mode: str, output_dir: Path, error: Exception) -> None:
    failure = {
        "asset": asset,
        "level": level,
        "mode": mode,
        "stage": _detect_failure_stage(output_dir),
        "error": str(error),
        "output_dir": str(output_dir),
        "stdout_log": str(output_dir / "worker_stdout.log"),
        "stderr_log": str(output_dir / "worker_stderr.log"),
    }
    write_json(output_dir / "failure.json", failure)
    failures.append(failure)


def run(args) -> int:
    specs = resolve_asset_specs(
        manifest_paths=args.manifest,
        scene_names=args.scene or [],
        tags=args.tag or [],
        scenarios=args.scenario or [],
        scenario_families=args.scenario_family or [],
        select_all=args.all,
        revision=args.revision,
        local_repo=args.local_repo,
    )
    levels = ordered_levels(args.levels)
    if "fp64" not in levels:
        raise RuntimeError("--levels must include fp64")
    builds = resolve_builds(levels, args.build, args.config)

    run_root = args.run_root.resolve() if args.run_root else default_output_root() / make_run_id("mixed_assets")
    run_root.mkdir(parents=True, exist_ok=True)
    selection = selection_payload(
        specs,
        manifest_paths=args.manifest,
        scene_names=args.scene or [],
        tags=args.tag or [],
        scenarios=args.scenario or [],
        scenario_families=args.scenario_family or [],
        select_all=args.all,
    )
    write_json(run_root / "selection.json", selection)
    save_manifest(run_root / "assets_snapshot.json", specs)

    cache_dir = args.cache_dir.resolve() if args.cache_dir else default_output_root() / "hf_cache"
    dataset_state = {
        "repo_id": "MuGdxy/uipc-assets",
        "remote_sha": args.revision,
        "last_modified": None,
        "runtime_versions": {},
        "assets": {spec.name: None for spec in specs},
    }
    if not args.dry_run:
        dataset_state = sync_assets(specs, cache_dir, run_root / "dataset_state.json", revision=args.revision)
    run_meta = {
        "run_root": str(run_root),
        "levels": levels,
        "config": args.config,
        "resume": bool(getattr(args, "resume", False)),
        "cache_dir": str(cache_dir),
        "dataset_state": dataset_state,
        "modes": {
            "perf": args.perf,
            "quality": args.quality,
            "timers": args.timers,
            "visual_export": args.visual_export,
        },
        "builds": builds,
        "failures": [],
    }
    write_json(run_root / "run_meta.json", run_meta)
    write_json(run_root / "builds.json", builds)
    if args.dry_run:
        write_json(run_root / "dataset_state.json", dataset_state)

    if args.dry_run:
        print(f"dry run ready: {run_root}")
        return 0

    revision = dataset_state["remote_sha"]
    frames = parse_visual_frames(args.frames, args.frame_range)
    cli_path = Path(__file__).resolve().parents[1] / "cli.py"
    compare_levels = [level for level in levels if level != "fp64"]
    failures: list[dict] = []

    for spec in specs:
        fp64_perf_dir = run_root / "runs" / spec.name / "fp64" / "perf"
        fp64_quality_dir = run_root / "runs" / spec.name / "fp64" / "quality"
        fp64_perf_ok = False
        fp64_quality_ok = False

        if args.perf:
            try:
                run_worker_subprocess(
                    cli_path=cli_path,
                    python_exe=sys.executable,
                    asset_spec=spec,
                    mode="perf",
                    level="fp64",
                    module_dir=Path(builds["fp64"]["module_dir"]),
                    pyuipc_src_dir=Path(builds["fp64"]["pyuipc_src_dir"]),
                    output_dir=fp64_perf_dir,
                    revision=revision,
                    cache_dir=cache_dir,
                    dump_surface=args.visual_export,
                    reference_dir=None,
                    visual_frames=frames,
                    resume=args.resume,
                )
                fp64_perf_ok = True
            except Exception as exc:
                _record_failure(failures, asset=spec.name, level="fp64", mode="perf", output_dir=fp64_perf_dir, error=exc)

        if args.quality and spec.quality_enabled:
            try:
                fp64_quality = run_worker_subprocess(
                    cli_path=cli_path,
                    python_exe=sys.executable,
                    asset_spec=spec,
                    mode="quality",
                    level="fp64",
                    module_dir=Path(builds["fp64"]["module_dir"]),
                    pyuipc_src_dir=Path(builds["fp64"]["pyuipc_src_dir"]),
                    output_dir=fp64_quality_dir,
                    revision=revision,
                    cache_dir=cache_dir,
                    dump_surface=False,
                    reference_dir=None,
                    visual_frames=None,
                    resume=args.resume,
                )
                reference_dir = Path(fp64_quality["solution_dump_dir"])
                fp64_quality_ok = True
            except Exception as exc:
                _record_failure(failures, asset=spec.name, level="fp64", mode="quality", output_dir=fp64_quality_dir, error=exc)
                reference_dir = None
        else:
            reference_dir = None

        for level in compare_levels:
            if args.perf:
                perf_dir = run_root / "runs" / spec.name / level / "perf"
                try:
                    run_worker_subprocess(
                        cli_path=cli_path,
                        python_exe=sys.executable,
                        asset_spec=spec,
                        mode="perf",
                        level=level,
                        module_dir=Path(builds[level]["module_dir"]),
                        pyuipc_src_dir=Path(builds[level]["pyuipc_src_dir"]),
                        output_dir=perf_dir,
                        revision=revision,
                        cache_dir=cache_dir,
                        dump_surface=args.visual_export,
                        reference_dir=None,
                        visual_frames=frames,
                        resume=args.resume,
                    )
                except Exception as exc:
                    _record_failure(failures, asset=spec.name, level=level, mode="perf", output_dir=perf_dir, error=exc)
            if args.quality and spec.quality_enabled and reference_dir is not None and fp64_quality_ok:
                quality_dir = run_root / "runs" / spec.name / level / "quality"
                try:
                    run_worker_subprocess(
                        cli_path=cli_path,
                        python_exe=sys.executable,
                        asset_spec=spec,
                        mode="quality",
                        level=level,
                        module_dir=Path(builds[level]["module_dir"]),
                        pyuipc_src_dir=Path(builds[level]["pyuipc_src_dir"]),
                        output_dir=quality_dir,
                        revision=revision,
                        cache_dir=cache_dir,
                        dump_surface=False,
                        reference_dir=reference_dir,
                        visual_frames=None,
                        resume=args.resume,
                    )
                except Exception as exc:
                    _record_failure(failures, asset=spec.name, level=level, mode="quality", output_dir=quality_dir, error=exc)

    run_meta["failures"] = failures
    write_json(run_root / "run_meta.json", run_meta)
    report_data = collect_report_data(run_root)
    summary = build_summary_payload(report_data)
    write_report_files(run_root, summary)
    print(f"run finished: {run_root}")
    return 0
