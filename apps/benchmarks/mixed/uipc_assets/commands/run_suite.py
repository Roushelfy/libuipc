from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..core.artifacts import default_output_root, make_run_id, write_json
from ..core.builds import resolve_builds
from ..core.manifest import AssetSpec, ordered_levels, save_manifest
from ..core.report_schema import build_summary_payload, collect_report_data, write_report_files
from ..core.runner import SEARCH_DIRECTION_INVALID_PREFIX, parse_visual_frames, run_worker_subprocess, sync_assets
from ..core.selection import resolve_asset_specs, selection_payload


def _load_failure_log_text(output_dir: Path) -> str:
    log_texts: list[str] = []
    for log_path in (output_dir / "worker_stderr.log", output_dir / "worker_stdout.log"):
        if log_path.exists():
            log_texts.append(log_path.read_text(encoding="utf-8", errors="replace"))
    failure_json = output_dir / "failure.json"
    if failure_json.exists():
        try:
            failure_payload = json.loads(failure_json.read_text(encoding="utf-8", errors="replace"))
            error_text = failure_payload.get("error")
            if isinstance(error_text, str) and error_text.strip():
                log_texts.append(error_text)
        except json.JSONDecodeError:
            pass
    return "\n".join(log_texts)


def _extract_exit_code(error_text: str) -> int | None:
    match = re.search(r"\(exit=(\d+)\)", error_text)
    if match is None:
        return None
    return int(match.group(1))


def _detect_failure_classification(output_dir: Path, error: Exception | None = None) -> dict:
    text = _load_failure_log_text(output_dir)
    error_text = "" if error is None else str(error)
    combined_text = "\n".join(part for part in (text, error_text) if part)
    exit_code = _extract_exit_code(error_text) or _extract_exit_code(combined_text)

    def result(stage: str, reason_code: str, reason: str) -> dict:
        return {
            "stage": stage,
            "reason_code": reason_code,
            "reason": reason,
            "exit_code": exit_code,
        }

    stage_checks = [
        (
            SEARCH_DIRECTION_INVALID_PREFIX,
            result("search direction", "search_direction_invalid", "search direction invalid"),
        ),
        ("FusedPCG", result("FusedPCG", "fused_pcg", "fused PCG failure")),
        ("quality_metrics", result("quality compare", "quality_compare", "quality compare failure")),
        ("JSONDecodeError", result("timer parse", "timer_parse", "timer frame parse failure")),
        (
            "dump_solution_x",
            result("solution dump", "solution_dump", "solution dump failure"),
        ),
        (
            "Solution x dump not found",
            result("solution dump", "solution_dump_missing", "solution dump missing"),
        ),
    ]
    for needle, classification in stage_checks:
        if needle in combined_text:
            return classification

    if any(
        needle in combined_text
        for needle in ("SimplicialSurfaceIntersectionCheck", "SimplicialSurfaceDistanceCheck", "Intersection detected")
    ):
        return result("worker", "sanity_check", "sanity check failure")

    if any(
        needle in combined_text
        for needle in ("Residual is nan", "check_rz_nan_inf", "norm(z) = nan", "norm(r) = inf")
    ):
        return result("worker", "linear_solver_nan_inf", "linear solver nan/inf")

    if (
        ("LineSearcher::compute_energy" in combined_text or "Energy [" in combined_text)
        and (" is inf" in combined_text or " is nan" in combined_text)
    ):
        return result("worker", "energy_nan_inf", "energy nan/inf")

    if "worker simulation became invalid" in combined_text:
        return result("worker", "world_invalid", "world became invalid")

    if (
        "Traceback (most recent call last):" in combined_text
        and "matplotlib" in combined_text
        and ("summary_report" in combined_text or "_draw_system_dependency_graph" in combined_text)
    ):
        return result("worker", "python_report_generation", "python report generation failure")

    if "No module named" in combined_text:
        return result("worker", "python_import_error", "python import error")

    if "FileNotFoundError" in combined_text:
        return result("worker", "missing_file", "missing file")

    if "Traceback (most recent call last):" in combined_text:
        return result("worker", "python_exception", "python exception")

    if "Assertion " in combined_text:
        return result("worker", "native_assertion", "native assertion")

    if exit_code == 3221225477:
        return result("worker", "native_access_violation", "native access violation")
    if exit_code == 3221226505:
        return result("worker", "native_stack_buffer_overrun", "native stack buffer overrun")
    if exit_code is not None:
        return result("worker", f"exit_{exit_code}", f"worker exited with code {exit_code}")
    if not combined_text.strip():
        return result("unknown", "unknown", "unknown failure")
    return result("worker", "worker", "worker failure")


def _detect_failure_stage(output_dir: Path) -> str:
    return _detect_failure_classification(output_dir)["stage"]


def _record_failure(failures: list[dict], *, asset: str, level: str, mode: str, output_dir: Path, error: Exception) -> None:
    classification = _detect_failure_classification(output_dir, error)
    failure = {
        "asset": asset,
        "level": level,
        "mode": mode,
        "stage": classification["stage"],
        "reason_code": classification["reason_code"],
        "reason": classification["reason"],
        "exit_code": classification["exit_code"],
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
