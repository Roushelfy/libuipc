from __future__ import annotations

import sys
from pathlib import Path

from ..core.artifacts import read_json
from ..core.builds import resolve_builds
from ..core.manifest import AssetSpec
from ..core.runner import copy_visual_exports, parse_visual_frames, run_worker_subprocess


def run(args) -> int:
    run_root = args.run_root.resolve()
    run_meta = read_json(run_root / "run_meta.json")
    selection = read_json(run_root / "selection.json")
    selected_assets = {row["name"] for row in selection["assets"]}
    if args.scene:
        selected_assets &= set(args.scene)
    if args.scenario:
        selected_assets &= {
            row["name"] for row in selection["assets"] if row.get("scenario") in set(args.scenario)
        }
    if args.scenario_family:
        selected_assets &= {
            row["name"]
            for row in selection["assets"]
            if row.get("scenario_family") in set(args.scenario_family)
        }
    if not selected_assets:
        raise RuntimeError("No assets selected for export")

    levels = args.levels or run_meta["levels"]
    builds = resolve_builds(levels, args.build, args.config)
    revision = run_meta["dataset_state"]["remote_sha"]
    cache_dir = Path(run_meta["cache_dir"])
    frames = parse_visual_frames(args.frames, args.frame_range)
    cli_path = Path(__file__).resolve().parents[1] / "cli.py"

    for asset_json in selection["assets"]:
        if asset_json["name"] not in selected_assets:
            continue
        spec = AssetSpec.from_json(asset_json)
        for level in levels:
            visual_dir = run_root / "runs" / spec.name / level / "visual"
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
                    dump_surface=True,
                    reference_dir=None,
                    visual_frames=frames,
                )
            except RuntimeError:
                workspace = perf_dir / f"workspace_{spec.name}"
                if not any(workspace.rglob("scene_surface*.obj")):
                    raise
                copy_visual_exports(
                    workspace,
                    visual_dir,
                    frames,
                    source_frame_offset=spec.perf_warmup_frames,
                    frame_count=spec.frames_perf,
                )
                print(f"visual export recovered from workspace: {visual_dir}")
                continue
            print(f"visual export refreshed: {visual_dir}")
    return 0
