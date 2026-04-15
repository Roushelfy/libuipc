from __future__ import annotations

from ..core.selection import list_remote_assets, resolve_asset_specs


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
    available = set()
    if args.all or args.local_repo is not None:
        available = set(list_remote_assets(revision=args.revision, local_repo=args.local_repo))
    for spec in specs:
        origin = "dataset" if spec.name in available else "manifest-only"
        print(
            f"{spec.name}\t"
            f"scenario={spec.scenario or '-'}\t"
            f"family={spec.scenario_family or '-'}\t"
            f"tags={','.join(spec.tags or []) or '-'}\t"
            f"quality={'on' if spec.quality_enabled else 'off'}\t"
            f"perf={spec.frames_perf}\twarmup={spec.perf_warmup_frames}\tquality_frames={spec.frames_quality}\t"
            f"origin={origin}\t"
            f"notes={spec.notes}"
        )
    return 0
