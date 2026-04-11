from __future__ import annotations

import json

from ..core.selection import resolve_asset_specs, selection_payload


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
    print(
        json.dumps(
            selection_payload(
                specs,
                manifest_paths=args.manifest,
                scene_names=args.scene or [],
                tags=args.tag or [],
                scenarios=args.scenario or [],
                scenario_families=args.scenario_family or [],
                select_all=args.all,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0
