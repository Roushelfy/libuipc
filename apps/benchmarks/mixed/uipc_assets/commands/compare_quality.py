from __future__ import annotations

from pathlib import Path

from ..core.artifacts import read_json, write_json
from ..core.quality import collect_solution_metrics


def run(args) -> int:
    run_root = args.run_root.resolve()
    run_meta = read_json(run_root / "run_meta.json")
    selection = read_json(run_root / "selection.json")
    levels = [level for level in run_meta["levels"] if level != "fp64"]
    for asset in selection["assets"]:
        asset_name = asset["name"]
        fp64_dir = run_root / "runs" / asset_name / "fp64" / "quality" / "x_dumps"
        if not fp64_dir.exists():
            continue
        for level in levels:
            quality_dir = run_root / "runs" / asset_name / level / "quality"
            compare_dir = quality_dir / "x_dumps"
            if not compare_dir.exists():
                continue
            metrics = collect_solution_metrics(fp64_dir, compare_dir)
            metrics["reference_dir"] = str(fp64_dir)
            metrics["compare_dir"] = str(compare_dir)
            write_json(quality_dir / "quality_metrics.json", metrics)
    print(f"quality comparison updated: {run_root}")
    return 0

