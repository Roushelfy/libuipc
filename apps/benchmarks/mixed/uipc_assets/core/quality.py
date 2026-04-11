from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any, Dict, List


def collect_solution_metrics(reference_dir: Path, compare_dir: Path) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[5]
    solution_metrics_py = repo_root / "apps" / "benchmarks" / "common" / "solution_metrics.py"
    spec = importlib.util.spec_from_file_location("uipc_solution_metrics", solution_metrics_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load solution_metrics from {solution_metrics_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.collect_solution_dir_metrics(reference_dir, compare_dir)


def quality_row(asset: str, level: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asset": asset,
        "level": level,
        "rel_l2_max": metrics.get("rel_l2_max", 0.0),
        "abs_linf_max": metrics.get("abs_linf_max", 0.0),
        "nan_inf_count": metrics.get("nan_inf_count", 0),
        "record_count": metrics.get("record_count", 0),
        "missing_in_compare_count": metrics.get("missing_in_compare_count", 0),
        "missing_in_reference_count": metrics.get("missing_in_reference_count", 0),
        "reference_dir": metrics.get("reference_dir"),
        "compare_dir": metrics.get("compare_dir"),
    }


def write_quality_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "asset",
        "level",
        "rel_l2_max",
        "abs_linf_max",
        "nan_inf_count",
        "record_count",
        "missing_in_compare_count",
        "missing_in_reference_count",
        "reference_dir",
        "compare_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
