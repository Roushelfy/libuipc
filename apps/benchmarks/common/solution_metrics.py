#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_vector_market(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Matrix Market vector: {path}")

    rows = None
    cols = None
    values: List[float] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if rows is None:
            dims = line.split()
            if len(dims) != 2:
                raise ValueError(f"Invalid Matrix Market vector header in {path}: {line}")
            rows = int(dims[0])
            cols = int(dims[1])
            if cols != 1:
                raise ValueError(f"Expected column vector in {path}, got {rows}x{cols}")
            continue
        values.append(float(line))

    if rows is None or cols is None:
        raise ValueError(f"Missing Matrix Market dimensions in {path}")
    if len(values) != rows:
        raise ValueError(f"Vector length mismatch in {path}: expected {rows}, got {len(values)}")
    return values


def compare_solution_vectors(test: List[float], ref: List[float], eps: float = 1e-15) -> Dict[str, Any]:
    if len(test) != len(ref):
        raise ValueError(f"Vector size mismatch: test={len(test)} ref={len(ref)}")

    sq_diff = 0.0
    sq_ref = 0.0
    abs_linf = 0.0

    for t, r in zip(test, ref):
        d = t - r
        sq_diff += d * d
        sq_ref += r * r
        abs_linf = max(abs_linf, abs(d))

    denom = max(math.sqrt(sq_ref), eps)
    rel_l2 = math.sqrt(sq_diff) / denom
    nan_inf_flag = not (math.isfinite(rel_l2) and math.isfinite(abs_linf))
    return {
        "rel_l2_x": rel_l2,
        "abs_linf_x": abs_linf,
        "nan_inf_flag": nan_inf_flag,
    }


def _parse_solution_filename(name: str) -> Tuple[int | None, int | None]:
    stem = Path(name).stem
    parts = stem.split(".")
    if len(parts) != 3 or parts[0] != "x":
        return None, None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None, None


def _solution_file_map(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing solution dump dir: {root}")

    files: Dict[str, Path] = {}
    for entry in root.rglob("x.*.mtx"):
        if not entry.is_file():
            continue
        files[entry.name] = entry

    if not files:
        raise FileNotFoundError(f"No x.*.mtx files found under {root}")
    return files


def collect_solution_dir_metrics(reference_dir: Path, compare_dir: Path) -> Dict[str, Any]:
    ref_files = _solution_file_map(reference_dir)
    cmp_files = _solution_file_map(compare_dir)

    missing_in_compare = sorted(set(ref_files) - set(cmp_files))
    missing_in_reference = sorted(set(cmp_files) - set(ref_files))
    matched_names = sorted(set(ref_files) & set(cmp_files))
    if not matched_names:
        raise ValueError(
            "No matched solution dump files: "
            f"missing_in_compare={len(missing_in_compare)} "
            f"missing_in_reference={len(missing_in_reference)}"
        )

    rel_l2_max = 0.0
    abs_linf_max = 0.0
    nan_inf_count = 0
    records: List[Dict[str, Any]] = []

    for name in matched_names:
        ref_vec = load_vector_market(ref_files[name])
        cmp_vec = load_vector_market(cmp_files[name])
        metrics = compare_solution_vectors(cmp_vec, ref_vec)
        frame, newton_iter = _parse_solution_filename(name)
        record = {
            "file": name,
            "reference_file": str(ref_files[name]),
            "compare_file": str(cmp_files[name]),
            "frame": frame,
            "newton_iter": newton_iter,
            **metrics,
        }
        records.append(record)
        rel_l2_max = max(rel_l2_max, float(metrics["rel_l2_x"]))
        abs_linf_max = max(abs_linf_max, float(metrics["abs_linf_x"]))
        nan_inf_count += 1 if metrics["nan_inf_flag"] else 0

    return {
        "rel_l2_max": rel_l2_max,
        "abs_linf_max": abs_linf_max,
        "nan_inf_count": nan_inf_count,
        "record_count": len(records),
        "matched_record_count": len(records),
        "missing_in_compare_count": len(missing_in_compare),
        "missing_in_reference_count": len(missing_in_reference),
        "missing_in_compare": missing_in_compare,
        "missing_in_reference": missing_in_reference,
        "records": records,
        "reference_dir": str(reference_dir),
        "compare_dir": str(compare_dir),
    }
