from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def script_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_root() -> Path:
    return repo_root() / "output" / "benchmarks" / "mixed" / "uipc_assets"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_root_path(run_id: str | None = None) -> Path:
    root = default_output_root()
    return root / (run_id or make_run_id("assets"))

