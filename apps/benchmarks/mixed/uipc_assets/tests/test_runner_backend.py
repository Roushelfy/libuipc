from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.manifest import AssetSpec
from uipc_assets.core import runner


def test_run_worker_subprocess_forces_agg_backend(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "worker"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "worker_result.json").write_text(
        json.dumps({"asset": "test_asset", "mode": "perf", "level": "path6"}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run(cmd, *, check, env, cwd, capture_output, text):
        captured["env"] = env
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("MPLBACKEND", "TkAgg")

    result = runner.run_worker_subprocess(
        cli_path=tmp_path / "apps" / "benchmarks" / "mixed" / "uipc_assets" / "cli.py",
        python_exe=sys.executable,
        asset_spec=AssetSpec(name="test_asset"),
        mode="perf",
        level="path6",
        module_dir=tmp_path / "module_dir",
        pyuipc_src_dir=tmp_path / "pyuipc_src_dir",
        output_dir=output_dir,
        revision="main",
        cache_dir=tmp_path / "cache",
        dump_surface=False,
        reference_dir=None,
        visual_frames=None,
    )

    assert result["asset"] == "test_asset"
    assert captured["env"]["MPLBACKEND"] == runner.BENCHMARK_MPLBACKEND
    assert os.environ["MPLBACKEND"] == "TkAgg"
