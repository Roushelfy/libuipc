from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.commands import run_suite
from uipc_assets.core.manifest import AssetSpec
from uipc_assets.core.runner import SEARCH_DIRECTION_INVALID_PREFIX, extract_search_direction_message


def test_extract_search_direction_message_from_traceback() -> None:
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "worker.py", line 1, in <module>',
            "RuntimeError: search_direction_invalid: frame=12 newton=34 consecutive_hits=10 threshold=10 line_search_max_iter=8",
        ]
    )
    assert (
        extract_search_direction_message(text)
        == "search_direction_invalid: frame=12 newton=34 consecutive_hits=10 threshold=10 line_search_max_iter=8"
    )


def test_run_suite_records_search_direction_failure_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    specs = [
        AssetSpec(name="rigid_ipc_double_pendulum", frames_perf=200, frames_quality=30),
        AssetSpec(name="abd_external_force", frames_perf=20, frames_quality=10),
    ]
    calls: list[tuple[str, str, str]] = []

    def fake_run_worker_subprocess(**kwargs):
        asset_spec = kwargs["asset_spec"]
        level = kwargs["level"]
        mode = kwargs["mode"]
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        calls.append((asset_spec.name, level, mode))

        if asset_spec.name == "rigid_ipc_double_pendulum" and level == "path5" and mode == "perf":
            (output_dir / "worker_stdout.log").write_text(
                "RuntimeError: "
                f"{SEARCH_DIRECTION_INVALID_PREFIX} frame=12 newton=34 consecutive_hits=10 threshold=10 line_search_max_iter=8\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{SEARCH_DIRECTION_INVALID_PREFIX} frame=12 newton=34 consecutive_hits=10 threshold=10 line_search_max_iter=8"
            )

        result = {
            "asset": asset_spec.name,
            "mode": mode,
            "level": level,
            "solution_dump_dir": None,
        }
        if mode == "quality":
            dump_dir = output_dir / "x_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)
            result["solution_dump_dir"] = str(dump_dir)
        (output_dir / "worker_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(run_suite, "resolve_asset_specs", lambda **_: specs)
    monkeypatch.setattr(
        run_suite,
        "resolve_builds",
        lambda levels, build_overrides, config: {
            level: {
                "build_dir": str(tmp_path / "build" / level),
                "module_dir": str(tmp_path / "bin" / level),
                "pyuipc_src_dir": str(tmp_path / "py" / level),
            }
            for level in levels
        },
    )
    monkeypatch.setattr(
        run_suite,
        "selection_payload",
        lambda specs, **_: {"assets": [spec.to_json() for spec in specs]},
    )
    monkeypatch.setattr(
        run_suite,
        "sync_assets",
        lambda specs, cache_dir, state_file, revision=None: {
            "repo_id": "MuGdxy/uipc-assets",
            "remote_sha": revision or "main",
            "last_modified": None,
            "runtime_versions": {},
            "assets": {spec.name: str(tmp_path / spec.name) for spec in specs},
        },
    )
    monkeypatch.setattr(run_suite, "run_worker_subprocess", fake_run_worker_subprocess)
    monkeypatch.setattr(run_suite, "collect_report_data", lambda run_root: {"run_root": str(run_root)})
    monkeypatch.setattr(run_suite, "build_summary_payload", lambda report_data: report_data)
    monkeypatch.setattr(run_suite, "write_report_files", lambda run_root, summary: None)

    run_root = tmp_path / "run"
    args = SimpleNamespace(
        manifest=[],
        scene=None,
        tag=None,
        scenario=None,
        scenario_family=None,
        all=False,
        revision="main",
        local_repo=None,
        levels=["fp64", "path5"],
        build={},
        config="RelWithDebInfo",
        run_root=run_root,
        cache_dir=tmp_path / "cache",
        dry_run=False,
        resume=False,
        perf=True,
        quality=True,
        timers=True,
        visual_export=False,
        frames=None,
        frame_range=None,
    )

    assert run_suite.run(args) == 0

    failure = json.loads(
        (run_root / "runs" / "rigid_ipc_double_pendulum" / "path5" / "perf" / "failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["stage"] == "search direction"
    assert (run_root / "runs" / "abd_external_force" / "path5" / "perf" / "worker_result.json").exists()
    assert ("abd_external_force", "path5", "perf") in calls
