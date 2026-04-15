from __future__ import annotations

import json
import sys
import types
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.manifest import AssetSpec
from uipc_assets.core import runner


def _install_fake_uipc(monkeypatch, tmp_path: Path) -> dict[str, int]:
    recorder = {
        "advance_calls": 0,
        "retrieve_calls": 0,
        "collect_calls": 0,
    }

    class FakeConfig:
        def __init__(self):
            self.values: dict[str, list[int | bool | float | str]] = {}

        def find(self, path: str):
            return self.values.get(path)

        def create(self, path: str, value):
            self.values[path] = [value]

    class FakeScene:
        @staticmethod
        def default_config():
            return {}

        def __init__(self, _cfg):
            self._config = FakeConfig()

        def config(self):
            return self._config

    class FakeEngine:
        def __init__(self, backend: str, workspace: str):
            self.backend = backend
            self.workspace = workspace

    class FakeWorld:
        def __init__(self, engine: FakeEngine):
            self.engine = engine
            self.scene: FakeScene | None = None

        def init(self, scene: FakeScene):
            self.scene = scene
            Path(self.engine.workspace).mkdir(parents=True, exist_ok=True)

        def advance(self):
            recorder["advance_calls"] += 1

        def retrieve(self):
            recorder["retrieve_calls"] += 1
            assert self.scene is not None
            dump_solution = self.scene.config().values.get("extras/debug/dump_solution_x", [0])[0]
            if dump_solution:
                dump_dir = Path(self.engine.workspace) / "solver"
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"x.{recorder['retrieve_calls']:04d}.mtx"
                dump_path.write_text("%%MatrixMarket", encoding="utf-8")

        def is_valid(self):
            return True

    class FakeSimulationStats:
        def __init__(self):
            self._frames = []

        @property
        def num_frames(self):
            return len(self._frames)

        def collect(self):
            recorder["collect_calls"] += 1
            self._frames.append(
                {
                    "name": "Pipeline",
                    "duration": 0.001,
                    "count": 0,
                    "children": [
                        {
                            "name": "Newton Iteration",
                            "duration": 0.001,
                            "count": 1,
                            "children": [
                                {
                                    "name": "Line Search Iteration",
                                    "duration": 0.0005,
                                    "count": 1,
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            )

    class FakeLogger:
        class Level:
            Warn = "warn"

        @staticmethod
        def set_level(_level):
            return None

    def fake_save_result(result: dict, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        benchmark = {
            "name": result["name"],
            "num_frames": result["num_frames"],
            "wall_time": result["wall_time"],
            "summary": result["summary"],
            "workspace": result.get("workspace"),
        }
        (out / "benchmark.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
        (out / "timer_frames.json").write_text(json.dumps(result["timer_frames"], indent=2), encoding="utf-8")

    uipc_module = types.ModuleType("uipc")
    uipc_module.default_config = lambda: {}
    uipc_module.init = lambda _cfg: None
    uipc_module.view = lambda slot: slot
    uipc_module.Engine = FakeEngine
    uipc_module.Logger = FakeLogger
    uipc_module.Scene = FakeScene
    uipc_module.World = FakeWorld

    uipc_stats = types.ModuleType("uipc.stats")
    uipc_stats.SimulationStats = FakeSimulationStats

    uipc_profile = types.ModuleType("uipc.profile")
    uipc_profile._save_result = fake_save_result

    monkeypatch.setitem(sys.modules, "uipc", uipc_module)
    monkeypatch.setitem(sys.modules, "uipc.stats", uipc_stats)
    monkeypatch.setitem(sys.modules, "uipc.profile", uipc_profile)
    monkeypatch.setattr(runner, "load_asset_scene", lambda *args, **kwargs: tmp_path / "asset")
    monkeypatch.setattr(runner, "collect_solution_metrics", lambda *args, **kwargs: {})
    return recorder


def test_run_profile_worker_perf_warmup_collects_profile_only(tmp_path: Path, monkeypatch) -> None:
    recorder = _install_fake_uipc(monkeypatch, tmp_path)
    asset_spec = AssetSpec(name="warm_scene", frames_perf=3, perf_warmup_frames=2, frames_quality=2)
    output_dir = tmp_path / "perf"

    result = runner.run_profile_worker(
        asset_spec=asset_spec,
        mode="perf",
        level="path8",
        module_dir=tmp_path / "module_dir",
        output_dir=output_dir,
        revision="main",
        cache_dir=tmp_path / "cache",
        dump_surface=False,
        reference_dir=None,
        visual_frames=None,
    )

    assert recorder["advance_calls"] == 5
    assert recorder["retrieve_calls"] == 5
    assert recorder["collect_calls"] == 3
    assert result["frames"] == 3
    assert result["warmup_frames"] == 2

    benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["num_frames"] == 3
    assert benchmark["warmup_frames"] == 2
    assert benchmark["warmup_wall_time_s"] >= 0.0
    assert benchmark["end_to_end_wall_time_s"] >= benchmark["wall_time"]

    timer_frames = json.loads((output_dir / "timer_frames.json").read_text(encoding="utf-8"))
    assert len(timer_frames) == 3
    stage_summary = json.loads((output_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert stage_summary["frame_count"] == 3


def test_run_profile_worker_quality_ignores_perf_warmup_frames(tmp_path: Path, monkeypatch) -> None:
    recorder = _install_fake_uipc(monkeypatch, tmp_path)
    asset_spec = AssetSpec(name="quality_scene", frames_perf=5, perf_warmup_frames=4, frames_quality=2)
    output_dir = tmp_path / "quality"

    result = runner.run_profile_worker(
        asset_spec=asset_spec,
        mode="quality",
        level="path8",
        module_dir=tmp_path / "module_dir",
        output_dir=output_dir,
        revision="main",
        cache_dir=tmp_path / "cache",
        dump_surface=False,
        reference_dir=None,
        visual_frames=None,
    )

    assert recorder["advance_calls"] == 2
    assert recorder["retrieve_calls"] == 2
    assert recorder["collect_calls"] == 2
    assert result["frames"] == 2
    assert result["warmup_frames"] == 0
    assert Path(result["solution_dump_dir"]).is_dir()

    benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["num_frames"] == 2
    assert benchmark["warmup_frames"] == 0


def test_copy_visual_exports_uses_profile_local_frame_numbers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    visual_dir = tmp_path / "visual"
    workspace.mkdir(parents=True, exist_ok=True)
    for frame in (4, 5, 15, 21):
        (workspace / f"scene_surface{frame:04d}.obj").write_text(f"frame={frame}", encoding="utf-8")

    manifest = runner.copy_visual_exports(
        workspace,
        visual_dir,
        {0, 10},
        source_frame_offset=5,
        frame_count=20,
    )

    exported_frames = sorted(row["frame"] for row in manifest["files"])
    assert exported_frames == [0, 10]
    assert sorted(row["source_frame"] for row in manifest["files"]) == [5, 15]
    assert (visual_dir / "frame_0000" / "scene_surface0000.obj").exists()
    assert (visual_dir / "frame_0010" / "scene_surface0010.obj").exists()
    assert not (visual_dir / "frame_0016").exists()
