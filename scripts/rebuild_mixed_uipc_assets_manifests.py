#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT / "apps" / "benchmarks" / "mixed"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from uipc_assets.core.manifest_regen import regenerate_manifest_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate mixed uipc_assets manifests from scene.py.")
    parser.add_argument(
        "--assets-root",
        type=Path,
        required=True,
        help="Path to the external uipc-assets assets directory containing <asset>/scene.py.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPO_ROOT / "apps" / "benchmarks" / "mixed" / "uipc_assets" / "manifests",
        help="Manifest directory to overwrite.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payloads = regenerate_manifest_dir(
        manifest_dir=args.manifest_dir.resolve(),
        assets_root=args.assets_root.resolve(),
    )

    scenario_counts = {}
    for spec in payloads["assets_catalog.json"]:
        scenario_counts[spec.scenario] = scenario_counts.get(spec.scenario, 0) + 1

    print(f"Regenerated manifests in {args.manifest_dir.resolve()}")
    print(f"Assets: {len(payloads['assets_catalog.json'])}")
    print(
        "Scenarios: "
        + ", ".join(f"{name}={scenario_counts[name]}" for name in sorted(scenario_counts))
    )
    for filename in sorted(payloads):
        print(f"  {filename}: {len(payloads[filename])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
