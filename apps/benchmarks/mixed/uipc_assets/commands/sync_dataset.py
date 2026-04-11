from __future__ import annotations

from pathlib import Path

from ..core.artifacts import default_output_root, make_run_id, now_utc_iso, write_json
from ..core.runner import ensure_runtime_dependencies, fetch_remote_dataset_info
from ..core.selection import REPO_ID, load_assets_catalog, validate_catalog_coverage


def run(args) -> int:
    ensure_runtime_dependencies(require_uipc=False)

    cache_dir = args.cache_dir.resolve() if args.cache_dir else default_output_root() / "hf_cache"
    output_path = (
        args.output.resolve()
        if args.output
        else default_output_root() / make_run_id("dataset_sync") / "dataset_state.json"
    )

    coverage = validate_catalog_coverage(revision=args.revision, local_repo=args.local_repo)
    if coverage["missing"] or coverage["extra"]:
        problems = []
        if coverage["missing"]:
            problems.append("uncatalogued remote assets: " + ", ".join(coverage["missing"]))
        if coverage["extra"]:
            problems.append("catalog-only assets missing remotely: " + ", ".join(coverage["extra"]))
        raise RuntimeError("; ".join(problems))

    specs = load_assets_catalog()
    catalog_names = [spec.name for spec in specs if spec.enabled]

    snapshot_dir: Path | None = None
    remote_info = {
        "repo_id": REPO_ID,
        "remote_sha": args.revision,
        "last_modified": None,
    }

    if args.local_repo is not None:
        assets_root = args.local_repo.resolve() / "assets"
        if not assets_root.is_dir():
            raise FileNotFoundError(f"Local assets directory not found: {assets_root}")
    else:
        from huggingface_hub import snapshot_download

        remote_info = fetch_remote_dataset_info(revision=args.revision)
        revision = remote_info.get("remote_sha") or args.revision
        snapshot_dir = Path(
            snapshot_download(
                REPO_ID,
                allow_patterns=["assets/**", "README.md"],
                revision=revision,
                cache_dir=str(cache_dir),
                repo_type="dataset",
            )
        )
        remote_info["remote_sha"] = revision
        assets_root = snapshot_dir / "assets"

    assets = {}
    failures = []
    for name in catalog_names:
        path = assets_root / name
        if path.is_dir():
            assets[name] = str(path.resolve())
        else:
            failures.append({"asset": name, "reason": f"missing asset directory: {path}"})

    state = {
        "repo_id": REPO_ID,
        "remote_sha": remote_info.get("remote_sha"),
        "last_modified": remote_info.get("last_modified"),
        "checked_at": now_utc_iso(),
        "cache_dir": str(cache_dir.resolve()),
        "snapshot_dir": None if snapshot_dir is None else str(snapshot_dir.resolve()),
        "local_repo": None if args.local_repo is None else str(args.local_repo.resolve()),
        "catalog_asset_count": len(catalog_names),
        "remote_asset_count": len(coverage["remote_assets"]),
        "synced_asset_count": len(assets),
        "failure_count": len(failures),
        "missing_assets": list(coverage["missing"]),
        "extra_assets": list(coverage["extra"]),
        "failures": failures,
        "assets": assets,
    }
    write_json(output_path, state)
    print(f"sync complete: {len(assets)}/{len(catalog_names)} assets -> {output_path}")
    return 0
