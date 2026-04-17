#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = (
    REPO_ROOT / "src" / "backends" / "cuda_mixed" / "mixed_precision" / "precision_contracts.json"
)
SOURCE_SUFFIXES = {".cu", ".cpp", ".h", ".hpp", ".inl"}


def _load_contract() -> dict:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    seen_ids: set[str] = set()
    for component in payload.get("components", []):
        component_id = str(component["id"])
        if component_id in seen_ids:
            raise RuntimeError(f"duplicate component id in {CONTRACT_PATH}: {component_id}")
        seen_ids.add(component_id)
        for rel_path in component.get("files", []):
            if not (REPO_ROOT / rel_path).exists():
                raise RuntimeError(
                    f"component {component_id} references missing file: {rel_path}"
                )

    return payload


def _source_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _line_text(text: str, index: int) -> str:
    begin = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end == -1:
        end = len(text)
    return text[begin:end]


def _match_message(path: Path, text: str, match: re.Match[str], reason: str) -> str:
    rel_path = path.relative_to(REPO_ROOT)
    line_no = _line_number(text, match.start())
    line = _line_text(text, match.start()).strip()
    return f"{rel_path}:{line_no}: {reason}: {line}"


def _scan_file_rules(contract: dict, errors: list[str]) -> None:
    for rule in contract.get("file_rules", []):
        path = REPO_ROOT / rule["path"]
        text = path.read_text(encoding="utf-8", errors="replace")

        for required in rule.get("required_patterns", []):
            regex = re.compile(required["pattern"], re.MULTILINE)
            if regex.search(text) is None:
                errors.append(
                    f"{path.relative_to(REPO_ROOT)}: missing required pattern: {required['reason']}"
                )

        for forbidden in rule.get("forbidden_patterns", []):
            regex = re.compile(forbidden["pattern"], re.MULTILINE)
            for match in regex.finditer(text):
                errors.append(_match_message(path, text, match, forbidden["reason"]))

        for watcher in rule.get("line_watch_patterns", []):
            regex = re.compile(watcher["pattern"], re.MULTILINE)
            allow_patterns = [
                re.compile(pattern, re.MULTILINE)
                for pattern in watcher.get("allow_line_patterns", [])
            ]
            for match in regex.finditer(text):
                line = _line_text(text, match.start())
                if any(pattern.search(line) for pattern in allow_patterns):
                    continue
                errors.append(_match_message(path, text, match, watcher["reason"]))


def _scan_global_rules(contract: dict, errors: list[str]) -> None:
    for root_rel in contract.get("source_roots", []):
        root = REPO_ROOT / root_rel
        files = _source_files(root)
        for rule in contract.get("global_forbidden_patterns", []):
            regex = re.compile(rule["pattern"], re.MULTILINE)
            for path in files:
                text = path.read_text(encoding="utf-8", errors="replace")
                for match in regex.finditer(text):
                    errors.append(_match_message(path, text, match, rule["reason"]))


def main() -> int:
    contract = _load_contract()
    errors: list[str] = []

    _scan_global_rules(contract, errors)
    _scan_file_rules(contract, errors)

    if errors:
        print("cuda_mixed precision contract scan failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    component_count = len(contract.get("components", []))
    print(
        f"cuda_mixed precision contract scan passed ({component_count} documented components)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
