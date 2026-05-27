#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FileDiff:
    path: str
    status: str
    details: list[str]


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(obj: Any) -> Any:
    """
    Normalize YAML data for stable comparison.

    - Dict keys are sorted recursively.
    - Lists are kept in order, because order often matters in configs.
    """
    if isinstance(obj, dict):
        return {key: normalize(obj[key]) for key in sorted(obj)}
    if isinstance(obj, list):
        return [normalize(item) for item in obj]
    return obj


def collect_yaml_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".yml", ".yaml"}:
            rel = path.relative_to(root).as_posix()
            files[rel] = path

    return files


def compare_values(old: Any, new: Any, prefix: str = "") -> list[str]:
    diffs: list[str] = []

    if type(old) is not type(new):
        diffs.append(
            f"{prefix or '<root>'}: type changed "
            f"{type(old).__name__} -> {type(new).__name__}"
        )
        return diffs

    if isinstance(old, dict):
        old_keys = set(old)
        new_keys = set(new)

        for key in sorted(old_keys - new_keys):
            key_path = f"{prefix}.{key}" if prefix else str(key)
            diffs.append(f"{key_path}: removed key")

        for key in sorted(new_keys - old_keys):
            key_path = f"{prefix}.{key}" if prefix else str(key)
            diffs.append(f"{key_path}: added key = {new[key]!r}")

        for key in sorted(old_keys & new_keys):
            key_path = f"{prefix}.{key}" if prefix else str(key)
            diffs.extend(compare_values(old[key], new[key], key_path))

        return diffs

    if isinstance(old, list):
        if len(old) != len(new):
            diffs.append(f"{prefix or '<root>'}: list length {len(old)} -> {len(new)}")

        for i, (old_item, new_item) in enumerate(zip(old, new)):
            item_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            diffs.extend(compare_values(old_item, new_item, item_path))

        if len(new) > len(old):
            for i in range(len(old), len(new)):
                item_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
                diffs.append(f"{item_path}: added item = {new[i]!r}")

        if len(old) > len(new):
            for i in range(len(new), len(old)):
                item_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
                diffs.append(f"{item_path}: removed item = {old[i]!r}")

        return diffs

    if old != new:
        diffs.append(f"{prefix or '<root>'}: {old!r} -> {new!r}")

    return diffs


def compare_dirs(old_dir: Path, new_dir: Path) -> list[FileDiff]:
    old_files = collect_yaml_files(old_dir)
    new_files = collect_yaml_files(new_dir)

    all_paths = sorted(set(old_files) | set(new_files))
    results: list[FileDiff] = []

    for rel_path in all_paths:
        old_path = old_files.get(rel_path)
        new_path = new_files.get(rel_path)

        if old_path is None:
            results.append(FileDiff(rel_path, "added_file", [f"Only exists in new: {new_path}"]))
            continue

        if new_path is None:
            results.append(FileDiff(rel_path, "removed_file", [f"Only exists in old: {old_path}"]))
            continue

        try:
            old_data = normalize(load_yaml(old_path))
            new_data = normalize(load_yaml(new_path))
        except Exception as exc:
            results.append(FileDiff(rel_path, "parse_error", [repr(exc)]))
            continue

        diffs = compare_values(old_data, new_data)

        if diffs:
            results.append(FileDiff(rel_path, "changed", diffs))
        else:
            results.append(FileDiff(rel_path, "unchanged", []))

    return results


def write_json_report(results: list[FileDiff], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(result) for result in results]

    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown_report(results: list[FileDiff], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    changed = [r for r in results if r.status != "unchanged"]

    with output.open("w", encoding="utf-8") as f:
        f.write("# Concept Config Comparison Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total YAML files checked: {len(results)}\n")
        f.write(f"- Changed/new/removed/parse-error files: {len(changed)}\n")
        f.write(f"- Unchanged files: {len(results) - len(changed)}\n\n")

        counts: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1

        f.write("## Status counts\n\n")
        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

        f.write("\n## Details\n\n")

        for result in changed:
            f.write(f"### `{result.path}`\n\n")
            f.write(f"Status: `{result.status}`\n\n")

            for detail in result.details:
                f.write(f"- `{detail}`\n")

            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old and generated OpenICU concept YAML configs."
    )
    parser.add_argument(
        "--old",
        required=True,
        type=Path,
        help="Path to old concept config directory.",
    )
    parser.add_argument(
        "--new",
        required=True,
        type=Path,
        help="Path to newly generated concept config directory.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("concept_diff_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("concept_diff_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Print unchanged files too.",
    )

    args = parser.parse_args()

    old_dir = args.old.resolve()
    new_dir = args.new.resolve()

    if not old_dir.exists():
        raise FileNotFoundError(f"Old directory does not exist: {old_dir}")

    if not new_dir.exists():
        raise FileNotFoundError(f"New directory does not exist: {new_dir}")

    results = compare_dirs(old_dir, new_dir)

    write_markdown_report(results, args.report_md)
    write_json_report(results, args.report_json)

    for result in results:
        if result.status == "unchanged" and not args.show_unchanged:
            continue

        print(f"[{result.status}] {result.path}")
        for detail in result.details[:10]:
            print(f"  - {detail}")

        if len(result.details) > 10:
            print(f"  ... {len(result.details) - 10} more differences")

    changed_count = sum(result.status != "unchanged" for result in results)

    print()
    print(f"Compared files: {len(results)}")
    print(f"Changed/new/removed/parse-error files: {changed_count}")
    print(f"Markdown report: {args.report_md}")
    print(f"JSON report: {args.report_json}")


if __name__ == "__main__":
    main()