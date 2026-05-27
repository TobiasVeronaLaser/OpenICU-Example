#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class ConceptOutputDiff:
    path: str
    status: str
    old_rows: int | None = None
    new_rows: int | None = None
    old_schema: dict[str, str] | None = None
    new_schema: dict[str, str] | None = None
    schema_diff: list[str] = field(default_factory=list)
    missing_in_new: int | None = None
    missing_in_old: int | None = None
    changed_values: int | None = None
    details: list[str] = field(default_factory=list)


def collect_parquet_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    for path in root.rglob("*.parquet"):
        rel = path.relative_to(root).as_posix()
        files[rel] = path

    return files


def schema_to_str_dict(schema: dict[str, pl.DataType]) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in schema.items()}


def compare_schema(old_schema: dict[str, str], new_schema: dict[str, str]) -> list[str]:
    diffs: list[str] = []

    old_cols = set(old_schema)
    new_cols = set(new_schema)

    for col in sorted(old_cols - new_cols):
        diffs.append(f"removed column: {col} ({old_schema[col]})")

    for col in sorted(new_cols - old_cols):
        diffs.append(f"added column: {col} ({new_schema[col]})")

    for col in sorted(old_cols & new_cols):
        if old_schema[col] != new_schema[col]:
            diffs.append(f"type changed: {col}: {old_schema[col]} -> {new_schema[col]}")

    return diffs


def normalize_df(df: pl.DataFrame, sort_columns: list[str] | None) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col(col).cast(pl.Utf8).alias(col)
            for col in df.columns
            if df.schema[col] in {pl.Categorical, pl.Enum}
        ]
    )

    if sort_columns:
        existing = [col for col in sort_columns if col in df.columns]
        if existing:
            df = df.sort(existing)

    return df


def compare_without_keys(
    old_df: pl.DataFrame,
    new_df: pl.DataFrame,
    rel_path: str,
) -> ConceptOutputDiff:
    old_schema = schema_to_str_dict(old_df.schema)
    new_schema = schema_to_str_dict(new_df.schema)

    schema_diff = compare_schema(old_schema, new_schema)

    result = ConceptOutputDiff(
        path=rel_path,
        status="unchanged",
        old_rows=old_df.height,
        new_rows=new_df.height,
        old_schema=old_schema,
        new_schema=new_schema,
        schema_diff=schema_diff,
    )

    if schema_diff:
        result.status = "schema_changed"
        result.details.extend(schema_diff)
        return result

    old_sorted = old_df.sort(old_df.columns)
    new_sorted = new_df.sort(new_df.columns)

    if old_sorted.equals(new_sorted):
        return result

    result.status = "changed"

    old_unique = old_sorted.unique()
    new_unique = new_sorted.unique()

    missing_in_new = old_unique.join(new_unique, on=old_unique.columns, how="anti")
    missing_in_old = new_unique.join(old_unique, on=new_unique.columns, how="anti")

    result.missing_in_new = missing_in_new.height
    result.missing_in_old = missing_in_old.height

    if old_df.height != new_df.height:
        result.details.append(f"row count changed: {old_df.height} -> {new_df.height}")

    result.details.append(f"rows only in old: {missing_in_new.height}")
    result.details.append(f"rows only in new: {missing_in_old.height}")

    return result


def compare_with_keys(
    old_df: pl.DataFrame,
    new_df: pl.DataFrame,
    rel_path: str,
    key_columns: list[str],
    tolerance: float,
) -> ConceptOutputDiff:
    old_schema = schema_to_str_dict(old_df.schema)
    new_schema = schema_to_str_dict(new_df.schema)

    schema_diff = compare_schema(old_schema, new_schema)

    result = ConceptOutputDiff(
        path=rel_path,
        status="unchanged",
        old_rows=old_df.height,
        new_rows=new_df.height,
        old_schema=old_schema,
        new_schema=new_schema,
        schema_diff=schema_diff,
    )

    missing_keys = [col for col in key_columns if col not in old_df.columns or col not in new_df.columns]
    if missing_keys:
        result.status = "missing_key_columns"
        result.details.append(f"missing key columns: {missing_keys}")
        return result

    if schema_diff:
        result.status = "schema_changed"
        result.details.extend(schema_diff)
        return result

    compare_columns = [col for col in old_df.columns if col not in key_columns]

    old_keys = old_df.select(key_columns).unique()
    new_keys = new_df.select(key_columns).unique()

    missing_in_new = old_keys.join(new_keys, on=key_columns, how="anti")
    missing_in_old = new_keys.join(old_keys, on=key_columns, how="anti")

    result.missing_in_new = missing_in_new.height
    result.missing_in_old = missing_in_old.height

    joined = old_df.join(
        new_df,
        on=key_columns,
        how="inner",
        suffix="__new",
    )

    changed_count = 0

    for col in compare_columns:
        new_col = f"{col}__new"

        if new_col not in joined.columns:
            continue

        dtype = old_df.schema[col]

        if dtype.is_float():
            changed_expr = (
                (pl.col(col) - pl.col(new_col)).abs() > tolerance
            ) & ~(pl.col(col).is_null() & pl.col(new_col).is_null())
        else:
            changed_expr = (
                pl.col(col) != pl.col(new_col)
            ) & ~(pl.col(col).is_null() & pl.col(new_col).is_null())

        n_changed = joined.select(changed_expr.sum().alias("n")).item()

        if n_changed:
            changed_count += int(n_changed)
            result.details.append(f"{col}: {n_changed} changed values")

    result.changed_values = changed_count

    if (
        old_df.height != new_df.height
        or missing_in_new.height
        or missing_in_old.height
        or changed_count
    ):
        result.status = "changed"

    if old_df.height != new_df.height:
        result.details.append(f"row count changed: {old_df.height} -> {new_df.height}")

    if missing_in_new.height:
        result.details.append(f"keys only in old: {missing_in_new.height}")

    if missing_in_old.height:
        result.details.append(f"keys only in new: {missing_in_old.height}")

    return result


def compare_parquet_file(
    old_path: Path,
    new_path: Path,
    rel_path: str,
    key_columns: list[str] | None,
    tolerance: float,
) -> ConceptOutputDiff:
    old_df = pl.read_parquet(old_path)
    new_df = pl.read_parquet(new_path)

    if key_columns:
        return compare_with_keys(
            old_df=old_df,
            new_df=new_df,
            rel_path=rel_path,
            key_columns=key_columns,
            tolerance=tolerance,
        )

    return compare_without_keys(
        old_df=old_df,
        new_df=new_df,
        rel_path=rel_path,
    )


def compare_output_dirs(
    old_root: Path,
    new_root: Path,
    key_columns: list[str] | None,
    tolerance: float,
) -> list[ConceptOutputDiff]:
    old_files = collect_parquet_files(old_root)
    new_files = collect_parquet_files(new_root)

    results: list[ConceptOutputDiff] = []

    for rel_path in sorted(set(old_files) | set(new_files)):
        old_path = old_files.get(rel_path)
        new_path = new_files.get(rel_path)

        if old_path is None:
            results.append(
                ConceptOutputDiff(
                    path=rel_path,
                    status="added_file",
                    details=[f"only exists in new: {new_path}"],
                )
            )
            continue

        if new_path is None:
            results.append(
                ConceptOutputDiff(
                    path=rel_path,
                    status="removed_file",
                    details=[f"only exists in old: {old_path}"],
                )
            )
            continue

        try:
            result = compare_parquet_file(
                old_path=old_path,
                new_path=new_path,
                rel_path=rel_path,
                key_columns=key_columns,
                tolerance=tolerance,
            )
        except Exception as exc:
            result = ConceptOutputDiff(
                path=rel_path,
                status="error",
                details=[repr(exc)],
            )

        results.append(result)

    return results


def write_json_report(results: list[ConceptOutputDiff], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)


def write_markdown_report(results: list[ConceptOutputDiff], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    changed = [r for r in results if r.status != "unchanged"]

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    with output.open("w", encoding="utf-8") as f:
        f.write("# Concept Output Comparison Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Compared Parquet files: {len(results)}\n")
        f.write(f"- Changed/new/removed/error files: {len(changed)}\n")
        f.write(f"- Unchanged files: {len(results) - len(changed)}\n\n")

        f.write("## Status counts\n\n")
        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

        f.write("\n## Details\n\n")

        for result in changed:
            f.write(f"### `{result.path}`\n\n")
            f.write(f"- Status: `{result.status}`\n")

            if result.old_rows is not None or result.new_rows is not None:
                f.write(f"- Rows: `{result.old_rows}` -> `{result.new_rows}`\n")

            if result.missing_in_new is not None:
                f.write(f"- Keys/rows only in old: `{result.missing_in_new}`\n")

            if result.missing_in_old is not None:
                f.write(f"- Keys/rows only in new: `{result.missing_in_old}`\n")

            if result.changed_values is not None:
                f.write(f"- Changed values: `{result.changed_values}`\n")

            if result.schema_diff:
                f.write("\nSchema differences:\n")
                for diff in result.schema_diff:
                    f.write(f"- `{diff}`\n")

            if result.details:
                f.write("\nDetails:\n")
                for detail in result.details[:50]:
                    f.write(f"- `{detail}`\n")

                if len(result.details) > 50:
                    f.write(f"- `... {len(result.details) - 50} more details`\n")

            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old and new OpenICU concept output Parquet files."
    )
    parser.add_argument(
        "--old",
        required=True,
        type=Path,
        help="Old concept output root directory.",
    )
    parser.add_argument(
        "--new",
        required=True,
        type=Path,
        help="New concept output root directory.",
    )
    parser.add_argument(
        "--key",
        nargs="*",
        default=None,
        help=(
            "Optional key columns for row alignment. "
            "Example: --key subject_id time code"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for float comparisons.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("concept_output_diff_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("concept_output_diff_report.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Print unchanged files too.",
    )

    args = parser.parse_args()

    old_root = args.old.resolve()
    new_root = args.new.resolve()

    if not old_root.exists():
        raise FileNotFoundError(f"Old output directory does not exist: {old_root}")

    if not new_root.exists():
        raise FileNotFoundError(f"New output directory does not exist: {new_root}")

    results = compare_output_dirs(
        old_root=old_root,
        new_root=new_root,
        key_columns=args.key,
        tolerance=args.tolerance,
    )

    write_markdown_report(results, args.report_md)
    write_json_report(results, args.report_json)

    for result in results:
        if result.status == "unchanged" and not args.show_unchanged:
            continue

        print(f"[{result.status}] {result.path}")

        if result.old_rows is not None or result.new_rows is not None:
            print(f"  rows: {result.old_rows} -> {result.new_rows}")

        for detail in result.details[:10]:
            print(f"  - {detail}")

        if len(result.details) > 10:
            print(f"  ... {len(result.details) - 10} more details")

    changed_count = sum(result.status != "unchanged" for result in results)

    print()
    print(f"Compared parquet files: {len(results)}")
    print(f"Changed/new/removed/error files: {changed_count}")
    print(f"Markdown report: {args.report_md}")
    print(f"JSON report: {args.report_json}")


if __name__ == "__main__":
    main()