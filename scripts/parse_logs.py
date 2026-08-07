#!/usr/bin/env python3
"""Parse and analyze HypreDrive logs.

The ``block_norms`` mode extracts diagnostics emitted at
``HYPREDRV_LOG_LEVEL=3``. A log may contain several solver objects and linear
system setups, so every ``matrix block Frobenius norms`` record is a snapshot.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


_NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
_SUMMARY = re.compile(
    rf"(?P<context>.*?)matrix block Frobenius norms:\s*"
    rf"blocks=(?P<blocks>\d+)\s+matrix_norm=(?P<norm>{_NUMBER})\s+"
    rf"ignored_nnz=(?P<ignored>\d+)",
    re.IGNORECASE,
)
_ROW = re.compile(
    r"block Frobenius row\s+"
    r"(?:(?P<label>.*?)\(id=(?P<id>-?\d+)\)|(?P<bare_id>-?\d+)):"
    r"\s*(?P<entries>.*)"
)
_ENTRY = re.compile(
    rf"(?:^|\s)(?:(?P<label>.*?)\(id=(?P<id>-?\d+)\)|(?P<bare_id>-?\d+))="
    rf"(?P<norm>{_NUMBER})\(nnz=(?P<nnz>\d+)\)",
    re.IGNORECASE,
)
_OBJECT = re.compile(r"\[obj-(?P<id>\d+)\]")
_LINEAR_SYSTEM = re.compile(r"\[ls=(?P<id>-?\d+)\]")


@dataclass
class BlockRow:
    label: str
    norms: dict[int, float]
    nonzeros: dict[int, int]
    column_labels: dict[int, str]


@dataclass
class Snapshot:
    index: int
    blocks: int
    matrix_norm: float
    ignored_nnz: int
    line_number: int
    object_id: int | None = None
    linear_system_id: int | None = None
    rows: dict[int, BlockRow] = field(default_factory=dict)

    def ordered_ids(self) -> list[int]:
        return sorted(self.rows)

    def labels(self) -> list[str]:
        return [self.rows[block_id].label for block_id in self.ordered_ids()]

    def norm_matrix(self) -> list[list[float]]:
        ids = self.ordered_ids()
        return [[self.rows[row_id].norms[col_id] for col_id in ids] for row_id in ids]

    def nnz_matrix(self) -> list[list[int]]:
        ids = self.ordered_ids()
        return [
            [self.rows[row_id].nonzeros[col_id] for col_id in ids]
            for row_id in ids
        ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and analyze HypreDrive logs.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("block_norms",),
        help="Log analysis mode",
    )
    parser.add_argument("log", help="HypreDrive log file, or '-' for standard input")
    parser.add_argument(
        "--format",
        choices=("table", "python", "csv"),
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        action="append",
        help="Print only this 1-based snapshot; may be specified more than once",
    )
    parser.add_argument(
        "--object",
        type=int,
        dest="object_id",
        help="Print only records carrying this HypreDrive object id",
    )
    parser.add_argument(
        "--nnz",
        action="store_true",
        help="Also print the matrix of nonzero counts",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Digits after the decimal in Frobenius values (default: 6)",
    )
    args = parser.parse_args(argv)
    if args.precision < 0:
        parser.error("--precision must be nonnegative")
    if args.snapshot and any(index < 1 for index in args.snapshot):
        parser.error("--snapshot values are 1-based and must be positive")
    return args


def _finish_snapshot(snapshot: Snapshot | None, source: str) -> None:
    if snapshot is None:
        return

    ids = snapshot.ordered_ids()
    if len(ids) != snapshot.blocks:
        raise ValueError(
            f"{source}:{snapshot.line_number}: snapshot {snapshot.index} declares "
            f"{snapshot.blocks} blocks but contains {len(ids)} Frobenius rows"
        )

    expected = set(ids)
    for row_id in ids:
        row = snapshot.rows[row_id]
        found = set(row.norms)
        if found != expected:
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            raise ValueError(
                f"{source}:{snapshot.line_number}: snapshot {snapshot.index}, "
                f"row id {row_id} has inconsistent columns "
                f"(missing={missing}, extra={extra})"
            )
        for col_id in ids:
            column_label = row.column_labels[col_id]
            expected_label = snapshot.rows[col_id].label
            if column_label != expected_label:
                raise ValueError(
                    f"{source}:{snapshot.line_number}: snapshot {snapshot.index}, "
                    f"column id {col_id} is labeled both {column_label!r} and "
                    f"{expected_label!r}"
                )


def parse_snapshots(stream: TextIO, source: str) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    current: Snapshot | None = None

    for line_number, line in enumerate(stream, start=1):
        summary = _SUMMARY.search(line)
        if summary:
            _finish_snapshot(current, source)
            context = summary.group("context")
            object_match = _OBJECT.search(context)
            ls_match = _LINEAR_SYSTEM.search(context)
            current = Snapshot(
                index=len(snapshots) + 1,
                blocks=int(summary.group("blocks")),
                matrix_norm=float(summary.group("norm")),
                ignored_nnz=int(summary.group("ignored")),
                line_number=line_number,
                object_id=int(object_match.group("id")) if object_match else None,
                linear_system_id=int(ls_match.group("id")) if ls_match else None,
            )
            snapshots.append(current)
            continue

        row_match = _ROW.search(line)
        if row_match is None or current is None:
            continue

        row_id_text = row_match.group("id") or row_match.group("bare_id")
        row_id = int(row_id_text)
        row_label = (
            row_match.group("label").strip()
            if row_match.group("id") is not None
            else row_id_text
        )
        entries = list(_ENTRY.finditer(row_match.group("entries")))
        if len(entries) != current.blocks:
            raise ValueError(
                f"{source}:{line_number}: expected {current.blocks} entries in "
                f"Frobenius row id {row_id}, found {len(entries)}"
            )
        if row_id in current.rows:
            raise ValueError(
                f"{source}:{line_number}: duplicate Frobenius row id {row_id} "
                f"in snapshot {current.index}"
            )

        entry_ids = [int(item.group("id") or item.group("bare_id")) for item in entries]
        entry_labels = [
            item.group("label").strip()
            if item.group("id") is not None
            else item.group("bare_id")
            for item in entries
        ]
        current.rows[row_id] = BlockRow(
            label=row_label,
            norms={
                block_id: float(item.group("norm"))
                for block_id, item in zip(entry_ids, entries)
            },
            nonzeros={
                block_id: int(item.group("nnz"))
                for block_id, item in zip(entry_ids, entries)
            },
            column_labels=dict(zip(entry_ids, entry_labels)),
        )

    _finish_snapshot(current, source)
    if not snapshots:
        raise ValueError(f"{source}: no block Frobenius norm snapshots found")
    return snapshots


def _metadata(snapshot: Snapshot) -> str:
    fields = [f"snapshot={snapshot.index}"]
    if snapshot.object_id is not None:
        fields.append(f"object={snapshot.object_id}")
    if snapshot.linear_system_id is not None:
        fields.append(f"linear_system={snapshot.linear_system_id}")
    fields.extend(
        (
            f"blocks={snapshot.blocks}",
            f"matrix_norm={snapshot.matrix_norm:.6e}",
            f"ignored_nnz={snapshot.ignored_nnz}",
        )
    )
    return " ".join(fields)


def _print_table_matrix(
    title: str,
    ids: list[int],
    matrix: list[list[float]] | list[list[int]],
    precision: int,
    integer: bool = False,
) -> None:
    values = (
        [[str(value) for value in row] for row in matrix]
        if integer
        else [[f"{value:.{precision}e}" for value in row] for row in matrix]
    )
    width = max(3, *(len(value) for row in values for value in row))
    print(title)
    print("id".rjust(3), *(str(block_id).rjust(width) for block_id in ids))
    for block_id, row in zip(ids, values):
        print(str(block_id).rjust(3), *(value.rjust(width) for value in row))


def print_table(snapshot: Snapshot, precision: int, include_nnz: bool) -> None:
    print(_metadata(snapshot))
    ids = snapshot.ordered_ids()
    labels = snapshot.labels()
    print("labels:")
    for block_id, label in zip(ids, labels):
        print(f"  {block_id}: {label}")
    _print_table_matrix(
        "frobenius (rows x columns):", ids, snapshot.norm_matrix(), precision
    )
    if include_nnz:
        _print_table_matrix(
            "nnz (rows x columns):",
            ids,
            snapshot.nnz_matrix(),
            precision,
            integer=True,
        )


def _python_matrix(
    matrix: list[list[float]] | list[list[int]], precision: int, integer: bool = False
) -> list[str]:
    if integer:
        return ["[" + ", ".join(str(value) for value in row) + "]" for row in matrix]
    return [
        "[" + ", ".join(f"{value:.{precision}e}" for value in row) + "]"
        for row in matrix
    ]


def print_python(snapshots: list[Snapshot], precision: int, include_nnz: bool) -> None:
    print("block_norm_snapshots = [")
    for snapshot in snapshots:
        print("    {")
        print(f"        'snapshot': {snapshot.index},")
        print(f"        'object': {snapshot.object_id!r},")
        print(f"        'linear_system': {snapshot.linear_system_id!r},")
        print(f"        'matrix_norm': {snapshot.matrix_norm:.{precision}e},")
        print(f"        'ignored_nnz': {snapshot.ignored_nnz},")
        print(f"        'block_ids': {snapshot.ordered_ids()!r},")
        print(f"        'labels': {snapshot.labels()!r},")
        print("        'frobenius': [")
        for row in _python_matrix(snapshot.norm_matrix(), precision):
            print(f"            {row},")
        print("        ],")
        if include_nnz:
            print("        'nnz': [")
            for row in _python_matrix(snapshot.nnz_matrix(), precision, integer=True):
                print(f"            {row},")
            print("        ],")
        print("    },")
    print("]")


def print_csv(snapshot: Snapshot, precision: int, include_nnz: bool) -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    print(f"# {_metadata(snapshot)}")
    labels = snapshot.labels()
    writer.writerow(["row/column", *labels])
    for label, row in zip(labels, snapshot.norm_matrix()):
        writer.writerow([label, *(f"{value:.{precision}e}" for value in row)])
    if include_nnz:
        print("# nnz")
        writer.writerow(["row/column", *labels])
        for label, row in zip(labels, snapshot.nnz_matrix()):
            writer.writerow([label, *row])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = "<stdin>" if args.log == "-" else args.log
    try:
        if args.log == "-":
            snapshots = parse_snapshots(sys.stdin, source)
        else:
            with Path(args.log).open(encoding="utf-8", errors="replace") as stream:
                snapshots = parse_snapshots(stream, source)
    except (OSError, ValueError) as error:
        raise SystemExit(error) from error

    if args.snapshot:
        selected = set(args.snapshot)
        snapshots = [snapshot for snapshot in snapshots if snapshot.index in selected]
    if args.object_id is not None:
        snapshots = [
            snapshot for snapshot in snapshots if snapshot.object_id == args.object_id
        ]
    if not snapshots:
        raise SystemExit("no block Frobenius norm snapshots match the selection")

    if args.format == "python":
        print_python(snapshots, args.precision, args.nnz)
        return

    for position, snapshot in enumerate(snapshots):
        if position:
            print()
        if args.format == "csv":
            print_csv(snapshot, args.precision, args.nnz)
        else:
            print_table(snapshot, args.precision, args.nnz)


if __name__ == "__main__":
    main()
