#!/usr/bin/env python3
"""Pack hypredrive datasets into deterministic, per-dataset Zenodo archives.

The current Zenodo dataset record is 22116856.  This packer defaults to the
v0.3.0 release and preserves the existing archive names, for example
``compflow6k.tar.gz``.  The release version is recorded in the generated
manifest and checksum filenames.

Examples::

    # Pack every dataset found below data/ into /tmp.
    python3 scripts/pack_datasets.py

    # Pack selected datasets into a new directory.
    python3 scripts/pack_datasets.py \
        --dataset spres1k --dataset tmpporo1k \
        --output-dir /tmp/hypredrive-datasets-v0.3.0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path


ZENODO_RECORD = "22116856"
CURRENT_ZENODO_VERSION = "0.1.0"
DEFAULT_VERSION = "0.3.0"
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "data"
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one deterministic tarball per hypredrive dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="directory containing dataset directories (default: repository data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for tarballs and checksums (default: /tmp/hypredrive-datasets-vVERSION)",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"release version recorded in the manifest (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="pack only this dataset; repeat for multiple datasets (default: discover all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite archives/checksum files that already exist in --output-dir",
    )
    return parser.parse_args()


def dataset_contains_matrix(dataset_dir: Path) -> bool:
    np1_dir = dataset_dir / "np1"
    if not np1_dir.is_dir():
        return False
    return any(
        path.is_file() and path.name.startswith("IJ.out.A.")
        for path in np1_dir.rglob("*")
    )


def discover_datasets(source_dir: Path) -> list[Path]:
    datasets = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and (path / "README.md").is_file()
        ),
        key=lambda path: path.name,
    )
    for dataset_dir in datasets:
        if not dataset_contains_matrix(dataset_dir):
            print(
                f"warning: {dataset_dir.name} has no np1/IJ.out.A.* matrix; "
                "including it in the archive anyway"
            )
    return datasets


def select_datasets(source_dir: Path, requested: list[str] | None) -> list[Path]:
    available = {path.name: path for path in discover_datasets(source_dir)}
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"requested dataset(s) not found: {names}")
        return [available[name] for name in sorted(set(requested))]
    if not available:
        raise ValueError(f"no dataset directories found below {source_dir}")
    return [available[name] for name in sorted(available)]


def archive_paths(dataset_dir: Path) -> list[Path]:
    paths = [dataset_dir, *dataset_dir.rglob("*")]
    paths.sort(key=lambda path: path.relative_to(dataset_dir.parent).as_posix())
    return paths


def normalized_tarinfo(archive: tarfile.TarFile, path: Path, arcname: str) -> tarfile.TarInfo:
    if not path.is_symlink() and not path.is_file() and not path.is_dir():
        raise ValueError(f"unsupported filesystem entry in dataset: {path}")

    info = archive.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    if info.isdir():
        info.mode = 0o755
    elif info.issym():
        info.mode = 0o777
    else:
        info.mode = 0o644
    return info


def write_archive(dataset_dir: Path, output_path: Path) -> None:
    """Write a gzip-compressed tar with stable metadata and member ordering."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for path in archive_paths(dataset_dir):
                    arcname = path.relative_to(dataset_dir.parent).as_posix()
                    info = normalized_tarinfo(archive, path, arcname)
                    if info.isfile():
                        with path.open("rb") as input_file:
                            archive.addfile(info, input_file)
                    else:
                        archive.addfile(info)


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    args = parse_args()
    if not VERSION_RE.fullmatch(args.version):
        raise ValueError(f"--version must be MAJOR.MINOR.PATCH, got {args.version!r}")

    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path("/tmp") / f"hypredrive-datasets-v{args.version}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = select_datasets(source_dir, args.datasets)
    planned_archives = [output_dir / f"{dataset.name}.tar.gz" for dataset in datasets]
    planned_files = [
        *planned_archives,
        output_dir / f"checksums-md5-v{args.version}.txt",
        output_dir / f"checksums-sha256-v{args.version}.txt",
        output_dir / f"manifest-v{args.version}.json",
    ]
    if not args.force:
        existing = [path for path in planned_files if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"output already exists ({names}); choose another --output-dir or use --force"
            )

    records = []
    for dataset_dir, archive_path in zip(datasets, planned_archives):
        write_archive(dataset_dir, archive_path)
        records.append(
            {
                "dataset": dataset_dir.name,
                "archive": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "md5": digest(archive_path, "md5"),
                "sha256": digest(archive_path, "sha256"),
            }
        )
        print(f"{dataset_dir.name}: {archive_path} ({records[-1]['bytes']} bytes)")

    md5_path = output_dir / f"checksums-md5-v{args.version}.txt"
    md5_path.write_text(
        "".join(f"{record['md5']}  {record['archive']}\n" for record in records)
    )
    sha256_path = output_dir / f"checksums-sha256-v{args.version}.txt"
    sha256_path.write_text(
        "".join(f"{record['sha256']}  {record['archive']}\n" for record in records)
    )
    manifest_path = output_dir / f"manifest-v{args.version}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project": "hypredrive",
                "zenodo_record": ZENODO_RECORD,
                "previous_zenodo_version": CURRENT_ZENODO_VERSION,
                "release": args.version,
                "source_directory": source_dir.name,
                "archive_layout": "<dataset>.tar.gz, containing the dataset directory at archive root",
                "datasets": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(records)} archives to {output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"pack_datasets.py: error: {error}")
