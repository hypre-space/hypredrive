#!/usr/bin/env python3
"""Prepare a flavor-specific Python wheel source tree.

The Python package always imports as ``hypredrive``. Binary MPI wheels are
published as flavor-specific distributions, so this helper copies the repository
to a temporary build tree and rewrites only Python distribution metadata.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


_DIST_NAMES = {
    "mpich": "hypredrive-mpich",
    "openmpi": "hypredrive-openmpi",
}


def _ignore(dirpath: str, names: list[str]) -> set[str]:
    del dirpath
    ignored: set[str] = set()
    for name in names:
        if name in {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            "__pycache__",
            "wheelhouse",
        }:
            ignored.add(name)
        elif name.startswith(".venv"):
            ignored.add(name)
        elif name.startswith("build"):
            ignored.add(name)
        elif name.startswith("install"):
            ignored.add(name)
    return ignored


def prepare(repo_root: Path, output: Path, flavor: str) -> None:
    dist_name = _DIST_NAMES[flavor]
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(repo_root, output, ignore=_ignore)

    pyproject = output / "interfaces" / "python" / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^name = "hypredrive"$', f'name = "{dist_name}"', text, count=1)
    text = re.sub(
        r'(?m)^description = "Python interface to the hypredrive sparse-linear-system driver"$',
        f'description = "Python interface to hypredrive with bundled {flavor} HYPRE/HYPREDRV"',
        text,
        count=1,
    )
    text += (
        "\n[tool.scikit-build.cmake.define]\n"
        f'HYPREDRV_PYTHON_DIST_NAME = "{dist_name}"\n'
        f'HYPREDRV_PYTHON_MPI_FLAVOR = "{flavor}"\n'
    )
    pyproject.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", choices=sorted(_DIST_NAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    prepare(repo_root, args.output.resolve(), args.flavor)


if __name__ == "__main__":
    main()
