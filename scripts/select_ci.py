#!/usr/bin/env python3
"""Select required checks for the pull request's tested merge tree."""

import json
import os
from pathlib import Path
import subprocess


HYPRE_RELEASE = (Path(__file__).resolve().parents[1] / ".github/hypre-release").read_text(
    encoding="utf-8"
).strip()


def matrix(expanded):
    baseline = [
        ("ubuntu-latest", "gcc", "Release", HYPRE_RELEASE, "OFF", "ON"),
        ("ubuntu-latest", "clang", "Debug", HYPRE_RELEASE, "ON", "OFF"),
        ("macos-latest", "clang", "Debug", HYPRE_RELEASE, "ON", "OFF"),
        ("ubuntu-latest", "gcc", "Debug", "v2.20.0", "ON", "OFF"),
    ]
    if expanded:
        baseline.extend(
            (os_name, compiler, "Debug", version, "ON", "OFF")
            for os_name, compiler in [
                ("ubuntu-latest", "gcc"), ("ubuntu-latest", "clang"),
                ("macos-latest", "clang"),
            ]
            for version in ["master", HYPRE_RELEASE, "v2.20.0"]
            if (os_name, compiler, "Debug", version, "ON", "OFF") not in baseline
        )
    keys = ("os", "compiler", "build_type", "hypre_version", "shared", "cpu_features")
    return [dict(zip(keys, entry)) for entry in baseline]


def select(paths, expanded):
    def documentation_only(path):
        return path in {"LICENSE", "NOTICE", "COPYRIGHT", "CHANGELOG", ".gitignore"} or (
            path != "README.md" and (path.startswith("docs/") or path.endswith(".md"))
        )

    def starts_with(prefixes):
        return any(path.startswith(prefixes) for path in paths)

    shared_binding_inputs = (
        starts_with(("include/", "src/", "cmake/", ".github/")) or "CMakeLists.txt" in paths
    )
    native_binding_inputs = tuple(
        f"interfaces/{name}/" for name in ("cpp", "fortran", "matlab-octave", "python", "rust")
    )
    return {
        "core": expanded or any(not documentation_only(path) for path in paths),
        "expanded": expanded,
        "docs": expanded or starts_with(("docs/", "include/", "src/", ".github/", "cmake/"))
        or bool(paths.intersection({"Doxyfile.in", "CMakeLists.txt", ".readthedocs.yaml"})),
        "bindings": expanded or shared_binding_inputs or starts_with(native_binding_inputs),
        "julia": expanded or shared_binding_inputs or starts_with(("interfaces/julia/",)),
        "abi": expanded or starts_with(("include/", "src/", "cmake/", ".github/"))
        or "CMakeLists.txt" in paths,
        "matrix": matrix(expanded),
    }


def main():
    base = os.environ["BASE_SHA"].strip()
    has_base = base not in ("", "0" * 40)
    expanded = not has_base or os.environ["EXPANDED"] == "true"
    paths = set()
    if has_base:
        diff = subprocess.check_output(["git", "diff", "--name-only", "--no-renames", "-z", base, "HEAD"])
        paths = set(os.fsdecode(diff).split("\0")[:-1])
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as outputs:
        print(f"hypre_release={HYPRE_RELEASE}", file=outputs)
        for key, value in select(paths, expanded).items():
            print(f"{key}={json.dumps(value, separators=(',', ':'))}", file=outputs)


if __name__ == "__main__":
    main()
