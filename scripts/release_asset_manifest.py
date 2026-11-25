#!/usr/bin/env python3
"""Record and check the source tree and SHA-256 digests of staged release assets."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def source_tree(kind, ref="HEAD"):
    args = ["git", "ls-tree", "-r", "-z", ref]
    if kind == "julia":
        # The binding commit records hashes of the tarballs we just built.
        args += ["--", ".", ":(exclude)interfaces/julia/Artifacts.toml"]
    return hashlib.sha256(subprocess.check_output(args)).hexdigest()


def digests(directory, kind):
    suffix = {"python": ".whl", "julia": ".tar.gz"}[kind]
    expected_count = {"python": 45, "julia": 5}[kind]
    paths = sorted(path for path in directory.iterdir() if path.name.endswith(suffix))
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} {kind} assets, got {len(paths)}")

    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("write", "verify"))
    parser.add_argument("kind", choices=("python", "julia"))
    parser.add_argument("tag")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-ref", default="HEAD")
    args = parser.parse_args()

    manifest = args.directory / f"{args.kind}-asset-manifest.json"
    expected = {
        "tag": args.tag,
        "source_tree_sha256": source_tree(args.kind, args.source_ref),
        "assets": digests(args.directory, args.kind),
    }
    if args.operation == "write":
        manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    elif json.loads(manifest.read_text(encoding="utf-8")) != expected:
        parser.error(f"tag, source tree, or asset digests differ from {manifest}")
    print(f"{args.kind} manifest verified for {args.tag}")


if __name__ == "__main__":
    main()
