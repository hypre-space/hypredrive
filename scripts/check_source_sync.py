#!/usr/bin/env python3
"""
Verify CMake and Autotools library source lists stay in sync.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CMAKE_FILE = ROOT / "CMakeLists.txt"
MAKEFILE_AM = ROOT / "Makefile.am"


def _extract_sources_from_cmake(text: str) -> set[str]:
    match = re.search(r"set\s*\(\s*SOURCE_FILES(.*?)\)\s*", text, re.DOTALL)
    if not match:
        raise RuntimeError("Could not find SOURCE_FILES block in CMakeLists.txt")
    block = match.group(1)
    return set(re.findall(r"\bsrc/[A-Za-z0-9_./-]+\.c\b", block))


def _extract_sources_from_makefile(text: str) -> set[str]:
    lines = text.splitlines()
    collecting = False
    block_parts: list[str] = []
    for line in lines:
        if not collecting:
            if line.startswith("libHYPREDRV_la_SOURCES"):
                collecting = True
                block_parts.append(line.split("=", 1)[1])
        else:
            block_parts.append(line)
            if not line.rstrip().endswith("\\"):
                break

    if not block_parts:
        raise RuntimeError("Could not find libHYPREDRV_la_SOURCES in Makefile.am")

    block = "\n".join(block_parts).replace("\\\n", " ")
    return set(re.findall(r"\bsrc/[A-Za-z0-9_./-]+\.c\b", block))


def main() -> int:
    cmake_text = CMAKE_FILE.read_text(encoding="utf-8")
    make_text = MAKEFILE_AM.read_text(encoding="utf-8")

    cmake_sources = _extract_sources_from_cmake(cmake_text)
    make_sources = _extract_sources_from_makefile(make_text)

    only_cmake = sorted(cmake_sources - make_sources)
    only_make = sorted(make_sources - cmake_sources)

    if not only_cmake and not only_make:
        print("Source lists are in sync.")
        return 0

    print("Source list mismatch detected.")
    if only_cmake:
        print("Only in CMakeLists.txt:")
        for path in only_cmake:
            print(f"  - {path}")
    if only_make:
        print("Only in Makefile.am:")
        for path in only_make:
            print(f"  - {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
