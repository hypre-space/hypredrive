#!/usr/bin/env python3
"""Pack IJ multipart seed parts into the fuzz harness envelope."""

import argparse
import struct


MAGIC = b"HDFZMP01"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output envelope path")
    parser.add_argument("parts", nargs="+", help="input part files, in part-id order")
    parser.add_argument(
        "--memory",
        choices=("host", "device"),
        default="host",
        help="memory location passed to the multipart reader",
    )
    args = parser.parse_args()

    if len(args.parts) > 3:
        raise SystemExit("the fuzz envelope supports at most 3 parts")

    payloads = []
    for path in args.parts:
        with open(path, "rb") as fp:
            payloads.append(fp.read())

    with open(args.output, "wb") as fp:
        fp.write(MAGIC)
        fp.write(bytes([(len(payloads) - 1) & 0xFF]))
        fp.write(bytes([1 if args.memory == "device" else 0]))
        for payload in payloads:
            fp.write(struct.pack("<I", len(payload)))
        for payload in payloads:
            fp.write(payload)


if __name__ == "__main__":
    main()
