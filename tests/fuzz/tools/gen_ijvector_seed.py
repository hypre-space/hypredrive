#!/usr/bin/env python3
"""Generate a multipart IJVector seed part."""

import argparse
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output .bin part path")
    parser.add_argument("--value-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--nrows", type=int, default=3)
    args = parser.parse_args()

    header = [0] * 8
    header[1] = args.value_size
    header[5] = args.nrows

    with open(args.output, "wb") as fp:
        fp.write(struct.pack("<8Q", *header))
        value_fmt = "<f" if args.value_size == 4 else "<d"
        for i in range(args.nrows):
            fp.write(struct.pack(value_fmt, 1.0 + float(i)))


if __name__ == "__main__":
    main()
