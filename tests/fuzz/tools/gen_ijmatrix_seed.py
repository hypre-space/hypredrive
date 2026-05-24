#!/usr/bin/env python3
"""Generate a multipart IJMatrix seed part."""

import argparse
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output .bin part path")
    parser.add_argument("--index-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--value-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--nrows", type=int, default=1)
    parser.add_argument("--entries", type=int, default=1)
    args = parser.parse_args()

    header = [0] * 11
    header[1] = args.index_size
    header[2] = args.value_size
    header[5] = args.nrows
    header[6] = args.entries
    header[7] = 0
    header[8] = max(args.nrows - 1, 0)

    rows = [i % max(args.nrows, 1) for i in range(args.entries)]
    cols = [(i + 1) % max(args.nrows, 1) for i in range(args.entries)]

    with open(args.output, "wb") as fp:
        fp.write(struct.pack("<11Q", *header))
        index_fmt = "<I" if args.index_size == 4 else "<Q"
        value_fmt = "<f" if args.value_size == 4 else "<d"
        for row in rows:
            fp.write(struct.pack(index_fmt, row))
        for col in cols:
            fp.write(struct.pack(index_fmt, col))
        for i in range(args.entries):
            fp.write(struct.pack(value_fmt, 1.0 + float(i)))


if __name__ == "__main__":
    main()
