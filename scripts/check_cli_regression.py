#!/usr/bin/env python3
"""Verify the solver statistics for the serial and distributed CLI examples."""

import argparse
import math
from pathlib import Path
import subprocess


def check_output(output, tolerance):
    rows = [
        [column.strip() for column in line.strip().strip("|").split("|")]
        for line in output.splitlines()
        if line.startswith("|") and line.split("|")[1].strip().isdigit()
        and len(line.split("|")) == 9
    ]
    if len(rows) != 1 or rows[0][0] != "0":
        raise ValueError(f"expected exactly one solve, got {rows}")
    residual = float(rows[0][-2])
    if not math.isfinite(residual) or not 0 <= residual <= tolerance:
        raise ValueError(f"relative residual {residual} exceeds tolerance {tolerance}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)

    # The default PCG tolerance used by ex1 and the explicit ex2 tolerance agree.
    for example, ranks in ((1, 1), (2, 4)):
        log = args.logs / f"ex{example}-{ranks}ranks.log"
        with log.open("w") as stream:
            subprocess.run(
                ["mpiexec", "-n", str(ranks), str(args.cli.resolve()), f"examples/ex{example}.yml"],
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, check=True,
            )
        check_output(log.read_text(), 1.0e-6)
        print(f"ex{example} passed with {ranks} MPI rank(s)")


if __name__ == "__main__":
    main()
