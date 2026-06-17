#!/usr/bin/env python3
"""Parse Krylov convergence histories and plot residual vs iteration.

Reads one or more solver logs produced with ``print_level: 2`` on the Krylov
solver (e.g. the Darcy driver run with different preconditioner YAML files) and
overlays their convergence histories on a single semilog plot.

Each log is expected to contain a hypre Krylov history table of the form::

    Iters      resid.norm     conv.rate   rel.res.norm
    -----    ------------    ----------   ------------
        1    6.950522e+01      0.973268   9.732678e-01
        2    1.491124e+01      0.214534   2.087991e-01
        ...

Only the first such table in each file is used (the first linear solve).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A history data row: iter, resid.norm, conv.rate, rel.res.norm
_ROW = re.compile(
    r"^\s*(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
_HEADER = re.compile(r"Iters.*resid\.norm.*rel\.res\.norm")
_INITIAL = re.compile(r"Initial L2 norm of residual:\s*([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Krylov convergence histories from print_level:2 logs.",
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Solver log files containing a Krylov convergence table.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels, one per log (default: log file stem).",
    )
    parser.add_argument(
        "--metric",
        choices=("relative", "absolute"),
        default="relative",
        help="Plot the relative or absolute residual norm (default: relative).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("convergence.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        default="Krylov convergence",
        help="Figure title.",
    )
    return parser.parse_args()


def parse_history(path: Path, metric: str) -> Tuple[List[int], List[float]]:
    """Return (iterations, residuals) for the first history table in ``path``."""
    iters: List[int] = [0]
    initial: Optional[float] = None
    # Column index into the regex groups: 2 = resid.norm, 4 = rel.res.norm.
    col = 4 if metric == "relative" else 2
    resids: List[float] = []

    in_table = False
    saw_table = False
    for line in path.read_text().splitlines():
        if initial is None:
            m_init = _INITIAL.search(line)
            if m_init:
                initial = float(m_init.group(1))
        if not in_table:
            if _HEADER.search(line):
                in_table = True
            continue
        m = _ROW.match(line)
        if m:
            saw_table = True
            iters.append(int(m.group(1)))
            resids.append(float(m.group(col)))
        elif saw_table:
            # First non-data line after the table ends the first solve.
            break

    if not resids:
        raise SystemExit(f"{path}: no Krylov convergence table found")

    # Prepend the initial residual (iteration 0): 1.0 relative, or the parsed
    # absolute norm when available.
    iter0 = 1.0 if metric == "relative" else (initial if initial is not None else resids[0])
    return iters, [iter0] + resids


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.logs):
        raise SystemExit("--labels must provide one label per log file")

    ylabel = (
        "relative residual norm"
        if args.metric == "relative"
        else "residual norm"
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for idx, log in enumerate(args.logs):
        label = args.labels[idx] if args.labels else log.stem
        iters, resids = parse_history(log, args.metric)
        ax.semilogy(iters, resids, marker="o", markersize=4, linewidth=1.8, label=label)

    ax.set_xlabel("Krylov iteration", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(args.title, fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
