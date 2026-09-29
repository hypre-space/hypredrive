#!/usr/bin/env python3
"""Parse Krylov convergence histories and plot residual vs iteration.

Reads one or more solver logs produced with ``print_level`` > 1 on the Krylov
solver and overlays their convergence histories on a single semilog plot.

Hypre prints several equivalent tables. GMRES, BiCGSTAB, and FlexGMRES use::

    Iters      resid.norm     conv.rate   rel.res.norm
    -----    ------------    ----------   ------------
        1    6.950522e+01      0.973268   9.732678e-01

PCG, CGNR, and the structured solvers use::

    Iters       ||r||_2     conv.rate  ||r||_2/||b||_2
    -----    ------------   ---------  ------------
        1    4.882464e-05    0.238848    2.388480e-01

The same parser accepts the three-column forms (no relative column), the
preconditioned PCG norm ``||r||_C``, GMRES error tables (``error.norm``), and
the variable-width tagged residual/error tables. Only the first table in each
file is used (the first linear solve). Iteration 0 is included when the log
prints it, and otherwise reconstructed from the initial residual or from the
first convergence rate.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Header of a Krylov (or structured-solver) convergence table.
_HEADER = re.compile(
    r"^\s*Iters\b.*(?:"
    r"conv\.rate|resid\.norm|rel\.res\.norm|error\.norm|rel\.err\.norm|"
    r"\|\|r\|\|_[2C]|\|[re]\d*\|_2"
    r")"
)
_RELATIVE = re.compile(r"rel\.res\.norm|rel\.err\.norm|/")
_ERROR = re.compile(r"error\.norm|rel\.err\.norm|\|e\d*\|_2")
_INITIAL = re.compile(r"Initial L2 norm of residual:\s*([-+0-9.eE]+)")
_BNORM = re.compile(r"L2 norm of b:\s*([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Krylov convergence histories from print_level>1 logs.",
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


def _parse_data_row(line: str) -> Optional[Tuple[int, List[float]]]:
    """Return (iteration, numbers) when ``line`` is a history row."""
    parts = line.split()
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    numbers: List[float] = []
    for part in parts[1:]:
        try:
            numbers.append(float(part))
        except ValueError:
            return None
    return int(parts[0]), numbers


def _previous(value: float, rate: Optional[float]) -> Optional[float]:
    """Recover the value at the previous iteration from a convergence rate."""
    if rate is None or rate == 0.0:
        return None
    return value / rate


def _prepend_initial(
    iters: Sequence[int],
    values: Sequence[float],
    initial: Optional[float],
) -> Tuple[List[int], List[float]]:
    """Add iteration 0 when the table starts at iteration 1 and it is known."""
    out_iters = list(iters)
    out_values = list(values)
    if initial is None or not out_iters or out_iters[0] != 1:
        return out_iters, out_values
    return [0] + out_iters, [initial] + out_values


def parse_history(path: Path, metric: str) -> Tuple[List[int], List[float], str]:
    """Return (iterations, residuals, quantity) for the first table in ``path``.

    ``quantity`` is ``"residual"`` or ``"error"``, matching the printed column.
    """
    iters: List[int] = []
    absolutes: List[float] = []
    relatives: List[Optional[float]] = []
    rates: List[Optional[float]] = []
    initial: Optional[float] = None
    b_norm: Optional[float] = None
    quantity = "residual"
    has_rate = False
    has_relative = False
    in_table = False
    saw_row = False

    for line in path.read_text().splitlines():
        if initial is None:
            match = _INITIAL.search(line)
            if match:
                initial = float(match.group(1))
        if b_norm is None:
            match = _BNORM.search(line)
            if match:
                b_norm = float(match.group(1))
        if not in_table:
            if _HEADER.search(line):
                in_table = True
                has_rate = "conv.rate" in line
                has_relative = _RELATIVE.search(line) is not None
                if _ERROR.search(line):
                    quantity = "error"
            continue

        parsed = _parse_data_row(line)
        needed = 3 if has_rate and has_relative else (2 if has_rate else 1)
        if parsed is None or len(parsed[1]) < needed:
            if saw_row:
                break
            continue

        saw_row = True
        iteration, numbers = parsed
        iters.append(iteration)
        if has_rate:
            absolutes.append(numbers[0])
            rates.append(numbers[1])
            relatives.append(numbers[2] if has_relative else None)
        else:
            # Tagged tables print the global norm first, then one value per tag.
            absolutes.append(numbers[0])
            rates.append(None)
            relatives.append(numbers[0] if has_relative else None)

    if not iters:
        raise SystemExit(f"{path}: no Krylov convergence table found")

    if metric == "relative" and any(value is not None for value in relatives):
        values = [float(value) for value in relatives]
        iter0 = _previous(values[0], rates[0])
        if iter0 is None and initial is not None and b_norm not in (None, 0.0):
            iter0 = initial / b_norm
        if iter0 is None:
            iter0 = 1.0
    elif metric == "relative":
        reference = initial
        if reference is None:
            reference = _previous(absolutes[0], rates[0])
        if reference is None or reference == 0.0:
            values = list(absolutes)
            iter0 = None
        else:
            values = [value / reference for value in absolutes]
            iter0 = 1.0
    else:
        values = list(absolutes)
        iter0 = initial if initial is not None else _previous(absolutes[0], rates[0])

    iters, values = _prepend_initial(iters, values, iter0)
    return iters, values, quantity


def _ylabel(metric: str, quantities: Sequence[str]) -> str:
    unique = set(quantities)
    if unique == {"error"}:
        name = "error norm"
    elif unique == {"residual"}:
        name = "residual norm"
    else:
        name = "norm"
    if metric == "relative":
        return f"relative {name}"
    return name


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.logs):
        raise SystemExit("--labels must provide one label per log file")

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    quantities: List[str] = []
    for idx, log in enumerate(args.logs):
        label = args.labels[idx] if args.labels else log.stem
        iters, resids, quantity = parse_history(log, args.metric)
        quantities.append(quantity)
        ax.semilogy(iters, resids, marker="o", markersize=4, linewidth=1.8, label=label)

    ax.set_xlabel("Krylov iteration", fontsize=14)
    ax.set_ylabel(_ylabel(args.metric, quantities), fontsize=14)
    ax.set_title(args.title, fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
