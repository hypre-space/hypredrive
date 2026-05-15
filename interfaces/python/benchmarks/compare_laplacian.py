#!/usr/bin/env python3
"""Compare serial hypredrive and PyAMG solves on the same Laplacian matrix."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from common import emit_records, laplacian_matrix, residual_norm, rhs_for_matrix, time_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--pyamg-method",
        action="append",
        choices=("sa", "rootnode", "pairwise", "ruge-stuben", "air"),
        help=(
            "PyAMG method to include. May be repeated. Defaults to "
            "sa, rs, rootnode, and pairwise."
        ),
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        help="write JSON records to PATH, or stdout when PATH is omitted",
    )
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.pyamg_method is None:
        args.pyamg_method = ["sa", "ruge-stuben", "rootnode", "pairwise"]
    return args


def hypredrive_options(tol: float, maxiter: int) -> dict[str, Any]:
    import hypredrive as hd

    return hd.configure(
        general={"statistics": False, "exec_policy": "host"},
        linear_system={"init_guess_mode": "zeros"},
        solver="pcg",
        preconditioner="amg",
        pcg={"max_iter": maxiter, "relative_tol": tol, "print_level": 0},
        amg={"print_level": 0},
    )


def run_hypredrive_solve_cycle(drv) -> dict[str, Any]:
    """Time the hypredrive setup/apply cycle using the private benchmark hook."""
    timings: dict[str, Any] = {}
    _, timings["initial_guess_s"] = time_call(drv._core.setup_initial_guess)
    _, timings["solver_create_s"] = time_call(drv._core.solver_create)
    try:
        _, timings["solver_setup_s"] = time_call(drv._core.solver_setup)
        _, timings["solver_apply_s"] = time_call(drv._core.solver_apply)
        timings["iterations"] = drv._core.solver_iterations()
    finally:
        _, timings["solver_destroy_s"] = time_call(drv._core.solver_destroy)
    timings["solve_s"] = timings["solver_setup_s"] + timings["solver_apply_s"]
    return timings


def run_hypredrive(matrix, rhs: np.ndarray, tol: float, maxiter: int) -> dict[str, Any]:
    import hypredrive as hd

    options = hypredrive_options(tol, maxiter)
    total_start = time.perf_counter()
    drv, create_s = time_call(lambda: hd.HypreDrive(options=options))
    try:
        _, set_matrix_s = time_call(lambda: drv.set_matrix_from_csr(matrix))
        _, set_rhs_s = time_call(lambda: drv.set_rhs(rhs))
        solve_timings = run_hypredrive_solve_cycle(drv)
        x, get_solution_s = time_call(drv.get_solution)
    finally:
        drv.close()
    total_s = time.perf_counter() - total_start
    setup_s = create_s + set_matrix_s + set_rhs_s
    return {
        "backend": "hypredrive",
        "create_s": create_s,
        "set_matrix_s": set_matrix_s,
        "set_rhs_s": set_rhs_s,
        "setup_s": setup_s,
        "get_solution_s": get_solution_s,
        "total_s": total_s,
        "residual_l2": residual_norm(matrix, x, rhs),
        "solution_l2": float(np.linalg.norm(x)),
        **solve_timings,
    }


def pyamg_solver_factory(pyamg, method: str):
    factories = {
        "sa": pyamg.smoothed_aggregation_solver,
        "rootnode": pyamg.rootnode_solver,
        "pairwise": pyamg.pairwise_solver,
        "ruge-stuben": pyamg.ruge_stuben_solver,
        "air": pyamg.air_solver,
    }
    return factories[method]


def run_pyamg_pcg(
    matrix,
    rhs: np.ndarray,
    tol: float,
    maxiter: int,
    method: str,
) -> dict[str, Any]:
    import pyamg

    total_start = time.perf_counter()
    backend_name = "rs" if method == "ruge-stuben" else method
    backend = f"pyamg-{backend_name}+pcg"
    try:
        ml, setup_s = time_call(lambda: pyamg_solver_factory(pyamg, method)(matrix))
    except Exception as exc:
        return {
            "backend": backend,
            "setup_s": None,
            "solve_s": None,
            "total_s": time.perf_counter() - total_start,
            "residual_l2": None,
            "solution_l2": None,
            "iterations": None,
            "status": "setup-failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    residuals: list[float] = []
    try:
        x, solve_s = time_call(
            lambda: ml.solve(
                rhs,
                tol=tol,
                maxiter=maxiter,
                accel="cg",
                residuals=residuals,
            )
        )
    except Exception as exc:
        return {
            "backend": backend,
            "setup_s": setup_s,
            "solve_s": None,
            "total_s": time.perf_counter() - total_start,
            "residual_l2": None,
            "solution_l2": None,
            "iterations": None,
            "status": "solve-failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    total_s = time.perf_counter() - total_start
    return {
        "backend": backend,
        "setup_s": setup_s,
        "solve_s": solve_s,
        "total_s": total_s,
        "residual_l2": residual_norm(matrix, x, rhs),
        "solution_l2": float(np.linalg.norm(x)),
        "iterations": max(0, len(residuals) - 1) if residuals else None,
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    matrix, build_s = time_call(lambda: laplacian_matrix(args.dim, args.n))
    rhs = rhs_for_matrix(matrix)

    records: list[dict[str, Any]] = []
    for repeat in range(args.repeat):
        benchmark_runs = [
            ("hypredrive", lambda: run_hypredrive(matrix, rhs, args.tol, args.maxiter)),
            *[
                (
                    f"pyamg-{method}",
                    lambda method=method: run_pyamg_pcg(
                        matrix, rhs, args.tol, args.maxiter, method
                    ),
                )
                for method in args.pyamg_method
            ],
        ]
        for _name, run in benchmark_runs:
            record = run()
            record.update(
                {
                    "repeat": repeat,
                    "dim": args.dim,
                    "n": args.n,
                    "unknowns": int(matrix.shape[0]),
                    "nnz": int(matrix.nnz),
                    "matrix_build_s": build_s,
                    "tol": args.tol,
                    "maxiter": args.maxiter,
                }
            )
            records.append(record)

    if not args.json:
        print(
            f"problem: {args.dim}D Laplacian, n={args.n}, "
            f"unknowns={matrix.shape[0]}, nnz={matrix.nnz}"
        )
        print(f"tolerance: {args.tol:.1e}")
        print()
    emit_records(records, args.json)


if __name__ == "__main__":
    main()
