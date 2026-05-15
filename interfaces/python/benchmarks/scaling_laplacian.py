#!/usr/bin/env python3
"""MPI hypredrive scaling benchmark for a distributed 3D Laplacian."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from mpi4py import MPI

import numpy as np

from common import time_call


def parse_args(comm: MPI.Comm) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs=3, type=int, default=(16, 16, 16),
                        metavar=("NX", "NY", "NZ"))
    parser.add_argument("--P", nargs=3, type=int, default=(1, 1, 1),
                        metavar=("PX", "PY", "PZ"))
    parser.add_argument("--stencil", type=int, choices=(7, 19, 27), default=7)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if any(v <= 0 for v in args.n + args.P):
        parser.error("--n and --P entries must be positive")
    if int(np.prod(args.P)) != comm.Get_size():
        parser.error(f"--P product {int(np.prod(args.P))} must equal MPI size {comm.Get_size()}")
    if any(p > n for p, n in zip(args.P, args.n)):
        parser.error("each processor dimension must be <= the matching grid dimension")
    return args


def split_points(n: int, p: int) -> np.ndarray:
    base, extra = divmod(n, p)
    return np.array([base * i + min(i, extra) for i in range(p + 1)], dtype=np.int64)


def block_size(starts: tuple[np.ndarray, np.ndarray, np.ndarray],
               coords: tuple[int, int, int]) -> int:
    return int(np.prod([starts[d][coords[d] + 1] - starts[d][coords[d]] for d in range(3)]))


def block_offset(starts: tuple[np.ndarray, np.ndarray, np.ndarray],
                 proc_shape: tuple[int, int, int],
                 coords: tuple[int, int, int]) -> int:
    offset = 0
    for bz in range(proc_shape[2]):
        for by in range(proc_shape[1]):
            for bx in range(proc_shape[0]):
                block = (bx, by, bz)
                if block == coords:
                    return offset
                offset += block_size(starts, block)
    raise ValueError(f"invalid processor coordinates: {coords}")


def block_for_coord(coord: int, starts: np.ndarray) -> int:
    return int(np.searchsorted(starts, coord, side="right") - 1)


def gid(x: int, y: int, z: int, starts: tuple[np.ndarray, np.ndarray, np.ndarray],
        proc_shape: tuple[int, int, int]) -> int:
    block = tuple(block_for_coord(coord, start) for coord, start in zip((x, y, z), starts))
    x0, y0, z0 = (int(starts[d][block[d]]) for d in range(3))
    nx, ny, _ = (int(starts[d][block[d] + 1] - starts[d][block[d]]) for d in range(3))
    local_id = ((z - z0) * ny + (y - y0)) * nx + (x - x0)
    return block_offset(starts, proc_shape, block) + local_id


def stencil_offsets(stencil: int) -> list[tuple[int, int, int, float]]:
    offsets: list[tuple[int, int, int, float]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                distance = abs(dx) + abs(dy) + abs(dz)
                if distance == 0:
                    continue
                if stencil == 7 and distance != 1:
                    continue
                if stencil == 19 and distance == 3:
                    continue
                offsets.append((dx, dy, dz, -1.0 / float(distance)))
    return offsets


def build_local_problem(comm: MPI.Comm, shape: tuple[int, int, int],
                        proc_shape: tuple[int, int, int], stencil: int):
    import hypredrive as hd

    cart = comm.Create_cart(proc_shape, periods=(False, False, False), reorder=False)
    coords = tuple(cart.Get_coords(cart.Get_rank()))
    starts = tuple(split_points(n, p) for n, p in zip(shape, proc_shape))
    slices = tuple(slice(int(starts[d][coords[d]]), int(starts[d][coords[d] + 1]))
                   for d in range(3))
    row_start = block_offset(starts, proc_shape, coords)
    local_shape = tuple(s.stop - s.start for s in slices)
    row_end = row_start + int(np.prod(local_shape)) - 1

    offsets = stencil_offsets(stencil)
    indptr = [0]
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    nx, ny, nz = shape

    xs, ys, zs = slices
    for z in range(zs.start, zs.stop):
        for y in range(ys.start, ys.stop):
            for x in range(xs.start, xs.stop):
                row = gid(x, y, z, starts, proc_shape)
                diag = 0.0
                for dx, dy, dz, value in offsets:
                    xn, yn, zn = x + dx, y + dy, z + dz
                    diag -= value
                    if 0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz:
                        cols.append(gid(xn, yn, zn, starts, proc_shape))
                        data.append(value)
                cols.append(row)
                data.append(diag)
                rhs.append(1.0)
                indptr.append(len(cols))

    return (
        np.asarray(indptr, dtype=hd.BIGINT_DTYPE),
        np.asarray(cols, dtype=hd.BIGINT_DTYPE),
        np.asarray(data, dtype=hd.REAL_DTYPE),
        np.asarray(rhs, dtype=hd.REAL_DTYPE),
        row_start,
        row_end,
    )


def local_residual_l2(indptr, cols, data, rhs, x_global) -> float:
    r2 = 0.0
    for row in range(rhs.size):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        residual = float(rhs[row]) - float(np.dot(data[start:stop], x_global[cols[start:stop]]))
        r2 += residual * residual
    return r2


def gather_solution(comm: MPI.Comm, row_start: int, x_local: np.ndarray,
                    global_size: int) -> np.ndarray:
    counts = np.asarray(comm.allgather(x_local.size), dtype=np.int32)
    displs = np.asarray(comm.allgather(row_start), dtype=np.int32)
    x_global = np.empty(global_size, dtype=x_local.dtype)
    mpi_dtype = MPI.DOUBLE if x_local.dtype == np.float64 else MPI.FLOAT
    comm.Allgatherv(x_local, [x_global, counts, displs, mpi_dtype])
    return x_global


def options(tol: float, maxiter: int) -> dict[str, Any]:
    import hypredrive as hd

    return hd.configure(
        general={"statistics": False, "exec_policy": "host"},
        linear_system={"init_guess_mode": "zeros"},
        solver="pcg",
        preconditioner="amg",
        pcg={"max_iter": maxiter, "relative_tol": tol, "print_level": 0},
        amg={"print_level": 0},
    )


def run_hypredrive_solve_cycle(drv) -> dict[str, float]:
    timings: dict[str, float] = {}
    _, timings["initial_guess"] = time_call(drv._core.setup_initial_guess)
    _, timings["solver_create"] = time_call(drv._core.solver_create)
    try:
        _, timings["solver_setup"] = time_call(drv._core.solver_setup)
        _, timings["solver_apply"] = time_call(drv._core.solver_apply)
    finally:
        _, timings["solver_destroy"] = time_call(drv._core.solver_destroy)
    timings["solve"] = timings["solver_setup"] + timings["solver_apply"]
    return timings


def reduce_times(comm: MPI.Comm, timings: dict[str, float]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for name, value in timings.items():
        reduced[f"{name}_max_s"] = comm.allreduce(value, op=MPI.MAX)
        reduced[f"{name}_min_s"] = comm.allreduce(value, op=MPI.MIN)
        reduced[f"{name}_avg_s"] = comm.allreduce(value, op=MPI.SUM) / comm.Get_size()
    return reduced


def main() -> None:
    comm = MPI.COMM_WORLD
    args = parse_args(comm)
    shape = tuple(args.n)
    proc_shape = tuple(args.P)
    global_size = int(np.prod(shape))

    records = []
    for repeat in range(args.repeat):
        (problem, build_s) = time_call(
            lambda: build_local_problem(comm, shape, proc_shape, args.stencil)
        )
        indptr, cols, data, rhs, row_start, row_end = problem

        import hypredrive as hd

        total_start = time.perf_counter()
        drv, create_s = time_call(lambda: hd.HypreDrive(options=options(args.tol, args.maxiter), comm=comm))
        try:
            _, set_matrix_s = time_call(
                lambda: drv.set_matrix_from_csr(
                    indptr, cols, data, row_start=row_start, row_end=row_end
                )
            )
            _, set_rhs_s = time_call(lambda: drv.set_rhs(rhs))
            solve_timings = run_hypredrive_solve_cycle(drv)
            x_local, get_solution_s = time_call(drv.get_solution)
        finally:
            drv.close()
        x_global, gather_s = time_call(lambda: gather_solution(comm, row_start, x_local, global_size))
        local_r2, residual_s = time_call(lambda: local_residual_l2(indptr, cols, data, rhs, x_global))
        residual_l2 = float(np.sqrt(comm.allreduce(local_r2, op=MPI.SUM)))
        total_s = time.perf_counter() - total_start

        timings = reduce_times(
            comm,
            {
                "build": build_s,
                "create": create_s,
                "set_matrix": set_matrix_s,
                "set_rhs": set_rhs_s,
                "get_solution": get_solution_s,
                "gather": gather_s,
                "residual": residual_s,
                "total": total_s,
                **solve_timings,
            },
        )
        local_unknowns_min = comm.allreduce(rhs.size, op=MPI.MIN)
        local_unknowns_max = comm.allreduce(rhs.size, op=MPI.MAX)
        if comm.Get_rank() == 0:
            record = {
                "backend": "hypredrive",
                "repeat": repeat,
                "shape": list(shape),
                "proc_shape": list(proc_shape),
                "ranks": comm.Get_size(),
                "stencil": args.stencil,
                "unknowns": global_size,
                "local_unknowns_min": local_unknowns_min,
                "local_unknowns_max": local_unknowns_max,
                "residual_l2": residual_l2,
                "tol": args.tol,
                "maxiter": args.maxiter,
                **timings,
            }
            records.append(record)

    if comm.Get_rank() == 0:
        if args.json:
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            for record in records:
                print(
                    "hypredrive MPI "
                    f"ranks={record['ranks']} shape={tuple(record['shape'])} "
                    f"stencil={record['stencil']} repeat={record['repeat']}"
                )
                print(
                    f"  solve max: {record['solve_max_s']:.6e} s, "
                    f"total max: {record['total_max_s']:.6e} s, "
                    f"residual: {record['residual_l2']:.6e}"
                )


if __name__ == "__main__":
    main()
