"""Distributed 3D Laplacian example using the hypredrive Python interface.

The example intentionally follows a compact, PyAMG-like style: build a problem,
solve it, report a residual, and optionally write a VTK file.

Run as:

    mpiexec -n 4 python interfaces/python/examples/laplacian.py \
        -n 16 16 16 -P 2 2 1 -s 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

from mpi4py import MPI

import numpy as np


class Grid(NamedTuple):
    comm: MPI.Cartcomm
    shape: tuple[int, int, int]
    proc_shape: tuple[int, int, int]
    proc_coords: tuple[int, int, int]
    starts: tuple[np.ndarray, np.ndarray, np.ndarray]
    slices: tuple[slice, slice, slice]

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        return self.comm.Get_size()

    @property
    def local_shape(self) -> tuple[int, int, int]:
        return tuple(s.stop - s.start for s in self.slices)

    @property
    def row_start(self) -> int:
        return block_offset(self.starts, self.proc_shape, self.proc_coords)

    @property
    def row_end(self) -> int:
        return self.row_start + np.prod(self.local_shape) - 1


def parse_args(comm: MPI.Comm):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", help="YAML solver options")
    parser.add_argument("-n", nargs=3, type=int, default=(10, 10, 10),
                        metavar=("NX", "NY", "NZ"))
    parser.add_argument("-P", nargs=3, type=int, default=(1, 1, 1),
                        metavar=("PX", "PY", "PZ"))
    parser.add_argument("-c", nargs=3, type=float, default=(1.0, 1.0, 1.0),
                        metavar=("CX", "CY", "CZ"))
    parser.add_argument("-c2", nargs=3, type=float, default=(1.0, 1.0, 1.0),
                        metavar=("CXY", "CXZ", "CYZ"))
    parser.add_argument("-s", "--stencil", type=int, choices=(7, 19, 27, 125),
                        default=7)
    parser.add_argument("-ns", "--nsolve", type=int, default=1)
    parser.add_argument("-vis", "--visualize", action="store_true")
    parser.add_argument("-v", "--verbose", type=int, default=1)
    args = parser.parse_args()

    shape = tuple(args.n)
    proc_shape = tuple(args.P)
    if any(n <= 0 for n in shape + proc_shape):
        parser.error("grid and processor dimensions must be positive")
    if np.prod(proc_shape) != comm.Get_size():
        parser.error(f"-P product {np.prod(proc_shape)} must match MPI size {comm.Get_size()}")
    if any(p > n for p, n in zip(proc_shape, shape)):
        parser.error("each processor dimension must be <= the matching grid dimension")
    if args.nsolve <= 0:
        parser.error("--nsolve must be positive")
    return args


def split_points(n: int, p: int) -> np.ndarray:
    base, extra = divmod(n, p)
    return np.array([base * i + min(i, extra) for i in range(p + 1)], dtype=np.int64)


def make_grid(comm: MPI.Comm, shape: tuple[int, int, int],
              proc_shape: tuple[int, int, int]) -> Grid:
    cart = comm.Create_cart(proc_shape, periods=(False, False, False), reorder=False)
    coords = tuple(cart.Get_coords(cart.Get_rank()))
    starts = tuple(split_points(n, p) for n, p in zip(shape, proc_shape))
    slices = tuple(slice(int(starts[d][coords[d]]), int(starts[d][coords[d] + 1]))
                   for d in range(3))
    return Grid(cart, shape, proc_shape, coords, starts, slices)


def block_for_coord(coord: int, starts: np.ndarray) -> int:
    return int(np.searchsorted(starts, coord, side="right") - 1)


def block_size(starts: tuple[np.ndarray, np.ndarray, np.ndarray],
               coords: tuple[int, int, int]) -> int:
    return int(np.prod([starts[d][coords[d] + 1] - starts[d][coords[d]]
                        for d in range(3)]))


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


def gid(x: int, y: int, z: int, grid: Grid) -> int:
    block = tuple(block_for_coord(coord, starts)
                  for coord, starts in zip((x, y, z), grid.starts))
    x0, y0, z0 = (int(grid.starts[d][block[d]]) for d in range(3))
    nx, ny, _ = (int(grid.starts[d][block[d] + 1] - grid.starts[d][block[d]])
                 for d in range(3))
    local_id = ((z - z0) * ny + (y - y0)) * nx + (x - x0)
    return block_offset(grid.starts, grid.proc_shape, block) + local_id


def owned_points(grid: Grid):
    xs, ys, zs = grid.slices
    for z in range(zs.start, zs.stop):
        for y in range(ys.start, ys.stop):
            for x in range(xs.start, xs.stop):
                yield x, y, z


def stencil_offsets(stencil: int, c: tuple[float, float, float],
                    c2: tuple[float, float, float]):
    if stencil == 7:
        return [
            (-1, 0, 0, -c[0]), (1, 0, 0, -c[0]),
            (0, -1, 0, -c[1]), (0, 1, 0, -c[1]),
            (0, 0, -1, -c[2]), (0, 0, 1, -c[2]),
        ]

    if stencil == 125:
        return [
            (dx, dy, dz, -1.0 if abs(dx) + abs(dy) + abs(dz) == 1 else -0.01)
            for dz in range(-2, 3)
            for dy in range(-2, 3)
            for dx in range(-2, 3)
            if (dx, dy, dz) != (0, 0, 0)
        ]

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ndiff = abs(dx) + abs(dy) + abs(dz)
                if ndiff == 0 or (stencil == 19 and ndiff == 3):
                    continue
                if stencil == 19:
                    if ndiff == 1:
                        weight = -(c[0] if dx else c[1] if dy else c[2])
                    elif dx == 0:
                        weight = -0.5 * c2[2]
                    elif dy == 0:
                        weight = -0.5 * c2[1]
                    else:
                        weight = -0.5 * c2[0]
                else:
                    adj = (abs(dx) * c[0] + abs(dy) * c[1] + abs(dz) * c[2]) / ndiff
                    weight = -adj
                offsets.append((dx, dy, dz, weight))
    return offsets


def build_csr(grid: Grid, stencil: int, c: tuple[float, float, float],
              c2: tuple[float, float, float]):
    import hypredrive as hd

    offsets = stencil_offsets(stencil, c, c2)
    indptr = [0]
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    nx, ny, nz = grid.shape

    for x, y, z in owned_points(grid):
        row = gid(x, y, z, grid)
        diag = 0.0
        row_rhs = 1.0 if y == 0 else 0.0
        for dx, dy, dz, weight in offsets:
            xn, yn, zn = x + dx, y + dy, z + dz
            diag -= weight
            if 0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz:
                cols.append(gid(xn, yn, zn, grid))
                data.append(weight)
        cols.append(row)
        data.append(diag)
        rhs.append(row_rhs)
        indptr.append(len(cols))

    return (
        np.asarray(indptr, dtype=hd.BIGINT_DTYPE),
        np.asarray(cols, dtype=hd.BIGINT_DTYPE),
        np.asarray(data, dtype=hd.REAL_DTYPE),
        np.asarray(rhs, dtype=hd.REAL_DTYPE),
    )


def default_options():
    import hypredrive as hd

    return hd.configure(
        general={"statistics": False, "exec_policy": "host"},
        linear_system={"init_guess_mode": "zeros"},
        solver="pcg",
        preconditioner="amg",
        pcg={"max_iter": 200, "relative_tol": 1.0e-8, "print_level": 0},
        amg={"print_level": 0},
    )


def mpi_dtype(dtype: np.dtype):
    return MPI.DOUBLE if np.dtype(dtype) == np.float64 else MPI.FLOAT


def gather_solution(grid: Grid, x_local: np.ndarray) -> np.ndarray:
    counts = np.asarray(grid.comm.allgather(x_local.size), dtype=np.int32)
    displs = np.asarray(grid.comm.allgather(grid.row_start), dtype=np.int32)
    x = np.empty(np.prod(grid.shape), dtype=x_local.dtype)
    grid.comm.Allgatherv(x_local, [x, counts, displs, mpi_dtype(x_local.dtype)])
    return x


def residual_norm(indptr, cols, data, rhs, x_global) -> float:
    r2 = 0.0
    for row in range(rhs.size):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        r = float(rhs[row]) - float(np.dot(data[start:stop], x_global[cols[start:stop]]))
        r2 += r * r
    return r2


def write_vtk(grid: Grid, x_local: np.ndarray, stencil: int) -> Path:
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise RuntimeError("Install `vtk` to use -vis: python -m pip install vtk") from exc

    def vtk_array(a):
        return numpy_support.numpy_to_vtk(np.asarray(a, dtype=np.float64))

    nx, ny, nz = grid.local_shape
    xs, ys, zs = grid.slices
    hx, hy, hz = (1.0 / (n - 1) if n > 1 else 1.0 for n in grid.shape)

    vtk_grid = vtk.vtkRectilinearGrid()
    vtk_grid.SetDimensions(nx, ny, nz)
    vtk_grid.SetXCoordinates(vtk_array((xs.start + np.arange(nx)) * hx))
    vtk_grid.SetYCoordinates(vtk_array((ys.start + np.arange(ny)) * hy))
    vtk_grid.SetZCoordinates(vtk_array((zs.start + np.arange(nz)) * hz))

    values = vtk_array(x_local)
    values.SetName("solution")
    vtk_grid.GetPointData().SetScalars(values)

    filename = Path(f"laplacian_{stencil}pt_rank{grid.rank:04d}.vtr")
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(vtk_grid)
    writer.SetDataModeToAscii()
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {filename}")
    return filename


def main() -> None:
    comm = MPI.COMM_WORLD
    args = parse_args(comm)

    import hypredrive as hd

    grid = make_grid(comm, tuple(args.n), tuple(args.P))
    if args.verbose and grid.rank == 0:
        print(f"grid           : {args.n[0]} x {args.n[1]} x {args.n[2]}")
        print(f"processor grid : {args.P[0]} x {args.P[1]} x {args.P[2]}")
        print(f"stencil        : {args.stencil}-point")

    indptr, cols, data, rhs = build_csr(grid, args.stencil, tuple(args.c), tuple(args.c2))
    options = args.input if args.input else default_options()

    with hd.HypreDrive(options=options, comm=grid.comm) as drv:
        drv.set_matrix_from_csr(indptr, cols, data,
                                row_start=grid.row_start, row_end=grid.row_end)
        drv.set_rhs(rhs)
        for i in range(args.nsolve):
            if args.verbose and grid.rank == 0:
                print(f"solve {i + 1}/{args.nsolve}")
            drv.solve()
        x_local = drv.get_solution()
        solution_norm = drv.solution_norm()

    x_global = gather_solution(grid, x_local)
    local_r2 = residual_norm(indptr, cols, data, rhs, x_global)
    res_norm = np.sqrt(grid.comm.allreduce(local_r2, op=MPI.SUM))

    if args.visualize:
        try:
            vtk_file = write_vtk(grid, x_local, args.stencil)
        except RuntimeError as exc:
            if grid.rank == 0:
                print(exc)
            raise SystemExit(1) from None
        if args.verbose:
            print(f"rank {grid.rank}: wrote {vtk_file}")

    if grid.rank == 0:
        print(f"ranks          : {grid.size}")
        print(f"unknowns       : {np.prod(grid.shape)}")
        print(f"solution norm  : {solution_norm:.6e}")
        print(f"||b - A x||_2  : {res_norm:.6e}")


if __name__ == "__main__":
    main()
