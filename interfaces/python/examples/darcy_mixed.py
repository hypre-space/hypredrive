"""Mixed-form Darcy on a structured Cartesian mesh (1D / 2D / 3D).

Solves

    q + K grad u = 0        in Ω
    div q        = f        in Ω
    u            = u_D      on Γ_D
    q · n        = g_N      on Γ_N

with cell-centred piecewise-constant pressure (P0) and face-centred
lowest-order Raviart-Thomas flux (RT0). The RT0 DOF on a face F is the
integrated normal flux ``∫_F q · n_F dS`` using the global face normal
``+x̂`` / ``+ŷ`` / ``+ẑ``. The resulting block linear system is

    [ M   B ] [ q ]   [ g_D ]
    [ Bᵀ  0 ] [ u ] = [ -f  ]

with

    M[F,F'] = ∫_Ω K⁻¹ ψ_F · ψ_F' dV
    B[F,K]  = ∫_K div ψ_F dV     (= ±1 on the two cells of F)

The system is saddle-point, so we solve with GMRES preconditioned by
two-level MGR: the flux block is the F-block (eliminated via Jacobi
relaxation on its block diagonal), and the resulting Schur complement on
the cell pressures is handled by BoomerAMG.

Default boundary conditions (unit hypercube, ``K=I``, ``f=0``):
    u = 0 on x=0,  u = 1 on x=L_x,   q · n = 0 on every other face.
Analytical solution: ``u(x) = x / L_x``, ``q = (-1/L_x, 0, 0)``. The
script prints the relative residual plus the discrete L²-error of the
computed cell pressures against this reference.

VTI output (cell data): scalar ``pressure`` and 3-component ``flux``.
Each ``flux`` cell-centre value is the per-direction average of the two
opposite-face normal velocities (DOF divided by |F|), with the inactive
components set to zero.

Run as::

    mpirun -np 1 .venv/bin/python interfaces/python/examples/darcy_mixed.py \\
        --nx 16 --ny 16 --nz 16 --output darcy3d.vti
"""

from __future__ import annotations

import argparse
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Sequence

import mpi4py.MPI  # noqa: F401  -- side effect: calls MPI_Init
import numpy as np
import scipy.sparse as sp

import hypredrive as hd


# ----------------------------------------------------------------------
# Mesh / DOF bookkeeping
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Mesh:
    nx: int
    ny: int
    nz: int
    hx: float
    hy: float
    hz: float

    @property
    def dim(self) -> int:
        return sum(1 for n in (self.nx, self.ny, self.nz) if n > 1)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def n_x_faces(self) -> int:
        return (self.nx + 1) * self.ny * self.nz

    @property
    def n_y_faces(self) -> int:
        if self.dim < 2:
            return 0
        return self.nx * (self.ny + 1) * self.nz

    @property
    def n_z_faces(self) -> int:
        if self.dim < 3:
            return 0
        return self.nx * self.ny * (self.nz + 1)

    @property
    def n_faces(self) -> int:
        return self.n_x_faces + self.n_y_faces + self.n_z_faces

    @property
    def y_face_offset(self) -> int:
        return self.n_x_faces

    @property
    def z_face_offset(self) -> int:
        return self.n_x_faces + self.n_y_faces

    @property
    def face_areas(self) -> tuple[float, float, float]:
        # |F| for an x-face is hy*hz (and 1 when that direction is inactive).
        ax = (self.hy if self.ny > 1 else 1.0) * (self.hz if self.nz > 1 else 1.0)
        ay = self.hx * (self.hz if self.nz > 1 else 1.0)
        az = self.hx * self.hy
        return ax, ay, az

    @property
    def cell_volume(self) -> float:
        vx = self.hx
        vy = self.hy if self.ny > 1 else 1.0
        vz = self.hz if self.nz > 1 else 1.0
        return vx * vy * vz

    # ----- index helpers --------------------------------------------------

    def cell_index(self, i: int, j: int, k: int) -> int:
        return i + self.nx * (j + self.ny * k)

    def x_face_index(self, i: int, j: int, k: int) -> int:
        return i + (self.nx + 1) * (j + self.ny * k)

    def y_face_index(self, i: int, j: int, k: int) -> int:
        return self.y_face_offset + i + self.nx * (j + (self.ny + 1) * k)

    def z_face_index(self, i: int, j: int, k: int) -> int:
        return self.z_face_offset + i + self.nx * (j + self.ny * k)


# ----------------------------------------------------------------------
# K tensor
# ----------------------------------------------------------------------


def build_K_inv(mesh: Mesh, K: np.ndarray) -> np.ndarray:
    """Per-cell inverse permeability tensor, shape ``(n_cells, 3, 3)``.

    ``K`` may be either:

    * a single ``(3, 3)`` symmetric tensor used for every cell, or
    * a per-cell ``(n_cells, 3, 3)`` field (e.g. read from a tabulated
      permeability file).

    The assembly path consumes ``(n_cells, 3, 3)`` either way.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 3):
        if not np.allclose(K, K.T, atol=1e-12):
            raise ValueError("K must be symmetric")
        K_inv = np.linalg.inv(K)
        return np.broadcast_to(K_inv, (mesh.n_cells, 3, 3)).copy()
    if K.shape == (mesh.n_cells, 3, 3):
        if not np.allclose(K, np.transpose(K, (0, 2, 1)), atol=1e-12):
            raise ValueError("per-cell K must be symmetric")
        return np.linalg.inv(K)
    raise ValueError(
        f"K must have shape (3, 3) or ({mesh.n_cells}, 3, 3), got {K.shape}"
    )


def cell_centers(mesh: Mesh) -> np.ndarray:
    """Physical coordinates of every cell centre, shape ``(n_cells, 3)``.

    Cells are listed in ``cell_index`` order. Inactive directions use the
    mesh's collapsed spacing, so the centre still has a well-defined
    coordinate (e.g. ``0.5 * Ly`` for a 1D mesh).
    """
    cells = np.arange(mesh.n_cells, dtype=np.int64)
    i = cells % mesh.nx
    j = (cells // mesh.nx) % mesh.ny
    k = cells // (mesh.nx * mesh.ny)
    centers = np.empty((mesh.n_cells, 3), dtype=np.float64)
    centers[:, 0] = (i + 0.5) * mesh.hx
    centers[:, 1] = (j + 0.5) * mesh.hy
    centers[:, 2] = (k + 0.5) * mesh.hz
    return centers


# ----------------------------------------------------------------------
# Cell-local RT0 mass matrix
# ----------------------------------------------------------------------


def local_face_directions(dim: int) -> tuple[int, ...]:
    """Component index (0,1,2 = x,y,z) of each local face's normal."""
    if dim == 1:
        return (0, 0)  # W, E
    if dim == 2:
        return (0, 0, 1, 1)  # W, E, S, N
    return (0, 0, 1, 1, 2, 2)  # W, E, S, N, B, T


def local_face_is_low(dim: int) -> tuple[bool, ...]:
    """True for the "low-side" face in each direction (W, S, B)."""
    if dim == 1:
        return (True, False)
    if dim == 2:
        return (True, False, True, False)
    return (True, False, True, False, True, False)


def cell_mass_prefactor(mesh: Mesh) -> np.ndarray:
    """Cell-local prefactor matrix ``V * coef[a, b] / (|F_a| · |F_b|)``.

    The RT0 mass matrix factors cleanly: with the DOF convention
    ``σ_F(v) = ∫_F v · n_F dS``, each pair of local faces (a, b) of
    direction (d_a, d_b) contributes

        M_local[a, b] = K⁻¹[d_a, d_b] · V · coef(a, b) / (|F_a| · |F_b|)

    where ``coef(a, b)`` is purely geometric:

      * ``1/3`` if ``d_a == d_b`` and both faces are on the same side
        (low-low or high-high);
      * ``1/6`` if ``d_a == d_b`` but opposite sides (low-high);
      * ``1/4`` if ``d_a != d_b``.

    Returning that prefactor lets us assemble all cells in one
    broadcasted multiplication.
    """
    dim = mesh.dim
    n_local = 2 * dim
    dirs = np.asarray(local_face_directions(dim))
    is_low = np.asarray(local_face_is_low(dim))
    areas = np.asarray(mesh.face_areas)

    da = dirs[:, None]
    db = dirs[None, :]
    la = is_low[:, None]
    lb = is_low[None, :]

    same_dir = da == db
    same_low = la == lb
    coef = np.where(same_dir, np.where(same_low, 1.0 / 3.0, 1.0 / 6.0), 0.25)
    area_factor = 1.0 / (areas[da] * areas[db])
    return mesh.cell_volume * coef * area_factor


# ----------------------------------------------------------------------
# Global assembly
# ----------------------------------------------------------------------


def local_face_signs(dim: int) -> tuple[int, ...]:
    """Sign of B[K, F]: + for the high-side face, - for the low-side."""
    return tuple(-1 if low else +1 for low in local_face_is_low(dim))


def _cell_face_index_arrays(mesh: Mesh) -> np.ndarray:
    """Return per-cell global face indices as an ``(n_cells, n_local)`` array.

    Cells are listed in ``cell_index`` order (i fastest, then j, then k).
    Local face order matches :func:`local_face_directions`: x-faces (W,E)
    first, then y-faces (S,N) for 2D/3D, then z-faces (B,T) for 3D.
    """
    n_cells = mesh.n_cells
    cells = np.arange(n_cells, dtype=np.int64)
    nx, ny = mesh.nx, mesh.ny
    i = cells % nx
    j = (cells // nx) % ny
    k = cells // (nx * ny)

    blocks = [i + (nx + 1) * (j + ny * k),                 # W
              (i + 1) + (nx + 1) * (j + ny * k)]           # E
    if mesh.dim >= 2:
        y_off = mesh.y_face_offset
        blocks.append(y_off + i + nx * (j + (ny + 1) * k))         # S
        blocks.append(y_off + i + nx * ((j + 1) + (ny + 1) * k))   # N
    if mesh.dim >= 3:
        z_off = mesh.z_face_offset
        blocks.append(z_off + i + nx * (j + ny * k))               # B
        blocks.append(z_off + i + nx * (j + ny * (k + 1)))         # T
    return np.stack(blocks, axis=1)


def assemble_system(
    mesh: Mesh, K_inv: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Assemble (M, B) over the whole mesh, fully vectorised.

    For every cell, M's local block is
    ``M_local[c, a, b] = K⁻¹[c, d_a, d_b] · prefactor[a, b]``
    where ``prefactor`` is the geometric factor returned by
    :func:`cell_mass_prefactor`. Triplets are produced via broadcasting
    and handed once to ``sp.coo_matrix``, which sums shared-face
    contributions on the conversion to CSR.
    """
    dim = mesh.dim
    n_local = 2 * dim
    n_cells = mesh.n_cells

    faces = _cell_face_index_arrays(mesh)            # (n_cells, n_local)
    dirs = np.asarray(local_face_directions(dim))    # (n_local,)
    prefactor = cell_mass_prefactor(mesh)            # (n_local, n_local)

    # K_inv_local[c, a, b] = K_inv[c, dirs[a], dirs[b]]
    K_inv_local = K_inv[:, dirs[:, None], dirs[None, :]]
    m_vals = (K_inv_local * prefactor[None, :, :]).reshape(-1)

    shape = (n_cells, n_local, n_local)
    m_rows = np.broadcast_to(faces[:, :, None], shape).reshape(-1)
    m_cols = np.broadcast_to(faces[:, None, :], shape).reshape(-1)

    M = sp.coo_matrix(
        (m_vals, (m_rows, m_cols)), shape=(mesh.n_faces, mesh.n_faces)
    ).tocsr()

    signs = np.asarray(local_face_signs(dim), dtype=np.float64)
    b_rows = faces.reshape(-1)
    b_cols = np.repeat(np.arange(n_cells, dtype=np.int64), n_local)
    b_vals = np.tile(signs, n_cells)
    B = sp.coo_matrix(
        (b_vals, (b_rows, b_cols)), shape=(mesh.n_faces, mesh.n_cells)
    ).tocsr()
    return M, B


# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------


@dataclass
class BoundaryData:
    """One Dirichlet value or one Neumann value per face."""

    dirichlet: dict[int, float]      # face_idx -> u_D
    neumann: dict[int, float]        # face_idx -> g_N (q · n_outward)


def _axis_boundary_faces(mesh: Mesh, axis: int, high: bool):
    """Yield global face indices on the boundary perpendicular to ``axis``.

    ``high=False`` selects the low-coordinate boundary (i/j/k = 0),
    ``high=True`` the high-coordinate boundary (i/j/k = n_axis). ``axis``
    must be an active direction (``axis < mesh.dim``).
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    if axis == 0:
        i = nx if high else 0
        for k in range(nz):
            for j in range(ny):
                yield mesh.x_face_index(i, j, k)
    elif axis == 1:
        j = ny if high else 0
        for k in range(nz):
            for i in range(nx):
                yield mesh.y_face_index(i, j, k)
    else:
        k = nz if high else 0
        for j in range(ny):
            for i in range(nx):
                yield mesh.z_face_index(i, j, k)


def boundary_pressure_drop(mesh: Mesh, axis: int) -> BoundaryData:
    """Unit pressure drop along ``axis``; no-flow on the other boundaries.

    Sets ``u = 1`` on the low-coordinate boundary perpendicular to
    ``axis`` and ``u = 0`` on the high-coordinate boundary, so the
    pressure decreases along ``axis`` and the flux ``q = -K∇u`` points in
    the positive ``axis`` direction. Every boundary perpendicular to the
    other active directions is ``q · n = 0``. ``axis`` is 0/1/2 for x/y/z
    and must satisfy ``axis < mesh.dim``.
    """
    if not 0 <= axis < mesh.dim:
        raise ValueError(
            f"drive axis {axis} is not an active direction for a "
            f"{mesh.dim}-D mesh"
        )
    bd = BoundaryData(dirichlet={}, neumann={})
    for f in _axis_boundary_faces(mesh, axis, high=False):
        bd.dirichlet[f] = 1.0
    for f in _axis_boundary_faces(mesh, axis, high=True):
        bd.dirichlet[f] = 0.0
    for other in range(mesh.dim):
        if other == axis:
            continue
        for f in _axis_boundary_faces(mesh, other, high=False):
            bd.neumann[f] = 0.0
        for f in _axis_boundary_faces(mesh, other, high=True):
            bd.neumann[f] = 0.0
    return bd


def _face_outward_sign(mesh: Mesh, face_idx: int) -> int:
    """+1 if global face normal == domain outward normal, -1 otherwise.

    For a boundary face, this is +1 on east / north / top and -1 on
    west / south / bottom. Returns 0 for interior faces (not used here).
    """
    if face_idx < mesh.y_face_offset:
        # x-face: low-side (i=0) is west boundary, high-side (i=nx) is east
        i = face_idx % (mesh.nx + 1)
        if i == 0:
            return -1
        if i == mesh.nx:
            return +1
        return 0
    if face_idx < mesh.z_face_offset:
        f = face_idx - mesh.y_face_offset
        j = (f // mesh.nx) % (mesh.ny + 1)
        if j == 0:
            return -1
        if j == mesh.ny:
            return +1
        return 0
    f = face_idx - mesh.z_face_offset
    k = f // (mesh.nx * mesh.ny)
    if k == 0:
        return -1
    if k == mesh.nz:
        return +1
    return 0


def _face_area_of(mesh: Mesh, face_idx: int) -> float:
    ax, ay, az = mesh.face_areas
    if face_idx < mesh.y_face_offset:
        return ax
    if face_idx < mesh.z_face_offset:
        return ay
    return az


def apply_boundary_conditions(
    M: sp.csr_matrix,
    B: sp.csr_matrix,
    bd: BoundaryData,
    mesh: Mesh,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Assemble the saddle-point system with BC applied.

    Symmetric saddle-point form ``[[M, -B], [-Bᵀ, 0]] (q; u) = (-f_D; 0)``
    — matches the variational system
        ``∫ K⁻¹ q · v - ∫ u div v = -∫_{Γ_D} u_D (v · n_out)``
        ``∫ w div q = 0``
    after negating the second equation to symmetrise.

    Dirichlet BCs enter as a flux RHS. Neumann BCs are enforced strongly
    via the standard "pinned DOF" pattern: zero the pinned rows and
    columns, set 1 on their diagonal, and move the previously-coupled
    column contributions onto the RHS so the surviving free equations
    stay consistent.
    """
    n_f = mesh.n_faces
    n_c = mesh.n_cells
    n_total = n_f + n_c

    rhs = np.zeros(n_total, dtype=np.float64)
    for face_idx, u_d in bd.dirichlet.items():
        rhs[face_idx] -= _face_outward_sign(mesh, face_idx) * u_d

    Bt = B.T.tocsr()
    A = sp.bmat(
        [[M, -B], [-Bt, sp.csr_matrix((n_c, n_c))]],
        format="csr",
    )

    # Pack Neumann targets: q_F = sign · g_N · |F|.
    pinned_mask = np.zeros(n_total, dtype=bool)
    pinned_target = np.zeros(n_total, dtype=np.float64)
    for face_idx, g_n in bd.neumann.items():
        sign = _face_outward_sign(mesh, face_idx)
        if sign == 0:
            raise ValueError(
                f"Neumann face {face_idx} is not on the domain boundary"
            )
        pinned_mask[face_idx] = True
        pinned_target[face_idx] = sign * g_n * _face_area_of(mesh, face_idx)

    # Strong-enforcement via three sparse ops:
    #   D       = diag(free)            keeps free rows/cols, zeros pinned ones
    #   D_pin   = diag(pinned)          identity on pinned diagonal
    #   A_freeR = D · A                 pinned rows zeroed
    #   rhs    -= A_freeR · pinned_target   subtract pinned-col contributions
    #                                       (zero on pinned rows by construction)
    #   A_new   = A_freeR · D + D_pin    pinned cols zeroed; identity on pinned diag
    keep = (~pinned_mask).astype(np.float64)
    D = sp.diags(keep)
    D_pinned = sp.diags(pinned_mask.astype(np.float64))

    A_free_rows = D @ A
    rhs -= A_free_rows @ pinned_target
    rhs[pinned_mask] = pinned_target[pinned_mask]
    A_new = (A_free_rows @ D + D_pinned).tocsr()

    return A_new, rhs


# ----------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------


def mgr_options() -> dict:
    """GMRES + 1-level MGR; F-points = flux block, AMG on the Schur."""
    return {
        "general": {"statistics": False, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": "zeros"},
        "solver": {
            "gmres": {
                "max_iter": 200,
                "krylov_dim": 60,
                "relative_tol": 1.0e-10,
                "absolute_tol": 0.0,
                "print_level": 0,
            }
        },
        "preconditioner": {
            "mgr": {
                "tolerance": 0.0,
                "max_iter": 1,
                "print_level": 0,
                "coarse_th": 0.0,
                "level": {
                    "0": {
                        "f_dofs": [1],
                        "f_relaxation": "jacobi",
                        "g_relaxation": "none",
                        "restriction_type": "injection",
                        "prolongation_type": "jacobi",
                        "coarse_level_type": "rap",
                    },
                },
                "coarsest_level": {
                    "amg": {
                        "tolerance": 0.0,
                        "max_iter": 1,
                        "print_level": 0,
                    },
                },
            }
        },
    }


def make_dofmap(mesh: Mesh) -> np.ndarray:
    """Label 1 for flux DOFs, label 0 for cell DOFs."""
    labels = np.empty(mesh.n_faces + mesh.n_cells, dtype=np.intc)
    labels[: mesh.n_faces] = 1
    labels[mesh.n_faces :] = 0
    return labels


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------


def cell_centred_flux(mesh: Mesh, q: np.ndarray) -> np.ndarray:
    """Average opposite-face velocities to produce a cell-centred 3-vector.

    The RT0 DOF on face F is the integrated normal flux. We convert to a
    face-averaged velocity (DOF / |F|) and average the two opposite
    faces of each cell in each active direction. Inactive components are
    zero, so the output is always shape (n_cells, 3) for downstream VTI.
    """
    ax, ay, az = mesh.face_areas
    out = np.zeros((mesh.n_cells, 3), dtype=np.float64)
    for k in range(mesh.nz):
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                c = mesh.cell_index(i, j, k)
                vw = q[mesh.x_face_index(i, j, k)] / ax
                ve = q[mesh.x_face_index(i + 1, j, k)] / ax
                out[c, 0] = 0.5 * (vw + ve)
                if mesh.dim >= 2:
                    vs = q[mesh.y_face_index(i, j, k)] / ay
                    vn = q[mesh.y_face_index(i, j + 1, k)] / ay
                    out[c, 1] = 0.5 * (vs + vn)
                if mesh.dim >= 3:
                    vb = q[mesh.z_face_index(i, j, k)] / az
                    vt = q[mesh.z_face_index(i, j, k + 1)] / az
                    out[c, 2] = 0.5 * (vb + vt)
    return out


# VTK stores a symmetric second-order tensor as six components in the
# order (xx, yy, zz, xy, yz, xz). The matching (row, col) index pairs:
_SYM_TENSOR_INDICES = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
_SYM_TENSOR_LABELS = ("xx", "yy", "zz", "xy", "yz", "xz")


def symmetric_tensor_components(K_cells: np.ndarray) -> np.ndarray:
    """Pack a per-cell ``(n_cells, 3, 3)`` symmetric tensor as ``(n_cells, 6)``.

    Component order matches VTK's symmetric-tensor convention
    (xx, yy, zz, xy, yz, xz), so ParaView reads the array back as a
    symmetric tensor.
    """
    out = np.empty((K_cells.shape[0], 6), dtype=np.float64)
    for c, (r, s) in enumerate(_SYM_TENSOR_INDICES):
        out[:, c] = K_cells[:, r, s]
    return out


def _encode_appended(arr: np.ndarray) -> bytes:
    """Encode one DataArray for VTK ``<AppendedData encoding="raw">``.

    Single-block zlib compression. Header is four little-endian uint64s
    (``num_blocks=1``, ``block_size``, ``last_block_size``, and the
    compressed payload size), followed by the zlib bytes.
    """
    raw = np.ascontiguousarray(arr, dtype=np.float64).tobytes(order="C")
    compressed = zlib.compress(raw)
    header = struct.pack("<QQQQ", 1, len(raw), len(raw), len(compressed))
    return header + compressed


def _pad_cell_array(arr: np.ndarray, n_cells_vtk: int, n_real: int) -> np.ndarray:
    """Zero-pad a cell array up to the VTK cell count (collapsed dims)."""
    if arr.shape[0] == n_cells_vtk:
        return arr
    shape = (n_cells_vtk,) + arr.shape[1:]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:n_real] = arr
    return padded


def write_vti(
    filename: str,
    mesh: Mesh,
    pressure: np.ndarray,
    flux_cell: np.ndarray,
    permeability: np.ndarray,
) -> None:
    """Write a binary zlib-compressed appended VTK ImageData file (.vti).

    Cell data written: scalar ``pressure``, 3-vector ``flux``, and the
    symmetric ``permeability`` tensor as a 6-component array with named
    components (xx, yy, zz, xy, yz, xz).

    ``permeability`` is the ``(n_cells, 6)`` packing produced by
    :func:`symmetric_tensor_components`. Even in 1D / 2D the file is a
    3-D ImageData with collapsed dimensions; ParaView handles that
    without trouble.
    """
    # ImageData WholeExtent uses point indices. For Nx cells in x there
    # are Nx grid points spanning [0, Nx], etc. Inactive directions are
    # given a 1-cell "slab" of unit thickness so the file remains valid.
    ex_x = mesh.nx
    ex_y = mesh.ny if mesh.ny > 1 else 1
    ex_z = mesh.nz if mesh.nz > 1 else 1
    sx = mesh.hx
    sy = mesh.hy if mesh.ny > 1 else 1.0
    sz = mesh.hz if mesh.nz > 1 else 1.0

    # Pressure cell order: cell_index(i,j,k) = i + nx*(j + ny*k).
    # That matches VTI's expected i-fastest-then-j-then-k ordering.
    n_cells_vtk = ex_x * ex_y * ex_z
    n_real = mesh.n_cells
    pressure = _pad_cell_array(pressure, n_cells_vtk, n_real)
    flux_cell = _pad_cell_array(flux_cell, n_cells_vtk, n_real)
    permeability = _pad_cell_array(permeability, n_cells_vtk, n_real)

    # Build the appended blocks first so we know the per-array offsets
    # that go into the XML header.
    pressure_bytes = _encode_appended(pressure)
    flux_bytes = _encode_appended(flux_cell)
    perm_bytes = _encode_appended(permeability)
    offset_pressure = 0
    offset_flux = len(pressure_bytes)
    offset_perm = offset_flux + len(flux_bytes)

    perm_component_names = "".join(
        f'ComponentName{c}="{label}" '
        for c, label in enumerate(_SYM_TENSOR_LABELS)
    )

    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64" compressor="vtkZLibDataCompressor">\n'
        f'  <ImageData WholeExtent="0 {ex_x} 0 {ex_y} 0 {ex_z}" '
        f'Origin="0 0 0" Spacing="{sx} {sy} {sz}">\n'
        f'    <Piece Extent="0 {ex_x} 0 {ex_y} 0 {ex_z}">\n'
        '      <CellData Scalars="pressure" Vectors="flux" '
        'Tensors="permeability">\n'
        '        <DataArray type="Float64" Name="pressure" '
        f'format="appended" offset="{offset_pressure}"/>\n'
        '        <DataArray type="Float64" Name="flux" '
        'NumberOfComponents="3" format="appended" '
        f'offset="{offset_flux}"/>\n'
        '        <DataArray type="Float64" Name="permeability" '
        f'NumberOfComponents="6" {perm_component_names}'
        f'format="appended" offset="{offset_perm}"/>\n'
        '      </CellData>\n'
        '      <PointData></PointData>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n'
        '   _'
    )
    footer = b"\n  </AppendedData>\n</VTKFile>\n"

    with open(filename, "wb") as fh:
        fh.write(header.encode("utf-8"))
        fh.write(pressure_bytes)
        fh.write(flux_bytes)
        fh.write(perm_bytes)
        fh.write(footer)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_K(values: Sequence[float] | None) -> np.ndarray:
    """Build a symmetric 3x3 K from CLI input.

    Accepts:
      - None / empty: identity
      - 1 value: isotropic k·I
      - 3 values: diag(kxx, kyy, kzz)
      - 6 values: full symmetric [kxx, kyy, kzz, kxy, kxz, kyz]
    """
    if not values:
        return np.eye(3)
    vals = list(values)
    if len(vals) == 1:
        return vals[0] * np.eye(3)
    if len(vals) == 3:
        return np.diag(vals)
    if len(vals) == 6:
        kxx, kyy, kzz, kxy, kxz, kyz = vals
        return np.array(
            [[kxx, kxy, kxz], [kxy, kyy, kyz], [kxz, kyz, kzz]],
            dtype=np.float64,
        )
    raise ValueError(
        "--K expects 1, 3, or 6 values "
        "(isotropic / diagonal / full symmetric); got "
        f"{len(vals)}"
    )


def _diagonal_K_from_components(
    kx: np.ndarray, ky: np.ndarray, kz: np.ndarray
) -> np.ndarray:
    """Pack three per-cell permeability components into ``(n_cells, 3, 3)``.

    The tabulated reader produces a diagonal (orthotropic) tensor; the
    assembly path still treats it as a full symmetric tensor, so the
    off-diagonal entries are simply left at zero.
    """
    n = kx.shape[0]
    if np.any(kx <= 0) or np.any(ky <= 0) or np.any(kz <= 0):
        raise SystemExit("permeability values must be strictly positive")
    K = np.zeros((n, 3, 3), dtype=np.float64)
    K[:, 0, 0] = kx
    K[:, 1, 1] = ky
    K[:, 2, 2] = kz
    return K


def _read_flat_components(
    path: str, n_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a flat permeability stream into ``(kx, ky, kz)`` components.

    The file is a whitespace-separated stream of numbers. The total count
    selects the layout, relative to the expected ``n_points``:

    * ``n_points`` values     → isotropic (kx = ky = kz);
    * ``3 * n_points`` values → three contiguous blocks
      ``[kx ...][ky ...][kz ...]``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        values = np.array(fh.read().split(), dtype=np.float64)

    if values.size == n_points:
        return values, values, values
    if values.size == 3 * n_points:
        return (
            values[:n_points],
            values[n_points : 2 * n_points],
            values[2 * n_points :],
        )
    raise SystemExit(
        f"permeability file '{path}' has {values.size} values; expected "
        f"{n_points} (isotropic) or {3 * n_points} (kx,ky,kz blocks)"
    )


def _read_perm_one_to_one(path: str, mesh: Mesh, k_order: str) -> np.ndarray:
    """Read a flat SPE10-style permeability stream onto the internal mesh.

    No coordinates: the stream maps one-to-one onto the mesh cells. Each
    component block has ``n_cells`` values running with the I index
    fastest, then J, then K (the reservoir-simulation convention).
    ``k_order`` selects how the K layers map onto the mesh:

    * ``"bottom-up"``: file layer 0 is the lowest-z layer (k = 0). No flip.
    * ``"top-down"``:  file layer 0 is the highest-z layer (the common
      SPE10 convention, ``Kmax:-1:Kmin``); the K axis is reversed so it
      lands on the mesh with k = 0 at the bottom.
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    blocks = _read_flat_components(path, mesh.n_cells)

    components = []
    for block in blocks:
        # I-fastest, then J, then K  →  reshape so [k, j, i] addresses it.
        grid = block.reshape(nz, ny, nx)
        if k_order == "top-down":
            grid = grid[::-1]
        components.append(np.ascontiguousarray(grid).reshape(-1))
    return _diagonal_K_from_components(*components)


def _read_separable_axes(
    coords_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a separable (rectilinear) coordinate file.

    Layout (whitespace-separated, ``#`` comment lines ignored):

    * the three integer counts ``nX nY nZ``;
    * ``nX`` X-coordinates, then ``nY`` Y-coordinates, then ``nZ``
      Z-coordinates.

    Storing the axes rather than the full ``nX*nY*nZ`` outer product keeps
    the file compact and lets the reader use a fast tensor-product
    interpolator.
    """
    tokens = [
        tok
        for line in open(coords_path, "r", encoding="utf-8")
        if not line.lstrip().startswith("#")
        for tok in line.split()
    ]
    vals = np.array(tokens, dtype=np.float64)
    if vals.size < 3:
        raise SystemExit(
            f"coordinate file '{coords_path}' must start with the three "
            "counts 'nX nY nZ'"
        )
    n_x, n_y, n_z = (int(vals[0]), int(vals[1]), int(vals[2]))
    rest = vals[3:]
    if rest.size != n_x + n_y + n_z:
        raise SystemExit(
            f"coordinate file '{coords_path}' declares counts "
            f"{n_x} {n_y} {n_z} (sum {n_x + n_y + n_z}) but provides "
            f"{rest.size} coordinate values"
        )
    x_axis = rest[:n_x]
    y_axis = rest[n_x : n_x + n_y]
    z_axis = rest[n_x + n_y :]
    return x_axis, y_axis, z_axis


def _read_perm_interpolated(
    path: str, coords_path: str, mesh: Mesh, method: str
) -> np.ndarray:
    """Resample a tabulated permeability field onto the mesh cell centres.

    The permeability values come from ``path`` as a flat stream (see
    :func:`_read_flat_components`); the table grid comes from the separable
    ``coords_path`` file (see :func:`_read_separable_axes`). The value
    stream describes the ``nX * nY * nZ`` grid points with the I index
    (X) fastest, then J (Y), then K (Z) — so it has ``n_points`` (isotropic)
    or ``3 * n_points`` entries with ``n_points = nX * nY * nZ``.

    The value stream's Z-block ``k`` is placed at the ``k``-th z-coordinate
    listed in the coordinate file, so the *coordinate file's z-axis order*
    selects the physical layering. For a top-down value stream (e.g. SPE10,
    where the first block is the top layer) list the z-axis descending
    (high z first); for a bottom-up stream list it ascending. Listed axes
    are sorted ascending for the interpolator (the value grid is reordered
    to match), so either direction is accepted.

    ``method`` is the ``RegularGridInterpolator`` mode over the table axes
    with more than one sample:

    * ``"nearest"``: piecewise-constant — each cell takes the value of the
      closest table point. The physically appropriate choice for a cell
      property, and it keeps every value within the table's range (so it
      stays positive and never extrapolates spuriously when the mesh is
      finer/coarser than, or extends beyond, the table).
    * ``"linear"``: tensor-product linear, extrapolating linearly outside
      the table box (suitable for smooth fields).
    """
    from scipy.interpolate import RegularGridInterpolator

    x_axis, y_axis, z_axis = _read_separable_axes(coords_path)
    n_x, n_y, n_z = x_axis.size, y_axis.size, z_axis.size
    n_points = n_x * n_y * n_z
    kx, ky, kz = _read_flat_components(path, n_points)

    # Grid arrays index as [k, j, i] = (z, y, x), matching the value stream.
    grid_axes = (z_axis, y_axis, x_axis)
    grid_shape = (n_z, n_y, n_x)
    orders = tuple(np.argsort(ax) for ax in grid_axes)
    sorted_axes = tuple(grid_axes[d][orders[d]] for d in range(3))
    # Interpolate only over axes with more than one sample.
    keep = [d for d in range(3) if grid_shape[d] > 1]

    # Cell-centre coordinates in (z, y, x) order to match grid_axes.
    dst_zyx = cell_centers(mesh)[:, [2, 1, 0]]
    dst = dst_zyx[:, keep]
    interp_axes = [sorted_axes[d] for d in keep]

    components = []
    for comp in (kx, ky, kz):
        grid = comp.reshape(grid_shape)
        grid = grid[np.ix_(*orders)]
        # Drop degenerate (single-sample) axes.
        grid = grid[tuple(slice(None) if grid_shape[d] > 1 else 0 for d in range(3))]
        interp = RegularGridInterpolator(
            interp_axes, grid, method=method,
            bounds_error=False, fill_value=None,
        )
        components.append(interp(dst))
    return _diagonal_K_from_components(*components)


def read_permeability_field(
    path: str, mesh: Mesh, coords_path: str | None, k_order: str, method: str
) -> np.ndarray:
    """Dispatch to the interpolated or one-to-one tabulated reader.

    When ``coords_path`` is given, the value stream is resampled from the
    table grid to the mesh cell centres with ``method`` ("nearest" or
    "linear"); the physical layering is set by the z-axis order in the
    coordinate file (``k_order`` is unused). Otherwise the stream maps
    one-to-one onto the mesh and ``k_order`` selects the K-layer direction.
    """
    if coords_path:
        return _read_perm_interpolated(path, coords_path, mesh, method)
    return _read_perm_one_to_one(path, mesh, k_order)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nx", type=int, default=16, help="cells along x (default 16)")
    p.add_argument("--ny", type=int, default=1, help="cells along y (1 ⇒ 1D)")
    p.add_argument("--nz", type=int, default=1, help="cells along z (1 ⇒ 2D)")
    p.add_argument("--Lx", type=float, default=1.0, help="domain length in x")
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--Lz", type=float, default=1.0)
    p.add_argument(
        "--gradient-direction",
        dest="gradient_direction",
        choices=("x", "y", "z"),
        default="x",
        help=(
            "axis along which the unit pressure drop (u=0 to u=1) is "
            "enforced; the other active boundaries are no-flow. Must be an "
            "active direction (default x)"
        ),
    )
    k_group = p.add_mutually_exclusive_group()
    k_group.add_argument(
        "--K",
        type=float,
        nargs="+",
        default=None,
        help=(
            "constant permeability tensor: one value (isotropic), three "
            "values (diagonal), or six values [kxx,kyy,kzz,kxy,kxz,kyz]"
        ),
    )
    k_group.add_argument(
        "--K-file",
        dest="K_file",
        default=None,
        help=(
            "read a tabulated permeability field from an ASCII file: a flat "
            "SPE10-style value stream (n_points isotropic values, or kx/ky/kz "
            "blocks of n_points each). Without --K-coords-file the stream maps "
            "one-to-one onto the mesh (n_points = n_cells, I-fastest then J "
            "then K). With --K-coords-file the values are interpolated to cell "
            "centres."
        ),
    )
    p.add_argument(
        "--K-coords-file",
        dest="K_coords_file",
        default=None,
        help=(
            "ASCII separable grid for the --K-file table: the counts "
            "'nX nY nZ' followed by nX X-coordinates, nY Y-coordinates and "
            "nZ Z-coordinates. The value stream must list the nX*nY*nZ grid "
            "points X-fastest then Y then Z. When given, the permeability is "
            "interpolated from this grid onto the mesh cell centres."
        ),
    )
    p.add_argument(
        "--K-file-k-order",
        dest="K_file_k_order",
        choices=("bottom-up", "top-down"),
        default="bottom-up",
        help=(
            "layer ordering of a one-to-one (no --K-coords-file) value "
            "stream: 'bottom-up' (layer 0 = lowest z) or 'top-down' (layer 0 "
            "= highest z, the SPE10 Kmax:-1:Kmin convention). Ignored when "
            "--K-coords-file is set."
        ),
    )
    p.add_argument(
        "--K-interp",
        dest="K_interp",
        choices=("nearest", "linear"),
        default="nearest",
        help=(
            "resampling mode for --K-coords-file: 'nearest' (piecewise "
            "constant — each cell takes the closest table point; the right "
            "choice for a cell property and robust to refining/coarsening) "
            "or 'linear' (tensor-product, for smooth fields). Default nearest."
        ),
    )
    p.add_argument(
        "--output",
        default="darcy_mixed.vti",
        help="VTI output filename (set to '' to skip writing)",
    )
    p.add_argument(
        "--print-level",
        type=int,
        default=0,
        help="GMRES print_level (0 silent, 2 per-iter convergence)",
    )
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def analytic_cell_pressure(mesh: Mesh, axis: int, length: float) -> np.ndarray:
    """Cell-centred ``u = 1 - coord_axis / length`` for the pressure-drop problem.

    Exact solution when the drive is along ``axis`` (u=1 on the low
    boundary, u=0 on the high boundary), K is homogeneous and diagonal,
    and the lateral boundaries are no-flow.
    """
    return 1.0 - cell_centers(mesh)[:, axis] / length


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.nx, args.ny, args.nz) < 1:
        raise SystemExit("nx, ny, nz must each be >= 1")
    if args.nx <= 1 and args.ny <= 1 and args.nz <= 1:
        raise SystemExit("at least one of nx/ny/nz must be > 1")

    mesh = Mesh(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        hx=args.Lx / args.nx,
        hy=args.Ly / max(args.ny, 1),
        hz=args.Lz / max(args.nz, 1),
    )

    drive_axis = {"x": 0, "y": 1, "z": 2}[args.gradient_direction]
    if drive_axis >= mesh.dim:
        active = ", ".join(["x", "y", "z"][: mesh.dim])
        raise SystemExit(
            f"--gradient-direction {args.gradient_direction} requires an "
            f"active {args.gradient_direction} dimension; this "
            f"{mesh.dim}-D mesh only drives along: {active}"
        )
    drive_length = (args.Lx, args.Ly, args.Lz)[drive_axis]

    if args.K_coords_file and not args.K_file:
        raise SystemExit("--K-coords-file requires --K-file")

    read_time = None
    if args.K_file:
        t0 = time.perf_counter()
        K = read_permeability_field(
            args.K_file, mesh, args.K_coords_file, args.K_file_k_order,
            args.K_interp,
        )
        read_time = time.perf_counter() - t0
        analytic_valid = False
    else:
        K = _parse_K(args.K)
        # The analytic reference u(x)=x/Lx holds only for a homogeneous
        # diagonal tensor (off-diagonal K makes the no-flow BC inconsistent
        # with a purely-x gradient).
        analytic_valid = max(abs(K[0, 1]), abs(K[0, 2]), abs(K[1, 2])) < 1e-14
    K_inv = build_K_inv(mesh, K)

    t0 = time.perf_counter()
    M, B = assemble_system(mesh, K_inv)
    assemble_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    bd = boundary_pressure_drop(mesh, drive_axis)
    A, rhs = apply_boundary_conditions(M, B, bd, mesh)
    bc_time = time.perf_counter() - t0

    opts = mgr_options()
    if args.print_level:
        opts["solver"]["gmres"]["print_level"] = args.print_level

    with hd.HypreDrive(options=opts) as drv:
        n_total = mesh.n_faces + mesh.n_cells
        drv.set_matrix_from_csr(A, row_start=0, row_end=n_total - 1)
        drv.set_rhs(rhs)
        drv.set_dofmap(make_dofmap(mesh))
        drv.solve()
        iterations = drv.last_iterations
        setup_time = drv.last_setup_time
        solve_time = drv.last_solve_time
        x = drv.get_solution()

    q = x[: mesh.n_faces]
    u = x[mesh.n_faces :]

    residual = float(np.linalg.norm(rhs - A @ x))
    rhs_norm = float(np.linalg.norm(rhs))
    rel_res = residual / rhs_norm if rhs_norm > 0 else residual

    print(f"dim                  : {mesh.dim}")
    print(
        f"grid                 : "
        f"{mesh.nx} x {mesh.ny} x {mesh.nz}   "
        f"(cells={mesh.n_cells}, flux DOFs={mesh.n_faces}, "
        f"total={mesh.n_faces + mesh.n_cells})"
    )
    print(f"gradient direction   : {args.gradient_direction}")
    if args.K_file:
        diag = np.diagonal(K, axis1=1, axis2=2)
        print(
            f"permeability         : file '{args.K_file}' "
            f"(diag range [{diag.min():.3e}, {diag.max():.3e}])"
        )
        print(f"read time [s]        : {read_time:.6e}")
    print(f"GMRES iterations     : {iterations}")
    print(f"||b - A x||_2 / ||b||: {rel_res:.3e}")
    # Print the analytic L²-error only when the reference u(x)=x/Lx is
    # valid: a homogeneous diagonal tensor. Heterogeneous (file-based) or
    # full-tensor K do not admit this closed form.
    if analytic_valid:
        u_ref = analytic_cell_pressure(mesh, drive_axis, drive_length)
        err_l2 = float(np.linalg.norm(u - u_ref) / np.sqrt(mesh.n_cells))
        print(f"L2 error vs analytic : {err_l2:.3e}")

    flux_cell = cell_centred_flux(mesh, q)
    output_time = 0.0
    if args.output:
        if K.shape == (3, 3):
            K_cells = np.broadcast_to(K, (mesh.n_cells, 3, 3))
        else:
            K_cells = K
        perm_components = symmetric_tensor_components(K_cells)
        t0 = time.perf_counter()
        write_vti(args.output, mesh, u, flux_cell, perm_components)
        output_time = time.perf_counter() - t0
        print(f"wrote                : {args.output}")

    # Timing breakdown (host-side stages plus the HYPRE solver phases).
    stages = [
        ("read perm", read_time or 0.0),
        ("assemble", assemble_time),
        ("boundary cond", bc_time),
        ("solver setup", setup_time),
        ("solver solve", solve_time),
        ("write vti", output_time),
    ]
    total = sum(t for _, t in stages)
    print("timings [s]          :")
    for name, t in stages:
        pct = 100.0 * t / total if total > 0 else 0.0
        print(f"  {name:<14}: {t:.6e}  ({pct:5.1f}%)")
    print(f"  {'total':<14}: {total:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
