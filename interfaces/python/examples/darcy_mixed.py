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

    ``K`` is the single 3x3 symmetric tensor used for every cell. We
    invert once and broadcast — cell-varying K is a trivial extension
    (build the (n_cells, 3, 3) array directly).
    """
    if K.shape != (3, 3):
        raise ValueError(f"K must be a 3x3 matrix, got shape {K.shape}")
    if not np.allclose(K, K.T, atol=1e-12):
        raise ValueError("K must be symmetric")
    K_inv = np.linalg.inv(K)
    out = np.broadcast_to(K_inv, (mesh.n_cells, 3, 3)).copy()
    return out


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


def boundary_default_pressure_drop(mesh: Mesh) -> BoundaryData:
    """u=0 on x=0, u=1 on x=L_x, no-flow on every other boundary face."""
    bd = BoundaryData(dirichlet={}, neumann={})

    # x = 0 (i = 0): Dirichlet u = 0
    for k in range(mesh.nz):
        for j in range(mesh.ny):
            bd.dirichlet[mesh.x_face_index(0, j, k)] = 0.0
    # x = L_x (i = nx): Dirichlet u = 1
    for k in range(mesh.nz):
        for j in range(mesh.ny):
            bd.dirichlet[mesh.x_face_index(mesh.nx, j, k)] = 1.0

    # y boundaries: no-flow
    if mesh.dim >= 2:
        for k in range(mesh.nz):
            for i in range(mesh.nx):
                bd.neumann[mesh.y_face_index(i, 0, k)] = 0.0
                bd.neumann[mesh.y_face_index(i, mesh.ny, k)] = 0.0
    # z boundaries: no-flow
    if mesh.dim >= 3:
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                bd.neumann[mesh.z_face_index(i, j, 0)] = 0.0
                bd.neumann[mesh.z_face_index(i, j, mesh.nz)] = 0.0
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

    # Strong-enforcement, three sparse ops (mirrors the MATLAB pattern):
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


def write_vti(
    filename: str,
    mesh: Mesh,
    pressure: np.ndarray,
    flux_cell: np.ndarray,
) -> None:
    """Write a binary zlib-compressed appended VTK ImageData file (.vti).

    Even in 1D / 2D the file is a 3-D ImageData with collapsed
    dimensions; ParaView handles that without trouble.
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
    if mesh.n_cells != n_cells_vtk:
        # Pad with zeros if we collapsed a dimension to a single slab.
        p_padded = np.zeros(n_cells_vtk, dtype=np.float64)
        f_padded = np.zeros((n_cells_vtk, 3), dtype=np.float64)
        p_padded[: mesh.n_cells] = pressure
        f_padded[: mesh.n_cells] = flux_cell
        pressure = p_padded
        flux_cell = f_padded

    # Build the appended block first so we know the per-array offsets
    # that go into the XML header.
    pressure_bytes = _encode_appended(pressure)
    flux_bytes = _encode_appended(flux_cell)
    offset_pressure = 0
    offset_flux = len(pressure_bytes)

    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64" compressor="vtkZLibDataCompressor">\n'
        f'  <ImageData WholeExtent="0 {ex_x} 0 {ex_y} 0 {ex_z}" '
        f'Origin="0 0 0" Spacing="{sx} {sy} {sz}">\n'
        f'    <Piece Extent="0 {ex_x} 0 {ex_y} 0 {ex_z}">\n'
        '      <CellData Scalars="pressure" Vectors="flux">\n'
        '        <DataArray type="Float64" Name="pressure" '
        f'format="appended" offset="{offset_pressure}"/>\n'
        '        <DataArray type="Float64" Name="flux" '
        'NumberOfComponents="3" format="appended" '
        f'offset="{offset_flux}"/>\n'
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nx", type=int, default=16, help="cells along x (default 16)")
    p.add_argument("--ny", type=int, default=1, help="cells along y (1 ⇒ 1D)")
    p.add_argument("--nz", type=int, default=1, help="cells along z (1 ⇒ 2D)")
    p.add_argument("--Lx", type=float, default=1.0, help="domain length in x")
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--Lz", type=float, default=1.0)
    p.add_argument(
        "--K",
        type=float,
        nargs="+",
        default=None,
        help=(
            "permeability tensor: one value (isotropic), three values "
            "(diagonal), or six values [kxx,kyy,kzz,kxy,kxz,kyz]"
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


def analytic_cell_pressure(mesh: Mesh, Lx: float) -> np.ndarray:
    """Cell-centered values of u(x) = x / L_x for the default BC problem."""
    u = np.empty(mesh.n_cells, dtype=np.float64)
    for k in range(mesh.nz):
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                x_centre = (i + 0.5) * mesh.hx
                u[mesh.cell_index(i, j, k)] = x_centre / Lx
    return u


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
    K = _parse_K(args.K)
    K_inv = build_K_inv(mesh, K)

    M, B = assemble_system(mesh, K_inv)
    bd = boundary_default_pressure_drop(mesh)
    A, rhs = apply_boundary_conditions(M, B, bd, mesh)

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
    print(f"GMRES iterations     : {iterations}")
    print(f"setup time [s]       : {setup_time:.6e}")
    print(f"solve time [s]       : {solve_time:.6e}")
    print(f"total time [s]       : {setup_time + solve_time:.6e}")
    print(f"||b - A x||_2 / ||b||: {rel_res:.3e}")
    # The analytic reference u(x)=x/Lx is only correct for diagonal K
    # (off-diagonal K makes the no-flow BC inconsistent with a purely-x
    # gradient). Print it only in that case so the L²-error line stays
    # meaningful.
    off_diag = max(abs(K[0, 1]), abs(K[0, 2]), abs(K[1, 2]))
    if off_diag < 1e-14:
        u_ref = analytic_cell_pressure(mesh, args.Lx)
        err_l2 = float(np.linalg.norm(u - u_ref) / np.sqrt(mesh.n_cells))
        print(f"L2 error vs analytic : {err_l2:.3e}")

    flux_cell = cell_centred_flux(mesh, q)
    if args.output:
        write_vti(args.output, mesh, u, flux_cell)
        print(f"wrote                : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
