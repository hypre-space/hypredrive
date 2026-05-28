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


def cell_mass_matrix(mesh: Mesh, K_inv_cell: np.ndarray) -> np.ndarray:
    """RT0 cell-local mass matrix ``∫_cell K⁻¹ ψ_a · ψ_b dV``.

    The integrals are exact for affine basis functions in a Cartesian
    cell. With the DOF convention "σ_F(v) = ∫_F v · n_F dS" (integrated
    normal flux), the basis on cell K of size (hx, hy, hz) is, for the
    low-side x-face,

        ψ_W = ((h_x - x_loc) / (h_x · |F_W|),  0,  0)

    and similarly for the other faces. The volume integral splits into a
    product of 1D integrals, giving the entries below. See the source
    for the explicit derivation; coefficients are exact rationals.

    Returned shape: (n_local, n_local) where n_local = 2·dim.
    """
    dim = mesh.dim
    n_local = 2 * dim
    M = np.zeros((n_local, n_local), dtype=np.float64)
    dirs = local_face_directions(dim)
    is_low = local_face_is_low(dim)
    ax, ay, az = mesh.face_areas
    areas = (ax, ay, az)
    V = mesh.cell_volume

    # For two local faces a, b with normal directions d_a, d_b:
    #  - if d_a == d_b: opposite or same face along that axis. Contribution
    #    is (K⁻¹)[d,d] times a 1D mass-matrix entry of the linear "hat"
    #    pair, integrated over the perpendicular face area, normalised by
    #    |F_a| · |F_b|.
    #  - if d_a != d_b: cross-direction. Contribution is (K⁻¹)[d_a, d_b]
    #    times an integral that factors as (∫ hat_a) · (∫ hat_b) · (perp).

    h = (mesh.hx, mesh.hy, mesh.hz)
    h_active = (
        mesh.hx,
        mesh.hy if mesh.ny > 1 else 1.0,
        mesh.hz if mesh.nz > 1 else 1.0,
    )

    for a in range(n_local):
        for b in range(n_local):
            da, db = dirs[a], dirs[b]
            la, lb = is_low[a], is_low[b]
            if da == db:
                # 1D mass-matrix entry over the normal direction.
                # ∫_0^h ((h - t)/h)^2 dt = h/3      (low-low)
                # ∫_0^h ((h - t)/h)(t/h) dt = h/6   (low-high)
                # ∫_0^h (t/h)^2 dt = h/3            (high-high)
                hd_ = h_active[da]
                if la == lb:
                    coef = hd_ / 3.0
                else:
                    coef = hd_ / 6.0
                # Normalisation: ψ_F = (basis raw) / |F|, so two factors
                # of 1/|F|; integration over the perpendicular plane
                # contributes |F| (the cell volume V divided by the
                # active spacing hd_). Net: V * coef / (|F_a| · |F_b|).
                M[a, b] = K_inv_cell[da, db] * V * coef / (hd_ * areas[da] * areas[db])
            else:
                # ∫_0^{h_a} hat_a(t) dt = h_a / 2  (regardless of low/high)
                # cross term factors as
                #   (1/|F_a|) · (1/|F_b|) · (h_a/2) · (h_b/2) · (perp)
                # where the "perpendicular" thickness is whatever
                # direction is neither d_a nor d_b: V / (h_{d_a} · h_{d_b}).
                ha = h_active[da]
                hb = h_active[db]
                perp = V / (ha * hb)
                M[a, b] = (
                    K_inv_cell[da, db]
                    * (ha / 2.0)
                    * (hb / 2.0)
                    * perp
                    / (areas[da] * areas[db])
                )
    return M


# ----------------------------------------------------------------------
# Global assembly
# ----------------------------------------------------------------------


def cell_face_indices(mesh: Mesh, i: int, j: int, k: int) -> tuple[int, ...]:
    """Global face-DOF indices for the local faces of cell (i,j,k)."""
    dim = mesh.dim
    if dim == 1:
        return (mesh.x_face_index(i, 0, 0), mesh.x_face_index(i + 1, 0, 0))
    if dim == 2:
        return (
            mesh.x_face_index(i, j, 0),
            mesh.x_face_index(i + 1, j, 0),
            mesh.y_face_index(i, j, 0),
            mesh.y_face_index(i, j + 1, 0),
        )
    return (
        mesh.x_face_index(i, j, k),
        mesh.x_face_index(i + 1, j, k),
        mesh.y_face_index(i, j, k),
        mesh.y_face_index(i, j + 1, k),
        mesh.z_face_index(i, j, k),
        mesh.z_face_index(i, j, k + 1),
    )


def local_face_signs(dim: int) -> tuple[int, ...]:
    """Sign of B[K, F]: + for the high-side face, - for the low-side."""
    return tuple(-1 if low else +1 for low in local_face_is_low(dim))


def assemble_system(
    mesh: Mesh, K_inv: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Assemble (M, B) over the whole mesh."""
    n_local = 2 * mesh.dim
    n_cells = mesh.n_cells

    # Triplet accumulators sized exactly: M has n_local² triplets per
    # cell, B has n_local triplets per cell.
    m_rows = np.empty(n_cells * n_local * n_local, dtype=np.int64)
    m_cols = np.empty(n_cells * n_local * n_local, dtype=np.int64)
    m_vals = np.empty(n_cells * n_local * n_local, dtype=np.float64)
    b_rows = np.empty(n_cells * n_local, dtype=np.int64)
    b_cols = np.empty(n_cells * n_local, dtype=np.int64)
    b_vals = np.empty(n_cells * n_local, dtype=np.float64)

    signs = local_face_signs(mesh.dim)
    m_ptr = 0
    b_ptr = 0
    for k in range(mesh.nz):
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                c = mesh.cell_index(i, j, k)
                faces = cell_face_indices(mesh, i, j, k)
                M_local = cell_mass_matrix(mesh, K_inv[c])
                # M contributions
                for a in range(n_local):
                    fa = faces[a]
                    for b in range(n_local):
                        fb = faces[b]
                        m_rows[m_ptr] = fa
                        m_cols[m_ptr] = fb
                        m_vals[m_ptr] = M_local[a, b]
                        m_ptr += 1
                # B contributions
                for a in range(n_local):
                    b_rows[b_ptr] = faces[a]
                    b_cols[b_ptr] = c
                    b_vals[b_ptr] = signs[a]
                    b_ptr += 1

    M = sp.csr_matrix(
        (m_vals, (m_rows, m_cols)), shape=(mesh.n_faces, mesh.n_faces)
    )
    B = sp.csr_matrix(
        (b_vals, (b_rows, b_cols)), shape=(mesh.n_faces, mesh.n_cells)
    )
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
    """Assemble the saddle-point system [[M, B],[Bᵀ, 0]] with BC applied.

    Returns the global CSR matrix and the right-hand side. Dirichlet BCs
    enter as a flux RHS, Neumann BCs are enforced strongly (rows pinned
    on the affected flux DOFs).
    """
    n_f = mesh.n_faces
    n_c = mesh.n_cells
    n_total = n_f + n_c

    # Symmetric saddle-point form: [[M, -B], [-Bᵀ, 0]] (q; u) = (-f_D; 0)
    # — this matches the variational system
    #   ∫ K⁻¹ q · v - ∫ u div v = -∫_{Γ_D} u_D (v · n_out)
    #   ∫ w div q = 0
    # after negating the second equation to symmetrise.
    rhs = np.zeros(n_total, dtype=np.float64)
    for face_idx, u_d in bd.dirichlet.items():
        rhs[face_idx] -= _face_outward_sign(mesh, face_idx) * u_d

    Bt = B.T.tocsr()
    A = sp.bmat(
        [[M, -B], [-Bt, sp.csr_matrix((n_c, n_c))]],
        format="csr",
    )

    # Enforce Neumann (strong): pin q_F = g_N · |F| · outward_sign for
    # each Neumann face. For the standard q · n_out = g_N condition, we
    # have q_F (global-normal DOF) = (outward_sign) · g_N · |F|.
    A = A.tolil()
    pinned_values: dict[int, float] = {}
    for face_idx, g_n in bd.neumann.items():
        sign = _face_outward_sign(mesh, face_idx)
        if sign == 0:
            raise ValueError(
                f"Neumann face {face_idx} is not on the domain boundary"
            )
        target = sign * g_n * _face_area_of(mesh, face_idx)
        pinned_values[face_idx] = target

    # Move pinned columns to the RHS, then zero the rows and columns.
    for face_idx, target in pinned_values.items():
        col_dense = A[:, face_idx].toarray().ravel()
        # Skip the pinned row itself (we'll overwrite it below).
        col_dense[face_idx] = 0.0
        rhs -= col_dense * target

    for face_idx, target in pinned_values.items():
        A.rows[face_idx] = [face_idx]
        A.data[face_idx] = [1.0]
        rhs[face_idx] = target
        # Zero the corresponding column.
        for r in range(n_total):
            if r == face_idx:
                continue
            try:
                idx = A.rows[r].index(face_idx)
            except ValueError:
                continue
            A.rows[r].pop(idx)
            A.data[r].pop(idx)

    return A.tocsr(), rhs


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


def write_vti(
    filename: str,
    mesh: Mesh,
    pressure: np.ndarray,
    flux_cell: np.ndarray,
) -> None:
    """Hand-write an ASCII VTK ImageData file (.vti).

    Even in 1D / 2D we emit a 3-D ImageData with collapsed dimensions;
    ParaView handles that without trouble. ASCII format keeps the file
    human-readable for small examples.
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

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write(
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        )
        fh.write(
            f'  <ImageData WholeExtent="0 {ex_x} 0 {ex_y} 0 {ex_z}" '
            f'Origin="0 0 0" Spacing="{sx} {sy} {sz}">\n'
        )
        fh.write(
            f'    <Piece Extent="0 {ex_x} 0 {ex_y} 0 {ex_z}">\n'
        )
        fh.write('      <CellData Scalars="pressure" Vectors="flux">\n')
        fh.write(
            '        <DataArray type="Float64" Name="pressure" format="ascii">\n'
        )
        fh.write("          " + " ".join(f"{v:.10e}" for v in pressure) + "\n")
        fh.write("        </DataArray>\n")
        fh.write(
            '        <DataArray type="Float64" Name="flux" '
            'NumberOfComponents="3" format="ascii">\n'
        )
        fh.write(
            "          "
            + " ".join(
                f"{v:.10e}"
                for row in flux_cell
                for v in row
            )
            + "\n"
        )
        fh.write("        </DataArray>\n")
        fh.write("      </CellData>\n")
        fh.write("      <PointData></PointData>\n")
        fh.write("    </Piece>\n")
        fh.write("  </ImageData>\n")
        fh.write("</VTKFile>\n")


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
