#!/usr/bin/env python3
"""Generate SPE10 Darcy figures from C Darcy VTK output."""

from __future__ import annotations

import argparse
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SPE10 permeability and pressure figures from VTK output.",
    )
    parser.add_argument(
        "--result-file",
        required=True,
        type=Path,
        help="C Darcy .vti result, or .pvti master for a parallel run.",
    )
    parser.add_argument(
        "--mode",
        choices=("layer", "3d", "both"),
        default="layer",
        help="Figure mode to generate.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=35,
        help="Physical z-layer for the layer figure, 0-based.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("docs/usrman-src/figures/spe10_darcy_fields.png"),
        help="Output path for the layer figure.",
    )
    parser.add_argument(
        "--figure-3d-path",
        type=Path,
        default=Path("docs/usrman-src/figures/spe10_darcy_3d.png"),
        help="Output path for the 3D figure.",
    )
    return parser.parse_args()


def parse_extent(value: str) -> Tuple[int, int, int, int, int, int]:
    vals = tuple(int(v) for v in value.split())
    if len(vals) != 6:
        raise ValueError(f"invalid VTK extent: {value}")
    return vals


def cell_shape(extent: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int]:
    x0, x1, y0, y1, z0, z1 = extent
    return z1 - z0, y1 - y0, x1 - x0


def attrs_from_text(text: str) -> Dict[str, str]:
    return dict(re.findall(r'([A-Za-z0-9_]+)="([^"]*)"', text))


def vtk_dtype(vtk_type: str) -> np.dtype:
    if vtk_type == "Float64":
        return np.dtype("<f8")
    if vtk_type == "UInt32":
        return np.dtype("<u4")
    if vtk_type == "UInt64":
        return np.dtype("<u8")
    raise ValueError(f"unsupported VTK DataArray type: {vtk_type}")


def read_data_array(raw: bytes, appended_start: int, attrs: Dict[str, str]) -> np.ndarray:
    offset = int(attrs["offset"])
    pos = appended_start + offset
    nbytes = struct.unpack_from("<Q", raw, pos)[0]
    pos += 8
    payload = raw[pos : pos + nbytes]
    if len(payload) != nbytes:
        raise ValueError("truncated VTK appended data block")
    return np.frombuffer(payload, dtype=vtk_dtype(attrs["type"])).copy()


def read_vti_piece(path: Path) -> Tuple[Tuple[int, int, int, int, int, int], Dict[str, np.ndarray]]:
    raw = path.read_bytes()
    appended_tag = raw.find(b"<AppendedData")
    if appended_tag < 0:
        raise ValueError(f"{path} does not contain appended VTK data")
    appended_start = raw.find(b"_", appended_tag)
    if appended_start < 0:
        raise ValueError(f"{path} has no appended data marker")
    appended_start += 1

    header = raw[:appended_tag].decode("utf-8")
    piece_match = re.search(r"<Piece\s+Extent=\"([^\"]+)\"", header)
    if not piece_match:
        raise ValueError(f"{path} has no ImageData Piece extent")
    extent = parse_extent(piece_match.group(1))

    arrays: Dict[str, np.ndarray] = {}
    data_array_matches = re.findall(r"<DataArray\s+([^>]*)/>", header)
    for match in data_array_matches:
        attrs = attrs_from_text(match)
        name = attrs.get("Name")
        if name in {"pressure", "permeability", "GlobalCellId"}:
            arrays[name] = read_data_array(raw, appended_start, attrs)

    if "pressure" not in arrays or "permeability" not in arrays:
        raise ValueError(f"{path} is missing pressure or permeability arrays")
    return extent, arrays


def read_pvti(path: Path) -> Tuple[Tuple[int, int, int, int, int, int], List[Path]]:
    root = ET.parse(path).getroot()
    image = root.find("PImageData")
    if image is None:
        raise ValueError(f"{path} is not a PImageData file")
    whole = parse_extent(image.attrib["WholeExtent"])
    pieces = [path.parent / piece.attrib["Source"] for piece in image.findall("Piece")]
    if not pieces:
        raise ValueError(f"{path} does not reference any VTI pieces")
    return whole, pieces


def read_result(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".pvti":
        whole, pieces = read_pvti(path)
    else:
        whole, _arrays = read_vti_piece(path)
        pieces = [path]

    nz, ny, nx = cell_shape(whole)
    pressure = np.empty((nz, ny, nx), dtype=np.float64)
    kx = np.empty((nz, ny, nx), dtype=np.float64)

    for piece in pieces:
        extent, arrays = read_vti_piece(piece)
        z0, y0, x0 = extent[4], extent[2], extent[0]
        nzp, nyp, nxp = cell_shape(extent)
        n_cells = nzp * nyp * nxp
        piece_pressure = arrays["pressure"].reshape((nzp, nyp, nxp))
        piece_perm = arrays["permeability"].reshape((n_cells, 6))
        piece_kx = piece_perm[:, 0].reshape((nzp, nyp, nxp))
        pressure[z0 : z0 + nzp, y0 : y0 + nyp, x0 : x0 + nxp] = piece_pressure
        kx[z0 : z0 + nzp, y0 : y0 + nyp, x0 : x0 + nxp] = piece_kx

    return pressure, kx


def write_layer_figure(
    pressure: np.ndarray,
    kx: np.ndarray,
    physical_layer: int,
    figure_path: Path,
) -> None:
    nz, _ny, _nx = pressure.shape
    if physical_layer < 0 or physical_layer >= nz:
        raise SystemExit(f"--layer must be in [0, {nz - 1}]")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(
        np.log10(kx[physical_layer]), origin="lower", aspect="auto", cmap="viridis"
    )
    axes[0].set_title(f"SPE10 log10(Kx), layer {physical_layer}")
    axes[0].set_xlabel("Nx")
    axes[0].set_ylabel("Ny")
    fig.colorbar(im0, ax=axes[0], label="log10(mD)")

    im1 = axes[1].imshow(
        pressure[physical_layer], origin="lower", aspect="auto", cmap="magma"
    )
    axes[1].set_title("Pressure-drop solution on layer")
    axes[1].set_xlabel("Nx")
    axes[1].set_ylabel("Ny")
    fig.colorbar(im1, ax=axes[1], label="pressure")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    print(f"Wrote {figure_path}")


def add_surface_box(ax, vol: np.ndarray, cmap_name: str, title: str, cbar_label: str, fig) -> None:
    from matplotlib import cm, colors

    nz, ny, nx = vol.shape
    norm = colors.Normalize(float(np.nanmin(vol)), float(np.nanmax(vol)))
    cmap = cm.get_cmap(cmap_name)
    x_edges = np.arange(nx + 1)
    y_edges = np.arange(ny + 1)
    z_edges = np.arange(nz + 1)

    def colors_for(values: np.ndarray) -> np.ndarray:
        return cmap(norm(values))

    xgrid, ygrid = np.meshgrid(x_edges, y_edges)
    ax.plot_surface(
        xgrid,
        ygrid,
        np.zeros_like(xgrid),
        facecolors=colors_for(vol[0]),
        linewidth=0,
        shade=False,
        antialiased=False,
    )
    ax.plot_surface(
        xgrid,
        ygrid,
        np.full_like(xgrid, nz),
        facecolors=colors_for(vol[-1]),
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    xgrid, zgrid = np.meshgrid(x_edges, z_edges)
    ax.plot_surface(
        xgrid,
        np.zeros_like(xgrid),
        zgrid,
        facecolors=colors_for(vol[:, 0, :]),
        linewidth=0,
        shade=False,
        antialiased=False,
    )
    ax.plot_surface(
        xgrid,
        np.full_like(xgrid, ny),
        zgrid,
        facecolors=colors_for(vol[:, -1, :]),
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    ygrid, zgrid = np.meshgrid(y_edges, z_edges)
    ax.plot_surface(
        np.zeros_like(ygrid),
        ygrid,
        zgrid,
        facecolors=colors_for(vol[:, :, 0]),
        linewidth=0,
        shade=False,
        antialiased=False,
    )
    ax.plot_surface(
        np.full_like(ygrid, nx),
        ygrid,
        zgrid,
        facecolors=colors_for(vol[:, :, -1]),
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    ax.set_title(title)
    ax.set_xlabel("Nx", labelpad=4)
    ax.set_ylabel("Ny", labelpad=4)
    ax.set_zlabel("Nz", labelpad=4)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_xticks(np.linspace(0, nx, 5, dtype=int))
    ax.set_yticks(np.linspace(0, ny, 5, dtype=int))
    ax.set_zticks(np.linspace(0, nz, 5, dtype=int))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
    ax.zaxis.set_tick_params(labelsize=8, pad=0)
    ax.view_init(elev=24, azim=-54)
    ax.set_box_aspect((nx, ny, nz))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08, label=cbar_label)


def write_3d_figure(pressure: np.ndarray, kx: np.ndarray, figure_3d_path: Path) -> None:
    nz, ny, nx = pressure.shape
    fig = plt.figure(figsize=(12.4, 5.8), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    add_surface_box(
        ax0,
        np.log10(kx),
        "viridis",
        f"SPE10 log10(Kx), {nx} x {ny} x {nz}",
        "log10(mD)",
        fig,
    )
    add_surface_box(
        ax1,
        pressure,
        "magma",
        "3D pressure-drop solution",
        "pressure",
        fig,
    )
    figure_3d_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_3d_path, dpi=180)
    print(f"Wrote {figure_3d_path}")


def main() -> None:
    args = parse_args()
    pressure, kx = read_result(args.result_file)

    if args.mode in ("layer", "both"):
        write_layer_figure(pressure, kx, args.layer, args.figure_path)
    if args.mode in ("3d", "both"):
        write_3d_figure(pressure, kx, args.figure_3d_path)


if __name__ == "__main__":
    main()
