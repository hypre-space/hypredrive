#!/usr/bin/env python3
# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
"""Render a 3D visualization of the C_elasticity solution with PyVista.

Reads the VTK ``RectilinearGrid`` output written by ``elasticity -vis`` (a per-rank
``.vtr`` plus a ``.pvd`` collection in parallel) and produces a publication-quality
3D rendering of the displacement field, in the same style as the Maxwell (AMS),
grad-div (ADS), and Laplace examples.

The deformed (warped) configuration is drawn as a solid surface (with the FEM mesh
edges) colored by the displacement magnitude :math:`\\|\\mathbf{u}\\|_2`, shown together
with the original (undeformed) configuration as a light box outline. Downward load
arrows on the free-end top surface (opposite the clamped end) indicate the applied
traction/gravity. The camera reproduces the original figure's y-up 3/4 viewpoint.

The warp scale (``--warp-factor``) defaults to an automatic value chosen so that the
largest displacement is a modest fraction of the smallest domain dimension; pass an
explicit factor to override it (``--warp-factor 1`` shows the true deformation).

Requires PyVista (``pip install pyvista``), which reads the files via VTK and
renders with real depth compositing. The same ``.pvd``/``.vtr`` files can also be
opened directly in ParaView (use the "Warp By Vector" filter for the same effect).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FIELD = "displacement"
MAGNITUDE = "displacement_magnitude"
SOLUTION_LABEL = "|u|"
DEFAULT_OUTPUT = "elasticity_solution_3d.png"
DEFAULT_TITLE = "Elasticity displacement"
# Auto warp scale: make max|u| this fraction of the smallest domain dimension.
AUTO_WARP_FRACTION = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="VTK input: a .pvd collection (parallel) or a single .vtr")
    p.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                   help=f"output image path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--cmap", default="viridis", help="colormap (default: viridis)")
    p.add_argument("--warp-factor", type=float, default=0.0,
                   help="displacement scale for the deformed shape "
                        "(default: 0 = auto-pick a modest, clearly visible scale)")
    p.add_argument("--zoom", type=float, default=1.15, help="camera zoom (default: 1.15)")
    # A y-up, front-right-above 3/4 view matching the original figure: the clamped (x=0)
    # end recedes to the upper-left and the loaded free end is large on the right. The
    # azimuth/elevation below are extra rotations applied on top of that base view.
    p.add_argument("--azimuth", type=float, default=0.0, help="extra camera azimuth (deg)")
    p.add_argument("--elevation", type=float, default=0.0, help="extra camera elevation (deg)")
    p.add_argument("--load-arrows", action=argparse.BooleanOptionalAction, default=True,
                   help="draw downward load arrows on the free-end top surface (default: on)")
    p.add_argument("--window", type=int, nargs=2, metavar=("W", "H"), default=[1200, 800],
                   help="render window size in pixels (default: 1200 800)")
    p.add_argument("--background", default=None,
                   help="background color (default: white)")
    p.add_argument("--label", default=None, help=f"scalar-bar label (default: {SOLUTION_LABEL})")
    p.add_argument("--title", default=None,
                   help=f"image title (default: '{DEFAULT_TITLE}'; pass '' to omit)")
    return p.parse_args()


def render(args: argparse.Namespace) -> None:
    try:
        import pyvista as pv
    except ImportError:
        sys.exit("PyVista is required for this script: pip install pyvista\n"
                 "(Alternatively, open the .pvd/.vtr file directly in ParaView.)")
    import numpy as np

    pv.OFF_SCREEN = True
    obj = pv.read(str(args.input))
    # A .pvd collection (one block per rank) reads as a MultiBlock; merge to one grid.
    mesh = obj.combine() if isinstance(obj, pv.MultiBlock) else obj

    if FIELD not in mesh.point_data:
        sys.exit(f"{args.input} has no point vector named '{FIELD}'")
    disp = np.asarray(mesh.point_data[FIELD])
    if disp.ndim != 2 or disp.shape[1] != 3:
        sys.exit(f"'{FIELD}' is not a 3-component vector (shape {disp.shape})")
    mag = np.linalg.norm(disp, axis=1)
    mesh[MAGNITUDE] = mag
    clim = [float(mag.min()), float(mag.max())]
    label = args.label or SOLUTION_LABEL

    # Pick a warp factor: explicit if given, else a modest auto scale so the deflection
    # reads clearly without grossly distorting the beam.
    b = mesh.bounds
    dims = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]
    smallest = min(d for d in dims if d > 0.0)
    max_mag = clim[1]
    if args.warp_factor and args.warp_factor > 0.0:
        factor = args.warp_factor
    elif max_mag > 0.0:
        factor = AUTO_WARP_FRACTION * smallest / max_mag
    else:
        factor = 1.0

    # Deformed configuration: warp the grid by the displacement vector, then take the
    # exterior surface so it renders as a clean solid (not internal cells).
    warped = mesh.warp_by_vector(FIELD, factor=factor).extract_surface(algorithm=None)

    background = args.background or "white"
    fg = "black"

    pl = pv.Plotter(off_screen=True, window_size=list(args.window))
    pl.background_color = background
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    try:
        pl.enable_depth_peeling(10, 0.0)  # correct ordering for translucent surfaces
    except Exception:
        pass

    # Original (undeformed) configuration as a light box outline for reference.
    pl.add_mesh(mesh.outline(), color="gray", line_width=2)

    # Scalar bar parked in the right margin (the camera zoom keeps the object clear of it).
    sbar = dict(title=label, color=fg, title_font_size=36, label_font_size=28, n_labels=5,
                fmt="%.2f", vertical=True, position_x=0.9, position_y=0.18, height=0.62,
                width=0.05)
    # Deformed configuration: solid colored by |u| with the FEM mesh edges shown.
    pl.add_mesh(warped, scalars=MAGNITUDE, cmap=args.cmap, clim=clim, show_edges=True,
                edge_color="dimgray", line_width=1, scalar_bar_args=sbar)

    # Downward load arrows on the free-end top surface (the side opposite the x=0 clamp),
    # representing the applied traction / gravity that bends the cantilever.
    if args.load_arrows:
        Lx, Ly, Lz = dims
        alen = 0.45 * Ly
        xs = np.linspace(b[1] - 0.35 * Lx, b[1] - 0.05 * Lx, 4)
        zs = [b[4] + 0.3 * Lz, b[5] - 0.3 * Lz]
        pts = np.array([[x, b[3] + alen, z] for x in xs for z in zs])
        dirs = np.tile([0.0, -1.0, 0.0], (len(pts), 1))
        pl.add_arrows(pts, dirs, mag=alen, color="red")

    # Camera: reproduce the original figure's viewpoint -- a y-up, front-right-above 3/4
    # view with the clamped (x=0) end receding to the upper-left.
    center = [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
    diag = float(np.linalg.norm(dims))
    vdir = np.array([1.4, 0.5, 1.25])
    vdir /= np.linalg.norm(vdir)
    pos = np.array(center) + vdir * diag * 2.0
    pl.camera_position = [tuple(pos), tuple(center), (0.0, 1.0, 0.0)]
    pl.camera.azimuth += args.azimuth
    pl.camera.elevation += args.elevation
    pl.camera.zoom(args.zoom)
    pl.add_axes(color=fg, line_width=4)

    title = DEFAULT_TITLE if args.title is None else args.title
    if title:
        pl.add_text(title, position="upper_edge", color=fg, font_size=24)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(args.output))
    pl.close()
    print(f"Wrote {args.output}  (warp factor {factor:.3g}, max|u| {max_mag:.4g})")


if __name__ == "__main__":
    render(parse_args())
