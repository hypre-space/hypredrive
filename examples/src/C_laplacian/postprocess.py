#!/usr/bin/env python3
# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
"""Render a 3D visualization of the C_laplacian solution with PyVista.

Reads the VTK ``RectilinearGrid`` output written by ``laplacian -vis`` (a per-rank
``.vtr`` plus a ``.pvd`` collection in parallel) and produces a publication-quality
3D rendering of the scalar solution ``u``, in the same style as the Maxwell (AMS)
and grad-div (ADS) examples.

Styles (``--style``):

* ``iso``     (default) nested translucent isosurfaces (level sets) of the field.
* ``clip``    a cutaway: one octant is removed so the interior is exposed.
* ``volume``  smooth volume rendering with an opacity ramp (dark background).
* ``slices``  three orthogonal slice planes through the centre.

The color scale is logarithmic by default (``--no-log-scale`` for linear), which suits
the orders-of-magnitude decay of the diffusion field away from the Dirichlet face.

Requires PyVista (``pip install pyvista``), which reads the files via VTK and
renders with real depth compositing. The same ``.pvd``/``.vtr`` files can also be
opened directly in ParaView.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOLUTION_LABEL = "u"
DEFAULT_OUTPUT = "laplacian_solution_3d.png"
DEFAULT_TITLE = "Laplace solution"
LOG_DECADES = 3  # dynamic range shown on the log color scale (decades below the max)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="VTK input: a .pvd collection (parallel) or a single .vtr")
    p.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                   help=f"output image path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--style", choices=("iso", "clip", "volume", "slices"), default="iso",
                   help="visualization style (default: iso)")
    p.add_argument("--field", default="solution",
                   help="scalar array to color by (default: solution)")
    p.add_argument("--cmap", default="viridis", help="colormap (default: viridis)")
    p.add_argument("--clip-frac", type=float, default=0.5,
                   help="cutaway corner position as a fraction of each axis (default: 0.5)")
    p.add_argument("--n-contours", type=int, default=8,
                   help="number of isosurfaces for --style iso (default: 8)")
    p.add_argument("--log-scale", action=argparse.BooleanOptionalAction, default=True,
                   help="logarithmic color scale, ideal for the diffusion decay "
                        "(default: on; use --no-log-scale for a linear scale)")
    p.add_argument("--zoom", type=float, default=0.9, help="camera zoom (default: 0.9)")
    # Start from the isometric view and rotate by 90 deg (which keeps it isometric) to a
    # corner that faces the y=0 Dirichlet face, since the field is not symmetric.
    p.add_argument("--azimuth", type=float, default=-90.0, help="extra camera azimuth (deg)")
    p.add_argument("--elevation", type=float, default=0.0, help="extra camera elevation (deg)")
    p.add_argument("--window", type=int, nargs=2, metavar=("W", "H"), default=[1200, 900],
                   help="render window size in pixels (default: 1200 900)")
    p.add_argument("--background", default=None,
                   help="background color (default: white, or dark for --style volume)")
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

    field = args.field
    if field in mesh.point_data:
        point_mesh = mesh
    elif field in mesh.cell_data:
        point_mesh = mesh.cell_data_to_point_data()
    else:
        sys.exit(f"{args.input} has no array named '{field}'")
    arr = np.asarray(point_mesh.point_data[field])
    vmax = float(arr.max())
    if args.log_scale:
        # Clamp to a few decades below the peak (and stay strictly positive for log).
        floor = vmax * 10.0 ** (-LOG_DECADES)
        positive = arr[arr > 0.0]
        vmin = max(float(positive.min()) if positive.size else floor, floor)
        clim = [vmin, vmax]
    else:
        clim = [float(arr.min()), vmax]
    span = clim[1] - clim[0]
    label = args.label or SOLUTION_LABEL

    dark = args.style == "volume" if args.background is None else False
    background = args.background or ((0.11, 0.11, 0.15) if dark else "white")
    fg = "white" if dark else "black"

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

    # Scalar bar parked in the right margin (the camera zoom keeps the object clear of it).
    sbar = dict(title=label, color=fg, title_font_size=38,
                label_font_size=28 if args.log_scale else 30,
                n_labels=6 if args.log_scale else 5,
                fmt="%.0e" if args.log_scale else "%.2f",
                vertical=True, position_x=0.855, position_y=0.16, height=0.66, width=0.07)
    mesh_kw = dict(scalars=field, cmap=args.cmap, clim=clim, log_scale=args.log_scale,
                   scalar_bar_args=sbar)

    if args.style == "iso":
        if args.log_scale:
            levels = np.logspace(np.log10(clim[0]), np.log10(clim[1]), max(1, args.n_contours))
        else:
            levels = np.linspace(clim[0] + 0.10 * span, clim[1] - 0.02 * span, max(1, args.n_contours))
        iso = point_mesh.contour(levels.tolist(), scalars=field)
        pl.add_mesh(iso, opacity=0.45, smooth_shading=True, **mesh_kw)
        pl.add_mesh(mesh.outline(), color=fg, line_width=2)
    elif args.style == "clip":
        b = mesh.bounds
        f = args.clip_frac
        cx, cy, cz = b[0] + f * (b[1] - b[0]), b[2] + f * (b[3] - b[2]), b[4] + f * (b[5] - b[4])
        cut = mesh.clip_box((cx, b[1], cy, b[3], cz, b[5]), invert=True)
        pl.add_mesh(cut, **mesh_kw)
        pl.add_mesh(mesh.outline(), color=fg, line_width=2)
    elif args.style == "slices":
        pl.add_mesh(mesh.slice_orthogonal(), **mesh_kw)
        pl.add_mesh(mesh.outline(), color=fg, line_width=2)
    elif args.style == "volume":
        # The volume mapper needs ImageData, so resample the field onto a uniform grid.
        b = point_mesh.bounds
        d = 96
        grid = pv.ImageData(dimensions=(d, d, d), origin=(b[0], b[2], b[4]),
                            spacing=((b[1] - b[0]) / (d - 1), (b[3] - b[2]) / (d - 1),
                                     (b[5] - b[4]) / (d - 1)))
        grid = grid.sample(point_mesh)
        pl.add_volume(grid, scalars=field, cmap=args.cmap, clim=clim,
                      opacity=[0.0, 0.03, 0.10, 0.30, 0.75], scalar_bar_args=sbar)

    pl.camera_position = "iso"
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
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    render(parse_args())
