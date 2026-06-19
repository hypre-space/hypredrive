#!/usr/bin/env python3
# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
"""Render an animated visualization of the C_heatflow transient with PyVista.

Reads the time-series VTK ``RectilinearGrid`` output written by
``heatflow -vis 4`` (one ``.vtr`` per rank and timestep, plus a ``.pvd``
collection) and produces a light animated GIF of the temperature
isosurfaces (level sets) evolving as the body cools toward its cold base,
in the same style as the Maxwell (AMS), grad-div (ADS) and Laplace examples.

The isosurface levels and color scale are computed once over the whole
transient and then held fixed, so the nested level sets visibly shrink and
fade as the temperature decays. The camera is also fixed, so only the field
animates.

A single ``.vtr`` (or a ``.pvd`` with one timestep) is also accepted and
renders as a single still frame.

Requires PyVista (``pip install pyvista``), which reads the files via VTK and
renders with real depth compositing. The same ``.pvd``/``.vtr`` files can also
be opened directly in ParaView.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOLUTION_LABEL = "T"
DEFAULT_OUTPUT = "heatflow_transient.gif"
DEFAULT_TITLE = "Heat conduction"
MAX_FRAMES = 25  # subsample longer series to keep the GIF light


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="VTK input: a .pvd time-series collection or a single .vtr")
    p.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                   help=f"output GIF path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--field", default="temperature",
                   help="scalar array to color by (default: temperature)")
    p.add_argument("--cmap", default="viridis", help="colormap (default: viridis)")
    p.add_argument("--n-contours", type=int, default=7,
                   help="number of isosurfaces (default: 7)")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help=f"cap on rendered frames; longer series are subsampled "
                        f"(default: {MAX_FRAMES})")
    p.add_argument("--fps", type=int, default=8, help="GIF frames per second (default: 8)")
    p.add_argument("--zoom", type=float, default=0.9, help="camera zoom (default: 0.9)")
    p.add_argument("--window", type=int, nargs=2, metavar=("W", "H"), default=[700, 600],
                   help="render window size in pixels (default: 700 600)")
    p.add_argument("--label", default=None, help=f"scalar-bar label (default: {SOLUTION_LABEL})")
    p.add_argument("--title", default=None,
                   help=f"animation title (default: '{DEFAULT_TITLE}'; pass '' to omit)")
    return p.parse_args()


def _optimize_gif(path: Path) -> None:
    """Shrink the GIF losslessly: crop the static white margins, share one palette across
    all frames, and store only the pixels that change from frame to frame. With a fixed
    camera the background/outline/colorbar are identical every frame, so this is a large,
    quality-preserving reduction. Best-effort: silently skipped if Pillow is unavailable."""
    try:
        from PIL import Image, ImageSequence
        import numpy as np
    except Exception:
        return
    im = Image.open(str(path))
    dur = im.info.get("duration", 125)
    loop = im.info.get("loop", 0)
    frames = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    if len(frames) < 2:
        return
    # Crop the uniform white border (the same for every frame under a fixed camera).
    nonwhite = np.zeros(frames[0].shape[:2], bool)
    for a in frames:
        nonwhite |= a.min(axis=2) < 248
    ys, xs = np.where(nonwhite)
    if ys.size == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = [a[y0:y1, x0:x1] for a in frames]
    # One shared adaptive palette (255 colors; reserve index 255 for transparency).
    stack = Image.fromarray(np.concatenate(crop, axis=0))
    pal = stack.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE)
    idx = [np.asarray(Image.fromarray(c).quantize(palette=pal, dither=Image.NONE)) for c in crop]
    trans = 255
    first = Image.fromarray(idx[0], mode="P")
    first.putpalette(pal.getpalette())
    out = [first]
    prev = idx[0]
    for cur in idx[1:]:
        diff = cur.copy()
        diff[cur == prev] = trans  # unchanged pixels -> transparent (kept from prior frame)
        f = Image.fromarray(diff, mode="P")
        f.putpalette(pal.getpalette())
        out.append(f)
        prev = cur
    out[0].save(str(path), save_all=True, append_images=out[1:], duration=dur, loop=loop,
                disposal=1, transparency=trans, optimize=True)


def render(args: argparse.Namespace) -> None:
    try:
        import pyvista as pv
    except ImportError:
        sys.exit("PyVista is required for this script: pip install pyvista\n"
                 "(Alternatively, open the .pvd/.vtr file directly in ParaView.)")
    import numpy as np

    pv.OFF_SCREEN = True
    field = args.field
    label = args.label or SOLUTION_LABEL
    fg = "black"

    reader = pv.get_reader(str(args.input))
    # A .pvd time-series exposes its frames as time values; a bare .vtr has none.
    times = list(getattr(reader, "time_values", []) or [])

    def read_at(t):
        """Read one timestep and merge any per-rank blocks into a point mesh."""
        if t is not None:
            reader.set_active_time_value(t)
        obj = reader.read()
        mesh = obj.combine() if isinstance(obj, pv.MultiBlock) else obj
        if field in mesh.point_data:
            return mesh
        if field in mesh.cell_data:
            return mesh.cell_data_to_point_data()
        raise SystemExit(f"{args.input} has no array named '{field}'")

    if not times:
        times = [None]  # single static frame (a lone .vtr or single-step .pvd)

    # Subsample to keep the GIF light while preserving the first and last frames.
    if len(times) > args.max_frames > 0:
        idx = np.unique(np.linspace(0, len(times) - 1, args.max_frames).round().astype(int))
        times = [times[i] for i in idx]

    # One pass over the whole transient to fix a global color range and a single
    # set of isosurface levels, so the decay is visible against a constant scale.
    gmin, gmax = np.inf, -np.inf
    for t in times:
        arr = np.asarray(read_at(t).point_data[field])
        gmin = min(gmin, float(arr.min()))
        gmax = max(gmax, float(arr.max()))
    clim = [max(0.0, gmin), gmax]
    # Fixed iso levels spanning ~[0.1, 0.95] of the global peak.
    levels = np.linspace(clim[1] * 0.10, clim[1] * 0.95, max(1, args.n_contours)).tolist()

    pl = pv.Plotter(off_screen=True, window_size=list(args.window))
    pl.background_color = "white"
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        # SSAA is an optional rendering enhancement; depending on the VTK/OpenGL
        # backend (e.g. headless) it may be unavailable, so render without it.
        pass
    try:
        pl.enable_depth_peeling(10, 0.0)  # correct ordering for translucent surfaces
    except Exception:
        # Ordered transparency via depth peeling is optional and not supported on
        # every backend; proceed with the default compositing if it is unavailable.
        pass

    # Scalar bar parked in the right margin (the camera zoom keeps the object clear of it).
    sbar = dict(title=label, color=fg, title_font_size=38, label_font_size=30, n_labels=5,
                fmt="%.2f", vertical=True, position_x=0.855, position_y=0.16, height=0.66,
                width=0.07)

    title = DEFAULT_TITLE if args.title is None else args.title

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.open_gif(str(args.output), fps=args.fps)

    # Fix the camera once so only the field animates across frames.
    camera_set = False
    n_written = 0
    for t in times:
        mesh = read_at(t)
        iso = mesh.contour(levels, scalars=field)

        pl.clear()
        # Add the colored isosurfaces (skip on a degenerate frame whose field never
        # crosses the lowest level, which would otherwise drop the scalar bar).
        if iso.n_points:
            pl.add_mesh(iso, scalars=field, cmap=args.cmap, clim=clim, opacity=0.45,
                        smooth_shading=True, scalar_bar_args=sbar)
        pl.add_mesh(mesh.outline(), color=fg, line_width=2)
        pl.add_axes(color=fg, line_width=4)
        if t is not None:
            pl.add_text(f"t = {t:.3f}", position="lower_left", color=fg, font_size=16)
        if title:
            pl.add_text(title, position="upper_edge", color=fg, font_size=20)

        if not camera_set:
            pl.camera_position = "iso"
            pl.camera.zoom(args.zoom)
            camera_set = True

        pl.write_frame()
        n_written += 1

    pl.close()
    _optimize_gif(args.output)
    try:
        mb = args.output.stat().st_size / 1e6
        print(f"Wrote {args.output} ({n_written} frames at {args.fps} fps, {mb:.2f} MB)")
    except OSError:
        print(f"Wrote {args.output} ({n_written} frames at {args.fps} fps)")


if __name__ == "__main__":
    render(parse_args())
