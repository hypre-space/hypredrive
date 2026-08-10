#!/usr/bin/env python3
# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
"""Render a 3D visualization of the C_convdif solution with PyVista.

Reads the VTK ``RectilinearGrid`` output written by ``convdif -vis`` (a per-rank
``.vtr`` plus a ``.pvd`` collection in parallel) and produces a publication-quality
3D rendering of the concentration field ``c``, in the same style as the Maxwell (AMS)
and grad-div (ADS) examples.

When the input is a time-series ``.pvd`` (written by ``convdif -vis 2``) and the
output ends in ``.gif``, an animation of the advancing front is produced instead of a
static image; otherwise the frame selected by ``--time-index`` is rendered.

Styles (``--style``):

* ``iso``     nested translucent isosurfaces (level sets) of the field.
* ``clip``    a cutaway: one octant is removed so the interior is exposed.
* ``volume``  smooth volume rendering with an opacity ramp (dark background).
* ``slices``  (default) three orthogonal slice planes through the centre, which show
  the advected front most clearly.

The color scale is linear by default, since the concentration varies between 0 at the
initial state and 1 at the inlet (``--log-scale`` switches to a logarithmic scale).

Requires PyVista (``pip install pyvista``), which reads the files via VTK and
renders with real depth compositing. The same ``.pvd``/``.vtr`` files can also be
opened directly in ParaView.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOLUTION_LABEL = "c"
DEFAULT_OUTPUT = "convdif_solution_3d.png"
DEFAULT_TITLE = "Convection-diffusion solution"
LOG_DECADES = 3  # dynamic range shown on the log color scale (decades below the max)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="VTK input: a .pvd collection (parallel) or a single .vtr")
    p.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                   help=f"output image path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--style", choices=("iso", "clip", "volume", "slices"), default="slices",
                   help="visualization style (default: slices)")
    p.add_argument("--field", default="concentration",
                   help="scalar array to color by (default: concentration)")
    p.add_argument("--cmap", default="viridis", help="colormap (default: viridis)")
    p.add_argument("--clip-frac", type=float, default=0.5,
                   help="cutaway corner position as a fraction of each axis (default: 0.5)")
    p.add_argument("--n-contours", type=int, default=8,
                   help="number of isosurfaces for --style iso (default: 8)")
    p.add_argument("--log-scale", action=argparse.BooleanOptionalAction, default=False,
                   help="logarithmic color scale (default: off, since the concentration "
                        "varies linearly between 0 and 1)")
    p.add_argument("--zoom", type=float, default=0.9, help="camera zoom (default: 0.9)")
    p.add_argument("--time-index", type=int, default=-1,
                   help="frame to render from a time-series input (default: -1, the last)")
    p.add_argument("--max-frames", type=int, default=25,
                   help="cap on rendered GIF frames; longer series are subsampled "
                        "(default: 25)")
    p.add_argument("--fps", type=int, default=10,
                   help="GIF frames per second (default: 10)")
    p.add_argument("--gif-max-width", type=int, default=1400,
                   help="downscale stored GIF frames to this width; the docs display "
                        "the animation smaller, so the default loses nothing visibly "
                        "(default: 1400; 0 disables)")
    p.add_argument("--clim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="fixed color range (default: data range; the animation always "
                        "uses 0 1)")
    p.add_argument("--colorbar", choices=("horizontal", "vertical"), default="horizontal",
                   help="scalar bar orientation; horizontal suits the duct's wide "
                        "aspect (default: horizontal)")
    p.add_argument("--image-scale", type=int, default=2,
                   help="supersampling factor for the render window (default: 2)")
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


def _scalar_bar_args(args: argparse.Namespace, label: str, fg: str,
                     log_scale: bool = False) -> dict:
    """Scalar-bar layout: horizontal under the duct (its wide aspect leaves the
    bottom margin free) or the classic vertical bar in the right margin."""
    sbar = dict(title=label, color=fg, title_font_size=44,
                label_font_size=36 if log_scale else 44,
                n_labels=6 if log_scale else 5,
                fmt="%.0e" if log_scale else "%.2f")
    if args.colorbar == "horizontal":
        # No title: for a horizontal bar VTK stacks it on the middle tick label.
        # The surrounding caption/prose names the quantity instead.
        sbar.update(title="", vertical=False, position_x=0.24, position_y=0.05,
                    width=0.40, height=0.12, n_labels=3)
    else:
        sbar.update(vertical=True, position_x=0.855, position_y=0.16,
                    height=0.66, width=0.07)
    return sbar


def render(args: argparse.Namespace) -> None:
    try:
        import pyvista as pv
    except ImportError:
        sys.exit("PyVista is required for this script: pip install pyvista\n"
                 "(Alternatively, open the .pvd/.vtr file directly in ParaView.)")
    import numpy as np

    pv.OFF_SCREEN = True
    reader = pv.get_reader(str(args.input))
    # A time-series .pvd exposes frames as time values; pick the requested one.
    times = list(getattr(reader, "time_values", []) or [])
    if times:
        reader.set_active_time_value(times[args.time_index])
    obj = reader.read()
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
    if args.clim is not None:
        clim = list(args.clim)
    span = clim[1] - clim[0]
    label = args.label or SOLUTION_LABEL

    dark = args.style == "volume" if args.background is None else False
    background = args.background or ((0.11, 0.11, 0.15) if dark else "white")
    fg = "white" if dark else "black"

    pl = pv.Plotter(off_screen=True, window_size=list(args.window),
                    image_scale=args.image_scale)
    pl.background_color = background
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        # SSAA anti-aliasing is an optional visual nicety that some VTK/OpenGL
        # backends (especially headless ones) reject; proceed without it.
        pass
    try:
        pl.enable_depth_peeling(10, 0.0)  # correct ordering for translucent surfaces
    except Exception:
        # Depth peeling is an optional rendering enhancement for translucency and is
        # not supported on every backend; continue rendering without it.
        pass

    sbar = _scalar_bar_args(args, label, fg, log_scale=args.log_scale)
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




def _optimize_gif(path: Path, max_width: int = 1400) -> None:
    """Shrink the GIF: crop the static white margins, downscale to ``max_width`` (the
    docs display the animation at no more than about half that, so this is visually
    lossless), share one palette across all frames, and store only the pixels that
    change from frame to frame. With a fixed camera the background/outline/colorbar are
    identical every frame, so this is a large, quality-preserving reduction.
    Best-effort: silently skipped if Pillow is unavailable."""
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
    # Downscale with a high-quality filter before quantization.
    w = x1 - x0
    if max_width > 0 and w > max_width:
        h = int(round((y1 - y0) * max_width / w))
        crop = [np.asarray(Image.fromarray(a).resize((max_width, h), Image.LANCZOS))
                for a in crop]
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


def animate(args: argparse.Namespace) -> None:
    """Render the time series written by ``convdif -vis 2`` as an animated GIF."""
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
    times = list(getattr(reader, "time_values", []) or [])
    if not times:
        sys.exit(f"{args.input} is not a time series; run 'convdif -vis 2' to write one.")

    def read_at(t):
        reader.set_active_time_value(t)
        obj = reader.read()
        mesh = obj.combine() if isinstance(obj, pv.MultiBlock) else obj
        if field in mesh.point_data:
            return mesh
        if field in mesh.cell_data:
            return mesh.cell_data_to_point_data()
        raise SystemExit(f"{args.input} has no array named '{field}'")

    # Subsample to keep the GIF light while preserving the first and last frames.
    if len(times) > args.max_frames > 0:
        idx = np.unique(np.linspace(0, len(times) - 1, args.max_frames).round().astype(int))
        times = [times[i] for i in idx]

    # The concentration is bounded by the inlet value, so the color range is fixed.
    clim = [0.0, 1.0]

    pl = pv.Plotter(off_screen=True, window_size=list(args.window),
                    image_scale=args.image_scale)
    pl.background_color = "white"
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        # Optional rendering nicety; unavailable on some headless backends.
        pass

    sbar = _scalar_bar_args(args, label, fg)
    title = DEFAULT_TITLE if args.title is None else args.title

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.open_gif(str(args.output), fps=args.fps)

    # Fix the camera once so only the field animates across frames.
    camera_set = False
    n_written = 0
    for t in times:
        mesh = read_at(t)
        pl.clear()
        pl.add_mesh(mesh.slice_orthogonal(), scalars=field, cmap=args.cmap, clim=clim,
                    log_scale=False, scalar_bar_args=sbar)
        pl.add_mesh(mesh.outline(), color=fg, line_width=2)
        pl.add_axes(color=fg, line_width=4)
        pl.add_text(f"t = {t:.2f}", position=(0.24, 0.84), viewport=True, color=fg,
                    font_size=66)
        if title:
            pl.add_text(title, position="upper_edge", color=fg, font_size=20)

        if not camera_set:
            pl.camera_position = "iso"
            pl.camera.azimuth += args.azimuth
            pl.camera.elevation += args.elevation
            pl.camera.zoom(args.zoom)
            camera_set = True

        pl.write_frame()
        n_written += 1

    pl.close()
    _optimize_gif(args.output, max_width=args.gif_max_width)
    try:
        mb = args.output.stat().st_size / 1e6
        print(f"Wrote {args.output} ({n_written} frames at {args.fps} fps, {mb:.2f} MB)")
    except OSError:
        print(f"Wrote {args.output} ({n_written} frames at {args.fps} fps)")


if __name__ == "__main__":
    _args = parse_args()
    if _args.output.suffix.lower() == ".gif":
        animate(_args)
    else:
        render(_args)
