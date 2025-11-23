#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

import argparse
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle


def read_eigenvalues(path: str) -> Tuple[List[float], List[float]]:
    re_vals: List[float] = []
    im_vals: List[float] = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Eigenvalues file not found: {path}")

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                if len(parts) == 1:
                    re_vals.append(float(parts[0]))
                    im_vals.append(0.0)
                elif len(parts) >= 2:
                    re_vals.append(float(parts[0]))
                    im_vals.append(float(parts[1]))
            except ValueError:
                # Skip malformed lines
                continue

    return re_vals, im_vals


def compute_summary(re_vals: List[float], im_vals: List[float], zero_tol: float, imag_tol: float,
                    inset_halfwidth: float | None = None, one_tol: float | None = None) -> str:
    n = len(re_vals)
    if n == 0:
        return "No eigenvalues"
    # Core stats
    abs_vals = [math.hypot(re, im) for re, im in zip(re_vals, im_vals)]
    rho = max(abs_vals)
    min_abs = min(abs_vals)
    min_re = min(re_vals)
    max_re = max(re_vals)
    # Classifications
    is_complex = [abs(im) > imag_tol for im in im_vals]
    n_complex = sum(1 for c in is_complex if c)
    n_real = n - n_complex
    n_neg_re = sum(1 for re in re_vals if re < -zero_tol)
    n_pos_re = sum(1 for re in re_vals if re > zero_tol)
    n_zero_re = n - n_neg_re - n_pos_re
    # Spectral abscissa
    alpha = max_re
    # Near-zero magnitude counts
    n_near_zero_mag = sum(1 for a in abs_vals if a <= zero_tol)
    # Near-one magnitude counts (| |λ| - 1 | <= tol)
    n_near_one = None
    if one_tol is not None and one_tol > 0.0:
        n_near_one = sum(1 for a in abs_vals if abs(a - 1.0) <= one_tol)
    # Real-only condition estimate (min nonzero |λ|)
    cond_est = None
    if n_complex == 0:
        nonzero_abs = [abs(re) for re in re_vals if abs(re) > zero_tol]
        if nonzero_abs:
            cond_est = rho / min(nonzero_abs)
    lines = [
        f"rows = {n}",
        f"|λ|min = {min_abs:.6e}",
        f"|λ|max = {rho:.6e} (spectral radius)",
        f"Re[min,max] = [{min_re:.6g}, {max_re:.6g}]",
        f"Im[min,max] = [{min(im_vals):.6g}, {max(im_vals):.6g}]",
        f"LHP(Re<0) = {n_neg_re} ({n_neg_re/n*100:.1f}%)",
        f"RHP(Re>0) = {n_pos_re} ({n_pos_re/n*100:.1f}%)",
        f"Re≈0 = {n_zero_re} ({n_zero_re/n*100:.1f}%)",
        f"complex = {n_complex} ({n_complex/n*100:.1f}%)",
        f"real = {n_real} ({n_real/n*100:.1f}%)",
        f"|λ|≤{zero_tol:g} = {n_near_zero_mag} ({n_near_zero_mag/n*100:.1f}%)",
    ]
    if cond_est is not None:
        lines.append(f"cond_est≈ρ/min|λ|={cond_est:.6g}")
    # Count inside inset box if requested
    if inset_halfwidth is not None and inset_halfwidth > 0.0:
        n_inset = sum(1 for re, im in zip(re_vals, im_vals)
                      if abs(re) <= inset_halfwidth and abs(im) <= inset_halfwidth)
        lines.append(f"inset(|Re|≤{inset_halfwidth:g}, |Im|≤{inset_halfwidth:g}) = {n_inset} ({n_inset/n*100:.1f}%)")
    if n_near_one is not None:
        lines.append(f"|λ|≈1 (±{one_tol:g}) = {n_near_one} ({n_near_one/n*100:.1f}%)")
    return "\n".join(lines)


def plot_scatter_with_marginals(ax_scatter, ax_histx, ax_histy,
                                re_vals: List[float], im_vals: List[float],
                                re_bins: int, im_bins: int,
                                ms: float, alpha: float,
                                cmap: str, color: str, use_colormap: bool,
                                imag_tol: float):
    # Color by magnitude if multiple files or user wants colormap
    if use_colormap:
        mags = [math.hypot(re, im) for re, im in zip(re_vals, im_vals)]
        sc = ax_scatter.scatter(re_vals, im_vals, s=ms, c=mags, cmap=cmap, alpha=alpha, edgecolors='none')
    else:
        # By default, plot real eigenvalues in the base color and complex ones in red
        re_real = [re for re, im in zip(re_vals, im_vals) if abs(im) <= imag_tol]
        im_real = [im for im in im_vals if abs(im) <= imag_tol]
        re_cplx = [re for re, im in zip(re_vals, im_vals) if abs(im) > imag_tol]
        im_cplx = [im for im in im_vals if abs(im) > imag_tol]
        if re_real:
            ax_scatter.scatter(re_real, im_real, s=ms, alpha=alpha, color=color, edgecolors='none', zorder=2)
        if re_cplx:
            ax_scatter.scatter(re_cplx, im_cplx, s=ms, alpha=alpha, color='#d62728', edgecolors='none', zorder=3)
        sc = None  # no colorbar in non-colormap mode

    # Histograms
    if ax_histx is not None:
        ax_histx.hist(re_vals, bins=re_bins, color='#808080', alpha=0.6)
    if ax_histy is not None:
        ax_histy.hist(im_vals, bins=im_bins, color='#808080', alpha=0.6, orientation='horizontal')

    return sc


def make_figure(mode: str, figsize: Tuple[float, float], constrained: bool):
    if mode == 'scatter':
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=constrained)
        return fig, ax, None, None
    elif mode in ('scatter+hist', 'joint'):
        fig = plt.figure(figsize=figsize, constrained_layout=constrained)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05, figure=fig)
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_histy = fig.add_subplot(gs[1, 1])
        # Hide axis labels for marginals
        ax_histx.tick_params(axis='x', labelbottom=False)
        ax_histy.tick_params(axis='y', labelleft=False)
        return fig, ax_scatter, ax_histx, ax_histy
    elif mode == 'hist':
        fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=figsize, sharex=False, constrained_layout=constrained)
        return fig, (ax_re, ax_im), None, None
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description='Plot eigenspectrum from values file(s). Supports real (one-column) and complex (two-column) formats.')
    parser.add_argument('-f', '--files', nargs='+', required=True, help='Path(s) to <prefix>.values.txt files')
    parser.add_argument('--mode', choices=['scatter', 'scatter+hist', 'hist'], default='scatter+hist', help='Plot mode')
    parser.add_argument('--save', type=str, default=None, help='Output image filename (if omitted, just shows)')
    parser.add_argument('--title', type=str, default=None, help='Figure title')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    parser.add_argument('--figsize', type=float, nargs=2, default=(9.0, 7.0), help='Figure size (W H) inches')
    parser.add_argument('--alpha', type=float, default=0.8, help='Marker alpha for scatter')
    parser.add_argument('--ms', type=float, default=8.0, help='Marker size for scatter')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for coloring by |λ|')
    parser.add_argument('--color', type=str, default='#4C72B0', help='Fixed color when not using colormap')
    parser.add_argument('--use_colormap', action='store_true', help='Color by |λ| and enable colorbar with --colorbar')
    parser.add_argument('--equal_aspect', action='store_true', help='Use equal aspect ratio for scatter')
    parser.add_argument('--xscale', choices=['linear', 'log', 'symlog'], default='linear', help='X-axis scale for Re(λ)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help='x-axis limits (min max)')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='y-axis limits (min max)')
    parser.add_argument('--tfs', type=int, default=18, help='Title font size')
    parser.add_argument('--alfs', type=int, default=16, help='Axis label font size')
    parser.add_argument('--lgfs', type=int, default=14, help='Legend font size')
    parser.add_argument('--re_bins', type=int, default=50, help='Number of bins used for Re values')
    parser.add_argument('--im_bins', type=int, default=50, help='Number of bins used for Im values')
    parser.add_argument('--zero_tol', type=float, default=1.0e-12, help='Tolerance for classifying Re≈0 and |λ|≈0')
    parser.add_argument('--imag_tol', type=float, default=1.0e-12, help='Tolerance for classifying complex eigenvalues (|Im| > imag_tol)')
    parser.add_argument('--one_tol', type=float, default=0.05, help='Tolerance for classifying |λ| ≈ 1 (absolute window ±one_tol)')
    parser.add_argument('--inset', action='store_true', help='Add zoomed inset near the origin')
    parser.add_argument('--inset_halfwidth', type=float, default=0.1, help='Half-width around zero for inset view')
    parser.add_argument('--inset_size', type=float, nargs=2, default=(0.38, 0.38), help='Inset size as fraction of main axes (width height)')
    parser.add_argument('--inset_loc', type=str, default='upper right', help='Inset location (e.g., upper right)')
    parser.add_argument('--inset_nticks', type=int, default=3, help='Approximate number of major ticks on inset axes')
    parser.add_argument('--inset_labelfont', type=int, default=None, help='Inset axis label font size (defaults to alfs-2)')
    parser.add_argument('--inset_tickfont', type=int, default=None, help='Inset tick font size (defaults to alfs-4)')
    parser.add_argument('--show_radius', action='store_true', help='Overlay spectral-radius circle |λ|=ρ on the scatter')
    parser.add_argument('--radius_color', type=str, default='#d62728', help='Spectral radius circle color')
    parser.add_argument('--radius_alpha', type=float, default=0.8, help='Spectral radius circle alpha')
    parser.add_argument('--radius_lw', type=float, default=1.6, help='Spectral radius circle linewidth')

    args = parser.parse_args()

    # Global font sizes
    rcParams.update({
        'axes.titlesize': args.tfs,
        'axes.labelsize': args.alfs,
        'xtick.labelsize': args.alfs - 2,
        'ytick.labelsize': args.alfs - 2,
        'legend.fontsize': args.lgfs,
    })

    # Prepare figure
    # Always prefer constrained layout to avoid tight_layout warnings
    use_constrained = True
    fig, ax_main, ax_histx, ax_histy = make_figure(args.mode, tuple(args.figsize), constrained=use_constrained)
    if use_constrained:
        try:
            from matplotlib.figure import Figure
            # nudge paddings for clarity
            fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
        except Exception:
            pass

    # Plot per file (allow overlay)
    for idx, path in enumerate(args.files):
        re_vals, im_vals = read_eigenvalues(path)

        if args.mode == 'hist':
            ax_re, ax_im = ax_main
            ax_re.hist(re_vals, bins=100, color='#4C72B0', alpha=0.8)
            ax_re.set_xlabel('Re(λ)')
            ax_re.set_ylabel('Count')
            ax_im.hist(im_vals, bins=100, color='#55A868', alpha=0.8)
            ax_im.set_xlabel('Im(λ)')
            ax_im.set_ylabel('Count')
        else:
            sc = plot_scatter_with_marginals(ax_main, ax_histx, ax_histy,
                                             re_vals, im_vals, args.re_bins, args.im_bins,
                                             ms=args.ms, alpha=args.alpha,
                                             cmap=args.cmap, color=args.color, use_colormap=args.use_colormap,
                                             imag_tol=args.imag_tol)
            # Optional colorbar: place as an inset inside main axes to avoid overlapping hist axes
            if args.use_colormap and sc is not None:
                # Place colorbar below the x-label using figure coordinates; do not move hist axes
                fig.canvas.draw_idle()
                bbox = ax_main.get_position()
                cb_width = 0.6 * bbox.width
                cb_left  = bbox.x0 + 0.5 * (bbox.width - cb_width)
                cb_height = 0.025
                # Fixed margin from figure bottom to keep it clearly below xlabel
                cb_bottom = 0.02
                cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
                cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
                cb.set_label('|λ|', fontsize=args.alfs)
            # Apply requested x-axis scale to scatter and top histogram
            try:
                ax_main.set_xscale(args.xscale)
                if ax_histx is not None:
                    ax_histx.set_xscale(args.xscale)
            except Exception:
                pass

            # Match histogram axis ranges to scatter ranges
            xlim = ax_main.get_xlim()
            ylim = ax_main.get_ylim()
            if ax_histx is not None:
                ax_histx.set_xlim(xlim)
            if ax_histy is not None:
                ax_histy.set_ylim(ylim)
            # Spectral radius overlay
            if args.show_radius:
                # Compute spectral radius and ensure axis covers it
                rho = max((math.hypot(re, im) for re, im in zip(re_vals, im_vals)), default=0.0)
                if rho > 0.0:
                    circ = Circle((0.0, 0.0), radius=rho, fill=False,
                                  edgecolor=args.radius_color, alpha=args.radius_alpha,
                                  linewidth=args.radius_lw, label='|λ| = ρ')
                    ax_main.add_patch(circ)
                    # Ensure aspect is equal for a true circle and axes include the circle
                    ax_main.set_aspect('equal', adjustable='box')
                    xmin, xmax = ax_main.get_xlim()
                    ymin, ymax = ax_main.get_ylim()
                    pad = 0.05 * rho
                    xmin = min(xmin, -rho - pad)
                    xmax = max(xmax,  rho + pad)
                    ymin = min(ymin, -rho - pad)
                    ymax = max(ymax,  rho + pad)
                    ax_main.set_xlim((xmin, xmax))
                    ax_main.set_ylim((ymin, ymax))

        # Summary box on main axes and print to stdout
        summary = compute_summary(re_vals, im_vals,
                                  zero_tol=args.zero_tol,
                                  imag_tol=args.imag_tol,
                                  inset_halfwidth=(args.inset_halfwidth if args.inset else None),
                                  one_tol=args.one_tol)
        print(f"\nFile: {path}\n{summary}")

    # Labels and layout
    if args.mode != 'hist':
        ax_main.set_xlabel('Re(λ)')
        ax_main.set_ylabel('Im(λ)')
        ax_main.grid(True, linestyle='--', alpha=0.4)
        if args.equal_aspect:
            ax_main.set_aspect('equal', adjustable='box')
        if args.xlim:
            ax_main.set_xlim(args.xlim)
        if args.ylim:
            ax_main.set_ylim(args.ylim)
        if args.show_radius:
            ax_main.legend(loc='best', fontsize=args.lgfs, frameon=True, edgecolor='#888888')

        # Optional zoomed inset near origin
        if args.inset:
            w, h = args.inset_size
            inset = inset_axes(ax_main, width=f"{int(w*100)}%", height=f"{int(h*100)}%", loc=args.inset_loc)
            inset.scatter(re_vals, im_vals, s=max(2.0, args.ms*0.5), alpha=args.alpha,
                          color=(args.color if not args.use_colormap else '#4C72B0'), edgecolors='none')
            hw = args.inset_halfwidth
            inset.set_xlim([-hw, hw])
            inset.set_ylim([-hw, hw])
            inset.grid(True, linestyle='--', alpha=0.4)
            # Configure inset axes with their own ticks and labels
            inset.xaxis.set_major_locator(MaxNLocator(nbins=max(1, args.inset_nticks)))
            inset.yaxis.set_major_locator(MaxNLocator(nbins=max(1, args.inset_nticks)))
            inset_tickfs = args.inset_tickfont if args.inset_tickfont is not None else max(6, args.alfs - 4)
            inset_labelfs = args.inset_labelfont if args.inset_labelfont is not None else max(8, args.alfs - 2)
            inset.tick_params(axis='both', labelsize=inset_tickfs)
            #inset.set_xlabel('Re(λ)', fontsize=inset_labelfs)
            #inset.set_ylabel('Im(λ)', fontsize=inset_labelfs)
    else:
        # Apply requested x-axis scale to Re histogram
        try:
            ax_main[0].set_xscale(args.xscale)
        except Exception:
            pass
        ax_main[0].grid(True, linestyle='--', alpha=0.4)
        ax_main[1].grid(True, linestyle='--', alpha=0.4)

    if args.title:
        fig.suptitle(args.title, fontsize=12)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved figure to: {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
