#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

import argparse
import glob
import os
import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


def read_hypre_binary_matrix(parts: List[str], threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    vals_all: List[np.ndarray] = []
    nrows_glob = -1
    ncols_glob = -1
    tot_before = 0
    tot_after = 0

    for p in sorted(parts):
        logging.debug(f"Reading part file: {p}")
        with open(p, 'rb') as f:
            header = np.fromfile(f, count=11, dtype=np.uint64)
            if header.size != 11:
                raise RuntimeError(f"Invalid header in {p}")

            int_bytes = int(header[1])
            real_bytes = int(header[2])
            g_nrows = int(header[3])
            g_ncols = int(header[4])
            loc_nnz = int(header[6])

            if nrows_glob < 0:
                nrows_glob = g_nrows
                ncols_glob = g_ncols
            else:
                if g_nrows != nrows_glob or g_ncols != ncols_glob:
                    raise RuntimeError("Mismatched global shape across parts")

            if loc_nnz == 0:
                continue

            if int_bytes == 4:
                idtype = np.uint32
            elif int_bytes == 8:
                idtype = np.uint64
            else:
                raise RuntimeError(f"Unsupported integer byte width: {int_bytes}")

            if real_bytes == 4:
                fdtype = np.float32
            elif real_bytes == 8:
                fdtype = np.float64
            else:
                raise RuntimeError(f"Unsupported real byte width: {real_bytes}")

            r = np.fromfile(f, count=loc_nnz, dtype=idtype).astype(np.int64, copy=False)
            c = np.fromfile(f, count=loc_nnz, dtype=idtype).astype(np.int64, copy=False)
            v = np.fromfile(f, count=loc_nnz, dtype=fdtype).astype(np.float64, copy=False)
            logging.debug(f"  local nnz before threshold: {loc_nnz}")

            if threshold > 0.0:
                mask = np.abs(v) > threshold
                kept = int(mask.sum())
                logging.debug(f"  applying threshold {threshold}: keeping {kept} / {loc_nnz} (100.0 * {kept} / {loc_nnz:.0f}% = {100.0 * kept / loc_nnz:.1f}%)")
                r = r[mask]
                c = c[mask]
                v = v[mask]
                tot_before += loc_nnz
                tot_after += kept
            else:
                tot_before += loc_nnz
                tot_after += loc_nnz

            rows_all.append(r)
            cols_all.append(c)
            vals_all.append(v)

    if nrows_glob < 0:
        raise RuntimeError("No part files found or empty matrix")

    if rows_all:
        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        vals = np.concatenate(vals_all)
    else:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0, dtype=np.float64)

    if threshold > 0.0:
        logging.info(f"Threshold {threshold}: kept {tot_after}/{tot_before} ({100.0 * tot_after / tot_before:.1f}%) entries across all parts")

    return rows, cols, vals, (nrows_glob, ncols_glob)


def main():
    parser = argparse.ArgumentParser(description='Interactive Plotly spy plot for sequence of Hypre IJ binary matrices (per-process files).')
    parser.add_argument('-f', '--filename', nargs='+', default=None, help='Explicit path(s) to binary part file(s). If multiple, treated as parts of a single system')
    parser.add_argument('-d', '--directory', type=str, default='.', help='Root directory containing ls_00000, ls_00001, ... subdirectories')
    parser.add_argument('-p', '--pattern', type=str, default='ls_*', help='Subdirectory pattern to discover matrices')
    parser.add_argument('-P', '--prefix', type=str, default='IJ.out.A', help='Matrix file prefix inside each subdirectory')
    parser.add_argument('-t', '--threshold', type=float, default=0.0, help='Drop entries with |A_ij| <= threshold')
    parser.add_argument('-l', '--log', action='store_true', help='Use log(|A_ij|) as color')
    parser.add_argument('-r', '--range', type=str, default=None, help='Range as START:END (inclusive). Open ended like :END or START: are allowed')
    parser.add_argument('-S', '--start', type=int, default=None, help='First linear system index (alias)')
    parser.add_argument('-E', '--end', type=int, default=None, help='Last linear system index inclusive (alias)')
    parser.add_argument('-W', '--width', type=int, default=1000, help='Figure width')
    parser.add_argument('-H', '--height', type=int, default=900, help='Figure height')
    parser.add_argument('--tfs', type=int, default=22, help='Title font size')
    parser.add_argument('--alfs', type=int, default=18, help='Axis label font size')
    parser.add_argument('--tickfs', type=int, default=14, help='Axis tick font size')
    parser.add_argument('--cbfs', type=int, default=16, help='Colorbar title font size')
    parser.add_argument('--ms', type=float, default=2.5, help='Marker size (pixels)')
    parser.add_argument('--colorscale', type=str, default='Jet', help='Plotly colorscale name (default: Jet)')
    parser.add_argument('--no_grid', action='store_true', help='Disable grid lines')
    parser.add_argument('-s', '--save', type=str, default=None, help='Save HTML to this path instead of opening a browser')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v for INFO, -vv for DEBUG)')

    args = parser.parse_args()

    # Configure logging
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')

    files_mode = args.filename is not None and len(args.filename) > 0
    labels: List[str] = []

    if files_mode:
        logging.info("Files mode: using provided filename(s) as part files")
        parts = args.filename
        # Load single system from provided parts
        logging.debug(f"  using {len(parts)} file(s): {parts}")
        r, c, v, shape = read_hypre_binary_matrix(parts, threshold=args.threshold)
        logging.info(f"Loaded matrix from files: shape={shape} nnz={r.size}")
        if args.log and v.size:
            av = np.abs(v)
            pos = av[av > 0]
            vmin = float(pos.min()) if pos.size else 1.0
            cv = np.log10(np.maximum(vmin, av))
        else:
            cv = v
        mats = [(r, c, cv, shape, v)]
        ls_ids = [0]
        labels = [os.path.basename(parts[0])]
        cmins = [float(cv.min())] if cv.size else []
        cmaxs = [float(cv.max())] if cv.size else []
        cmin = min(cmins) if cmins else 0.0
        cmax = max(cmaxs) if cmaxs else 1.0
        logging.info(f"Global color scale: cmin={cmin:.3g} cmax={cmax:.3g}")
    else:
        logging.info(f"Scanning directory={args.directory!r} pattern={args.pattern!r}")
        subdirs = sorted([d for d in glob.glob(os.path.join(args.directory, args.pattern)) if os.path.isdir(d)])
        if not subdirs:
            raise SystemExit(f"No subdirectories found under {args.directory!r} matching {args.pattern!r}")
        logging.info(f"Found {len(subdirs)} subdirectories")

    import re
    def idx_of(path: str) -> int | None:
        m = re.search(r"(\d+)$", os.path.basename(path))
        return int(m.group(1)) if m else None

    if not files_mode:
        s = e = None
        if args.range:
            a, b = (args.range.strip().split(':', 1) + [''])[:2]
            s = int(a) if a else None
            e = int(b) if b else None
        if s is None and args.start is not None:
            s = args.start
        if e is None and args.end is not None:
            e = args.end

        pairs = [(idx_of(d), d) for d in subdirs]
        if s is not None or e is not None:
            if s is not None and e is not None and s > e:
                s, e = e, s
            logging.info(f"Filtering range: start={s} end={e}")
            def in_range(i: int) -> bool:
                return (s is None or i >= s) and (e is None or i <= e)
            pairs = [(i, d) for (i, d) in pairs if (i is not None and in_range(i))]
            if not pairs:
                raise SystemExit("No subdirectories matched the requested range.")
        logging.info(f"Selected {len(pairs)} systems")
        logging.debug("Selected directories: %s", [os.path.basename(d) for (_, d) in pairs])

    if not files_mode:
        ls_ids = [i if i is not None else k for k, (i, _) in enumerate(pairs)]
        subdirs = [d for (_, d) in pairs]
        labels = [f"{os.path.basename(d)}/{args.prefix}" for d in subdirs]

        # Load all selected matrices (preload for smooth slider)
        mats = []
        cmins = []
        cmaxs = []
        for d in subdirs:
            logging.info(f"Loading matrix from {d}")
            parts = sorted(glob.glob(os.path.join(d, f"{args.prefix}.*.bin")))
            if not parts:
                raise SystemExit(f"No part files found in {d} for prefix {args.prefix}")
            logging.debug(f"  found {len(parts)} part files")
            r, c, v, shape = read_hypre_binary_matrix(parts, threshold=args.threshold)
            logging.info(f"  shape={shape} nnz={r.size}")
            if args.log and v.size:
                av = np.abs(v)
                pos = av[av > 0]
                vmin = float(pos.min()) if pos.size else 1.0
                cv = np.log10(np.maximum(vmin, av))
            else:
                cv = v
            mats.append((r, c, cv, shape, v))
            if cv.size:
                cmins.append(float(cv.min()))
                cmaxs.append(float(cv.max()))
        cmin = min(cmins) if cmins else 0.0
        cmax = max(cmaxs) if cmaxs else 1.0
        logging.info(f"Global color scale: cmin={cmin:.3g} cmax={cmax:.3g}")

    # Build one trace per matrix and toggle visibility via slider (more robust than frames)
    if not mats:
        raise SystemExit("No matrices to plot")

    traces = []
    for idx, (r, c, cv, shape, hov) in enumerate(mats):
        trace_kwargs = {
            'customdata': hov,
            'hovertemplate': '%{customdata:.6g}<extra></extra>'
        }
        traces.append(go.Scattergl(
            x=c, y=r, mode='markers', visible=(idx == 0),
            marker=dict(size=args.ms, color=cv, colorscale=args.colorscale, cmin=cmin, cmax=cmax, showscale=True,
                        colorbar=dict(title='log10(|A_ij|)' if args.log else 'A_ij',
                                      titlefont=dict(size=args.cbfs), tickfont=dict(size=args.tickfs))),
            name=f"ls {ls_ids[idx]}",
            **trace_kwargs
        ))

    r0, c0, cv0, shape0 = mats[0][0], mats[0][1], mats[0][2], mats[0][3]
    fig = go.Figure(data=traces)
    showgrid = not args.no_grid
    fig.update_layout(
        template='seaborn', #'plotly_white',
        width=args.width,
        height=args.height,
        margin=dict(l=80, r=100, t=90, b=80),
        title=dict(text=f"Sparsity pattern plot of {labels[0]}", font=dict(size=args.tfs), x=0.5, xanchor='center'),
        font=dict(size=args.tickfs),
        xaxis=dict(title='col', titlefont=dict(size=args.alfs), tickfont=dict(size=args.tickfs),
                   range=[0, shape0[1]], showgrid=showgrid, gridcolor='#dddddd', zeroline=False,
                   linewidth=1, linecolor='#333', mirror=True),
        yaxis=dict(title='row', titlefont=dict(size=args.alfs), tickfont=dict(size=args.tickfs),
                   autorange=False, range=[shape0[0], 0], scaleanchor='x', scaleratio=1,
                   showgrid=showgrid, gridcolor='#dddddd', zeroline=False,
                   linewidth=1, linecolor='#333', mirror=True)
    )

    steps = []
    for i in range(len(mats)):
        shape = mats[i][3]
        vis = [False] * len(mats)
        vis[i] = True
        steps.append(dict(
            method='update',
            args=[
                {'visible': vis},
                {'title': {'text': f"Sparsity pattern plot of {labels[i]}", 'font': {'size': args.tfs}, 'x': 0.5, 'xanchor': 'center'},
                 'xaxis': {'range': [0, shape[1]], 'title': {'font': {'size': args.alfs}}, 'tickfont': {'size': args.tickfs}},
                 'yaxis': {'range': [shape[0], 0], 'title': {'font': {'size': args.alfs}}, 'tickfont': {'size': args.tickfs}}}
            ],
            label=f"{ls_ids[i]}"
        ))

    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix='Index: ', font=dict(size=args.alfs)), steps=steps, pad=dict(t=30))]
    )

    if args.save:
        fig.write_html(args.save, include_plotlyjs='cdn')
        logging.info(f"Saved to {args.save}")
    else:
        logging.info("Opening interactive viewer")
        fig.show()


if __name__ == '__main__':
    main()
