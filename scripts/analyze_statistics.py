#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

import re
import os
import pandas as pd
import argparse
import bisect
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Global variables
fgs  = (10, 6)       # Figure size
tfs  = 18            # Title font size
alfs = 14            # Axis label font size
lgfs = 14            # Legends font size

logger = logging.getLogger(__name__)

def parse_statistics_summary(filename, exclude, source_label=None, filter_table=None):
    """
    Parse statistics from a log file.  A single file may contain multiple
    STATISTICS SUMMARY tables (e.g. from separate HYPREDRV objects).  Each
    table is returned as an independent (series_list, time_unit) pair so the
    caller can treat them as separate data sources.

    Args:
        filename: Path to the log file
        exclude: List of entry numbers to exclude
        source_label: Optional label to use for this source (defaults to filename)
        filter_table: If given, only return results from the table whose name
            matches this string (e.g. 'fractureMechSolver').  When a filter is
            active the source label is always the file-level label so that data
            from multiple files remains distinguishable.

    Returns:
        list of (series_list, time_unit) tuples — one per table found.
    """
    logger.info(f"Parsing statistics from {filename = }")

    # Regular expressions to extract statistics and auxiliary data
    summary_pattern = re.compile(r"^\s*STATISTICS SUMMARY(?:\s+for\s+(.+?))?\s*:\s*$")
    table_divider_pattern = re.compile(r"^\+-+(?:\+-+)+\+\s*$")
    rows_and_nonzeros_pattern = re.compile(
        r"Solving linear system #\d+ with (\d+) rows and (\d+) nonzeros..."
    )
    mpi_rank_pattern = re.compile(
        r"Running on (?P<p1>\d+) MPI rank[s]?|^Num ranks:\s+(?P<p2>\d+)"
    )
    # Format: "Linear Solver | Status | Unknowns: X | Nonzeros: Y | ..."
    unknowns_pattern = re.compile(
        r"Linear Solver\s*\|[^|]*\|\s*Unknowns:\s*([\d,]+)\s*\|\s*Nonzeros:\s*([\d,]+)"
    )
    # "  solverName: Global solution scaling factor" — appears right after the
    # "Linear Solver |..." line and lets us associate it with a named solver.
    solver_global_pattern = re.compile(r"^\s+(\S+?):\s+Global solution scaling factor")
    time_unit_pattern = re.compile(r"\s*use_millisec:\s*(\S+)")

    # Per-table accumulators
    tables = []         # list of (table_name, data, use_path_column)
    current_data = None
    current_name = None
    current_use_path = False

    # Shared across the file
    rows = []
    nonzeros = []
    statistics_found = False
    in_summary_table = False
    time_unit = "[s]"
    nranks = 1
    # Pper-solver row/nonzero counts: solver_name -> (rows, nonzeros)
    # Populated by pairing "Linear Solver |...| Unknowns: X |..." with the
    # "solverName: Global solution scaling factor" line that follows it.
    table_default_rows = {}      # solver_name -> rows count
    table_default_nonzeros = {}  # solver_name -> nonzeros count
    _pending_rows = None         # rows parsed from most recent "Unknowns:" line
    _pending_nz   = None

    with open(filename, 'r') as fn:
        for line in fn:
            if match := time_unit_pattern.match(line):
                if match.group(1) in ("on", "1", "true", "y", "yes"):
                    time_unit = "[ms]"
                else:
                    time_unit = "[s]"

            if mpi_rank_search := mpi_rank_pattern.search(line):
                raw_rank = mpi_rank_search.group('p1') or mpi_rank_search.group('p2')
                nranks = int(raw_rank)
                logger.debug(f"Detected MPI ranks: {nranks}")
                continue

            # Logic for matching solver unknowns
            if rows_nonzeros_match := rows_and_nonzeros_pattern.match(line):
                rows.append(int(rows_nonzeros_match.group(1)))
                nonzeros.append(int(rows_nonzeros_match.group(2)))
                continue

            # "Linear Solver | ... | Unknowns: X | Nonzeros: Y |..." line.
            # Save the counts; we'll associate them with the solver on the next
            # "solverName: Global solution scaling factor" line.
            if unknowns_match := unknowns_pattern.search(line):
                # Strip commas if numbers look like "1,000,000"
                _pending_rows = int(unknowns_match.group(1).replace(',', ''))
                _pending_nz   = int(unknowns_match.group(2).replace(',', ''))
                continue

            if _pending_rows is not None:
                if solver_global_match := solver_global_pattern.match(line):
                    sname = solver_global_match.group(1)
                    table_default_rows[sname]     = _pending_rows
                    table_default_nonzeros[sname] = _pending_nz
                    _pending_rows = None
                    _pending_nz   = None
                    continue

            if summary_match := summary_pattern.match(line):
                # Flush previous table (if any)
                if current_data is not None and current_data:
                    tables.append((current_name, current_data, current_use_path))
                statistics_found = True
                in_summary_table = False
                current_data = []
                current_name = summary_match.group(1)  # None when unnamed
                current_use_path = False
                continue

            if statistics_found and table_divider_pattern.match(line):
                in_summary_table = True
                continue

            if in_summary_table and line.lstrip().startswith("|"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 9:
                    continue

                first_col = parts[1]

                # Detect column type from header row
                if first_col == 'Path':
                    current_use_path = True
                    continue
                elif first_col == 'Entry':
                    continue

                # Keep only per-entry rows; skip sub-headers and aggregate
                # rows such as Min./Max./Avg./Std./Total.
                if current_use_path:
                    if not all(s.isdigit() for s in first_col.split('.')):
                        continue
                else:
                    if not first_col.isdigit():
                        continue

                # parts layout for data rows:
                # ['', entry_or_path, build, setup, solve, r0, rr, iters, '']
                current_data.append((
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                    parts[6],
                    parts[7],
                ))

    # Flush last table
    if current_data:
        tables.append((current_name, current_data, current_use_path))

    if not tables:
        if not statistics_found:
            raise ValueError("no STATISTICS SUMMARY found")
        raise ValueError("STATISTICS SUMMARY found but contained no data")

    # Optionally restrict to a single named table
    if filter_table is not None:
        filtered = [(tn, d, up) for tn, d, up in tables if tn == filter_table]
        if not filtered:
            available = [tn or "(unnamed)" for tn, _, _ in tables]
            raise ValueError(
                f"table '{filter_table}' not found; available: {available}"
            )
        tables = filtered

    # Build source labels for each table
    base_label = source_label if source_label is not None else filename
    num_tables = len(tables)

    results = []
    global_entry_offset = 0  # track cumulative row offset for rows/nonzeros lookup
    for table_idx, (table_name, data, use_path_column) in enumerate(tables):
        if filter_table is not None:
            # Filtering active: always use the file-level label so that data
            # from multiple files remains distinguishable.
            label = base_label
        elif table_name:
            label = table_name if num_tables > 1 else base_label
        elif num_tables > 1:
            label = f"{base_label} ({table_idx + 1})"
        else:
            label = base_label

        series_list = []
        for i, row in enumerate(data):
            if use_path_column:
                segments = [int(s) for s in row[0].split('.')]
                entry_num = i
            else:
                segments = None
                entry_num = int(row[0])

            if entry_num in exclude:
                continue

            gi = global_entry_offset + i  # global index for rows/nonzeros
            build_time = float(row[1]) if row[1] not in (None, "") else None

            # Per-entry rows/nonzeros from "Solving linear system..." (hypredrive format).
            # Fall back to the per-solver map built from "Unknowns:" lines when the index-based list is empty.
            if gi < len(rows):
                entry_rows = int(rows[gi])
                entry_nz   = int(nonzeros[gi]) if gi < len(nonzeros) else None
            else:
                entry_rows = table_default_rows.get(table_name)
                entry_nz   = table_default_nonzeros.get(table_name)
            entry_data = {
                'entry': entry_num,
                'source': label,
                'nranks': int(nranks),
                'rows': entry_rows,
                'nonzeros': entry_nz,
                'build': build_time,
                'setup': float(row[2]),
                'solve': float(row[3]),
                'total': float(row[2]) + float(row[3]),
                'r0norm': float(row[4]),
                'resnorm': float(row[5]),
                'iters': int(row[6])
            }

            if segments is not None:
                entry_data['path'] = row[0]
                entry_data['ls_id'] = segments[-1]
                if len(segments) >= 2:
                    entry_data['timestep'] = segments[0]
                if len(segments) >= 3:
                    entry_data['nl_step'] = segments[1]

            series = pd.Series(entry_data, name=f'log_{filename}_t{table_idx}_entry_{entry_num}')
            series_list.append(series)

        global_entry_offset += len(data)

        if use_path_column:
            logger.info(f"Detected Path column in table {table_idx + 1} of {filename}")
        logger.debug(f"Parsed {len(series_list)} entries from table {table_idx + 1} of {filename} (time_unit={time_unit}, nranks={nranks})")
        results.append((series_list, time_unit))

    logger.info(f"Found {num_tables} statistics table(s) in {filename}")
    return results

def save_and_show_plot(savefig=None):
    """
    Saves the plot to a file with appropriate DPI settings for bitmap formats and always displays the plot.

    Parameters:
    - savefig (str, optional): File path to save the figure to. If provided, the plot is saved.
    """
    if savefig and savefig[-4:] != "None":
        # Determine the format from the file extension
        file_extension = savefig.split('.')[-1].lower()
        if file_extension in ['png', 'jpg', 'jpeg', 'bmp']:
            dpi = 600
        else:
            dpi = None  # Use default for non-bitmap formats or vector graphics

        logger.info(f"Saving figure: {savefig = } ...")
        plt.savefig(savefig, dpi=dpi)  # Save the figure with the appropriate DPI

    # Always display the plot regardless of saving
    plt.show()
    plt.close()

def plot_iterations(df, cumulative, xtype, xlabel, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None, latex=False, show_nl_iters=True, log_x=False, log_y=False):
    """
    Plots iteration counts as a function of a specified column in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data. It must include columns specified by 'xtype' and 'iters'.
    - cumulative (boolean): Plot cumulative sums of the quantities if True.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - use_title (boolean, optional): Turn on figure's title.
    - savefig (str, optional): File path to save the figure to. If not provided, the plot is displayed.

    Globals:
    - fgs (tuple): Figure size for the plot.
    - tfs (int): Font size for the title.
    - alfs (int): Font size for the axis labels.

    The function does not return anything but displays a plot.
    """

    # Determine grouping by source (if present)
    has_source = 'source' in df.columns
    sources = df['source'].unique().tolist() if has_source else []
    multiple_sources = has_source and len(sources) > 1

    agg_str = "agg_" if cumulative else ''

    logger.debug(f"Plotting iterations (cumulative={cumulative}, xtype={xtype})")
    # Plot figure
    plt.figure(figsize=fgs)

    # Resolve marker size to a concrete value to avoid passing None
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']

    def resolve_ls(user_ls, default_ls='-'):
        if user_ls == 'auto':
            return default_ls
        if user_ls == 'none':
            return 'None'
        return user_ls

    def resolve_single_legend():
        if sources:
            return get_legend_name(sources[0], legend_names or {}, latex)
        if legend_names:
            return next(iter(legend_names.values()))
        return "data"

    has_nl_iters = 'nl_iters_10x' in df.columns
    show_nl = has_nl_iters and show_nl_iters

    if multiple_sources:
        for src in sources:
            grp = df[df['source'] == src].sort_values(by=xtype)
            y = grp['iters'].cumsum() if cumulative else grp['iters']
            ls = resolve_ls(linestyle, '-')
            legend_name = get_legend_name(src, legend_names or {}, latex)
            linear_label = f"Linear Iters. ({legend_name})" if show_nl else legend_name
            plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=linear_label)
            if show_nl:
                nl = grp['nl_iters_10x'].cumsum() if cumulative else grp['nl_iters_10x']
                plt.plot(grp[xtype], nl, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                         label=f"Non-linear iters. (10x) ({legend_name})")
    else:
        grp = df.sort_values(by=xtype)
        y = grp['iters'].cumsum() if cumulative else grp['iters']
        ls = resolve_ls(linestyle, '-')
        legend_name = "Linear Iters." if show_nl else resolve_single_legend()
        plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=legend_name)
        if show_nl:
            nl = grp['nl_iters_10x'].cumsum() if cumulative else grp['nl_iters_10x']
            plt.plot(grp[xtype], nl, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                     label="Non-linear iters. (10x)")

    plt.legend(loc="upper left", fontsize=lgfs)
    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver iterations vs {xlabel}', fontsize=tfs, fontweight='bold')
    plt.ylabel('Iterations', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    ax = plt.gca()
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    else:
        plt.ylim(bottom=0.0)
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    if not log_x:
        try:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        except Exception as exc:
            logger.debug("integer MaxNLocator not applied: %s", exc)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"iters_{agg_str}{savefig}")

def plot_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None, latex=False, log_x=False, log_y=False):
    """
    Plots setup and solve times, as well as their total, as a function of a specified column in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data with 'setup' and 'solve' times among its columns.
    - cumulative (boolean): Plot cumulative sums of the quantities if True.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - time_unit (str): Unit used for time in the y-axis
    - use_title (boolean, optional): Turn on figure's title.
    - savefig (str, optional): File path to save the figure to. If not provided, the plot is displayed.

    Globals:
    - fgs (tuple): Figure size for the plot.
    - tfs (int): Font size for the title.
    - alfs (int): Font size for the axis labels.
    - lgfs (int): Font size for the legend.

    The plot includes three lines representing the 'setup' time, 'solve' time, and their total time for each entry in
    the DataFrame, as determined by the xtype column. The function does not return anything but displays the plot.
    """

    # Determine grouping by source (if present)
    has_source = 'source' in df.columns
    sources = df['source'].unique().tolist() if has_source else []
    multiple_sources = has_source and len(sources) > 1

    agg_str = "agg_" if cumulative else ''

    logger.debug(f"Plotting times (cumulative={cumulative}, xtype={xtype})")
    # Plot figure
    plt.figure(figsize=fgs)

    # Resolve marker size
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']

    def resolve_ls(user_ls, default_ls='-'):
        if user_ls == 'auto':
            return default_ls
        if user_ls == 'none':
            return 'None'
        return user_ls

    if multiple_sources:
        for src in sources:
            grp = df[df['source'] == src].sort_values(by=xtype)
            setup_data = grp['setup'].cumsum() if cumulative else grp['setup']
            solve_data = grp['solve'].cumsum() if cumulative else grp['solve']
            total_data = grp['total'].cumsum() if cumulative else grp['total']
            ls = resolve_ls(linestyle, '-')
            legend_name = get_legend_name(src, legend_names or {}, latex)
            plt.plot(grp[xtype], setup_data, marker='o', linestyle=ls, markersize=ms, label=f"Setup ({legend_name})")
            plt.plot(grp[xtype], solve_data, marker='o', linestyle=ls, markersize=ms, label=f"Solve ({legend_name})")
            plt.plot(grp[xtype], total_data, marker='o', linestyle=ls, markersize=ms, label=f"Total ({legend_name})")
        plt.legend(loc="best", fontsize=lgfs)
    else:
        grp = df.sort_values(by=xtype)
        setup_data = grp['setup'].cumsum() if cumulative else grp['setup']
        solve_data = grp['solve'].cumsum() if cumulative else grp['solve']
        total_data = grp['total'].cumsum() if cumulative else grp['total']
        ls = resolve_ls(linestyle, '-')
        plt.plot(grp[xtype], setup_data, marker='o', linestyle=ls, markersize=ms, label="Setup")
        plt.plot(grp[xtype], solve_data, marker='o', linestyle=ls, markersize=ms, label="Solve")
        plt.plot(grp[xtype], total_data, marker='o', linestyle=ls, markersize=ms, label="Total")
        plt.legend(loc="best", fontsize=lgfs)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver times vs {xlabel}', fontsize=tfs, fontweight='bold')
    plt.ylabel(f'Times {time_unit}', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    ax = plt.gca()
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    if not log_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    if not log_y:
        plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"times_{agg_str}{savefig}")

def plot_time_metric(df, cumulative, xtype, xlabel, time_unit, metric, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None, latex=False, log_x=False, log_y=False):
    """
    Plots a single time metric (one of 'setup', 'solve', 'total') across entries and files.
    Groups by 'source' when multiple input files are provided.
    """
    if metric not in ('setup', 'solve', 'total'):
        raise ValueError(f"Unsupported metric: {metric}")

    has_source = 'source' in df.columns
    sources = df['source'].unique().tolist() if has_source else []
    multiple_sources = has_source and len(sources) > 1

    agg_str = "agg_" if cumulative else ''
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']

    def resolve_ls(user_ls, default_ls='-'):
        if user_ls == 'auto':
            return default_ls
        if user_ls == 'none':
            return 'None'
        return user_ls

    logger.debug(f"Plotting metric '{metric}' (cumulative={cumulative}, xtype={xtype})")
    plt.figure(figsize=fgs)

    if multiple_sources:
        for src in sources:
            grp = df[df['source'] == src].sort_values(by=xtype)
            y = grp[metric].cumsum() if cumulative else grp[metric]
            ls = resolve_ls(linestyle, '-')
            legend_name = get_legend_name(src, legend_names or {}, latex)
            plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=f"{metric.capitalize()} ({legend_name})")
        plt.legend(loc="best", fontsize=lgfs)
    else:
        grp = df.sort_values(by=xtype)
        y = grp[metric].cumsum() if cumulative else grp[metric]
        ls = resolve_ls(linestyle, '-')
        plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=f"{metric.capitalize()}")
        plt.legend(loc="best", fontsize=lgfs)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f"{prefix}{metric.capitalize()} time vs {xlabel}", fontsize=tfs, fontweight='bold')
    plt.ylabel(f"Times {time_unit}", fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    ax = plt.gca()
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    if not log_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    if not log_y:
        plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"{metric}_{agg_str}{savefig}")

def plot_iters_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None, latex=False, show_nl_iters=True, log_x=False, log_y=False):
    """
    Plots setup and solve times, as well as iteration counts, as a function of a specified column in the DataFrame.
    Setup and solve times are plotted on the primary Y-axis, while iteration counts are plotted on a secondary Y-axis.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data with 'setup', 'solve', and 'iters' among its columns.
    - cumulative (boolean): Plot cumulative sums of the quantities if True.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - time_unit (str): Unit used for time in the y-axis
    - use_title (boolean, optional): Turn on figure's title.
    - savefig (str, optional): File path to save the figure to. If not provided, the plot is displayed.

    Globals:
    - fgs (tuple): Figure size for the plot. Must be defined elsewhere in the global scope.
    - tfs (int): Font size for the title. Must be defined elsewhere in the global scope.
    - alfs (int): Font size for the axis labels. Must be defined elsewhere in the global scope.
    - lgfs (int): Font size for the legend. Must be defined elsewhere in the global scope.

    The plot includes lines representing the 'setup' and 'solve' times on the primary Y-axis, and 'iters' on the secondary Y-axis.
    The function does not return anything but displays the plot.
    """
    logger.debug(f"Plotting iters-and-times (cumulative={cumulative}, xtype={xtype})")
    fig, ax1 = plt.subplots(figsize=fgs)

    # Determine grouping by source (if present)
    has_source = 'source' in df.columns
    sources = df['source'].unique().tolist() if has_source else []
    multiple_sources = has_source and len(sources) > 1

    agg_str = "agg_" if cumulative else ''

    # Plot setup and solve times on the primary Y-axis
    ax1.set_xlabel(xlabel, fontsize=alfs)
    ax1.set_ylabel(f'Times {time_unit}', fontsize=alfs)
    if log_x:
        ax1.set_xscale('log')
    if log_y:
        ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelsize=alfs)
    ax1.tick_params(axis='x', labelsize=alfs)
    if not log_x:
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # Secondary Y-axis for iteration counts
    ax2 = ax1.twinx()
    ax2.set_ylabel('Iterations', fontsize=alfs)
    ax2.tick_params(axis='y', labelsize=alfs)
    if log_y:
        ax2.set_yscale('log')

    lines = []
    labels = []

    # Resolve marker size
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']
    has_nl_iters = 'nl_iters_10x' in df.columns
    show_nl = has_nl_iters and show_nl_iters

    def resolve_ls(user_ls, default_ls='-'):
        if user_ls == 'auto':
            return default_ls
        if user_ls == 'none':
            return 'None'
        return user_ls

    if multiple_sources:
        max_iters = 0
        for src in sources:
            grp = df[df['source'] == src].sort_values(by=xtype)
            setup_data = grp['setup'].cumsum() if cumulative else grp['setup']
            solve_data = grp['solve'].cumsum() if cumulative else grp['solve']
            iters_data = grp['iters'].cumsum() if cumulative else grp['iters']
            nl_iters_data = grp['nl_iters_10x'].cumsum() if (show_nl and cumulative) else (grp.get('nl_iters_10x') if show_nl else None)
            ls_main = resolve_ls(linestyle, '-')
            ls_iter = resolve_ls(linestyle, '--')
            legend_name = get_legend_name(src, legend_names or {}, latex)
            l1, = ax1.plot(grp[xtype], setup_data, marker='o', linestyle=ls_main, markersize=ms, label=f"Setup ({legend_name})")
            l2, = ax1.plot(grp[xtype], solve_data, marker='o', linestyle=ls_main, markersize=ms, alpha=0.7, label=f"Solve ({legend_name})")
            linear_label = f"Linear Iters. ({legend_name})" if show_nl else f"Iterations ({legend_name})"
            l3, = ax2.plot(grp[xtype], iters_data, marker='o', linestyle=ls_iter, markersize=ms, label=linear_label)

            lines.extend([l1, l2, l3])
            labels.extend([l.get_label() for l in (l1, l2, l3)])
            max_iters = max(max_iters, max(iters_data) if len(iters_data) else 0)
            if show_nl and nl_iters_data is not None:
                l4, = ax2.plot(grp[xtype], nl_iters_data, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                               label=f"Non-linear iters. (10x) ({legend_name})")
                lines.append(l4)
                labels.append(l4.get_label())
                if len(nl_iters_data):
                    max_iters = max(max_iters, max(nl_iters_data))

        if log_y:
            ax2.set_ylim(bottom=1, top=max_iters * 2.0 if max_iters > 1 else 10)
        else:
            ax2.set_ylim(bottom=0, top=max_iters * 2.0 if max_iters > 0 else 1)
    else:
        grp = df.sort_values(by=xtype)
        setup_data = grp['setup'].cumsum() if cumulative else grp['setup']
        solve_data = grp['solve'].cumsum() if cumulative else grp['solve']
        iters_data = grp['iters'].cumsum() if cumulative else grp['iters']
        nl_iters_data = grp['nl_iters_10x'].cumsum() if (show_nl and cumulative) else (grp.get('nl_iters_10x') if show_nl else None)

        ls_main = resolve_ls(linestyle, '-')
        ls_iter = resolve_ls(linestyle, '--')
        l1, = ax1.plot(grp[xtype], setup_data, marker='o', linestyle=ls_main, markersize=ms, color='#E69F00', label="Setup")
        l2, = ax1.plot(grp[xtype], solve_data, marker='o', linestyle=ls_main, markersize=ms, color='#009E73', label="Solve", alpha=0.5)
        linear_label = "Linear Iters." if show_nl else "Iterations"
        l3, = ax2.plot(grp[xtype], iters_data, marker='o', linestyle=ls_iter, markersize=ms, color='#0072B2', label=linear_label)

        lines  = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        max_iters = max(iters_data) if len(iters_data) else 0
        if show_nl and nl_iters_data is not None:
            l4, = ax2.plot(grp[xtype], nl_iters_data, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                           color='#D55E00', label="Non-linear iters. (10x)")
            lines.append(l4)
            labels.append(l4.get_label())
            if len(nl_iters_data):
                max_iters = max(max_iters, max(nl_iters_data))
        if log_y:
            ax2.set_ylim(bottom=1, top=max_iters * 2.0 if max_iters > 1 else 10)
        else:
            ax2.set_ylim(bottom=0, top=max_iters * 2.0 if max_iters > 0 else 1)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver data vs {xlabel}', fontsize=tfs, fontweight='bold')

    lg = ax2.legend(lines, labels, loc="upper left", fontsize=lgfs)
    lg.set_zorder(100)

    fig.tight_layout()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, zorder=0)
    save_and_show_plot(f"iters_times_{agg_str}{savefig}")

def plot_bar_time_metric(df, metric, time_unit, labels, use_title=False, savefig=None, title=None):
    """
    Plots a bar chart for a single metric (one of 'setup', 'solve', 'total', 'iters')
    across entries in a single log file. Labels are provided by the caller.
    """
    if metric not in ('setup', 'solve', 'total', 'iters'):
        raise ValueError(f"Unsupported metric: {metric}")

    logger.debug(f"Plotting bar chart for metric '{metric}'")
    plt.figure(figsize=fgs)

    # Keep entry order stable (typically sorted by entry index)
    grp = df.sort_values(by='entry')
    y = grp[metric].tolist()
    x = list(range(len(y)))

    plt.bar(x, y, color='#4C72B0', zorder=2)
    plt.xticks(x, labels, fontsize=alfs, rotation=20, ha='right')
    if metric == 'iters':
        plt.ylabel("Iterations", fontsize=alfs)
    else:
        plt.ylabel(f"Times {time_unit}", fontsize=alfs)
    plt.xlabel("Solver", fontsize=alfs)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        if metric == 'iters':
            plt.title("Iterations by solver", fontsize=tfs, fontweight='bold')
        else:
            plt.title(f"{metric.capitalize()} time by solver", fontsize=tfs, fontweight='bold')

    plt.grid(True, axis='y', zorder=0)
    plt.tight_layout()
    save_and_show_plot(f"{metric}_bar_{savefig}")

def plot_throughput(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, title=None, log_x=False, log_y=False):
    """
    Plots throughput (degrees of freedom per second) as a function of a specified column in the DataFrame.
    Throughput is calculated as number of rows (degrees of freedom) divided by total time (setup + solve).

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data with 'total' and 'rows' among its columns.
    - cumulative (boolean): Plot cumulative sums of the quantities if True.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - time_unit (str): Unit used for time (e.g., "[s]" or "[ms]")
    - use_title (boolean, optional): Turn on figure's title.
    - savefig (str, optional): File path to save the figure to. If not provided, the plot is displayed.

    Globals:
    - fgs (tuple): Figure size for the plot.
    - tfs (int): Font size for the title.
    - alfs (int): Font size for the axis labels.
    - lgfs (int): Font size for the legend.

    The function does not return anything but displays the plot.
    """
    has_source = 'source' in df.columns
    sources = df['source'].unique().tolist() if has_source else []
    multiple_sources = has_source and len(sources) > 1

    agg_str = "agg_" if cumulative else ''
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']

    def resolve_ls(user_ls, default_ls='-'):
        if user_ls == 'auto':
            return default_ls
        if user_ls == 'none':
            return 'None'
        return user_ls

    logger.debug(f"Plotting throughput (cumulative={cumulative}, xtype={xtype})")
    plt.figure(figsize=fgs)

    # Calculate throughput: rows / total time (DOFs per second)
    # Handle None values in rows column and zero/negative time values
    df_with_throughput = df.copy()
    df_with_throughput['throughput'] = df_with_throughput.apply(
        lambda row: row['rows'] / row['total'] if (pd.notna(row['rows']) and row['rows'] > 0 and
                                                    pd.notna(row['total']) and row['total'] > 0) else None,
        axis=1
    )

    if multiple_sources:
        for src in sources:
            grp = df_with_throughput[df_with_throughput['source'] == src].sort_values(by=xtype)
            # Filter out rows where throughput is None
            grp = grp[grp['throughput'].notna()]
            if len(grp) == 0:
                continue
            if cumulative:
                # For cumulative: cumulative_rows / cumulative_total_time (average throughput)
                cum_rows = grp['rows'].cumsum()
                cum_total = grp['total'].cumsum()
                y = cum_rows / cum_total
            else:
                y = grp['throughput']
            ls = resolve_ls(linestyle, '-')
            plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=f"Throughput ({src})")
        plt.legend(loc="best", fontsize=lgfs)
    else:
        grp = df_with_throughput.sort_values(by=xtype)
        # Filter out rows where throughput is None
        grp = grp[grp['throughput'].notna()]
        if len(grp) > 0:
            if cumulative:
                # For cumulative: cumulative_rows / cumulative_total_time (average throughput)
                cum_rows = grp['rows'].cumsum()
                cum_total = grp['total'].cumsum()
                y = cum_rows / cum_total
            else:
                y = grp['throughput']
            ls = resolve_ls(linestyle, '-')
            plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label="Throughput")
            plt.legend(loc="best", fontsize=lgfs)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Throughput (DOFs/s) vs {xlabel}', fontsize=tfs, fontweight='bold')

    # Format y-axis label based on time unit
    if time_unit == "[s]":
        throughput_unit = "DOFs/s"
    elif time_unit == "[ms]":
        throughput_unit = "DOFs/ms"
    else:
        throughput_unit = f"DOFs/{time_unit}"

    plt.ylabel(f'Throughput ({throughput_unit})', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    ax = plt.gca()
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    if not log_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    if not log_y:
        plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"throughput_{agg_str}{savefig}")

def plot_weak_scaling(df, xtype, xlabel, metrics, metric_labels, time_unit,
                      agg='mean', errbar=False, use_title=False, savefig=None,
                      linestyle='auto', markersize=None, title=None,
                      log_x=False, log_y=False, show_ideal=False,
                      annotate_iters=False):
    """
    Plots weak scalability: metrics aggregated by xtype (typically nranks) across files.

    Parameters:
    - df: DataFrame with an xtype column and the requested metric columns.
    - xtype: Column to group by and use as x-axis (e.g. 'nranks').
    - xlabel: X-axis label string.
    - metrics: List of metric column names to plot (e.g. ['setup', 'solve', 'total']).
    - metric_labels: Corresponding legend/axis labels for each metric.
    - time_unit: Time unit string used in the y-axis label for time metrics.
    - agg: Aggregation applied within each xtype group ('mean', 'median', 'min', 'max', 'sum').
    - errbar: If True, shade the min–max range around each line.
    - show_ideal: If True, draw a dashed horizontal reference at the smallest-xtype value
        (ideal weak scaling = constant metric regardless of problem size).
    - annotate_iters: If True and 'iter_times' is among the metrics, annotate each point on
        the iter_times curve with the aggregated average iteration count.
    """
    logger.debug(f"Plotting weak scaling (xtype={xtype}, metrics={metrics}, agg={agg})")

    if df[xtype].isna().all():
        raise ValueError(f"Column '{xtype}' has no valid data for weak scaling plot.")

    # Determine y-axis label
    time_metrics = {'setup', 'solve', 'total', 'build', 'iter_times'}
    if metrics == ['iters']:
        ylabel = 'Iterations'
    elif metrics == ['throughput']:
        unit_str = 'DOFs/s' if time_unit == '[s]' else f'DOFs/{time_unit.strip("[]")}'
        ylabel = f'Throughput ({unit_str})'
    elif all(m in time_metrics for m in metrics):
        ylabel = f'Time {time_unit}'
    else:
        ylabel = 'Value'

    # Build aggregation spec: always compute agg + min/max for optional errbar.
    # Also pull in 'iters' (mean) when annotation is requested for iter_times.
    agg_spec = {m: [agg, 'min', 'max'] for m in metrics if m in df.columns}
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Metric column(s) not found in data: {missing}")
    need_iters_agg = annotate_iters and 'iter_times' in metrics and 'iters' in df.columns
    if need_iters_agg:
        agg_spec['iters'] = ['mean']

    grouped = df.groupby(xtype, as_index=False).agg(agg_spec).sort_values(xtype)
    # Flatten MultiIndex columns produced by multi-function agg
    grouped.columns = [
        '_'.join(str(c) for c in col).rstrip('_') if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    # Build per-x nranks array for secondary tick labels (when xtype is not nranks itself)
    nranks_per_x = None
    if xtype != 'nranks' and 'nranks' in df.columns:
        nranks_lookup = df.groupby(xtype)['nranks'].first()
        nranks_per_x = grouped[xtype].map(nranks_lookup).to_numpy(dtype=float, na_value=float('nan'))

    plt.figure(figsize=fgs)
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']

    def resolve_ls(user_ls, default_ls='-'):
        if user_ls == 'auto':
            return default_ls
        if user_ls == 'none':
            return 'None'
        return user_ls

    import numpy as np

    ls = resolve_ls(linestyle)
    # Cast to plain float to avoid pandas nullable-integer/extension-array
    # incompatibilities with matplotlib (fill_between, axhline, etc.)
    x = grouped[xtype].to_numpy(dtype=float, na_value=float('nan'))

    ax = plt.gca()
    for metric, mlabel in zip(metrics, metric_labels):
        agg_col = f'{metric}_{agg}'
        y = grouped[agg_col].to_numpy(dtype=float, na_value=float('nan'))
        ax.plot(x, y, marker='o', linestyle=ls, markersize=ms, label=mlabel)

        if errbar:
            y_lo = grouped[f'{metric}_min'].to_numpy(dtype=float, na_value=float('nan'))
            y_hi = grouped[f'{metric}_max'].to_numpy(dtype=float, na_value=float('nan'))
            ax.fill_between(x, y_lo, y_hi, alpha=0.2)

        if show_ideal and len(x) > 0:
            ref = y[0]
            ax.axhline(ref, linestyle='--', linewidth=1, alpha=0.6,
                       label=f'Ideal ({mlabel})')

        if annotate_iters and metric == 'iter_times' and need_iters_agg:
            iters_avg = grouped['iters_mean'].to_numpy(dtype=float, na_value=float('nan'))
            for xi, yi, ni in zip(x, y, iters_avg):
                if np.isnan(xi) or np.isnan(yi) or np.isnan(ni):
                    continue
                ax.annotate(
                    f"{ni:.0f}",
                    xy=(xi, yi),
                    xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=max(alfs - 4, 8),
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='gray', alpha=0.75),
                )

    plt.legend(loc='best', fontsize=lgfs)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        agg_str_label = agg.capitalize()
        plt.title(f'Weak Scaling — {agg_str_label} vs {xlabel}', fontsize=tfs, fontweight='bold')

    plt.ylabel(ylabel, fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    else:
        if metrics != ['iters']:
            plt.ylim(bottom=0.0)
        else:
            plt.ylim(bottom=0)
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    if not log_x:
        if nranks_per_x is not None and not np.isnan(nranks_per_x).all():
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{int(xi):,}\n({int(ni):,} ranks)" if not (np.isnan(xi) or np.isnan(ni)) else f"{xi}"
                 for xi, ni in zip(x, nranks_per_x)],
                fontsize=alfs,
            )
        else:
            try:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
            except Exception as exc:
                logging.debug("Could not set integer x-axis locator; using default locator.", exc_info=exc)
    plt.grid(True)
    plt.tight_layout()
    metric_tag = '_'.join(metrics)
    save_and_show_plot(f"weakscaling_{metric_tag}_{savefig}" if savefig else None)

def check_mode_exact_match(mode, word):
    # Split the mode string into parts separated by '+'
    parts = mode.split('+')

    # Check if the word exactly matches any of the parts
    return word in parts

def get_legend_name(source, legend_names, latex=False):
    """Get legend name for a source, using custom mapping if available."""
    name = legend_names.get(source, str(source))
    return f"${name}$" if latex else name

def read_timesteps_file(path):
    """
    Reads a timesteps file with format:
      <num_timesteps>
      <timestep_index> <ls_start>
    Returns a list of (timestep_index, ls_start) sorted by ls_start.
    """
    entries = []
    with open(path, "r") as f:
        header = f.readline()
        if not header:
            raise ValueError(f"Empty timesteps file: {path}")
        try:
            total = int(header.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid timesteps header in {path}") from exc
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid timesteps line in {path}: {line}")
            tstep = int(parts[0])
            start = int(parts[1])
            entries.append((tstep, start))

    if len(entries) != total:
        raise ValueError(f"Expected {total} timesteps, found {len(entries)} in {path}")

    entries.sort(key=lambda x: x[1])
    return entries

def map_entry_to_timestep(entries, entry):
    starts = [s for _, s in entries]
    pos = bisect.bisect_right(starts, entry) - 1
    if pos < 0:
        return entries[0][0]
    return entries[pos][0]

def map_entry_to_timestep_offset(entries, entry):
    starts = [s for _, s in entries]
    pos = bisect.bisect_right(starts, entry) - 1
    if pos < 0:
        pos = 0
    tstep = entries[pos][0]
    start = entries[pos][1]
    next_start = starts[pos + 1] if pos + 1 < len(starts) else None
    if next_start is None:
        count = max(1, entry - start + 1)
    else:
        count = max(1, next_start - start)
    local_index = entry - start
    return tstep + (local_index / count)

def infer_nonlinear_iters_per_timestep(entries, max_entry):
    """Infer nonlinear iteration count per timestep from timestep start indices."""
    counts = {}
    starts = [s for _, s in entries]
    for i, (tstep, start) in enumerate(entries):
        next_start = starts[i + 1] if i + 1 < len(starts) else (max_entry + 1)
        counts[tstep] = max(1, next_start - start)
    return counts

def main():
    # List of pre-defined labels
    labels = {'rows': "Number of rows",
              'nonzeros': "Number of nonzeros",
              'entry': "Linear system number",
              'nranks': "Number of MPI ranks",
              'timestep': "Timestep",
              'timestep_offset': "Timestep",
              'ls_id': "Linear system ID",
              'nl_step': "Nonlinear step"}

    # List of pre-defined modes:
    mode_choices = ('iters', 'times', 'iters-and-times', 'setup', 'solve', 'total', 'throughput', 'bar', 'weak-scaling')

    # Metric qualifiers accepted after 'weak-scaling+' or 'bar+'
    ws_metric_choices  = ('setup', 'solve', 'total', 'times', 'iters', 'throughput', 'iter-times')
    bar_metric_choices = ('setup', 'solve', 'total', 'iters')

    # Parser for plus-separated modes.
    # Accepted forms:
    #   • plain modes combined with '+': "setup+solve"
    #   • "weak-scaling[+metric+...]"  e.g. "weak-scaling+total+iter-times"
    #   • "bar[+metric]"               e.g. "bar+total"
    def parse_modes(value):
        parts = [p.replace('_', '-') for p in value.split('+')]
        if parts[0] == 'weak-scaling':
            metrics = parts[1:]
            invalid = [p for p in metrics if p not in ws_metric_choices]
            if invalid:
                raise argparse.ArgumentTypeError(
                    f"Invalid weak-scaling metric(s): {', '.join(invalid)}. "
                    f"Valid: {', '.join(ws_metric_choices)}"
                )
            return '+'.join(['weak-scaling'] + metrics)
        if parts[0] == 'bar':
            metrics = parts[1:]
            if len(metrics) > 1:
                raise argparse.ArgumentTypeError("'bar' mode accepts at most one metric qualifier")
            if metrics and metrics[0] not in bar_metric_choices:
                raise argparse.ArgumentTypeError(
                    f"Invalid bar metric: '{metrics[0]}'. "
                    f"Valid: {', '.join(bar_metric_choices)}"
                )
            return '+'.join(['bar'] + metrics)
        invalid = [p for p in parts if p not in mode_choices]
        if invalid:
            raise argparse.ArgumentTypeError(f"Invalid mode(s): {', '.join(invalid)}. Valid: {', '.join(mode_choices)}")
        return '+'.join(parts)

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse the Statistics Summary produced by hypredrive")
    parser.add_argument("-f", "--filename", type=str, nargs="+", required=True, help="Path to the log file")
    parser.add_argument("-e", "--exclude", type=int, nargs="+", default=[], help="Exclude certain entries from the statistics")
    parser.add_argument("-m", "--mode", type=parse_modes, default='iters-and-times',
                        help="What to plot. Combine plain modes with '+' (e.g. 'setup+solve'). "
                             "For weak-scaling append metric qualifiers: 'weak-scaling+total+iter-times'. "
                             "For bar append the metric: 'bar+iters'. "
                             f"Plain modes: {', '.join(mode_choices)}.")
    parser.add_argument("-t", "--xtype", type=str, default='entry', choices=labels.keys(), help="Variable type for the abscissa")
    parser.add_argument("-l", "--xlabel", type=str, default=None, help="Label for the abscissa")
    parser.add_argument("-s", "--savefig", default=None, help="Save figure(s) given this name suffix")
    parser.add_argument("-c", "--cumulative", action='store_true', help='Plot cumulative quantities')
    parser.add_argument("-u", "--use_title", action='store_true', help='Show title in plots')
    parser.add_argument("-T", "--title", type=str, default=None, help="Custom title for plots")
    parser.add_argument("-ls", "--linestyle", type=str, default='auto', choices=['auto', '-', '--', '-.', ':', 'none'], help="Line style for plots; 'none' draws markers only; 'auto' preserves defaults")
    parser.add_argument("-ms", "--markersize", type=float, default=None, help="Marker size (points); defaults to Matplotlib rcParams")
    parser.add_argument("-ln", "--legend-names", type=str, nargs="+", default=None, help="Custom legend labels for each input file (must match number of files)")
    parser.add_argument("--latex-legend", action="store_true", help="Wrap legend labels in $...$ for LaTeX math rendering")
    parser.add_argument("--nl-iters", action=argparse.BooleanOptionalAction, default=None,
                        help="Show non-linear iteration counts (default: on for single file, off for multiple files)")
    parser.add_argument("--tsteps", type=str, default=None, help="Timesteps file mapping timestep index to starting ls id")
    parser.add_argument("--tsteps-aggregate", action="store_true",
                        help="Aggregate stats within each timestep (requires --tsteps)")
    parser.add_argument("--table-name", type=str, default=None,
                        help="Select a specific named statistics table from files that contain "
                             "multiple tables (e.g. 'fractureMechSolver'). When set the source "
                             "label always reflects the input file, not the table name.")
    parser.add_argument("--weak-scaling-agg", type=str, default='mean',
                        choices=['mean', 'median', 'min', 'max', 'sum'],
                        help="Aggregation applied to each metric within a problem-size group "
                             "for weak-scaling plots (default: mean)")
    parser.add_argument("--weak-scaling-errbar", action='store_true',
                        help="Shade the min–max range around the aggregated line in "
                             "weak-scaling plots")
    parser.add_argument("--weak-scaling-ideal", action='store_true',
                        help="Draw a dashed reference line at the smallest problem-size value "
                             "(ideal weak scaling = constant metric)")
    parser.add_argument("--weak-scaling-annotate-iters", action='store_true',
                        help="Annotate each point on the iter-times curve with the avg. "
                             "iteration count (only effective when iter-times is plotted)")
    parser.add_argument("--log-x", action="store_true", help="Use log scale for X axis")
    parser.add_argument("--log-y", action="store_true", help="Use log scale for Y axis")
    parser.add_argument("-v", "--verbose", action='count', default=0, help='Increase verbosity (-v=INFO, -vv=DEBUG)')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging level based on verbosity
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    # Suppress noisy third-party DEBUG logs (e.g., Matplotlib font manager, PIL PNG plugin)
    for noisy_logger in (
        'matplotlib',
        'matplotlib.font_manager',
        'PIL',
        'PIL.PngImagePlugin',
        'fontTools'
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.debug(f"Arguments parsed: {vars(args) = }")

    bar_mode = check_mode_exact_match(args.mode, 'bar')
    show_nl_iters = args.nl_iters if args.nl_iters is not None else (len(args.filename) == 1)

    # Create label mapping from source filenames to custom labels
    label_map = {}
    if args.legend_names and not bar_mode:
        if len(args.legend_names) != len(args.filename):
            raise ValueError(f"Number of legend names ({len(args.legend_names)}) must match number of files ({len(args.filename)})")
        for filename, label in zip(args.filename, args.legend_names):
            source_key = os.path.basename(filename)
            label_map[source_key] = label
        logger.debug(f"Label mapping: {label_map = }")

    # Parse the statistics summary
    data = []
    for idx, filename in enumerate(args.filename):
        # Use custom label if provided, otherwise generate a unique label
        if args.legend_names and idx < len(args.legend_names):
            source_label = args.legend_names[idx]
        elif len(args.filename) > 1:
            # If multiple files, use a shortened path to distinguish them
            # Try to use a meaningful part of the path (e.g., parent directory)
            path_parts = os.path.normpath(filename).split(os.sep)
            if len(path_parts) > 1:
                # Use parent directory + filename if available
                source_label = os.path.join(path_parts[-2], path_parts[-1])
            else:
                source_label = filename
        else:
            # Single file: use basename
            source_label = os.path.basename(filename)

        try:
            table_results = parse_statistics_summary(filename, args.exclude, source_label,
                                                          filter_table=args.table_name)
        except ValueError as exc:
            logger.warning(f"Skipping {filename}: {exc}")
            continue
        for series_list, time_unit in table_results:
            data.extend(series_list)
    num_input_files  = len(args.filename)
    num_data_entries = len(data)
    logger.info(f"Parsed {num_input_files = }")
    logger.info(f"Found {num_data_entries = }")

    if not data:
        sys.exit("Error: no usable statistics data found in any of the provided files.")

    # Assemble all series into a single DataFrame
    df = pd.concat(data, axis=1).T.reset_index(drop=True)

    # Note: Custom labels are now applied during parsing, so label_map is no longer needed here
    # But we keep it for backward compatibility if someone passes both --legend-labels and expects mapping
    if label_map:
        df['source'] = df['source'].map(lambda x: label_map.get(x, x))

    # Explicitly specify data types for each column
    # Use nullable integer types (Int64) for columns that can have None values
    data_types = {
        'entry':    'int',
        'nranks':   'int',
        'rows':     'Int64',  # Nullable integer type
        'nonzeros': 'Int64',  # Nullable integer type
        'build':    'float',
        'setup':    'float',
        'solve':    'float',
        'total':    'float',
        'r0norm':   'float',
        'resnorm':  'float',
        'iters':    'int'
    }
    # Path-derived columns (present when stats table uses Path column)
    has_path_data = 'path' in df.columns
    if has_path_data:
        data_types['ls_id'] = 'int'
    if 'timestep' in df.columns:
        data_types['timestep'] = 'int'
    if 'nl_step' in df.columns:
        data_types['nl_step'] = 'Int64'

    # Convert data types (only for columns that exist)
    df = df.astype({k: v for k, v in data_types.items() if k in df.columns})

    if args.tsteps:
        tsteps = read_timesteps_file(args.tsteps)
        df['timestep'] = df['entry'].apply(lambda e: map_entry_to_timestep(tsteps, int(e)))
        df['timestep_offset'] = df['entry'].apply(lambda e: map_entry_to_timestep_offset(tsteps, int(e)))
        if args.xtype == 'entry':
            args.xtype = 'timestep_offset'
        if args.tsteps_aggregate:
            max_entry = int(df['entry'].max())
            nl_iters_per_tstep = infer_nonlinear_iters_per_timestep(tsteps, max_entry)
            group_cols = ['timestep']
            if 'source' in df.columns:
                group_cols = ['source', 'timestep']
            agg_map = {
                'entry': 'min',
                'nranks': 'first',
                'rows': 'sum',
                'nonzeros': 'sum',
                'build': 'sum',
                'setup': 'sum',
                'solve': 'sum',
                'total': 'sum',
                'r0norm': 'mean',
                'resnorm': 'mean',
                'iters': 'sum',
            }
            df = df.groupby(group_cols, as_index=False).agg(agg_map)
            df['nl_iters_10x'] = (10 * df['timestep'].map(lambda t: nl_iters_per_tstep.get(int(t), 0))).astype(int)
            if args.xtype in ('entry', 'timestep_offset'):
                args.xtype = 'timestep'
    elif has_path_data and 'timestep' in df.columns:
        # Path column provides timestep info — no --tsteps file needed
        if args.tsteps_aggregate:
            group_cols = ['timestep']
            if 'source' in df.columns:
                group_cols = ['source', 'timestep']
            # Count entries per group before aggregating (= nonlinear iters)
            nl_counts = df.groupby(group_cols).size().reset_index(name='_nl_count')
            agg_map = {
                'entry': 'min',
                'nranks': 'first',
                'rows': 'sum',
                'nonzeros': 'sum',
                'build': 'sum',
                'setup': 'sum',
                'solve': 'sum',
                'total': 'sum',
                'r0norm': 'mean',
                'resnorm': 'mean',
                'iters': 'sum',
            }
            if 'ls_id' in df.columns:
                agg_map['ls_id'] = 'min'
            df = df.groupby(group_cols, as_index=False).agg(agg_map)
            df = df.merge(nl_counts, on=group_cols)
            df['nl_iters_10x'] = (10 * df['_nl_count']).astype(int)
            df.drop(columns=['_nl_count'], inplace=True)
            if args.xtype == 'entry':
                args.xtype = 'timestep'
        elif args.xtype == 'entry' and not check_mode_exact_match(args.mode, 'weak-scaling'):
            args.xtype = 'ls_id'
    elif args.tsteps_aggregate:
        raise ValueError("--tsteps-aggregate requires --tsteps or Path column data")

    # Create legend name mapping
    legend_names = {}
    if args.legend_names and not bar_mode:
        if len(args.legend_names) != len(args.filename):
            raise ValueError(f"Number of legend labels ({len(args.legend_names)}) must match number of input files ({len(args.filename)})")
        # Map source (basename of filename) to custom legend name
        for filename, legend_name in zip(args.filename, args.legend_names):
            source = os.path.basename(filename)
            legend_names[source] = legend_name
        logger.info(f"Using custom legend names: {legend_names}")
    else:
        # No custom names provided, will use source filenames
        logger.debug("Using default legend names (source filenames)")

    # Update label
    xlabel = args.xlabel if args.xlabel else labels[args.xtype]

    # Optional DataFrame logging
    if args.verbose >= 2:
        logger.debug(f"DataFrame contents:\n{df.to_string(index=False)}")
    if args.verbose >= 1:
        logger.info(f"Sum total time: {df['total'].sum() = }")

    # Update savefig string
    savefig = args.savefig if args.savefig != "." else f"{(args.filename)[0].split('.')[0]}.png"

    # Produce plots
    if check_mode_exact_match(args.mode, 'weak-scaling'):
        mode_parts = args.mode.split('+')
        # metrics follow 'weak-scaling'; default to 'total' when none specified
        phase = '+'.join(mode_parts[1:]) if len(mode_parts) > 1 else 'total'

        # Determine x-axis: default to rows (problem size) when user left --xtype at
        # its default 'entry'.  Fall back to nranks if rows was not parsed.
        if args.xtype != 'entry':
            ws_xtype = args.xtype
        elif 'rows' in df.columns and df['rows'].notna().any():
            ws_xtype = 'rows'
        else:
            ws_xtype = 'nranks'
            logger.warning("rows not available; using nranks as weak-scaling x-axis")
        ws_xlabel = args.xlabel if args.xlabel else labels.get(ws_xtype, ws_xtype)

        df = df.copy()
        ws_metrics, ws_mlabels = [], []
        for ph in phase.split('+'):
            if ph == 'iters':
                ws_metrics.append('iters')
                ws_mlabels.append('Iterations')
            elif ph in ('setup', 'solve', 'total'):
                ws_metrics.append(ph)
                ws_mlabels.append(ph.capitalize())
            elif ph == 'times':
                ws_metrics += ['setup', 'solve', 'total']
                ws_mlabels += ['Setup', 'Solve', 'Total']
            elif ph == 'throughput':
                df['throughput'] = df.apply(
                    lambda r: r['rows'] / r['total']
                    if (pd.notna(r['rows']) and r['rows'] > 0
                        and pd.notna(r['total']) and r['total'] > 0)
                    else None,
                    axis=1,
                )
                unit_str = 'DOFs/s' if time_unit == '[s]' else f'DOFs/{time_unit.strip("[]")}'
                ws_metrics.append('throughput')
                ws_mlabels.append(f'Throughput ({unit_str})')
            elif ph == 'iter-times':
                df['iter_times'] = df.apply(
                    lambda r: r['solve'] / r['iters']
                    if (pd.notna(r['iters']) and r['iters'] > 0
                        and pd.notna(r['solve']))
                    else None,
                    axis=1,
                )
                ws_metrics.append('iter_times')
                ws_mlabels.append(f'Solve time / iter {time_unit}')

        plot_weak_scaling(df, ws_xtype, ws_xlabel, ws_metrics, ws_mlabels, time_unit,
                          agg=args.weak_scaling_agg,
                          errbar=args.weak_scaling_errbar,
                          use_title=args.use_title,
                          savefig=savefig,
                          linestyle=args.linestyle,
                          markersize=args.markersize,
                          title=args.title,
                          log_x=args.log_x,
                          log_y=args.log_y,
                          show_ideal=args.weak_scaling_ideal,
                          annotate_iters=args.weak_scaling_annotate_iters)
        return

    if check_mode_exact_match(args.mode, 'bar'):
        if len(args.filename) != 1:
            raise ValueError("Mode 'bar' expects a single input file.")
        if not args.legend_names:
            raise ValueError("Mode 'bar' requires -ln/--legend-names to label entries.")
        expected = len(df)
        if len(args.legend_names) != expected:
            raise ValueError(f"Number of legend names ({len(args.legend_names)}) must match number of entries ({expected}) after exclusions.")
        mode_parts = args.mode.split('+')
        bar_metric = mode_parts[1] if len(mode_parts) > 1 else 'total'
        plot_bar_time_metric(df, bar_metric, time_unit, args.legend_names, args.use_title, savefig, args.title)
        return

    log_x = args.log_x
    log_y = args.log_y

    # When a single file contains multiple tables, produce one set of plots
    # per table instead of merging everything into one plot.
    sources = df['source'].unique().tolist() if 'source' in df.columns else []
    split_by_table = len(args.filename) == 1 and len(sources) > 1

    plot_groups = []
    if split_by_table:
        for src in sources:
            src_df = df[df['source'] == src].copy()
            src_title = args.title or src
            src_savefig = f"{src.replace(' ', '_')}_{savefig}" if savefig else None
            plot_groups.append((src_df, src_title, src_savefig))
    else:
        plot_groups.append((df, args.title, savefig))

    for plot_df, plot_title, plot_savefig in plot_groups:
        if check_mode_exact_match(args.mode, 'iters'):
            plot_iterations(plot_df, args.cumulative, args.xtype, xlabel, args.use_title, plot_savefig, args.linestyle, args.markersize, legend_names, plot_title, args.latex_legend, show_nl_iters, log_x, log_y)

        if check_mode_exact_match(args.mode, 'times'):
            plot_times(plot_df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, plot_savefig, args.linestyle, args.markersize, legend_names, plot_title, args.latex_legend, log_x, log_y)

        if check_mode_exact_match(args.mode, 'iters-and-times'):
            plot_iters_times(plot_df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, plot_savefig, args.linestyle, args.markersize, legend_names, plot_title, args.latex_legend, show_nl_iters, log_x, log_y)

        if check_mode_exact_match(args.mode, 'setup'):
            plot_time_metric(plot_df, args.cumulative, args.xtype, xlabel, time_unit, 'setup', args.use_title, plot_savefig, args.linestyle, args.markersize, legend_names, plot_title, args.latex_legend, log_x, log_y)

        if check_mode_exact_match(args.mode, 'solve'):
            plot_time_metric(plot_df, args.cumulative, args.xtype, xlabel, time_unit, 'solve', args.use_title, plot_savefig, args.linestyle, args.markersize, legend_names, plot_title, args.latex_legend, log_x, log_y)

        if check_mode_exact_match(args.mode, 'total'):
            plot_time_metric(plot_df, args.cumulative, args.xtype, xlabel, time_unit, 'total', args.use_title, plot_savefig, args.linestyle, args.markersize, legend_names, plot_title, args.latex_legend, log_x, log_y)

        if check_mode_exact_match(args.mode, 'throughput'):
            plot_throughput(plot_df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, plot_savefig, args.linestyle, args.markersize, plot_title, log_x, log_y)

if __name__ == "__main__":
    main()
