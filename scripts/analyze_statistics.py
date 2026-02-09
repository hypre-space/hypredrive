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
from itertools import combinations

# Global variables
fgs  = (10, 6)       # Figure size
tfs  = 18            # Title font size
alfs = 14            # Axis label font size
lgfs = 14            # Legends font size

logger = logging.getLogger(__name__)

def parse_statistics_summary(filename, exclude, source_label=None):
    """
    Parse statistics from a log file.

    Args:
        filename: Path to the log file
        exclude: List of entry numbers to exclude
        source_label: Optional label to use for this source (defaults to filename)
    """
    logger.info(f"Parsing statistics from {filename = }")
    data = []
    rows = []
    nonzeros = []

    # Regular expressions to extract statistics and auxiliary data
    summary_pattern = re.compile(r"^\s*STATISTICS SUMMARY:\s*$")
    table_divider_pattern = re.compile(r"^\+-+(?:\+-+)+\+\s*$")
    rows_and_nonzeros_pattern = re.compile(
        r"Solving linear system #\d+ with (\d+) rows and (\d+) nonzeros..."
    )
    mpi_rank_pattern = re.compile(r"Running on (\d+) MPI rank[s]?")
    time_unit_pattern = re.compile(r"\s*use_millisec:\s*(\S+)")

    statistics_found = False
    in_summary_table = False
    time_unit = "[s]"
    nranks = 1
    with open(filename, 'r') as fn:
        for line in fn:
            if match := time_unit_pattern.match(line):
                if match.group(1) in ("on", "1", "true", "y", "yes"):
                    time_unit = "[ms]"
                else:
                    time_unit = "[s]"

            if mpi_rank_match := mpi_rank_pattern.match(line):
                nranks = int(mpi_rank_match.group(1))
                logger.debug(f"Detected MPI ranks: {nranks = }")
                continue

            if rows_nonzeros_match := rows_and_nonzeros_pattern.match(line):
                rows.append(int(rows_nonzeros_match.group(1)))
                nonzeros.append(int(rows_nonzeros_match.group(2)))
                continue

            if summary_pattern.match(line):
                statistics_found = True
                in_summary_table = False
                continue

            if statistics_found and table_divider_pattern.match(line):
                in_summary_table = True
                continue

            if in_summary_table and line.lstrip().startswith("|"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 9:
                    continue

                # Keep only per-entry rows; skip headers and aggregate rows
                # such as Min./Max./Avg./Std./Total.
                if not parts[1].isdigit():
                    continue

                # parts layout for data rows:
                # ['', entry, build, setup, solve, r0, rr, iters, '']
                data.append((
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                    parts[6],
                    parts[7],
                ))

    if not data:
        raise ValueError(f"Data info not found in {filename = } ")

    if not statistics_found:
        raise ValueError(f"Statistics info not found in {filename = } ")

    # Create a series for each log file entry
    series_list = []
    for i, row in enumerate(data):
        entry_num = int(row[0])
        if entry_num is not None and entry_num in exclude:
            continue

        build_time = float(row[1]) if row[1] not in (None, "") else None
        entry_data = {
            'entry': int(row[0]),
            'source': source_label if source_label is not None else filename,
            'nranks': int(nranks),
            'rows': int(rows[i]) if i < len(rows) else None,
            'nonzeros': int(nonzeros[i]) if i < len(nonzeros) else None,
            'build': build_time,
            'setup': float(row[2]),
            'solve': float(row[3]),
            'total': float(row[2]) + float(row[3]),
            'r0norm': float(row[4]),
            'resnorm': float(row[5]),
            'iters': int(row[6])
        }
        series = pd.Series(entry_data, name=f'log_{filename}_entry_{row[0]}')
        series_list.append(series)

    logger.debug(f"Parsed {len(series_list) = } entries from {filename = } (time_unit={time_unit}, nranks={nranks})")
    return series_list, time_unit

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

def plot_iterations(df, cumulative, xtype, xlabel, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None):
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
            return get_legend_name(sources[0], legend_names or {})
        if legend_names:
            return next(iter(legend_names.values()))
        return "data"

    has_nl_iters = 'nl_iters_10x' in df.columns

    if multiple_sources:
        for src in sources:
            grp = df[df['source'] == src].sort_values(by=xtype)
            y = grp['iters'].cumsum() if cumulative else grp['iters']
            ls = resolve_ls(linestyle, '-')
            legend_name = get_legend_name(src, legend_names or {})
            linear_label = f"Linear Iters. ({legend_name})" if has_nl_iters else legend_name
            plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=linear_label)
            if has_nl_iters:
                nl = grp['nl_iters_10x'].cumsum() if cumulative else grp['nl_iters_10x']
                plt.plot(grp[xtype], nl, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                         label=f"Non-linear iters. (10x) ({legend_name})")
    else:
        grp = df.sort_values(by=xtype)
        y = grp['iters'].cumsum() if cumulative else grp['iters']
        ls = resolve_ls(linestyle, '-')
        legend_name = "Linear Iters." if has_nl_iters else resolve_single_legend()
        plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=legend_name)
        if has_nl_iters:
            nl = grp['nl_iters_10x'].cumsum() if cumulative else grp['nl_iters_10x']
            plt.plot(grp[xtype], nl, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                     label="Non-linear iters. (10x)")

    plt.legend(loc="best", fontsize=lgfs)
    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver iterations vs {xlabel}', fontsize=tfs, fontweight='bold')
    plt.ylabel('Iterations', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    plt.ylim(bottom=0.0)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    # Use an integer tick locator with pruning to avoid bloated labels
    try:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    except Exception:
        pass
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"iters_{agg_str}{savefig}")

def plot_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None):
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
            legend_name = get_legend_name(src, legend_names or {})
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
    plt.tick_params(axis='x', labelsize=alfs)
    plt.tick_params(axis='y', labelsize=alfs)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"times_{agg_str}{savefig}")

def plot_time_metric(df, cumulative, xtype, xlabel, time_unit, metric, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None):
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
            legend_name = get_legend_name(src, legend_names or {})
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
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"{metric}_{agg_str}{savefig}")

def plot_iters_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None, title=None):
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
    ax1.tick_params(axis='y', labelsize=alfs)
    ax1.tick_params(axis='x', labelsize=alfs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    # Secondary Y-axis for iteration counts
    ax2 = ax1.twinx()
    ax2.set_ylabel('Iterations', fontsize=alfs)
    ax2.tick_params(axis='y', labelsize=alfs)

    lines = []
    labels = []

    # Resolve marker size
    ms = markersize if markersize is not None else plt.rcParams['lines.markersize']
    has_nl_iters = 'nl_iters_10x' in df.columns

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
            nl_iters_data = grp['nl_iters_10x'].cumsum() if (has_nl_iters and cumulative) else grp.get('nl_iters_10x')
            ls_main = resolve_ls(linestyle, '-')
            ls_iter = resolve_ls(linestyle, '--')
            legend_name = get_legend_name(src, legend_names or {})
            l1, = ax1.plot(grp[xtype], setup_data, marker='o', linestyle=ls_main, markersize=ms, label=f"Setup ({legend_name})")
            l2, = ax1.plot(grp[xtype], solve_data, marker='o', linestyle=ls_main, markersize=ms, alpha=0.7, label=f"Solve ({legend_name})")
            linear_label = f"Linear Iters. ({legend_name})" if has_nl_iters else f"Iterations ({legend_name})"
            l3, = ax2.plot(grp[xtype], iters_data, marker='o', linestyle=ls_iter, markersize=ms, label=linear_label)

            lines.extend([l1, l2, l3])
            labels.extend([l.get_label() for l in (l1, l2, l3)])
            max_iters = max(max_iters, max(iters_data) if len(iters_data) else 0)
            if has_nl_iters and nl_iters_data is not None:
                l4, = ax2.plot(grp[xtype], nl_iters_data, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                               label=f"Non-linear iters. (10x) ({legend_name})")
                lines.append(l4)
                labels.append(l4.get_label())
                if len(nl_iters_data):
                    max_iters = max(max_iters, max(nl_iters_data))

        ax2.set_ylim(bottom=0, top=max_iters * 2.0 if max_iters > 0 else 1)
    else:
        grp = df.sort_values(by=xtype)
        setup_data = grp['setup'].cumsum() if cumulative else grp['setup']
        solve_data = grp['solve'].cumsum() if cumulative else grp['solve']
        iters_data = grp['iters'].cumsum() if cumulative else grp['iters']
        nl_iters_data = grp['nl_iters_10x'].cumsum() if (has_nl_iters and cumulative) else grp.get('nl_iters_10x')

        ls_main = resolve_ls(linestyle, '-')
        ls_iter = resolve_ls(linestyle, '--')
        l1, = ax1.plot(grp[xtype], setup_data, marker='o', linestyle=ls_main, markersize=ms, color='#E69F00', label="Setup")
        l2, = ax1.plot(grp[xtype], solve_data, marker='o', linestyle=ls_main, markersize=ms, color='#009E73', label="Solve", alpha=0.5)
        linear_label = "Linear Iters." if has_nl_iters else "Iterations"
        l3, = ax2.plot(grp[xtype], iters_data, marker='o', linestyle=ls_iter, markersize=ms, color='#0072B2', label=linear_label)

        lines  = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        max_iters = max(iters_data) if len(iters_data) else 0
        if has_nl_iters and nl_iters_data is not None:
            l4, = ax2.plot(grp[xtype], nl_iters_data, marker='s', linestyle=resolve_ls(linestyle, ':'), markersize=ms,
                           color='#D55E00', label="Non-linear iters. (10x)")
            lines.append(l4)
            labels.append(l4.get_label())
            if len(nl_iters_data):
                max_iters = max(max_iters, max(nl_iters_data))
        ax2.set_ylim(bottom=0, top=max_iters * 2.0 if max_iters > 0 else 1)

    if title:
        plt.title(title, fontsize=tfs, fontweight='bold')
    elif use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver data vs {xlabel}', fontsize=tfs, fontweight='bold')

    lg = ax2.legend(lines, labels, loc="best", fontsize=lgfs)
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

def plot_throughput(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, title=None):
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
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(f"throughput_{agg_str}{savefig}")

def check_mode_exact_match(mode, word):
    # Split the mode string into parts separated by '+'
    parts = mode.split('+')

    # Check if the word exactly matches any of the parts
    return word in parts

def get_legend_name(source, legend_names):
    """Get legend name for a source, using custom mapping if available."""
    #return legend_names.get(source, str(source))
    return f"${legend_names.get(source, str(source))}$"

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
              'timestep_offset': "Timestep"}

    # List of pre-defined modes:
    mode_choices = ('iters', 'times', 'iters-and-times', 'setup', 'solve', 'total', 'throughput', 'bar')

    # Parser for plus-separated multiple modes, e.g., "setup+solve"
    def parse_modes(value):
        parts = value.split('+')
        invalid = [p for p in parts if p not in mode_choices]
        if invalid:
            raise argparse.ArgumentTypeError(f"Invalid mode(s): {', '.join(invalid)}. Valid: {', '.join(mode_choices)}")
        return value

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse the Statistics Summary produced by hypredrive")
    parser.add_argument("-f", "--filename", type=str, nargs="+", required=True, help="Path to the log file")
    parser.add_argument("-e", "--exclude", type=int, nargs="+", default=[], help="Exclude certain entries from the statistics")
    parser.add_argument("-m", "--mode", type=parse_modes, default='iters-and-times', help="What information to plot; combine multiple with '+' (e.g., 'setup+solve')")
    parser.add_argument("-t", "--xtype", type=str, default='entry', choices=labels.keys(), help="Variable type for the abscissa")
    parser.add_argument("-l", "--xlabel", type=str, default=None, help="Label for the abscissa")
    parser.add_argument("-s", "--savefig", default=None, help="Save figure(s) given this name suffix")
    parser.add_argument("-c", "--cumulative", action='store_true', help='Plot cumulative quantities')
    parser.add_argument("-u", "--use_title", action='store_true', help='Show title in plots')
    parser.add_argument("-T", "--title", type=str, default=None, help="Custom title for plots")
    parser.add_argument("-ls", "--linestyle", type=str, default='auto', choices=['auto', '-', '--', '-.', ':', 'none'], help="Line style for plots; 'none' draws markers only; 'auto' preserves defaults")
    parser.add_argument("-ms", "--markersize", type=float, default=None, help="Marker size (points); defaults to Matplotlib rcParams")
    parser.add_argument("-ln", "--legend-names", type=str, nargs="+", default=None, help="Custom legend labels for each input file (must match number of files)")
    parser.add_argument("--tsteps", type=str, default=None, help="Timesteps file mapping timestep index to starting ls id")
    parser.add_argument("--tsteps-aggregate", action="store_true",
                        help="Aggregate stats within each timestep (requires --tsteps)")
    parser.add_argument("-p", "--phase", type=str, default='total', choices=['setup', 'solve', 'total', 'iters'],
                        help="Phase for bar mode: setup, solve, total (setup+solve), or iters")
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

        series_list, time_unit = parse_statistics_summary(filename, args.exclude, source_label)
        data.extend(series_list)
    num_input_files  = len(args.filename)
    num_data_entries = len(data)
    logger.info(f"Parsed {num_input_files = }")
    logger.info(f"Found {num_data_entries = }")

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

    # Convert data types
    df = df.astype(data_types)

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
    elif args.tsteps_aggregate:
        raise ValueError("--tsteps-aggregate requires --tsteps")

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
    if check_mode_exact_match(args.mode, 'bar'):
        if args.mode != 'bar':
            raise ValueError("Mode 'bar' cannot be combined with other modes.")
        if len(args.filename) != 1:
            raise ValueError("Mode 'bar' expects a single input file.")
        if not args.legend_names:
            raise ValueError("Mode 'bar' requires -ln/--legend-names to label entries.")
        expected = len(df)
        if len(args.legend_names) != expected:
            raise ValueError(f"Number of legend names ({len(args.legend_names)}) must match number of entries ({expected}) after exclusions.")
        plot_bar_time_metric(df, args.phase, time_unit, args.legend_names, args.use_title, savefig, args.title)
        return

    if check_mode_exact_match(args.mode, 'iters'):
        plot_iterations(df, args.cumulative, args.xtype, xlabel, args.use_title, savefig, args.linestyle, args.markersize, legend_names, args.title)

    if check_mode_exact_match(args.mode, 'times'):
        plot_times(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig, args.linestyle, args.markersize, legend_names, args.title)

    if check_mode_exact_match(args.mode, 'iters-and-times'):
        plot_iters_times(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig, args.linestyle, args.markersize, legend_names, args.title)

    if check_mode_exact_match(args.mode, 'setup'):
        plot_time_metric(df, args.cumulative, args.xtype, xlabel, time_unit, 'setup', args.use_title, savefig, args.linestyle, args.markersize, legend_names, args.title)

    if check_mode_exact_match(args.mode, 'solve'):
        plot_time_metric(df, args.cumulative, args.xtype, xlabel, time_unit, 'solve', args.use_title, savefig, args.linestyle, args.markersize, legend_names, args.title)

    if check_mode_exact_match(args.mode, 'total'):
        plot_time_metric(df, args.cumulative, args.xtype, xlabel, time_unit, 'total', args.use_title, savefig, args.linestyle, args.markersize, legend_names, args.title)

    if check_mode_exact_match(args.mode, 'throughput'):
        plot_throughput(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig, args.linestyle, args.markersize, args.title)

if __name__ == "__main__":
    main()
