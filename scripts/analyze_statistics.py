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

def parse_statistics_summary(filename, exclude):
    logger.info(f"Parsing statistics from {filename = }")
    # Initialize an empty string to hold the current section being processed
    target_section = ""
    data = []
    rows = []
    nonzeros = []

    # Regular expressions to extract statistics and auxiliary data
    start_pattern = re.compile(r"\+\-+\+\-+\+\-+\+\-+\+\-+\+\-+\+")
    end_pattern   = re.compile(r"\+\-+\+\-+\+\-+\+\-+\+\-+\+\-+\+")
    data_pattern  = re.compile(
        r"\|\s+(\d+)\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+e[+-]\d+)\s+\|\s+(\d+)\s+\|"
    )
    rows_and_nonzeros_pattern = re.compile(
        r"Solving linear system #\d+ with (\d+) rows and (\d+) nonzeros..."
    )
    mpi_rank_pattern = re.compile(r"Running on (\d+) MPI rank[s]?")
    time_unit_pattern = re.compile(r"\s*use_millisec:\s*(\S+)")

    statistics_found = False
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

            if start_pattern.match(line):
                statistics_found = True
                target_section = line
                continue

            elif target_section and end_pattern.match(line):
                break  # End of the statistics summary table

            elif target_section:
                data.extend(data_pattern.findall(line))

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

        entry_data = {
            'entry': int(row[0]),
            'source': os.path.basename(filename),
            'nranks': int(nranks),
            'rows': int(rows[i]) if i < len(rows) else None,
            'nonzeros': int(nonzeros[i]) if i < len(nonzeros) else None,
            'build': float(row[1]),
            'setup': float(row[2]),
            'solve': float(row[3]),
            'total': float(row[2]) + float(row[3]),
            'resnorm': float(row[4]),
            'iters': int(row[5])
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

def plot_iterations(df, cumulative, xtype, xlabel, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None):
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

    if multiple_sources:
        for src in sources:
            grp = df[df['source'] == src].sort_values(by=xtype)
            y = grp['iters'].cumsum() if cumulative else grp['iters']
            ls = resolve_ls(linestyle, '-')
            legend_name = get_legend_name(src, legend_names or {})
            plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=legend_name)
    else:
        grp = df.sort_values(by=xtype)
        y = grp['iters'].cumsum() if cumulative else grp['iters']
        ls = resolve_ls(linestyle, '-')
        legend_name = legend_names[sources[0]]
        plt.plot(grp[xtype], y, marker='o', linestyle=ls, markersize=ms, label=legend_name)

    plt.legend(loc="best", fontsize=lgfs)
    if use_title:
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

def plot_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None):
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

    if use_title:
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

def plot_time_metric(df, cumulative, xtype, xlabel, time_unit, metric, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None):
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

    if use_title:
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

def plot_iters_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None, linestyle='auto', markersize=None, legend_names=None):
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
            ls_main = resolve_ls(linestyle, '-')
            ls_iter = resolve_ls(linestyle, '--')
            legend_name = get_legend_name(src, legend_names or {})
            l1, = ax1.plot(grp[xtype], setup_data, marker='o', linestyle=ls_main, markersize=ms, label=f"Setup ({legend_name})")
            l2, = ax1.plot(grp[xtype], solve_data, marker='o', linestyle=ls_main, markersize=ms, alpha=0.7, label=f"Solve ({legend_name})")
            l3, = ax2.plot(grp[xtype], iters_data, marker='o', linestyle=ls_iter, markersize=ms, label=f"Iterations ({legend_name})")

            lines.extend([l1, l2, l3])
            labels.extend([l.get_label() for l in (l1, l2, l3)])
            max_iters = max(max_iters, max(iters_data) if len(iters_data) else 0)

        ax2.set_ylim(bottom=0, top=max_iters * 2.0 if max_iters > 0 else 1)
    else:
        grp = df.sort_values(by=xtype)
        setup_data = grp['setup'].cumsum() if cumulative else grp['setup']
        solve_data = grp['solve'].cumsum() if cumulative else grp['solve']
        iters_data = grp['iters'].cumsum() if cumulative else grp['iters']

        ls_main = resolve_ls(linestyle, '-')
        ls_iter = resolve_ls(linestyle, '--')
        l1, = ax1.plot(grp[xtype], setup_data, marker='o', linestyle=ls_main, markersize=ms, color='#E69F00', label="Setup")
        l2, = ax1.plot(grp[xtype], solve_data, marker='o', linestyle=ls_main, markersize=ms, color='#009E73', label="Solve", alpha=0.5)
        l3, = ax2.plot(grp[xtype], iters_data, marker='o', linestyle=ls_iter, markersize=ms, color='#0072B2', label="Iterations")

        lines  = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        ax2.set_ylim(bottom=0, top=max(iters_data)*2.0 if len(iters_data) else 1)

    if use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver data vs {xlabel}', fontsize=tfs, fontweight='bold')

    lg = ax2.legend(lines, labels, loc="best", fontsize=lgfs)
    lg.set_zorder(100)

    fig.tight_layout()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, zorder=0)
    save_and_show_plot(f"iters_times_{agg_str}{savefig}")

def check_mode_exact_match(mode, word):
    # Split the mode string into parts separated by '+'
    parts = mode.split('+')

    # Check if the word exactly matches any of the parts
    return word in parts

def get_legend_name(source, legend_names):
    """Get legend name for a source, using custom mapping if available."""
    #return legend_names.get(source, str(source))
    return f"${legend_names.get(source, str(source))}$"

def main():
    # List of pre-defined labels
    labels = {'rows': "Number of rows",
              'nonzeros': "Number of nonzeros",
              'entry': "Linear system number",
              'nranks': "Number of MPI ranks"}

    # List of pre-defined modes:
    mode_choices = ('iters', 'times', 'iters-and-times', 'setup', 'solve', 'total')

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
    parser.add_argument("-ls", "--linestyle", type=str, default='auto', choices=['auto', '-', '--', '-.', ':', 'none'], help="Line style for plots; 'none' draws markers only; 'auto' preserves defaults")
    parser.add_argument("-ms", "--markersize", type=float, default=None, help="Marker size (points); defaults to Matplotlib rcParams")
    parser.add_argument("-ll", "--legend-labels", type=str, nargs="+", default=None, help="Custom legend labels for each input file (must match number of files)")
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

    # Create label mapping from source filenames to custom labels
    label_map = {}
    if args.legend_labels:
        if len(args.legend_labels) != len(args.filename):
            raise ValueError(f"Number of legend labels ({len(args.legend_labels)}) must match number of files ({len(args.filename)})")
        for filename, label in zip(args.filename, args.legend_labels):
            source_key = os.path.basename(filename)
            label_map[source_key] = label
        logger.debug(f"Label mapping: {label_map = }")

    # Parse the statistics summary
    data = []
    for filename in args.filename:
        series_list, time_unit = parse_statistics_summary(filename, args.exclude)
        data.extend(series_list)
    num_input_files  = len(args.filename)
    num_data_entries = len(data)
    logger.info(f"Parsed {num_input_files = }")
    logger.info(f"Found {num_data_entries = }")

    # Assemble all series into a single DataFrame
    df = pd.concat(data, axis=1).T.reset_index(drop=True)

    # Apply custom labels to the DataFrame if provided
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
        'resnorm':  'float',
        'iters':    'int'
    }

    # Convert data types
    df = df.astype(data_types)

    # Create legend name mapping
    legend_names = {}
    if args.names:
        if len(args.names) != len(args.filename):
            raise ValueError(f"Number of legend names ({len(args.names)}) must match number of input files ({len(args.filename)})")
        # Map source (basename of filename) to custom legend name
        for filename, legend_name in zip(args.filename, args.names):
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
    if check_mode_exact_match(args.mode, 'iters'):
        plot_iterations(df, args.cumulative, args.xtype, xlabel, args.use_title, savefig, args.linestyle, args.markersize, legend_names)

    if check_mode_exact_match(args.mode, 'times'):
        plot_times(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig, args.linestyle, args.markersize, legend_names)

    if check_mode_exact_match(args.mode, 'iters-and-times'):
        plot_iters_times(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig, args.linestyle, args.markersize, legend_names)

    if check_mode_exact_match(args.mode, 'setup'):
        plot_time_metric(df, args.cumulative, args.xtype, xlabel, time_unit, 'setup', args.use_title, savefig, args.linestyle, args.markersize, legend_names)

    if check_mode_exact_match(args.mode, 'solve'):
        plot_time_metric(df, args.cumulative, args.xtype, xlabel, time_unit, 'solve', args.use_title, savefig, args.linestyle, args.markersize, legend_names)

    if check_mode_exact_match(args.mode, 'total'):
        plot_time_metric(df, args.cumulative, args.xtype, xlabel, time_unit, 'total', args.use_title, savefig, args.linestyle, args.markersize, legend_names)

if __name__ == "__main__":
    main()
