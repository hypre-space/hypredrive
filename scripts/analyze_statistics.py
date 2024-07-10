#!/usr/bin/python
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from itertools import combinations

# Global variables
fgs  = (10, 6)       # Figure size
tfs  = 18            # Title font size
alfs = 14            # Axis label font size
lgfs = 14            # Legends font size

def parse_statistics_summary(filename, exclude):
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
    mpi_rank_pattern = re.compile(r"Running on (\d+) MPI rank")
    time_unit_pattern = re.compile(r"\s*use_millisec:\s*(\S+)")

    statistics_found = False
    time_unit = "[s]"
    with open(filename, 'r') as fn:
        for line in fn:
            if match := time_unit_pattern.match(line):
                if match.group(1) in ("on", "1", "true", "y", "yes"):
                    time_unit = "[ms]"
                else:
                    time_unit = "[s]"

            if mpi_rank_match := mpi_rank_pattern.match(line):
                nranks = int(mpi_rank_match.group(1))
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

        print(f"- Saving figure: {savefig}...")
        plt.savefig(savefig, dpi=dpi)  # Save the figure with the appropriate DPI

    # Always display the plot regardless of saving
    plt.show()
    plt.close()

def plot_iterations(df, cumulative, xtype, xlabel, use_title=False, savefig=None):
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

    # Determine data to plot, possibly as cumulative sums
    iters_data = df['iters'].cumsum() if cumulative else df['iters']
    agg_str = "agg_" if cumulative else ''

    # Plot figure
    plt.figure(figsize=fgs)
    plt.plot(df[xtype], iters_data, marker='o', linestyle='-')
    if use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver iterations vs {xlabel}', fontsize=tfs, fontweight='bold')
    plt.ylabel('Iterations', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    plt.tick_params(axis='x', labelsize=alfs)
    plt.tick_params(axis='y', labelsize=alfs)
    plt.grid(True)
    plt.xticks(df[xtype], fontsize=alfs)
    plt.tight_layout()
    save_and_show_plot(f"iters_{agg_str}{savefig}")

def plot_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None):
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

    # Determine data to plot, possibly as cumulative sums
    setup_data = df['setup'].cumsum() if cumulative else df['setup']
    solve_data = df['solve'].cumsum() if cumulative else df['solve']
    total_data = df['total'].cumsum() if cumulative else df['total']
    agg_str = "agg_" if cumulative else ''

    # Plot figure
    plt.figure(figsize=fgs)
    plt.plot(df[xtype], setup_data, marker='o', linestyle='-', label="Setup")
    plt.plot(df[xtype], solve_data, marker='o', linestyle='-', label="Solve")
    plt.plot(df[xtype], total_data, marker='o', linestyle='-', label="Total")
    if use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver times vs {xlabel}', fontsize=tfs, fontweight='bold')
    plt.ylabel(f'Times {time_unit}', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    plt.tick_params(axis='x', labelsize=alfs)
    plt.tick_params(axis='y', labelsize=alfs)
    plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.legend(loc="best", fontsize=lgfs)
    plt.tight_layout()
    save_and_show_plot(f"times_{agg_str}{savefig}")

def plot_iters_times(df, cumulative, xtype, xlabel, time_unit, use_title=False, savefig=None):
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
    fig, ax1 = plt.figure(figsize=fgs), plt.gca()

    # Determine data to plot, possibly as cumulative sums
    setup_data = df['setup'].cumsum() if cumulative else df['setup']
    solve_data = df['solve'].cumsum() if cumulative else df['solve']
    iters_data = df['iters'].cumsum() if cumulative else df['iters']
    agg_str = "agg_" if cumulative else ''

    # Plot setup and solve times on the primary Y-axis
    ax1.set_xlabel(xlabel, fontsize=alfs)
    ax1.set_ylabel(f'Times {time_unit}', fontsize=alfs)
    l1, = ax1.plot(df[xtype], setup_data, marker='o', linestyle='-', color='#E69F00', label="Setup")
    l2, = ax1.plot(df[xtype], solve_data, marker='o', linestyle='-', color='#009E73', label="Solve", alpha=0.5)
    ax1.tick_params(axis='y', labelsize=alfs)
    ax1.tick_params(axis='x', labelsize=alfs)

    # Create a secondary Y-axis for iteration counts
    ax2 = ax1.twinx()
    ax2.set_ylabel('Iterations', fontsize=alfs)
    l3, = ax2.plot(df[xtype], iters_data, marker='o', linestyle='--', color='#0072B2', label="Iterations")
    ax2.tick_params(axis='y', labelsize=alfs)
    ax2.set_ylim(bottom=0, top=max(iters_data)*2.0)
    if use_title:
        prefix = 'Cumulative ' if cumulative else ''
        plt.title(f'{prefix}Linear solver data vs {xlabel}', fontsize=tfs, fontweight='bold')

    # Combine all the lines for the legend
    lines  = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
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

def main():
    # List of pre-defined labels
    labels = {'rows': "Number of rows",
              'nonzeros': "Number of nonzeros",
              'entry': "Linear system ID",
              'nranks': "Number of MPI ranks"}

    # List of pre-defined modes:
    modes = ('iters', 'times', 'iters-and-times')
    mode_choices = tuple('+'.join(comb) for i in range(1, len(modes) + 1) for comb in combinations(modes, i))

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse the Statistics Summary produced by hypredrive")
    parser.add_argument("-f", "--filename", type=str, nargs="+", required=True, help="Path to the log file")
    parser.add_argument("-e", "--exclude", type=int, nargs="+", default=[], help="Exclude certain entries from the statistics")
    parser.add_argument("-m", "--mode", type=str, default='iters-and-times', choices=mode_choices, help="What information to plot")
    parser.add_argument("-t", "--xtype", type=str, default='entry', choices=labels.keys(), help="Variable type for the abscissa")
    parser.add_argument("-l", "--xlabel", type=str, default=None, help="Label for the abscissa")
    parser.add_argument("-s", "--savefig", default=None, help="Save figure(s) given this name suffix")
    parser.add_argument("-c", "--cumulative", action='store_true', help='Plot cumulative quantities')
    parser.add_argument("-u", "--use_title", action='store_true', help='Show title in plots')
    parser.add_argument("-v", "--verbose", action='store_true', help='Print dataframe contents')

    # Parse arguments
    args = parser.parse_args()

    # Parse the statistics summary
    data = []
    for filename in args.filename:
        series_list, time_unit = parse_statistics_summary(filename, args.exclude)
        data.extend(series_list)
    num_input_files  = len(args.filename)
    num_data_entries = len(data)
    print(f"- Parsed {num_input_files = }")
    print(f"- Found {num_data_entries = }")

    # Assemble all series into a single DataFrame
    df = pd.concat(data, axis=1).T.reset_index(drop=True)

    # Explicitly specify data types for each column
    data_types = {
        'entry':    'int',
        'nranks':   'int',
        'rows':     'int',
        'nonzeros': 'int',
        'build':    'float',
        'setup':    'float',
        'solve':    'float',
        'total':    'float',
        'resnorm':  'float',
        'iters':    'int'
    }

    # Convert data types
    df = df.astype(data_types)

    # Update label
    xlabel = args.xlabel if args.xlabel else labels[args.xtype]

    # Show DataFrame?
    if args.verbose:
        print(df)
        print(f"Sum total time: {sum(df['total']) = }")

    # Update savefig string
    savefig = args.savefig if args.savefig != "." else f"{(args.filename)[0].split('.')[0]}.png"

    # Produce plots
    if check_mode_exact_match(args.mode, 'iters'):
        plot_iterations(df, args.cumulative, args.xtype, xlabel, args.use_title, savefig)

    if check_mode_exact_match(args.mode, 'times'):
        plot_times(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig)

    if check_mode_exact_match(args.mode, 'iters-and-times'):
        plot_iters_times(df, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig)

if __name__ == "__main__":
    main()
