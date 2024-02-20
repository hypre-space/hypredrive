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

    statistics_found = False
    with open(filename, 'r') as fn:
        for line in fn:
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

    return series_list

def plot_iterations(df, xtype, xlabel, use_title=False):
    """
    Plots iteration counts as a function of a specified column in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data. It must include columns specified by 'xtype' and 'iters'.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - use_title (boolean, optional): Turn on figure's title.

    Globals:
    - fgs (tuple): Figure size for the plot.
    - tfs (int): Font size for the title.
    - alfs (int): Font size for the axis labels.

    The function does not return anything but displays a plot.
    """
    plt.figure(figsize=fgs)
    plt.plot(df[xtype], df['iters'], marker='o', linestyle='-')
    if use_title:
        plt.title('Linear solver iterations vs ' + xlabel, fontsize=tfs, fontweight='bold')
    plt.ylabel('Iterations', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    plt.grid(True)
    plt.xticks(df[xtype], fontsize=alfs)
    plt.tight_layout()
    plt.show()

def plot_times(df, xtype, xlabel, use_title=False):
    """
    Plots setup and solve times, as well as their total, as a function of a specified column in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data with 'setup' and 'solve' times among its columns.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - use_title (boolean, optional): Turn on figure's title.

    Globals:
    - fgs (tuple): Figure size for the plot.
    - tfs (int): Font size for the title.
    - alfs (int): Font size for the axis labels.
    - lgfs (int): Font size for the legend.

    The plot includes three lines representing the 'setup' time, 'solve' time, and their total time for each entry in
    the DataFrame, as determined by the xtype column. The function does not return anything but displays the plot.
    """
    plt.figure(figsize=fgs)
    plt.plot(df[xtype], df['setup'], marker='o', linestyle='-', label="Setup")
    plt.plot(df[xtype], df['solve'], marker='o', linestyle='-', label="Solve")
    plt.plot(df[xtype], df['total'], marker='o', linestyle='-', label="Total")
    if use_title:
        plt.title('Linear solver times vs ' + xlabel, fontsize=tfs, fontweight='bold')
    plt.ylabel('Times [s]', fontsize=alfs)
    plt.xlabel(xlabel, fontsize=alfs)
    plt.grid(True)
    plt.legend(loc="best", fontsize=lgfs)
    plt.tight_layout()
    plt.show()

def plot_iters_times(df, xtype, xlabel, use_title=False):
    """
    Plots setup and solve times, as well as iteration counts, as a function of a specified column in the DataFrame.
    Setup and solve times are plotted on the primary Y-axis, while iteration counts are plotted on a secondary Y-axis.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the log data with 'setup', 'solve', and 'iters' among its columns.
    - xtype (str): Column name in 'df' to use as the x-axis for the plot.
    - xlabel (str): Label for the x-axis.
    - use_title (str, optional): Custom title for the figure. If not provided, the figure will have no title.

    Globals:
    - fgs (tuple): Figure size for the plot. Must be defined elsewhere in the global scope.
    - tfs (int): Font size for the title. Must be defined elsewhere in the global scope.
    - alfs (int): Font size for the axis labels. Must be defined elsewhere in the global scope.
    - lgfs (int): Font size for the legend. Must be defined elsewhere in the global scope.

    The plot includes lines representing the 'setup' and 'solve' times on the primary Y-axis, and 'iters' on the secondary Y-axis.
    The function does not return anything but displays the plot.
    """
    fig, ax1 = plt.figure(figsize=fgs), plt.gca()

    # Plot setup and solve times on the primary Y-axis
    color = 'tab:blue'
    ax1.set_xlabel(xlabel, fontsize=alfs)
    ax1.set_ylabel('Times [s]', fontsize=alfs, color=color)
    l1, = ax1.plot(df[xtype], df['setup'], marker='o', linestyle='-', color=color, label="Setup")
    l2, = ax1.plot(df[xtype], df['solve'], marker='o', linestyle='-', color=color, label="Solve", alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=alfs)
    ax1.tick_params(axis='x', labelsize=alfs)

    # Create a secondary Y-axis for iteration counts
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Iterations', fontsize=alfs, color=color)
    l3, = ax2.plot(df[xtype], df['iters'], marker='o', linestyle='--', color=color, label="Iterations")
    ax2.tick_params(axis='y', labelcolor=color, labelsize=alfs)
    if use_title:
        plt.title('Linear solver data vs ' + xlabel, fontsize=tfs, fontweight='bold')

    # Combine all the lines for the legend
    lines  = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    lg = ax2.legend(lines, labels, loc="best", fontsize=lgfs)
    lg.set_zorder(100)

    fig.tight_layout()
    plt.grid(True, zorder=2)
    plt.show()

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
    modes = ('iters', 'timers', 'iters-and-times')
    mode_choices = tuple('+'.join(comb) for i in range(1, len(modes) + 1) for comb in combinations(modes, i))

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse the Statistics Summary produced by hypredrive")
    parser.add_argument("-f", "--filename", type=str, nargs="+", required=True, help="Path to the log file")
    parser.add_argument("-e", "--exclude", type=int, nargs="+", default=[], help="Exclude certain entries from the statistics")
    parser.add_argument("-m", "--mode", type=str, default='iters-and-times', choices=mode_choices, help="What information to plot")
    parser.add_argument("-t", "--xtype", type=str, default='entry', choices=labels.keys(), help="Variable type for the abscissa")
    parser.add_argument("-l", "--xlabel", type=str, default=None, help="Label for the abscissa")
    parser.add_argument("-u", "--use_title", action='store_true', help='Show title in plots')
    parser.add_argument("-v", "--verbose", action='store_true', help='Print dataframe contents')

    # Parse arguments
    args = parser.parse_args()

    # Parse the statistics summary
    data = []
    for filename in args.filename:
        series_list = parse_statistics_summary(filename, args.exclude)
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

    # Produce plots
    if check_mode_exact_match(args.mode, 'iters'):
        plot_iterations(df, args.xtype, xlabel, args.use_title)

    if check_mode_exact_match(args.mode, 'times'):
        plot_times(df, args.xtype, xlabel, args.use_title)

    if check_mode_exact_match(args.mode, 'iters-and-times'):
        plot_iters_times(df, args.xtype, xlabel, args.use_title)

if __name__ == "__main__":
    main()
