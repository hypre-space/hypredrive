#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

"""
Analyze Statistics - Hypredrive Solver Performance Analysis Tool

This script analyzes output files from runs using hypredrive,
extracting convergence data and performance metrics to help understand solver behavior.

The script provides functionality to:
1. Parse convergence history data from HYPRE solver output
2. Extract iteration counts, residual norms, and convergence rates
3. Visualize convergence behavior through semi-log plots
4. Compare convergence across multiple linear systems and/or simulation runs

The primary focus is tracking how quickly iterative solvers like CG, GMRES, or BiCGSTAB
converge to a solution, allowing researchers to optimize solver settings, preconditioners,
and numerical methods for better performance.

Usage:
    ./analyze_statistics.py -f <logfile> -m conv-hist [--savefig <output.png>]
    ./analyze_statistics.py -d <directory> -m conv-hist [--savefig <output.png>]

Examples:
    # Plot convergence for a single file
    ./analyze_statistics.py -f log.out -m conv-hist

    # Plot convergence for all .out files in a directory
    ./analyze_statistics.py -d results/ -m conv-hist --savefig convergence.png

    # Exclude specific linear systems from analysis
    ./analyze_statistics.py -f simulation.out -m conv-hist -e 0 1
"""

import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from itertools import combinations
import os # For os.path.basename
import numpy as np # For np.nan
import glob

# Global variables
fgs  = (10, 6)       # Figure size
tfs  = 18            # Title font size
alfs = 14            # Axis label font size
lgfs = 14            # Legends font size

def parse_statistics_summary(filename, exclude, debug=True):
    """
    Parses the STATISTICS SUMMARY section and related system info from a log file.
    Returns a list of pandas Series, each representing an entry, and the time unit.
    """
    system_details = {} # Key: entry_id, Value: {'rows': ..., 'nonzeros': ...}

    # Regular expressions
    mpi_rank_pattern = re.compile(r"Running on (\\d+) MPI rank")
    time_unit_pattern = re.compile(r"\\s*use_millisec:\\s*(\\S+)")
    system_info_pattern = re.compile(r"Solving linear system #(\\d+) with (\\d+) rows and (\\d+) nonzeros...")
    stats_summary_start_pattern = re.compile(r"STATISTICS SUMMARY:")
    stats_table_boundary_pattern = re.compile(r"^\+-+\+-+\+-+\+-+\+-+\+-+\+$")
    stats_data_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*(\d+)\s*\|"
    )

    time_unit = "[s]" # Default
    nranks = 1 # Default
    statistics_found_in_summary = False

    if debug:
        print(f"\nProcessing file: {filename}")

    with open(filename, 'r') as fn:
        lines = fn.readlines()

    if debug:
        print(f"Found {len(lines)} lines")

    # First pass: get nranks, time_unit, and system details (rows, nonzeros)
    for line in lines:
        if match := time_unit_pattern.match(line):
            if match.group(1).lower() in ("on", "1", "true", "y", "yes"):
                time_unit = "[ms]"
                if debug:
                    print(f"Using milliseconds as time unit")

        if mpi_rank_match := mpi_rank_pattern.match(line):
            nranks = int(mpi_rank_match.group(1))
            if debug:
                print(f"Running on {nranks} MPI ranks")

        if rows_nonzeros_match := system_info_pattern.match(line):
            entry_id = int(rows_nonzeros_match.group(1))
            rows_val = int(rows_nonzeros_match.group(2))
            nonzeros_val = int(rows_nonzeros_match.group(3))
            system_details[entry_id] = {'rows': rows_val, 'nonzeros': nonzeros_val}
            if debug:
                print(f"Found system #{entry_id}: {rows_val} rows, {nonzeros_val} nonzeros")

    series_list = []

    # Second pass (or continuation): parse the statistics summary table
    in_summary_section = False
    in_table_data = False
    boundary_count = 0  # Count the number of boundary lines seen
    for line_idx, line_content in enumerate(lines): # Use a different name for line content
        if stats_summary_start_pattern.search(line_content):
            in_summary_section = True
            if debug:
                print(f"\nFound statistics summary at line {line_idx+1}")
            continue

        if in_summary_section:
            if stats_table_boundary_pattern.match(line_content.strip()):
                boundary_count += 1
                if boundary_count == 1:  # First boundary - start of table
                    in_table_data = True
                    if debug:
                        print("Found table header boundary")
                elif boundary_count == 3:  # Third boundary - end of table
                    if debug:
                        print("Found table end boundary")
                    break
                continue

            if in_table_data:
                if stat_match := stats_data_pattern.match(line_content.strip()):
                    statistics_found_in_summary = True
                    row_tuple = stat_match.groups() # Use a different name from the loop variable
                    entry_num = int(row_tuple[0])

                    if entry_num in exclude:
                        if debug:
                            print(f"Skipping excluded entry {entry_num}")
                        continue

                    detail = system_details.get(entry_num, {'rows': np.nan, 'nonzeros': np.nan})

                    entry_data = {
                        'entry': entry_num,
                        'nranks': nranks,
                        'rows': detail['rows'],
                        'nonzeros': detail['nonzeros'],
                        'build': float(row_tuple[1]),
                        'setup': float(row_tuple[2]),
                        'solve': float(row_tuple[3]),
                        'total': float(row_tuple[2]) + float(row_tuple[3]),
                        'resnorm': float(row_tuple[4]),
                        'iters': int(row_tuple[5]),
                        'filename': filename
                    }
                    if debug:
                        print(f"Processing entry {entry_num}:")
                        print(f"  Build: {entry_data['build']} {time_unit}")
                        print(f"  Setup: {entry_data['setup']} {time_unit}")
                        print(f"  Solve: {entry_data['solve']} {time_unit}")
                        print(f"  Total: {entry_data['total']} {time_unit}")
                        print(f"  Resnorm: {entry_data['resnorm']}")
                        print(f"  Iters: {entry_data['iters']}")

                    series_list.append(pd.Series(entry_data, name=f'log_{filename}_entry_{entry_num}'))

    if not series_list and not statistics_found_in_summary:
        if not system_details:
             # Keep previous error only if nothing at all was found
            pass # Allow returning empty list if no summary, but systems were found
        elif not statistics_found_in_summary:
            print(f"Warning: Statistics summary table not found or empty in {filename}. No summary data generated for this file.")
            # No raise here, let create_main_dataframe handle if all files are like this

    if not series_list and not system_details: # If truly nothing useful
         raise ValueError(f"No usable data (system info or statistics summary) found in {filename}")

    if debug:
        print(f"\nSummary for {filename}:")
        print(f"  Time unit: {time_unit}")
        print(f"  MPI ranks: {nranks}")
        print(f"  Systems found: {len(system_details)}")
        print(f"  Statistics entries: {len(series_list)}")

    return series_list, time_unit

def parse_convergence_history_for_file(filename, exclude_entries, debug=False):
    """
    Parses HYPRE solver convergence histories from a log file, extracting iteration data.

    This function extracts convergence history data from iterative solver logs, particularly
    HYPRE output. It identifies each linear system solve block in the file, extracts the
    convergence data for each one, and creates a DataFrame of iterations vs. residuals.

    The function handles two types of convergence tables:

    1. Standard format with 4 columns:
    ```
      Iters     resid.norm     conv.rate  rel.res.norm
      -----    ------------    ---------- ------------
          1    9.958407e+01    1.000000   1.000000e+00
    ```

    2. Extended format with multiple residual/error columns:
    ```
      Iters       |r|_2/|b|_2     |r0|_2/|b|_2     |r1|_2/|b|_2     |r2|_2/|b|_2
     ------     -------------    -------------    -------------    -------------
          1      6.602754e-01     3.665768e-04     2.610796e-04     4.094180e-08
    ```

    Parameters
    ----------
    filename : str
        Path to the log file containing convergence history data
    exclude_entries : list of int
        List of entry IDs (linear system numbers) to exclude from processing

    Returns
    -------
    list of dict
        List of dictionaries, each containing:
        - 'filename': source file path
        - 'entry_id': linear system ID (from "Solving linear system #X")
        - 'history_df': pandas DataFrame with columns:
          * 'iter': iteration number
          * 'rel_res_norm' or 'rel_error_norm': relative residual/error norm
          * 'resid_norm': absolute residual norm (if available)
          * 'conv_rate': convergence rate (if available)
          * Additional columns 'r0', 'r1', 'r2', etc. (if present in extended format)
        - 'format_type': 'standard' or 'extended'
    """
    # Regular expression patterns to identify key parts of the convergence history

    # Pattern to match the line that indicates the start of a linear system solve
    system_info_pattern = re.compile(r"Solving linear system #(\d+) with (\d+) rows and (\d+) nonzeros...")

    # Pattern to match the standard header row of the convergence table:
    # "Iters     resid.norm     conv.rate  rel.res.norm"
    standard_header_pattern = re.compile(r"^\s*Iters\s+resid\.norm\s+conv\.rate\s+rel\.res\.norm")

    # Pattern to match the extended header row with multiple
    # relative residual columns: "Iters |r|_2/|b|_2 |r0|_2/|b|_2 |r1|_2/|b|_2 ..."
    # Or relative error columns: "Iters |e|_2/|eI|_2 |e0|_2/|eI0|_2 |e1|_2/|eI1|_2 ..."
    # Or absolute error norms:   "Iters ||e||_2 ||e0||_2 ||e1||_2 ..."
    extended_header_pattern = re.compile(r"^\s*Iters\s+(?:\|(r|e)\|_2/\|(b|eI)\|_2|\|(r|e)\|_2)")

    # Pattern to match extended column headers for additional residual or error norms
    # Now handles both relative and absolute norms
    residual_col_pattern = re.compile(r"(?:\|(r|e)(\d+)\|_2/\|(b|eI)\d*\|_2|\|(r|e)(\d+)\|_2)")

    # Pattern to match data rows in the standard table format
    standard_data_pattern = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)")

    # Patterns to identify the end of a convergence history section
    final_l2_norm_pattern = re.compile(r"Final L2 norm of residual:")
    stats_summary_section_pattern = re.compile(r"STATISTICS SUMMARY:")

    file_histories = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error opening file {filename}: {e}")
        return file_histories

    if debug:
        print(f"Processing file: {filename} - Found {len(lines)} lines")

    # State tracking variables
    active_entry_id = None          # Current linear system ID being processed
    current_history_data = []       # Collected data points for current system
    parsing_this_entry_history = False  # True when we're inside a convergence history section
    found_conv_header = False       # True when we've found the convergence table header
    format_type = "standard"        # Type of convergence table format: "standard" or "extended"
    column_names = []               # Column names for extended format

    # Process the file line by line
    for line_idx, line in enumerate(lines):
        line_strip = line.strip()

        # Check for a new linear system section
        if system_match := system_info_pattern.search(line):
            # If we were parsing a previous system, save its data before starting a new one
            if active_entry_id is not None and parsing_this_entry_history and current_history_data:
                if active_entry_id not in exclude_entries:
                    if debug:
                        print(f"Saving history for entry {active_entry_id} with {len(current_history_data)} points")
                    history_df = pd.DataFrame(current_history_data)

                    # Create iter=0 row with appropriate initial values
                    iter0_row = pd.DataFrame([{
                        'iter': 0,
                    }])

                    # Add appropriate norm value for either residual or error
                    if 'rel_res_norm' in history_df.columns:
                        iter0_row['rel_res_norm'] = 1.0
                    elif 'rel_error_norm' in history_df.columns:
                        iter0_row['rel_error_norm'] = 1.0
                    elif 'abs_res_norm' in history_df.columns:
                        iter0_row['abs_res_norm'] = history_df['abs_res_norm'].iloc[0] if not history_df.empty else 1.0
                    elif 'abs_error_norm' in history_df.columns:
                        iter0_row['abs_error_norm'] = history_df['abs_error_norm'].iloc[0] if not history_df.empty else 1.0

                    # Add values for other columns if they exist
                    if format_type == "standard" and 'resid_norm' in history_df.columns:
                        iter0_row['resid_norm'] = history_df['resid_norm'].iloc[0] if not history_df.empty else 1.0
                        iter0_row['conv_rate'] = 1.0  # Convention for first iteration
                    elif format_type == "extended":
                        # Set all r0/e0, r1/e1, etc values to initial values for iteration 0
                        for col in history_df.columns:
                            if (col.startswith('r') or col.startswith('e') or col.startswith('abs_')) and \
                               col not in ['rel_res_norm', 'rel_error_norm', 'abs_res_norm', 'abs_error_norm'] and \
                               col not in iter0_row:
                                # For absolute norms, use the first value, for relative norms use 1.0
                                if col.startswith('abs_'):
                                    iter0_row[col] = history_df[col].iloc[0] if not history_df.empty else 1.0
                                else:
                                    iter0_row[col] = 1.0  # Always start at 1.0 for relative metrics

                    # Concatenate and sort
                    history_df = pd.concat([iter0_row, history_df]).sort_values('iter').reset_index(drop=True)

                    if not history_df.empty:
                        file_histories.append({
                            'filename': filename,
                            'entry_id': active_entry_id,
                            'history_df': history_df,
                            'format_type': format_type
                        })

            # Extract system ID and set up for parsing the new system
            active_entry_id = int(system_match.group(1))
            if debug:
                print(f"Set active_entry_id = {active_entry_id}")
            current_history_data = []
            parsing_this_entry_history = False
            found_conv_header = False
            format_type = "standard"  # Reset for each new system
            column_names = []
            continue

        # Only process lines if we're inside a system block
        if active_entry_id is not None:
            # Look for the standard convergence table header
            if standard_header_pattern.search(line):
                if debug:
                    print(f"Found standard convergence header at line {line_idx+1}: {line_strip}")
                parsing_this_entry_history = True
                found_conv_header = True
                format_type = "standard"
                current_history_data = []
                continue

            # Look for the extended convergence table header
            elif extended_header_pattern.search(line):
                if debug:
                    print(f"Found extended convergence header at line {line_idx+1}: {line_strip}")
                parsing_this_entry_history = True
                found_conv_header = True
                format_type = "extended"

                # Initialize column names with iter column
                column_names = ['iter']

                # Check if this is a residual or error table
                is_residual = '|r' in line
                is_absolute = '/|' not in line

                # Add the main norm column
                if is_absolute:
                    column_names.append('abs_res_norm' if is_residual else 'abs_error_norm')
                else:
                    column_names.append('rel_res_norm' if is_residual else 'rel_error_norm')

                # Find all additional residual or error columns
                matches = residual_col_pattern.finditer(line)
                for match in matches:
                    if is_absolute:
                        # For absolute norms, we have either ||r0||_2 or ||e0||_2
                        prefix = match.group(4)  # r or e from the absolute pattern
                        index = match.group(5)   # index from the absolute pattern
                    else:
                        # For relative norms, we have |r0|_2/|b0|_2 or |e0|_2/|eI0|_2
                        prefix = match.group(1)  # r or e from the relative pattern
                        index = match.group(2)   # index from the relative pattern

                    # Add column name with appropriate prefix
                    if is_absolute:
                        column_names.append(f'abs_{prefix}{index}_norm')
                    else:
                        column_names.append(f'rel_{prefix}{index}_norm')

                if debug:
                    print(f"Extended format columns: {column_names}")

                current_history_data = []
                continue

            # Check if we're in the header decoration line (------)
            if parsing_this_entry_history and found_conv_header and re.match(r'^\s*[-]+\s+[-]+', line):
                if debug:
                    print(f"Found header decoration line at {line_idx+1}")
                continue

            # If we're inside a convergence table, parse the data rows
            if parsing_this_entry_history and found_conv_header:
                if format_type == "standard" and (data_match := standard_data_pattern.search(line)):
                    # Extract all four columns from the standard table
                    iteration = int(data_match.group(1))
                    resid_norm = float(data_match.group(2))  # Absolute residual norm
                    conv_rate = float(data_match.group(3))   # Convergence rate
                    rel_res_norm = float(data_match.group(4)) # Relative residual norm

                    if debug and len(current_history_data) == 0:
                        print(f"Found first iteration data (standard): {iteration}, {resid_norm}, {conv_rate}, {rel_res_norm}")

                    # Store all extracted data in our history collection
                    current_history_data.append({
                        'iter': iteration,
                        'rel_res_norm': rel_res_norm,
                        'resid_norm': resid_norm,
                        'conv_rate': conv_rate
                    })

                elif format_type == "extended" and re.match(r'^\s*\d+\s+[\d.eE+-]+', line):
                    # Parse extended format with multiple residual or error columns
                    # The pattern is more complex, so we'll split by whitespace and process
                    values = re.findall(r'[\d.eE+-]+', line)

                    if len(values) >= 2:  # At least iteration and main metric
                        iteration = int(values[0])
                        # Create data row with the main column (either rel_res_norm, rel_error_norm, abs_res_norm, or abs_error_norm)
                        data_row = {'iter': iteration}

                        # Add the main norm value based on the column names
                        if 'rel_res_norm' in column_names:
                            data_row['rel_res_norm'] = float(values[1])
                        elif 'rel_error_norm' in column_names:
                            data_row['rel_error_norm'] = float(values[1])
                        elif 'abs_res_norm' in column_names:
                            data_row['abs_res_norm'] = float(values[1])
                        elif 'abs_error_norm' in column_names:
                            data_row['abs_error_norm'] = float(values[1])

                        # Add additional norm values
                        for i, col_name in enumerate(column_names[2:], start=2):
                            if i < len(values):
                                data_row[col_name] = float(values[i])

                        # Add the row to our data
                        current_history_data.append(data_row)

                # Check for various conditions that indicate the end of a convergence table
                elif (not line_strip or
                      final_l2_norm_pattern.search(line) or
                      stats_summary_section_pattern.search(line) or
                      line_idx == len(lines) - 1):

                    # Save the convergence history for this system if we have data and it's not excluded
                    if active_entry_id not in exclude_entries and current_history_data:
                        if debug:
                            print(f"End of convergence data at line {line_idx+1}, saving {len(current_history_data)} points for entry {active_entry_id}")
                        history_df = pd.DataFrame(current_history_data)

                        # Create iter=0 row with appropriate initial values
                        iter0_row = pd.DataFrame([{
                            'iter': 0,
                        }])

                        # Add appropriate norm value for either residual or error
                        if 'rel_res_norm' in history_df.columns:
                            iter0_row['rel_res_norm'] = 1.0
                        elif 'rel_error_norm' in history_df.columns:
                            iter0_row['rel_error_norm'] = 1.0
                        elif 'abs_res_norm' in history_df.columns:
                            iter0_row['abs_res_norm'] = history_df['abs_res_norm'].iloc[0] if not history_df.empty else 1.0
                        elif 'abs_error_norm' in history_df.columns:
                            iter0_row['abs_error_norm'] = history_df['abs_error_norm'].iloc[0] if not history_df.empty else 1.0

                        # Add values for other columns if they exist
                        if format_type == "standard" and 'resid_norm' in history_df.columns:
                            iter0_row['resid_norm'] = history_df['resid_norm'].iloc[0] if not history_df.empty else 1.0
                            iter0_row['conv_rate'] = 1.0  # Convention for first iteration

                        elif format_type == "extended":
                            # Set all r0/e0, r1/e1, etc values to initial values for iteration 0
                            for col in history_df.columns:
                                if (col.startswith('r') or col.startswith('e') or col.startswith('abs_')) and\
                                   col not in ['rel_res_norm', 'rel_error_norm', 'abs_res_norm', 'abs_error_norm'] and\
                                   col not in iter0_row:
                                    # For absolute norms, use the first value, for relative norms use 1.0
                                    if col.startswith('abs_'):
                                        iter0_row[col] = history_df[col].iloc[0] if not history_df.empty else 1.0
                                    else:
                                        iter0_row[col] = 1.0  # Always start at 1.0 for relative metrics

                        # Concatenate and sort
                        history_df = pd.concat([iter0_row, history_df]).sort_values('iter').reset_index(drop=True)

                        if not history_df.empty:
                            file_histories.append({
                                'filename': filename,
                                'entry_id': active_entry_id,
                                'history_df': history_df,
                                'format_type': format_type
                            })

                    # Reset state for next section
                    parsing_this_entry_history = False
                    found_conv_header = False
                    current_history_data = []

                    # If we've reached the statistics summary, we're done with all systems
                    if stats_summary_section_pattern.search(line):
                        if debug:
                            print(f"Found stats summary at line {line_idx+1}, ending entry {active_entry_id}")
                        active_entry_id = None

    # Handle any remaining data at the end of file if the last system wasn't properly closed
    if active_entry_id is not None and parsing_this_entry_history and current_history_data:
        if active_entry_id not in exclude_entries:
            if debug:
                print(f"Saving final history data for entry {active_entry_id} with {len(current_history_data)} points")
            history_df = pd.DataFrame(current_history_data)

            # Create iter=0 row with appropriate initial values
            iter0_row = pd.DataFrame([{
                'iter': 0,
            }])

            # Add appropriate norm value for either residual or error
            if 'rel_res_norm' in history_df.columns:
                iter0_row['rel_res_norm'] = 1.0
            elif 'rel_error_norm' in history_df.columns:
                iter0_row['rel_error_norm'] = 1.0
            elif 'abs_res_norm' in history_df.columns:
                iter0_row['abs_res_norm'] = history_df['abs_res_norm'].iloc[0] if not history_df.empty else 1.0
            elif 'abs_error_norm' in history_df.columns:
                iter0_row['abs_error_norm'] = history_df['abs_error_norm'].iloc[0] if not history_df.empty else 1.0

            # Add values for other columns if they exist
            if format_type == "standard" and 'resid_norm' in history_df.columns:
                iter0_row['resid_norm'] = history_df['resid_norm'].iloc[0] if not history_df.empty else 1.0
                iter0_row['conv_rate'] = 1.0  # Convention for first iteration
            elif format_type == "extended":
                # Set all r0/e0, r1/e1, etc values to initial values for iteration 0
                for col in history_df.columns:
                    if (col.startswith('r') or col.startswith('e') or col.startswith('abs_')) and col not in ['rel_res_norm', 'rel_error_norm', 'abs_res_norm', 'abs_error_norm'] and col not in iter0_row:
                        # For absolute norms, use the first value, for relative norms use 1.0
                        if col.startswith('abs_'):
                            iter0_row[col] = history_df[col].iloc[0] if not history_df.empty else 1.0
                        else:
                            iter0_row[col] = 1.0  # Always start at 1.0 for relative metrics

            # Concatenate and sort
            history_df = pd.concat([iter0_row, history_df]).sort_values('iter').reset_index(drop=True)

            if not history_df.empty:
                file_histories.append({
                    'filename': filename,
                    'entry_id': active_entry_id,
                    'history_df': history_df,
                    'format_type': format_type
                })

    if debug:
        print(f"Total histories found: {len(file_histories)}")
        for i, hist in enumerate(file_histories):
            print(f"  History {i+1}: Entry {hist['entry_id']}, Format: {hist['format_type']}, {len(hist['history_df'])} points")
            if hist['format_type'] == "extended":
                print(f"    Columns: {list(hist['history_df'].columns)}")

    return file_histories

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

        print(f"- Saving figure: {savefig} ...")
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

def plot_convergence_histories(file_histories, use_title=False, savefig=None, annotate_last_point=False):
    """
    Plots convergence histories from one or more linear system solves.

    Creates a semi-log plot (log scale on y-axis) showing the convergence behavior
    of iterative solvers. Each line represents one linear system solve, with markers
    at each iteration point. The plot shows relative residual or error norm vs. iteration number.

    Parameters
    ----------
    file_histories : list of dict
        List of dictionaries containing convergence history data, each with:
        - 'filename': source file path (str)
        - 'entry_id': linear system ID (int)
        - 'history_df': pandas DataFrame with columns:
          * 'iter': iteration number
          * 'rel_res_norm' or 'rel_error_norm': relative residual/error norm
    use_title : bool, optional
        Whether to display a title on the plot
    savefig : str, optional
        If provided, saves the figure to this path instead of displaying it
    annotate_last_point : bool, optional
        If True, adds an annotation to the last point of each curve showing
        the final iteration number

    Returns
    -------
    None
        Either displays the plot or saves it to a file

    Notes
    -----
    - Y-axis uses logarithmic scale to better visualize convergence
    - Legend is organized in two columns when there are more than 5 entries
    - Grid lines are set to 40% opacity for improved visibility
    """
    if not file_histories:
        print("No convergence history data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    distinct_files = set(fh['filename'] for fh in file_histories)

    # Markers for different entries within the same file
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']

    # First determine if we're plotting residuals or errors (or both)
    contains_residuals = any('rel_res_norm' in fh['history_df'].columns for fh in file_histories)
    contains_errors = any('rel_error_norm' in fh['history_df'].columns for fh in file_histories)

    metric_type = "residual" if contains_residuals and not contains_errors else \
                 "error" if contains_errors and not contains_residuals else \
                 "convergence"  # Mixed or neither

    # First collect all the final points to plan annotation positions
    final_points = []
    for i, history_data in enumerate(file_histories):
        if not history_data['history_df'].empty:
            df = history_data['history_df']
            last_row = df.sort_values('iter').iloc[-1]

            # Determine which column to use (rel_res_norm or rel_error_norm)
            if 'rel_res_norm' in df.columns:
                value_col = 'rel_res_norm'
            else:
                value_col = 'rel_error_norm'

            final_points.append({
                'index': i,
                'iter': int(last_row['iter']),
                'value': float(last_row[value_col])
            })

    # Sort points by iteration number to handle overlaps systematically
    final_points.sort(key=lambda x: x['iter'])

    # Plot each history
    for i, history_data in enumerate(file_histories):
        filename = history_data['filename']
        entry_id = history_data['entry_id']
        df = history_data['history_df']

        # Determine which column to use (rel_res_norm or rel_error_norm)
        if 'rel_res_norm' in df.columns:
            value_col = 'rel_res_norm'
            prefix = "residual"
        else:
            value_col = 'rel_error_norm'
            prefix = "error"

        # Get basename of file for label
        label = f"{os.path.basename(filename)}, system #{entry_id} ({prefix})"

        # Choose marker based on index
        marker = markers[0] #markers[i % len(markers)]

        ax.semilogy(df['iter'], df[value_col], label=label,
                   marker=marker, markersize=5, linestyle='-', linewidth=1.0)

    # Add annotations in a separate pass after all lines are plotted
    if annotate_last_point and final_points:
        # Check for potential overlaps and adjust positions
        min_x_distance = 5  # minimum distance between annotations in x-axis units

        # Assign vertical positions (all annotations above points)
        vertical_offset = 12  # fixed vertical offset above the point

        # Add the annotations
        for i, point in enumerate(final_points):
            idx = point['index']
            last_iter = point['iter']
            last_value = point['value']

            # Check if we need to adjust this annotation because it's close to the previous one
            if i > 0 and (last_iter - final_points[i-1]['iter']) < min_x_distance:
                # Increase vertical offset to avoid overlap with previous annotation
                current_offset = vertical_offset + 15 * (i % 3 + 1)  # Use a pattern of 3 levels
            else:
                current_offset = vertical_offset

            ax.annotate(f"{last_iter}",
                       xy=(last_iter, last_value),
                       xytext=(0, current_offset),  # centered above point
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.7))

    # Set plot title and labels
    if use_title:
        ax.set_title(f'{metric_type.capitalize()} Convergence History', fontsize=tfs, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=alfs)

    # Use appropriate y-axis label
    if metric_type == "residual":
        y_label = 'Relative Residual Norm'
    elif metric_type == "error":
        y_label = 'Relative Error Norm'
    else:
        y_label = 'Relative Norm'
    ax.set_ylabel(y_label, fontsize=alfs)

    # Determine number of columns for legend based on number of entries
    ncols = 2 if len(file_histories) > 5 else 1
    ax.legend(loc='best', ncol=ncols, fancybox=True, shadow=True, fontsize=lgfs)

    # Configure tick labels
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)

    # Configure grid with higher opacity (40%) for better visibility
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()

    # Determine filename suffix based on metric type
    metric_prefix = ""
    if metric_type == "residual":
        metric_prefix = "res_"
    elif metric_type == "error":
        metric_prefix = "err_"

    savefig_name = f"{metric_prefix}conv_hist.png" # Default name if savefig is None or generic
    if savefig and savefig[-4:] != "None": # Check if savefig is a specific suffix
        savefig_name = f"{metric_prefix}conv_hist_{savefig}"

    save_and_show_plot(savefig_name)

def plot_multi_residual_convergence(history_data, use_title=False, savefig=None, annotate_last_point=False):
    """
    Plots convergence histories with multiple residual or error columns from a single solve.

    This function creates a semi-log plot showing the convergence behavior of
    different residual components (r, r0, r1, r2, etc.) or error components
    (e, e0, e1, e2, etc.) for a single linear system solve.

    Parameters
    ----------
    history_data : dict
        Dictionary containing data for a single convergence history with:
        - 'filename': source file path (str)
        - 'entry_id': linear system ID (int)
        - 'history_df': pandas DataFrame with multiple columns
        - 'format_type': Should be 'extended' for this function

    use_title : bool, optional
        Whether to display a title on the plot

    savefig : str, optional
        If provided, saves the figure to this path instead of displaying it

    annotate_last_point : bool, optional
        If True, adds an annotation to the last point of each curve showing
        the final iteration number

    Returns
    -------
    None
        Either displays the plot or saves it to a file

    Notes
    -----
    - This function is specifically for plotting the extended format with multiple columns
    - Y-axis uses logarithmic scale to better visualize convergence
    - Each component (r, r0, r1, etc. or e, e0, e1, etc.) is plotted as a separate curve
    """
    if not history_data or 'history_df' not in history_data or history_data['format_type'] != 'extended':
        print("No extended convergence history data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    df = history_data['history_df']
    entry_id = history_data['entry_id']
    filename = history_data['filename']

    # Determine if this is a residual or error plot and if it's absolute or relative
    is_residual = any(col.startswith(('rel_res_', 'abs_res_')) for col in df.columns)
    is_absolute = any(col.startswith('abs_') for col in df.columns)

    # Get the main metric column
    if is_absolute:
        main_column = 'abs_res_norm' if is_residual else 'abs_error_norm'
    else:
        main_column = 'rel_res_norm' if is_residual else 'rel_error_norm'

    metric_type = 'residual' if is_residual else 'error'

    # Get all columns that represent residual or error norms
    metric_columns = [main_column]  # Main metric

    # Find all additional metric columns
    for col in df.columns:
        if col == 'iter' or col == main_column:
            continue

        # For absolute norms, look for abs_r*_norm or abs_e*_norm
        if is_absolute:
            if (is_residual and col.startswith('abs_r') and col.endswith('_norm')) or \
               (not is_residual and col.startswith('abs_e') and col.endswith('_norm')):
                metric_columns.append(col)
        # For relative norms, look for r* or e* columns
        else:
            if (is_residual and col.startswith('rel_r') and col != 'rel_res_norm') or \
               (not is_residual and col.startswith('rel_e') and col != 'rel_error_norm'):
                metric_columns.append(col)

    # Different marker styles for variety
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Plot each component
    for i, col in enumerate(metric_columns):
        # Format the label: rel_res_norm -> "r", r0 -> "r₀", r1 -> "r₁", etc.
        # or rel_error_norm -> "e", e0 -> "e₀", e1 -> "e₁", etc.
        if col == main_column:
            label = "r" if is_residual else "e"
        else:
            label = f"{col[4:-5]}" # Remove 'abs_' prefix and '_norm' suffix

        # Choose marker
        marker = markers[0] #markers[i % len(markers)]  # Use different markers for each line

        # Plot the curve
        ax.semilogy(df['iter'], df[col], label=label,
                   marker=marker, markersize=5, linestyle='-', linewidth=1.5)

        # Add annotation for the last point if requested
        if annotate_last_point:
            # Get the last row in the sorted dataframe
            last_row = df.sort_values('iter').iloc[-1]
            last_iter = int(last_row['iter'])
            last_value = float(last_row[col])

            # Add annotation with vertical spacing based on the column
            vertical_offset = 12  # Base vertical offset

            ax.annotate(f"{last_iter}",
                       xy=(last_iter, last_value),
                       xytext=(0, vertical_offset),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.7))

    # Set plot title and labels
    file_basename = os.path.basename(filename)
    metric_label = "Absolute Residual Norm" if is_absolute and is_residual else \
                  "Absolute Error Norm" if is_absolute else \
                  "Relative Residual Norm" if is_residual else \
                  "Relative Error Norm"

    if use_title:
        ax.set_title(f'{metric_type.capitalize()} Convergence History - {file_basename}, System #{entry_id}', fontsize=tfs, fontweight='bold')
    else:
        ax.set_title(f'{file_basename}, System #{entry_id}', fontsize=tfs-2)

    ax.set_xlabel('Iteration', fontsize=alfs)
    ax.set_ylabel(metric_label, fontsize=alfs)

    # Configure tick labels
    ax.tick_params(axis='x', labelsize=alfs)
    ax.tick_params(axis='y', labelsize=alfs)

    # Set a nice legend with a frame and shadow
    ax.legend(loc='best', ncol=2, fancybox=True, shadow=True, fontsize=lgfs)

    # Configure grid
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()

    metric_prefix = 'abs_res' if is_absolute and is_residual else \
                   'abs_err' if is_absolute else \
                   'res' if is_residual else 'err'
    savefig_name = f"multi_{metric_prefix}_conv_hist_{os.path.splitext(os.path.basename(filename))[0]}_sys{entry_id}.png"
    if savefig and savefig[-4:] != "None":
        savefig_name = f"multi_{metric_prefix}_conv_hist_{savefig}"

    save_and_show_plot(savefig_name)

def check_mode_exact_match(mode, word):
    # Split the mode string into parts separated by '+'
    parts = mode.split('+')

    # Check if the word exactly matches any of the parts
    return word in parts

def create_main_dataframe(filenames_list, exclude_list):
    data_series = []
    time_unit_overall = "[s]"
    first_file_processed_for_time_unit = False

    for f_name in filenames_list:
        try:
            current_series_list, current_time_unit = parse_statistics_summary(f_name, exclude_list)
            if current_series_list:
                if not first_file_processed_for_time_unit:
                    time_unit_overall = current_time_unit
                    first_file_processed_for_time_unit = True
                data_series.extend(current_series_list)
            elif not first_file_processed_for_time_unit :
                 # Attempt to get time_unit even if no valid series are returned (e.g. all excluded)
                 # This assumes parse_statistics_summary can still return a time_unit
                 try:
                     _, temp_time_unit = parse_statistics_summary(f_name, [])
                     time_unit_overall = temp_time_unit
                     first_file_processed_for_time_unit = True # Mark as processed for time_unit
                 except ValueError: # If parsing completely fails
                     pass
        except ValueError as e:
            print(f"Error parsing summary from {f_name}: {e}")

    if not data_series:
        # No data series were created, but time_unit might have been found
        return None, time_unit_overall

    df = pd.concat(data_series, axis=1).T.reset_index(drop=True)

    data_types = {
        'entry':    'Int64',
        'nranks':   'Int64',
        'rows':     'Int64',
        'nonzeros': 'Int64',
        'build':    'float',
        'setup':    'float',
        'solve':    'float',
        'total':    'float',
        'resnorm':  'float',
        'iters':    'Int64',
        'filename': 'str'
    }

    for col, dtype_str in data_types.items():
        if col in df.columns:
            # For Int64, pandas handles NaN conversion from float automatically if needed.
            # If column is all NaN, astype(Int64) works. If mixed with actual numbers, pd.to_numeric first.
            if dtype_str == 'Int64':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            else:
                df[col] = df[col].astype(dtype_str)
        else:
             # If a column is entirely missing (e.g., 'filename' if not added properly by parser)
             # Add it as an empty series of the correct type.
            if dtype_str == 'object' or dtype_str == 'str':
                 df[col] = pd.Series(dtype='object', index=df.index)
            elif dtype_str == 'Int64':
                 df[col] = pd.Series(dtype='Int64', index=df.index)
            else: # float
                 df[col] = pd.Series(dtype='float', index=df.index)


    if 'filename' in df.columns and 'entry' in df.columns:
        # Ensure 'entry' is numeric for sorting if it became object due to NaNs before Int64
        df['entry'] = pd.to_numeric(df['entry'], errors='coerce')
        df = df.sort_values(by=['filename', 'entry']).reset_index(drop=True)

    return df, time_unit_overall

def parse_arguments():
    """
    Parse command-line arguments for the script.

    Defines and processes all command-line options for analyzing HYPRE solver output,
    including file selection, plotting modes, axis customization, and output options.

    Returns
    -------
    tuple
        (labels, args) where:
        - labels: Dictionary mapping data types to axis labels
        - args: Parsed command-line arguments
    """
    # List of pre-defined labels
    labels = {'rows': "Number of rows",
              'nonzeros': "Number of nonzeros",
              'entry': "Linear system ID",
              'nranks': "Number of MPI ranks"}

    # List of pre-defined modes:
    modes = ('iters', 'times', 'iters-and-times', 'conv-hist')
    mode_choices_list = []
    for i in range(1, len(modes) + 1):
        for comb in combinations(modes, i):
            mode_choices_list.append('+'.join(sorted(list(comb))))
    mode_choices = tuple(sorted(list(set(mode_choices_list))))

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse and visualize statistics from HYPRE solver output")
    parser.add_argument("-f", "--filename", type=str, nargs="+", required=True,
                       help="Path to the log file(s)")
    parser.add_argument("-e", "--exclude", type=int, nargs="+", default=[],
                       help="Exclude certain entries from the statistics")
    parser.add_argument("-m", "--mode", type=str, default='iters-and-times+conv-hist',
                       choices=mode_choices,
                       help="What information to plot")
    parser.add_argument("-t", "--xtype", type=str, default='entry',
                       choices=list(labels.keys()),
                       help="Variable type for the abscissa")
    parser.add_argument("-l", "--xlabel", type=str, default=None,
                       help="Label for the abscissa")
    parser.add_argument("-s", "--savefig", default=None,
                       help="Save figure(s) using this name suffix (e.g., my_run.png). If 'None' or not set, only shows plots.")
    parser.add_argument("-c", "--cumulative", action='store_true',
                       help='Plot cumulative quantities for iters/times plots')
    parser.add_argument("-u", "--use_title", action='store_true',
                       help='Show title in plots')
    parser.add_argument("-v", "--verbose", action='store_true',
                       help='Print dataframe contents')
    parser.add_argument("-a", "--annotate_last_point", action='store_true',
                       help='Annotate the last point in each convergence curve with iteration count')

    return labels, parser.parse_args()

def main():
    """
    Main function that processes command line arguments and generates plots.

    Handles all the processing workflow:
    1. Parses command-line arguments
    2. Creates dataframes for statistical summary data if needed
    3. Generates appropriate plots based on the selected modes
    4. Saves or displays the figures based on user preferences
    """
    labels, args = parse_arguments()

    # Update label
    xlabel = args.xlabel if args.xlabel else labels[args.xtype]

    # Update savefig string
    # Ensure savefig is either a valid string for suffix or None
    savefig_suffix = args.savefig
    if args.savefig and args.savefig.lower() == "none":
        savefig_suffix = None
    elif args.savefig == ".": # Special case to use filename
         if len(args.filename) == 1:
             savefig_suffix = f"{os.path.splitext(os.path.basename(args.filename[0]))[0]}.png"
         else:
             savefig_suffix = "multi_file_plot.png" # Default for multiple files if "." is used

    # Dataframe for summary stats, and overall time_unit
    df_summary = None
    time_unit = "[s]" # Default

    # Create DataFrame for summary stats if needed by the selected modes
    if any(check_mode_exact_match(args.mode, m) for m in ['iters', 'times', 'iters-and-times']):
        df_summary, time_unit = create_main_dataframe(args.filename, args.exclude)
        if df_summary is None or df_summary.empty:
            print("No summary statistics data found or all excluded. Skipping summary-based plots.")
        elif args.verbose:
            print("--- Summary DataFrame ---")
            print(df_summary)
            if 'total' in df_summary.columns and not df_summary['total'].empty:
                 print(f"Sum total time from summary: {df_summary['total'].sum()} {time_unit}")


    # Produce plots based on mode
    if check_mode_exact_match(args.mode, 'iters'):
        if df_summary is not None and not df_summary.empty:
            plot_iterations(df_summary, args.cumulative, args.xtype, xlabel, args.use_title, savefig_suffix)
        else:
            print("Skipping iterations plot: No summary data available.")

    if check_mode_exact_match(args.mode, 'times'):
        if df_summary is not None and not df_summary.empty:
            plot_times(df_summary, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig_suffix)
        else:
            print("Skipping times plot: No summary data available.")

    if check_mode_exact_match(args.mode, 'iters-and-times'):
        if df_summary is not None and not df_summary.empty:
            plot_iters_times(df_summary, args.cumulative, args.xtype, xlabel, time_unit, args.use_title, savefig_suffix)
        else:
            print("Skipping iterations and times plot: No summary data available.")

    if check_mode_exact_match(args.mode, 'conv-hist'):
        all_histories_for_plotting = []
        multi_residual_histories = []

        for f_name_loop in args.filename: # Iterate over each filename provided
            try:
                histories_in_file = parse_convergence_history_for_file(f_name_loop, args.exclude)

                # Sort histories into standard and extended formats
                for hist in histories_in_file:
                    if hist['format_type'] == 'standard':
                        all_histories_for_plotting.append(hist)
                    elif hist['format_type'] == 'extended':
                        multi_residual_histories.append(hist)

            except ValueError as e:
                 print(f"Error parsing convergence history from {f_name_loop}: {e}")
            except FileNotFoundError:
                 print(f"Error: File not found {f_name_loop} for convergence history parsing.")

        # Plot standard convergence histories if found
        if all_histories_for_plotting:
            if args.verbose:
                print("\n--- Standard Convergence Histories Found ---")
                for item in all_histories_for_plotting:
                    print(f"File: {item['filename']}, Entry: {item['entry_id']}, Points: {len(item['history_df'])}")
            plot_convergence_histories(all_histories_for_plotting, args.use_title, savefig_suffix, args.annotate_last_point)

        # Plot each multi-residual history separately
        if multi_residual_histories:
            if args.verbose:
                print("\n--- Extended Multi-Residual Convergence Histories Found ---")
                for item in multi_residual_histories:
                    print(f"File: {item['filename']}, Entry: {item['entry_id']}, Points: {len(item['history_df'])}")
                    print(f"  Columns: {list(item['history_df'].columns)}")

            for hist in multi_residual_histories:
                plot_multi_residual_convergence(hist, args.use_title, savefig_suffix, args.annotate_last_point)

        if not all_histories_for_plotting and not multi_residual_histories:
            print("No convergence history data found or all relevant entries were excluded.")

if __name__ == "__main__":
    main()
