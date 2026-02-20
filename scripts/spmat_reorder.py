#!/usr/bin/env python3
# /******************************************************************************
# * Copyright (c) 2024 Lawrence Livermore National Security, LLC
# * SPDX-License-Identifier: MIT
# ******************************************************************************/

"""
Reorder a matrix in block format according to dofmap indices.

Reads a matrix from HYPRE IJ format files and reorders rows/columns based on
dofmap indices. The reordering is specified as a list of dofmap index values.
"""

import argparse
import glob
import os
import numpy as np
from typing import List, Tuple, Optional


def read_dofmap(dofmap_file: str) -> np.ndarray:
    """Read dofmap from file.
    
    Format: First line is count, then one index per line.
    Returns array of dofmap indices (one per row).
    """
    with open(dofmap_file, 'r') as f:
        count = int(f.readline().strip())
        dofmap = np.array([int(f.readline().strip()) for _ in range(count)], dtype=np.int64)
    return dofmap


def read_hypre_binary_matrix(parts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Read HYPRE binary matrix from part files.
    
    Args:
        parts: List of binary matrix file paths
        
    Returns:
        (rows, cols, vals, (nrows_glob, ncols_glob))
    """
    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    vals_all: List[np.ndarray] = []
    nrows_glob = -1
    ncols_glob = -1

    for p in sorted(parts):
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

    return rows, cols, vals, (nrows_glob, ncols_glob)


def read_matrix_coordinate(parts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Read matrix in coordinate format from text files.
    
    Format: First line may be header (4 integers), then row col value per line.
    Returns (rows, cols, values, (nrows, ncols)) arrays.
    """
    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    vals_all: List[np.ndarray] = []
    nrows_glob = -1
    ncols_glob = -1
    
    for p in sorted(parts):
        rows_part = []
        cols_part = []
        vals_part = []
        
        with open(p, 'r') as f:
            # Check if first line is header
            first_line = f.readline().strip()
            first_values = first_line.split()
            
            # If first line has 4 numbers, it might be a header (nrows_start nrows_end ncols_start ncols_end)
            header_read = False
            if len(first_values) == 4:
                try:
                    v0, v1, v2, v3 = map(int, first_values)
                    if v0 == 0 and v3 > 0:  # Common header format
                        nrows_part = v3
                        ncols_part = v3  # Assume square if not specified
                        header_read = True
                except ValueError:
                    header_read = False
            
            if not header_read:
                # First line is data, rewind
                f.seek(0)
            
            # Read coordinate entries
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts_line = line.split()
                if len(parts_line) >= 3:
                    try:
                        r = int(parts_line[0])
                        c = int(parts_line[1])
                        v = float(parts_line[2])
                        rows_part.append(r)
                        cols_part.append(c)
                        vals_part.append(v)
                    except (ValueError, IndexError):
                        continue
        
        if not rows_part:
            continue
        
        r = np.array(rows_part, dtype=np.int64)
        c = np.array(cols_part, dtype=np.int64)
        v = np.array(vals_part, dtype=np.float64)
        
        # Determine global dimensions from max indices
        if nrows_glob < 0:
            nrows_glob = int(r.max()) + 1 if r.size > 0 else 0
            ncols_glob = int(c.max()) + 1 if c.size > 0 else 0
        else:
            nrows_glob = max(nrows_glob, int(r.max()) + 1 if r.size > 0 else 0)
            ncols_glob = max(ncols_glob, int(c.max()) + 1 if c.size > 0 else 0)
        
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
    
    return rows, cols, vals, (nrows_glob, ncols_glob)


def write_matrix_coordinate(matrix_file: str, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, 
                           shape: Optional[Tuple[int, int]] = None):
    """Write matrix in coordinate format.
    
    If shape is provided, writes header line (0 end_row 0 end_col) where end indices are inclusive.
    Format matches HYPRE: 0 (nrows-1) 0 (ncols-1)
    """
    with open(matrix_file, 'w') as f:
        # Write header if shape is provided
        if shape is not None:
            nrows, ncols = shape
            end_row = max(0, nrows - 1) if nrows > 0 else 0
            end_col = max(0, ncols - 1) if ncols > 0 else 0
            f.write(f"0 {end_row} 0 {end_col}\n")
        
        # Write coordinate entries
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r} {c} {v:.15e}\n")


def write_info_file(info_file: str, dofmap: np.ndarray, row_mapping: np.ndarray, 
                    row_reorder_list: List[int], col_mapping: np.ndarray, 
                    col_reorder_list: List[int], total_vertices: int):
    """Write info file with partitioning information based on dof boundaries.
    
    Format:
        # Number of partitions: <ndoms>
        # Total vertices: <total_vertices>
        original_vertex new_position partition
    
    Partitions are determined by the order of dofmap indices in the reorder list.
    """
    # Determine number of partitions (unique dofmap values in reorder list)
    ndoms = len(row_reorder_list)
    
    # Create mapping from dofmap index to partition number for rows
    dof_to_row_partition = {}
    for part_idx, dof_idx in enumerate(row_reorder_list):
        dof_to_row_partition[dof_idx] = part_idx
    
    # Create mapping from dofmap index to partition number for columns
    dof_to_col_partition = {}
    for part_idx, dof_idx in enumerate(col_reorder_list):
        dof_to_col_partition[dof_idx] = part_idx
    
    with open(info_file, 'w') as f:
        # Write header
        f.write(f"# Number of partitions: {ndoms}\n")
        f.write(f"# Total vertices: {total_vertices}\n")
        f.write("# Format: original_vertex new_position partition\n")
        
        # Write vertex mappings (using row partitioning as primary)
        # For each original vertex, determine its partition and new position
        for original_vertex in range(len(dofmap)):
            new_position = row_mapping[original_vertex]
            dof_idx = dofmap[original_vertex]
            partition = dof_to_row_partition.get(dof_idx, ndoms - 1)  # Default to last partition if not found
            
            f.write(f"{original_vertex} {new_position} {partition}\n")


def create_reordering_map(dofmap: np.ndarray, reorder_list: List[int]) -> np.ndarray:
    """Create mapping from old indices to new indices based on dofmap reordering.
    
    Args:
        dofmap: Array of dofmap indices for each row/column
        reorder_list: List of dofmap indices in desired order
        
    Returns:
        Mapping array: new_index[old_index] gives the new position
    """
    n = len(dofmap)
    mapping = np.zeros(n, dtype=np.int64)
    
    # Build reverse mapping: which old indices have each dofmap value
    dofmap_to_old_indices = {}
    for old_idx, dof_idx in enumerate(dofmap):
        if dof_idx not in dofmap_to_old_indices:
            dofmap_to_old_indices[dof_idx] = []
        dofmap_to_old_indices[dof_idx].append(old_idx)
    
    # Assign new indices according to reorder_list
    new_idx = 0
    for target_dof in reorder_list:
        if target_dof in dofmap_to_old_indices:
            for old_idx in dofmap_to_old_indices[target_dof]:
                mapping[old_idx] = new_idx
                new_idx += 1
    
    # Handle any dofmap indices not in reorder_list (keep original order)
    all_dofs = set(dofmap)
    for target_dof in sorted(all_dofs):
        if target_dof not in reorder_list:
            if target_dof in dofmap_to_old_indices:
                for old_idx in dofmap_to_old_indices[target_dof]:
                    mapping[old_idx] = new_idx
                    new_idx += 1
    
    return mapping


def reorder_matrix(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                   row_mapping: np.ndarray, col_mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply row and column reordering to matrix.
    
    Args:
        rows: Original row indices
        cols: Original column indices
        vals: Matrix values
        row_mapping: Mapping from old row indices to new row indices
        col_mapping: Mapping from old column indices to new column indices
        
    Returns:
        (new_rows, new_cols, new_vals)
    """
    new_rows = row_mapping[rows]
    new_cols = col_mapping[cols]
    return new_rows, new_cols, vals


def compute_partition_boundaries(dofmap: np.ndarray, reorder_list: List[int], 
                                 mapping: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute partition boundaries from dofmap and reorder list.
    
    Returns array of boundaries where boundaries[i] is the start of partition i.
    boundaries[0] = 0, boundaries[ndoms] = total_rows
    
    Args:
        dofmap: Array of dofmap indices
        reorder_list: List of dofmap indices in desired order
        mapping: Optional pre-computed mapping (if None, will compute)
    """
    if mapping is None:
        mapping = create_reordering_map(dofmap, reorder_list)
    
    n = len(dofmap)
    ndoms = len(reorder_list)
    boundaries = np.zeros(ndoms + 1, dtype=np.int64)
    
    # Create mapping from dofmap index to partition
    dof_to_partition = {}
    for part_idx, dof_idx in enumerate(reorder_list):
        dof_to_partition[dof_idx] = part_idx
    
    # Count vertices per partition
    partition_counts = {}
    for dof_idx in dofmap:
        part = dof_to_partition.get(dof_idx, ndoms - 1)
        partition_counts[part] = partition_counts.get(part, 0) + 1
    
    # Build boundaries (cumulative)
    boundaries[0] = 0
    for i in range(ndoms):
        boundaries[i + 1] = boundaries[i] + partition_counts.get(i, 0)
    
    return boundaries


def compute_block_statistics(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                            row_boundaries: np.ndarray, col_boundaries: np.ndarray) -> dict:
    """Compute statistics for each sub-block of the matrix.
    
    Returns dictionary with keys like 'block_0_0', 'block_0_1', etc.
    Each entry contains: 'frobenius_norm', 'avg_diag', 'nnz', 'min_val', 'max_val', 'size'
    """
    ndoms_row = len(row_boundaries) - 1
    ndoms_col = len(col_boundaries) - 1
    stats = {}
    
    for i in range(ndoms_row):
        for j in range(ndoms_col):
            # Extract block indices
            row_start = row_boundaries[i]
            row_end = row_boundaries[i + 1]
            col_start = col_boundaries[j]
            col_end = col_boundaries[j + 1]
            
            # Find entries in this block
            mask = (rows >= row_start) & (rows < row_end) & (cols >= col_start) & (cols < col_end)
            block_vals = vals[mask]
            block_rows = rows[mask]
            block_cols = cols[mask]
            
            nnz = len(block_vals)
            block_size = (row_end - row_start, col_end - col_start)
            
            if nnz > 0:
                frobenius_norm = np.sqrt(np.sum(block_vals ** 2))
                min_val = np.min(block_vals)
                max_val = np.max(block_vals)
                abs_vals = np.abs(block_vals)
                max_abs = np.max(abs_vals)
                min_abs = np.min(abs_vals)
                
                # Average number of coefficients per row
                # Count non-zeros per row in this block
                nrows_block = row_end - row_start
                if nrows_block > 0:
                    # Use bincount to count entries per row (relative to block start)
                    relative_rows = block_rows - row_start
                    row_counts = np.bincount(relative_rows, minlength=nrows_block)
                    avg_nnz_per_row = np.mean(row_counts)
                else:
                    avg_nnz_per_row = 0.0
                
                # Average diagonal value (for square blocks, including off-diagonal)
                avg_diag = None
                ncols_block = col_end - col_start
                is_square = (nrows_block == ncols_block) and (nrows_block > 0)
                
                if is_square:
                    # For square blocks, diagonal is when relative row == relative col
                    # i.e., block_rows - row_start == block_cols - col_start
                    relative_cols = block_cols - col_start
                    diag_mask = (relative_rows == relative_cols)
                    if diag_mask.any():
                        diag_vals = block_vals[diag_mask]
                        avg_diag = np.mean(diag_vals)
                    else:
                        avg_diag = 0.0  # No diagonal entries
            else:
                frobenius_norm = 0.0
                min_val = 0.0
                max_val = 0.0
                max_abs = 0.0
                min_abs = 0.0
                avg_nnz_per_row = 0.0
                # Check if block is square for avg_diag computation
                nrows_block = row_end - row_start
                ncols_block = col_end - col_start
                is_square = (nrows_block == ncols_block) and (nrows_block > 0)
                avg_diag = 0.0 if is_square else None
            
            stats[f'block_{i}_{j}'] = {
                'frobenius_norm': frobenius_norm,
                'avg_diag': avg_diag,
                'nnz': nnz,
                'avg_nnz_per_row': avg_nnz_per_row,
                'min_val': min_val,
                'max_val': max_val,
                'max_abs': max_abs,
                'min_abs': min_abs,
                'size': block_size
            }
    
    return stats


def print_block_statistics(stats: dict, row_boundaries: np.ndarray, col_boundaries: np.ndarray,
                          stat_name: str = 'frobenius_norm'):
    """Print block statistics in a matrix format.
    
    Args:
        stats: Dictionary from compute_block_statistics
        row_boundaries: Row partition boundaries
        col_boundaries: Column partition boundaries
        stat_name: Which statistic to display ('frobenius_norm', 'avg_diag', 'nnz', etc.)
    """
    ndoms_row = len(row_boundaries) - 1
    ndoms_col = len(col_boundaries) - 1
    
    # Determine format based on stat type
    if stat_name == 'nnz':
        col_width = 12
        fmt_str = '{:>12d}'
        na_str = '         N/A'
    elif stat_name == 'avg_nnz_per_row':
        col_width = 14
        fmt_str = '{:>14.2f}'
        na_str = '           N/A'
    elif stat_name in ['frobenius_norm', 'avg_diag', 'min_val', 'max_val', 'max_abs', 'min_abs']:
        col_width = 14
        fmt_str = '{:>14.2e}'
        na_str = '           N/A'
    else:
        col_width = 12
        fmt_str = '{:>12}'
        na_str = '         N/A'
    
    # Print header
    print(f"\n{stat_name.upper().replace('_', ' ')} by Block:")
    separator = "=" * (col_width * (ndoms_col + 1) + 10)
    print(separator)
    
    # Column headers
    header = "Block R\\C"
    for j in range(ndoms_col):
        header += f"{j:>{col_width}}"
    print(header)
    print("-" * (col_width * (ndoms_col + 1) + 10))
    
    # Print each row
    for i in range(ndoms_row):
        row_str = f"   {i:>3}  "
        for j in range(ndoms_col):
            key = f'block_{i}_{j}'
            if key in stats:
                val = stats[key][stat_name]
                if val is not None:
                    # For avg_nnz_per_row, show 0.0 as it's meaningful (indicates no entries per row)
                    if stat_name == 'avg_nnz_per_row' or val != 0.0:
                        row_str += fmt_str.format(val)
                    else:
                        row_str += na_str
                else:
                    row_str += na_str
            else:
                row_str += na_str
        print(row_str)
    
    print(separator)


def main():
    parser = argparse.ArgumentParser(
        description='Reorder matrix according to dofmap indices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f hypre-data-tmp/ls_00010 -r 0 1 2 4 5 6 3 -c 0 1 2 4 6 5 3
  %(prog)s -f /path/to/data --row-order 0 1 2 --col-order 2 1 0
        """
    )
    parser.add_argument('-f', '--file-dir', type=str, required=True,
                        help='Directory containing IJ.out.A.* and dofmap.out.* files')
    parser.add_argument('-r', '--row-order', type=int, nargs='+', required=True,
                        help='Dofmap indices for row reordering (in desired order)')
    parser.add_argument('-c', '--col-order', type=int, nargs='+', required=True,
                        help='Dofmap indices for column reordering (in desired order)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: same as input with _reordered suffix)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-s', '--stats', action='store_true',
                        help='Compute and display block statistics')
    
    args = parser.parse_args()
    
    # Find input files - handle both single and multiple part files
    matrix_files_all = sorted(glob.glob(os.path.join(args.file_dir, 'IJ.out.A.*')))
    dofmap_files_all = sorted(glob.glob(os.path.join(args.file_dir, 'dofmap.out.*')))
    
    if not matrix_files_all:
        raise SystemExit(f"Error: No matrix files found (IJ.out.A.*) in {args.file_dir}")
    if not dofmap_files_all:
        raise SystemExit(f"Error: No dofmap files found (dofmap.out.*) in {args.file_dir}")
    
    # Group matrix files by base name (to handle multiple parts)
    # For now, treat all matrix files as parts of a single matrix
    # This matches spyplot.py behavior where all IJ.out.A.* files in a directory are parts
    matrix_parts = sorted(matrix_files_all)
    
    # For dofmap, typically there's one per rank, but we'll use the first one
    # In practice, dofmap should be the same across all ranks
    dofmap_files = sorted(dofmap_files_all)
    if len(dofmap_files) > 1:
        if args.verbose:
            print(f"Warning: Found {len(dofmap_files)} dofmap files, using first one: {dofmap_files[0]}")
        dofmap_file = dofmap_files[0]
    else:
        dofmap_file = dofmap_files[0] if dofmap_files else None
    
    if dofmap_file is None:
        raise SystemExit(f"Error: No dofmap files found in {args.file_dir}")
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = args.file_dir.rstrip('/') + '_reordered'
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Input directory: {args.file_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Row dofmap order: {args.row_order}")
        print(f"Column dofmap order: {args.col_order}")
        print(f"Matrix part files: {len(matrix_parts)}")
        print(f"Dofmap file: {os.path.basename(dofmap_file)}")
    
    # Process the matrix (all parts together)
    if args.verbose:
        print(f"\nProcessing matrix parts with {os.path.basename(dofmap_file)}")
    
    # Read dofmap
    dofmap = read_dofmap(dofmap_file)
    if args.verbose:
        print(f"  Read dofmap: {len(dofmap)} entries, unique dofs: {sorted(set(dofmap))}")
    
    # Read matrix with auto-detection: try binary first, fall back to text
    try:
        rows, cols, vals, shape = read_hypre_binary_matrix(matrix_parts)
        if args.verbose:
            print(f"  Successfully read as binary format")
    except (RuntimeError, ValueError, IOError) as e:
        if args.verbose:
            print(f"  Binary read failed ({e}), trying text format")
        rows, cols, vals, shape = read_matrix_coordinate(matrix_parts)
        if args.verbose:
            print(f"  Successfully read as text format")
    
    nrows, ncols = shape
    if args.verbose:
        print(f"  Read matrix: {len(rows)} nonzeros, shape ({nrows}, {ncols})")
    
    # Check consistency
    if len(dofmap) != nrows:
        print(f"Warning: Dofmap length ({len(dofmap)}) != number of rows ({nrows})")
    
    # Create reordering mappings
    row_mapping = create_reordering_map(dofmap, args.row_order)
    col_mapping = create_reordering_map(dofmap, args.col_order)
    
    if args.verbose:
        print(f"  Row mapping: {len(np.unique(row_mapping))} unique target indices")
        print(f"  Column mapping: {len(np.unique(col_mapping))} unique target indices")
    
    # Apply reordering
    new_rows, new_cols, new_vals = reorder_matrix(rows, cols, vals, row_mapping, col_mapping)
    
    # Compute statistics if requested
    if args.stats:
        row_boundaries = compute_partition_boundaries(dofmap, args.row_order, row_mapping)
        col_boundaries = compute_partition_boundaries(dofmap, args.col_order, col_mapping)
        stats = compute_block_statistics(new_rows, new_cols, new_vals, row_boundaries, col_boundaries)
        
        print(f"\n{'='*80}")
        print(f"Block Statistics for: matrix with {len(matrix_parts)} part file(s)")
        print(f"{'='*80}")
        
        # Print partition boundaries
        print(f"\nRow Partition Boundaries: {row_boundaries}")
        print(f"Col Partition Boundaries: {col_boundaries}")
        
        # Print different statistics
        print_block_statistics(stats, row_boundaries, col_boundaries, 'frobenius_norm')
        print_block_statistics(stats, row_boundaries, col_boundaries, 'avg_diag')
        print_block_statistics(stats, row_boundaries, col_boundaries, 'nnz')
        print_block_statistics(stats, row_boundaries, col_boundaries, 'avg_nnz_per_row')
        #print_block_statistics(stats, row_boundaries, col_boundaries, 'max_val')
        print_block_statistics(stats, row_boundaries, col_boundaries, 'max_abs')
        print_block_statistics(stats, row_boundaries, col_boundaries, 'min_abs')
    
    # Determine output shape (based on max indices after reordering)
    new_nrows = int(new_rows.max()) + 1 if new_rows.size > 0 else 0
    new_ncols = int(new_cols.max()) + 1 if new_cols.size > 0 else 0
    
    # Write output - write as single text file (could be enhanced to write binary or multiple parts)
    output_mat_file = os.path.join(args.output_dir, 'IJ.out.A.00000')
    write_matrix_coordinate(output_mat_file, new_rows, new_cols, new_vals, 
                           shape=(new_nrows, new_ncols))
    
    # Also copy dofmap if needed (with reordered indices)
    # The dofmap should be reordered according to row mapping
    new_dofmap = dofmap[np.argsort(row_mapping)]
    output_dof_file = os.path.join(args.output_dir, os.path.basename(dofmap_file))
    with open(output_dof_file, 'w') as f:
        f.write(f"{len(new_dofmap)}\n")
        for dof_idx in new_dofmap:
            f.write(f"{dof_idx}\n")
    
    # Write info file with partitioning information
    info_file = os.path.join(args.output_dir, 'partitioning.info')
    write_info_file(info_file, dofmap, row_mapping, args.row_order, 
                   col_mapping, args.col_order, nrows)
    
    if args.verbose:
        print(f"\n  Wrote reordered matrix: {output_mat_file}")
        print(f"  Wrote reordered dofmap: {output_dof_file}")
        print(f"  Wrote partitioning info: {info_file}")
    
    print(f"\nDone! Reordered matrices written to: {args.output_dir}")


if __name__ == '__main__':
    main()

