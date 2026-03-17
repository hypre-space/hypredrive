# Scripts

This directory contains utility scripts for validation, data preparation, plotting, profiling, and output analysis.

## Bash Scripts

- `check_private_prefix.sh`: Checks that private `libHYPREDRV` callables use the `hypredrv_` prefix and can optionally auto-fix offending names.
- `list_public_apis.sh`: Generates a sorted list of all public `HYPREDRV_` API function names by parsing `include/HYPREDRV.h`. Use `--check` to validate that all public APIs start with `HYPREDRV_`.
- `compare_output.sh`: Normalizes timestamps, versions, and paths before diffing an output file against a reference output.
- `download_and_extract.sh`: Downloads a tarball, verifies its MD5 checksum, and extracts it into a destination directory.
- `generate_example_output.sh`: Runs the example YAML inputs and regenerates normalized reference outputs for the examples.
- `perf_laplacian.sh`: Builds and runs the standalone Laplacian driver across hypre versions with optional perf/Caliper profiling and scaling summaries.

## Python Scripts

- `analyze_caliper.py`: Parses Caliper runtime reports and builds an interactive hierarchical timing visualization.
- `analyze_coverage.py`: Summarizes a `gcovr` XML coverage report with low-coverage files and overall coverage statistics.
- `analyze_statistics.py`: Parses statistics outputs and generates plots for iterations, timings, throughput, and related solver metrics.
- `eigplot.py`: Reads eigenvalue files and plots the spectrum with summary statistics and optional histogram or inset views.
- `spmat_reorder.py`: Reorders HYPRE IJ matrices by dofmap index groups and can write reordered matrices and partition info.
- `spyplot.py`: Reads HYPRE matrix files and generates sparse matrix spy plots from text or binary matrix parts.
