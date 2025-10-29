# hypredrive datasets

**Important:** The datasets are not included in the repository. You must download them from Zenodo before running examples that use these datasets.

## Obtaining the datasets

The datasets are hosted on Zenodo at https://zenodo.org/records/17471036.

The most convenient way to download and extract all datasets is using the CMake `data` target:

```bash
cmake --build <build-dir> --target data
```

Alternatively, you can download the datasets manually from the Zenodo record and extract them into this directory.

This directory hosts small datasets used by hypredrive examples and tests. All matrices and vectors are provided in HYPRE IJ format using a consistent naming convention:
- Matrix prefix: `IJ.out.A`
- RHS prefix: `IJ.out.b`
- Partitioned runs store per-rank parts in subdirectories like `np4/`, with file suffixes `.00000[.bin]`, `.00001[.bin]`, etc.
- Single-rank ASCII uses files without `.bin`; multi-rank (and some single-rank cases) use binary `.bin`.

## Datasets

- ps3d10pt7: 3D Laplacian (scalar) with the standard 7‑point finite difference stencil
- compflow6k: compositional multiphase flow problem simulated in GEOS.
- poromech2k: sequence of linear systems a multiphase poromechanics simulation in GEOS.

## Notes

- Binary IJ files encode indices and values with widths recorded in the file header; hypredrive detects these automatically.
- For provenance of third‑party inputs, see the dataset‑specific READMEs.