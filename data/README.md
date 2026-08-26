# hypredrive datasets

This directory hosts small datasets used by hypredrive examples and tests. All matrices and vectors are provided in HYPRE IJ format using a consistent naming convention:
- Matrix prefix: `IJ.out.A`
- RHS prefix: `IJ.out.b`
- Partitioned runs store per-rank parts in subdirectories like `np4/`, with file suffixes `.00000[.bin]`, `.00001[.bin]`, etc.
- Single-rank ASCII uses files without `.bin`; multi-rank (and some single-rank cases) use binary `.bin`.

**Important:** The datasets are not included in the repository. You must download them from Zenodo before running examples that use these datasets.

## Obtaining the datasets

The datasets are hosted on Zenodo at https://zenodo.org/records/17471036.

The most convenient way to download and extract all datasets is using the CMake `data` target:

```bash
cmake --build <build-dir> --target data
```

Alternatively, you can download the datasets manually from the Zenodo record and extract them into this directory.

For release preparation, run `scripts/pack_datasets.py` to create one
deterministic tarball per dataset, plus MD5/SHA256 checksums and a v0.3.0
manifest under `/tmp/hypredrive-datasets-v0.3.0`.

## Datasets

- ps3d10pt7: 3D Laplacian (scalar) with the standard 7‑point finite difference stencil
- compflow6k: compositional multiphase flow problem simulated in GEOS.
- poromech2k: sequence of linear systems a multiphase poromechanics simulation in GEOS.
- MGR strategy datasets: 27 small offline cases, one for every GEOS MGR strategy, with np1 and np4 variants.

The MGR strategy datasets are not checked into the hypredrive sources. Include
their top-level short-named directories (for example, `spres1k` and
`cmpres1k`) in the same Zenodo archive as the other datasets. The generated
`mgr_strategy_cases.md` file lists every case, its GEOS strategy, and its total
DOF count. Run serial cases from their `np1/` directory with
`hypredrive-cli input.yml`, or run the four-rank `np4/` variant with
`mpiexec -n 4 hypredrive-cli input.yml`.

## Notes

- Binary IJ files encode indices and values with widths recorded in the file header; hypredrive detects these automatically.
- For provenance of third‑party inputs, see the dataset-specific READMEs.
