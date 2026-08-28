# hypredrive datasets

This directory hosts small datasets used by hypredrive examples and tests. All matrices and vectors are provided in HYPRE IJ format using a consistent naming convention:
- Matrix prefix: `IJ.out.A`
- RHS prefix: `IJ.out.b`
- Partitioned runs store per-rank parts in subdirectories like `np4/`, with file suffixes `.00000[.bin]`, `.00001[.bin]`, etc.
- Single-rank ASCII uses files without `.bin`; multi-rank (and some single-rank cases) use binary `.bin`.

**Important:** The datasets are not included in the repository. You must download them from Zenodo before running examples that use these datasets.

## Obtaining the datasets

The datasets are hosted on Zenodo at https://zenodo.org/records/22116856.

The most convenient way to download and extract all datasets is using the CMake `data` target:

```bash
cmake --build <build-dir> --target data
```

Alternatively, you can download the datasets manually from the Zenodo record and extract them into this directory.

## Datasets

- ps3d10pt7: 3D Laplacian (scalar) with the standard 7‑point finite difference stencil.
- compflow6k: compositional multiphase flow problem simulated in GEOS.
- poromech2k: sequence of linear systems from a multiphase poromechanics simulation in GEOS.
- spres1k: single phase flow in reservoir coupled with wells simulated in GEOS.
- tspres1k: thermal single phase flow in reservoir coupled with wells simulated in GEOS.
- sphyb1k: single phase flow using a hybrid formulation simulated in GEOS.
- spreshyb1k: single phase flow in reservoir using a hybrid formulation coupled with wells simulated in GEOS.
- spporo1k: single phase poromechanics simulated in GEOS.
- tspporo1k: thermal single phase poromechanics simulated in GEOS.
- hspporo1k: hybrid single phase poromechanics simulated in GEOS.
- spporoef1k: single phase poromechanics with embedded fractures simulated in GEOS.
- spporocf1k: single phase poromechanics with conforming fractures simulated in GEOS.
- spporores1k: single phase poromechanics in reservoir coupled with wells simulated in GEOS.
- tspporores1k: thermal single phase poromechanics in reservoir coupled with wells simulated in GEOS.
- cmpf1k: compositional multiphase flow simulated in GEOS.
- cmphyb1k: compositional multiphase flow using a hybrid formulation simulated in GEOS.
- cmpres1k: compositional multiphase flow in reservoir coupled with wells simulated in GEOS.
- cmpreshyb1k: compositional multiphase flow in reservoir using a hybrid formulation coupled with wells simulated in GEOS.
- immf1k: immiscible multiphase flow simulated in GEOS.
- rcmpobl1k: reactive compositional multiphase flow using operator-based linearization simulated in GEOS.
- tcmpf1k: thermal compositional multiphase flow simulated in GEOS.
- tcmpres1k: thermal compositional multiphase flow in reservoir coupled with wells simulated in GEOS.
- mpporo1k: multiphase poromechanics simulated in GEOS.
- mpporores1k: multiphase poromechanics in reservoir coupled with wells simulated in GEOS.
- tmpporo1k: thermal multiphase poromechanics simulated in GEOS.
- hydrofrac1k: hydrofracture simulation in GEOS.
- lcontact1k: Lagrangian contact mechanics simulated in GEOS.
- alcontact1k: augmented Lagrangian contact mechanics simulated in GEOS.
- lcontactbs1k: Lagrangian contact mechanics with bubble stabilization simulated in GEOS.
- smef1k: solid mechanics with embedded fractures simulated in GEOS.
- mhd3ddbdt1k: 3D coupled full-induction magnetohydrodynamics from a duct flow problem simulated in VERTEX-CFD.
- mhd2dldc1k: 2D coupled full-induction magnetohydrodynamics from a lid-driven cavity problem simulated in VERTEX-CFD.

## Notes

- Binary IJ files encode indices and values with widths recorded in the file header; hypredrive detects these automatically.
- For provenance of third-party inputs, see the dataset-specific READMEs.
