[![DOI](https://joss.theoj.org/papers/10.21105/joss.06654/status.svg)](https://doi.org/10.21105/joss.06654)
[![Docs](https://github.com/hypre-space/hypredrive/workflows/Docs/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/docs.yml)
[![Coverage](https://github.com/hypre-space/hypredrive/workflows/Code%20Coverage/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/coverage.yml)
[![Analysis](https://github.com/hypre-space/hypredrive/workflows/Code%20Analysis/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/analysis.yml)
[![CI](https://github.com/hypre-space/hypredrive/workflows/CI/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/ci.yml)

# hypredrive

High-level interface to [hypre](https://github.com/hypre-space/hypre) for solving sparse linear systems of equations. It can be used as:

- **Driver**: `hypredrive-cli` executable with YAML input files, for example [`examples/ex2.yml`](examples/ex2.yml)
- **Library**: C API via [`include/HYPREDRV.h`](include/HYPREDRV.h) driven by YAML input (file or in-memory), see [`examples/src/`](examples/src/)

## Build

```bash
git clone --depth 1 https://github.com/hypre-space/hypredrive.git
cmake -S hypredrive -B build && cmake --build build --parallel
```

- hypre is fetched automatically if not found. Pass `-DHYPRE_ROOT=<path>` to use an existing install.
- Check [installation instructions](https://hypredrive.readthedocs.io/en/latest/installation.html) for details, including the available library options.

## Examples

**Driver** -- solve a system from a YAML file (see [examples files](https://hypredrive.readthedocs.io/en/latest/driver_examples.html) in the docs).

```bash
mpirun -np 1 ./build/hypredrive-cli examples/ex2.yml -q
```

**Library** -- call the API from your own code (see [example drivers](https://hypredrive.readthedocs.io/en/latest/library_examples.html) in the docs).

```C
   // 1. Setup hypredrive options from YAML input
   HYPREDRV_t hdrv = NULL;
   HYPREDRV_Initialize();
   HYPREDRV_Create(MPI_COMM_WORLD, &hdrv);
   HYPREDRV_SetLibraryMode(hdrv);
   HYPREDRV_InputArgsParse(1, yaml_text, hdrv); // Solver options are parsed here

   // 2. Setup linear system ("A" and "b" are built previously)
   HYPREDRV_LinearSystemSetMatrix(hdrv, (HYPRE_Matrix) A);
   HYPREDRV_LinearSystemSetRHS(hdrv, (HYPRE_Vector) b);

   // 3. Solve lifecycle (Solve for "x" in "A x = b")
   HYPREDRV_LinearSolverCreate(hdrv);
   HYPREDRV_LinearSolverSetup(hdrv);
   HYPREDRV_LinearSolverApply(hdrv);
   HYPREDRV_LinearSolverDestroy(hdrv);

   // 4. Cleanup
   HYPREDRV_Destroy(&hdrv);
   HYPREDRV_Finalize();
```

Documentation is available at [hypredrive.readthedocs.io](https://hypredrive.readthedocs.io/en/latest/). For contribution instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

> Magri, V. A. P. (2024). Hypredrive: High-level interface for solving linear systems with hypre. *JOSS*, 9(98), 6654. <https://doi.org/10.21105/joss.06654>

See also [CITATION.cff](CITATION.cff).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-861860
