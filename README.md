[![DOI](https://joss.theoj.org/papers/10.21105/joss.06654/status.svg)](https://doi.org/10.21105/joss.06654)
[![Docs](https://github.com/hypre-space/hypredrive/workflows/Docs/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/docs.yml)
[![Coverage](https://github.com/hypre-space/hypredrive/workflows/Coverage/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/coverage.yml)
[![Analysis](https://github.com/hypre-space/hypredrive/workflows/Static%20Analysis/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/analysis.yml)
[![CI](https://github.com/hypre-space/hypredrive/workflows/CI/badge.svg)](https://github.com/hypre-space/hypredrive/actions/workflows/ci.yml)


# *hypredrive*

High-level interface for solving linear systems with hypre, providing a user-friendly way to leverage its functionalities. Key features are:

1. **YAML Input**: Accepts configuration parameters written in the structured and human-readable YAML format.
2. **Intuitive Interface**: Offers a clear and concise API that encapsulates the functionalities of *hypre*.
3. **Prototyping**: Establishes a quick prototyping framework various solver/preconditioner combinations.
4. **Testing**: Enables the construction of an integrated testing framework, accommodating problems from applications built on hypre.

## Getting Started

The instructions for building *hypredrive* using CMake are given below:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${PWD}/install \
      -B build -S .
cmake --build build --parallel
cmake --build build --target check
cmake --install build
```

Notes:
1. [hypre](https://github.com/hypre-space/hypre) is automatically fetched and built from source if not found.
   To use a pre-built HYPRE, specify `-DHYPRE_ROOT=<path>`.
2. To download example datasets, use `cmake --build build --target data`.
3. For full documentation and installation options, see the [user's manual](https://hypredrive.readthedocs.io/en/latest/installation.html).

## Examples

The user's manual has a detailed section about running examples. For a quick-start, try
running the first example from the top-level folder:

```
$ mpirun -np 1 ./hypredrive examples/ex1.yml -q

Date and time: YYYY-MM-DD HH:MM:SS

Using HYPRE_DEVELOP_STRING: HYPRE_VERSION_GOES_HERE

Running on 1 MPI rank
---------------------------------------------------------------------------------------
general:
  use_millisec: on
linear_system:
  rhs_filename: data/ps3d10pt7/np1/IJ.out.b
  matrix_filename: data/ps3d10pt7/np1/IJ.out.A
solver: pcg
preconditioner: amg
---------------------------------------------------------------------------------------
=======================================================================================
Solving linear system #0 with 1000 rows and 6400 nonzeros...
=======================================================================================


STATISTICS SUMMARY:

+--------+-------------+-------------+-------------+------------+------------+--------+
|        |    LS build |       setup |       solve |    initial |   relative |        |
|  Entry |  times [ms] |  times [ms] |  times [ms] |  res. norm |  res. norm |  iters |
+--------+-------------+-------------+-------------+------------+------------+--------+
|      0 |       1.500 |       1.286 |       0.262 |   3.16e+01 |   4.98e-08 |      6 |
+--------+-------------+-------------+-------------+------------+------------+--------+

Date and time: YYYY-MM-DD HH:MM:SS
${HYPREDRIVE_PATH}/hypredrive done!
```

## Documentation

Check *hypredrive*'s manual [here](https://hypredrive.readthedocs.io/en/latest/).

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING.md) for details on contributing to *hypredrive*,
including the process for submitting pull requests to us.

## Citing

If you are referencing *hypredrive* in a publication, please cite the following paper:

    Magri, V. A. P. (2024). Hypredrive: High-level interface for solving linear systems
    with hypre. Journal of Open Source Software,
    9(98), 6654. https://doi.org/10.21105/joss.06654

You can easily obtain the citation in APA or BibTeX format directly from GitHub. Navigate
to the "Cite this repository" option located in the "About" section. Alternatively, you can
find the raw BibTeX format in the comments section of the [CITATION](CITATION.cff) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
details. All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-861860
