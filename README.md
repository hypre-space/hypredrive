[![DOI](https://joss.theoj.org/papers/10.21105/joss.06654/status.svg)](https://doi.org/10.21105/joss.06654)

# *hypredrive*

High-level interface for solving linear systems with hypre, providing a user-friendly way to leverage its functionalities. Key features are:

1. **YAML Input**: Accepts configuration parameters written in the structured and human-readable YAML format.
2. **Intuitive Interface**: Offers a clear and concise API, encapsulating the functionalities of *hypre* to ensure user-friendly interactions.
3. **Prototyping**: Establishes a quick prototyping framework, facilitating the exploration of various solver/preconditioner setups.
4. **Testing**: Enables the construction of an integrated testing framework, accommodating problems from applications built on hypre.

## Getting Started

The instructions for building *hypredrive* are given below:

```
 $ autoreconf -i
 $ ./configure --prefix=${PWD}/install --with-hypre-dir=${HYPRE_INSTALL_DIR}
 $ make all
 $ make check
 $ make install
```

Note:
1. The first step must be executed only once after cloning this repository.
2. [hypre](https://github.com/hypre-space/hypre) needs to be installed at
   `${HYPRE_INSTALL_DIR}`.
3. For GPU support, add `--with-cuda` (NVIDIA GPUs) or `--with-hip` (AMD GPUs) to
   `./configure`.

## Examples

The user's manual has a detailed section about running examples. For a quick-start, try
running the first example from the top-level folder:

```
$ mpirun -np 1 ./hypredrive examples/ex1.yml

Date and time: YYYY-MM-DD HH:MM:SS

Using HYPRE_DEVELOP_STRING: HYPRE_VERSION_GOES_HERE

Running on 1 MPI rank
------------------------------------------------------------------------------------
general:
  use_millisec: on
linear_system:
  rhs_filename: data/ps3d10pt7/np1/IJ.out.b
  matrix_filename: data/ps3d10pt7/np1/IJ.out.A
solver: pcg
preconditioner: amg
------------------------------------------------------------------------------------
====================================================================================
Solving linear system #0 with 1000 rows and 6400 nonzeros...
====================================================================================


STATISTICS SUMMARY:

+------------+-------------+-------------+-------------+-------------+-------------+
|            |    LS build |       setup |       solve |    relative |             |
|      Entry |  times [ms] |  times [ms] |  times [ms] |   res. norm |       iters |
+------------+-------------+-------------+-------------+-------------+-------------+
|          0 |       2.706 |       2.846 |       1.361 |    4.98e-08 |           6 |
+------------+-------------+-------------+-------------+-------------+-------------+

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
