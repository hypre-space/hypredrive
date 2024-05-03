---
title: '*Hypredrive*: High-level interface for solving linear systems with *hypre*'
tags:
  - linear solvers
  - preconditioning
  - algebraic multigrid
  - high performance computing
authors:
  - name: Victor A. P. Magri
    orcid: 0000-0002-3389-523X
    affiliation: 1
affiliations:
 - name: Lawrence Livermore National Laboratory, CA, USA
   index: 1
date: 30 March 2024
bibliography: paper.bib
---

# Summary

Solving sparse linear systems of equations is an essential problem for many application
codes in computational science and engineering (CSE). Commonly used mathematical libraries
for tackling these problems are Trilinos [@trilinos], PETSc [@petsc] and *hypre*
[@hypre]. The first two provide means of quickly testing different solution strategies via
input files, while such a capability is missing in *hypre*. *Hypredrive* aims to fill this
gap by providing a simple and user-friendly interface to *hypre* [@hypre]. Particularly,
inspired by the solver composability support in PETSc via `.petscrc` files [@petscrc] and
the flexible configuration framework used in spack [@spack], *hypredrive* allows users to
easily configure and switch solver options in *hypre* through input files in @YAML format,
making experimentation with different solver techniques more accessible to researchers and
software developers who work with numerical simulation codes.

# Statement of need

*Hypre* is a widely used and efficient linear solver package; however, the complexity
associated with its direct use might limit *hypre* users and application developers to
explore with different solver options. *Hypredrive* bridges this gap by providing a
high-level and lightweight interface to *hypre*, encapsulating its complexity while
retaining its capabilities with minimal computational overhead.


# Software capabilities

*Hypredrive* is a software package written in C that includes a library with APIs designed
to simplify the interaction with *hypre* and an executable driver for performing the
solution of linear systems defined via YAML input files. The types of linear systems
solvable with *hypredrive* are determined by *hypre*. Key features of the software are:

* **Encapsulation**: `libHYPREDRV` wraps the function calls for building solvers and
  preconditioners in *hypre* through an intuitive YAML interface driven by configuration
  files. This design ensures a straightforward way of setting up solvers in *hypre* and
  sharing options without recompiling the user's application code.


* **Prototyping**: `hypredrive` allows users to prototype rapidly, comparing the
  performance of various solver options and tweaking parameters directly through the YAML
  configuration file. This flexibility encourages experimentation, helping users identify
  the most effective solver strategies for their specific problems.


* **Testing**: `hypredrive` enables the creation of an integrated testing framework to
  evaluate solvers against a set of predefined linear systems. This feature lets users
  understand whether updates to *hypre* lead to different solver convergence or
  performance for their problems of interest.

# Example usage

As an example usage of `hypredrive`, consider the solution of a linear system arising from
a seven points finite difference discretization of the Laplace equation on a `10x10x10`
cartesian grid. Both sparse matrix and right hand side vector are stored at
`data/ps3d10pt7/np1/`. For solving this linear system with algebraic multigrid (BoomerAMG)
as a preconditioner to the conjugate gradient iterative solver, the YAML input file
`input.yml` would look like:

```yaml
linear_system:
  rhs_filename: data/ps3d10pt7/np1/IJ.out.b
  matrix_filename: data/ps3d10pt7/np1/IJ.out.A
solver: pcg
preconditioner: amg
```

while `hypredrive` can be executed via

```bash
$ mpirun -np 1 ./hypredrive input.yml
```

yielding an output that looks like:

```
Date and time: YYYY-MM-DD HH:MM:SS

Using HYPRE_DEVELOP_STRING: HYPRE_VERSION_GOES_HERE

Running on 1 MPI rank
-------------------------------------------------------------------------------
linear_system:
  rhs_filename: data/ps3d10pt7/np1/IJ.out.b
  matrix_filename: data/ps3d10pt7/np1/IJ.out.A
solver: pcg
preconditioner: amg
-------------------------------------------------------------------------------
===============================================================================
Solving linear system #0 with 1000 rows and 6400 nonzeros...
===============================================================================


STATISTICS SUMMARY:

+----------+-------------+-------------+-------------+-------------+----------+
|          |    LS build |       setup |       solve |    relative |          |
|    Entry |       times |       times |       times |   res. norm |    iters |
+----------+-------------+-------------+-------------+-------------+----------+
|        0 |       0.004 |       0.002 |       0.001 |    4.98e-08 |        6 |
+----------+-------------+-------------+-------------+-------------+----------+

Date and time: YYYY-MM-DD HH:MM:SS
${HYPREDRIVE_PATH}/hypredrive done!
```

This example shows the minimal set of options in `hypredrive`'s input file needed for
running the executable. Specific parameter/value pairs for controlling the setup of
preconditioners and solvers can be added to the input file, leading to different
convergence behaviors. For a complete list of available parameters and more detailed input
file examples, see [*hypredrive*'s manual](https://hypredrive.readthedocs.io/en/latest/).

# Acknowledgements

This work performed under the auspices of the U.S. Department of Energy by Lawrence
Livermore National Laboratory under Contract DE-AC52-07NA27344.

# References
