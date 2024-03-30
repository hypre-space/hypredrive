---
title: 'Hypredrive: High-level interface for solving linear systems with hypre'
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
date: 1 April 2024
bibliography: paper.bib
---

# Summary

Solving sparse linear systems of equations is an essential problem to many application
codes in computational science and engineering (CSE). *Hypredrive* aims to facilitate this
problem via a simple and user-friendly interface to *hypre* [@hypre], a well-established
package featuring multigrid methods. *Hypredrive* allows users to easily configure and
switch solver options in *hypre* through YAML input files, making experimentation with
different solver techniques more accessible to researchers and software developers working
with numerical simulation codes.

# Statement of need

*Hypre* is a widely used and efficient linear solver package; however, the complexity
associated with its direct use may make it difficult to experimentat with different solver
options. *Hypredrive* bridges this gap by providing a high-level and lightweight interface
to *hypre*, encapsulating its complexity while retaining its capabilities with minimal
computational overhead.

# Software capabilities

*Hypredrive* is a software written in C that includes a library with APIs designed to
simplify the interaction with *hypre* and an executable for performing the solution of
linear systems defined via YAML input files. Key features of the software are:

* **Encapsulation**: `libHYPREDRV` wraps the function calls for building solvers and
  preconditioners in *hypre* through an intuitive YAML interface driven by configuration
  files.


* **Prototyping**: `hypredrive` allows users to prototype rapidly, comparing the
  performance of various solver options updated directly through the YAML configuration file.


* **Testing**: `hypredrive` enables the creation of an integrated testing framework to
  evaluate solvers against a set of predefined linear systems.

For instructions on installation, input file structure, and examples, see [*hypredrive*'s
manual](https://hypredrive.readthedocs.io/en/latest/).

# Acknowledgements

This work performed under the auspices of the U.S. Department of Energy by Lawrence
Livermore National Laboratory under Contract DE-AC52-07NA27344.

# References
