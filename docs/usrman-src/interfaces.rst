.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Interfaces:

Interfaces
==========

`hypredrive` has several entry points:

- The ``hypredrive-cli`` driver reads YAML input files and matrix/RHS data from disk.
- The C library API exposes ``HYPREDRV_t`` handles for applications that already manage
  their own HYPRE objects or sparse data.
- Language interfaces provide higher-level bindings for scripting, prototyping, and
  application integration.

Language interfaces live under ``interfaces/`` in the source tree. Each language-specific
page documents installation, build requirements, and the public API exposed by that
binding. Each page includes a minimal serial solve of the same two-row system,
with solution ``[1, 1]``, followed by configuration and MPI usage where supported.

.. toctree::
   :maxdepth: 2

   interfaces_cpp
   interfaces_fortran
   interfaces_python
   interfaces_matlab
   interfaces_julia
