.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Interfaces:

Interfaces
==========

Hypredrive can be used through several entry points:

- The ``hypredrive-cli`` driver reads YAML input files and matrix/RHS data from disk.
- The C library API exposes ``HYPREDRV_t`` handles for applications that already manage
  their own HYPRE objects or sparse data.
- Language interfaces provide higher-level bindings for scripting, prototyping, and
  application integration.

Language interfaces live under ``interfaces/`` in the source tree. Each language-specific
page documents installation, build requirements, and the public API exposed by that
binding.

.. toctree::
   :maxdepth: 2

   interfaces_python
