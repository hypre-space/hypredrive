.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _InputFileStructure:

Input Structure (YAML)
======================

`hypredrive` uses YAML for its parameters and settings. The driver reads YAML from a
file. Application code can pass a YAML string to ``HYPREDRV_InputArgsParse``. Both forms
use the same YAML structure.

Most keywords are optional and have default values. In a file-based driver workflow,
the ``preconditioner`` section is required; ``solver`` defaults to ``pcg`` and
``linear_system`` must provide a file-backed system. Library callers can omit the
``linear_system`` section when they attach in-memory matrices and vectors through the
C API. The reference marks conditionally required keywords as `possibly required`.

.. note::

   The `hypredrive` YAML parser is not case-sensitive. Keys and values can use lowercase,
   uppercase, or mixed-case text.

.. tip::

   Use the driver option ``-h`` or ``--help`` to browse the schema. It prints valid keys,
   accepted values, and nested topics without a solve. See :ref:`CLIHelp`.

General Settings
----------------

The ``general`` section contains global settings that apply to the entire execution of
`hypredrive`. This section is optional.

- ``warmup`` - A value of `yes` runs a warmup before the timed operation. This
  gives more accurate timing measurements. A value of `no` skips the warmup.
  The default is `no`.

- ``name`` - Optional display label for the current ``HYPREDRV_t`` object. When set,
  statistics banners use it in messages such as ``STATISTICS SUMMARY for flow-solver:``.
  The default is empty, which preserves the unlabeled banner.

- ``statistics`` - Controls the verbosity of statistics reporting. Accepts integer values
  or boolean strings (`yes`/`no`, `on`/`off`, `true`/`false`). The default value is `1`
  (or `yes`). Available levels:

  - ``0`` (or `no`/`off`/`false`) - No statistics reporting.
  - ``1`` (or `yes`/`on`/`true`) - Display the basic statistics summary table with
    execution times, residual norms, and iteration counts for each solve entry.
  - ``2`` - Display the basic summary and an aggregate table. The aggregate includes the
    minimum, maximum, average, standard deviation, and total. It covers build, setup, and
    solve times, plus iteration counts. Multiple entries, such as repetitions, are required.

  In library mode, hypredrive prints the configured summary on rank 0 when the
  application destroys the owning ``HYPREDRV_t`` object. Applications can call
  ``HYPREDRV_StatsPrint`` earlier for an additional snapshot.

- ``statistics_filename`` - Optional file path for statistics output. An empty path
  sends output to ``stdout`` and is the default. A file path makes hypredrive append
  each statistics snapshot to that file. If hypredrive cannot open the file, it
  writes a warning to ``stderr`` and sends that snapshot to ``stdout``.

- ``use_millisec`` - Show timings on the statistics summary table in milliseconds. The
  default value is `no`, which uses seconds instead.

- ``device_lazy_init`` - Defer GPU device and vendor-library initialization until the
  first device operation. Values: ``yes`` / ``no``. The default is ``no``. Set this
  to ``yes`` when an application needs device initialization deferred until the first
  device operation.

- ``print_config_params`` - Print the parsed YAML tree to stdout on rank 0 after input
  parsing. Values: ``yes`` / ``no``. Default: ``yes`` in driver mode and ``no`` in
  library mode.

- ``num_repetitions`` - Number of times that hypredrive repeats the operation.
  Use repetitions for benchmarks and profiles. The default is `1`.

- ``dev_pool_size`` - Initial size of the umpire device memory pool. hypre ignores
  this parameter without umpire support. The default is `2 GB`.

- ``uvm_pool_size`` - Initial size of the umpire unified virtual memory pool. hypre
  ignores this parameter without umpire support. The default is `2 GB`.

- ``host_pool_size`` - Initial size of the umpire host memory pool. hypre ignores
  this parameter without umpire support. The default is `2 GB`.

- ``pinned_pool_size`` - Initial size of the umpire pinned host memory pool. hypre
  ignores this parameter without umpire support. The default is `0.1 GB` (100 MB).

- ``exec_policy`` - Selects the ``host`` (CPU) or ``device`` (GPU) for computations.
  Without GPU support, ``host`` is the only valid value. Otherwise, the default is
  ``device``.

- ``use_vendor_spgemm`` - Use vendor-optimized sparse matrix-matrix multiplication (SpGEMM)
  kernels when available. Values: ``yes`` / ``no``. Default: ``yes`` on GPU-enabled builds
  and ``no`` otherwise.

- ``use_vendor_spmv`` - Use vendor-optimized sparse matrix-vector multiplication (SpMV)
  kernels when available. Values: ``yes`` / ``no``. Default: ``yes`` on GPU-enabled builds
  and ``no`` otherwise.


This example shows the ``general`` section:

.. code-block:: yaml

    general:
      warmup: no
      statistics: yes
      use_millisec: no
      device_lazy_init: no
      print_config_params: yes
      num_repetitions: 1
      dev_pool_size: 2.0
      uvm_pool_size: 2.0
      host_pool_size: 2.0
      pinned_pool_size: 0.1
      exec_policy: device
      use_vendor_spgemm: yes
      use_vendor_spmv: yes

Linear System
-------------

The optional ``linear_system`` section describes the linear system that
`hypredrive` solves. A file-based driver run needs a matrix source here; a library
caller may instead attach the matrix and vectors programmatically.

- ``type`` - The format of the linear system matrix. Available options are ``online``,
  ``ij``, ``parcsr``, and ``mtx``. The default value is ``ij``. The file reader supports
  ``ij`` and ``mtx``; ``online`` and ``parcsr`` are for programmatic/in-memory workflows.

- ``matrix_filename`` - (Possibly required) Filename of the linear system matrix. A
  file-based run can use this, ``matrix_basename``, or ``sequence_filename`` as its
  matrix source. This parameter has no default. Library callers can omit all file
  sources and use ``HYPREDRV_LinearSystemSetMatrix`` or
  ``HYPREDRV_LinearSystemSetMatrixFromCSR``.

- ``precmat_filename`` - Filename of the matrix that hypredrive uses to compute the
  preconditioner. ``precmat_basename`` and ``precmat_sequence_filename`` are alternative
  sources. The original system matrix is the default.
  In the C API, ``HYPREDRV_LinearSystemSetPrecMatrix(h, mat)`` overrides this file-based
  path when ``mat`` is non-NULL.

- ``rhs_filename`` - (Possibly required) Filename of the right-hand side vector.
  ``rhs_basename`` is an alternative for a sequence of files. These parameters have no
  default and a file source is required when ``rhs_mode`` is ``file``.

- ``x0_filename`` - (Possibly required) Filename of the initial guess for the
  left-hand side vector. This parameter has no default and is required when
  ``init_guess_mode`` is ``file``.
  When present, it is read as the literal path supplied; it takes precedence over
  ``init_guess_mode`` and sequence filename resolution. In the C API,
  ``HYPREDRV_LinearSystemSetInitialGuess(h, vec)`` overrides this file/default path
  when ``vec`` is non-NULL.

- ``xref_filename`` - (Optional) The filename of the reference solution vector used by
  tagged residual/error reporting. ``xref_basename`` is an alternative file source. An
  explicit filename or basename replaces a reference generated by ``rhs_mode: randsol``.
  In the C API,
  ``HYPREDRV_LinearSystemSetReferenceSolution(h, vec)`` overrides file/default behavior
  when ``vec`` is non-NULL.

- ``sol_filename`` - Accepted for compatibility, but it does not currently select a
  solution output path. Scheduled ``print_system`` dumps use ``solution.out`` in the
  selected output directory, and ``HYPREDRV_LinearSystemPrint`` writes the matrix, RHS,
  and dofmap rather than using this key.

.. _linear_system_dofmap:

Degrees of Freedom Map
~~~~~~~~~~~~~~~~~~~~~~

- ``dofmap_filename`` - (Possibly required) Filename of the degree-of-freedom
  mapping array (`dofmap`). This parameter has no default and is required for
  the ``mgr`` preconditioner.

- ``dof_labels`` - Optional mapping from symbolic degree-of-freedom labels to integer
  tags. It can be written as a YAML block or flow mapping. For example:

  .. code-block:: yaml

      linear_system:
        dof_labels:
          v_x: 0
          v_y: 1
          p: 2

  Keys are normalized to lowercase. MGR ``f_dofs`` entries can use these labels as
  well as integer tags. A symbolic label requires this mapping and an unknown label
  is an error.

- ``init_guess_mode`` - Choice of initial guess vector. Available options are:

  - ``zeros``: Generates a vector of zeros.
  - ``ones``: Generates a vector of ones.
  - ``random``: Generates a vector of random numbers between `0` and `1`.
  - ``file``: Reads the vector from a file.
  - ``previous``: Reuses the solution of the previous linear solve. When no compatible
    previous solution exists (first solve, or a system with a different size or
    distribution), it falls back to a vector of zeros.

  The default value is ``zeros``. When ``x0_filename`` is present, `hypredrive` reads the
  literal path supplied there, regardless of this mode.

  .. note::
     In the library API, passing ``NULL`` to
     ``HYPREDRV_LinearSystemSetInitialGuess`` /
     ``HYPREDRV_LinearSystemSetReferenceSolution`` /
     ``HYPREDRV_LinearSystemSetPrecMatrix`` preserves the file/default behavior described
     in this section.

- ``rhs_mode`` - Selects the source of the right-hand side vector. Available options are:

  - ``zeros``: Generates a vector of zeros.
  - ``ones``: Generates a vector of ones.
  - ``random``: Generates a vector of random numbers between `0` and `1`.
  - ``file``: Reads the vector from ``rhs_filename``, ``rhs_basename``, or a sequence
    container. This is the default.
  - ``randsol``: Generates a random reference solution, computes ``b = A*xref``, and
    stores that reference for tagged error checks.

  When the mode is not ``file``, hypredrive does not read ``rhs_filename`` or
  ``rhs_basename``. The generated random vectors use the fixed seed `2023`.

- ``dirname`` - (Possibly required) Name of the top-level data directory. Use this
  parameter to remove repeated directory names from file paths. This parameter
  has no default.

- ``sequence_filename`` - (Optional) Path to a lossless compressed sequence
  container from ``hypredrive-lsseq``. When set, hypredrive reads the matrix,
  file-mode right-hand side, and optional dofmap from the container. It does not use
  the directory, filename, or basename fields. With a non-file ``rhs_mode``, the RHS
  is generated instead. Container metadata gives the number of systems.

- ``precmat_sequence_filename`` - (Optional) Path to a lossless compressed sequence
  container whose matrix is used to build the preconditioner. This avoids unpacking
  an auxiliary matrix archive. It cannot be combined with ``precmat_filename`` or
  ``precmat_basename``.

- ``precmat_sequence_system_id`` - (Optional) Zero-based matrix index in
  ``precmat_sequence_filename``. The default value, ``-1``, selects the current
  linear-system index. Set this to ``0`` to reuse the only matrix in a one-system
  archive while solving another sequence.

- ``matrix_basename`` - (Possibly required) Common prefix for matrix filenames.
  Use it to solve multiple matrices in a shared directory. This parameter has
  no default.

- ``precmat_basename`` - (Possibly required) Common prefix for the filenames of
  preconditioner matrices. By default, hypredrive uses the original system
  matrices that ``matrix_basename`` identifies.

- ``rhs_basename`` - (Possibly required) Common prefix for right-hand side
  filenames. Use it for multiple right-hand sides in a shared directory. This
  parameter has no default.

- ``dofmap_basename`` - (Possibly required) Common prefix for `dofmap` array
  filenames. This parameter has no default.

- ``timestep_filename`` - (Optional) File that maps timesteps to linear-system IDs
  for preconditioner reuse. You can omit this file when the sequence container
  includes timestep metadata.

- ``init_suffix`` - (Possibly required) Suffix number of the first linear system of a
  sequence of systems to be solved. Cannot be used together with ``set_suffix``.

- ``last_suffix`` - (Possibly required) Suffix number of the last linear system of a
  sequence of systems to be solved. Cannot be used together with ``set_suffix``.

- ``set_suffix`` - (Optional) A list of suffix numbers for the linear systems. One example
  is ``set_suffix: [0, 2, 5]``. Use this key for nonconsecutive suffixes. Do not combine
  it with ``init_suffix`` or ``last_suffix``. Its list length sets the number of systems.

- ``digits_suffix`` - (Optional) Number of digits used to build complete filenames when
  using the ``basename`` or ``dirname`` options. This parameter has a default value of 5.

Eigenspectrum diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``eigspec`` subsection enables a diagnostic full eigenspectrum calculation
for small systems. The driver invokes it after building a system when hypredrive is
built with ``-DHYPREDRV_ENABLE_EIGSPEC=ON``; library callers can invoke
``HYPREDRV_LinearSystemComputeEigenspectrum`` directly.

- ``enable`` - Enable the calculation. Default: ``off``.
- ``vectors`` - Also write eigenvectors. Default: ``off``.
- ``hermitian`` - Use the symmetric/Hermitian LAPACK path. Default: ``off``.
- ``preconditioned`` - Compute the spectrum of :math:`M^{-1}A` using the configured
  preconditioner instead of :math:`A`. Default: ``off``.
- ``output_prefix`` - Prefix for output files. Default: ``eig``.

The feature gathers the matrix and expands it to a dense array on one rank, so it is
not a distributed eigensolver. It uses :math:`O(n^2)` memory and :math:`O(n^3)` time,
requires a square positive-size matrix, and rejects a matrix dimension greater than
``20000``. The build also requires LAPACK. Rank 0 writes
``<output_prefix>.values.txt`` and, when ``vectors`` is enabled,
``<output_prefix>.vectors.bin`` in the current directory. General problems may produce
real/imaginary pairs in the values file; the Hermitian path writes real values. When
the feature is not compiled in, the API call is a successful no-op with a one-time
rank-0 warning.

Scheduled linear-system dumps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``print_system`` subsection can emit solver artifacts automatically at
``build``, ``setup``, and/or ``apply`` lifecycle boundaries.

Defaults:

- ``enabled: off``
- ``type: all``
- ``stage: build``
- ``artifacts: [matrix, rhs, dofmap]``
- ``output_dir: hypredrive-data``
- ``overwrite: off``

Available ``type`` values:

- ``all`` - dump every matched stage.
- ``every_n_systems`` - dump every ``N`` linear systems (requires ``every``).
- ``every_n_timesteps`` - dump every ``N`` timesteps (requires ``every``).
- ``ids`` - dump specific **0-based** linear-system ids (requires ``ids``).
- ``ranges`` - dump ids in one or more inclusive ranges (requires ``ranges``).
- ``iterations_over`` - dump after ``apply`` when the last linear solve used at least
  ``threshold`` iterations.
- ``setup_time_over`` - dump after ``setup`` or ``apply`` when the last setup time was
  at least ``threshold``.
- ``solve_time_over`` - dump after ``apply`` when the last solve time was at least
  ``threshold``.
- ``selectors`` - union of selector blocks using ``basis`` + ``every`` / ``ids`` /
  ``ranges`` for index-based matching, or ``basis`` + ``threshold`` for metric-based
  matching.

Time-based thresholds use the current stats time units:

- seconds by default
- milliseconds when ``general.use_millisec: on``

Supported artifact names are ``matrix``, ``precmat``, ``rhs``, ``x0``, ``xref``,
``solution``, ``dofmap``, and ``metadata``.

Dump layout:

- The driver writes dumps under ``output_dir/<object_name>/`` without a stage
  subdirectory.
- The driver assigns each system a continuous folder index in creation order:
  ``ls_00000``, ``ls_00001``, and subsequent values.
- ``systems_index.txt`` maps each ``ls_*`` folder to detailed context (stage,
  original linear-system id, timestep, variant, repetition, object name).

For multi-range selection, use block-sequence items as range pairs:

.. code-block:: yaml

    linear_system:
      print_system:
        enabled: on
        type: ranges
        stage: apply
        artifacts: [matrix, rhs, solution, metadata]
        output_dir: hypre-dumps
        overwrite: off
        ranges:
          - [20, 24]
          - [100, 150]

For mixed selectors:

.. code-block:: yaml

    linear_system:
      print_system:
        enabled: on
        type: selectors
        stage: all
        selectors:
          - basis: ids
            every: 10
          - basis: timestep
            ranges:
              - [2, 4]
              - [10, 12]
          - basis: level
            level: 1
            ids: [0, 3, 5]
          - basis: iterations
            threshold: 80
          - basis: solve_time
            threshold: 25.0

For threshold-based top-level triggers:

.. code-block:: yaml

    general:
      use_millisec: on

    linear_system:
      print_system:
        enabled: on
        type: setup_time_over
        stage: apply
        artifacts: [matrix, rhs, metadata]
        output_dir: hypre-dumps
        overwrite: off
        threshold: 50.0

Metric-based selectors support these ``basis`` values:

- ``iterations`` - compares the last linear-iteration count using ``threshold``.
- ``setup_time`` - compares the last setup time using ``threshold``.
- ``solve_time`` - compares the last solve time using ``threshold``.

Threshold-based triggers use an inclusive ``>= threshold`` comparison. Metric availability
depends on the current stage:

- ``iterations`` and ``solve_time`` are available on ``apply``.
- ``setup_time`` is available on ``setup`` and ``apply``.

CLI overrides can target nested keys, for example:

.. code-block:: bash

    hypredrive-cli input.yml -a --linear_system:print_system:enabled on


This example shows the ``linear_system`` section:

.. code-block:: yaml

    linear_system:
      type: ij
      x0_filename: IJ.out.x0
      rhs_filename: IJ.out.b
      matrix_filename: IJ.out.A
      precmat_filename: IJ.out.A
      dofmap_filename: dofmap
      rhs_mode: file
      init_guess_mode: file

Compressed sequence containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``hypredrive-lsseq`` utility can convert a directory-based sequence into a
single lossless container. The output filename extension identifies the low-level
backend (for example ``.zst.bin`` or ``.zlib.bin``).

Example packing command:

.. code-block:: bash

    # Minimal form: auto-detect suffix range + filenames from the first ls_XXXXX directory
    hypredrive-lsseq \
      --dirname data/poromech2k/np1 \
      --output poromech2k_np1_lsseq

    # Explicit form (overrides auto-detection)
    hypredrive-lsseq \
      --dirname data/poromech2k/np1/ls \
      --matrix-filename IJ.out.A \
      --rhs-filename IJ.out.b \
      --dofmap-filename dofmap.out \
      --init-suffix 0 \
      --last-suffix 4 \
      --algo zstd \
      --output poromech2k_np1_lsseq

Inspect metadata from an existing packed sequence:

.. code-block:: bash

    hypredrive-lsseq metadata \
      --input poromech2k_np1_lsseq.zst.bin

Unpack back to directory layout:

.. code-block:: bash

    hypredrive-lsseq unpack \
      --input poromech2k_np1_lsseq.zst.bin \
      --output-dir poromech2k_np1_unpacked

Example use in YAML:

.. code-block:: yaml

    linear_system:
      sequence_filename: poromech2k_np1_lsseq.zst.bin
      rhs_mode: file

An independently packed matrix can be used for preconditioner construction without
unpacking either archive:

.. code-block:: yaml

    linear_system:
      sequence_filename: operator.zst.bin
      precmat_sequence_filename: approximate-operator.zst.bin
      precmat_sequence_system_id: 0
      rhs_mode: file

When ``sequence_filename`` is present, hypredrive reads the matrix, right-hand
side, and optional dofmap from the container. If the container includes timestep
metadata, reuse can use it when you omit ``timestep_filename``.

Packed sequence files use batched compression for each part. Separate blobs contain
values, right-hand sides, and degree-of-freedom maps across all systems. A small,
uncompressed manifest contains provenance and debug information. This includes the suffix
range, input paths, codec, and build metadata. The manifest does not affect run-time
reconstruction. See :ref:`utilities` for the container structure.

Solver
------

The optional ``solver`` section specifies the Krylov solver configuration. If it is
omitted, hypredrive uses ``pcg``. These solver types are available:

- ``pcg`` - Preconditioned conjugate gradient.
- ``bicgstab`` - Bi-conjugate gradient stabilized.
- ``gmres`` - Generalized minimal residual.
- ``fgmres`` - Flexible generalized minimal residual.

Put the solver type on a new indentation level under ``solver``.

Scaling
~~~~~~~

The ``scaling`` subsection enables optional diagonal scaling before setup and
solve. Most scaling types use congruence scaling: for a diagonal matrix
:math:`D`, hypredrive solves
:math:`(D A D)y = D b` and recovers :math:`x = D y`. This form preserves a
symmetric matrix and is therefore the appropriate default for solvers and
preconditioners that require symmetry.

The custom dofmap variants also provide left, right, and similarity scaling.
They are mathematically different operations, not aliases for
``dofmap_custom``. Let :math:`D_{ii}` be the entry from ``custom_values``
selected by the dofmap tag of unknown :math:`i`. The transformed systems are:

.. list-table:: Custom dofmap transformations
   :header-rows: 1
   :widths: 24 26 22 38

   * - Type
     - Transformed system
     - Solution recovery
     - Intended use
   * - ``dofmap_custom``
     - :math:`(D A D)y = D b`
     - :math:`x = D y`
     - Congruence scaling that preserves symmetry. Use this for symmetric or
       positive-definite systems.
   * - ``dofmap_row_custom``
     - :math:`(D A)y = D b`
     - :math:`x = y`
     - Left scaling for normalizing or changing the sign of equations. Krylov
       residual norms are measured in the resulting weighted equation norm.
   * - ``dofmap_col_custom``
     - :math:`(A D)y = b`
     - :math:`x = D y`
     - Right scaling for changing unknown units while retaining the original
       physical residual :math:`b-Ax`.
   * - ``dofmap_similarity_custom``
     - :math:`(D^{-1} A D)y = D^{-1}b`
     - :math:`x = D y`
     - Similarity scaling for balancing asymmetric block couplings while
       preserving the eigenvalues of :math:`A`.

The exact vector and matrix transformations are applied before solver setup and
reversed after the solve. For ``rhs_l2``, with
:math:`s = 1/\sqrt{\|b\|_2}`, application uses
:math:`A \leftarrow s^2 A`, :math:`M \leftarrow s^2 M`,
:math:`b \leftarrow s b`, and :math:`v \leftarrow v/s` for unknown and reference
vectors. Recovery uses :math:`x \leftarrow s x`, :math:`b \leftarrow b/s`, and
:math:`A,M \leftarrow A/s^2,M/s^2`. A zero RHS uses :math:`s=1`.

For the congruence dofmap modes (``dofmap_mag`` and ``dofmap_custom``), application
uses :math:`A \leftarrow D A D`, :math:`M \leftarrow D M D`,
:math:`b \leftarrow D b`, and :math:`v \leftarrow v/D`; recovery uses
:math:`x \leftarrow D x`, :math:`b \leftarrow b/D`, and
:math:`A,M \leftarrow D^{-1} A D^{-1},D^{-1} M D^{-1}`. The row, column, and
similarity variants use the corresponding transformations in the table above, so
only the affected RHS or unknown vectors are scaled.

Congruence scaling multiplies both opposite coupling blocks :math:`A_{ij}` and
:math:`A_{ji}` by the same factor :math:`D_{ii}D_{jj}`. Consequently,
``dofmap_custom`` cannot change the magnitude ratio between those blocks.
Similarity scaling multiplies them by reciprocal factors and can balance that
ratio. Left and right scaling are useful when only equation units or only
unknown units should change. Except for congruence scaling, these operations
generally do not preserve matrix symmetry.

Available keywords:

- ``enabled`` - Turns scaling on or off. Values are ``yes`` and ``no``. The default
  is ``no``.

- ``type`` - Scaling strategy. Available values are:

  - ``rhs_l2`` - Scalar scaling based on the L2 norm of the RHS vector :math:`b`. Computes :math:`s = 1/\sqrt{\|b\|_2}` and applies uniform scaling :math:`M = s I`.

  - ``dofmap_mag`` - Uses hypre's tagged scaling API for vector scaling. This type
    requires a dofmap. It calls ``HYPRE_ParCSRMatrixComputeScalingTagged`` with
    scaling type 1. The call computes weights for each degree-of-freedom type
    from the matrix magnitude. See :ref:`linear_system_dofmap`.

  - ``dofmap_custom`` - Uses custom values for vector scaling. This type requires
    a dofmap. Provide one value for each unique degree-of-freedom type. Each
    value in ``custom_values`` defines the corresponding diagonal entry of
    :math:`D`. It applies congruence scaling as described above. See
    :ref:`linear_system_dofmap`.

  - ``dofmap_row_custom`` - Applies custom left (row/equation) scaling
    :math:`D A` and :math:`D b`.

  - ``dofmap_col_custom`` - Applies custom right (column/unknown) scaling
    :math:`A D` and recovers :math:`x = D y`.

  - ``dofmap_similarity_custom`` - Applies the custom similarity transform
    :math:`D^{-1} A D` and scales the RHS by :math:`D^{-1}`.

- ``custom_values`` - Required for all four custom dofmap types. Provide one
  nonzero value for each unique DOF tag as a YAML sequence. Entry :math:`i`
  defines :math:`D_{jj}` for every unknown whose dofmap tag is :math:`i`. For
  tags 0, 1, and 2, provide three entries in that order.

Scaling requires hypre 3.0.0 or newer. On older builds, YAML parsing succeeds,
but hypredrive logs a message and disables scaling at run time.

Example configuration with RHS L2 scaling:

.. code-block:: yaml

   solver:
     pcg:
       max_iter: 100
       relative_tol: 1.0e-6
     scaling:
       enabled: yes
       type: rhs_l2

Example configuration with dofmap_mag scaling:

.. code-block:: yaml

   solver:
     gmres:
       max_iter: 300
       relative_tol: 1.0e-6
     scaling:
       enabled: yes
       type: dofmap_mag
   linear_system:
     dofmap_filename: dofmap.dat

Example configuration with dofmap_custom scaling:

.. code-block:: yaml

   solver:
     gmres:
       max_iter: 300
       relative_tol: 1.0e-6
     scaling:
       enabled: yes
       type: dofmap_custom
       custom_values:
         - 1.0e5
         - 1.0e8
         - 1.0e4
   linear_system:
     dofmap_filename: dofmap.dat

For example, similarity scaling can balance an asymmetric coupling between
tags 0 and 2 without changing the matrix eigenvalues:

.. code-block:: yaml

   solver:
     gmres:
       max_iter: 300
       relative_tol: 1.0e-6
     scaling:
       enabled: yes
       type: dofmap_similarity_custom
       custom_values: [1.0, 1.0, 31.0]
   linear_system:
     dofmap_filename: dofmap.dat


.. _pcg:

PCG
~~~

The available keywords to further configure the preconditioned conjugate gradient solver
(``pcg``) are all optional and given below:

- ``max_iter`` - Maximum number of iterations. Available values are any positive integer.

- ``two_norm`` - Turn on/off L2 norm for the residual. Available values are ``yes`` or
  ``no``. Default value is ``yes``.

- ``stop_crit`` - Select the stopping criterion used by hypre. Available values are
  ``yes`` or ``no``. Default value is ``no``.

- ``rel_change`` - Turn on/off an additional convergence criteria that checks for a relative
  change in the solution vector. Available values are ``yes`` or ``no``. Default value is
  ``no``.

- ``print_level`` - Verbosity level for the iterative solver. `1` turns on convergence
  history reporting. The default is `1` in the block form shown below. The scalar
  shorthand ``solver: pcg`` preserves the quiet legacy behavior and sets it to `0`.

- ``recompute_res`` - Recompute the residual during PCG iterations. Available values are
  ``yes`` or ``no``. Default value is ``no``.

- ``relative_tol`` - Relative tolerance based on the norm of the residual vector and used
  for determining convergence of the iterative solver. Available values are any positive
  floating point number. Default value is ``1.0e-6``.

- ``absolute_tol`` - Absolute tolerance used for determining convergence of the iterative
  solver. Available values are any positive floating point number. Default value is
  ``0.0``, meaning that the absolute tolerance-based convergence criteria is inactive.

- ``residual_tol`` - Tolerance used for determining convergence of the iterative solver
  and based on the norm of the difference between subsequent residual vectors. Available
  values are any positive floating point number. Default value is ``0.0``, meaning that
  the residual tolerance-based convergence criterion is inactive.

- ``conv_fac_tol`` - Tolerance used for determining convergence of the iterative solver
  and based on the convergence factor ratio of subsequent iterations. Available values are
  any positive floating point number. Default value is ``0.0``, meaning that the
  convergence factor tolerance-based convergence criteria is inactive.

The code block representing the default parameter values for the ``solver:pcg`` section is
given below:

.. code-block:: yaml

    solver:
      pcg:
        max_iter: 100
        two_norm: yes
        stop_crit: no
        rel_change: no
        print_level: 1
        recompute_res: no
        relative_tol: 1.0e-6
        absolute_tol: 0.0
        residual_tol: 0.0
        conv_fac_tol: 0.0

BiCGSTAB
~~~~~~~~

The available keywords to further configure the bi-conjugate gradient stabilized solver
(``bicgstab``) are all optional and given below:

- ``min_iter`` - Minimum number of iterations. Available values are any non-negative
  integer. Default value is ``0``.

- ``max_iter``, ``print_level``, ``relative_tol``, ``absolute_tol``, and ``conv_fac_tol``
  - See :ref:`pcg` for a description of these variables.

- ``stop_crit`` - Select the stopping criterion used by hypre. Available values are
  ``yes`` or ``no``. Default value is ``no``.

- ``logging`` - Enable hypre logging. Available values are ``yes`` or ``no``. Default
  value is ``yes``.

The code block representing the default parameter values for the ``solver:bicgstab`` section is
given below:

.. code-block:: yaml

    solver:
      bicgstab:
        min_iter: 0
        max_iter: 100
        stop_crit: no
        logging: yes
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0
        conv_fac_tol: 0.0

.. _gmres:

GMRES
~~~~~

The available keywords to further configure the generalized minimal residual solver
(``gmres``) are all optional and given below:

- ``skip_real_res_check`` - Skip calculation of the real residual when evaluating
  convergence. Available values are `yes` and `no`. Default value is `no`.

- ``krylov_dim`` - Dimension of the krylov space. Available values are any positive
  integer. Default value is `30`.

- ``min_iter``, ``max_iter``, ``print_level``, ``rel_change``, ``relative_tol``,
  ``absolute_tol``, and ``conv_fac_tol`` - See :ref:`pcg` for a description of these
  variables.

- ``stop_crit`` - Select the stopping criterion used by hypre. Available values are
  ``yes`` or ``no``. Default value is ``no``.

- ``logging`` - Enable hypre logging. Available values are ``yes`` or ``no``. Default
  value is ``yes``.

The code block representing the default parameter values for the ``solver:gmres`` section is
given below:

.. code-block:: yaml

    solver:
      gmres:
        min_iter: 0
        max_iter: 300
        stop_crit: no
        skip_real_res_check: no
        krylov_dim: 30
        rel_change: no
        logging: yes
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0
        conv_fac_tol: 0.0

FGMRES
~~~~~~

The available keywords to further configure the flexible generalized minimal residual
solver (``fgmres``) are all optional and given below:

- ``min_iter``, ``max_iter``, ``krylov_dim``, ``print_level``, ``relative_tol``,
  ``absolute_tol``, and ``logging`` - See :ref:`gmres` for a description of these
  variables.

The code block representing the default parameter values for the ``solver:fgmres`` section is
given below:

.. code-block:: yaml

    solver:
      fgmres:
        min_iter: 0
        max_iter: 300
        krylov_dim: 30
        logging: yes
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0

Preconditioner
--------------

The ``preconditioner`` section is mandatory and it specifies the preconditioner
configuration. Available options for the preconditioner type are:

- ``amg`` - algebraic multigrid (BoomerAMG).
- ``jacobi`` - one Jacobi sweep.
- ``gauss-seidel`` - one forward hybrid Gauss-Seidel sweep.
- ``ilu``: incomplete LU factorization.
- ``fsai``: factorized sparse approximate inverse.
- ``mgr``: multigrid reduction.
- ``ams``: auxiliary-space Maxwell solver (for H(curl) / edge-element problems).
- ``ads``: auxiliary-space divergence solver (for H(div) / face-element problems).
- ``schwarz``: overlapping Schwarz (available with HYPRE 3.1.0 develop number 55 or newer).

Put a configurable preconditioner type at a new indentation level below
``preconditioner``. The fixed ``jacobi`` and ``gauss-seidel`` types use scalar forms.
These forms are ``preconditioner: jacobi`` and ``preconditioner: gauss-seidel``.
Internally, both forms limit BoomerAMG to one level and configure a V(1,0) cycle.
They also set the global relaxation type for the single-level execution path.

Preconditioner presets
~~~~~~~~~~~~~~~~~~~~~~

Presets are named default configurations. Each preset selects a preconditioner
and applies a small set of tuned settings. A preset supplies a default
without a full option list.

.. code-block:: yaml

    preconditioner:
      preset: elasticity-2D

Available presets:

- ``poisson``: BoomerAMG defaults (same as ``preconditioner: amg`` with defaults).
- ``elasticity-2D``: BoomerAMG defaults with
  ``coarsening.num_functions = 2`` and ``coarsening.strong_th = 0.8``.
- ``elasticity-3D``: BoomerAMG defaults with
  ``coarsening.num_functions = 3`` and ``coarsening.strong_th = 0.8``.

Preset names are case-insensitive.
Applications can also register additional preset names at runtime with
``HYPREDRV_PreconPresetRegister()`` before parsing or selecting inputs.

.. _PreconReuse:

Preconditioner reuse
~~~~~~~~~~~~~~~~~~~~

For a **sequence of linear systems**, you can reuse one preconditioner across multiple
systems. Examples include multiple right-hand sides and time steps. This reuse reduces
setup work. `hypredrive` rebuilds the preconditioner only at selected system indices. At
other indices, it applies the previous factorization.

A ``reuse`` subsection under ``preconditioner`` configures this behavior.

Static reuse
^^^^^^^^^^^^

The default policy is ``static``. It rebuilds on a fixed schedule:

- **``enabled``** – Turns reuse logic on or off. Values: ``yes`` / ``no``. Default: ``no``.
- **``policy``** – ``static`` or ``adaptive``. Default: ``static`` unless an
  ``adaptive`` subsection is present.
- **``frequency``** – Nonnegative integer. hypredrive rebuilds when
  ``(linear_system_index) mod (frequency + 1) == 0``. A value of ``0`` rebuilds
  every system. A value of ``1`` rebuilds every other system.
- **``linear_system_ids``** – Explicit list of **0-based** system indices for
  rebuilds, such as ``[0, 5, 10]``. Alias: ``linear_solver_ids``. Do not combine
  this key with ``frequency`` or ``per_timestep``.
- **``per_timestep``** – A value of ``yes`` applies ``frequency`` to each timestep.
  hypredrive rebuilds at the first system. It then rebuilds at every
  ``(frequency + 1)``-th system in that timestep.
  Driver workflows require a timestep file in
  ``linear_system.timestep_filename``. In library workflows, level-0 timestep
  annotations on the active ``HYPREDRV_t`` object supply the same information.
  Values: ``yes`` / ``no``. Do not combine this key with ``linear_system_ids``.

The timestep file maps linear systems to timesteps. Put the total number of
timesteps on the first line. Put ``timestep_id ls_start`` on each subsequent
line. ``ls_start`` is the zero-based index of the first system for that timestep.

Example: frequency-based reuse (rebuild every 3rd system):

.. code-block:: yaml

   preconditioner:
     amg: {}
     reuse:
       enabled: yes
       frequency: 2

Example: explicit list of systems at which to rebuild:

.. code-block:: yaml

   preconditioner:
     amg: {}
     reuse:
       enabled: yes
       linear_system_ids: [0, 10, 20, 30]

Example: Reuse for each time step. This form rebuilds at the first system and requires
``linear_system.timestep_filename``:

.. code-block:: yaml

   linear_system:
     timestep_filename: timesteps.txt
   preconditioner:
     amg: {}
     reuse:
       enabled: yes
       per_timestep: yes
       frequency: 0

In embedded library mode, pair the same YAML block with object-scoped annotations.
For example, use ``HYPREDRV_AnnotateLevelBegin(h, 0, "timestep-3", -1)`` and
``HYPREDRV_AnnotateLevelEnd(h, 0, "timestep-3", -1)`` instead of a timestep file.

Adaptive reuse
^^^^^^^^^^^^^^

Set ``type: adaptive`` to score recent solver behavior. HYPREDRV rebuilds when the score
crosses a threshold. Adaptive mode is separate from the static schedule. Do not combine
it with ``frequency``, ``linear_system_ids``, or ``per_timestep``.

For the simplest opt-in, ``reuse: adaptive`` is valid shorthand. That form enables adaptive
reuse with HYPREDRV's built-in default score model plus default guardrails:
``guards.min_history_points: 3``, ``guards.bad_decisions_to_rebuild: 2``, and
``guards.max_iteration_ratio: 3.0``.

.. code-block:: yaml

   preconditioner:
     mgr: {}
     reuse: adaptive

Top-level adaptive keys:

- **``guards.min_reuse_solves``** – Requires this number of reuse solves before
  another rebuild.
- **``guards.max_reuse_solves``** – Rebuilds after this number of preconditioner
  reuses. ``-1`` disables the guard.
- **``guards.min_history_points``** – Minimum number of samples that a score
  component requires before it triggers a rebuild.
- **``guards.bad_decisions_to_rebuild``** – Number of consecutive score-threshold breaches
  required before adaptive scoring triggers a rebuild.
- **``guards.max_iteration_ratio``** – Immediate hard guard. Rebuild when the most recent
  iteration count reaches this multiple of the rebuild baseline, bypassing the
  score/streak logic. ``-1`` disables the guard.
- **``guards.max_solve_time_ratio``** – Immediate hard guard on solve time relative to the
  rebuild baseline. ``-1`` disables the guard.
- **``guards.rebuild_on_new_level``** – Optional list of annotation levels that force a rebuild
  when the active level entry changes.
- **``adaptive.rebuild_threshold``** – Rebuild when the weighted score is at least this value.
- **``adaptive.positive_floor``** – Positive clamp used by geometric / harmonic / nonpositive
  power means and ratio-like transforms. Defaults to ``1.0e-12``.
- **``adaptive.components``** – Optional sequence of score components. If omitted,
  HYPREDRV installs a built-in default component set.

The HYPREDRV logging framework reports adaptive decisions when
``HYPREDRV_LOG_LEVEL >= 2``.

Each component contributes a nonnegative term to the total score. The current formula is:

.. math::

   \mathrm{score} = \sum_i w_i \max(0, d_i)

where ``w_i`` is ``weight`` and

.. math::

   d_i =
   \begin{cases}
   (\mathrm{aggregate}_i - \mathrm{target}_i) / \mathrm{scale}_i, & \text{direction = above} \\
   (\mathrm{target}_i - \mathrm{aggregate}_i) / \mathrm{scale}_i, & \text{direction = below}
   \end{cases}

``aggregate_i`` is the generalized power mean of the transformed history samples for that
component. HYPREDRV rebuilds when ``score >= adaptive.rebuild_threshold``.

Default adaptive components
"""""""""""""""""""""""""""

If you select ``type: adaptive`` and omit ``adaptive.components``, HYPREDRV uses
these built-in components:

- **``efficiency``** – ``metric: solve_overhead_vs_setup``, arithmetic mean, ``target: 1.0``,
  ``scale: 1.0``, ``transform.kind: raw``, ``transform.amortization_window: 10``,
  ``history.source: linear_solves``, ``history.max_points: 5``.
- **``stability``** – ``metric: iterations``, RMS mean, ``target: 1.5``,
  ``scale: 0.5``, ``transform.kind: ratio_to_baseline``, ``transform.baseline: rebuild``,
  ``history.source: linear_solves``, ``history.max_points: 5``.

The default model has one efficiency term and one iteration-stability term. Thus,
``reuse: adaptive`` does not require level annotations or a component list. The default
``guards.max_iteration_ratio: 3.0`` detects large one-step increases. The default
``guards.bad_decisions_to_rebuild: 2`` prevents a rebuild after one ordinary bad sample.
You can override the model with ``adaptive.components`` or explicit ``guards`` entries.

Fields available per component:

- **``name``** – Free-form label used in adaptive decision logs.
- **``metric``** – ``iterations``, ``solve_time``, ``setup_time``, ``total_time``, or
  ``solve_overhead_vs_setup``.
- **``direction``** – ``above`` or ``below``. ``above`` is the common case.
- **``target``** / **``scale``** – Normalize the aggregated metric before weighting.
- **``mean.kind``** – ``arithmetic``, ``geometric``, ``harmonic``, ``rms``, ``min``, ``max``,
  or ``power``.
- **``mean.power``** – Exponent used only when ``mean.kind: power``.
- **``transform.kind``** – ``raw``, ``delta_from_baseline``, ``ratio_to_baseline``, or
  ``relative_increase``.
- **``transform.baseline``** – ``rebuild`` or ``window_mean``.
- **``transform.amortization_window``** – Used by ``solve_overhead_vs_setup`` to compare solve
  overhead against the setup cost amortized over a chosen horizon.
- **``history.source``** – ``linear_solves``, ``active_level``, or ``completed_level``.
- **``history.level``** – Required for level-based history sources.
- **``history.max_points``** – Rolling history length for that component.
- **``history.reduction``** – ``none``, ``sum``, or ``mean``. For completed-level history,
  ``mean`` reports per-solve averages inside each level entry.

Named means are specialized forms of the generalized power mean:

- ``arithmetic`` = power ``1``
- ``geometric`` = power ``0``
- ``harmonic`` = power ``-1``
- ``rms`` = power ``2``

The following adaptive policy mirrors the original sketch: an amortization-style efficiency
term plus an RMS iteration spike detector.

.. code-block:: yaml

   preconditioner:
     mgr: {}
     reuse:
       enabled: yes
       type: adaptive
       guards:
         min_history_points: 2
         min_reuse_solves: 1
       adaptive:
         rebuild_threshold: 1.0
         components:
           - name: efficiency
             metric: solve_overhead_vs_setup
             target: 1.0
             scale: 1.0
             mean:
               kind: arithmetic
             transform:
               kind: raw
               amortization_window: 10
             history:
               source: linear_solves
               max_points: 5
           - name: stability
             metric: iterations
             target: 1.5
             scale: 0.5
             mean:
               kind: rms
             transform:
               kind: ratio_to_baseline
               baseline: rebuild
             history:
               source: active_level
               level: 0
               max_points: 3

Level-based history uses the same object-scoped annotation API described above. For example,
``history.source: active_level`` with ``level: 0`` tracks only solves inside the active
timestep, while ``history.source: completed_level`` evaluates the summaries produced by
completed level entries.

The ``mgr`` keyword does not fully define an MGR preconditioner. Use ``f_dofs`` to select
the degree-of-freedom types that are F points at each level. This selection identifies the
last level that contains each type. See :ref:`Example3` for a minimal MGR configuration.

.. _amg:

AMG
~~~

Use these optional keywords to configure the algebraic multigrid (BoomerAMG)
preconditioner:

- ``max_iter`` - Number of preconditioner applications for each call. Use a
  positive integer. The default is `1`.

- ``tolerance`` - convergence tolerance of AMG when applied multiple times. Available
  values are any positive floating point number. Default value is `0.0`.

- ``print_level`` - Verbosity level for the preconditioner. Default value is `0`.

  - ``0`` - no printout.
  - ``1`` - print setup statistics.
  - ``2`` - print solve statistics.

- ``interpolation`` - subsection detailing interpolation options:

  - ``prolongation_type`` - choose the prolongation operator. For detailed information,
    see `HYPRE_BoomerAMGSetInterpType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_BoomerAMGSetInterpType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``mod_classical``
    - ``least_squares``
    - ``mod_classical_he``
    - ``direct_sep_weights``
    - ``multipass``
    - ``multipass_sep_weights``
    - ``extended+i`` (default)
    - ``extended+i_c``
    - ``standard``
    - ``standard_sep_weights``
    - ``blk_classical``
    - ``blk_classical_diag``
    - ``f_f``
    - ``f_f1``
    - ``extended``
    - ``direct_sep_weights``
    - ``mm_extended``
    - ``mm_extended+i``
    - ``mm-ext+i`` (alias)
    - ``mm_extended+e``
    - ``mm-ext+e`` (alias)
    - ``blk_direct``
    - ``one_point``

  - ``restriction_type`` - choose the restriction operator. For detailed information, see
    `HYPRE_BoomerAMGSetRestriction
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv429HYPRE_BoomerAMGSetRestriction12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``p_transpose`` (default)
    - ``air_1``
    - ``air_2``
    - ``neumann_air_0``
    - ``neumann_air_1``
    - ``neumann_air_2``
    - ``air_1.5``

    The ``air_*`` and ``neumann_air_*`` operators implement *approximate ideal
    restriction*, which targets nonsymmetric, convection-dominated problems. They are
    normally combined with ``prolongation_type: one_point`` and with
    ``relaxation: points: air`` (see below), and require a nonsymmetric Krylov solver such
    as GMRES or BiCGSTAB.

  - ``restrict_strong_th`` - strength threshold used when building the approximate ideal
    restriction operator. Only relevant for the ``air_*`` and ``neumann_air_*`` restriction
    types. For detailed information, see `HYPRE_BoomerAMGSetStrongThresholdR
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv434HYPRE_BoomerAMGSetStrongThresholdR12HYPRE_Solver10HYPRE_Real>`_. Available values are
    any non-negative floating point number. Default value is `0.25`.

  - ``restrict_filter_th`` - post-filtering threshold applied to the approximate ideal
    restriction operator, used to drop small entries and keep it sparse. Only relevant for
    the ``air_*`` and ``neumann_air_*`` restriction types. For detailed information, see
    `HYPRE_BoomerAMGSetFilterThresholdR
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv434HYPRE_BoomerAMGSetFilterThresholdR12HYPRE_Solver10HYPRE_Real>`_. Available values are
    any non-negative floating point number. Default value is `0.0`.

  - ``trunc_factor`` - truncation factor for computing interpolation. Available values are
    any non-negative floating point number. Default value is `0.0`.

  - ``max_nnz_row`` - maximum number of elements per row for interpolation. Available values are
    any non-negative integer. Default value is `4`.

- ``coarsening`` - subsection detailing coarsening options:

  - ``type`` - choose the coarsening method. For detailed information, see
    `HYPRE_BoomerAMGSetCoarsenType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv429HYPRE_BoomerAMGSetCoarsenType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``cljp``
    - ``rs``
    - ``rs3``
    - ``falgout``
    - ``pmis``
    - ``hmis`` (default)

  - ``strong_th`` - strength threshold used for computing the strength of connection
    matrix. Available values are any non-negative floating point number. Default value is
    `0.25`.

  - ``seq_amg_th`` - Maximum size for an agglomeration or redundant coarse-grid
    solve. hypre then solves smaller systems with sequential AMG. Use a
    nonnegative integer. The default is `0`.

  - ``max_coarse_size`` - maximum size of the coarsest grid. Available values are any
    non-negative integer. Default value is `64`.

  - ``min_coarse_size`` - minimum size of the coarsest grid. Available values are any
    non-negative integer. Default value is `0`.

  - ``max_levels`` - maximum number of levels in the multigrid hierarchy. Available values
    are any non-negative integer. Default value is `25`.

  - ``num_functions`` - size of the system of PDEs, when using the systems
    version. Available values are any positive integer. Default value is `1`.

  - ``filter_functions`` - turn on/off filtering based on inter-variable couplings for
    systems of equations. For more information, see
    `HYPRE_BoomerAMGSetFilterFunctions
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv433HYPRE_BoomerAMGSetFilterFunctions12HYPRE_Solver9HYPRE_Int>`_.
    Default value is `off`.

  - ``nodal`` - turn on/off nodal (block) coarsening, in which the strength of connection
    matrix is built from small blocks of unknowns rather than single entries. Required for
    the rigid-body-mode (GM/LN) interpolation used by 3D elasticity, and combined with the
    near-null-space modes set through ``HYPREDRV_LinearSystemSetNearNullSpace``. Available
    values are any non-negative integer. Default value is `0` (off).

  - ``nodal_type`` - norm used to build the nodal strength matrix. For available options,
    see `HYPRE_BoomerAMGSetNodal
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv423HYPRE_BoomerAMGSetNodal12HYPRE_Solver9HYPRE_Int>`_.
    Only relevant when ``nodal`` is on. Default value is `4` (row-sum norm, which also
    enables ``HYPRE_BoomerAMGSetNodalDiag``); `2` (block sum-of-absolute-values) is a
    common alternative for elasticity.

  - ``interp_vec_variant`` - variant of the GM/LN interpolation that incorporates the
    supplied near-null-space vectors: `1` (GM1), `2` (GM2), or `3` and above (LN). For
    detailed information, see `HYPRE_BoomerAMGSetInterpVecVariant
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv434HYPRE_BoomerAMGSetInterpVecVariant12HYPRE_Solver9HYPRE_Int>`_.
    Only relevant when ``nodal`` is on. Default value is `-1` (auto), which selects `2`
    for a full-system near-null space and `1` when the modes are projected onto an MGR
    F-block.

  - ``smooth_interp_vecs`` - turn on/off smoothing of the interpolation (near-null-space)
    vectors before they are used. For detailed information, see
    `HYPRE_BoomerAMGSetSmoothInterpVectors
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv437HYPRE_BoomerAMGSetSmoothInterpVectors12HYPRE_Solver9HYPRE_Int>`_.
    Only relevant when ``nodal`` is on. Default value is `-1` (auto), which smooths them
    only for the GM2 variant.

  - ``interp_vec_qmax`` - maximum number of nonzeros per row added to interpolation by the
    GM/LN expansion. For detailed information, see `HYPRE_BoomerAMGSetInterpVecQMax
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv431HYPRE_BoomerAMGSetInterpVecQMax12HYPRE_Solver9HYPRE_Int>`_.
    Only relevant when ``nodal`` is on. Default value is `-1` (auto), which uses `4` for
    the GM2 variant and `0` (no limit) otherwise.

  - ``interp_vec_abs_q_trunc`` - drop entries of the GM/LN expansion below this magnitude.
    For detailed information, see `HYPRE_BoomerAMGSetInterpVecAbsQTrunc
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv435HYPRE_BoomerAMGSetInterpVecAbsQTrunc12HYPRE_Solver10HYPRE_Real>`_.
    Only relevant when ``nodal`` is on. Available values are any non-negative floating
    point number. Default value is `0.0`.

  - ``rap2`` - whether or not to use two matrix products to compute coarse
    level matrices. Available values are any non-negative integer. Default value is `0`.

  - ``mod_rap2`` - whether or not to use two matrix products with modularized kernels for
    computing coarse level matrices. Available values are any non-negative
    integer. Default value is `0` for CPU runs or `1` for GPU runs.

  - ``keep_transpose`` - whether or not to save local interpolation transposes for more
    efficient matvecs during the solve phase. Available values are any non-negative
    integer. Default value is `0` for CPU runs or `1` for GPU runs.

  - ``max_row_sum`` - parameter that modifies the definition of strength for diagonal
    dominant portions of the matrix. Available values are any non-negative floating point
    number. Default value is `0.9`.

- ``aggressive`` - subsection detailing aggressive coarsening options:

  - ``prolongation_type`` - choose the prolongation type used in levels with aggressive
    coarsening turned on. For detailed information, see
    `HYPRE_ParCSRHybridSetAggInterpType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv434HYPRE_ParCSRHybridSetAggInterpType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``2_stage_extended+i``
    - ``2_stage_standard``
    - ``2_stage_extended``
    - ``multipass`` (default)
    - ``mm_extended``
    - ``mm_extended+i``
    - ``mm_extended+e``

  - ``num_levels`` - number of levels with aggressive coarsening turned on. Available
    values are any positive integer. Default value is `0`.

  - ``num_paths`` - degree of aggressive coarsening. Available values are any positive
    integer. Default value is `1`.

  - ``trunc_factor`` - truncation factor for computing interpolation in aggressive
    coarsening levels. Available values are any non-negative floating point
    number. Default value is `0.0`.

  - ``max_nnz_row`` - maximum number of elements per row for computing interpolation in
    aggressive coarsening levels. Available values are any non-negative integer. Default
    value is `0` (no limit).

  - ``P12_trunc_factor`` - Truncation factor for the P1 and P2 matrices. These
    matrices build two-stage interpolation. Use a nonnegative floating-point
    number. The default is `0.0`.

  - ``P12_max_elements`` - Maximum number of elements in each P1 and P2 matrix row.
    These matrices build two-stage interpolation. Use a nonnegative integer. The
    default of `0` removes the limit.

- ``relaxation`` - subsection detailing relaxation options:

  - ``down_type`` - Relaxation method for the pre-smoothing stage. For detailed
    information, see `HYPRE_BoomerAMGSetRelaxType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_BoomerAMGSetRelaxType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``jacobi_non_mv``: legacy Jacobi implementation.
    - ``forward-hgs``: forward hybrid Gauss-Seidel.
    - ``backward-hgs``: backward hybrid Gauss-Seidel.
    - ``chaotic-hgs``: chaotic hybrid Gauss-Seidel.
    - ``hsgs``: hybrid symmetric Gauss-Seidel.
    - ``jacobi``: Jacobi (based on SpMVs).
    - ``l1-hsgs``: L1-scaled hybrid symmetric Gauss-Seidel.
    - ``forward-solve``: on-processor direct forward solve, for matrices with triangular structure.
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
    - ``forward-hl1gs``: forward hybrid L1-scaled Gauss-Seidel (default).
    - ``backward-hl1gs``: backward hybrid L1-scaled Gauss-Seidel.
    - ``cg``: conjugate gradient.
    - ``chebyshev``: Chebyshev polynomial.
    - ``l1-jacobi``: L1-scaled Jacobi.
    - ``l1sym-hgs``: L1-scaled symmetric hybrid Gauss-Seidel (with convergent L1 factor).

  - ``up_type`` - Relaxation method for the post-smoothing stage. For detailed
    information, see `HYPRE_BoomerAMGSetRelaxType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_BoomerAMGSetRelaxType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``jacobi_non_mv``: legacy Jacobi implementation.
    - ``forward-hgs``: forward hybrid Gauss-Seidel.
    - ``backward-hgs``: backward hybrid Gauss-Seidel.
    - ``chaotic-hgs``: chaotic hybrid Gauss-Seidel.
    - ``hsgs``: hybrid symmetric Gauss-Seidel.
    - ``jacobi``: Jacobi (based on SpMVs).
    - ``l1-hsgs``: L1-scaled hybrid symmetric Gauss-Seidel.
    - ``forward-solve``: on-processor direct forward solve, for matrices with triangular structure.
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
    - ``forward-hl1gs``: forward hybrid L1-scaled Gauss-Seidel.
    - ``backward-hl1gs``: backward hybrid L1-scaled Gauss-Seidel (default).
    - ``cg``: conjugate gradient.
    - ``chebyshev``: Chebyshev polynomial.
    - ``l1-jacobi``: L1-scaled Jacobi.
    - ``l1sym-hgs``: L1-scaled symmetric hybrid Gauss-Seidel (with convergent L1 factor).

  - ``coarse_type`` - Relaxation method for the coarsest levels. For detailed
    information, see `HYPRE_BoomerAMGSetRelaxType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_BoomerAMGSetRelaxType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``jacobi_non_mv``: legacy Jacobi implementation.
    - ``hsgs``: hybrid symmetric Gauss-Seidel.
    - ``jacobi``: Jacobi (based on SpMVs).
    - ``l1-hsgs``: L1-scaled hybrid symmetric Gauss-Seidel.
    - ``ge``: hypre's gaussian elimination.
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
    - ``cg``: conjugate gradient.
    - ``chebyshev``: Chebyshev polynomial.
    - ``l1-jacobi``: L1-scaled Jacobi.
    - ``l1sym-hgs``: L1-scaled symmetric hybrid Gauss-Seidel (with convergent L1 factor).
    - ``lu_piv``: LU factorization with pivoting.
    - ``lu_inv``: explicit LU inverse.

  - ``down_sweeps`` - number of pre-smoothing sweeps. Available values are any integer
    greater or equal than `-1`, which turns off the selection of sweeps at the specific
    cycle. Default value is `-1`.

  - ``up_sweeps`` - number of post-smoothing sweeps. Available values are any integer
    greater or equal than `-1`, which turns off the selection of sweeps at the specific
    cycle. Default value is `-1`.

  - ``coarse_sweeps`` - number of smoothing sweeps in the coarsest level. Available values
    are any integer greater or equal than `-1`, which turns off the selection of sweeps at
    the specific cycle. Default value is `1`.

  - ``num_sweeps`` - number of pre and post-smoothing sweeps. Available values are any
    non-negative integer. Default value is `1`.

  - ``order`` - Order in which hypre relaxes the points. For available
    options, see `HYPRE_BoomerAMGSetRelaxOrder
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_BoomerAMGSetRelaxOrder12HYPRE_Solver9HYPRE_Int>`_. Default value is `0`.

  - ``points`` - which grid points each sweep of the cycle relaxes. For detailed
    information, see `HYPRE_BoomerAMGSetGridRelaxPoints
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv433HYPRE_BoomerAMGSetGridRelaxPoints12HYPRE_SolverPP9HYPRE_Int>`_. Available options are:

    - ``all`` (default) - leave hypre's default schedule untouched, i.e. every sweep
      relaxes all points.
    - ``air`` - use the schedule expected by approximate ideal restriction: the down cycle
      and the coarsest level relax all points, while the up cycle relaxes F-points and
      finishes with a C-point sweep when ``up_sweeps`` is greater than two.

    .. warning::

       ``points: air`` is meant to be used with ``order: 0``. Combining an ``air_*``
       restriction with ``order: 1`` and a hybrid Gauss-Seidel down-cycle relaxation type
       (``forward-hl1gs`` or ``backward-hl1gs``) is known to produce a hierarchy
       containing non-numeric values in hypre 2.33.0 and later.

  - ``weight`` - relaxation weight for smoothed Jacobi and hybrid SOR. For available
    options, see `HYPRE_BoomerAMGSetRelaxWt
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv425HYPRE_BoomerAMGSetRelaxWt12HYPRE_Solver10HYPRE_Real>`_. Default value is `1.0`.

  - ``outer_weight`` - outer relaxation weight for hybrid SOR and SSOR. For available
    options, see `HYPRE_BoomerAMGSetOuterWt
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv425HYPRE_BoomerAMGSetOuterWt12HYPRE_Solver10HYPRE_Real>`_. Default value is `1.0`.

- ``smoother`` - subsection detailing complex smoother options:

  - ``type`` - complex smoother type. For detailed information, see `HYPRE_BoomerAMGSetSmoothType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_BoomerAMGSetSmoothType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``fsai``: factorized sparse approximate inverse.
    - ``ilu``: incomplete LU factorization.
    - ``schwarz``: Additive/Multiplicative overlapping Schwarz.
    - ``pilut``: Incomplete LU factorization from PILUT.
    - ``parasails``: Sparse approximate inverse from Parasails.
    - ``euclid``: Incomplete LU factorization from Euclid.

  - ``num_levels`` - Number of levels that use complex smoothers, starting at the
    finest level. Use a nonnegative integer. The default is `0`.

  - ``num_sweeps`` - number of pre and post-smoothing sweeps used for the complex
    smoother. Available values are any non-negative integer. Default value is `1`.

This example shows the default values for ``preconditioner:amg``:

.. code-block:: yaml

    preconditioner:
      amg:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        interpolation:
          prolongation_type: extended+i
          restriction_type: p_transpose
          restrict_strong_th: 0.25
          restrict_filter_th: 0.0
          trunc_factor: 0.0
          max_nnz_row: 4
        coarsening:
          type: hmis # pmis for GPU runs
          strong_th: 0.25
          seq_amg_th: 0
          max_coarse_size: 64
          min_coarse_size: 0
          max_levels: 25
          num_functions: 1
          filter_functions: off
          rap2: off
          mod_rap2: off # on for GPU runs
          keep_transpose: off # on for GPU runs
          max_row_sum: 0.9
        aggressive:
          num_levels: 0
          num_paths: 1
          prolongation_type: multipass
          trunc_factor: 0
          max_nnz_row: 0
          P12_trunc_factor: 0.0
          P12_max_elements: 0
        relaxation:
          down_type: forward-hl1gs
          up_type: backward-hl1gs
          coarse_type: ge
          down_sweeps: -1
          up_sweeps: -1
          coarse_sweeps: 1
          num_sweeps: 1
          order: 0
          points: all
          weight: 1.0
          outer_weight: 1.0
        smoother:
          type: ilu
          num_levels: 0
          num_sweeps: 1

.. _ilu:

ILU
~~~

Use these optional keywords to configure the incomplete LU factorization (ILU)
preconditioner:

- ``max_iter``, ``tolerance``, and ``print_level`` - See :ref:`amg` for a description of
  these variables.

- ``type`` - ILU type. For available
  options, see `HYPRE_ILUSetType
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv416HYPRE_ILUSetType12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `0` (Block-Jacobi ILU0).

- ``fill_level`` - level of fill when using ILUK. Available values are any non-negative
  integer. Default value is `0`.

- ``reordering`` - reordering method. For available
  options, see `HYPRE_ILUSetLocalReordering
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_ILUSetLocalReordering12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `0` (no reordering).

- ``tri_solve`` - whether or not to turn on direct triangular solves in the
  preconditioner's application phase. Default value is `1`.

- ``lower_jac_iters`` - Number of iterations for solving the lower triangular system
  during the preconditioner's application phase. Available values are any positive
  integer. Default value is `5`. This option has effect only when ``tri_solve`` is set to
  zero.

- ``upper_jac_iters`` - Number of iterations for solving the upper triangular system
  during the preconditioner's application phase. Available values are any positive
  integer. Default value is `5`. This option has effect only when ``tri_solve`` is set to
  zero.

- ``max_row_nnz`` - Maximum number if nonzeros per row when using ILUT. Available values
  are any positive integer. Default value is `200`.

- ``schur_max_iter`` - Maximum number of the Schur system solve. Available values
  are any positive integer. Default value is `3`. This option has effect only when
  ``type`` is greater or equal than `10`.

- ``droptol`` - Dropping tolerance for computing the triangular factors when using
  ILUT. Available values are any non-negative floating point numbers. Default value is
  `1.0e-2`.

- ``nsh_droptol`` - Dropping tolerance for computing the triangular factors when using
  NSH. Available values are any non-negative floating point numbers. Default value is
  `1.0e-2`.

This example shows the default values for ``preconditioner:ilu``:

.. code-block:: yaml

    preconditioner:
      ilu:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        type: 0
        fill_level: 0
        reordering: 0
        tri_solve: 1
        lower_jac_iters: 5
        upper_jac_iters: 5
        max_row_nnz: 200
        schur_max_iter: 3
        droptol: 1.0e-2
        nsh_droptol: 1.0e-2

.. _fsai:

FSAI
~~~~

Use these optional keywords to configure the factorized sparse approximate
inverse (FSAI) preconditioner:

- ``max_iter``, ``tolerance``, and ``print_level`` - See :ref:`amg` for a description of
  these variables.

- ``algo_type`` - algorithm type used for building FSAI. Available values are ``bj-afsai``
  (`1`, the default), ``bj-afsai-omp`` (`2`), and ``bj-sfsai`` (`3`). For additional
  options, see `HYPRE_FSAISetAlgoType
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv421HYPRE_FSAISetAlgoType12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `1`.

- ``ls_type`` - solver type for the local linear systems in FSAI. For available
  options, see `HYPRE_FSAISetLocalSolveType
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_FSAISetLocalSolveType12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `1`.

- ``max_steps`` - maximum number of steps for computing the sparsity pattern
  of G. Available values are any positive integer. Default value is `5`.

- ``max_step_size`` - step size for computing the sparsity pattern of G. Available values
  are any positive integer. Default value is `3`.

- ``max_nnz_row`` - maximum number of nonzeros per row for computing the sparsity pattern
  of G. Available values are any positive integer. Default value is `15`.

- ``num_levels`` - number of levels for computing the candidate pattern matrix. Available
  values are any positive integer. Default value is `1`.

- ``eig_max_iters`` - number of iterations for estimating the largest eigenvalue of G. Available
  values are any positive integer. Default value is `5`.

- ``threshold`` - Dropping tolerance for building the candidate pattern matrix. Available
  values are any non-negative floating point numbers. Default value is `1.0e-3`.

- ``kap_tolerance`` - Kaporin reduction factor. Available values are any non-negative
  floating point numbers. Default value is `1.0e-3`.

This example shows the default values for ``preconditioner:fsai``:

.. code-block:: yaml

    preconditioner:
      fsai:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        algo_type: 1
        ls_type: 1
        max_steps: 5
        max_step_size: 3
        max_nnz_row: 15
        num_levels: 1
        eig_max_iters: 5
        threshold: 1.0e-3
        kap_tolerance: 1.0e-3

.. _mgr:

MGR
~~~

Use these optional keywords to configure the multigrid reduction (MGR)
preconditioner:

- ``max_iter`` and ``tolerance`` - See :ref:`amg` for a description of these variables.

- ``print_level`` - verbosity level for the preconditioner. For available
  options, see `HYPRE_MGRSetPrintLevel
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv422HYPRE_MGRSetPrintLevel12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `0` (no printout).

- ``non_c_to_f`` - Whether non-C points are treated as F points. This is an integer
  switch passed to hypre. Default value is `1`.

- ``pmax`` - Maximum number of interpolation elements retained per row. A value of
  `0` leaves hypre's default/unlimited behavior. Default value is `0`.

- ``num_levels`` - Number of MGR levels. When levels are supplied in the YAML ``level``
  and ``coarsest_level`` sections, hypredrive derives this value from those entries.
  The standalone default is `0`.

- ``relax_type`` - Base MGR relaxation type. Named values include ``jacobi`` (`7`),
  ``h-fgs`` (`3`), ``h-bgs`` (`4`), ``ch-gs`` (`5`), ``h-ssor`` (`6`),
  ``hl1-ssor`` (`8`), ``l1-fgs`` (`13`), ``l1-bgs`` (`14`), ``chebyshev`` (`16`),
  and ``l1-jacobi`` (`18`). The default is ``jacobi`` (`7`).

- ``nonglk_max_elmts`` - Maximum number of entries retained per row in the non-Galerkin
  coarse operator. Default value is `1`; HYPRE versions without this setter ignore it.

- ``coarse_th`` - threshold for dropping small entries on the coarse grid. Available
  values are any non-negative floating point numbers. Default value is `0.0`, which means
  no dropping.

- ``interp_sweeps`` and ``interp_weight`` - opt-in damped Jacobi refinement of
  type-2 (``jacobi``) prolongation. Defaults are `0` and `1.0`, respectively.
  The damping weight must lie in ``[HYPRE_REAL_MIN, 1]``.
  Positive sweeps require a HYPRE build advertising bounded P2 refinement, CPU
  execution, ``pmax > 0``, injection restriction, and Galerkin coarse operators.
  Each sweep retains at most the ``pmax`` largest-magnitude entries per row
  without row-sum rescaling, and the refined prolongation is used by the full
  RAP product so its Schur correction is consistent.

- ``injection_upcycle`` - default-off, block-LDU-shaped solve mode. With ``on``, MGR
  retains type-2 (Jacobi) interpolation when constructing the Galerkin coarse
  operator, but uses injection for the upward correction; post-F relaxation
  then performs the fine-variable backsolve. This bounded mode requires
  ``cycle: v(1,1)``, ``interp_sweeps: 0``, and P2/R0/Galerkin with active F
  relaxation at every reduction level. It becomes an exact block-LDU action
  only when the configured F solver applies the exact inverse of its F block.

- ``matched_q_sweeps`` and ``matched_q_weight`` - construction controls for the
  per-level ``matched_q: polynomial`` mode (``on`` is an alias). Defaults are
  ``0`` and ``1.0``. A polynomial-selected level requires at least one sweep
  and a weight in ``[HYPRE_REAL_MIN, 1]``. ``matched_q: afsai`` ignores both
  keys. They are separate from ``interp_sweeps`` so an earlier TPFA level can
  retain ordinary Jacobi prolongation while a later, much larger MFD block uses
  matched sparse-Q elimination.

- ``cycle`` - MGR traversal cycle. Numeric values preserve the legacy encoding.
  `1` selects a V-cycle and `2` selects a W-cycle. Symbolic values ``v``, ``w``,
  ``v(1,0)``, ``v(0,1)``, ``v(1,1)``, ``w(1,0)``, ``w(0,1)``, and ``w(1,1)``
  are also valid. The tuple selects pre-smoothing, post-smoothing, or both with
  HYPRE ``>= 3.1.0`` develop number ``>= 50``. On older HYPRE versions, the key is
  accepted for compatibility but ignored.

- ``level`` - Defines parameters for each MGR level. A numeric identifier selects each
  level. The finest level has identifier `0`. Put the levels in increasing order at the
  next YAML indentation level.

  - ``f_dofs`` - (Mandatory) Array containing the identifiers of F (fine) degrees of
    freedom to be treated in the current level. Available values are integer numbers
    from `0` to `n_dofs - 1`, where `n_dofs` represents the unique number of degree-of-
    freedom identifiers, or symbolic labels defined by ``linear_system.dof_labels``.

  - ``matched_q`` - opt into a CPU-only matched approximate block
    factorization on this level. Available values are ``off`` (the default),
    ``polynomial`` (with ``on`` as an alias), and ``afsai``. For
    ``A = [[A_FF, A_FC], [A_CF, A_CC]]``, both nonzero modes form
    ``S = diag(A_FF)^(-1/2)`` and materialize one sparse inverse approximation
    ``Q``. The identical ``Q`` is used for the pre-F correction, untruncated
    ``W = -Q A_FC``, and the Galerkin Schur approximation
    ``A_CC - A_CF Q A_FC``. This matching avoids using different F inverses in
    relaxation and in the Schur complement.

    ``polynomial`` applies ``matched_q_sweeps`` damped Richardson updates to
    approximate the inverse of ``S A_FF S``. After each update it retains the
    diagonal plus at most ``pmax - 1`` largest remaining entries per row,
    without rescaling, then forms ``Q = S Qs S``. Thus ``pmax`` caps Q in this
    mode.

    ``afsai`` is available only in a real-valued host build advertising matched
    aFSAI Q. On the F closure produced by the preceding active reductions, it
    treats the stored local lower triangle as an SPD operator and builds a
    private, symmetrically scaled adaptive FSAI factor using three one-entry
    pattern-growth steps, then materializes
    ``Q = S G^T G S`` and destroys the private factor. Each G row has at most
    four entries. Accordingly ``pmax`` must be exactly ``4`` as a construction
    contract for G; it is not a cap on Q. Neither the resulting Q nor W is
    truncated, so their row fill and memory are not strictly bounded by four.
    The adaptive pattern is rank-block local and can therefore depend on the MPI
    row partition. A failed local Cholesky/FSAI construction is rejected; this
    mode is not a general approximate inverse for nonsymmetric F blocks. Small
    upper/lower-triangle skew therefore needs an application-specific gate.

    Both modes require ``prolongation_type: jacobi``,
    ``restriction_type: injection``, ``coarse_level_type: rap``, one built-in
    Jacobi F sweep, no global smoother, ``cycle: v(1,0)``, ``pmax > 0``,
    ``coarse_th: 0``, ``interp_sweeps: 0``, and
    ``injection_upcycle: off``. Invalid/nonpositive A_FF diagonals, incompatible
    extracted block dimensions/partitions, nonfinite A_FF/Q/W construction
    values, device execution, P/W truncation, and reserved-C insertion are
    rejected collectively. This check does not assert that every A_CF, A_CC, or
    final RAP entry is finite.

  - ``matched_f_backsolve`` - select a CPU-only final-level matched block
    action. The default, ``off``, preserves ordinary MGR. ``on`` retains
    ordinary Jacobi P2 for setup, Galerkin RAP, and upward transfer; after
    prolongation it discards only that level's F entries and applies the
    existing zero-guess F solver once to ``f_F - A_FC u_C``. The result is
    ``Q(f_F - A_FC u_C)`` while the coarse operator remains the diagonal-P2
    approximation ``A_CC - A_CF diag(A_FF)^(-1) A_FC``.

    ``gmres1`` retains the original ``C=A_FC``, ``B=A_CF``, and ``K=A_CC``
    blocks while keeping the same P2/R0 Galerkin hierarchy. On the down leg it
    caches ``q0=Q f_F`` and ``g=f_C-B q0``. Given the one-cycle coarse-AMG
    direction ``z``, its up leg bypasses P and computes
    ``h=Q(C z)``, ``w=K z-B h``,
    ``alpha=(w^H g)/(w^H w)``, ``q=q0-alpha h``, and ``p=alpha z``.
    Thus each application uses the same managed, zero-guess Q child twice and
    the managed coarse AMG once. Because ``alpha`` depends on the current
    residual, the outer solver must be ``fgmres``. This mode is an overwrite
    preconditioner action and clears its top-level output before forming
    ``q0`` and ``g``. A nonzero ``g`` with a zero
    direction, a nonfinite child result/coefficient, or a nonfinite final
    correction is reported collectively without partially committing q or p.

    Exactly one final active level may select either nonzero mode. Both require
    ``max_iter: 1``, ``tolerance: 0``, ``cycle: v(1,0)``, ``coarse_th: 0``,
    P2/R0/Galerkin, and one managed AMG F sweep with
    ``symmetric_diagonal_scaling: on``, AMG ``max_iter: 1``, and AMG
    ``tolerance: 0``. Global relaxation, nested Krylov, component reuse,
    projected rigid-body modes, P2 refinement, injection upcycle, and
    ``matched_q`` are incompatible. ``gmres1`` additionally requires a managed
    coarsest-level AMG with ``max_iter: 1`` and ``tolerance: 0`` (no nested
    Krylov or component reuse). If the selected configured level has no F labels
    in a system and is compacted away, the option is a no-op for that system.
    The outer-solver requirement is scoped to the configuration rather than to
    an individual system, so ``gmres1`` requires ``fgmres`` even when the
    selected level is compacted away; the configuration is therefore accepted or
    rejected identically for every system in a sequence.

  - ``f_relaxation`` - relaxation method targeting F points. For available options, see
    `HYPRE_MGRSetLevelFRelaxType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_MGRSetLevelFRelaxType12HYPRE_SolverP9HYPRE_Int>`_. Default
    value is `7` (Jacobi). Use ``none`` to deactivate F-relaxation.

    A nested managed ``amg`` F-relaxation may opt into
    ``symmetric_diagonal_scaling: on``. On CPUs this clones the extracted F block,
    forms the diagonal congruence ``D^{-1/2} A_FF D^{-1/2}``, and maps the AMG
    right-hand side and correction so the outer matrix and residual norm remain
    unchanged. The option defaults to ``off`` and currently requires finite positive
    F-block diagonals and HYPRE 3.1 or newer; it is incompatible with device
    execution, nested Krylov, component reuse, and projected rigid-body modes.

  - ``g_relaxation`` - global relaxation method targeting F and C points. For available
    options, see `HYPRE_MGRSetGlobalSmoothType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_MGRSetGlobalSmoothType12HYPRE_Solver9HYPRE_Int>`_. Default
    value is `-1` (none). Use a named smoother such as ``blk-jacobi`` to enable global
    relaxation.

  - ``f_relaxation`` and ``g_relaxation`` also accept a nested Krylov solver block with an
    optional nested preconditioner. Supported Krylov solvers are ``pcg``, ``gmres``,
    ``fgmres``, and ``bicgstab``. Supported nested preconditioners are ``amg``, ``ilu``,
    and ``fsai``. Nested ``preconditioner: mgr`` is not supported.

    Example:

    .. code-block:: yaml

        f_relaxation:
          gmres:
            max_iter: 2
            preconditioner:
              amg:
                max_iter: 1
                coarsening:
                  num_levels: 1

  - ``f_relaxation`` also supports a nested ``mgr`` block. Nested MGR requires `hypre`
    3.1.0 or newer with development number 5 or newer. Nested levels use the parent
    degree-of-freedom labels. Only labels in the parent F-block remain available.

    Example:

    .. code-block:: yaml

        level:
          0:
            f_dofs: [0, 2]
            f_relaxation:
              mgr:
                level:
                  0:
                    f_dofs: [2] # refers to the same parent label 2 (no relabeling)
                coarsest_level:
                  amg: {}

  - ``restriction_type`` - algorithm for computing the restriction operator. Accepted values
    are ``injection``, ``jacobi``, ``approx-inv``, ``blk-jacobi``, ``cpr-like``,
    ``columped``, and ``columped-partial``. With HYPRE 2.32 or newer, ``air_1`` and
    ``air_1.5`` select distance-1 and distance-1.5 approximate ideal restriction,
    respectively. See `HYPRE_MGRSetRestrictType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv424HYPRE_MGRSetRestrictType12HYPRE_Solver9HYPRE_Int>`_. Default
    value is `0` (Injection).

  - ``prolongation_type`` - algorithm for computing the prolongation operator. Accepted
    values are ``injection``, ``l1-jacobi``, ``jacobi``, ``classical-mod``,
    ``approx-inv``, ``blk-jacobi``, ``blk-rowlump``, and ``blk-absrowsum``. With
    HYPRE 2.24 or newer, ``mm-ext``, ``mm-ext+i``, and ``mm-ext+e`` are also
    accepted. For implementation details, see
    `HYPRE_MGRSetInterpType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv422HYPRE_MGRSetInterpType12HYPRE_Solver9HYPRE_Int>`_. Default
    value is `0` (Injection).

  - ``coarse_level_type`` - algorithm for computing the coarse level matrices. For available
    options, see `HYPRE_MGRSetCoarseGridMethod
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_MGRSetCoarseGridMethod12HYPRE_SolverP9HYPRE_Int>`_. Default
    value is `0` (Galerkin).

- ``coarsest_level`` - special keyword for defining specific parameters for MGR's coarsest
  level.

  ``coarsest_level`` also supports the same nested Krylov solver block described above.

This example shows the default values for ``preconditioner:mgr``:

.. code-block:: yaml

    preconditioner:
      mgr:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        non_c_to_f: 1
        pmax: 0
        num_levels: 0
        relax_type: jacobi
        nonglk_max_elmts: 1
        coarse_th: 0.0
        interp_sweeps: 0
        interp_weight: 1.0
        matched_q_sweeps: 0
        matched_q_weight: 1.0
        level:
          0:
            f_dofs: [1, 2] # Example usage where DOFs 1 and 2 are treated in MGR's 1st level
            f_relaxation: single
            g_relaxation: none
            restriction_type: injection
            prolongation_type: jacobi
            coarse_level_type: rap
            matched_q: off
            matched_f_backsolve: off

          1:
            f_dofs: [0] # Example usage where DOF 0 is treated in MGR's 2nd level
            f_relaxation: none
            g_relaxation:
              ilu: # ILU parameters can be specified with a new indentation level
            restriction_type: injection
            prolongation_type: jacobi
            coarse_level_type: rap
            matched_q: off
            matched_f_backsolve: off

        coarsest_level:
          amg: # AMG parameters can be specified with a new indentation level

Nested Krylov-in-MGR requires the vendored `hypre` build in ``hypre/``. Build `hypre`
with ``hypre/build-hypre.sh`` before you build `hypredrive`.

.. _ams:

AMS
~~~

The auxiliary-space Maxwell solver (AMS) preconditioner targets definite Maxwell
(curl-curl + mass) problems discretized with Nedelec (edge) elements in H(curl).
In addition to the system matrix, AMS requires two operator inputs from the
programmatic API. These inputs do not have YAML keys:

- The **discrete gradient** ``G`` through ``HYPREDRV_LinearSystemSetDiscreteGradient()``.
- The **vertex coordinate vectors** through ``HYPREDRV_LinearSystemSetCoordinates()``.

See the ``examples/src/C_maxwell`` driver for a complete usage example. The scalar
options below map directly to the corresponding ``HYPRE_AMSSet*`` routines.
Integer-valued options accept the raw `hypre` values in the
`hypre AMS reference <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html>`_.

- ``max_iter``, ``tolerance``, and ``print_level`` - See :ref:`amg` for a description of
  these variables. As a preconditioner the defaults are ``max_iter = 1`` and
  ``tolerance = 0.0``.

- ``dimension`` - spatial dimension (``2`` or ``3``). Default value is ``3``.

- ``cycle_type`` - AMS cycle type (multiplicative/additive variants). Default value is ``1``.

- ``relax_type``, ``relax_times``, ``relax_weight``, ``omega`` - smoothing options on the
  original matrix (``HYPRE_AMSSetSmoothingOptions``). Defaults are ``2`` (l1-scaled GS),
  ``1``, ``1.0``, and ``1.0``.

- ``proj_freq`` - subspace-projection frequency. Default value is ``5``.

- ``alpha_coarsen_type``, ``alpha_agg_levels``, ``alpha_relax_type``,
  ``alpha_strength_threshold``, ``alpha_interp_type``, ``alpha_Pmax``,
  ``alpha_coarse_relax_type`` - BoomerAMG options for the vector Poisson (Pi-space)
  problem (``HYPRE_AMSSetAlphaAMGOptions`` / ``...CoarseRelaxType``). Defaults are
  ``10``, ``1``, ``3``, ``0.25``, ``0``, ``0``, and ``8``.

- ``beta_coarsen_type``, ``beta_agg_levels``, ``beta_relax_type``,
  ``beta_strength_threshold``, ``beta_interp_type``, ``beta_Pmax``,
  ``beta_coarse_relax_type`` - BoomerAMG options for the scalar Poisson (G-space)
  problem (``HYPRE_AMSSetBetaAMGOptions`` / ``...CoarseRelaxType``). Defaults match the
  alpha options.

.. _ads:

ADS
~~~

The auxiliary-space divergence solver (ADS) preconditioner targets grad-div problems
discretized with Raviart-Thomas (face) elements in H(div). In addition to the system
matrix it requires three operator inputs provided through the programmatic API:

- The **discrete gradient** ``G`` through ``HYPREDRV_LinearSystemSetDiscreteGradient()``.
- The **discrete curl** ``C`` through ``HYPREDRV_LinearSystemSetDiscreteCurl()``.
- The **vertex coordinate vectors** through ``HYPREDRV_LinearSystemSetCoordinates()``.

The scalar options map onto the corresponding ``HYPRE_ADSSet*`` routines:

- ``max_iter``, ``tolerance``, ``print_level`` - as for AMS (defaults ``1``, ``0.0``, ``0``).

- ``cycle_type`` - ADS cycle type. Default value is ``1``.

- ``relax_type``, ``relax_times``, ``relax_weight``, ``omega`` - smoothing options
  (``HYPRE_ADSSetSmoothingOptions``). Defaults ``2``, ``1``, ``1.0``, ``1.0``.

- ``cheby_order``, ``cheby_fraction`` - Chebyshev smoothing options
  (``HYPRE_ADSSetChebySmoothingOptions``). Defaults ``2`` and ``0.3``.

- ``ams_cycle_type``, ``ams_coarsen_type``, ``ams_agg_levels``, ``ams_relax_type``,
  ``ams_strength_threshold``, ``ams_interp_type``, ``ams_Pmax`` - options for the
  auxiliary AMS (curl-curl) solve (``HYPRE_ADSSetAMSOptions``). Defaults ``11``, ``10``,
  ``1``, ``3``, ``0.25``, ``0``, ``0``.

- ``amg_coarsen_type``, ``amg_agg_levels``, ``amg_relax_type``,
  ``amg_strength_threshold``, ``amg_interp_type``, ``amg_Pmax`` - options for the
  auxiliary vector-Poisson AMG solve (``HYPRE_ADSSetAMGOptions``). Defaults ``10``,
  ``1``, ``3``, ``0.25``, ``0``, ``0``.
