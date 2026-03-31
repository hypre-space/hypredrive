.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _InputFileStructure:

Input File Structure
====================

`hypredrive` uses YAML to specify parameters and settings. The YAML content can be
provided either as a **file on disk** (driver usage) or as an **in-memory string**
passed directly to ``HYPREDRV_InputArgsParse`` from application code. The structure
of the YAML is the same in both cases.

In general, the various keywords are optional and, if not explicitly defined by the
user, default values are used for them. Some keywords such as ``linear_system`` are
mandatory for driver use and are marked with `required` or `possibly required`,
depending on the value of other keywords.

.. note::

   The YAML parser in `hypredrive` is case-insensitive, meaning that it works
   regardless of the presence of lower-case, upper-case, or a mixture of both when
   defining keys and values.

General Settings
----------------

The ``general`` section contains global settings that apply to the entire execution of
`hypredrive`. This section is optional.

- ``warmup`` - If set to `yes`, `hypredrive` will perform a warmup execution to
  ensure more accurate timing measurements. If `no`, no warmup is performed. The default
  value for this parameter is `no`.

- ``name`` - Optional display label for the current ``HYPREDRV_t`` object. When set,
  statistics banners use it in messages such as ``STATISTICS SUMMARY for flow-solver:``.
  The default is empty, which preserves the unlabeled banner.

- ``statistics`` - Controls the verbosity of statistics reporting. Accepts integer values
  or boolean strings (`yes`/`no`, `on`/`off`, `true`/`false`). The default value is `1`
  (or `yes`). Available levels:

  - ``0`` (or `no`/`off`/`false`) - No statistics reporting.
  - ``1`` (or `yes`/`on`/`true`) - Display the basic statistics summary table with
    execution times, residual norms, and iteration counts for each solve entry.
  - ``2`` - Display the basic statistics summary table plus an aggregate summary table
    showing min, max, average, standard deviation, and total values for build, setup,
    and solve times, as well as iteration counts. The aggregate table is only shown when
    there are multiple entries (e.g., when using ``num_repetitions > 1``).

  In library mode, the configured statistics summary is printed automatically on rank 0
  when the owning ``HYPREDRV_t`` object is destroyed. Applications can still call
  ``HYPREDRV_StatsPrint`` earlier if they want an additional snapshot before teardown.

- ``statistics_filename`` - Optional file path for statistics output. Default is empty,
  which keeps writing to ``stdout``. When set, hypredrive appends each statistics
  snapshot to that file (for example, both explicit ``HYPREDRV_StatsPrint`` calls and
  library-mode auto-print on destroy). If the file cannot be opened, hypredrive emits a
  warning on ``stderr`` and falls back to ``stdout`` for that print.

- ``use_millisec`` - Show timings on the statistics summary table in milliseconds. The
  default value is `no`, which uses seconds instead.

- ``print_config_params`` - Print the parsed YAML tree to stdout on rank 0 after input
  parsing. Values: ``yes`` / ``no``. Default: ``yes`` in driver mode and ``no`` in
  library mode.

- ``num_repetitions`` - Specifies the number of times the operation should be
  repeated. Useful for benchmarking and profiling. The default value for this parameter is
  `1`.

- ``dev_pool_size`` - Initial size of the umpire's device memory pool. This parameter is
  neglected when hypre is *not* configured with umpire support. The default value for this
  parameter is `8 GB`.

- ``uvm_pool_size`` - Initial size of the umpire's unified virtual memory pool. This
  parameter is neglected when hypre is *not* configured with umpire support. The default
  value for this parameter is `8 GB`.

- ``host_pool_size`` - Initial size of the umpire's host memory pool. This parameter is
  neglected when hypre is *not* configured with umpire support. The default value for this
  parameter is `8 GB`.

- ``pinned_pool_size`` - Initial size of the umpire's host memory pool. This parameter is
  neglected when hypre is *not* configured with umpire support. The default value for this
  parameter is `512 MB`.

- ``exec_policy`` - Determines whether computations are performed on the ``host`` (CPU) or
  ``device`` (GPU). When hypre is built without GPU support the only valid value is ``host``;
  otherwise the default is ``device``.

- ``use_vendor_spgemm`` - Use vendor-optimized sparse matrix-matrix multiplication (SpGEMM)
  kernels when available. Values: ``yes`` / ``no``. Default: ``yes`` on GPU-enabled builds
  and ``no`` otherwise.

- ``use_vendor_spmv`` - Use vendor-optimized sparse matrix-vector multiplication (SpMV)
  kernels when available. Values: ``yes`` / ``no``. Default: ``yes`` on GPU-enabled builds
  and ``no`` otherwise.


An example code block for the ``general`` section is given below:

.. code-block:: yaml

    general:
      warmup: no
      statistics: yes
      use_millisec: no
      print_config_params: yes
      num_repetitions: 1
      dev_pool_size: 8.0
      uvm_pool_size: 8.0
      host_pool_size: 8.0
      pinned_pool_size: 0.5
      exec_policy: device
      use_vendor_spgemm: yes
      use_vendor_spmv: yes

Linear System
-------------

The ``linear_system`` section describes the linear system that `hypredrive` will solve. This
section is required.

- ``type`` - The format of the linear system matrix. Available options are ``ij`` and
  ``mtx``. The default value for this parameter is ``ij``.

- ``matrix_filename`` - (Required) The filename of the linear system matrix. This
  parameter does not have a default value.

- ``precmat_filename`` - The filename of the linear system matrix used for computing the
  preconditioner, which, by default, is set to the original linear system matrix.
  In the C API, ``HYPREDRV_LinearSystemSetPrecMatrix(h, mat)`` overrides this file-based
  path when ``mat`` is non-NULL.

- ``rhs_filename`` - (Possibly required) The filename of the linear system right hand side
  vector. This parameter does not have a default value and it is required when the
  ``rhs_mode`` is set to ``file``.

- ``x0_filename`` - (Possibly required) The filename of the initial guess for the linear
  system left hand side vector. This parameter does not have a default value and it is
  required when the ``init_guess_mode`` is set to ``file``.
  In the C API, ``HYPREDRV_LinearSystemSetInitialGuess(h, vec)`` overrides this
  file/default path when ``vec`` is non-NULL.

- ``xref_filename`` - (Optional) The filename of the reference solution vector used by
  tagged residual/error reporting. In the C API,
  ``HYPREDRV_LinearSystemSetReferenceSolution(h, vec)`` overrides file/default behavior
  when ``vec`` is non-NULL.

.. _linear_system_dofmap:

Degrees of Freedom Map
~~~~~~~~~~~~~~~~~~~~~~

- ``dofmap_filename`` - (Possibly required) The filename of the degrees of freedom maping
  array (`dofmap`) for the linear system. This parameter does not have a default value and it is
  required when the ``mgr`` preconditioner is used.

- ``init_guess_mode`` - Choice of initial guess vector. Available options are:

  - ``zeros``: generates a vector of zeros.
  - ``ones``: generates a vector of ones.
  - ``random``: generates a vector of random numbers between `0` and `1`.
  - ``file``: vector is read from file.

  The default value for this parameter is ``file``.

  .. note::
     In the library API, passing ``NULL`` to
     ``HYPREDRV_LinearSystemSetInitialGuess`` /
     ``HYPREDRV_LinearSystemSetReferenceSolution`` /
     ``HYPREDRV_LinearSystemSetPrecMatrix`` preserves the file/default behavior described
     in this section.

- ``rhs_mode`` - Determines how the right-hand side vector is provided. Available options
  are the same as for ``init_guess_mode``.

- ``dirname`` - (Possibly required) Name of the top-level directory storing the linear
  system data. This option helps remove possible redundancies when informing filenames
  for the linear system data. This parameter does not have a default value.

- ``sequence_filename`` - (Optional) Path to a lossless-compressed sequence container
  produced by ``hypredrive-lsseq``. When set, matrix/RHS/(optional) dofmap are read
  from the container instead of ``dirname``/``*_filename``/``*_basename`` fields. The
  number of systems is inferred from the container metadata.

- ``matrix_basename`` - (Possibly required) Common prefix used for the filenames of linear
  system matrices. It can be used to solve multiple matrices stored in a shared
  directory. This parameter does not have a default value.

- ``precmat_basename`` - (Possibly required) Common prefix used for the filenames of
  linear system matrices employed in the computation of the preconditioner. If not specified,
  the matrices used for preconditioning purposes are set to the original linear system
  matrices formed with `matrix_basename`.

- ``rhs_basename`` - (Possibly required) Common prefix used for the filenames of linear
  system right hand sides. It can be used to solve multiple RHS stored in a shared
  directory. This parameter does not have a default value.

- ``dofmap_basename`` - (Possibly required) Common prefix used for the filenames of
  `dofmap` arrays. This parameter does not have a default value.

- ``timestep_filename`` - (Optional) File that maps timesteps to linear-system ids for
  preconditioner reuse-by-timestep. When using ``sequence_filename``, this can be omitted
  if timestep metadata is embedded in the compressed container.

- ``init_suffix`` - (Possibly required) Suffix number of the first linear system of a
  sequence of systems to be solved. Cannot be used together with ``set_suffix``.

- ``last_suffix`` - (Possibly required) Suffix number of the last linear system of a
  sequence of systems to be solved. Cannot be used together with ``set_suffix``.

- ``set_suffix`` - (Optional) A list of suffix numbers for each linear system in the
  sequence (e.g. ``set_suffix: [0, 2, 5]``). Use this when the sequence of systems does
  not use consecutive suffixes. Cannot be used together with ``init_suffix`` or
  ``last_suffix``; if ``set_suffix`` is set, the number of systems is the length of the
  list.

- ``digits_suffix`` - (Optional) Number of digits used to build complete filenames when
  using the ``basename`` or ``dirname`` options. This parameter has a default value of 5.

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

- Dumps are written under ``output_dir/<object_name>/`` (no stage subdirectory).
- Each dumped system gets a continuous folder index: ``ls_00000``, ``ls_00001``, ...
  in dump creation order.
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

    hypredrive-cli --config input.yml --linear_system:print_system:enabled on


An example code block for the ``linear_system`` section is given below:

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
      --last-suffix 24 \
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

When ``sequence_filename`` is present, hypredrive reads matrix/RHS/(optional) dofmap from
the container. If timestep metadata is embedded in the container, preconditioner reuse by
timestep can use it when ``timestep_filename`` is omitted.

Packed sequence files use batched compression per part (one compressed blob per part for
values, RHS, and dofmap across all systems), which yields good compression ratios. They
embed a mandatory small manifest block (uncompressed) with provenance/debug information
(resolved suffix range, input paths, codec, build metadata). This does not affect runtime
reconstruction, but is useful when inspecting packed artifacts. See :ref:`utilities` for
the internal container structure.

Solver
------

The ``solver`` section is mandatory and it specifies the Krylov solver configuration. The
available options for the Krylov solver type are:

- ``pcg`` - preconditioned conjugate gradient.
- ``bicgstab`` - bi-conjugate gradient stabilized.
- ``gmres`` - generalized minimal residual.
- ``fgmres`` - flexible generalized minimal residual.

The solver type must be entered as a key in a new indentation level under ``solver``.

Scaling
~~~~~~~

The ``scaling`` subsection under ``solver`` enables optional diagonal scaling of the linear system before preconditioner setup and Krylov solve. When enabled, the system is transformed as :math:`B = M A M`, :math:`c = M b`, solved as :math:`B y = c`, and the solution is recovered as :math:`x = M y`.

Available keywords:

- ``enabled`` - Turn on/off scaling. Available values are ``yes`` or ``no``. Default value is ``no``.

- ``type`` - Scaling strategy. Available values are:

  - ``rhs_l2`` - Scalar scaling based on the L2 norm of the RHS vector :math:`b`. Computes :math:`s = 1/\sqrt{\|b\|_2}` and applies uniform scaling :math:`M = s I`.

  - ``dofmap_mag`` - Vector scaling computed using Hypre's tagged scaling API. Requires a dofmap to be provided (see :ref:`linear_system_dofmap`). Uses ``HYPRE_ParCSRMatrixComputeScalingTagged`` with scaling type 1 to compute per-DOF-type scaling weights based on matrix magnitude.

  - ``dofmap_custom`` - Vector scaling using user-provided custom scaling values. Requires a dofmap to be provided (see :ref:`linear_system_dofmap`). The number of custom values must match the number of unique DOF types in the dofmap. Each DOF type is scaled by the corresponding value in the ``custom_values`` array.

- ``custom_values`` - (Required for ``dofmap_custom``) Array of scaling values, one per unique DOF type in the dofmap. Must be provided as a YAML sequence. The number of entries must match the number of unique tags in the dofmap (e.g., if dofmap has tags 0, 1, 2, then ``custom_values`` must have 3 entries).

.. note::
   Scaling requires hypre >= 3.0.0. If scaling is enabled on an older build, YAML parsing
   succeeds but scaling is silently disabled at runtime.

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


.. _pcg:

PCG
~~~

The available keywords to further configure the preconditioned conjugate gradient solver
(``pcg``) are all optional and given below:

- ``max_iter`` - Maximum number of iterations. Available values are any positive integer.

- ``two_norm`` - Turn on/off L2 norm for the residual. Available values are ``yes`` or
  ``no``. Default value is ``yes``.

- ``rel_change`` - Turn on/off an additional convergence criteria that checks for a relative
  change in the solution vector. Available values are ``yes`` or ``no``. Default value is
  ``no``.

- ``print_level`` - Verbosity level for the iterative solver. `1` turns on convergence
  history reporting. Default value is `0`.

- ``relative_tol`` - Relative tolerance based on the norm of the residual vector and used
  for determining convergence of the iterative solver. Available values are any positive
  floating point number. Default value is ``1.0e-6``.

- ``absolute_tol`` - Absolute tolerance used for determining convergence of the iterative
  solver. Available values are any positive floating point number. Default value is
  ``0.0``, meaning that the absolute tolerance-based convergence criteria is inactive.

- ``residual_tol`` - Tolerance used for determining convergence of the iterative solver
  and based on the norm of the difference between subsequent residual vectors. Available
  values are any positive floating point number. Default value is ``0.0``, meaning that
  the residual tolerance-based convergence criteria is inactive.

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
        rel_change: no
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0
        residual_tol: 0.0
        conv_fac_tol: 0.0

BiCGSTAB
~~~~~~~~

The available keywords to further configure the bi-conjugate gradient stabilized solver
(``bicgstab``) are all optional and given below:

- ``min_iter`` - Minimum number of iterations. Available values are any positive integer.

- ``max_iter``, ``print_level``, ``relative_tol``, ``absolute_tol``, ``residual_tol``, and
  ``conv_fac_tol`` - See :ref:`pcg` for a description of these variables.

The code block representing the default parameter values for the ``solver:bicgstab`` section is
given below:

.. code-block:: yaml

    solver:
      bicgstab:
        min_iter: 0
        max_iter: 100
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0
        residual_tol: 0.0
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

The code block representing the default parameter values for the ``solver:gmres`` section is
given below:

.. code-block:: yaml

    solver:
      gmres:
        min_iter: 0
        max_iter: 300
        skip_real_res_check: no
        krylov_dim: 30
        rel_change: no
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0
        conv_fac_tol: 0.0

FGMRES
~~~~~~

The available keywords to further configure the flexible generalized minimal residual
solver (``fgmres``) are all optional and given below:

- ``min_iter``, ``max_iter``, ``krylov_dim``, ``print_level``, ``relative_tol``,
  ``absolute_tol`` - See :ref:`gmres` for a description of these variables.

The code block representing the default parameter values for the ``solver:fgmres`` section is
given below:

.. code-block:: yaml

    solver:
      fgmres:
        min_iter: 0
        max_iter: 300
        krylov_dim: 30
        print_level: 1
        relative_tol: 1.0e-6
        absolute_tol: 0.0

Preconditioner
--------------

The ``preconditioner`` section is mandatory and it specifies the preconditioner
configuration. Available options for the preconditioner type are:

- ``amg`` - algebraic multigrid (BoomerAMG).
- ``ilu``: incomplete LU factorization.
- ``fsai``: factorized sparse approximate inverse.
- ``mgr``: multigrid reduction.

The preconditioner type must be entered as a key in a new indentation level under
``preconditioner``.

Preconditioner presets
~~~~~~~~~~~~~~~~~~~~~~

Presets are named default configurations that select a preconditioner and apply
a small set of tuned settings. They are useful when you want a reasonable
default without enumerating all options.

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

When solving a **sequence of linear systems** (e.g. multiple right-hand sides or time
steps), you can reuse the same preconditioner across several systems to avoid repeated setup
cost. The preconditioner is rebuilt only at chosen linear-system indices; for the rest, the
previous factorization is applied as-is.

A ``reuse`` subsection under ``preconditioner`` configures this behavior.

Static reuse
^^^^^^^^^^^^

The default policy is ``static``. It rebuilds on a fixed schedule:

- **``enabled``** – Turn reuse logic on or off. Values: ``yes`` / ``no``. Default: ``no``.
- **``policy``** – ``static`` or ``adaptive``. Default: ``static`` unless an
  ``adaptive`` subsection is present.
- **``frequency``** – Nonnegative integer. Rebuild when ``(linear_system_index) mod
  (frequency + 1) == 0``. So ``0`` = rebuild every system, ``1`` = rebuild every other, etc.
- **``linear_system_ids``** – Explicit list of **0-based** linear-system indices at which to
  rebuild (e.g. ``[0, 5, 10]``). Alias: ``linear_solver_ids``. Cannot be combined with
  ``frequency`` or ``per_timestep``.
- **``per_timestep``** – If ``yes``, ``frequency`` is applied **per timestep**: rebuild at
  the first system of each timestep, then every ``(frequency+1)``-th system within that
  timestep. In file/driver workflows this requires ``linear_system.timestep_filename`` to point
  to a timestep file. In embedded library-mode workflows, the same behavior can be driven by
  object-scoped level-0 timestep annotations on the active ``HYPREDRV_t`` object.
  Values: ``yes`` / ``no``. Cannot be combined with ``linear_system_ids``.

The timestep file (used in driver / file-based workflows when ``per_timestep: yes``) must list
how linear systems map to timesteps. First line: total number of timesteps. Each following line:
``timestep_id ls_start``, where ``ls_start`` is the 0-based index of the first linear system for
that timestep.

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

Example: reuse per timestep (rebuild at the first system of each timestep; requires
``linear_system.timestep_filename``):

.. code-block:: yaml

   linear_system:
     timestep_filename: timesteps.txt
   preconditioner:
     amg: {}
     reuse:
       enabled: yes
       per_timestep: yes
       frequency: 0

In embedded library mode, the same YAML block can be paired with object-scoped annotations such
as ``HYPREDRV_AnnotateLevelBegin(h, 0, "timestep-3", -1)`` /
``HYPREDRV_AnnotateLevelEnd(h, 0, "timestep-3", -1)`` instead of a timestep file.

Adaptive reuse
^^^^^^^^^^^^^^

Set ``type: adaptive`` to let HYPREDRV score recent solver behavior and rebuild when the
score crosses a threshold. This mode is separate from the static schedule above; combining it
with ``frequency``, ``linear_system_ids``, or ``per_timestep`` produces a parse-time error.

For the simplest opt-in, ``reuse: adaptive`` is valid shorthand. That form enables adaptive
reuse with HYPREDRV's built-in default score model.

.. code-block:: yaml

   preconditioner:
     mgr: {}
     reuse: adaptive

Top-level adaptive keys:

- **``guards.min_reuse_solves``** – Force reuse until this many solves have completed after a
  rebuild.
- **``guards.max_reuse_solves``** – Force rebuild once the current preconditioner has been
  reused this many times. ``-1`` disables the guard.
- **``guards.min_history_points``** – Minimum number of samples a score component needs before
  it can trigger a rebuild.
- **``guards.rebuild_on_new_level``** – Optional list of annotation levels that force a rebuild
  when the active level entry changes.
- **``adaptive.rebuild_threshold``** – Rebuild when the weighted score is at least this value.
- **``adaptive.positive_floor``** – Positive clamp used by geometric / harmonic / nonpositive
  power means and ratio-like transforms. Defaults to ``1.0e-12``.
- **``adaptive.components``** – Optional sequence of score components. If omitted,
  HYPREDRV installs a built-in default component set.

Adaptive decisions are reported through HYPREDRV's existing internal logging framework when
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

If ``type: adaptive`` is selected and no explicit ``adaptive.components`` list is provided,
HYPREDRV uses the following built-in components:

- **``efficiency``** – ``metric: solve_overhead_vs_setup``, arithmetic mean, ``target: 1.0``,
  ``scale: 1.0``, ``transform.kind: raw``, ``transform.amortization_window: 10``,
  ``history.source: linear_solves``, ``history.max_points: 5``.
- **``stability``** – ``metric: iterations``, RMS mean, ``target: 1.5``,
  ``scale: 0.5``, ``transform.kind: ratio_to_baseline``, ``transform.baseline: rebuild``,
  ``history.source: linear_solves``, ``history.max_points: 5``.

The default model intentionally mixes one efficiency term and one iteration-stability term so
``reuse: adaptive`` works out of the box without level annotations or a hand-built component
list. Users can still override any part of the model by providing ``adaptive.components``
explicitly.

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

   MGR cannot be fully defined by the ``mgr`` keyword only. Instead, it is also necessary
   to specify which types of degrees of freedom are treated as F points in each MGR level,
   i.e., the last level where a degree of freedom of a given type is present. This is done
   via the ``f_dofs`` keyword. For a minimal MGR configuration input example, see
   :ref:`Example3`.

.. _amg:

AMG
~~~

The algebraic multigrid (BoomerAMG) preconditioner can be further configured by the
following optional keywords:

- ``max_iter`` - number of times the preconditioner is applied when it is
  called. Available values are any positive integer. Default value is `1`.

- ``tolerance`` - convergence tolerance of AMG when applied multiple times. Available
  values are any positive floating point number. Default value is `0.0`.

- ``print_level`` - Verbosity level for the preconditioner. Default value is `0`

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
    - ``mm_extended+e``
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

  - ``seq_amg_th`` - maximum size for agglomeration or redundant coarse grid
    solve. Smaller system are then solved with a sequential AMG. Available values are any
    non-negative integer. Default value is `0`.

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
    aggressive caorsening levels. Available values are any non-negative integer. Default
    value is `4`.

  - ``P12_trunc_factor`` - truncation factor for matrices P1 and P2 which are used to
    build 2-stage interpolation. Available values are any non-negative floating point
    number. Default value is `0.0`.

  - ``P12_max_elements`` - maximum number of elements per row for matrices P1 and P2 which
    are used to build 2-stage interpolation. Available values are any non-negative
    integer. Default value is `0`, meaning there is no maximum number of elements per row.

- ``relaxation`` - subsection detailing relaxation options:

  - ``down_type`` - relaxation method used in the pre-smoothing stage. For detailed
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
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
    - ``forward-hl1gs``: forward hybrid L1-scaled Gauss-Seidel (default).
    - ``backward-hl1gs``: backward hybrid L1-scaled Gauss-Seidel.
    - ``cg``: conjugate gradient.
    - ``chebyshev``: chebyshev polinomial.
    - ``l1-jacobi``: L1-scaled Jacobi.
    - ``l1sym-hgs``: L1-scaled symmetric hybrid Gauss-Seidel (with convergent L1 factor).

  - ``up_type`` - relaxation method used in the post-smoothing stage. For detailed
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
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
    - ``forward-hl1gs``: forward hybrid L1-scaled Gauss-Seidel.
    - ``backward-hl1gs``: backward hybrid L1-scaled Gauss-Seidel (default).
    - ``cg``: conjugate gradient.
    - ``chebyshev``: chebyshev polinomial.
    - ``l1-jacobi``: L1-scaled Jacobi.
    - ``l1sym-hgs``: L1-scaled symmetric hybrid Gauss-Seidel (with convergent L1 factor).

  - ``coarse_type`` - relaxation method used in the coarsest levels. For detailed
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
    - ``chebyshev``: chebyshev polinomial.
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
    the specific cycle. Default value is `-1`.

  - ``num_sweeps`` - number of pre and post-smoothing sweeps. Available values are any
    non-negative integer. Default value is `1`.

  - ``order`` - order in which the points are relaxed. For available
    options, see `HYPRE_BoomerAMGSetRelaxOrder
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_BoomerAMGSetRelaxOrder12HYPRE_Solver9HYPRE_Int>`_. Default value is `0`.

  - ``weight`` - relaxation weight for smoothed Jacobi and hybrid SOR. For available
    options, see `HYPRE_BoomerAMGSetRelaxWt
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv425HYPRE_BoomerAMGSetRelaxWt12HYPRE_Solver10HYPRE_Real>`_. Default value is `1.0`.

  - ``outer_weight`` - outer relaxation weight for hybrid SOR and SSOR. For available
    options, see `HYPRE_BoomerAMGSetOuterWt
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv425HYPRE_BoomerAMGSetOuterWt12HYPRE_Solver10HYPRE_Real>`_. Default value is `1.0`.

- ``relaxation`` - subsection detailing complex smoother options:

  - ``type`` - complex smoother type. For detailed information, see `HYPRE_BoomerAMGSetSmoothType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_BoomerAMGSetSmoothType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``fsai``: factorized sparse approximate inverse.
    - ``ilu``: incomplete LU factorization.
    - ``schwarz``: Additive/Multiplicative overlapping Schwarz.
    - ``pilut``: incomplete LU factorization via PILUT.
    - ``parasails``: sparse approximate inverse via Parasails.
    - ``euclid``: incomplete LU factorization via Euclid.

  - ``num_levels`` - number of levels starting from the finest one where complex smoothers
    are used. Available values are any non-negative integer. Default value is `0`.

  - ``num_sweeps`` - number of pre and post-smoothing sweeps used for the complex
    smoother. Available values are any non-negative integer. Default value is `1`.

The default parameter values for the ``preconditioner:amg`` section are represented in the
code block below:

.. code-block:: yaml

    preconditioner:
      amg:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        interpolation:
          prolongation_type: extended+i
          restriction_type: p_transpose
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
          coarse_sweeps: -1
          num_sweeps: 1
          order: 0
          weight: 1.0
          outer_weight: 1.0
        smoother:
          type: ilu
          num_levels: 0
          num_sweeps: 1

.. _ilu:

ILU
~~~

The incomplete LU factorization (ILU) preconditioner can be further configured by the
following optional keywords:

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

- ``lower_jac_iters`` - Number of iterations for solving the upper triangular system
  during the preconditioner's application phase. Available values are any positive
  integer. Default value is `5`. This option has effect only when ``tri_solve`` is set to
  zero.

- ``max_row_nnz`` - Maximum number if nonzeros per row when using ILUT. Available values
  are any positive integer. Default value is `200`.

- ``schur_max_iter`` - Maximum number of the Schur system solve. Available values
  are any positive integer. Default value is `5`. This option has effect only when
  ``type`` is greater or equal than `10`.

- ``droptol`` - Dropping tolerance for computing the triangular factors when using
  ILUT. Available values are any non-negative floating point numbers. Default value is
  `1.0e-2`.

- ``nsh_droptol`` - Dropping tolerance for computing the triangular factors when using
  NSH. Available values are any non-negative floating point numbers. Default value is
  `1.0e-2`.

The default parameter values for the ``preconditioner:ilu`` section are represented in the
code block below:

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

The factorized sparse approximate inverse (FSAI) preconditioner can be further configured by the
following optional keywords:

- ``max_iter``, ``tolerance``, and ``print_level`` - See :ref:`amg` for a description of
  these variables.

- ``type`` - algorithm type used for building FSAI. For available
  options, see `HYPRE_FSAISetAlgoType
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv421HYPRE_FSAISetAlgoType12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `1` (Adaptive) for CPUs and `3` (Static) for GPUs.

- ``ls_type`` - solver type for the local linear systems in FSAI. For available
  options, see `HYPRE_FSAISetLocalSolveType
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_FSAISetLocalSolveType12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `0` (Gauss-Jordan).

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

- ``threshold`` - Dropping tolerance for building the canditate pattern matrix. Available
  values are any non-negative floating point numbers. Default value is `1.0e-3`.

- ``kap_tolerance`` - Kaporin reduction factor. Available values are any non-negative
  floating point numbers. Default value is `1.0e-3`.

The default parameter values for the ``preconditioner:fsai`` section are represented in
the code block below:

.. code-block:: yaml

    preconditioner:
      fsai:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        algo_type: 1
        ls_type: 0
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

The multigrid reduction (MGR) preconditioner can be further configured by the following
optional keywords:

- ``max_iter`` and ``tolerance`` - See :ref:`amg` for a description of these variables.

- ``print_level`` - verbosity level for the preconditioner. For available
  options, see `HYPRE_MGRSetPrintLevel
  <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv422HYPRE_MGRSetPrintLevel12HYPRE_Solver9HYPRE_Int>`_. Default
  value is `0` (no printout).

- ``coarse_th`` - threshold for dropping small entries on the coarse grid. Available
  values are any non-negative floating point numbers. Default value is `0.0`, which means
  no dropping.

- ``level`` - special keyword for defining specific parameters for each MGR level. Each
  level is identified by its numeric ID starting from `0` (finest) and placed in
  increasing order on the next indentation level of the YAML input.

  - ``f_dofs`` - (Mandatory) Array containing the identifiers of F (fine) degrees of
    freedom to be treated in the current level. Available values are any integer numbers
    from `0` to `n_dofs - 1`, where `n_dofs` represent the unique number of degrees of
    freedom identifiers.

  - ``f_relaxation`` - relaxation method targeting F points. For available options, see
    `HYPRE_MGRSetLevelFRelaxType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_MGRSetLevelFRelaxType12HYPRE_SolverP9HYPRE_Int>`_. Default
    value is `0` (Jacobi). Use ``none`` to deactivate F-relaxation.

  - ``g_relaxation`` - global relaxation method targeting F and C points. For available
    options, see `HYPRE_MGRSetGlobalSmoothType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_MGRSetGlobalSmoothType12HYPRE_Solver9HYPRE_Int>`_. Default
    value is `2` (Jacobi). Use ``none`` to deactivate global relaxation.

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

  - ``f_relaxation`` also supports a nested ``mgr`` block (MGR-inside-MGR). When using
    nested MGR (requires Hypre ``>= 3.1.0`` with develop number ``>= 5``), the nested
    ``f_dofs`` labels are not re-labeled: nested levels continue to use the same
    DOF labels as the parent MGR dofmap (restricted to the parent level's F-block).

    Example:

    .. code-block:: yaml

        level:
          0:
            f_dofs: [0, 2]
            f_relaxation: mgr
              mgr:
                level:
                  0:
                    f_dofs: [2] # refers to the same parent label 2 (no relabeling)
                coarsest_level:
                  amg:

  - ``restriction_type`` - algorithm for computing the restriction operator. For available
    options, see `HYPRE_MGRSetRestrictType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv424HYPRE_MGRSetRestrictType12HYPRE_Solver9HYPRE_Int>`_. Default
    value is `0` (Injection).

  - ``prolongation_type`` - algorithm for computing the prolongation operator. For available
    options, see `HYPRE_MGRSetInterpType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv422HYPRE_MGRSetInterpType12HYPRE_Solver9HYPRE_Int>`_. Default
    value is `0` (Injection).

  - ``coarse_level_type`` - algorithm for computing the coarse level matrices. For available
    options, see `HYPRE_MGRSetCoarseGridMethod
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv428HYPRE_MGRSetCoarseGridMethod12HYPRE_SolverP9HYPRE_Int>`_. Default
    value is `0` (Galerkin).

- ``coarsest_level`` - special keyword for defining specific parameters for MGR's coarsest
  level.

  ``coarsest_level`` also supports the same nested Krylov solver block described above.

The default parameter values for the ``preconditioner:mgr`` section are represented in the
code block below:

.. code-block:: yaml

    preconditioner:
      mgr:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
        coarse_th: 0.0
        level:
          0:
            f_dofs: [1, 2] # Example usage where DOFs 1 and 2 are treated in MGR's 1st level
            f_relaxation: single
              sweeps: 1
            g_relaxation: none
            restriction_type: injection
            prolongation_type: jacobi
            coarse_level_type: rap

          1:
            f_dofs: [0] # Example usage where DOF 0 is treated in MGR's 2nd level
            f_relaxation: none
            g_relaxation:
              ilu: # ILU parameters can be specified with a new indentation level
            restriction_type: injection
            prolongation_type: jacobi
            coarse_level_type: rap

        coarsest_level:
          amg: # AMG parameters can be specified with a new indentation level

.. warning::

   Nested Krylov-in-MGR requires the vendored Hypre build in the ``hypre/`` folder. Make
   sure to build Hypre with ``hypre/build-hypre.sh`` before building HypreDrive.
