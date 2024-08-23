.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _InputFileStructure:

Input File Structure
====================

`hypredrive` uses a configuration file in YAML format to specify the parameters and settings
for the program's execution. Below is a detailed explanation of each section in the configuration
file. In general, the various keywords are optional and, if not explicitly defined by the
user, default values are used for them. On the other hand, some keywords such as
``linear_system`` are mandatory and, thus, are marked with `required` or `possibly
required`, depending on the value of other keywords.

.. note::

   The YAML file parser in `hypredrive` is case-insensitive, meaning that it works
   regardless of the presence of lower-case, upper-case, or a mixture of both when
   defining keys and values in the input file.


General Settings
----------------

The ``general`` section contains global settings that apply to the entire execution of
`hypredrive`. This section is optional.

- ``warmup`` - If set to `yes`, `hypredrive` will perform a warmup execution to
  ensure more accurate timing measurements. If `no`, no warmup is performed. The default
  value for this parameter is `yes`.

- ``statistics`` - If set to `yes`, `hypredrive` will display a statistics summary
  at the end of the run reporting execution times. If `no`, no statistics reporting is
  performed. The default value for this parameter is `yes`.

- ``use_milisec`` - Show timings on the statistics summary table in milliseconds. The
  default value is `no`, which uses seconds instead.

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


An example code block for the ``general`` section is given below:

.. code-block:: yaml

    general:
      warmup: yes
      statistics: yes
      use_millisec: no
      num_repetitions: 1
      dev_pool_size: 8.0
      uvm_pool_size: 8.0
      host_pool_size: 8.0
      pinned_pool_size: 0.5

Linear System
-------------

The ``linear_system`` section describes the linear system that `hypredrive` will solve. This
section is required.

- ``exec_policy`` - Determines whether the linear system is to be solved on the ``host``
  (CPU) or ``device`` (GPU). When hypre is built without GPU support, the default value
  for this parameter is ``host``; otherwise, the default value is ``device``.

- ``type`` - The format of the linear system matrix. Available options are ``ij`` and
  ``mtx``. The default value for this parameter is ``ij``.

- ``matrix_filename`` - (Required) The filename of the linear system matrix. This
  parameter does not have a default value.

- ``precmat_filename`` - The filename of the linear system matrix used for computing the
  preconditioner, which, by default, is set to the original linear system matrix.

- ``rhs_filename`` - (Possibly required) The filename of the linear system right hand side
  vector. This parameter does not have a default value and it is required when the
  ``rhs_mode`` is set to ``file``.

- ``x0_filename`` - (Possibly required) The filename of the initial guess for the linear
  system left hand side vector. This parameter does not have a default value and it is
  required when the ``init_guess_mode`` is set to ``file``.

- ``dofmap_filename`` - (Possibly required) The filename of the degrees of freedom maping
  array (`dofmap`) for the linear system. This parameter does not have a default value and it is
  required when the ``mgr`` preconditioner is used.

- ``init_guess_mode`` - Choice of initial guess vector. Available options are:

  - ``zeros``: generates a vector of zeros.
  - ``ones``: generates a vector of ones.
  - ``random``: generates a vector of random numbers between `0` and `1`.
  - ``file``: vector is read from file.

  The default value for this parameter is ``file``.

- ``rhs_mode`` - Choice of initial guess vector. Available options are the same as for
  ``init_guess_mode``.

- ``dirname`` - (Possibly required) Name of the top-level directory storing the linear
  system data. This option helps remove possible redundancies when informing filenames
  for the linear system data. This parameter does not have a default value.

- ``matrix_basename`` - (Possibly required) Common prefix used for the filenames of linear
  system matrices. It can be used to solve multiple matrices stored in a shared
  directory. This parameter does not have a default value.

- ``rhs_basename`` - (Possibly required) Common prefix used for the filenames of linear
  system right hand sides. It can be used to solve multiple RHS stored in a shared
  directory. This parameter does not have a default value.

- ``dofmap_basename`` - (Possibly required) Common prefix used for the filenames of
  `dofmap` arrays. This parameter does not have a default value.

- ``init_suffix`` - (Possibly required) Suffix number of the first linear system of a
  sequence of systems to be solved.

- ``last_suffix`` - (Possibly required) Suffix number of the last linear system of a
  sequence of systems to be solved.

- ``digits_suffix`` - (Optional) Number of digits used to build complete filenames when
  using the ``basename`` or ``dirname`` options. This parameter has a default value of 5.

- ``precon_reuse`` - (Optional) Frequency for reusing the preconditioner when solving multiple
  linear systems. This parameter has a default value of 0 meaning that the preconditioner
  is rebuilt for every linear system in a sequence.


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
      exec_policy: device

Solver
------

The ``solver`` section is mandatory and it specifies the Krylov solver configuration. The
available options for the Krylov solver type are:

- ``pcg`` - preconditioned conjugate gradient.
- ``bicgstab`` - bi-conjugate gradient stabilized.
- ``gmres`` - generalized minimal residual.
- ``fgmres`` - flexible generalized minimal residual.

The solver type must be entered as a key in a new indentation level under ``solver``.

.. _PCG:

PCG
^^^

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
^^^^^^^^

The available keywords to further configure the bi-conjugate gradient stabilized solver
(``bicgstab``) are all optional and given below:

- ``min_iter`` - Minimum number of iterations. Available values are any positive integer.

- ``max_iter``, ``print_level``, ``relative_tol``, ``absolute_tol``, ``residual_tol``, and
  ``conv_fac_tol`` - See :ref:`PCG` for a description of these variables.

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

.. _GMRES:

GMRES
^^^^^

The available keywords to further configure the generalized minimal residual solver
(``gmres``) are all optional and given below:

- ``skip_real_res_check`` - Skip calculation of the real residual when evaluating
  convergence. Available values are `yes` and `no`. Default value is `no`.

- ``krylov_dim`` - Dimension of the krylov space. Available values are any positive
  integer. Default value is `30`.

- ``min_iter``, ``max_iter``, ``print_level``, ``rel_change``, ``relative_tol``,
  ``absolute_tol``, and ``conv_fac_tol`` - See :ref:`PCG` for a description of these
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
^^^^^^

The available keywords to further configure the flexible generalized minimal residual
solver (``fgmres``) are all optional and given below:

- ``min_iter``, ``max_iter``, ``krylov_dim``, ``print_level``, ``relative_tol``,
  ``absolute_tol`` - See :ref:`GMRES` for a description of these variables.

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

.. _AMG:

AMG
^^^

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
    version. Available values are any positive integer. Default value is `off`.

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
    - ``chaotic-hgs``: chaotic hybrid Gauss-Seidel.
    - ``hsgs``: hybrid symmetric Gauss-Seidel.
    - ``jacobi``: Jacobi (based on SpMVs).
    - ``l1-hsgs``: L1-scaled hybrid symmetric Gauss-Seidel.
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
    - ``forward-hl1gs``: forward hybrid L1-scaled Gauss-Seidel (default).
    - ``cg``: conjugate gradient.
    - ``chebyshev``: chebyshev polinomial.
    - ``l1-jacobi``: L1-scaled Jacobi.
    - ``l1sym-hgs``: L1-scaled symmetric hybrid Gauss-Seidel (with convergent L1 factor).

  - ``up_type`` - relaxation method used in the post-smoothing stage. For detailed
    information, see `HYPRE_BoomerAMGSetRelaxType
    <https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html#_CPPv427HYPRE_BoomerAMGSetRelaxType12HYPRE_Solver9HYPRE_Int>`_. Available
    options are:

    - ``jacobi_non_mv``: legacy Jacobi implementation.
    - ``backward-hgs``: backward hybrid Gauss-Seidel.
    - ``chaotic-hgs``: chaotic hybrid Gauss-Seidel.
    - ``hsgs``: hybrid symmetric Gauss-Seidel.
    - ``jacobi``: Jacobi (based on SpMVs).
    - ``l1-hsgs``: L1-scaled hybrid symmetric Gauss-Seidel.
    - ``2gs-it1``: single iteration two stage Gauss-Seidel.
    - ``2gs-it2``: double iteration two stage Gauss-Seidel.
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

.. _ILU:

ILU
^^^

The incomplete LU factorization (ILU) preconditioner can be further configured by the
following optional keywords:

- ``max_iter``, ``tolerance``, and ``print_level`` - See :ref:`AMG` for a description of
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

.. _FSAI:

FSAI
^^^^

The factorized sparse approximate inverse (FSAI) preconditioner can be further configured by the
following optional keywords:

- ``max_iter``, ``tolerance``, and ``print_level`` - See :ref:`AMG` for a description of
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

MGR
^^^

The multigrid reduction (MGR) preconditioner can be further configured by the following
optional keywords:

- ``max_iter`` and ``tolerance`` - See :ref:`AMG` for a description of these variables.

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

   MGR cannot be fully defined by the ``mgr`` keyword only. Instead, it is also necessary
   to specify which types of degrees of freedom are treated as F points in each MGR level,
   i.e., the last level where a degree of freedom of a given type is present. This is done
   via the ``f_dofs`` keyword. For a minimal MGR configuration input example, see
   :ref:`Example3`.
