.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _DriverExamples:

Driver Examples (hypredrive-cli)
================================

This section shows how to prepare input files and use the ``hypredrive-cli`` driver.
The examples solve different linear systems. Input files are in ``examples``. Reference
output is in ``examples/refOutput``.

.. note::
   These examples use `hypredrive` as a standalone driver with YAML input files.
   :ref:`LibraryExamples` shows how an application uses the ``libHYPREDRV`` API.

Some examples require data from https://zenodo.org/records/17471036. The repository
does not contain these data files. Download and extract the data with the CMake
``data`` target:

.. code-block:: bash

   $ cmake --build <build-dir> --target data

For a manual download, extract the data into ``data/``. Read ``data/README.md`` for
more information.

.. _CLIHelp:

Exploring the Input Schema (``-h/--help``)
------------------------------------------

The driver can describe its own YAML input schema from the command line. Passing
``-h`` or ``--help`` prints an overview of the top-level sections and exits without
running a solve:

.. code-block:: bash

   mpirun -np 1 ./hypredrive-cli --help

.. code-block:: text

   Usage: ./hypredrive-cli [options] <input.yml>

   List of valid sections for input.yml:

   Valid keys:
     general         <section>  Global configuration settings
     linear_system   <section>  Linear system settings
     solver          <section>  Linear solver settings
     preconditioner  <section>  Preconditioner settings

   Nested topics:
     general
     linear_system
     solver
     preconditioner

To inspect a specific section, append a *topic path* after ``--help``. Use ``:``
to separate nested blocks, as in the :ref:`CLIOverrides`. For
example, ``--help solver`` lists the available solver types, the keys each accepts, and
the nested topics you can explore further:

.. code-block:: bash

   mpirun -np 1 ./hypredrive-cli --help solver

.. code-block:: text

   Help for solver
   Linear solver settings

   Accepted values:
     pcg (0)
     gmres (1)
     fgmres (2)
     bicgstab (3)

   Valid keys:
     pcg       <solver-type>  Preconditioned Conjugate Gradient solver block
     gmres     <solver-type>  Generalized Minimal RESidual solver block
     fgmres    <solver-type>  Flexible Generalized Minimal RESidual solver block
     bicgstab  <solver-type>  Bi-Conjugate Gradient Stabilized solver block

   Nested topics:
     pcg
     gmres
     fgmres
     bicgstab

You can descend as far as the schema goes, down to an individual key:

.. code-block:: bash

   mpirun -np 1 ./hypredrive-cli --help solver:gmres:max_iter

.. code-block:: text

   Help for solver:gmres:max_iter
   max_iter  <value>  Maximum number of iterations

A few conventions appear in the output:

- ``<one of>`` marks a key that takes an enumerated value. The output lists the accepted
  values below the key. The parenthetical integer is the internal `hypre` code. You can
  also supply this code as the value.
- Sections that are *sequences* (such as ``preconditioner:mgr:level`` or
  ``preconditioner:reuse:adaptive:components``) accept a numeric item index, shown as
  ``<index>`` under *Nested topics*. Use any non-negative integer in the path to inspect
  an item, for example ``--help preconditioner:mgr:level:0:f_relaxation``.
- The topic path is case-insensitive and accepts either ``:`` or ``.`` as the separator.

.. note::

   ``--help`` shows the schema in your build. It shows version-dependent options only
   when the current `hypre` version supports them.

.. _CLIOverrides:

CLI Overrides (``-a/--args``)
-----------------------------

You can keep a base YAML file and override selected parameters from the command
line.

The ``-a`` or ``--args`` option introduces command-line key/value pairs in this form:

- ``--path:to:key <value>``

The driver interprets ``path:to:key`` as a YAML path. The ``:`` character
separates nested blocks. The driver applies overrides after it reads the YAML
file. Thus, the command line takes precedence.

Examples (based on ``examples/ex1.yml``):

.. code-block:: bash

   # Keep solver as PCG, but change max_iter
   mpirun -np 1 ./hypredrive-cli examples/ex1.yml -q -a --solver:pcg:max_iter 50

.. code-block:: bash

   # Switch solver type and set a nested option
   mpirun -np 1 ./hypredrive-cli examples/ex1.yml -q -a --solver gmres --solver:gmres:max_iter 30

Use these override rules:

- Put each key/value pair after ``-a``.
- Add settings that do not occur in the base YAML file when necessary.
- Separate each key and value with a space, for example ``--solver:pcg:max_iter 50``.
- Quote values that contain spaces or YAML syntax, such as ``[1, 2]``.

.. _Example1:

Example 1: Minimal configuration
--------------------------------

In this example, we solve a basic linear system using an `AMG-PCG` solver with default
settings. The input file contains only the information that `hypredrive-cli`
requires.

The matrix comes from a seven-point finite-difference form of the Laplace equation on a
10×10×10 Cartesian grid. The right-hand side is a vector of ones. The files contain one
MPI partition. Thus, this example requires one process.

This example requires the ``ps3d10pt7`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Prepare your linear system files (``matrix_filename`` and ``rhs_filename``).
2. Use the YAML configuration file ``ex1.yml``:

.. literalinclude:: ../../examples/ex1.yml
   :language: yaml

3. Run `hypredrive-cli` with the configuration file:

.. code-block:: bash

    $ mpirun -np 1 ./hypredrive-cli examples/ex1.yml

4. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex1.txt
   :language: text

Run `hypredrive-cli` from the top project directory. Otherwise, set
``matrix_filename`` and ``rhs_filename`` relative to the current directory.

.. _Example2:

Example 2: Parallel run with full AMG configuration
---------------------------------------------------

In this example, we solve the same problem as in the previous example, but partitioned for
`4` processes. The configuration file shows all available `PCG` and `AMG`
input options.

This example requires the ``ps3d10pt7`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Prepare your linear system files.

2. Use the YAML configuration file ``ex2.yml``:

.. literalinclude:: ../../examples/ex2.yml
   :language: yaml

3. Run `hypredrive-cli` with the configuration file:

.. code-block:: bash

    $ mpirun -np 4 ./hypredrive-cli examples/ex2.yml

4. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex2.txt
   :language: text

.. _Example3:

Example 3: Minimal multigrid reduction strategy
-----------------------------------------------

In this example, we solve a linear system derived from the discretization of a
compositional flow problem from `GEOS <https://github.com/GEOS-DEV/GEOS>`_. Details about
the generation process are in ``data/compflow6k/README.md``. This
example uses a `MGR-GMRES` solver. It shows the minimum configuration for the
multigrid reduction preconditioner in this linear system.

This example requires the ``compflow6k`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Prepare your linear system files.

2. Use the YAML configuration file ``ex3.yml``:

.. literalinclude:: ../../examples/ex3.yml
   :language: yaml

3. Run `hypredrive-cli` with the configuration file:

.. code-block:: bash

    $ mpirun -np 1 ./hypredrive-cli examples/ex3.yml

4. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex3.txt
   :language: text

Example 4: Advanced multigrid reduction strategy
------------------------------------------------

In this example, we solve the same problem as before, but partitioned for 4
processes. This example shows an advanced `MGR` setup with multiple options.

This example requires the ``compflow6k`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Prepare your linear system files.

2. Use the YAML configuration file ``ex4.yml``:

.. literalinclude:: ../../examples/ex4.yml
   :language: yaml

3. Run `hypredrive-cli` with the configuration file:

.. code-block:: bash

    $ mpirun -np 4 ./hypredrive-cli examples/ex4.yml

4. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex4.txt
   :language: text

Example 5: Spreading input parameters in multiple files
-------------------------------------------------------

In this example, we solve the same problem as in example 3, but using the same solver and
preconditioner parameters as in example 4. In addition, we define these parameters in
separate files. The ``include`` keyword adds these files to the main input file.

This example requires the ``compflow6k`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Prepare your linear system files.

2. Define the input file containing the solver parameters ``ex5-gmres.yml``:

.. literalinclude:: ../../examples/ex5-gmres.yml
   :language: yaml

3. Define the input file containing the preconditioner parameters ``ex5-mgr.yml``:

.. literalinclude:: ../../examples/ex5-mgr.yml
   :language: yaml

4. Define the main input file ``ex5.yml``:

.. literalinclude:: ../../examples/ex5.yml
   :language: yaml

5. Run `hypredrive-cli` with the configuration file:

.. code-block:: bash

    $ mpirun -np 1 ./hypredrive-cli examples/ex5.yml

6. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex5.txt
   :language: text

.. _Example6:

Example 6: Full eigenspectrum computation (single rank)
-------------------------------------------------------

In this example, we enable the debug/analysis eigenspectrum capability to compute the
full eigenspectrum of the matrix with dense LAPACK routines. Use this path for
small matrices and single-rank runs.

This example requires eigenspectrum support. Configure `hypredrive` with
``-DHYPREDRV_ENABLE_EIGSPEC=ON``.

This example requires the ``compflow6k`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Use the YAML configuration file ``ex6.yml``:

.. literalinclude:: ../../examples/ex6.yml
   :language: yaml

2. Run `hypredrive-cli` with the configuration file (single rank):

.. code-block:: bash

   $ mpirun -np 1 ./hypredrive-cli examples/ex6.yml

3. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex6.txt
   :language: text

4. (Optional) Plot the full eigenspectrum of the preconditioned matrix using the `eigplot.py` script:

.. code-block:: bash

   $ scripts/eigplot.py -f data/compflow6k/np1/eig.values.txt

.. _Example7:

Example 7: Solving a sequence of linear systems
-----------------------------------------------

In this example, we solve a sequence of linear systems stored in a hierarchical directory layout.
The systems originate from a single-rank (np1) GEOS multiphase poromechanics benchmark.
We use MGR-preconditioned FGMRES (MGR-FGMRES) as the linear solver.
See ``data/poromech2k/README.md`` for more details about the problem in GEOS.

This example requires the ``poromech2k`` dataset. The data instructions at the start
of this page explain how to retrieve it.

1. Use the YAML configuration file ``ex7.yml``:

.. literalinclude:: ../../examples/ex7.yml
   :language: yaml

2. Run `hypredrive-cli` with the configuration file (single rank):

.. code-block:: bash

   $ mpirun -np 1 ./hypredrive-cli examples/ex7.yml

3. Compare your output with this shortened reference:

.. literalinclude:: ../../examples/refOutput/ex7.txt
   :language: text

4. (Optional) Plot the sparsity pattern of the matrices using the `spyplot.py` script:

.. code-block:: bash

   $ scripts/spyplot.py -d data/poromech2k/np1 -r 0:24 -l -t 1e-20

.. _Example8:

Example 8: Multiple preconditioner variants in one YAML
-------------------------------------------------------

This example demonstrates how to run several preconditioner variants defined as a
YAML sequence under a single ``preconditioner`` block. `hypredrive-cli` will loop over each
variant and report a separate stats entry per variant while reusing the same linear system.

This example uses the ``ps3d10pt7`` dataset for multiple ranks. The data instructions
at the start of this page explain how to retrieve it.

1. Use the YAML configuration file ``ex8.yml``:

.. literalinclude:: ../../examples/ex8.yml
   :language: yaml

2. Run `hypredrive-cli` with the configuration file (single rank):

.. code-block:: bash

   $ mpirun -np 1 ./hypredrive-cli examples/ex8.yml -q

3. Compare your output with this reference:

.. literalinclude:: ../../examples/refOutput/ex8.txt
   :language: text

4. (Optional) Plot the timing or iteration bars with ``scripts/analyze_statistics.py``.
   Use ``-ln`` to provide one label for each table entry in table order:

.. code-block:: bash

   $ scripts/analyze_statistics.py -f examples/refOutput/ex8.txt -m bar -p total \
       -ln "AMG-1" "AMG-2" "AMG-3" "AMG-4" "AMG-5" \
       -s ex8.svg -T "Example 8 total time"

   # Use -p iters to compare iteration counts instead of timings

.. only:: html

   .. figure:: figures/ex8_total_bar.svg
      :alt: Total time per preconditioner variant (Example 8)
      :width: 80%
      :align: center

      Total time (setup + solve) for each preconditioner variant in Example 8.

.. only:: latex

   The HTML manual includes a rendered bar chart for this example. The PDF build omits
   it because the source asset is SVG-only.

.. note::
   ``examples/ex8-multi-1.yml`` provides the same input in multiple files. Each YAML file
   contains one preconditioner variant. The ``include`` key combines the files. The solver
   settings, iteration counts, and residuals match the single-file version. Reported times
   can have small differences.
