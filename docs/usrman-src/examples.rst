.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Examples:

Input File Examples
===================

This section provides several examples demonstrating how to set up input files and use
`hypredrive` for the solution of different types of linear system problems. All example
inputs can be found at the ``examples`` folder and a reference output for each example can
be found at ``examples/refOutput``.

.. _Example1:

Example 1: Minimal configuration
--------------------------------

In this example, we solve a basic linear system using an `AMG-PCG` solver with default
settings. This example showcases the minimum amount of information required in the input
file to execute `hypredrive`.

We consider a linear system matrix arising from a seven points finite differences
discretizaion of the Laplace equation on a `10x10x10` cartesian grid. Furthermore, the
right hand side is the vector of ones. Both data are read from file and partitioned for a
single MPI rank. Therefore, this example must be executed on a single process.

1. Prepare your linear system files (``matrix_filename`` and ``rhs_filename``).
2. Use the YAML configuration file ``ex1.yml``:

.. literalinclude:: ../../examples/ex1.yml
   :language: yaml

3. Run `hypredrive` with the configuration file:

.. code-block:: bash

    mpirun -np 1 ./hypredrive ex1.yml

4. Your output should look like:

.. literalinclude:: ../../examples/refOutput/ex1.txt
   :language: text

.. warning::
   Make sure that `hypredrive` is executed from the top level project folder in order for
   the relative paths in ``matrix_filename`` and ``rhs_filename`` to be
   correct. Otherwise, adjust the relative paths for these entries accordingly.

.. _Example2:

Example 2: Parallel run with full AMG configuration
---------------------------------------------------

In this example, we solve the same problem as in the previous example, but partitioned for
`4` processes. We also showcase all available input options for `PCG` and `AMG` in the
configuration file.

1. Prepare your linear system files.

2. Use the YAML configuration file ``ex2.yml``:

.. literalinclude:: ../../examples/ex2.yml
   :language: yaml

3. Run `hypredrive` with the configuration file:

.. code-block:: bash

    mpirun -np 4 ./hypredrive ex2.yml

4. Your output should look like:

.. literalinclude:: ../../examples/refOutput/ex2.txt
   :language: text

.. _Example3:

Example 3: Minimal multigrid reduction strategy
-----------------------------------------------

In this example, we solve a linear system derived from the discretization of a
compositional flow problem from `GEOS <https://github.com/GEOS-DEV/GEOS>`_. Details about
how this linear system was generated can be found at ``data/compflow6k/README.md``. This
example uses a `MGR-GMRES` solver and showcases the minimal configuration for setting up
the multigrid reduction preconditioner for this particular kind of linear system.

1. Prepare your linear system files.

2. Use the YAML configuration file ``ex3.yml``:

.. literalinclude:: ../../examples/ex3.yml
   :language: yaml

3. Run `hypredrive` with the configuration file:

.. code-block:: bash

    mpirun -np 1 ./hypredrive ex3.yml

4. Your output should look like:

.. literalinclude:: ../../examples/refOutput/ex3.txt
   :language: text

Example 4: Advanced multigrid reduction strategy
------------------------------------------------

In this example, we solve the same problem as before, but partitioned for 4
processes. Here, we showcase a more advanced setup of `MGR` involving multiple options.

1. Prepare your linear system files.

2. Use the YAML configuration file ``ex4.yml``:

.. literalinclude:: ../../examples/ex4.yml
   :language: yaml

3. Run `hypredrive` with the configuration file:

.. code-block:: bash

    mpirun -np 4 ./hypredrive ex4.yml

4. Your output should look like:

.. literalinclude:: ../../examples/refOutput/ex4.txt
   :language: text
