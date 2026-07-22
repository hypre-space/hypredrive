.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _FAQ:

Frequently Asked Questions (FAQ)
================================

This section answers common questions about `hypredrive`.

What is `hypredrive`?
---------------------

`hypredrive` is a high-level interface for solving linear systems with hypre.

How do I install `hypredrive`?
------------------------------

Download and compile the `hypredrive` source files. See :ref:`Installation` for the
installation procedure.

Which linear system types can `hypredrive` solve?
-------------------------------------------------

`hypredrive` solves symmetric and nonsymmetric sparse linear systems. Available features
depend on the HYPRE build and the `hypredrive` configuration.

How do I configure `hypredrive` for my specific problem?
--------------------------------------------------------

Provide YAML as a file for ``hypredrive-cli`` or as a string for
``HYPREDRV_InputArgsParse``. The YAML contains the solver and preconditioner settings. See
:ref:`InputFileStructure` for the reference. See :ref:`DriverExamples` for driver examples.

How can I contribute to `hypredrive`?
-------------------------------------

You can file issues, submit pull requests, or improve the documentation. See
:ref:`Contributing` for the contribution procedure.

Can I use `hypredrive` on GPU-accelerated systems?
--------------------------------------------------

Yes. Build `hypre` with graphics processing unit (GPU) support. In these builds,
``general.exec_policy`` defaults to ``device``. Set ``host`` only when you require host
execution.

Can I compile `hypredrive` on Windows machines?
-----------------------------------------------

No. The project does not currently plan to support Windows.

How do I debug solver failures?
-------------------------------

Set the environment variable ``HYPREDRV_LOG_LEVEL`` to enable internal traces:

- ``1``: lifecycle boundaries (create, setup, solve, destroy)
- ``2``: decision and context messages (reuse policy, scaling choices)
- ``3``: detailed subphase traces (parsing, linear system assembly, scaling steps)

Rank 0 writes traces to ``stderr``. Set ``HYPREDRV_LOG_STREAM=stdout`` to write them to
``stdout``.

For error trapping, set ``HYPREDRV_DEBUG=1``. Then ``HYPREDRV_SAFE_CALL`` raises
``SIGTRAP`` instead of stopping the process.

What do I do if I encounter an issue with `hypredrive`?
--------------------------------------------------------

Open a `GitHub issue <https://github.com/hypre-space/hypredrive/issues>`_. Include the
configuration files, system information, error messages, and steps that reproduce the
problem.

How is `hypredrive` licensed?
-----------------------------

`hypredrive` uses the MIT License. See the ``LICENSE`` file in the source distribution.
