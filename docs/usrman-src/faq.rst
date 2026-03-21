.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _FAQ:

Frequently Asked Questions (FAQ)
================================

This section provides answers to some of the most commonly asked questions about `hypredrive`.

What is `hypredrive`?
---------------------

`hypredrive` is a high-level interface for solving linear systems with hypre.

How do I install `hypredrive`?
------------------------------

You can install `hypredrive` by downloading and compiling its source files. Please refer to
:ref:`Installation` for detailed installation instructions.

Which linear system types can `hypredrive` solve?
-------------------------------------------------

`hypredrive` is capable of solving both symmetric and non-symmetric sparse linear
systems. The specific capabilities depend on the underlying HYPRE library and the
configuration of `hypredrive`.

How do I configure `hypredrive` for my specific problem?
--------------------------------------------------------

You can configure `hypredrive` by providing YAML — either as a file on disk (for the
``hypredrive-cli`` driver) or as an in-memory string passed to ``HYPREDRV_InputArgsParse``
from application code. The YAML specifies all necessary settings, including solver and
preconditioner configurations. For the full reference, see :ref:`InputFileStructure`. For
driver usage examples, see :ref:`DriverExamples`.

How can I contribute to `hypredrive`?
-------------------------------------

Contributions to `hypredrive` are welcome! You can contribute by filing issues, submitting
pull requests or improving its documentation. Please refer to :ref:`Contributing` for
guidelines.

Can I use `hypredrive` on GPU-accelerated systems?
--------------------------------------------------

Yes, `hypredrive` supports GPU acceleration. Note that `hypre` also needs to be compiled
with GPU support. On GPU-enabled builds, ``general.exec_policy`` defaults to ``device``;
set it explicitly only if you want to force host execution.

Can I compile `hypredrive` on Windows machines?
-----------------------------------------------

No. At the moment, there are no plans to add Windows support.

What should I do if I encounter an issue with `hypredrive`?
-----------------------------------------------------------

If you encounter an issue, you can open a `GitHub issue
<https://github.com/hypre-space/hypredrive/issues>`_. Providing detailed information about
your problem, including configuration files, system details, and error messages, will help
in resolving issues more quickly.

How is `hypredrive` licensed?
-----------------------------

`hypredrive` is licensed under the MIT License. For more details, see the LICENSE file in
the source distribution.
