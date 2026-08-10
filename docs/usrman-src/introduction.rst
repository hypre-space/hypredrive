.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Introduction:

Introduction
============

What is `hypredrive`?
---------------------

`hypredrive` is a high-level interface for `hypre`, which solves sparse linear systems.
A YAML file or string configures the `hypre` solvers. Researchers and application
developers can use `hypredrive` with a small integration cost.

Key Features
------------

- **Encapsulation.** A YAML interface controls `hypre` and reduces direct solver setup.
  You can change and share solver settings without code changes.

- **Prototyping.** You can compare solver options and adjust their parameters in the YAML
  configuration.

- **Testing.** The integrated test framework evaluates solvers against defined linear
  systems. These tests detect changes in solver convergence and performance.

Getting Started
---------------

1. Install `hypredrive` with the instructions in :ref:`Installation`.

2. Read :ref:`InputFileStructure` to learn the YAML configuration.

3. Run ``hypredrive-cli`` on one process with a basic configuration file:

.. code-block:: bash

    $ mpirun -np 1 ./hypredrive-cli input.yml

Replace ``input.yml`` with the path of your configuration file. See
:ref:`DriverExamples` for input files and explanations.

.. _Contributing:

Contributing
------------

The project accepts community contributions. This section explains how you can contribute
to `hypredrive`.

Ways to Contribute
^^^^^^^^^^^^^^^^^^

You can contribute to `hypredrive` in these ways:

- **Report bugs.** Open an issue on the `GitHub issues page
  <https://github.com/hypre-space/hypredrive/issues>`_. Include the configuration, system
  information, error messages, and steps that reproduce the problem.

- **Request features.** Open an issue and identify it as a feature request.

- **Submit patches.** Open a pull request for a bug fix or feature. Follow the project
  coding standards and include applicable tests.

Before you submit a pull request, review these project areas:

- **Project structure.** Review the project layout.

- **Coding standards.** Follow the project style and development guidelines.

- **Testing.** Write and run tests for your changes.

Submitting a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Fork the repository on GitHub.
2. Clone your fork and create a branch for your contribution.
3. Implement your fix or feature.
4. Run the applicable tests and correct all failures.
5. Commit your changes with a clear message.
6. Push your changes to your fork.
7. Open a pull request to the main `hypredrive` repository.

Code Review Process
^^^^^^^^^^^^^^^^^^^

Project maintainers review each pull request. During this review:

- Review the feedback.
- Discuss questions with the reviewers.
- Make the requested revisions.

After approval, a maintainer merges the pull request.
