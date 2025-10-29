.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Introduction:

Introduction
============

What is `hypredrive`?
---------------------

`hypredrive` is a high-level interface driver for `hypre`, a package for solving sparse
linear systems of equations. It's designed to make the process of setting up linear
solvers in `hypre` easier via input files in YAML format. Whether you are a researcher or
an application library developer, `hypredrive` offers an easy way with low overhead to
test or access the linear solvers provided by `hypre`.

Key Features
------------

- **Encapsulation** hides the complexity of the Hypre library by offering an intuitive
  interface driven by YAML configurations. This allows for straightforward and
  error-resistant setup, enabling easy adjustments and sharing of solver settings.

- **Prototyping** with a variety of solver options. Users can effortlessly compare solver
  performance and adjust parameters through the YAML configuration, fostering
  experimentation and optimal solver strategy identification.

- **Testing** through an integrated framework to evaluate solvers against a diverse set of
  predefined linear system problems. Thus ensuring that future hypre developments do not
  negatively impact solver convergence and performance for these problems.

Getting Started
---------------

To get started with `hypredrive`, you should first ensure that the software is properly
installed on your system. For detailed installation instructions, please refer to the
:ref:`Installation` section.

Once the installation is complete, familiarize yourself with the the input file structure
for `hypredrive` by reading through the :ref:`InputFileStructure` section. This will
provide you with a good understanding of how to configure and run `hypredrive` for your
specific needs.

Here's an example command to run `hypredrive` on a single process with a basic configuration file:

.. code-block:: bash

    $ mpirun -np 1 ./hypredrive input.yml

In this command, ``input.yml`` should be replaced with the path to your actual configuration
file. You can find input file examples and detailed explanations in the :ref:`Examples` section.

.. _Contributing:

Contributing
------------

We welcome contributions from the community and are pleased that you're interested in helping improve `hypredrive`! This document provides guidelines and information on how you can contribute.

Ways to Contribute
^^^^^^^^^^^^^^^^^^

There are many ways to contribute to `hypredrive`:

- **Reporting Bugs:** If you encounter issues or bugs, please report them by opening an
  issue on our `GitHub issues page
  <https://github.com/hypre-space/hypredrive/issues>`_. Please provide as much detail as
  possible to help us understand and address the issue.

- **Feature Requests:** Are you a developer with ideas for new features or improvements?
  Feel free to submit them as issues, labeling them as feature requests.

- **Submitting Patches:** If you've fixed a bug or implemented a feature, you can submit a
  pull request. Make sure your code adheres to the project's coding standards and include
  tests if possible.

If you plan to submit a pull request, before you start, it's a good idea to get familiar
with the following:

- **Project Structure:** Understand how the project is organized.

- **Coding Standards:** Follow the coding style and guidelines of the project to ensure
  consistency.

- **Testing:** Write and run tests to make sure your changes don't introduce new issues.

Submitting a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Fork the Repository:** Start by forking the repository on GitHub.
2. **Clone Your Fork:** Clone your fork locally and create a new branch for your contribution.
3. **Make Your Changes:** Implement your fix or feature.
4. **Test Your Changes:** Ensure your changes pass all tests and don't introduce new issues.
5. **Commit Your Changes:** Commit your changes with a clear, descriptive commit message.
6. **Push Your Changes:** Push your changes to your fork on GitHub.
7. **Submit a Pull Request:** Open a pull request from your fork to the main `hypredrive`
   repository.

Code Review Process
^^^^^^^^^^^^^^^^^^^

After you submit a pull request, the project maintainers will review your changes. During
the review process:

- Be open to feedback and willing to make revisions.
- Discuss any suggestions or issues that reviewers bring up.
- Once your pull request is approved, a maintainer will merge it into the project.
