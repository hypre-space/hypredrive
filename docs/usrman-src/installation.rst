.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Installation:

Installation
============

Installing `hypredrive` is straightforward. Follow these steps to get it up and running on your system.

Prerequisites
-------------

Before installing `hypredrive`, ensure you have the following prerequisites installed:

- `m4 <https://www.gnu.org/software/m4/>`_: GNU package for expanding and processing
  macros. Minimum version: `1.4.6`.
- `Autoconf <https://www.gnu.org/software/autoconf/>`_: GNU package for generating
  portable configure scripts. Minimum version: `2.69`.
- `Automake <https://www.gnu.org/software/automake/>`_: GNU package for generating
  portable Makefiles. Minimum version: `1.11.3`.
- `libtool <https://www.gnu.org/software/libtool/>`_: GNU package for creating portable
  compiled libraries. Minimum version: `2.4.2`.
- `hypre <https://github.com/hypre-space/hypre>`_: high-performance preconditioners
  library. Minimum version: `2.31.0`.

.. note::
   The GNU packages (``m4``, ``autoconf``, ``automake``, and ``libtool``) are generally
   pre-installed in Unix distributions. If they are not present, they can be easily
   installed via package managers such as ``apt``, ``yum``, ``pacman``, ``homebrew`` or
   ``port`` or ``spack``.

.. note::
   On MacOS, ``libtool`` is a native binary tool used for creating static libraries and
   isn't related to GNU Libtool. To avoid conflict with MacOS's native libtool, the GNU
   Libtool is typically installed as ``glibtool`` when using package managers like
   ``homebrew`` or ``port``.


Installing `hypredrive`
-----------------------

Users can install `hypredrive` by compiling from source, according to the steps bellow:

1. Download `hypredrive`'s source code. This can be accomplished via ``git``:

    .. code-block:: bash

        $ git clone https://github.com/hypre-space/hypredrive.git

   Another option, which does not download the full repository history, is to use ``wget``:

    .. code-block:: bash

        $ wget https://github.com/hypre-space/hypredrive/archive/refs/heads/master.zip
        $ unzip master.zip
        $ rm master.zip
        $ mv hypredrive-master hypredrive

2. Navigate to the cloned directory and run ``autoreconf -i``:

    .. code-block:: bash

        $ cd hypredrive
        $ autoreconf -i

3. Run the configure script while informing where the `hypre` library and include files can
   be found:

    .. code-block:: bash

        $ ./configure --prefix=${HYPREDRIVE_INSTALL_DIR} --with-hypre-dir=${HYPRE_INSTALL_DIR}

   Replace ``${HYPREDRIVE_INSTALL_DIR}`` with your desired installation path for `hypredrive`,
   and ``${HYPRE_INSTALL_DIR}`` with the path to your installation of `hypre`.

   For GPU support, add `--with-cuda` in the case of NVIDIA GPUs or `--with-hip` in the
   case of AMD GPUs to the `./configure` line.

4. Run ``make``:

    .. code-block:: bash

        $ make -j
        $ make install

Verifying the Installation
--------------------------

After installation, you can verify that `hypredrive` is installed correctly by running:

.. code-block:: bash

    $ make check

You should see the output below:

.. code-block:: bash

    "Running with 1 MPI process... passed!"
    "Running with 4 MPI processes... passed!"


Troubleshooting
---------------

If you encounter any issues during the installation of `hypredrive`, please open a
`GitHub issue <https://github.com/hypre-space/hypredrive/issues>`_ and include a copy of the
``config.log`` file, which is generated after running the ``configure`` script.
