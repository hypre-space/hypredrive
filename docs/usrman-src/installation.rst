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

- `CMake <https://cmake.org/>`_: Cross-platform build system. Minimum version: `3.23`.
- `hypre <https://github.com/hypre-space/hypre>`_: high-performance preconditioners
  library. Minimum version: `2.31.0`.

.. note::
   CMake is the **preferred** and recommended build system for `hypredrive`. It provides
   better integration with modern development tools and more straightforward dependency
   management.

For Autotools Build (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer using Autotools (``autoconf``, ``automake``), the following additional
prerequisites are required:

- `m4 <https://www.gnu.org/software/m4/>`_: GNU package for expanding and processing
  macros. Minimum version: `1.4.6`.
- `Autoconf <https://www.gnu.org/software/autoconf/>`_: GNU package for generating
  portable configure scripts. Minimum version: `2.69`.
- `Automake <https://www.gnu.org/software/automake/>`_: GNU package for generating
  portable Makefiles. Minimum version: `1.11.3`.
- `libtool <https://www.gnu.org/software/libtool/>`_: GNU package for creating portable
  compiled libraries. Minimum version: `2.4.2`.

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

Users can install `hypredrive` by compiling from source. **CMake is the preferred build
system** and is recommended for all users.

Using CMake (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download `hypredrive`'s source code. This can be accomplished via ``git``:

    .. code-block:: bash

        $ git clone https://github.com/hypre-space/hypredrive.git

   Another option, which does not download the full repository history, is to use ``wget``:

    .. code-block:: bash

        $ wget https://github.com/hypre-space/hypredrive/archive/refs/heads/master.zip
        $ unzip master.zip
        $ rm master.zip
        $ mv hypredrive-master hypredrive

2. Create a build directory and configure:

    .. code-block:: bash

        $ cd hypredrive
        $ cmake -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
                -DHYPRE_ROOT=${HYPRE_INSTALL_DIR} \
                -B build -S .

   Replace ``${HYPREDRIVE_INSTALL_DIR}`` with your desired installation path for `hypredrive`,
   and ``${HYPRE_INSTALL_DIR}`` with the path to your installation of `hypre`.

   **Required Options:**

   - ``-DHYPRE_ROOT=<path>``: Path to the HYPRE installation directory (required).

   **Common Options:**

   - ``-DCMAKE_BUILD_TYPE=<type>``: Set the build type. Options: ``Release`` (default),
     ``Debug``, ``RelWithDebInfo``, ``MinSizeRel``.
   - ``-DCMAKE_INSTALL_PREFIX=<path>``: Installation directory prefix (default: system
     directories).

   **Optional Features:**

   - ``-DHYPREDRV_ENABLE_EIGSPEC=ON``: Enable full eigenspectrum computation support
     (requires LAPACK). Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_HWLOC=ON``: Enable hwloc support for detailed system topology
     information (CPU, GPU, NUMA, cache hierarchy, etc.). Requires the hwloc library to be
     installed. If hwloc is not available, hypredrive will fall back to basic system
     information. Default: ``OFF``.
   - ``-DHYPREDRV_BUILD_EXAMPLES=OFF``: Disable building example programs. Default: ``ON``.
   - ``-DHYPREDRV_ENABLE_TESTING=OFF``: Disable testing support and ``check`` target.
     Default: ``ON``.
   - ``-DHYPREDRV_ENABLE_DATA=OFF``: Disable the ``data`` target for downloading
     example datasets. Default: ``ON``.
   - ``-DBUILD_SHARED_LIBS=ON``: Build shared libraries instead of static libraries.
     Default: ``OFF``.

3. Build and install:

    .. code-block:: bash

        $ cmake --build build --parallel
        $ cmake --install build

4. (Optional) Download example datasets:

    .. code-block:: bash

        $ cmake --build build --target data

   This downloads datasets from Zenodo needed for examples. See :ref:`DriverExamples` for details.

Using Autotools (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer using Autotools, follow these steps:

1. Download `hypredrive`'s source code (same as CMake step 1 above).

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

After installation, you can verify that `hypredrive` is installed correctly:

**For CMake builds:**

.. code-block:: bash

    $ cmake --build build --target check

**For Autotools builds:**

.. code-block:: bash

    $ make check

You should see the output below:

.. code-block:: bash

    Running tests (equivalent to autotools make check)
    Test project /path/to/hypredrive/build
        Start 1: test_ex1_1proc
    1/2 Test #1: test_ex1_1proc ....................   Passed
        Start 2: test_ex2_4proc
    2/2 Test #2: test_ex2_4proc ....................   Passed

    100% tests passed, 2 tests passed out of 2


Troubleshooting
---------------

If you encounter any issues during the installation of `hypredrive`, please open a
`GitHub issue <https://github.com/hypre-space/hypredrive/issues>`_.

For **CMake builds**, include the output from ``cmake`` and ``cmake --build``.

For **Autotools builds**, include a copy of the ``config.log`` file, which is generated after
running the ``configure`` script.
