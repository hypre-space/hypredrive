.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Installation:

Installation
============

Installing `hypredrive` is straightforward. This page documents the current build
and installation behavior for both supported build systems, with CMake as the
recommended path.

Prerequisites
-------------

Before installing `hypredrive`, ensure you have the following prerequisites installed:

- `CMake <https://cmake.org/>`_: Cross-platform build system. Minimum version: `3.23`.
- `hypre <https://github.com/hypre-space/hypre>`_: high-performance preconditioners
  library. Minimum version: `2.20.0` when using a pre-installed copy.

.. note::
   CMake is the **preferred** and recommended build system for `hypredrive`. It provides
   better integration with modern development tools and more straightforward dependency
   management.

.. note::
   When HYPRE is not already installed, the CMake build can fetch and build it
   automatically. By default, the fetched HYPRE revision is ``master``.

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

   **Option A: Use an installed HYPRE via ``find_package``**

   If HYPRE provides a CMake package configuration, point CMake to the installation:

   .. code-block:: bash

       $ cd hypredrive
       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
               -DHYPRE_ROOT=${HYPRE_INSTALL_DIR} \
               -B build -S .

   Replace ``${HYPREDRIVE_INSTALL_DIR}`` with your desired installation path for
   `hypredrive`, and ``${HYPRE_INSTALL_DIR}`` with the path to your installation of
   `hypre`.

   **Option B: Use user-provided HYPRE include and library paths**

   This is useful when HYPRE was installed by Autotools or otherwise does not provide a
   CMake package configuration:

   .. code-block:: bash

       $ cd hypredrive
       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
               -DHYPRE_INCLUDE_DIRS=${HYPRE_INSTALL_DIR}/include \
               -DHYPRE_LIBRARIES=${HYPRE_INSTALL_DIR}/lib/libHYPRE.so \
               -B build -S .

   ``HYPRE_LIBRARY`` may be used instead of ``HYPRE_LIBRARIES`` when a single library file
   is sufficient.

   **Option C: Automatic HYPRE build**

   If HYPRE is not found, `hypredrive` can automatically download and build HYPRE from
   source. The default fetched revision is ``master``:

   .. code-block:: bash

       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
               -B build -S .

   To use a specific HYPRE release, branch, or tag:

   .. code-block:: bash

       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DHYPRE_VERSION=v3.0.0 \
               -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
               -B build -S .

   **Core configuration options**

   - ``-DCMAKE_BUILD_TYPE=<type>``: Build type. Default: ``Release``.
     Common values are ``Debug``, ``Release``, ``RelWithDebInfo``, and ``MinSizeRel``.
   - ``-DCMAKE_INSTALL_PREFIX=<path>``: Installation prefix.
   - ``-DBUILD_SHARED_LIBS=ON``: Build shared libraries instead of static libraries.
     Default: ``OFF``.

   **Dependency selection options**

   - ``-DHYPRE_ROOT=<path>``: Root of an installed HYPRE tree with a CMake package.
   - ``-DHYPRE_INCLUDE_DIRS=<path>``: HYPRE include directory for custom or Autotools
     installs.
   - ``-DHYPRE_LIBRARIES=<path>`` or ``-DHYPRE_LIBRARY=<path>``: HYPRE library path for
     custom or Autotools installs.
   - ``-DHYPRE_VERSION=<version>``: HYPRE revision to fetch when HYPRE must be built
     automatically. Default: ``master``.
   - ``-DCALIPER_VERSION=<version>``: Caliper revision to fetch when
     ``HYPREDRV_ENABLE_CALIPER=ON`` and Caliper is not already available. Default:
     ``master``.
   - ``CALIPER_DIR`` or ``CALIPER_ROOT``: Search hints for a pre-installed Caliper build.

   **hypredrive feature options**

   - ``-DHYPREDRV_ENABLE_EIGSPEC=ON``: Enable eigenspectrum support. Requires LAPACK.
     Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_TESTING=ON``: Enable the CTest-based unit and integration tests.
     Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_COVERAGE=ON``: Enable coverage instrumentation and add the
     ``coverage`` target. Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_ANALYSIS=ON``: Enable static-analysis integration, including
     sanitizers and optional ``clang-tidy`` / ``cppcheck`` targets when tools are present.
     Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_DATA=ON``: Enable dataset download targets such as ``data``.
     Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_DOCS=ON``: Enable documentation targets such as ``docs``,
     ``sphinx-doc``, and ``sphinx-latexpdf`` when the required tools are available.
     Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_EXAMPLES=ON``: Build the standalone example programs under
     ``examples/src``. Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_HWLOC=ON``: Enable optional hwloc-based topology reporting.
     Default: ``OFF``.
   - ``-DHYPREDRV_ENABLE_CALIPER=ON``: Enable Caliper instrumentation support. Default:
     ``OFF``.
   - ``-DHYPREDRV_ENABLE_COMPRESSION=ON``: Enable optional lossless compression backends
     and build ``hypredrive-lsseq`` when supported. Default: ``OFF``.

   **Option interactions**

   - ``HYPREDRV_ENABLE_TESTING=ON`` automatically sets ``HYPREDRV_ENABLE_DATA=ON``.
   - ``HYPREDRV_ENABLE_COVERAGE=ON`` automatically sets
     ``HYPREDRV_ENABLE_TESTING=ON``, ``HYPREDRV_ENABLE_EXAMPLES=ON``, and
     ``HYPREDRV_ENABLE_DATA=ON``.
   - The ``check`` target is always available as a smoke test for ``hypredrive-cli``,
     even when ``HYPREDRV_ENABLE_TESTING=OFF``.

   **Optional dependency behavior**

   - ``HYPREDRV_ENABLE_HWLOC=ON`` searches for `hwloc <https://www.open-mpi.org/projects/hwloc/>`_.
   - ``HYPREDRV_ENABLE_CALIPER=ON`` tries ``find_package(caliper)`` first and otherwise
     fetches Caliper automatically.
   - Caliper auto-fetch/build is not supported with the Ninja generator. Use the default
     Makefile generator or provide a pre-built Caliper installation instead.
   - ``HYPREDRV_ENABLE_COMPRESSION=ON`` probes for zlib, zstd, lz4, and blosc. Available
     backends are enabled automatically based on what is found at configure time.

   **Forwarding HYPRE build options**

   When HYPRE is built automatically, `hypredrive` forwards common HYPRE and TPL
   configuration variables to the HYPRE build instead of maintaining a duplicate option
   list. In practice, this includes:

   - Variables beginning with ``HYPRE_ENABLE_`` such as ``HYPRE_ENABLE_CUDA``,
     ``HYPRE_ENABLE_HIP``, ``HYPRE_ENABLE_CALIPER``, or ``HYPRE_ENABLE_MIXEDINT``.
   - Variables ending with ``_ROOT`` or ``_DIR`` such as ``CUDA_ROOT``, ``ROCM_PATH``,
     ``MAGMA_ROOT``, or ``CALIPER_DIR``.
   - Standard ``CMAKE_*`` variables such as compilers, build type, and install prefix.

   Examples:

   .. code-block:: bash

       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DHYPRE_ENABLE_CUDA=ON \
               -DCMAKE_CUDA_ARCHITECTURES=80 \
               -B build -S .

   .. code-block:: bash

       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DHYPRE_ENABLE_HIP=ON \
               -DROCM_PATH=/opt/rocm \
               -DCMAKE_HIP_ARCHITECTURES=gfx90a \
               -B build -S .

   .. code-block:: bash

       $ cmake -DCMAKE_BUILD_TYPE=Release \
               -DHYPREDRV_ENABLE_CALIPER=ON \
               -DCALIPER_DIR=${CALIPER_INSTALL_DIR} \
               -B build -S .

3. Build and install:

    .. code-block:: bash

        $ cmake --build build --parallel
        $ cmake --install build

4. (Optional) Download example datasets:

   If ``HYPREDRV_ENABLE_DATA=ON`` (either explicitly or implied by another option), you can
   fetch the example datasets with:

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
   be found. The Autotools build produces static libraries by default.

    .. code-block:: bash

        $ ./configure --prefix=${HYPREDRIVE_INSTALL_DIR} --with-hypre-dir=${HYPRE_INSTALL_DIR}

   Replace ``${HYPREDRIVE_INSTALL_DIR}`` with your desired installation path for `hypredrive`,
   and ``${HYPRE_INSTALL_DIR}`` with the path to your installation of `hypre`.

   If ``--with-hypre-dir`` is not used, both of the following are required:

   - ``--with-hypre-include=${HYPRE_INCLUDE_DIR}``
   - ``--with-hypre-lib=${HYPRE_LIBRARY_DIR}``

   Common Autotools options:

   - ``--with-cuda``: Enable CUDA support.
   - ``--with-cuda-home=${CUDA_HOME}``: Set the CUDA installation root.
   - ``--with-hip``: Enable HIP support.
   - ``--with-rocm-path=${ROCM_PATH}``: Set the ROCm installation root.
   - ``--enable-doxygen``: Enable Doxygen documentation generation. Default: ``no``.

   ``--with-cuda`` and ``--with-hip`` are mutually exclusive.

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

The ``check`` target runs a smoke test. To enable the full CTest-based unit and integration
suite, configure with ``-DHYPREDRV_ENABLE_TESTING=ON`` and run:

.. code-block:: bash

    $ ctest --test-dir build --output-on-failure

**For Autotools builds:**

.. code-block:: bash

    $ make check

For a successful smoke test, you should see output similar to the following:

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
