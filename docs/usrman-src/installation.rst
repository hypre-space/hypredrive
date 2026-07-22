.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Installation:

Installation
============

CMake is the recommended build system for `hypredrive`. It finds dependencies and
configures optional features through one interface. Autotools is also available when an
environment cannot use CMake. See :ref:`AutotoolsInstall`.

Prerequisites
-------------

- `CMake <https://cmake.org/>`_ 3.23 or newer.
- `hypre <https://github.com/hypre-space/hypre>`_ 2.20.0 or newer. Alternatively, CMake
  can fetch and build HYPRE.

.. _CMakeInstall:

Installing with CMake (Recommended)
-------------------------------------

**Step 1: Download the source**

.. code-block:: bash

    $ git clone https://github.com/hypre-space/hypredrive.git
    $ cd hypredrive

Or, to download without full git history:

.. code-block:: bash

    $ wget https://github.com/hypre-space/hypredrive/archive/refs/heads/master.zip
    $ unzip master.zip && rm master.zip
    $ mv hypredrive-master hypredrive
    $ cd hypredrive

**Step 2: Configure**

Choose one of the following options depending on how HYPRE is available:

**Option A — Existing HYPRE installation through** ``find_package``

.. code-block:: bash

    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
            -DHYPRE_ROOT=${HYPRE_INSTALL_DIR} \
            -B build -S .

**Option B — Custom HYPRE paths, such as an Autotools build**

.. code-block:: bash

    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
            -DHYPRE_INCLUDE_DIRS=${HYPRE_INSTALL_DIR}/include \
            -DHYPRE_LIBRARIES=${HYPRE_INSTALL_DIR}/lib/libHYPRE.so \
            -B build -S .

Use ``HYPRE_LIBRARY`` instead of ``HYPRE_LIBRARIES`` for one library file.

**Option C — Automatic HYPRE build (no HYPRE required)**

.. code-block:: bash

    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
            -B build -S .

To pin a specific HYPRE revision:

.. code-block:: bash

    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DHYPRE_VERSION=v3.0.0 \
            -DCMAKE_INSTALL_PREFIX=${HYPREDRIVE_INSTALL_DIR} \
            -B build -S .

**Step 3: Build and install**

.. code-block:: bash

    $ cmake --build build --parallel
    $ cmake --install build

**Step 4 (optional): Download example datasets**

.. code-block:: bash

    $ cmake --build build --target data

This fetches datasets from Zenodo needed for the :ref:`DriverExamples`. It runs
automatically when ``HYPREDRV_ENABLE_TESTING=ON`` is set.

CMake options reference
~~~~~~~~~~~~~~~~~~~~~~~

*Core options*

- ``-DCMAKE_BUILD_TYPE=<type>`` — ``Release`` (default), ``Debug``, ``RelWithDebInfo``, or ``MinSizeRel``.
- ``-DCMAKE_INSTALL_PREFIX=<path>`` — Installation prefix.
- ``-DBUILD_SHARED_LIBS=ON`` — Build shared libraries. Default: ``OFF``.

*HYPRE discovery*

- ``-DHYPRE_ROOT=<path>`` — Root of a CMake-configured HYPRE installation.
- ``-DHYPRE_INCLUDE_DIRS=<path>`` — HYPRE include directory (non-CMake installs).
- ``-DHYPRE_LIBRARIES=<path>`` / ``-DHYPRE_LIBRARY=<path>`` — HYPRE library path.
- ``-DHYPRE_VERSION=<version>`` — HYPRE revision to fetch when building automatically. Default: ``master``.

*Feature options*

- ``-DHYPREDRV_ENABLE_EIGSPEC=ON`` — Eigenspectrum support (requires LAPACK). Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_TESTING=ON`` — CTest-based unit and integration tests. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_COVERAGE=ON`` — Coverage instrumentation. This option also enables testing, examples, and data. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_ANALYSIS=ON`` — Sanitizers and optional ``clang-tidy`` / ``cppcheck`` targets. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_DATA=ON`` — Dataset download targets (implied by testing). Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_DOCS=ON`` — Documentation targets (``docs``, ``sphinx-doc``, ``sphinx-latexpdf``). Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_EXAMPLES=ON`` — Standalone example programs under ``examples/src``. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_HWLOC=ON`` — hwloc-based topology reporting. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_CALIPER=ON`` — Caliper instrumentation. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_COMPRESSION=ON`` — Lossless compression backends and the ``hypredrive-lsseq`` utility. Default: ``OFF``.
- ``-DHYPREDRV_ENABLE_PYTHON=ON`` — Build the Python extension as part of the top-level CMake build. Default: ``OFF``.

*Option interactions*

- ``HYPREDRV_ENABLE_TESTING=ON`` implies ``HYPREDRV_ENABLE_DATA=ON``.
- ``HYPREDRV_ENABLE_COVERAGE=ON`` implies testing, examples, and data.
- The ``check`` smoke-test target is always available regardless of ``HYPREDRV_ENABLE_TESTING``.

*GPU builds (forwarded to the automatic HYPRE build)*

When CMake builds HYPRE automatically, it forwards ``HYPRE_ENABLE_*``, ``*_ROOT``, ``*_DIR``,
and standard ``CMAKE_*`` variables to the HYPRE build. Examples:

.. code-block:: bash

    # CUDA
    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DHYPRE_ENABLE_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=80 \
            -B build -S .

.. code-block:: bash

    # HIP / ROCm
    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DHYPRE_ENABLE_HIP=ON \
            -DROCM_PATH=/opt/rocm \
            -DCMAKE_HIP_ARCHITECTURES=gfx90a \
            -B build -S .

*Caliper (optional performance profiling)*

- ``-DCALIPER_VERSION=<version>`` — Caliper revision to fetch. Default: ``master``.
- ``CALIPER_DIR`` / ``CALIPER_ROOT`` — Search hints for an existing Caliper installation.

Caliper automatic retrieval does not support the Ninja generator. Use the
default Makefile generator or an existing Caliper installation.

*SuperLU_DIST (optional HYPRE DSUPERLU support)*

- ``-DHYPRE_ENABLE_DSUPERLU=ON`` — Enable HYPRE's SuperLU_DIST integration.
- ``-DSUPERLU_DIST_VERSION=<version>`` — SuperLU_DIST revision to fetch if
  ``superlu_dist`` is not found. Default: ``v9.2.1``.
- ``DSUPERLU_DIR`` / ``DSUPERLU_ROOT`` / ``SUPERLU_DIST_ROOT`` — Search hints for an
  existing SuperLU_DIST installation.

*Compression backends*

``HYPREDRV_ENABLE_COMPRESSION=ON`` probes for zlib, zstd, lz4, and blosc. CMake
enables each available backend during configuration.

*Python interface*

The Python package lives under ``interfaces/python``. The standard package workflow is
``pip install ./interfaces/python`` against either an installed hypredrive or an in-tree
development build. The top-level CMake option ``HYPREDRV_ENABLE_PYTHON=ON`` is also
available for developer and CI builds that want the native Python extension built with
the C library. See :ref:`PythonInterface` for full installation and usage details.

Verifying the Installation
--------------------------

Run the smoke test:

.. code-block:: bash

    $ cmake --build build --target check

For the full CTest-based suite (requires ``-DHYPREDRV_ENABLE_TESTING=ON``):

.. code-block:: bash

    $ ctest --test-dir build --output-on-failure

A passing smoke test looks like:

.. code-block:: text

    Running tests (equivalent to autotools make check)
    Test project /path/to/hypredrive/build
        Start 1: test_ex1_1proc
    1/2 Test #1: test_ex1_1proc ....................   Passed
        Start 2: test_ex2_4proc
    2/2 Test #2: test_ex2_4proc ....................   Passed

    100% tests passed, 2 tests passed out of 2

Troubleshooting
---------------

If you encounter any issues, open a `GitHub issue
<https://github.com/hypre-space/hypredrive/issues>`_ and include the output from ``cmake``
and ``cmake --build``.

.. _AutotoolsInstall:

Autotools (Alternative)
------------------------

.. note::
   Autotools supports environments without CMake. CMake is the standard build path
   for all other environments.

*Prerequisites*

In addition to HYPRE, the following GNU tools are required:

- `m4 <https://www.gnu.org/software/m4/>`_ 1.4.6+
- `Autoconf <https://www.gnu.org/software/autoconf/>`_ 2.69+
- `Automake <https://www.gnu.org/software/automake/>`_ 1.11.3+
- `libtool <https://www.gnu.org/software/libtool/>`_ 2.4.2+

Linux systems usually include these tools. On macOS, Homebrew and MacPorts install GNU
libtool as ``glibtool`` to prevent a conflict with the native ``libtool`` binary.

*Build steps*

1. Download the source (same as CMake step 1).

2. Generate the configure script:

   .. code-block:: bash

       $ cd hypredrive
       $ autoreconf -i

3. Configure:

   .. code-block:: bash

       $ ./configure --prefix=${HYPREDRIVE_INSTALL_DIR} \
                     --with-hypre-dir=${HYPRE_INSTALL_DIR}

   If you do not use ``--with-hypre-dir``, provide both:

   - ``--with-hypre-include=${HYPRE_INCLUDE_DIR}``
   - ``--with-hypre-lib=${HYPRE_LIBRARY_DIR}``

   Common options:

   - ``--with-cuda`` / ``--with-cuda-home=${CUDA_HOME}`` — Enable CUDA.
   - ``--with-hip`` / ``--with-rocm-path=${ROCM_PATH}`` — Enable HIP.
   - ``--enable-doxygen`` — Enable Doxygen generation. Default: ``no``.

   ``--with-cuda`` and ``--with-hip`` are mutually exclusive.

4. Build and install:

   .. code-block:: bash

       $ make -j
       $ make install

5. Verify:

   .. code-block:: bash

       $ make check
