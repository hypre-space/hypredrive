.. _developer-notes:

Developer Notes
===============

This chapter collects practical guidance for contributing to hypredrive with a focus on:

- Continuous Integration (CI)
- Static code analysis (cppcheck, clang-tidy)
- Code coverage (gcov/gcovr, CTest)

It explains how the CI is structured, how to reproduce checks locally, and what
options and targets are available in CMake to enable these workflows. New
contributors should read this once before opening their first PR.


CI Overview
-----------

Workflows
~~~~~~~~~

Hypredrive uses GitHub Actions with the following workflows (see ``.github/workflows``):

- ``ci.yml``: main build-and-test matrix (Ubuntu + macOS; compilers: gcc/clang; build type: Debug)
- ``format.yml``: clang-format style checks for ``include/``, ``src/``, ``examples/src/``
- ``docs.yml``: builds documentation (Sphinx and Doxygen jobs)
- ``coverage.yml``: builds with coverage instrumentation and generates HTML/XML reports
- ``analysis.yml``: code analysis (static: cppcheck and clang-tidy; dynamic: ASan/UBSan with gcc/clang)

To reduce duplication, CI uses composite actions in ``.github/actions``:

- ``setup-ubuntu``: installs compilers, MPI, and tools
- ``build-hypre``: builds and caches HYPRE from source and exposes its install prefix via the
  ``hypre_prefix`` output

HYPRE reuse and caching
~~~~~~~~~~~~~~~~~~~~~~~

All jobs that need HYPRE call the ``build-hypre`` action and pass:

- ``hypre_version`` (default: ``master``)
- ``compiler`` (``gcc`` or ``clang``)
- ``build_type`` (``Debug``)
- ``hypre_shared_libs`` (``ON``)
- ``os`` (``ubuntu-latest`` or ``macos-latest``)

The action caches HYPRE under ``${{ runner.tool_cache }}/hypre/<version>`` with a descriptive key
that includes OS, compiler, build type, and whether libraries are shared (``shared``/``static``).

Automatic HYPRE build
~~~~~~~~~~~~~~~~~~~~~

When building locally, *hypredrive* can automatically fetch and build HYPRE from source using
CMake's `FetchContent` if ``HYPRE_ROOT`` is not specified. This feature:

- Automatically downloads HYPRE from GitHub (default: ``master`` branch)
- Inherits all CMake arguments from the parent project (build type, compilers, TPLs, etc.)
- Builds HYPRE in the same build tree with unified library and include directories
- Supports specifying HYPRE version via ``-DHYPRE_VERSION=<version>``

To use a pre-built HYPRE instead, specify ``-DHYPRE_ROOT=<path>``.

MPI configuration
~~~~~~~~~~~~~~~~~

- MPI implementation: MPICH (avoids oversubscription pitfalls common with OpenMPI in CI).
- C compiler for CMake: ``-DCMAKE_C_COMPILER=mpicc``.
- On Ubuntu, the action adds ``-DMPI_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/mpich``.

Local reproduction of CI
~~~~~~~~~~~~~~~~~~~~~~~~

On Ubuntu (gcc example):

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y cmake ninja-build mpich libmpich-dev ccache clang-format gcc
   # Build HYPRE once and set CMAKE_PREFIX_PATH
   git clone --depth 1 --branch master https://github.com/hypre-space/hypre.git
   cmake -S hypre/src -B hypre/build -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON \
     -DHYPRE_BUILD_TESTS=OFF -DHYPRE_BUILD_EXAMPLES=OFF \
     -DCMAKE_C_COMPILER=mpicc -DMPI_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/mpich \
     -DCMAKE_INSTALL_PREFIX=$HOME/.local/hypre/master
   cmake --build hypre/build --parallel
   cmake --install hypre/build

   export HYPRE_PREFIX=$HOME/.local/hypre/master
   cmake -S . -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug \
     -DHYPREDRV_ENABLE_TESTING=ON -DHYPREDRV_ENABLE_EXAMPLES=ON \
     -DHYPRE_ROOT=$HYPRE_PREFIX \
     -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_PREFIX_PATH=${HYPRE_PREFIX}
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure

On macOS (Apple clang):

.. code-block:: bash

   brew update && brew install cmake ninja mpich hypre clang-format
   cmake -S . -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug \
     -DHYPREDRV_ENABLE_TESTING=ON -DHYPREDRV_BUILD_EXAMPLES=ON \
     -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_PREFIX_PATH=$(brew --prefix hypre)
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure


Code Analysis
-------------

Overview
~~~~~~~~

Code analysis runs in ``analysis.yml`` and includes both static and dynamic analysis:

Static analysis:
- ``cppcheck`` (C99 rules, exhaustive checks)
- ``clang-tidy`` (driven by ``compile_commands.json``)

Dynamic analysis:
- ``sanitizers`` (AddressSanitizer and UndefinedBehaviorSanitizer with gcc/clang)

CMake options and targets
~~~~~~~~~~~~~~~~~~~~~~~~~

- Enable analysis flags and targets: ``-DHYPREDRV_ENABLE_ANALYSIS=ON``.
- Configure with export of compile commands for clang-tidy: ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``.
- Static analysis targets:
  - ``cppcheck``: runs analysis on ``src/`` only, with includes to ``include/`` and the build dir.
  - ``clang-tidy``: runs clang-tidy (non-fix) across the project using the compile database.
- Dynamic analysis: When ``HYPREDRV_ENABLE_ANALYSIS=ON``, sanitizers (ASan/UBSan) are automatically
  enabled for all targets. Run tests with ``ctest`` to exercise the sanitizers.

cppcheck configuration
~~~~~~~~~~~~~~~~~~~~~~

- Scope narrowed to ``src/`` to keep signal high and run time reasonable.
- HYPRE includes are added so macros/types resolve.
- Threads: cppcheck uses the same number as CMake parallel level when available.
- Known suppressions: irrelevant warnings for mixed precision or specific HYPRE headers can be
  ignored (e.g., ``_hypre_IJ_mv.h``, ``_hypre_utilities.h``).

clang-tidy usage
~~~~~~~~~~~~~~~~

- Use the ``clang-tidy`` target for read-only reports.
- Avoid running aggressive automatic fixes over the entire tree; they can mangle identifiers in
  macro-heavy code. If you must use ``clang-tidy -fix``, constrain to specific files and review
  diffs carefully. Prefer incremental, human-reviewed edits.
- If you add checks, ensure they don't fight our established style or C idioms in this project.

Local runs
~~~~~~~~~~

Static analysis:

.. code-block:: bash

   cmake -S . -B build-analysis -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug \
     -DHYPREDRV_ENABLE_ANALYSIS=ON \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
     -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_PREFIX_PATH=${HYPRE_PREFIX}
   cmake --build build-analysis --parallel
   # cppcheck
   cmake --build build-analysis --target cppcheck
   # clang-tidy
   cmake --build build-analysis --target clang-tidy

Dynamic analysis (sanitizers):

.. code-block:: bash

   cmake -S . -B build-sanitizers -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug \
     -DHYPREDRV_ENABLE_TESTING=ON -DHYPREDRV_ENABLE_EXAMPLES=ON \
     -DHYPREDRV_ENABLE_ANALYSIS=ON \
     -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_PREFIX_PATH=${HYPRE_PREFIX}
   cmake --build build-sanitizers --parallel
   cmake --build build-sanitizers --target data
   # Run tests with sanitizers enabled
   export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:print_stacktrace=1"
   export UBSAN_OPTIONS="print_stacktrace=1:abort_on_error=1"
   ctest --test-dir build-sanitizers --output-on-failure


Code Coverage
-------------

Overview
~~~~~~~~

Coverage is generated in ``coverage.yml`` using ``gcovr`` (HTML and XML artifacts). The build uses
``-O0 -g --coverage`` and runs unit tests via CTest. We also expose a simple percentage summary in
the job output.

CMake options and targets
~~~~~~~~~~~~~~~~~~~~~~~~~

- Enable instrumentation: ``-DHYPREDRV_ENABLE_COVERAGE=ON`` (Debug builds).
- Build normally, then run: ``ctest`` and ``cmake --build <build> --target coverage``.
- Artifacts: ``coverage.html`` and ``coverage.xml`` in the build directory.

Local reproduction
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install gcovr
   cmake -S . -B build-coverage -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug \
     -DHYPREDRV_ENABLE_TESTING=ON -DHYPREDRV_BUILD_EXAMPLES=ON \
     -DHYPREDRV_ENABLE_COVERAGE=ON \
     -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_PREFIX_PATH=${HYPRE_PREFIX}
   cmake --build build-coverage --parallel
   ctest --test-dir build-coverage --output-on-failure
   cmake --build build-coverage --target coverage
   # Open build-coverage/coverage.html

Documentation Builds
--------------------

Two documentation systems are wired:

- Doxygen (developer reference; target: ``doxygen-doc``)
- Sphinx (user manual; targets: ``sphinx-doc``, ``sphinx-latexpdf``)

Enable via CMake: ``-DHYPREDRV_ENABLE_DOCS=ON``. The combined ``docs`` target builds Doxygen first,
then invokes Sphinx (either via ``docs/Makefile`` if available or directly via ``sphinx-build``).

Notes:

- Doxygen LaTeX can build a ``refman.pdf``; when present, CI copies it to
  ``build-docs/docs/devman-hypredrive.pdf``.
- Sphinx PDF (``sphinx-latexpdf``) is copied to ``docs/usrman-hypredrive.pdf`` when available.


Coding Style & Project Conventions
----------------------------------

- ``clang-format`` is enforced in CI; run ``make format`` locally (or the ``format`` CMake target) to
  format ``.c``/``.h`` files under ``include/``, ``src/``, and ``examples/src/``.
- Some macros are guarded with ``// clang-format off``/``on`` to preserve required layout. Avoid
  editing those blocks unless necessary.
- Follow const-correctness and safe API patterns. Error-reporting helpers must be used consistently.
  The ``HYPREDRV_SAFE_CALL`` macro logs location info and aborts or traps under ``HYPREDRV_DEBUG=1``.
- Public API naming: ``HYPRE_`` for public HYPRE APIs, ``hypre_`` or project-private helpers are not
  exported. In hypredrive, public C API follows the ``HYPREDRV_`` prefix.


Troubleshooting CI
------------------

- If a composite action cannot be found, ensure it is not excluded by ``.gitignore`` and that the
  repo is checked out (``actions/checkout``) before using local actions.
- On Ubuntu, missing MPI headers require the include hint used by the action
  (``-DMPI_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/mpich``).
- If Doxygen fails with an ``OUTPUT_DIRECTORY`` error, the project already configures it to a build-
  relative ``docs`` directory; rebuild after cleaning stale artifacts.
- If ``clang-tidy-fix`` causes unexpected renames in macro contexts, restrict fixes to specific
  files and review diffs manually.


Where to Look in the Tree
-------------------------

- CI config: ``.github/workflows/*.yml``
- Composite actions: ``.github/actions/setup-ubuntu``, ``.github/actions/build-hypre``
- CMake options: see the top-level ``CMakeLists.txt`` and the ``cmake/`` modules
- Tests: ``tests/`` and ``cmake/HYPREDRV_Testing.cmake``


