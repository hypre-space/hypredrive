.. _developer-notes:

Developer Notes
===============

This chapter collects practical guidance for contributing to hypredrive with a focus on:

- Continuous Integration (CI)
- Static code analysis (cppcheck, clang-tidy)
- Code coverage (gcov/gcovr, CTest)
- Fuzzing replay and live fuzz campaigns

It explains the CI structure, local checks, and the CMake options for these
workflows. New contributors can use this chapter before they open their first
pull request.

Library API Ownership Semantics
-------------------------------

For public setter APIs that accept optional external hypre objects:

- ``HYPREDRV_LinearSystemSetInitialGuess(h, vec)``
- ``HYPREDRV_LinearSystemSetReferenceSolution(h, vec)``
- ``HYPREDRV_LinearSystemSetPrecMatrix(h, mat)``

the ``NULL`` argument keeps the existing file/default behavior from parsed input args.
When the argument is non-``NULL``, hypredrive uses the supplied object directly.

Ownership depends on library mode:

- ``HYPREDRV_SetLibraryMode`` enabled: hypredrive borrows supplied objects and
  never destroys them.
- ``HYPREDRV_SetLibraryMode`` disabled: Ownership transfers to `hypredrive`.
  `hypredrive` can destroy the objects during replacement or object teardown.

Keep this object-lifetime contract explicit in code comments and user
documentation.


CI Overview
-----------

Workflows
~~~~~~~~~

`hypredrive` uses these GitHub Actions workflows in ``.github/workflows``:

- ``ci.yml``: Main build and test matrix for Ubuntu and macOS. It uses GCC and Clang with Debug builds.
- ``format.yml``: code style checks (clang-format, private/public naming prefix validation, binary symbol prefix validation)
- ``docs.yml``: builds documentation (Sphinx and Doxygen jobs)
- ``coverage.yml``: builds with coverage instrumentation and generates HTML/XML reports
- ``analysis.yml``: Static analysis with cppcheck and clang-tidy. Dynamic analysis with ASan and UBSan.

To reduce duplication, CI uses composite actions in ``.github/actions``:

- ``setup-ubuntu``: installs compilers, MPI, and tools
- ``build-hypre``: Builds and caches HYPRE from source. Its ``hypre_prefix`` output
  provides the installation prefix.

HYPRE reuse and caching
~~~~~~~~~~~~~~~~~~~~~~~

All jobs that need HYPRE call the ``build-hypre`` action and pass:

- ``hypre_version`` (default: ``master``)
- ``compiler`` (``gcc`` or ``clang``)
- ``build_type`` (``Debug``)
- ``hypre_shared_libs`` (``ON``)
- ``os`` (``ubuntu-latest`` or ``macos-latest``)

The action caches HYPRE under ``${{ runner.tool_cache }}/hypre/<version>``. Its key
includes the OS, compiler, build type, and library type (``shared`` or ``static``).

Automatic HYPRE build
~~~~~~~~~~~~~~~~~~~~~

For a local build, CMake uses `FetchContent` to retrieve and build HYPRE when
``HYPRE_ROOT`` is not set. This process:

- Downloads HYPRE from the default ``master`` branch on GitHub.
- Passes the CMake arguments from the parent project to HYPRE.
- Builds HYPRE in the same build tree.
- Selects the HYPRE version with ``-DHYPRE_VERSION=<version>``.

To use an existing HYPRE installation, set ``-DHYPRE_ROOT=<path>``.

MPI configuration
~~~~~~~~~~~~~~~~~

- MPI implementation: OpenMPI (default in CI). MPICH is also supported.
- C compiler for CMake: ``-DCMAKE_C_COMPILER=mpicc``.
- On Ubuntu, the action adds ``-DMPI_INCLUDE_DIR=/usr/lib/x86_64-linux-gnu/openmpi/include``.

Local reproduction of CI
~~~~~~~~~~~~~~~~~~~~~~~~

On Ubuntu (gcc example):

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y cmake ninja-build libopenmpi-dev openmpi-bin ccache clang-format gcc
   # Build HYPRE once and set CMAKE_PREFIX_PATH
   git clone --depth 1 --branch master https://github.com/hypre-space/hypre.git
   cmake -S hypre/src -B hypre/build -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON \
     -DHYPRE_BUILD_TESTS=OFF -DHYPRE_BUILD_EXAMPLES=OFF \
     -DCMAKE_C_COMPILER=mpicc -DMPI_INCLUDE_DIR=/usr/lib/x86_64-linux-gnu/openmpi/include \
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
     -DHYPREDRV_ENABLE_TESTING=ON -DHYPREDRV_ENABLE_EXAMPLES=ON \
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
- Dynamic analysis: When ``HYPREDRV_ENABLE_ANALYSIS=ON``, CMake enables sanitizers
  for all targets. Run the tests with ``ctest`` to exercise the sanitizers.

cppcheck configuration
~~~~~~~~~~~~~~~~~~~~~~

- The cppcheck tool examines only ``src/`` to keep the signal high and the run time short.
- The configuration adds HYPRE includes so that macros and types resolve.
- Threads: cppcheck uses the same number as CMake parallel level when available.
- Known suppressions: Ignore irrelevant warnings for mixed precision or specific HYPRE
  headers, such as ``_hypre_IJ_mv.h`` and ``_hypre_utilities.h``.

clang-tidy usage
~~~~~~~~~~~~~~~~

- Use the ``clang-tidy`` target for read-only reports.
- Do not run aggressive automatic fixes on the full tree. They can damage identifiers in
  macro-heavy code. If you use ``clang-tidy -fix``, limit it to specific files and review
  each difference. Prefer incremental edits with human review.
- Confirm that new checks agree with the project style and C patterns.

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


Fuzzing
-------

Overview
~~~~~~~~

The ``-DHYPREDRV_ENABLE_FUZZING=ON`` option enables fuzzing. The fuzzing
tree lives under ``tests/fuzz/`` and uses a single harness source,
``tests/fuzz/harness.c``, compiled into mode-specific targets:

- ``hypredrv-fuzz-parse``: parses YAML and CLI-style argument fragments
- ``hypredrv-fuzz-solve``: parses YAML and exercises a small one-rank solve path
- ``hypredrv-fuzz-lsseq``: reads compressed linear-system sequence files when
  ``HYPREDRV_ENABLE_COMPRESSION=ON``
- ``hypredrv-fuzz-matrix``: reads IJ matrix files and multipart matrix sequences
- ``hypredrv-fuzz-vector``: reads IJ vector files and multipart vector sequences

The supported engines are:

- ``replay``: deterministic CTest replay of seeds and regressions
- ``libfuzzer``: live in-process fuzzing with Clang's libFuzzer
- ``afl``: AFL++ fuzzing with ``afl-clang-fast`` or ``afl-clang-lto``

Enabling fuzzing also enables testing and analysis, disables shared libraries, and
rejects coverage builds. Separate build trees prevent conflicts between coverage and
sanitizer or fuzzing instrumentation.

CMake options and targets
~~~~~~~~~~~~~~~~~~~~~~~~~

- Enable fuzzing: ``-DHYPREDRV_ENABLE_FUZZING=ON``.
- Select an engine: ``-DHYPREDRV_FUZZ_ENGINE=replay|libfuzzer|afl``. The default is
  ``replay``.
- Enable MemorySanitizer for fuzzing builds: ``-DHYPREDRV_FUZZ_MSAN=ON``. This requires
  Clang and a compatible dependency stack.
- CMake registers replay tests under the ``fuzz-replay`` CTest label.
- The build targets are ``hypredrv-fuzz-parse``, ``hypredrv-fuzz-solve``,
  ``hypredrv-fuzz-lsseq``, ``hypredrv-fuzz-matrix``, and ``hypredrv-fuzz-vector``.

The solve replay tests require HYPRE APIs guarded by
``HYPREDRV_HAVE_HYPRE_21900_DEV0``. When those APIs are unavailable, CMake prints a
status message and skips solve replay registration.

Replay builds
~~~~~~~~~~~~~

Replay is the CI-friendly mode. It builds normal executables that accept one or more
input files and registers every seed and regression as a CTest test.

.. code-block:: bash

   cmake -S . -B build-fuzz -G Ninja \
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DHYPREDRV_ENABLE_FUZZING=ON \
     -DHYPREDRV_ENABLE_TESTING=ON
   cmake --build build-fuzz --parallel
   ctest --test-dir build-fuzz -L fuzz-replay --output-on-failure

To replay one mode:

.. code-block:: bash

   ctest --test-dir build-fuzz -L fuzz-replay -R "matrix" --output-on-failure

CMake compiles replay binaries with a mode-specific ``FUZZ_MODE`` definition. For
manual debugging, ``HYPREDRV_FUZZ_MODE=<mode>`` can override that build mode in
replay builds.

Because parse replay uses ``examples/*.yml`` directly, checked-in example YAML files
are part of the fuzz replay corpus. Keep new examples valid as parser input. Otherwise,
update the fuzz CMake registration to exclude or control them.

Live libFuzzer runs
~~~~~~~~~~~~~~~~~~~

Use a separate Clang build tree for live libFuzzer campaigns:

.. code-block:: bash

   cmake -S . -B build-fuzz -G Ninja \
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DHYPREDRV_ENABLE_FUZZING=ON \
     -DHYPREDRV_FUZZ_ENGINE=libfuzzer \
     -DHYPREDRV_ENABLE_TESTING=ON
   cmake --build build-fuzz --target hypredrv-fuzz-parse --parallel
   tests/fuzz/fuzzing.sh parse 300 libfuzzer

The helper script accepts ``parse``, ``solve``, ``lsseq``, ``matrix``, or ``vector`` as
the first argument and the run duration in seconds as the second argument:

.. code-block:: bash

   tests/fuzz/fuzzing.sh matrix 600 libfuzzer
   tests/fuzz/fuzzing.sh vector 600 libfuzzer

Set ``HYPREDRV_FUZZ_BUILD_DIR`` if the fuzz build tree is not ``build-fuzz``.

AFL++ runs
~~~~~~~~~~

Configure AFL++ builds with the AFL compiler wrapper:

.. code-block:: bash

   CC=afl-clang-fast cmake -S . -B build-fuzz-afl -G Ninja \
     -DHYPREDRV_ENABLE_FUZZING=ON \
     -DHYPREDRV_FUZZ_ENGINE=afl \
     -DHYPREDRV_ENABLE_TESTING=ON
   cmake --build build-fuzz-afl --target hypredrv-fuzz-matrix --parallel
   HYPREDRV_FUZZ_BUILD_DIR=$PWD/build-fuzz-afl tests/fuzz/fuzzing.sh matrix 600 afl

The script chooses a default AFL timeout per mode. Override it with
``HYPREDRV_FUZZ_AFL_TIMEOUT_MS`` when investigating slower paths, especially ``solve``.
For parse and solve modes, the harness runs a small warmup input before starting the
AFL forkserver. This warmup initializes the HYPRE and hypredrive global state
before AFL forks persistent children.

Corpora, dictionaries, and generated seeds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The versioned fuzz inputs are small and deterministic:

- ``tests/fuzz/seeds/<mode>/`` contains mode-specific seed inputs that are not already
  represented elsewhere in the repository.
- ``tests/fuzz/regressions/<mode>/`` contains minimized inputs for fixed bugs.
- ``tests/fuzz/dicts/`` contains dictionaries for YAML/CLI fragments and IJ-like binary
  formats.
- ``tests/fuzz/tools/`` contains seed generation helpers for IJ matrix, IJ vector, and
  linear-system sequence inputs.

Do not duplicate existing example YAML files under ``tests/fuzz/seeds``. Parse replay
uses ``examples/*.yml`` directly, and solve replay uses ``examples/ex1.yml``,
``examples/ex2.yml``, and ``examples/ex7.yml`` by path before adding the fuzz-specific
solve seeds.

The live fuzzing helper creates per-run corpora under ``build-fuzz/fuzz-run/<mode>-<time>``
and copies in the checked-in seeds, regressions, and reused example YAML inputs. Those
run directories, generated corpora, crashes, hangs, and minimization artifacts are local
outputs. Do not commit these directories or their generated files.

Regression workflow
~~~~~~~~~~~~~~~~~~~

When a live campaign finds a crash, leak, timeout, or sanitizer report:

- Minimize the input with the engine that found it.
- Add the minimized reproducer to ``tests/fuzz/regressions/<mode>/`` with a descriptive
  filename.
- Confirm the replay test fails before the code fix.
- Fix the owning module rather than weakening the harness.
- Re-run the relevant replay subset, then the full ``fuzz-replay`` label.

Typical replay commands:

.. code-block:: bash

   ctest --test-dir build-fuzz -L fuzz-replay -R "solve" --output-on-failure
   ctest --test-dir build-fuzz -L fuzz-replay --output-on-failure

Use mode ownership as the initial triage path:

- ``parse`` maps primarily to ``src/yaml.c`` and ``src/args.c``.
- ``solve`` also reaches ``src/linsys.c``, ``src/solver.c``, ``src/precon.c``, and
  related setup paths.
- ``matrix`` maps to ``src/matrix.c``.
- ``vector`` maps to vector input paths.
- ``lsseq`` maps to ``src/lsseq.c`` and compression-dependent sequence handling.

CI fuzzing
~~~~~~~~~~

The fuzzing workflow has three tiers:

- Pull requests run replay tests for deterministic coverage of the checked-in corpus and
  regressions.
- Labeled pull requests can run short libFuzzer smoke jobs.
- Scheduled runs execute longer nightly libFuzzer campaigns and upload run artifacts.

Treat replay failures as normal test failures. Minimize live fuzz failures and add them to
``tests/fuzz/regressions/<mode>/``. Future CI can then reproduce the same bug.


Code Coverage
-------------

Overview
~~~~~~~~

The ``coverage.yml`` workflow uses ``gcovr`` to generate HTML and XML coverage files. The
build uses ``-O0 -g --coverage`` and runs unit tests with CTest. The job output also shows
a percentage summary.

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
     -DHYPREDRV_ENABLE_TESTING=ON -DHYPREDRV_ENABLE_EXAMPLES=ON \
     -DHYPREDRV_ENABLE_COVERAGE=ON \
     -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_PREFIX_PATH=${HYPRE_PREFIX}
   cmake --build build-coverage --parallel
   ctest --test-dir build-coverage --output-on-failure
   cmake --build build-coverage --target coverage
   # Open build-coverage/coverage.html

Performance Profiling with Caliper
-----------------------------------

Overview
~~~~~~~~

When you set ``HYPREDRV_ENABLE_CALIPER=ON``, hypredrive uses Caliper markers for
performance profiles. Caliper measures run-time performance and helps identify
code bottlenecks.

Building with Caliper
~~~~~~~~~~~~~~~~~~~~~

Enable Caliper support during CMake configuration:

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
     -DHYPREDRV_ENABLE_CALIPER=ON \
     -DCMAKE_PREFIX_PATH=${HYPRE_PREFIX}
   cmake --build build --parallel

By default, CMake retrieves and builds Caliper when it cannot find Caliper. To use an
existing Caliper installation, set ``CALIPER_DIR`` or ``CALIPER_ROOT``. Alternatively,
configure the search path so that ``find_package(caliper)`` finds Caliper.

Using Caliper for Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To collect performance data, set the ``CALI_CONFIG`` environment variable when running hypredrive:

.. code-block:: bash

   CALI_CONFIG=runtime-report mpirun -np 1 ./build/hypredrive-cli examples/ex1.yml

Common Caliper configurations:

- ``runtime-report``: Print a summary report to stdout at program end
- ``runtime-report,max_column_width=200,calc.inclusive,output=stdout,mpi-report``: Print a detailed
  report, including MPI-related information, to stdout at program end.
- ``spot``: Generate Caliper output files for analysis with Caliper's spot tool

For more information about Caliper configurations and services, see the
`Caliper documentation <https://software.llnl.gov/Caliper/>`_.

Documentation Builds
--------------------

The project uses two documentation systems:

- Doxygen provides the developer reference. Its target is ``doxygen-doc``.
- Sphinx provides the user manual. Its targets are ``sphinx-doc`` and ``sphinx-latexpdf``.

Enable the documentation with ``-DHYPREDRV_ENABLE_DOCS=ON``. The combined ``docs``
target builds Doxygen and then starts Sphinx. It uses ``docs/Makefile`` when that
file is available. Otherwise, it uses ``sphinx-build`` directly.

Notes:

- Doxygen LaTeX can build ``refman.pdf``. CI copies this file to
  ``build-docs/docs/devman-hypredrive.pdf``.
- The ``sphinx-latexpdf`` target copies an available Sphinx PDF to
  ``docs/usrman-hypredrive.pdf``.


Coding Style and Project Conventions
------------------------------------

- CI enforces ``clang-format``. Run ``make format`` or the ``format`` CMake target to
  format C source and header files.
- Some macros use ``// clang-format off`` and ``// clang-format on`` guards to
  preserve their layout. Edit those blocks only when necessary.
- Follow const-correctness and safe API patterns. Use error-reporting helpers consistently.
  The ``HYPREDRV_SAFE_CALL`` macro logs location info and aborts or traps under ``HYPREDRV_DEBUG=1``.
- For internal run-time traces, use ``HYPREDRV_LOG_LEVEL``:
  ``0`` disables traces (default), ``1`` logs lifecycle boundaries, ``2`` adds
  decision/context messages, and ``3`` enables deeper parse/linear-system/scaling subphase traces.
  At every nonzero level, HYPREDRV logs the effective HYPRE execution policy after input
  configuration and whenever it changes.
  By default, hypredrive writes trace output from rank 0 to ``stderr``. Set
  ``HYPREDRV_LOG_STREAM=stdout`` to write the same traces to ``stdout`` instead.
- ``HYPREDRV_LOG_LEVEL`` controls hypredrive traces only. Hypre's own logging remains controlled
  by ``HYPRE_LOG_LEVEL`` (forwarded during runtime initialization when supported by the linked
  Hypre version).
- Public API naming: ``HYPRE_`` for public HYPRE APIs, ``hypre_`` or project-private helpers are not
  exported. In hypredrive, public C API follows the ``HYPREDRV_`` prefix.


Troubleshooting CI
------------------

- If a composite action is missing, check whether ``.gitignore`` excludes it.
  Run ``actions/checkout`` before you use local actions.
- On Ubuntu, missing MPI headers require the include hint used by the action
  (``-DMPI_INCLUDE_DIR=/usr/lib/x86_64-linux-gnu/openmpi/include`` for OpenMPI, or
  ``/usr/include/x86_64-linux-gnu/mpich`` for MPICH).
- If Doxygen reports an ``OUTPUT_DIRECTORY`` error, remove stale build artifacts and build
  again. The project uses a build-relative ``docs`` directory.
- If ``clang-tidy-fix`` causes unexpected renames in macro contexts, restrict fixes to specific
  files and review diffs manually.


Where to Look in the Tree
-------------------------

- CI configuration: ``.github/workflows/*.yml``
- Composite actions: ``.github/actions/setup-ubuntu``, ``.github/actions/build-hypre``
- CMake options: see the top-level ``CMakeLists.txt`` and the ``cmake/`` modules
- Tests: ``tests/`` and ``cmake/HYPREDRV_Testing.cmake``
