.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _utilities:

Utilities
=========

This chapter documents the sequence compression utility and the corresponding runtime
decompression path in hypredrive.

Overview
--------

The sequence-compression workflow has two components:

- ``hypredrive-lsseq-pack`` (offline utility):
  - Reads a sequence of IJ multipart files from a directory-based layout.
  - Packs the sequence into one lossless binary container.
- Runtime decompressor (inside hypredrive):
  - Triggered by ``linear_system.sequence_filename`` in YAML.
  - Reconstructs matrix/RHS/(optional) dofmap for each linear-system step.

This split keeps the heavy, one-time packaging step outside the core solver loop, while
making decompression available directly in the main executable/library.

Philosophy Behind the Compressed Binary File
--------------------------------------------

The container is designed around five principles:

1. **Lossless by construction**
   - Matrix indices and numeric values are stored as raw bytes after compression.
   - No quantization, truncation, or value transformations are applied.

2. **One file per sequence**
   - A single file stores all timesteps/systems in the selected suffix range.
   - The extension encodes the low-level codec (for example ``.zst.bin`` or ``.zlib.bin``).

3. **Sparsity-pattern deduplication**
   - Repeated matrix sparsity patterns are detected and stored only once.
   - Each system/part references the appropriate stored pattern by ``pattern_id``.

4. **Portable metadata, explicit structure**
   - Fixed-width metadata fields are used in the top-level format.
   - Internal offsets/sizes define all sections explicitly, so readers can validate layout.

5. **Runtime alignment with existing multipart readers**
   - Decompressed data is fed through existing matrix/vector/dofmap read paths.
   - This minimizes behavioral drift between directory mode and sequence-container mode.


Internal Container Structure (v1)
---------------------------------

The format is a chunked binary container with an uncompressed metadata front matter and
compressed payload blobs.

High-level sections:

- ``LSSeqHeader`` (fixed-size)
- optional ``LSSeqInfoHeader`` + UTF-8 manifest payload (provenance/debug block)
- ``LSSeqPartMeta[]`` (one entry per global part)
- ``LSSeqPatternMeta[]`` (one entry per unique sparsity pattern)
- ``LSSeqSystemPartMeta[]`` (one entry per ``(system, part)`` pair)
- optional ``LSSeqTimestepEntry[]``
- blob area (compressed payloads)

Compact Offset Map
~~~~~~~~~~~~~~~~~~

The file is laid out in this order (all offsets are byte offsets from file start):

.. code-block:: text

   0
   +------------------------------+
   | LSSeqHeader                  |
   +------------------------------+
   | LSSeqInfoHeader (optional)   |
   | + manifest payload (optional)|
   +------------------------------+  <- offset_part_meta
   | LSSeqPartMeta[num_parts]     |
   +------------------------------+  <- offset_pattern_meta
   | LSSeqPatternMeta[num_patterns]|
   +------------------------------+  <- offset_sys_part_meta
   | LSSeqSystemPartMeta[systems*parts] |
   +------------------------------+  <- offset_timestep_meta (optional section)
   | LSSeqTimestepEntry[num_timesteps]   |
   +------------------------------+  <- offset_blob_data
   | Blob #0 (compressed)         |
   | Blob #1 (compressed)         |
   | ...                          |
   +------------------------------+  EOF

Field-By-Field Reference
~~~~~~~~~~~~~~~~~~~~~~~~

``LSSeqHeader``

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``magic``
     - ``uint64_t``
     - File signature. Must equal ``LSSEQ_MAGIC``.
   * - ``version``
     - ``uint32_t``
     - Format version. Current value is ``1``.
   * - ``flags``
     - ``uint32_t``
     - Bitmask for optional sections (dofmap, timesteps, info/manifest).
   * - ``codec``
     - ``uint32_t``
     - Compression backend enum (none/zlib/zstd/lz4/lz4hc/blosc).
   * - ``num_systems``
     - ``uint32_t``
     - Number of linear systems in the sequence.
   * - ``num_parts``
     - ``uint32_t``
     - Number of global multipart partitions.
   * - ``num_patterns``
     - ``uint32_t``
     - Number of unique sparsity patterns in ``LSSeqPatternMeta[]``.
   * - ``num_timesteps``
     - ``uint32_t``
     - Number of entries in optional timestep section.
   * - ``offset_part_meta``
     - ``uint64_t``
     - Start of ``LSSeqPartMeta[]``.
   * - ``offset_pattern_meta``
     - ``uint64_t``
     - Start of ``LSSeqPatternMeta[]``.
   * - ``offset_sys_part_meta``
     - ``uint64_t``
     - Start of ``LSSeqSystemPartMeta[]``.
   * - ``offset_timestep_meta``
     - ``uint64_t``
     - Start of ``LSSeqTimestepEntry[]`` (if present).
   * - ``offset_blob_data``
     - ``uint64_t``
     - Start of compressed blob payload region.

``LSSeqInfoHeader`` (optional)

When ``LSSEQ_FLAG_HAS_INFO`` is set in ``LSSeqHeader.flags``, the file stores an
``LSSeqInfoHeader`` immediately after ``LSSeqHeader``. It is followed by a small, uncompressed
UTF-8 manifest payload that records provenance/debug information (resolved input paths, suffix
range, codec, build metadata, etc).

The manifest payload format is a sequence of ``key=value`` lines (one per line).

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``magic``
     - ``uint64_t``
     - Must equal ``LSSEQ_INFO_MAGIC``.
   * - ``version``
     - ``uint32_t``
     - Info header version (current: ``1``).
   * - ``flags``
     - ``uint32_t``
     - Payload format flags (currently ``LSSEQ_INFO_FLAG_PAYLOAD_KV``).
   * - ``endian_tag``
     - ``uint32_t``
     - Endianness guard (writer sets ``0x01020304``).
   * - ``payload_size``
     - ``uint64_t``
     - Size (bytes) of the manifest payload following this header.
   * - ``payload_hash_fnv1a64``
     - ``uint64_t``
     - FNV-1a 64-bit hash of the manifest payload.
   * - ``blob_hash_fnv1a64``
     - ``uint64_t``
     - FNV-1a 64-bit hash of the *compressed* blob area bytes.
   * - ``blob_bytes``
     - ``uint64_t``
     - Number of bytes in the blob area (compressed payload region).

Inspecting the manifest from the shell:

.. code-block:: bash

   # Print the first few lines of the manifest payload (may include binary noise after it).
   strings -n 8 myseq.zst.bin | sed -n '1,80p'

Programmatic access:

- ``LSSeqReadInfo("myseq.zst.bin", &payload, &nbytes)``

``LSSeqPartMeta``

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``row_lower``
     - ``uint64_t``
     - Inclusive global lower row bound for this part.
   * - ``row_upper``
     - ``uint64_t``
     - Inclusive global upper row bound for this part.
   * - ``nrows``
     - ``uint64_t``
     - Local row count for RHS/dofmap payload sizing.
   * - ``row_index_size``
     - ``uint64_t``
     - Byte width of row/column index entries.
   * - ``value_size``
     - ``uint64_t``
     - Byte width of matrix/RHS numerical values.

``LSSeqPatternMeta``

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``part_id``
     - ``uint32_t``
     - Owning part id for this deduplicated pattern.
   * - ``reserved``
     - ``uint32_t``
     - Reserved for future format extensions (currently zero).
   * - ``nnz``
     - ``uint64_t``
     - Number of nonzeros in this pattern.
   * - ``rows_blob_offset``
     - ``uint64_t``
     - Offset to compressed ``rows`` payload in blob area.
   * - ``rows_blob_size``
     - ``uint64_t``
     - Compressed size of ``rows`` payload.
   * - ``cols_blob_offset``
     - ``uint64_t``
     - Offset to compressed ``cols`` payload in blob area.
   * - ``cols_blob_size``
     - ``uint64_t``
     - Compressed size of ``cols`` payload.

``LSSeqSystemPartMeta``

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``pattern_id``
     - ``uint32_t``
     - Index into ``LSSeqPatternMeta[]`` for this system/part.
   * - ``flags``
     - ``uint32_t``
     - Per-system/part flags (reserved in current implementation).
   * - ``nnz``
     - ``uint64_t``
     - Number of value entries expected for matrix chunk.
   * - ``values_blob_offset``
     - ``uint64_t``
     - Offset to compressed matrix values chunk.
   * - ``values_blob_size``
     - ``uint64_t``
     - Compressed byte size for matrix values.
   * - ``rhs_blob_offset``
     - ``uint64_t``
     - Offset to compressed RHS values chunk.
   * - ``rhs_blob_size``
     - ``uint64_t``
     - Compressed byte size for RHS values.
   * - ``dof_blob_offset``
     - ``uint64_t``
     - Offset to compressed dofmap chunk (optional).
   * - ``dof_blob_size``
     - ``uint64_t``
     - Compressed byte size for dofmap chunk.
   * - ``dof_num_entries``
     - ``uint64_t``
     - Number of dofmap entries expected after decompression.

``LSSeqTimestepEntry``

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``timestep``
     - ``int32_t``
     - Logical timestep id from source metadata.
   * - ``ls_start``
     - ``int32_t``
     - Linear-system start index for preconditioner reuse grouping.

Header fields include:

- magic and version
- codec id
- section counts (systems, parts, patterns, timesteps)
- section offsets
- flags (dofmap present, timesteps present)

Part metadata (``LSSeqPartMeta``) stores static part properties:

- row bounds and row count
- row-index byte width
- value byte width

Pattern metadata (``LSSeqPatternMeta``) stores deduplicated sparsity descriptors:

- owning ``part_id``
- ``nnz``
- offsets/sizes for compressed ``rows`` and ``cols`` arrays

System/part metadata (``LSSeqSystemPartMeta``) stores per-step payload references:

- ``pattern_id`` to recover sparsity
- ``nnz``
- offsets/sizes for compressed matrix value chunk
- offsets/sizes for compressed RHS value chunk
- optional dofmap chunk metadata

Optional timesteps section (``LSSeqTimestepEntry``):

- stores ``(timestep, ls_start)`` pairs
- consumed by preconditioner-reuse-by-timestep when external timestep file is absent

How decompression works at runtime:

1. Read and validate metadata sections.
2. For the current ``ls_id``, resolve local part ids.
3. Load referenced blobs (pattern chunks and value chunks), decompress, and validate size.
4. Rebuild temporary multipart files and reuse standard IJ readers.
5. Continue solver execution as if data came from directory mode.

Binary Compatibility Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Metadata uses fixed-width integer types.
- Payload arrays (indices/values/dofmap) are stored as raw bytes before compression.
- Current implementation writes/reads metadata and payloads with host-native layout and is
  intended for homogeneous environments (same ABI and endianness across writer/reader).


Why and When to Use This Capability
-----------------------------------

Use sequence containers when:

- running many linear systems with repeated sparsity structure
- reducing filesystem metadata traffic from many small files
- shipping or archiving a sequence as one artifact
- preserving optional timestep grouping metadata in-band

Benefits:

- less on-disk duplication for repeated patterns
- simpler distribution (single artifact)
- YAML-level switch between directory and sequence modes


How to Use
----------

Build with compression enabled:

.. code-block:: bash

   cmake -S . -B build \
     -DHYPREDRV_ENABLE_COMPRESSION=ON
   cmake --build build --target hypredrive-lsseq-pack --parallel

Pack a sequence:

.. code-block:: bash

   # Quickstart: only dirname + output are required.
   # The packer auto-detects:
   #   - init_suffix=0 (or smallest available)
   #   - last_suffix=largest available
   #   - matrix/rhs prefixes from the first ls_XXXXX directory
   #   - optional dofmap + timesteps when present
   build/hypredrive-lsseq-pack \
     --dirname data/poromech2k/np1 \
     --output build/poromech2k_np1_lsseq

   # Explicit form (equivalent, but overrides auto-detection):
   build/hypredrive-lsseq-pack \
     --dirname data/poromech2k/np1/ls \
     --matrix-filename IJ.out.A \
     --rhs-filename IJ.out.b \
     --dofmap-filename dofmap.out \
     --init-suffix 0 \
     --last-suffix 24 \
     --digits-suffix 5 \
     --algo zstd \
     --output build/poromech2k_np1_lsseq

   # MPI: split work across parts (one rank can handle one or more part ids).
   # Only rank 0 writes the output file; other ranks send their payloads for assembly.
   mpiexec -n 4 build/hypredrive-lsseq-pack \
     --dirname data/poromech2k/np1 \
     --output build/poromech2k_np1_lsseq_mpi

Use in YAML:

.. code-block:: yaml

   linear_system:
     sequence_filename: build/poromech2k_np1_lsseq.zst.bin
     rhs_mode: file

Notes:

- ``rhs_mode`` remains authoritative. Container RHS is used only when ``rhs_mode: file``.
- If ``timestep_filename`` is omitted, embedded timesteps are used when available.
- Extensions map to codec: ``.bin`` (none), ``.zlib.bin``, ``.zst.bin``, ``.lz4.bin``,
  ``.lz4hc.bin``, ``.blosc.bin``.


Current Scope and Limitations
-----------------------------

- Input is expected in binary IJ multipart format.
- Sequence container path currently targets matrix, RHS, and optional dofmap data.
- Initial guess and explicit reference-solution files remain separate inputs.
- For compatibility with older hypre releases, some sequence regression coverage is gated by
  hypre version in test registration.
