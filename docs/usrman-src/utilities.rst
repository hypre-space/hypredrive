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

- ``hypredrive-lsseq`` (offline utility):

  - Reads a sequence of IJ multipart files from a directory-based layout.
  - Packs the sequence into one lossless binary container.
- Runtime decompressor (inside hypredrive):

  - The ``linear_system.sequence_filename`` YAML key starts the decompressor.
  - The decompressor reconstructs the matrix, right-hand side, and optional degree-of-freedom map.

This design keeps the one-time packaging work outside the core solver loop. The main
executable and library perform the decompression.

Philosophy Behind the Compressed Binary File
--------------------------------------------

The container design follows five principles:

1. **Lossless by construction**

   - The utility stores matrix indices and numeric values as raw bytes after compression.
   - The utility does not apply quantization, truncation, or value transformations.

2. **One file per sequence**

   - A single file stores all timesteps/systems in the selected suffix range.
   - The extension encodes the low-level codec (for example ``.zst.bin`` or ``.zlib.bin``).

3. **Sparsity-pattern deduplication**

   - The utility detects and stores each repeated matrix sparsity pattern only once.
   - Each system/part references the appropriate stored pattern by ``pattern_id``.

4. **Portable metadata, explicit structure**

   - The top-level format uses fixed-width metadata fields.
   - Internal offsets/sizes define all sections explicitly, so readers can validate layout.

5. **Runtime alignment with existing multipart readers**

   - The runtime sends decompressed data through the existing data readers.
   - This minimizes behavioral drift between directory mode and sequence-container mode.


Internal Container Structure
----------------------------

The format is a chunked binary container. It has uncompressed metadata followed by
compressed data blobs. Each part has one batched blob for all system values. Each part
also has one blob for right-hand sides and one for the degree-of-freedom map. These
batches compress better than separate blobs for each system and part.

The part blob table gives the file offset and size of each batched blob.
``LSSeqSystemPartMeta`` gives the decompressed offset and size for each system in the blob.

High-level sections:

- ``LSSeqHeader`` (fixed-size)
- Mandatory ``LSSeqInfoHeader`` + UTF-8 manifest payload (provenance/debug block)
- ``LSSeqPartMeta[]`` (one entry per global part)
- ``LSSeqPatternMeta[]`` (one entry per unique sparsity pattern)
- ``LSSeqSystemPartMeta[]`` (one entry per ``(system, part)`` pair)
- Part blob table: ``6 * num_parts`` ``uint64_t`` (offset and size for values, RHS, dof per part)
- Optional ``LSSeqTimestepEntry[]``
- Blob area (compressed payloads)

Compact Offset Map
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   0
   +------------------------------------+
   | LSSeqHeader                        |
   +------------------------------------+
   | LSSeqInfoHeader + manifest payload |
   +------------------------------------+  <- offset_part_meta
   | LSSeqPartMeta[num_parts]           |
   +------------------------------------+  <- offset_pattern_meta
   | LSSeqPatternMeta[num_patterns]     |
   +------------------------------------+  <- offset_sys_part_meta
   | LSSeqSystemPartMeta[systems*parts] |
   +------------------------------------+  <- offset_part_blob_table
   | Part blob table [6*num_parts]      |
   +------------------------------------+  <- offset_timestep_meta (optional)
   | LSSeqTimestepEntry[num_timesteps]  |
   +------------------------------------+  <- offset_blob_data
   | Pattern blobs,                     |
   | part batched blobs (vals/rhs/dof)  |
   +------------------------------------+  EOF

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
     - The writer sets the file signature to ``LSSEQ_MAGIC``.
   * - ``version``
     - ``uint32_t``
     - Format version. Current value is ``1`` (batched per-part blobs).
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
   * - ``offset_part_blob_table``
     - ``uint64_t``
     - Start of part blob table (``6*num_parts`` ``uint64_t``).

``LSSeqInfoHeader`` (mandatory)

``LSSEQ_VERSION=1`` requires ``LSSEQ_FLAG_HAS_INFO`` in ``LSSeqHeader.flags``. It also
requires an ``LSSeqInfoHeader`` immediately after ``LSSeqHeader``. A small,
uncompressed UTF-8 manifest follows the information header. The manifest records
input paths, the suffix range, the codec, and build metadata.

The manifest payload format is a sequence of ``key=value`` lines (one per line).

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Meaning
   * - ``magic``
     - ``uint64_t``
     - The writer sets this field to ``LSSEQ_INFO_MAGIC``.
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

- ``hypredrv_LSSeqReadInfo("myseq.zst.bin", &payload, &nbytes)``

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
     - Decompressed byte offset within this part’s batched values blob.
   * - ``values_blob_size``
     - ``uint64_t``
     - Decompressed byte size (length of slice) for matrix values.
   * - ``rhs_blob_offset``
     - ``uint64_t``
     - Decompressed offset within this part’s batched RHS blob.
   * - ``rhs_blob_size``
     - ``uint64_t``
     - Decompressed byte size for RHS slice.
   * - ``dof_blob_offset``
     - ``uint64_t``
     - Decompressed offset within this part’s batched dofmap blob.
   * - ``dof_blob_size``
     - ``uint64_t``
     - Decompressed byte size for dofmap slice (or 0 if no dofmap).
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

- Magic and version
- Codec ID
- Section counts (systems, parts, patterns, timesteps)
- Section offsets
- Flags (dofmap present, timesteps present)

Part metadata (``LSSeqPartMeta``) stores static part properties:

- Row bounds and row count
- Row-index byte width
- Value byte width

Pattern metadata (``LSSeqPatternMeta``) stores deduplicated sparsity descriptors:

- Owning ``part_id``
- ``nnz``
- Offsets and sizes for compressed ``rows`` and ``cols`` arrays

System/part metadata (``LSSeqSystemPartMeta``) stores per-step payload references:

- ``pattern_id`` to recover sparsity
- ``nnz``
- Offsets and sizes for the compressed matrix value chunk
- Offsets and sizes for the compressed RHS value chunk
- Optional DOF map chunk metadata

Optional timesteps section (``LSSeqTimestepEntry``):

- Stores ``(timestep, ls_start)`` pairs
- Supplies time-step groups for preconditioner reuse when an external time-step file is absent

How decompression works at runtime:

1. Read and validate metadata sections.
2. For the current ``ls_id``, resolve local part ids.
3. Load the referenced blobs, decompress them, and validate their sizes.
4. Rebuild temporary multipart files and reuse standard IJ readers.
5. Continue solver execution as if data came from directory mode.

Binary Compatibility Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Metadata uses fixed-width integer types.
- The writer stores payload arrays as raw bytes before compression.
- The implementation reads and writes metadata and payloads with the host-native layout.
  Use the same ABI and byte order for the writer and reader.


Why and When to Use This Capability
-----------------------------------

Use sequence containers in these cases:

- You run many linear systems with a repeated sparsity structure.
- Many small files cause excessive file-system metadata traffic.
- You distribute or archive a sequence as one artifact.
- You preserve optional time-step group metadata in the container.

Benefits:

- Less on-disk duplication for repeated patterns
- Simpler distribution as one artifact
- YAML-level selection of directory or sequence mode


How to Use
----------

Build with compression enabled:

.. code-block:: bash

   cmake -S . -B build \
     -DHYPREDRV_ENABLE_COMPRESSION=ON
   cmake --build build --target hypredrive-lsseq --parallel

Pack a sequence:

.. code-block:: bash

   # Quickstart: only dirname + output are required.
   # The packer auto-detects:
   #   - init_suffix=0 (or smallest available)
   #   - last_suffix=largest available
   #   - matrix/rhs prefixes from the first ls_XXXXX directory
   #   - optional dofmap + timesteps when present
   build/hypredrive-lsseq \
     --dirname data/poromech2k/np1 \
     --output build/poromech2k_np1_lsseq

   # Explicit form (equivalent, but overrides auto-detection):
   build/hypredrive-lsseq \
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
   mpiexec -n 4 build/hypredrive-lsseq \
     --dirname data/poromech2k/np1 \
     --output build/poromech2k_np1_lsseq_mpi

Inspect packed metadata:

.. code-block:: bash

   build/hypredrive-lsseq metadata \
     --input build/poromech2k_np1_lsseq.zst.bin

Unpack a sequence back to directory layout:

.. code-block:: bash

   build/hypredrive-lsseq unpack \
     --input build/poromech2k_np1_lsseq.zst.bin \
     --output-dir build/poromech2k_np1_unpacked

Use in YAML:

.. code-block:: yaml

   linear_system:
     sequence_filename: build/poromech2k_np1_lsseq.zst.bin
     rhs_mode: file

Notes:

- ``rhs_mode`` remains authoritative. The runtime reads the container right-hand side
  only when ``rhs_mode: file``.
- If you omit ``timestep_filename``, the runtime uses available embedded timesteps.
- ``LSSeqInfoHeader`` + manifest payload are mandatory for ``LSSEQ_VERSION=1`` files.
- Extensions map to codec: ``.bin`` (none), ``.zlib.bin``, ``.zst.bin``, ``.lz4.bin``,
  ``.lz4hc.bin``, ``.blosc.bin``.


Current Scope and Limitations
-----------------------------

- The utility accepts input in binary IJ multipart format.
- Sequence container path currently targets matrix, RHS, and optional dofmap data.
- Initial guess and explicit reference-solution files remain separate inputs.
- Test registration selects some sequence tests according to the hypre version.
  This selection maintains compatibility with older hypre releases.
