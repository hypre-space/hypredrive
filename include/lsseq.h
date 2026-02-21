/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef LSSEQ_HEADER
#define LSSEQ_HEADER

#include <stddef.h>
#include <stdint.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_utilities.h"
#include "comp.h"
#include "compatibility.h"
#include "containers.h"

#define LSSEQ_MAGIC UINT64_C(0x3151534c56445248) /* "HDRVLSQ1" */
#define LSSEQ_VERSION UINT32_C(1) /* batched blobs per part (values/rhs/dof) */

enum
{
   LSSEQ_FLAG_HAS_DOFMAP    = 1u << 0,
   LSSEQ_FLAG_HAS_TIMESTEPS = 1u << 1,
   LSSEQ_FLAG_HAS_INFO      = 1u << 2
};

typedef struct LSSeqHeader_struct
{
   uint64_t magic;
   uint32_t version;
   uint32_t flags;
   uint32_t codec;
   uint32_t num_systems;
   uint32_t num_parts;
   uint32_t num_patterns;
   uint32_t num_timesteps;
   uint64_t offset_part_meta;
   uint64_t offset_pattern_meta;
   uint64_t offset_sys_part_meta;
   uint64_t offset_timestep_meta;
   uint64_t offset_blob_data;
   uint64_t offset_part_blob_table; /* v2 only: table of (offset,size)^3 per part for
                                       values/rhs/dof */
} LSSeqHeader;

/* Part blob table (v2): 6*uint64_t per part = values_offset, values_size, rhs_offset,
 * rhs_size, dof_offset, dof_size (relative to offset_blob_data) */
enum
{
   LSSEQ_PART_BLOB_ENTRIES = 6
};

/* Mandatory info/manifest block written immediately after LSSeqHeader.
 * For LSSEQ_VERSION=1, LSSEQ_FLAG_HAS_INFO must be set and this header/payload
 * must be present.
 *
 * The payload is stored as UTF-8 key/value lines (one "key=value" per line).
 * This block is intended for provenance/debuggability, not for runtime parsing. */
#define LSSEQ_INFO_MAGIC UINT64_C(0x31464E4956524448) /* "HDRVINF1" */
#define LSSEQ_INFO_VERSION UINT32_C(1)

enum
{
   LSSEQ_INFO_FLAG_PAYLOAD_KV = 1u << 0
};

typedef struct LSSeqInfoHeader_struct
{
   uint64_t magic;
   uint32_t version;
   uint32_t flags;
   uint32_t endian_tag; /* 0x01020304 */
   uint32_t reserved;
   uint64_t payload_size;
   uint64_t payload_hash_fnv1a64;
   uint64_t blob_hash_fnv1a64;
   uint64_t blob_bytes;
} LSSeqInfoHeader;

typedef struct LSSeqPartMeta_struct
{
   uint64_t row_lower;
   uint64_t row_upper;
   uint64_t nrows;
   uint64_t row_index_size;
   uint64_t value_size;
} LSSeqPartMeta;

typedef struct LSSeqPatternMeta_struct
{
   uint32_t part_id;
   uint32_t reserved;
   uint64_t nnz;
   uint64_t rows_blob_offset;
   uint64_t rows_blob_size;
   uint64_t cols_blob_offset;
   uint64_t cols_blob_size;
} LSSeqPatternMeta;

typedef struct LSSeqSystemPartMeta_struct
{
   uint32_t pattern_id;
   uint32_t flags;
   uint64_t nnz;
   uint64_t values_blob_offset;
   uint64_t values_blob_size;
   uint64_t rhs_blob_offset;
   uint64_t rhs_blob_size;
   uint64_t dof_blob_offset;
   uint64_t dof_blob_size;
   uint64_t dof_num_entries;
} LSSeqSystemPartMeta;

typedef struct LSSeqTimestepEntry_struct
{
   int32_t timestep;
   int32_t ls_start;
} LSSeqTimestepEntry;

int LSSeqReadSummary(const char *filename, int *num_systems, int *num_patterns,
                     int *has_dofmap, int *has_timesteps);
int LSSeqReadInfo(const char *filename, char **payload_ptr, size_t *payload_size);
int LSSeqReadMatrix(MPI_Comm comm, const char *filename, int ls_id,
                    HYPRE_MemoryLocation memory_location, HYPRE_IJMatrix *matrix_ptr);
int LSSeqReadRHS(MPI_Comm comm, const char *filename, int ls_id,
                 HYPRE_MemoryLocation memory_location, HYPRE_IJVector *rhs_ptr);
int LSSeqReadDofmap(MPI_Comm comm, const char *filename, int ls_id,
                    IntArray **dofmap_ptr);
int LSSeqReadTimesteps(const char *filename, IntArray **timestep_starts);

#endif /* LSSEQ_HEADER */
