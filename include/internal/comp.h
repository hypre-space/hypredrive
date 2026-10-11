/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef COMP_HEADER
#define COMP_HEADER

#include <stddef.h>
#include <stdint.h>

typedef enum
{
   COMP_NONE  = 0,
   COMP_ZLIB  = 1,
   COMP_ZSTD  = 2,
   COMP_LZ4   = 3,
   COMP_LZ4HC = 4,
   COMP_BLOSC = 5,
} comp_alg_t;

const char *hypredrv_compression_get_name(comp_alg_t algo);
const char *hypredrv_compression_get_extension(comp_alg_t algo);

/* compression_level: algorithm-dependent; use -1 for default. ZSTD: 1-22, -1=5. */
void hypredrv_compress(comp_alg_t algo, size_t isize, const void *input,
                       size_t *osize_ptr, void **output_ptr, int compression_level);
void hypredrv_decompress(comp_alg_t algo, size_t isize, const void *input,
                         size_t *osize_ptr, void **output_ptr);
int  hypredrv_decompress_slice(comp_alg_t algo, size_t isize, const void *input,
                               size_t offset, size_t size, void **output);

/* Where a slice stream reads its compressed blob (as written by
 * hypredrv_compress) from: read(ctx, offset, buffer, size) copies `size` bytes
 * starting `offset` bytes into the blob and returns 1 on success. */
typedef struct
{
   int (*read)(void *ctx, uint64_t offset, void *buffer, size_t size);
   void *ctx;
} hypredrv_SliceSource;
hypredrv_SliceSource hypredrv_SliceSourceFromMemory(const void *blob);

/* Resumable slice decoder over one compressed blob of `isize` bytes (zstd/zlib
 * only; Create returns NULL for other codecs). Compressed input is read from
 * the source passed to each call, so a stream keeps no copy of the blob; every
 * call must pass a source for the same blob. Reading slices in increasing
 * offset order decodes the blob once; an earlier offset restarts from the
 * beginning. */
typedef struct hypredrv_SliceStream_struct hypredrv_SliceStream;
hypredrv_SliceStream *hypredrv_SliceStreamCreate(comp_alg_t algo, size_t isize,
                                                 const hypredrv_SliceSource *src);
int  hypredrv_SliceStreamRead(hypredrv_SliceStream *, const hypredrv_SliceSource *src,
                              size_t offset, size_t size, void **output);
void hypredrv_SliceStreamDestroy(hypredrv_SliceStream **);

#endif /* COMP_HEADER */
