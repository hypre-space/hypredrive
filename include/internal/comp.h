/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef COMP_HEADER
#define COMP_HEADER

#include <stddef.h>

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
comp_alg_t  hypredrv_compression_from_filename(const char *filename);

/* compression_level: algorithm-dependent; use -1 for default. ZSTD: 1-22, -1=5. */
void hypredrv_compress(comp_alg_t algo, size_t isize, const void *input,
                       size_t *osize_ptr, void **output_ptr, int compression_level);
void hypredrv_decompress(comp_alg_t algo, size_t isize, const void *input,
                         size_t *osize_ptr, void **output_ptr);
int  hypredrv_decompress_slice(comp_alg_t algo, size_t isize, const void *input,
                               size_t offset, size_t size, void **output);

/* Resumable slice decoder over one compressed blob (zstd/zlib only; Create
 * returns NULL for other codecs). Reading slices in increasing offset order
 * decodes the blob once; an earlier offset restarts from the beginning. */
typedef struct hypredrv_SliceStream_struct hypredrv_SliceStream;
hypredrv_SliceStream *hypredrv_SliceStreamCreate(comp_alg_t algo, size_t isize,
                                                 const void *input);
int  hypredrv_SliceStreamRead(hypredrv_SliceStream *, size_t offset, size_t size,
                              void **output);
void hypredrv_SliceStreamDestroy(hypredrv_SliceStream **);

#endif /* COMP_HEADER */
