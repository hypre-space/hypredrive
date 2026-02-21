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
   COMP_BLOSC = 5
} comp_alg_t;

const char *hypredrv_compression_get_name(comp_alg_t algo);
const char *hypredrv_compression_get_extension(comp_alg_t algo);
comp_alg_t  hypredrv_compression_from_filename(const char *filename);

void hypredrv_compress(comp_alg_t algo, size_t isize, const void *input,
                       size_t *osize_ptr, void **output_ptr);
void hypredrv_decompress(comp_alg_t algo, size_t isize, const void *input,
                         size_t *osize_ptr, void **output_ptr);

#endif /* COMP_HEADER */
