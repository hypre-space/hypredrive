/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_FUZZ_COMMON_HEADER
#define HYPREDRV_FUZZ_COMMON_HEADER

#include <stddef.h>
#include <stdint.h>

#include <mpi.h>

#define FUZZ_MAX_INPUT (64u * 1024u)

typedef enum
{
   FUZZ_MODE_PARSE  = 1,
   FUZZ_MODE_SOLVE  = 2,
   FUZZ_MODE_LSSEQ  = 3,
   FUZZ_MODE_MATRIX = 4,
   FUZZ_MODE_VECTOR = 5
} hypredrv_FuzzMode;

int  fuzz_one(const uint8_t *data, size_t size);
void fuzz_mpi_init_once(int *argc, char ***argv);
void fuzz_runtime_mode_override(void);
void fuzz_require_startup_assets(void);
void fuzz_reset_error_state(void);
int  fuzz_current_mode(void);

int fuzz_write_temp_file(const uint8_t *data, size_t size, const char *suffix, char *path,
                         size_t path_size);
int fuzz_make_temp_prefix(char *prefix, size_t prefix_size);
int fuzz_read_file(const char *path, uint8_t **data_ptr, size_t *size_ptr);

#endif /* HYPREDRV_FUZZ_COMMON_HEADER */
