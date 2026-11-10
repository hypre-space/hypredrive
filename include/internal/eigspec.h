/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef EIGSPEC_HEADER
#define EIGSPEC_HEADER

#include <stdint.h>
#include "HYPRE_utilities.h"
#include "internal/containers.h"
#include "internal/object_fwd.h"

struct YAMLnode_struct;

/*--------------------------------------------------------------------------
 * Eigenspectrum arguments struct
 *--------------------------------------------------------------------------*/

typedef struct EigSpec_args_struct
{
   HYPRE_Int enable;         /* 0: off, 1: on */
   HYPRE_Int vectors;        /* 0: values only, 1: values+vectors */
   HYPRE_Int hermitian;      /* 0: general, 1: symmetric/Hermitian */
   HYPRE_Int preconditioned; /* 0: A, 1: M^{-1}A */
   char      output_prefix[MAX_FILENAME_LENGTH];
} EigSpec_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_EigSpecSetDefaultArgs(EigSpec_args *);
void hypredrv_EigSpecSetArgs(void *, struct YAMLnode_struct *);

/* The public entry point supplies a live object and, in enabled builds,
 * parsed arguments. This operation owns its work buffers and diagnostics. */
uint32_t hypredrv_EigSpecCompute(hypredrv_t *);

#endif /* EIGSPEC_HEADER */
