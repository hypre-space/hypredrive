/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef EIGSPEC_HEADER
#define EIGSPEC_HEADER

#include <stdint.h>
#include "HYPRE_utilities.h"
#include "internal/containers.h"

struct Stats_struct;
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
void hypredrv_EigSpecSetArgs(void *, const struct YAMLnode_struct *);

/* Internal helpers */
#ifdef HYPREDRV_ENABLE_EIGSPEC
typedef void (*hypredrv_PreconApplyFn)(void *ctx, void *b, void *x);
uint32_t hypredrv_EigSpecCompute(const EigSpec_args *, void *, void *,
                                 hypredrv_PreconApplyFn, struct Stats_struct *);
#endif

#endif /* EIGSPEC_HEADER */
