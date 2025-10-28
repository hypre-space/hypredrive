/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef EIGSPEC_HEADER
#define EIGSPEC_HEADER

#include "yaml.h"
#include "field.h"
#include "containers.h"

/*--------------------------------------------------------------------------
 * Eigenspectrum arguments struct
 *--------------------------------------------------------------------------*/

typedef struct EigSpec_args_struct {
   HYPRE_Int  enable;           /* 0: off, 1: on */
   HYPRE_Int  vectors;          /* 0: values only, 1: values+vectors */
   HYPRE_Int  hermitian;        /* 0: general, 1: symmetric/Hermitian */
   HYPRE_Int  preconditioned;   /* 0: A, 1: M^{-1}A */
   char       output_prefix[MAX_FILENAME_LENGTH];
} EigSpec_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void EigSpecSetDefaultArgs(EigSpec_args*);
void EigSpecSetArgs(void*, YAMLnode*);

/* Internal helpers */
#if defined(HYPREDRV_ENABLE_EIGSPEC)
typedef void (*hypredrv_PreconApplyFn)(void *ctx, void *b, void *x);
uint32_t hypredrv_EigSpecCompute(const EigSpec_args *eargs,
                                 void *mat_A,
                                 void *precon_ctx,
                                 hypredrv_PreconApplyFn precon_apply);
#endif

#endif /* EIGSPEC_HEADER */
