/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef LINSYS_HEADER
#define LINSYS_HEADER

#include "yaml.h"
#include "utils.h"
#include "field.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"

/*--------------------------------------------------------------------------
 * Linear system arguments struct
 *--------------------------------------------------------------------------*/

typedef struct LS_args_struct {
   char          matrix_filename[MAX_FILENAME_LENGTH];
   char          precmat_filename[MAX_FILENAME_LENGTH];
   char          rhs_filename[MAX_FILENAME_LENGTH];
   char          x0_filename[MAX_FILENAME_LENGTH];
   char          sol_filename[MAX_FILENAME_LENGTH];
   char          dofmap_filename[MAX_FILENAME_LENGTH];
   HYPRE_Int     init_guess_mode;
   HYPRE_Int     rhs_mode;
   HYPRE_Int     type;
} LS_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

StrArray LinearSystemGetValidKeys(void);
StrIntMapArray LinearSystemGetValidValues(const char*);

void LinearSystemSetDefaultArgs(LS_args*);
void LinearSystemSetArgsFromYAML(LS_args*, YAMLnode*);
int LinearSystemReadMatrix(MPI_Comm, LS_args*, HYPRE_IJMatrix*);
int LinearSystemSetRHS(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJVector*);
int LinearSystemSetInitialGuess(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector*);
int LinearSystemSetPrecMatrix(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJMatrix*);
int LinearSystemReadDofmap(MPI_Comm, LS_args*, HYPRE_IntArray*);

#endif
