/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "hypredrive.h"
#include "args.h"
#include "linsys.h"
#include "info.h"

/*-----------------------------------------------------------------------------
 * hypredrv_t data type
 *-----------------------------------------------------------------------------*/

typedef struct hypredrv_struct {
   MPI_Comm         comm;
   input_args      *iargs;

   IntArray        *dofmap;
   HYPRE_IJMatrix   mat_A;
   HYPRE_IJMatrix   mat_M;
   HYPRE_IJVector   vec_b;
   HYPRE_IJVector   vec_x;
   HYPRE_IJVector   vec_x0;

   HYPRE_Solver     precon;
   HYPRE_Solver     solver;
} hypredrv_t;

/*-----------------------------------------------------------------------------
 * HYPREDRV_Create
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *obj_ptr)
{
   HYPREDRV_t obj = (HYPREDRV_t) malloc(sizeof(hypredrv_t));

   obj->comm = comm;
   *obj_ptr  = obj;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Destroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *obj_ptr)
{
   HYPREDRV_t obj = *obj_ptr;

   if (obj)
   {
      if (obj->mat_A != obj->mat_M)
      {
         HYPRE_IJMatrixDestroy(obj->mat_M);
      }
      HYPRE_IJMatrixDestroy(obj->mat_A);
      HYPRE_IJVectorDestroy(obj->vec_b);
      HYPRE_IJVectorDestroy(obj->vec_x);
      HYPRE_IJVectorDestroy(obj->vec_x0);

      InputArgsDestroy(&obj->iargs);

      free(*obj_ptr);
      *obj_ptr = NULL;
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintLibInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintLibInfo(void)
{
   PrintLibInfo();

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintExitInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintExitInfo(const char *argv0)
{
   PrintExitInfo(argv0);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsParse
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t obj)
{
   if (obj)
   {
      InputArgsParse(obj->comm, argc, argv, &obj->iargs);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetExecPolicy
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetExecPolicy(HYPREDRV_t obj)
{
   return (obj) ? obj->iargs->ls.exec_policy : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetWarmup
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t obj)
{
   return (obj) ? obj->iargs->warmup : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumRepetitions
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t obj)
{
   return (obj) ? obj->iargs->num_repetitions : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemReadMatrix(obj->comm, &obj->iargs->ls, &obj->mat_A);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemSetRHS(obj->comm, &obj->iargs->ls, obj->mat_A, &obj->vec_b);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemSetInitialGuess(obj->comm, &obj->iargs->ls, obj->mat_A, obj->vec_b,
                                  &obj->vec_x0, &obj->vec_x);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemResetInitialGuess(obj->vec_x0, obj->vec_x);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemSetPrecMatrix(obj->comm, &obj->iargs->ls, obj->mat_A, &obj->mat_M);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemReadDofmap(obj->comm, &obj->iargs->ls, &obj->dofmap);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t obj)
{
   if (obj)
   {
      PreconCreate(obj->iargs->precon_method, &obj->iargs->precon, obj->dofmap, &obj->precon);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t obj)
{
   if (obj)
   {
      SolverCreate(obj->comm, obj->iargs->solver_method, &obj->iargs->solver, &obj->solver);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverSetup
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t obj)
{
   if (obj)
   {
      SolverSetup(obj->iargs->precon_method, obj->iargs->solver_method,
                  obj->precon, obj->solver, obj->mat_M, obj->vec_b, obj->vec_x);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverApply
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t obj)
{
   if (obj)
   {
      SolverApply(obj->iargs->solver_method, obj->solver, obj->mat_A,
                  obj->vec_b, obj->vec_x);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t obj)
{
   if (obj)
   {
      PreconDestroy(obj->iargs->precon_method, &obj->precon);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t obj)
{
   if (obj)
   {
      SolverDestroy(obj->iargs->solver_method, &obj->solver);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsPrint
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t obj)
{
   if (obj)
   {
      StatsPrint(obj->iargs->statistics);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}
