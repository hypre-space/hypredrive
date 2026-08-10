/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>

#include "HYPREDRV.h"

uint32_t
HYPREDRV_FortranTestCreateVector(MPI_Fint fcomm, int64_t row_start_in, int64_t row_end_in,
                                 HYPRE_Real value, HYPRE_IJVector *vec)
{
   HYPRE_BigInt row_start = (HYPRE_BigInt)row_start_in;
   HYPRE_BigInt row_end   = (HYPRE_BigInt)row_end_in;
   HYPRE_BigInt nrows     = row_end - row_start + 1;

   if (!vec || nrows <= 0 || (int64_t)row_start != row_start_in ||
       (int64_t)row_end != row_end_in)
   {
      return HYPREDRV_ErrorInvalidValue("invalid Fortran test vector range");
   }

   /* This helper calls HYPRE directly. When shared HYPREDRV embeds a static
    * HYPRE archive, those calls use a distinct HYPRE runtime from the one
    * initialized by HYPREDRV_Initialize(). Initializing is idempotent and also
    * covers builds where both interfaces resolve to the same shared HYPRE. */
   if (HYPRE_Initialize())
   {
      return HYPREDRV_ErrorInvalidValue(
         "failed to initialize Fortran test HYPRE runtime");
   }

   HYPRE_BigInt  *indices = (HYPRE_BigInt *)malloc((size_t)nrows * sizeof(*indices));
   HYPRE_Complex *values  = (HYPRE_Complex *)malloc((size_t)nrows * sizeof(*values));
   if (!indices || !values)
   {
      free(indices);
      free(values);
      return HYPREDRV_ErrorInvalidValue("failed to allocate Fortran test vector");
   }

   for (HYPRE_BigInt i = 0; i < nrows; i++)
   {
      indices[i] = row_start + i;
      values[i]  = (HYPRE_Complex)value;
   }

   uint32_t ierr =
      (uint32_t)HYPRE_IJVectorCreate(MPI_Comm_f2c(fcomm), row_start, row_end, vec);
   if (!ierr)
   {
      ierr = (uint32_t)HYPRE_IJVectorSetObjectType(*vec, HYPRE_PARCSR);
   }
   if (!ierr)
   {
      ierr = (uint32_t)HYPRE_IJVectorInitialize(*vec);
   }
   if (!ierr)
   {
      ierr = (uint32_t)HYPRE_IJVectorSetValues(*vec, (HYPRE_Int)nrows, indices, values);
   }
   if (!ierr)
   {
      ierr = (uint32_t)HYPRE_IJVectorAssemble(*vec);
   }

   free(indices);
   free(values);

   if (ierr)
   {
      if (*vec)
      {
         (void)HYPRE_IJVectorDestroy(*vec);
         *vec = NULL;
      }
      return HYPREDRV_ErrorInvalidValue("failed to build Fortran test vector");
   }

   return HYPREDRV_SUCCESS;
}

uint32_t
HYPREDRV_FortranTestDestroyVector(HYPRE_IJVector vec)
{
   if (!vec)
   {
      return HYPREDRV_SUCCESS;
   }
   return (uint32_t)HYPRE_IJVectorDestroy(vec);
}
