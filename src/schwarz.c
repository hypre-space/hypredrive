/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/compatibility.h"

#if HYPREDRV_HAVE_EXPERIMENTAL

#include "HYPRE_parcsr_ls.h"
#include "internal/gen_macros.h"
#include "internal/schwarz.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define Schwarz_FIELDS(_prefix)                                                  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, variant, hypredrv_FieldTypeIntSet)            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, overlap, hypredrv_FieldTypeIntSet)            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, domain_type, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_functions, hypredrv_FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, use_nonsymm, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, local_solver_type, hypredrv_FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, iluk_level_of_fill, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilut_max_nnz_row, hypredrv_FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, hypredrv_FieldTypeIntSet)           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, logging, hypredrv_FieldTypeIntSet)            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relax_weight, hypredrv_FieldTypeDoubleSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilut_droptol, hypredrv_FieldTypeDoubleSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, hypredrv_FieldTypeDoubleSet)

#define Schwarz_NUM_FIELDS \
   (sizeof(Schwarz_field_offset_map) / sizeof(Schwarz_field_offset_map[0]))

GENERATE_PREFIXED_COMPONENTS(Schwarz) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * SchwarzGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_SchwarzGetValidValues(const char *key)
{
   if (!strcmp(key, "variant"))
   {
      static StrIntMap map[] = {
         {"mp", 0},           {"ad", 1},        {"par-ad", 2},   {"par-mp", 3},
         {"mp-fw", 4},        {"ras-iluk", 10}, {"as-iluk", 11}, {"ras-ilut", 20},
         {"as-ilut", 21},     {"ras-amg", 30},  {"as-amg", 31},  {"ras-spdirect", 40},
         {"as-spdirect", 41},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "local_solver_type"))
   {
      static StrIntMap map[] = {
         {"iluk", 0}, {"ilut", 1}, {"amg", 2}, {"spdirect", 3}, {"superlu", 3},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * SchwarzSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_SchwarzSetDefaultArgs(Schwarz_args *args)
{
   args->variant            = 10;
   args->overlap            = 1;
   args->domain_type        = 2;
   args->num_functions      = 1;
   args->use_nonsymm        = 0;
   args->local_solver_type  = 0;
   args->iluk_level_of_fill = 0;
   args->ilut_max_nnz_row   = 1000;
   args->max_iter           = 1;
   args->print_level        = 0;
   args->logging            = 0;
   args->relax_weight       = 1.0;
   args->ilut_droptol       = 1.0e-2;
   args->tolerance          = 0.0;
}

/*-----------------------------------------------------------------------------
 * hypredrv_SchwarzCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_SchwarzCreate(const Schwarz_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon = NULL;

   HYPRE_SchwarzCreate(&precon);
   HYPRE_SchwarzSetVariant(precon, args->variant);
   HYPRE_SchwarzSetOverlap(precon, args->overlap);
   HYPRE_SchwarzSetDomainType(precon, args->domain_type);
   HYPRE_SchwarzSetRelaxWeight(precon, args->relax_weight);
   HYPRE_SchwarzSetNumFunctions(precon, args->num_functions);
   HYPRE_SchwarzSetNonSymm(precon, args->use_nonsymm);
   HYPRE_SchwarzSetLocalSolverType(precon, args->local_solver_type);
   HYPRE_SchwarzSetILUKLevelOfFill(precon, args->iluk_level_of_fill);
   HYPRE_SchwarzSetILUTMaxNnzPerRow(precon, args->ilut_max_nnz_row);
   HYPRE_SchwarzSetILUTDroptol(precon, args->ilut_droptol);
   HYPRE_SchwarzSetMaxIter(precon, args->max_iter);
   HYPRE_SchwarzSetTol(precon, args->tolerance);
   HYPRE_SchwarzSetPrintLevel(precon, args->print_level);
   HYPRE_SchwarzSetLogging(precon, args->logging);

   *precon_ptr = precon;
}

#endif /* HYPREDRV_HAVE_EXPERIMENTAL */
