/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "cheby.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define Cheby_FIELDS(_prefix)                                \
   ADD_FIELD_OFFSET_ENTRY(_prefix, order, FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, eig_est, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, variant, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, scale, FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fraction, FieldTypeDoubleSet)

/* Define num_fields macro */
#define Cheby_NUM_FIELDS \
   (sizeof(Cheby_field_offset_map) / sizeof(Cheby_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS(Cheby)      // LCOV_EXCL_LINE
DEFINE_VOID_GET_VALID_VALUES_FUNC(Cheby) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * ChebySetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
ChebySetDefaultArgs(Cheby_args *args)
{
   args->order    = 2;
   args->eig_est  = 10;
   args->variant  = 0;
   args->scale    = 1;
   args->fraction = 0.3;
}
