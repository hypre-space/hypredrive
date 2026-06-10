/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/cheby.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define Cheby_FIELDS(_X, _p)                     \
   _X(_p, order, hypredrv_FieldTypeIntSet, 2)    \
   _X(_p, eig_est, hypredrv_FieldTypeIntSet, 10) \
   _X(_p, variant, hypredrv_FieldTypeIntSet, 0)  \
   _X(_p, scale, hypredrv_FieldTypeIntSet, 1)    \
   _X(_p, fraction, hypredrv_FieldTypeDoubleSet, 0.3)

/* Define num_fields macro */
#define Cheby_NUM_FIELDS \
   (sizeof(Cheby_field_offset_map) / sizeof(Cheby_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(Cheby)          // LCOV_EXCL_LINE
hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC(hypredrv_Cheby) // LCOV_EXCL_LINE
