/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "field.h"

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeIntSet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeIntSet(void *field, const YAMLnode *node)
{
   sscanf(node->mapped_val, "%d", (int *)field);
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeIntArraySet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeIntArraySet(void *field, const YAMLnode *node)
{
   IntArray *int_array = NULL;

   hypredrv_StrToIntArray(node->mapped_val, &int_array);

   *((void **)field) = int_array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeStackIntArraySet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeStackIntArraySet(void *field, const YAMLnode *node)
{
   StackIntArray *int_array = ((StackIntArray *)field);

   hypredrv_StrToStackIntArray(node->mapped_val, int_array);
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeDoubleSet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeDoubleSet(void *field, const YAMLnode *node)
{
   sscanf(node->mapped_val, "%lf", (double *)field);
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeCharSet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeCharSet(void *field, const YAMLnode *node)
{
   sscanf(node->mapped_val, "%c", (char *)field);
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeStringSet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeStringSet(void *field, const YAMLnode *node)
{
   snprintf((char *)field, MAX_FILENAME_LENGTH, "%s", node->mapped_val);
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeDoubleArraySet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeDoubleArraySet(void *field, const YAMLnode *node)
{
   DoubleArray *double_array = NULL;

   hypredrv_StrToDoubleArray(node->mapped_val, &double_array);

   *((void **)field) = double_array;
}
