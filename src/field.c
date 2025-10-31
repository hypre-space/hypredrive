/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "field.h"

/*-----------------------------------------------------------------------------
 * FieldTypeIntSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeIntSet(void *field, YAMLnode *node)
{
   sscanf(node->mapped_val, "%d", (int *)field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeIntArraySet
 *-----------------------------------------------------------------------------*/

void
FieldTypeIntArraySet(void *field, YAMLnode *node)
{
   IntArray *int_array;

   StrToIntArray(node->mapped_val, &int_array);

   *((void **)field) = int_array;
}

/*-----------------------------------------------------------------------------
 * FieldTypeStackIntArraySet
 *-----------------------------------------------------------------------------*/

void
FieldTypeStackIntArraySet(void *field, YAMLnode *node)
{
   StackIntArray *int_array = ((StackIntArray *)field);

   StrToStackIntArray(node->mapped_val, int_array);
}

/*-----------------------------------------------------------------------------
 * FieldTypeDoubleSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeDoubleSet(void *field, YAMLnode *node)
{
   sscanf(node->mapped_val, "%lf", (double *)field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeCharSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeCharSet(void *field, YAMLnode *node)
{
   sscanf(node->mapped_val, "%c", (char *)field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeStringSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeStringSet(void *field, YAMLnode *node)
{
   snprintf((char *)field, MAX_FILENAME_LENGTH, "%s", node->mapped_val);
}
