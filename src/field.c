/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "field.h"

/*-----------------------------------------------------------------------------
 * FieldTypeIntSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeIntSet(void *field, YAMLnode *node)
{
   sscanf(node->mapped_val, "%d", (int*) field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeIntArraySet
 *-----------------------------------------------------------------------------*/

void
FieldTypeIntArraySet(void *field, YAMLnode *node)
{
   IntArray *int_array;

   StrToIntArray(node->mapped_val, &int_array);

   *((void**)field) = int_array;
}

/*-----------------------------------------------------------------------------
 * FieldTypeDoubleSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeDoubleSet(void *field, YAMLnode *node)
{
   sscanf(node->mapped_val, "%lf", (double*) field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeCharSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeCharSet(void *field, YAMLnode *node)
{
   sscanf(node->mapped_val, "%c", (char*) field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeStringSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeStringSet(void *field, YAMLnode *node)
{
   snprintf((char*) field, MAX_FILENAME_LENGTH, "%s", node->mapped_val);
}
