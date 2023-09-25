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
   sscanf(node->val, "%d", (int*) field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeDoubleSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeDoubleSet(void *field, YAMLnode *node)
{
   sscanf(node->val, "%lf", (double*) field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeCharSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeCharSet(void *field, YAMLnode *node)
{
   sscanf(node->val, "%c", (char*) field);
}

/*-----------------------------------------------------------------------------
 * FieldTypeStringSet
 *-----------------------------------------------------------------------------*/

void
FieldTypeStringSet(void *field, YAMLnode *node)
{
   snprintf((char*) field, MAX_FILENAME_LENGTH, "%s", node->val);
}
