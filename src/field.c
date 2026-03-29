/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/field.h"

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
   const char *src = (node && node->mapped_val) ? node->mapped_val : "";
   int         n   = snprintf((char *)field, MAX_FILENAME_LENGTH, "%s", src);
   if (n < 0)
   {
      ((char *)field)[0] = '\0';
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Failed to parse string value for key '%s'",
                           node ? node->key : "<unknown>");
      return;
   }
   if ((size_t)n >= MAX_FILENAME_LENGTH)
   {
      ((char *)field)[0] = '\0';
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Value for key '%s' exceeds %d characters",
                           node ? node->key : "<unknown>", MAX_FILENAME_LENGTH - 1);
   }
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

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeNoopSet
 *
 * No-op setter used for keys whose parsing is handled via special-case branches
 * in the caller (e.g. dof_labels in linear_system).  The entry in the field
 * offset map only exists so the validator accepts the key.
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeNoopSet(void *field, const YAMLnode *node)
{
   (void)field;
   (void)node;
}
