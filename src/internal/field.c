/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/field.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeIntSet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeIntSet(void *field, const YAMLnode *node)
{
   const char *src = (node && node->mapped_val) ? node->mapped_val : "";
   char       *end = NULL;
   long        val;

   errno = 0;
   val   = strtol(src, &end, 10);
   /* Reject empty/garbage input and out-of-range values instead of invoking the
    * undefined behavior of sscanf("%d") on an unrepresentable value. */
   if (end == src || *end != '\0' || errno == ERANGE || val < INT_MIN || val > INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid integer value '%s' for key '%s'", src,
                           node ? node->key : "<unknown>");
      return;
   }
   *((int *)field) = (int)val;
}

/*-----------------------------------------------------------------------------
 * hypredrv_FieldTypeIntArraySet
 *-----------------------------------------------------------------------------*/

void
hypredrv_FieldTypeIntArraySet(void *field, const YAMLnode *node)
{
   IntArray *int_array = NULL;

   hypredrv_StrToIntArray(node->mapped_val, &int_array);

   /* Free any array already stored in this field before overwriting it, so setting
    * the same key twice (e.g. file value plus a CLI override) does not leak. */
   IntArray *existing = *((IntArray **)field);
   if (existing)
   {
      hypredrv_IntArrayDestroy(&existing);
   }
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
   const char *src = (node && node->mapped_val) ? node->mapped_val : "";
   char       *end = NULL;
   double      val;

   errno = 0;
   val   = strtod(src, &end);
   if (end == src || *end != '\0')
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid floating-point value '%s' for key '%s'", src,
                           node ? node->key : "<unknown>");
      return;
   }
   *((double *)field) = val;
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

   /* Free any array already stored in this field before overwriting it. */
   DoubleArray *existing = *((DoubleArray **)field);
   if (existing)
   {
      hypredrv_DoubleArrayDestroy(&existing);
   }
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
