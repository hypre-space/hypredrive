/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/* Helpers for numerical assertions on caller-owned host solution values. */
#ifndef TEST_SOLUTION_HELPERS_HEADER
#define TEST_SOLUTION_HELPERS_HEADER

#include "HYPREDRV.h"
#include "test_helpers.h"

static HYPRE_Complex *test_solution_host_values;
static size_t         test_solution_host_capacity;

static void
test_solution_host_cleanup(void)
{
   free(test_solution_host_values);
}

static uint32_t
test_copy_solution_values(HYPREDRV_t obj, HYPRE_Complex **values)
{
   HYPRE_BigInt count = 0;
   uint32_t     code  = HYPREDRV_LinearSystemGetSolutionLength(obj, &count);
   if (code) return code;
   ASSERT_TRUE(count >= 0 && (uintmax_t)count <= SIZE_MAX / sizeof(**values));
   if ((size_t)count > test_solution_host_capacity)
   {
      if (!test_solution_host_capacity) ASSERT_EQ(atexit(test_solution_host_cleanup), 0);
      void *buffer = realloc(test_solution_host_values, (size_t)count * sizeof(**values));
      ASSERT_NOT_NULL(buffer);
      test_solution_host_values   = buffer;
      test_solution_host_capacity = (size_t)count;
   }
   *values = test_solution_host_values;
   return HYPREDRV_LinearSystemCopySolutionValues(obj, count, *values);
}

#endif
