/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/* When built with HYPREDRV_COVERAGE_TESTS, runs before main() so unit tests
 * exercise logging branches without editing every test file. */

#ifdef HYPREDRV_COVERAGE_TESTS

#include <stdlib.h>

#include "logging.h"

__attribute__((constructor))
static void
hypredrv_tests_coverage_log_init(void)
{
   /* Enable verbose logging for coverage; tests may still override via
    * hypredrv_LogInitializeFromEnv / hypredrv_LogReset as needed. */
   if (setenv("HYPREDRV_LOG_LEVEL", "3", 1) == 0)
   {
      hypredrv_LogInitializeFromEnv();
   }
}

#endif /* HYPREDRV_COVERAGE_TESTS */
