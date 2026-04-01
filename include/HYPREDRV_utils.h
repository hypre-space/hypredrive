/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_UTILS_HEADER
#define HYPREDRV_UTILS_HEADER

/**
 * @file HYPREDRV_utils.h
 *
 * @brief Optional utility macros for HYPREDRV.
 *
 * Include this header after HYPREDRV.h when you want to use the
 * HYPREDRV_SAFE_CALL or HYPREDRV_SAFE_CALL_COMM convenience macros.
 * These macros are kept in a separate header to avoid leaking system
 * symbols (strcmp, getenv, raise, ...) into translation units that only
 * need the public HYPREDRV API types and function declarations.
 *
 * @code
 *   #include <HYPREDRV.h>
 *   #include <HYPREDRV_utils.h>   // for HYPREDRV_SAFE_CALL
 * @endcode
 */

#include <mpi.h>
#include <signal.h> // For: raise
#include <stdio.h>  // For: fprintf
#include <stdlib.h> // For: getenv
#include <string.h> // For: strcmp

#include "HYPREDRV.h"

static inline void
hypredrv_SafeCallHandleError(uint32_t error_code, MPI_Comm comm, const char *file,
                             int line, const char *func)
{
   /* GCOVR_EXCL_BR_START */
   if (error_code != 0)
   {
      (void)fprintf(stderr, "At %s:%d in %s():\n", file, line, func);
      HYPREDRV_ErrorCodeDescribe(error_code);
      const char *debug_env = getenv("HYPREDRV_DEBUG");
      if (debug_env && strcmp(debug_env, "1") == 0)
      {
         raise(SIGTRAP); /* Breakpoint for gdb */
      }
      else
      {
         MPI_Abort(comm, (int)error_code);
      }
   }
   /* GCOVR_EXCL_BR_STOP */
}

/**
 * @brief Safely call a HYPREDRV function; abort on error via MPI_COMM_WORLD.
 *
 * On error this macro prints the source location, calls
 * HYPREDRV_ErrorCodeDescribe() to print the error message, then either
 * raises SIGTRAP (when @c HYPREDRV_DEBUG=1 is set in the environment, for
 * use with a debugger) or calls @c MPI_Abort(MPI_COMM_WORLD, error_code).
 *
 * @note Uses @c MPI_COMM_WORLD as the communicator for @c MPI_Abort. If
 * your application runs on a sub-communicator and you need the abort to
 * target a specific communicator, use HYPREDRV_SAFE_CALL_COMM() instead.
 *
 * @code
 *   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
 * @endcode
 */
#ifndef HYPREDRV_SAFE_CALL
#define HYPREDRV_SAFE_CALL(call)                                               \
   do                                                                          \
   {                                                                           \
      hypredrv_SafeCallHandleError((call), MPI_COMM_WORLD, __FILE__, __LINE__, \
                                   __func__);                                  \
   } while (0)
#endif

/**
 * @brief Safely call a HYPREDRV function; abort on error via a specific MPI
 * communicator.
 *
 * Identical to HYPREDRV_SAFE_CALL() but calls @c MPI_Abort(@p comm, ...) so
 * that the correct process group is aborted when the application uses a
 * communicator other than @c MPI_COMM_WORLD.
 *
 * @param comm An @c MPI_Comm handle to use for @c MPI_Abort.
 * @param call The HYPREDRV function call to execute.
 *
 * @code
 *   MPI_Comm my_comm = ...;
 *   HYPREDRV_SAFE_CALL_COMM(my_comm, HYPREDRV_Initialize());
 * @endcode
 */
#ifndef HYPREDRV_SAFE_CALL_COMM
#define HYPREDRV_SAFE_CALL_COMM(comm, call)                                       \
   do                                                                             \
   {                                                                              \
      hypredrv_SafeCallHandleError((call), (comm), __FILE__, __LINE__, __func__); \
   } while (0)
#endif

#endif /* ifndef HYPREDRV_UTILS_HEADER */
