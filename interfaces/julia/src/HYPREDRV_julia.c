/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "internal/bridge_row_bounds.h"

#include <mpi.h>

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*HYPREDRV_JuliaFunction)(void);

HYPREDRV_JuliaFunction
HYPREDRV_JuliaGetSymbol(const char *name)
{
#define HYPREDRV_JULIA_SYMBOL(api)          \
   if (strcmp(name, #api) == 0)             \
   {                                        \
      return (HYPREDRV_JuliaFunction)(api); \
   }
#include "HYPREDRV_julia_symbols.inc"
#undef HYPREDRV_JULIA_SYMBOL
   return NULL;
}

/* Process-global MPI ownership matches the current Python bridge policy. It is
 * suitable for one Julia runtime per process; do not mix independent MPI owners
 * inside the same process.
 */
static int hypredrv_julia_mpi_owned = 0;

_Static_assert(sizeof(HYPRE_BigInt) == 4u || sizeof(HYPRE_BigInt) == 8u,
               "Julia bridge supports only 32-bit or 64-bit HYPRE_BigInt");
_Static_assert(sizeof(HYPRE_Real) == sizeof(double),
               "Julia bridge requires double-precision HYPRE_Real");
_Static_assert(sizeof(HYPRE_Complex) == sizeof(double),
               "Julia bridge does not support complex HYPRE builds");

static uint32_t
HYPREDRV_JuliaBigIntFromI64(int64_t value, const char *name, HYPRE_BigInt *out)
{
   if (!hypredrv_BigIntFromI64(value, out))
   {
      char msg[256];
      int  nwritten = snprintf(
         msg, sizeof(msg), "%s=%" PRId64 " is outside HYPRE_BigInt range", name, value);
      if (nwritten < 0 || (size_t)nwritten >= sizeof(msg))
      {
         return HYPREDRV_ErrorInvalidValue("integer value is outside HYPRE_BigInt range");
      }
      return HYPREDRV_ErrorInvalidValue(msg);
   }

   return HYPREDRV_SUCCESS;
}

/** @brief Initialize MPI if Julia/HYPREDRV owns MPI startup. */
uint32_t
HYPREDRV_JuliaMPIInitialize(void)
{
   int finalized = 0;
   int ierr      = MPI_Finalized(&finalized);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Finalized failed");
   }
   if (finalized)
   {
      return HYPREDRV_ErrorInvalidValue("MPI has already been finalized");
   }

   int initialized = 0;
   ierr            = MPI_Initialized(&initialized);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Initialized failed");
   }
   if (!initialized)
   {
      int provided = 0;
      ierr         = MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
      if (ierr != MPI_SUCCESS)
      {
         return HYPREDRV_ErrorInvalidValue("MPI_Init_thread failed");
      }
      hypredrv_julia_mpi_owned = 1;
   }
   else
   {
      hypredrv_julia_mpi_owned = 0;
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Finalize MPI only if this Julia bridge initialized MPI. */
uint32_t
HYPREDRV_JuliaMPIFinalize(void)
{
   int finalized = 0;
   int ierr      = MPI_Finalized(&finalized);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Finalized failed");
   }
   if (hypredrv_julia_mpi_owned && !finalized)
   {
      ierr = MPI_Finalize();
      if (ierr != MPI_SUCCESS)
      {
         return HYPREDRV_ErrorInvalidValue("MPI_Finalize failed");
      }
      hypredrv_julia_mpi_owned = 0;
   }
   if (finalized)
   {
      hypredrv_julia_mpi_owned = 0;
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Return the byte width of HYPRE_BigInt in this HYPRE build. */
size_t
HYPREDRV_JuliaBigIntSize(void)
{
   return sizeof(HYPRE_BigInt);
}

/** @brief Return the byte width of HYPRE_Real in this HYPRE build. */
size_t
HYPREDRV_JuliaRealSize(void)
{
   return sizeof(HYPRE_Real);
}

/** @brief Return the byte width of HYPRE_Complex solution entries. */
size_t
HYPREDRV_JuliaSolutionEntrySize(void)
{
   return sizeof(HYPRE_Complex);
}

/** @brief Create a HYPREDRV handle on MPI_COMM_SELF. */
uint32_t
HYPREDRV_JuliaCreateWithSelf(HYPREDRV_t *hypredrv_ptr)
{
   if (hypredrv_ptr == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV output pointer is NULL");
   }
   return HYPREDRV_Create(MPI_COMM_SELF, hypredrv_ptr);
}

/** @brief Create a HYPREDRV handle on MPI_COMM_WORLD. */
uint32_t
HYPREDRV_JuliaCreateWithWorld(HYPREDRV_t *hypredrv_ptr)
{
   if (hypredrv_ptr == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV output pointer is NULL");
   }
   return HYPREDRV_Create(MPI_COMM_WORLD, hypredrv_ptr);
}

/** @brief Return the MPI_COMM_WORLD rank. */
uint32_t
HYPREDRV_JuliaWorldCommRank(int *rank)
{
   if (rank == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI rank output pointer is NULL");
   }
   int ierr = MPI_Comm_rank(MPI_COMM_WORLD, rank);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Comm_rank failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Return the MPI_COMM_WORLD size. */
uint32_t
HYPREDRV_JuliaWorldCommSize(int *size)
{
   if (size == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI size output pointer is NULL");
   }
   int ierr = MPI_Comm_size(MPI_COMM_WORLD, size);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Comm_size failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Allreduce a double value with SUM over MPI_COMM_WORLD. */
uint32_t
HYPREDRV_JuliaWorldAllreduceDoubleSum(double value, double *sum)
{
   if (sum == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI allreduce output pointer is NULL");
   }
   int ierr = MPI_Allreduce(&value, sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Allreduce failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Gather variable-length double arrays on MPI_COMM_WORLD. */
uint32_t
HYPREDRV_JuliaWorldAllgathervDouble(const double *send, int send_count, double *recv,
                                    const int *counts, const int *displs)
{
   if (send_count < 0)
   {
      return HYPREDRV_ErrorInvalidValue("send_count is negative");
   }
   if ((send_count > 0 && send == NULL) || recv == NULL || counts == NULL ||
       displs == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI allgatherv pointer is NULL");
   }

   int ierr = MPI_Allgatherv(send, send_count, MPI_DOUBLE, recv, counts, displs,
                             MPI_DOUBLE, MPI_COMM_WORLD);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Allgatherv failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Destroy a HYPREDRV handle. */
uint32_t
HYPREDRV_JuliaDestroy(HYPREDRV_t *hypredrv_ptr)
{
   if (hypredrv_ptr == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV pointer is NULL");
   }
   if (*hypredrv_ptr == NULL)
   {
      return HYPREDRV_SUCCESS;
   }
   return HYPREDRV_Destroy(hypredrv_ptr);
}

/** @brief Parse in-memory YAML text into the HYPREDRV input-argument state. */
uint32_t
HYPREDRV_JuliaInputArgsParseYaml(HYPREDRV_t hypredrv, const char *yaml_text)
{
   if (yaml_text == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("YAML text pointer is NULL");
   }

   char *argv[1] = {(char *)yaml_text};
   return HYPREDRV_InputArgsParse(1, argv, hypredrv);
}

/** @brief Parse a full HYPREDRV argv vector for Julia callers. */
uint32_t
HYPREDRV_JuliaInputArgsParseArgv(HYPREDRV_t hypredrv, int argc, const char **argv)
{
   if (argc <= 0)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV argv is empty");
   }
   if (argv == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV argv pointer is NULL");
   }

   /* Heap-allocate rather than using a caller-sized VLA: VLAs are optional in C11
    * (unsupported by MSVC) and a large argc would overflow the stack. */
   char **parse_argv = (char **)malloc((size_t)argc * sizeof(char *));
   if (!parse_argv)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV failed to allocate argv buffer");
   }
   for (int i = 0; i < argc; i++)
   {
      if (argv[i] == NULL)
      {
         free(parse_argv);
         return HYPREDRV_ErrorInvalidValue("HYPREDRV argv entry is NULL");
      }
      parse_argv[i] = (char *)argv[i];
   }

   uint32_t code = HYPREDRV_InputArgsParse(argc, parse_argv, hypredrv);
   free(parse_argv);
   return code;
}

/** @brief Install a CSR matrix slab from int64 row bounds and HYPRE_BigInt arrays. */
uint32_t
HYPREDRV_JuliaSetMatrixFromCSR(HYPREDRV_t hypredrv, int64_t row_start_i64,
                               int64_t row_end_i64, const void *indptr,
                               const void *col_indices, const void *data)
{
   if (indptr == NULL || col_indices == NULL || data == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("CSR input pointer is NULL");
   }

   HYPRE_BigInt row_start = 0;
   HYPRE_BigInt row_end   = 0;
   uint32_t code = hypredrv_BridgeRowBoundsFromI64(row_start_i64, row_end_i64, &row_start,
                                                   &row_end, HYPREDRV_JuliaBigIntFromI64);
   if (code != 0u)
   {
      return code;
   }

   /* Contract: callers must pass indptr/col_indices buffers whose element type
    * has exactly sizeof(HYPRE_BigInt), and data whose element type has exactly
    * sizeof(HYPRE_Real). The public Julia API enforces this before reaching the
    * bridge; these opaque casts keep the C ABI stable across HYPRE integer
    * widths.
    */
   code = HYPREDRV_AnnotateBegin(hypredrv, "matrix", -1);
   if (code != 0u)
   {
      return code;
   }
   code = HYPREDRV_LinearSystemSetMatrixFromCSR(
      hypredrv, row_start, row_end, (const HYPRE_BigInt *)indptr,
      (const HYPRE_BigInt *)col_indices, (const HYPRE_Real *)data);
   uint32_t end_code = HYPREDRV_AnnotateEnd(hypredrv, "matrix", -1);
   return code != 0u ? code : end_code;
}

/** @brief Install an RHS slab from int64 row bounds and HYPRE_Real values. */
uint32_t
HYPREDRV_JuliaSetRHSFromArray(HYPREDRV_t hypredrv, int64_t row_start_i64,
                              int64_t row_end_i64, const void *data)
{
   if (data == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("RHS data pointer is NULL");
   }

   HYPRE_BigInt row_start = 0;
   HYPRE_BigInt row_end   = 0;
   uint32_t code = hypredrv_BridgeRowBoundsFromI64(row_start_i64, row_end_i64, &row_start,
                                                   &row_end, HYPREDRV_JuliaBigIntFromI64);
   if (code != 0u)
   {
      return code;
   }

   /* Contract: callers must pass a data buffer whose element type has exactly
    * sizeof(HYPRE_Real). The public Julia API enforces double precision before
    * reaching the bridge.
    */
   return HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end,
                                               (const HYPRE_Real *)data);
}

/** @brief Copy local solution values into a caller-owned host array. */
uint32_t
HYPREDRV_JuliaCopySolutionValues(HYPREDRV_t hypredrv, void *sol_data, int64_t sol_length)
{
   HYPRE_BigInt count;
   uint32_t     code = HYPREDRV_JuliaBigIntFromI64(sol_length, "solution length", &count);
   if (code) return code;
   return HYPREDRV_LinearSystemCopySolutionValues(hypredrv, count,
                                                  (HYPRE_Complex *)sol_data);
}

/* ---------------------------------------------------------------------------
 * Additional shims covering the remaining public HYPREDRV API.
 *
 * Opaque HYPRE objects (HYPRE_Matrix, HYPRE_Vector, HYPRE_IJVector) cross the
 * ABI as void pointers. MPI_Comm arguments are selected with use_world:
 * nonzero maps to MPI_COMM_WORLD, zero to MPI_COMM_SELF.
 * ------------------------------------------------------------------------- */

static MPI_Comm
HYPREDRV_JuliaComm(int use_world)
{
   return use_world ? MPI_COMM_WORLD : MPI_COMM_SELF;
}

/** @brief Print library information. */
uint32_t
HYPREDRV_JuliaPrintLibInfo(int use_world, int print_datetime)
{
   return HYPREDRV_PrintLibInfo(HYPREDRV_JuliaComm(use_world), print_datetime);
}

/** @brief Print system information. */
uint32_t
HYPREDRV_JuliaPrintSystemInfo(int use_world)
{
   return HYPREDRV_PrintSystemInfo(HYPREDRV_JuliaComm(use_world));
}

/** @brief Print exit information. */
uint32_t
HYPREDRV_JuliaPrintExitInfo(int use_world, const char *argv0)
{
   return HYPREDRV_PrintExitInfo(HYPREDRV_JuliaComm(use_world), argv0);
}

/** @brief Borrow the native local solution pointer; is_host reports its location. */
uint32_t
HYPREDRV_JuliaLinearSystemGetSolutionValues(HYPREDRV_t hypredrv, double **data,
                                            int *is_host)
{
   if (data == NULL || is_host == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("solution values output pointer is NULL");
   }
   HYPRE_Complex       *ptr = NULL;
   HYPRE_MemoryLocation loc = HYPRE_MEMORY_HOST;
   uint32_t code = HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &ptr, &loc);
   if (code != 0u)
   {
      return code;
   }
   *data    = (double *)ptr;
   *is_host = (loc == HYPRE_MEMORY_HOST);
   return HYPREDRV_SUCCESS;
}

/** @brief Return the local solution length. */
uint32_t
HYPREDRV_JuliaLinearSystemGetSolutionLength(HYPREDRV_t hypredrv, int64_t *length)
{
   if (length == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("solution length output pointer is NULL");
   }
   HYPRE_BigInt n    = 0;
   uint32_t     code = HYPREDRV_LinearSystemGetSolutionLength(hypredrv, &n);
   if (code != 0u)
   {
      return code;
   }
   *length = (int64_t)n;
   return HYPREDRV_SUCCESS;
}

uint32_t HYPREDRV_JuliaVectorLocalLength(void *vec, int64_t *length);

/** @brief Return the local length of the RHS vector. */
uint32_t
HYPREDRV_JuliaLinearSystemGetRHSLength(HYPREDRV_t hypredrv, int64_t *length)
{
   if (length == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("RHS length output pointer is NULL");
   }
   HYPRE_Vector rhs  = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetRHS(hypredrv, &rhs);
   if (code != 0u)
   {
      return code;
   }
   if (rhs == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("no RHS vector available");
   }
   return HYPREDRV_JuliaVectorLocalLength(rhs, length);
}

/** @brief Return the local length of an IJ vector. */
uint32_t
HYPREDRV_JuliaVectorLocalLength(void *vec, int64_t *length)
{
   if (vec == NULL || length == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("vector or length output pointer is NULL");
   }
   HYPRE_BigInt lo = 0, hi = -1;
   if (HYPRE_IJVectorGetLocalRange((HYPRE_IJVector)vec, &lo, &hi))
   {
      HYPRE_ClearAllErrors();
      return HYPREDRV_ErrorInvalidValue("failed to query the vector local range");
   }
   *length = (int64_t)(hi - lo + 1);
   return HYPREDRV_SUCCESS;
}

/** @brief Return the local row range of the system matrix. */
uint32_t
HYPREDRV_JuliaGetMatrixRowRange(HYPREDRV_t hypredrv, int64_t *row_start, int64_t *row_end)
{
   if (row_start == NULL || row_end == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("row range output pointer is NULL");
   }
   HYPRE_Matrix mat  = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetMatrix(hypredrv, &mat);
   if (code != 0u)
   {
      return code;
   }
   HYPRE_BigInt ilower = 0, iupper = -1, jlower = 0, jupper = -1;
   if (mat == NULL || HYPRE_IJMatrixGetLocalRange((HYPRE_IJMatrix)mat, &ilower, &iupper,
                                                  &jlower, &jupper))
   {
      HYPRE_ClearAllErrors();
      return HYPREDRV_ErrorInvalidValue("failed to query the matrix local range");
   }
   *row_start = (int64_t)ilower;
   *row_end   = (int64_t)iupper;
   return HYPREDRV_SUCCESS;
}

/** @brief Copy exactly the local RHS (length entries) into caller-owned host memory. */
uint32_t
HYPREDRV_JuliaCopyRHSValues(HYPREDRV_t hypredrv, void *dst, int64_t length)
{
   int64_t  n    = 0;
   uint32_t code = HYPREDRV_JuliaLinearSystemGetRHSLength(hypredrv, &n);
   if (code != 0u)
   {
      return code;
   }
   if (n != length)
   {
      return HYPREDRV_ErrorInvalidValue("RHS copy length mismatch");
   }
#ifdef HYPRE_USING_GPU
   /* HYPREDRV_LinearSystemGetRHSValues reports no memory location, so the raw
    * pointer may be device memory on GPU builds. */
   (void)dst;
   return HYPREDRV_ErrorInvalidValue(
      "RHS values are not readable on GPU HYPRE builds; use get_rhs_handle");
#else
   HYPRE_Complex *data = NULL;
   code                = HYPREDRV_LinearSystemGetRHSValues(hypredrv, &data);
   if (code != 0u)
   {
      return code;
   }
   if (n > 0 && (data == NULL || dst == NULL))
   {
      return HYPREDRV_ErrorInvalidValue("RHS values are unavailable");
   }
   if (n > 0) memcpy(dst, data, (size_t)n * sizeof(HYPRE_Complex));
   return HYPREDRV_SUCCESS;
#endif
}

/** @brief Copy length entries of state vector index into caller-owned host memory. */
uint32_t
HYPREDRV_JuliaCopyStateVectorValues(HYPREDRV_t hypredrv, int index, void *dst,
                                    int64_t length)
{
   if (length < 0)
   {
      return HYPREDRV_ErrorInvalidValue("state vector length is negative");
   }
   HYPRE_Complex *data = NULL;
   uint32_t       code = HYPREDRV_StateVectorGetValues(hypredrv, index, &data);
   if (code != 0u)
   {
      return code;
   }
   if (length > 0 && (data == NULL || dst == NULL))
   {
      return HYPREDRV_ErrorInvalidValue("state vector values are unavailable");
   }
   if (length > 0) memcpy(dst, data, (size_t)length * sizeof(HYPRE_Complex));
   return HYPREDRV_SUCCESS;
}
