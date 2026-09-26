/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_COMPATIBILITY_HEADER
#define HYPREDRV_COMPATIBILITY_HEADER

#include <stdint.h>

#include "HYPRE_utilities.h"

/* MSVC exposes a number of POSIX-compatible CRT routines with a leading
 * underscore.  Keep the implementation sources portable without making the
 * public API depend on these names. */
#if defined(_MSC_VER)
#include <direct.h>
#include <io.h>
#include <process.h>
#include <string.h>

#define access(...) _access(__VA_ARGS__)
#define close(...) _close(__VA_ARGS__)
#define dup(...) _dup(__VA_ARGS__)
#define dup2(...) _dup2(__VA_ARGS__)
#define fdopen(...) _fdopen(__VA_ARGS__)
#define fileno(...) _fileno(__VA_ARGS__)
#define getpid(...) _getpid(__VA_ARGS__)
#define open(...) _open(__VA_ARGS__)
#define read(...) _read(__VA_ARGS__)
#define rmdir(...) _rmdir(__VA_ARGS__)
#define strdup(...) _strdup(__VA_ARGS__)
#define strtok_r(...) strtok_s(__VA_ARGS__)
#define strcasecmp(...) _stricmp(__VA_ARGS__)
#define strncasecmp(...) _strnicmp(__VA_ARGS__)
#define unlink(...) _unlink(__VA_ARGS__)
#define write(...) _write(__VA_ARGS__)
#endif

/* Provide stable version macros even when older hypre headers don't define them. */
#ifndef HYPREDRV_HYPRE_RELEASE_NUMBER
#ifdef HYPRE_RELEASE_NUMBER
#define HYPREDRV_HYPRE_RELEASE_NUMBER HYPRE_RELEASE_NUMBER
#else
#define HYPREDRV_HYPRE_RELEASE_NUMBER 0
#endif
#endif

#ifndef HYPREDRV_HYPRE_DEVELOP_NUMBER
#ifdef HYPRE_DEVELOP_NUMBER
#define HYPREDRV_HYPRE_DEVELOP_NUMBER HYPRE_DEVELOP_NUMBER
#else
#define HYPREDRV_HYPRE_DEVELOP_NUMBER 0
#endif
#endif

/* Older hypre releases provide HYPRE_DescribeError but not the public buffer
 * size macro used by newer headers. Keep the fallback local to hypredrive's
 * compatibility layer and let newer hypre define the canonical value. */
#ifndef HYPRE_MAX_MSG_LEN
#define HYPRE_MAX_MSG_LEN 256
#endif

/**
 * @brief Narrow a signed 64-bit value into the active HYPRE_BigInt type.
 *
 * @return 1 on success and writes @p out; 0 when @p value is outside the
 *         representable range of HYPRE_BigInt.
 */
static inline int
hypredrv_BigIntFromI64(int64_t value, HYPRE_BigInt *out)
{
   HYPRE_BigInt converted;

   if (!out)
   {
      return 0;
   }

   converted = (HYPRE_BigInt)value;
   if ((int64_t)converted != value)
   {
      return 0;
   }

   *out = converted;
   return 1;
}

/**
 * @brief Convert the pair of 64-bit row bounds used by language bridges.
 *
 * The bridge-specific conversion callback keeps each language's diagnostic
 * text intact while this helper centralizes the common start-then-end
 * conversion order.
 */
typedef uint32_t (*hypredrv_BridgeBigIntFromI64Fn)(int64_t value, const char *name,
                                                   HYPRE_BigInt *converted);

static inline uint32_t
hypredrv_BridgeRowBoundsFromI64(int64_t row_start_i64, int64_t row_end_i64,
                                HYPRE_BigInt *row_start, HYPRE_BigInt *row_end,
                                hypredrv_BridgeBigIntFromI64Fn convert)
{
   uint32_t code = convert(row_start_i64, "row_start", row_start);
   if (code != 0u)
   {
      return code;
   }

   return convert(row_end_i64, "row_end", row_end);
}

/**
 * @brief Narrow an unsigned 64-bit value into a non-negative HYPRE_BigInt.
 *
 * Rejects values that do not round-trip through HYPRE_BigInt, including any
 * conversion that would appear negative in a signed HYPRE_BigInt build.
 *
 * @return 1 on success and writes @p out; 0 on overflow or range error.
 */
static inline int
hypredrv_BigIntFromU64(uint64_t value, HYPRE_BigInt *out)
{
   HYPRE_BigInt converted;

   if (!out)
   {
      return 0;
   }

   converted = (HYPRE_BigInt)value;
   if (converted < 0 || (uint64_t)converted != value)
   {
      return 0;
   }

   *out = converted;
   return 1;
}

#endif /* HYPREDRV_COMPATIBILITY_HEADER */
