/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_LOGGING_HEADER
#define HYPREDRV_LOGGING_HEADER

#include <stdbool.h>
#include "HYPREDRV.h"

#if defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER))
#define HYPREDRV_PRINTF_FORMAT(_format_index, _argument_index) \
   __attribute__((format(printf, _format_index, _argument_index)))
#else
#define HYPREDRV_PRINTF_FORMAT(_format_index, _argument_index)
#endif

enum
{
   HYPREDRV_LOG_LEVEL_OFF = 0,
   HYPREDRV_LOG_LEVEL_MAX = 4,
};

void hypredrv_LogInitializeFromEnv(void);
void hypredrv_LogReset(void);

int  hypredrv_LogLevelGet(void);
bool hypredrv_LogEnabled(int level);
int  hypredrv_LogRankFromComm(MPI_Comm comm);
void hypredrv_LogCommf(int level, MPI_Comm comm, const char *object_name, int ls_id,
                       const char *fmt, ...) HYPREDRV_PRINTF_FORMAT(5, 6);

void hypredrv_Logf(int level, int mypid, const char *object_name, int ls_id,
                   const char *fmt, ...) HYPREDRV_PRINTF_FORMAT(5, 6);
void hypredrv_LogObjectf(int level, HYPREDRV_t hypredrv, const char *fmt, ...)
   HYPREDRV_PRINTF_FORMAT(3, 4);
void hypredrv_LogTextBlock(int level, int mypid, const char *object_name, int ls_id,
                           const char *header, const char *text);

#define HYPREDRV_LOGF(_level, _mypid, _object_name, _ls_id, _fmt, ...)       \
   do                                                                        \
   {                                                                         \
      /* GCOVR_EXCL_BR_START */                                              \
      if (hypredrv_LogEnabled((_level))) /* GCOVR_EXCL_BR_STOP */            \
      {                                                                      \
         hypredrv_Logf((_level), (_mypid), (_object_name), (_ls_id), (_fmt), \
                       ##__VA_ARGS__);                                       \
      }                                                                      \
   } while (0)

#define HYPREDRV_LOG_COMMF(_level, _comm, _object_name, _ls_id, _fmt, ...)      \
   do                                                                           \
   {                                                                            \
      /* GCOVR_EXCL_BR_START */                                                 \
      if (hypredrv_LogEnabled((_level))) /* GCOVR_EXCL_BR_STOP */               \
      {                                                                         \
         hypredrv_LogCommf((_level), (_comm), (_object_name), (_ls_id), (_fmt), \
                           ##__VA_ARGS__);                                      \
      }                                                                         \
   } while (0)

#define HYPREDRV_LOG_OBJECTF(_level, _hypredrv, _fmt, ...)                  \
   do                                                                       \
   {                                                                        \
      /* GCOVR_EXCL_BR_START */                                             \
      if (hypredrv_LogEnabled((_level))) /* GCOVR_EXCL_BR_STOP */           \
      {                                                                     \
         hypredrv_LogObjectf((_level), (_hypredrv), (_fmt), ##__VA_ARGS__); \
      }                                                                     \
   } while (0)

#define HYPREDRV_LOG_TEXTBLOCK(_level, _mypid, _object_name, _ls_id, _header, _text)    \
   do                                                                                   \
   {                                                                                    \
      /* GCOVR_EXCL_BR_START */                                                         \
      if (hypredrv_LogEnabled((_level))) /* GCOVR_EXCL_BR_STOP */                       \
      {                                                                                 \
         hypredrv_LogTextBlock((_level), (_mypid), (_object_name), (_ls_id), (_header), \
                               (_text));                                                \
      }                                                                                 \
   } while (0)

#endif /* HYPREDRV_LOGGING_HEADER */
