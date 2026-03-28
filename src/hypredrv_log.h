/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_LOG_HEADER
#define HYPREDRV_LOG_HEADER

#include <stdbool.h>
#include "HYPREDRV.h"

enum
{
   HYPREDRV_LOG_LEVEL_OFF = 0,
   HYPREDRV_LOG_LEVEL_MAX = 3,
};

void hypredrv_LogInitializeFromEnv(void);
void hypredrv_LogReset(void);

int  hypredrv_LogLevelGet(void);
bool hypredrv_LogEnabled(int level);
int  hypredrv_LogRankFromComm(MPI_Comm comm);
void hypredrv_LogCommf(int level, MPI_Comm comm, const char *object_name, int ls_id,
                       const char *fmt, ...) __attribute__((format(printf, 5, 6)));

void hypredrv_Logf(int level, int mypid, const char *object_name, int ls_id,
                   const char *fmt, ...) __attribute__((format(printf, 5, 6)));
void hypredrv_LogObjectf(int level, HYPREDRV_t hypredrv, const char *fmt, ...)
   __attribute__((format(printf, 3, 4)));
void hypredrv_LogTextBlock(int level, int mypid, const char *object_name, int ls_id,
                           const char *header, const char *text);

#define HYPREDRV_LOGF(_level, _mypid, _object_name, _ls_id, _fmt, ...)       \
   do                                                                        \
   {                                                                         \
      if (hypredrv_LogEnabled((_level)))                                     \
      {                                                                      \
         hypredrv_Logf((_level), (_mypid), (_object_name), (_ls_id), (_fmt), \
                       ##__VA_ARGS__);                                       \
      }                                                                      \
   } while (0)

#define HYPREDRV_LOG_COMMF(_level, _comm, _object_name, _ls_id, _fmt, ...)      \
   do                                                                           \
   {                                                                            \
      if (hypredrv_LogEnabled((_level)))                                        \
      {                                                                         \
         hypredrv_LogCommf((_level), (_comm), (_object_name), (_ls_id), (_fmt), \
                           ##__VA_ARGS__);                                      \
      }                                                                         \
   } while (0)

#define HYPREDRV_LOG_OBJECTF(_level, _hypredrv, _fmt, ...)                  \
   do                                                                       \
   {                                                                        \
      if (hypredrv_LogEnabled((_level)))                                    \
      {                                                                     \
         hypredrv_LogObjectf((_level), (_hypredrv), (_fmt), ##__VA_ARGS__); \
      }                                                                     \
   } while (0)

#define HYPREDRV_LOG_TEXTBLOCK(_level, _mypid, _object_name, _ls_id, _header, _text)    \
   do                                                                                   \
   {                                                                                    \
      if (hypredrv_LogEnabled((_level)))                                                \
      {                                                                                 \
         hypredrv_LogTextBlock((_level), (_mypid), (_object_name), (_ls_id), (_header), \
                               (_text));                                                \
      }                                                                                 \
   } while (0)

#endif /* HYPREDRV_LOG_HEADER */
