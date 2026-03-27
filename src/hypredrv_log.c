/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "hypredrv_log.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "hypredrv_object.h"

static int g_hypredrv_log_level = HYPREDRV_LOG_LEVEL_OFF;

static int
LogLevelParse(const char *env_log_level)
{
   if (!env_log_level || env_log_level[0] == '\0')
   {
      return HYPREDRV_LOG_LEVEL_OFF;
   }

   char *endptr    = NULL;
   long  raw_level = strtol(env_log_level, &endptr, 10);
   if (endptr == env_log_level)
   {
      return HYPREDRV_LOG_LEVEL_OFF;
   }

   while (*endptr && isspace((unsigned char)*endptr))
   {
      endptr++;
   }
   if (*endptr != '\0')
   {
      return HYPREDRV_LOG_LEVEL_OFF;
   }

   if (raw_level < HYPREDRV_LOG_LEVEL_OFF)
   {
      return HYPREDRV_LOG_LEVEL_OFF;
   }
   if (raw_level > HYPREDRV_LOG_LEVEL_MAX)
   {
      return HYPREDRV_LOG_LEVEL_MAX;
   }

   return (int)raw_level;
}

void
hypredrv_LogInitializeFromEnv(void)
{
   g_hypredrv_log_level = LogLevelParse(getenv("HYPREDRV_LOG_LEVEL"));
}

void
hypredrv_LogReset(void)
{
   g_hypredrv_log_level = HYPREDRV_LOG_LEVEL_OFF;
}

int
hypredrv_LogLevelGet(void)
{
   return g_hypredrv_log_level;
}

bool
hypredrv_LogEnabled(int level)
{
   if (level <= HYPREDRV_LOG_LEVEL_OFF)
   {
      return false;
   }
   if (level > g_hypredrv_log_level)
   {
      return false;
   }
   return true;
}

int
hypredrv_LogRankFromComm(MPI_Comm comm)
{
   int mpi_initialized = 0;
   MPI_Initialized(&mpi_initialized);
   if (!mpi_initialized)
   {
      return -1;
   }

   int mpi_finalized = 0;
   MPI_Finalized(&mpi_finalized);
   if (mpi_finalized)
   {
      return -1;
   }

   int mypid = -1;
   if (MPI_Comm_rank(comm, &mypid) != MPI_SUCCESS)
   {
      return -1;
   }

   return mypid;
}

static void
LogPrintPrefix(int level, int mypid, const char *object_name, int ls_id)
{
   const char *name = (object_name && object_name[0] != '\0') ? object_name : "unnamed";

   if (ls_id > 0)
   {
      (void)fprintf(stderr, "[HYPREDRV][L%d][rank=%d][obj=%s][ls=%d] ", level, mypid,
                    name, ls_id);
   }
   else
   {
      (void)fprintf(stderr, "[HYPREDRV][L%d][rank=%d][obj=%s] ", level, mypid, name);
   }
}

static void
LogVf(int level, int mypid, const char *object_name, int ls_id, const char *fmt,
      va_list args)
{
   if (!hypredrv_LogEnabled(level))
   {
      return;
   }
   if (mypid != 0)
   {
      return;
   }

   LogPrintPrefix(level, mypid, object_name, ls_id);
   (void)vfprintf(stderr, fmt, args);
   (void)fputc('\n', stderr);
   (void)fflush(stderr);
}

void
hypredrv_Logf(int level, int mypid, const char *object_name, int ls_id, const char *fmt,
              ...)
{
   if (!hypredrv_LogEnabled(level))
   {
      return;
   }

   va_list args;
   va_start(args, fmt);
   LogVf(level, mypid, object_name, ls_id, fmt, args);
   va_end(args);
}

void
hypredrv_LogObjectf(int level, HYPREDRV_t hypredrv, const char *fmt, ...)
{
   if (!hypredrv_LogEnabled(level))
   {
      return;
   }

   int         mypid       = -1;
   const char *object_name = NULL;
   int         ls_id       = 0;

   if (hypredrv)
   {
      mypid = hypredrv->mypid;
      if (hypredrv->stats)
      {
         object_name = hypredrv->stats->object_name;
         ls_id       = hypredrv_StatsGetLinearSystemID(hypredrv->stats);
      }
   }

   va_list args;
   va_start(args, fmt);
   LogVf(level, mypid, object_name, ls_id, fmt, args);
   va_end(args);
}
