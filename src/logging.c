/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "logging.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "object.h"

static int      g_hypredrv_log_level  = HYPREDRV_LOG_LEVEL_OFF;
static bool     g_hypredrv_log_stdout = false;
static MPI_Comm g_last_log_comm       = MPI_COMM_NULL;
static int      g_last_log_rank       = -1;

static void LogVf(int level, int mypid, const char *object_name, int ls_id,
                  const char *fmt, va_list args);

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

static bool
TextEqualsTrimmedIgnoreCase(const char *text, const char *expected)
{
   if (!text || !expected)
   {
      return false;
   }

   while (*text && isspace((unsigned char)*text))
   {
      text++;
   }

   while (*text && *expected)
   {
      if (tolower((unsigned char)*text) != tolower((unsigned char)*expected))
      {
         return false;
      }
      text++;
      expected++;
   }

   if (*expected != '\0')
   {
      return false;
   }

   while (*text && isspace((unsigned char)*text))
   {
      text++;
   }

   return *text == '\0';
}

static bool
LogToStdoutParse(const char *env_log_stream)
{
   return TextEqualsTrimmedIgnoreCase(env_log_stream, "stdout");
}

static FILE *
LogStreamGet(void)
{
   if (g_hypredrv_log_stdout)
   {
      return stdout;
   }
   return stderr;
}

void
hypredrv_LogInitializeFromEnv(void)
{
   g_hypredrv_log_level  = LogLevelParse(getenv("HYPREDRV_LOG_LEVEL"));
   g_hypredrv_log_stdout = LogToStdoutParse(getenv("HYPREDRV_LOG_STREAM"));
}

void
hypredrv_LogReset(void)
{
   g_hypredrv_log_level  = HYPREDRV_LOG_LEVEL_OFF;
   g_hypredrv_log_stdout = false;
   g_last_log_comm       = MPI_COMM_NULL;
   g_last_log_rank       = -1;
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
   if (comm == MPI_COMM_NULL)
   {
      return -1;
   }

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

   if (comm == g_last_log_comm)
   {
      return g_last_log_rank;
   }

   int mypid = -1;
   if (MPI_Comm_rank(comm, &mypid) != MPI_SUCCESS)
   {
      return -1;
   }

   g_last_log_comm = comm;
   g_last_log_rank = mypid;

   return mypid;
}

void
hypredrv_LogCommf(int level, MPI_Comm comm, const char *object_name, int ls_id,
                  const char *fmt, ...)
{
   if (!hypredrv_LogEnabled(level))
   {
      return;
   }

   int mypid = hypredrv_LogRankFromComm(comm);

   va_list args;
   va_start(args, fmt);
   LogVf(level, mypid, object_name, ls_id, fmt, args);
   va_end(args);
}

static void
LogPrintPrefix(int level, const char *object_name, int ls_id)
{
   const char *name   = (object_name && object_name[0] != '\0') ? object_name : "unnamed";
   FILE       *stream = LogStreamGet();

   if (ls_id > 0)
   {
      (void)fprintf(stream, "[HYPREDRV][L%d][%s][ls=%d] ", level, name, ls_id);
   }
   else
   {
      (void)fprintf(stream, "[HYPREDRV][L%d][%s] ", level, name);
   }
}

static void
LogVf(int level, int mypid, const char *object_name, int ls_id, const char *fmt,
      va_list args)
{
   if (mypid != 0 || !fmt)
   {
      return;
   }

   FILE *stream = LogStreamGet();
   LogPrintPrefix(level, object_name, ls_id);
   (void)vfprintf(stream, fmt, args);
   (void)fputc('\n', stream);
   (void)fflush(stream);
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
   char        default_object_name[32];
   default_object_name[0] = '\0';

   if (hypredrv)
   {
      mypid = hypredrv->mypid;
      if (hypredrv->stats)
      {
         object_name = hypredrv->stats->object_name;
         ls_id       = hypredrv_StatsGetLinearSystemID(hypredrv->stats);
      }

      if ((!object_name || object_name[0] == '\0') && hypredrv->runtime_object_id > 0)
      {
         snprintf(default_object_name, sizeof(default_object_name), "obj-%d",
                  hypredrv->runtime_object_id);
         object_name = default_object_name;
      }
   }

   va_list args;
   va_start(args, fmt);
   LogVf(level, mypid, object_name, ls_id, fmt, args);
   va_end(args);
}

void
hypredrv_LogTextBlock(int level, int mypid, const char *object_name, int ls_id,
                      const char *header, const char *text)
{
   if (!hypredrv_LogEnabled(level) || mypid != 0 || !text)
   {
      return;
   }

   FILE *stream = LogStreamGet();
   if (header)
   {
      LogPrintPrefix(level, object_name, ls_id);
      (void)fprintf(stream, "%s\n", header);
   }

   const char *line = text;
   while (*line)
   {
      const char *eol = strchr(line, '\n');
      int         len = eol ? (int)(eol - line) : (int)strlen(line);

      LogPrintPrefix(level, object_name, ls_id);
      (void)fprintf(stream, "  %.*s\n", len, line);

      if (!eol)
      {
         break;
      }
      line = eol + 1;
   }
   (void)fflush(stream);
}
