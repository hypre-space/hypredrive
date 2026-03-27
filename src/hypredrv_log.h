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

void hypredrv_Logf(int level, int mypid, const char *object_name, int ls_id,
                   const char *fmt, ...);
void hypredrv_LogObjectf(int level, HYPREDRV_t hypredrv, const char *fmt, ...);

#endif /* HYPREDRV_LOG_HEADER */
