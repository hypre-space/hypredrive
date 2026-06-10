#ifndef RUNTIME_HEADER
#define RUNTIME_HEADER

#include <stdbool.h>
#include <stdint.h>
#include "HYPREDRV.h"

typedef uint32_t (*hypredrv_RuntimeDestroyFunc)(HYPREDRV_t hypredrv);

uint32_t hypredrv_RuntimeInitialize(void);
uint32_t hypredrv_RuntimeFinalizeState(void);
uint32_t hypredrv_RuntimeDestroyAllLiveObjects(hypredrv_RuntimeDestroyFunc destroy_fn);

bool hypredrv_RuntimeIsInitialized(void);
bool hypredrv_RuntimeIsFinalizing(void);
bool hypredrv_RuntimeObjectIsLive(HYPREDRV_t hypredrv);

int      hypredrv_RuntimeGetActiveCount(void);
uint32_t hypredrv_RuntimeRegisterObject(HYPREDRV_t hypredrv);
void     hypredrv_RuntimeUnregisterObject(HYPREDRV_t hypredrv);

#endif /* RUNTIME_HEADER */
