/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef CONTAINERS_HEADER
#define CONTAINERS_HEADER

#include "HYPRE.h"
#include "HYPRE_utilities.h"

/* Undefine autotools package macros from hypre */
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal/utils.h"

enum
{
   MAX_FILENAME_LENGTH    = 2048,
   MAX_STACK_ARRAY_LENGTH = 128,
};

/*--------------------------------------------------------------------------
 * StackIntArray struct
 *--------------------------------------------------------------------------*/

typedef struct StackIntArray_struct
{
   int    data[MAX_STACK_ARRAY_LENGTH];
   size_t size;
} StackIntArray;

void hypredrv_StackIntArrayRead(StackIntArray *);
#define STACK_INTARRAY_CREATE() ((StackIntArray){.data = {0}, .size = 0})

/*--------------------------------------------------------------------------
 * IntArray struct
 *--------------------------------------------------------------------------*/

typedef struct IntArray_struct
{
   int   *data;
   size_t size;

   int   *unique_data;
   size_t unique_size;

   int   *g_unique_data;
   size_t g_unique_size;
} IntArray;

IntArray *hypredrv_IntArrayCreate(size_t);
void      hypredrv_IntArrayDestroy(IntArray **);
void      hypredrv_IntArrayBuild(MPI_Comm, int, const int *, IntArray **);
void      hypredrv_IntArrayBuildInterleaved(MPI_Comm, int, int, IntArray **);
void      hypredrv_IntArrayBuildContiguous(MPI_Comm, int, int, IntArray **);
void      hypredrv_IntArrayParRead(MPI_Comm, const char *, IntArray **);
void      hypredrv_IntArrayWriteAsciiByRank(MPI_Comm, const IntArray *, const char *);

/*--------------------------------------------------------------------------
 * DoubleArray struct
 *--------------------------------------------------------------------------*/

typedef struct DoubleArray_struct
{
   double *data;
   size_t  size;
} DoubleArray;

DoubleArray *hypredrv_DoubleArrayCreate(size_t);
void         hypredrv_DoubleArrayDestroy(DoubleArray **);

/*--------------------------------------------------------------------------
 * StrArray struct
 *--------------------------------------------------------------------------*/

typedef struct StrArray_struct
{
   const char **data;
   size_t       size;
} StrArray;

#define STR_ARRAY_CREATE(_str)                             \
   (StrArray)                                              \
   {                                                       \
      .data = _str, .size = sizeof(_str) / sizeof(_str[0]) \
   }

bool hypredrv_StrArrayEntryExists(StrArray, const char *);
void hypredrv_StrToIntArray(const char *, IntArray **);
void hypredrv_StrToDoubleArray(const char *, DoubleArray **);
void hypredrv_StrToStackIntArray(const char *, StackIntArray *);

/*--------------------------------------------------------------------------
 * StrIntMap struct (str <-> num)
 *--------------------------------------------------------------------------*/

typedef struct StrIntMap_struct
{
   const char *str;
   int         num;
} StrIntMap;

typedef struct StrIntMapArray_struct
{
   const StrIntMap *data;
   size_t           size;
} StrIntMapArray;

#define STR_INT_MAP_ARRAY_CREATE(map)                   \
   (StrIntMapArray)                                     \
   {                                                    \
      .data = map, .size = sizeof(map) / sizeof(map[0]) \
   }
#define STR_INT_MAP_ARRAY_CREATE_ON_OFF() hypredrv_OnOffMapArray
// clang-format off
#define STR_INT_MAP_ARRAY_VOID() (StrIntMapArray){.data = NULL, .size = 0}
// clang-format on

extern const StrIntMapArray hypredrv_OnOffMapArray;

int  hypredrv_StrIntMapArrayGetImage(StrIntMapArray, const char *);
bool hypredrv_StrIntMapArrayDomainEntryExists(StrIntMapArray, const char *);

/*--------------------------------------------------------------------------
 * StrStrIntMap struct (strA,strB <-> num)
 *--------------------------------------------------------------------------*/

typedef struct StrStrIntMap_struct
{
   const char *strA;
   const char *strB;
   int         num;
} StrStrIntMap;

/*--------------------------------------------------------------------------
 * DofLabelMap struct (name <-> integer dof type)
 *--------------------------------------------------------------------------*/

typedef struct DofLabelEntry_struct
{
   char name[64]; /* label string, e.g. "v_x" */
   int  value;    /* integer dof type */
} DofLabelEntry;

typedef struct DofLabelMap_struct
{
   DofLabelEntry *data;
   size_t         size;
   size_t         capacity;
} DofLabelMap;

DofLabelMap *hypredrv_DofLabelMapCreate(void);
void         hypredrv_DofLabelMapAdd(DofLabelMap *, const char *name, int value);
int          hypredrv_DofLabelMapLookup(const DofLabelMap *, const char *name);
void         hypredrv_DofLabelMapDestroy(DofLabelMap **);

#endif /* CONTAINERS_HEADER */
