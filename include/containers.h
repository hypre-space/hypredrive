/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef CONTAINERS_HEADER
#define CONTAINERS_HEADER

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HYPRE.h"
#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "utils.h"

enum
{
   MAX_FILENAME_LENGTH    = 2048,
   MAX_STACK_ARRAY_LENGTH = 128
};

/*--------------------------------------------------------------------------
 * StackIntArray struct
 *--------------------------------------------------------------------------*/

typedef struct StackIntArray_struct
{
   int    data[MAX_STACK_ARRAY_LENGTH];
   size_t size;
} StackIntArray;

void StackIntArrayRead(StackIntArray *);
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

IntArray *IntArrayCreate(size_t);
void      IntArrayDestroy(IntArray **);
void      IntArrayBuild(MPI_Comm, int, const int *, IntArray **);
void      IntArrayBuildInterleaved(MPI_Comm, int, int, IntArray **);
void      IntArrayBuildContiguous(MPI_Comm, int, int, IntArray **);
void      IntArrayParRead(MPI_Comm, const char *, IntArray **);
void      IntArrayWriteAsciiByRank(MPI_Comm, const IntArray*, const char*);

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

bool StrArrayEntryExists(StrArray, const char *);
void StrToIntArray(const char *, IntArray **);
void StrToStackIntArray(const char *, StackIntArray *);

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
#define STR_INT_MAP_ARRAY_CREATE_ON_OFF() OnOffMapArray
// clang-format off
#define STR_INT_MAP_ARRAY_VOID() (StrIntMapArray){.data = NULL, .size = 0}
// clang-format on

extern const StrIntMapArray OnOffMapArray;

int  StrIntMapArrayGetImage(StrIntMapArray, const char *);
bool StrIntMapArrayDomainEntryExists(StrIntMapArray, const char *);

/*--------------------------------------------------------------------------
 * StrStrIntMap struct (strA,strB <-> num)
 *--------------------------------------------------------------------------*/

typedef struct StrStrIntMap_struct
{
   const char *strA;
   const char *strB;
   int         num;
} StrStrIntMap;

#endif /* CONTAINERS_HEADER */
