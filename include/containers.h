/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CONTAINERS_HEADER
#define CONTAINERS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include "utils.h"
#include "HYPRE.h"
#include "HYPRE_config.h"
#include "HYPRE_utilities.h"

#define MAX_FILENAME_LENGTH 2048

/*--------------------------------------------------------------------------
 * IntArray struct
 *--------------------------------------------------------------------------*/

typedef struct IntArray_struct
{
   int      *data;
   size_t    size;
   size_t    num_unique_entries;
} IntArray;

IntArray* IntArrayCreate(size_t);
void IntArrayDestroy(IntArray**);
int IntArrayParRead(MPI_Comm, const char*, IntArray**);

/*--------------------------------------------------------------------------
 * StrArray struct
 *--------------------------------------------------------------------------*/

typedef struct StrArray_struct
{
   const char  **data;
   size_t        size;
} StrArray;

#define STR_ARRAY_CREATE(_str) \
   (StrArray){.data = _str, .size = sizeof(_str) / sizeof(_str[0])}

bool StrArrayEntryExists(const StrArray, const char*);
void StrToIntArray(const char*, IntArray**);

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
   const StrIntMap  *data;
   size_t            size;
} StrIntMapArray;

#define STR_INT_MAP_ARRAY_CREATE(map) \
   (StrIntMapArray){.data = map, .size = sizeof(map) / sizeof(map[0])}
#define STR_INT_MAP_ARRAY_CREATE_ON_OFF() OnOffMapArray
#define STR_INT_MAP_ARRAY_VOID() \
   (StrIntMapArray){.data = NULL, .size = 0}

extern const StrIntMapArray OnOffMapArray;
int  StrIntMapArrayGetImage(const StrIntMapArray, const char*);
bool StrIntMapArrayDomainEntryExists(const StrIntMapArray, const char*);

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
