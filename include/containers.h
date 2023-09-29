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

/*--------------------------------------------------------------------------
 * IntArray struct
 *--------------------------------------------------------------------------*/

typedef struct IntArray_struct
{
   int      *array;
   size_t    size;
} IntArray;

IntArray* IntArrayCreate(size_t);
void IntArrayDestroy(IntArray**);

/*--------------------------------------------------------------------------
 * StrArray struct
 *--------------------------------------------------------------------------*/

typedef struct StrArray_struct
{
   const char  **array;
   size_t        size;
} StrArray;

#define STR_ARRAY_CREATE(_str) \
   (StrArray){.array = _str, .size = sizeof(_str) / sizeof(_str[0])}

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
   const StrIntMap  *array;
   size_t            size;
} StrIntMapArray;

#define STR_INT_MAP_ARRAY_CREATE(map) \
   (StrIntMapArray){.array = map, .size = sizeof(map) / sizeof(map[0])}
#define STR_INT_MAP_ARRAY_CREATE_ON_OFF() OnOffMapArray
#define STR_INT_MAP_ARRAY_VOID() \
   (StrIntMapArray){.array = NULL, .size = 0}

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

#endif /* CONTAINERS_HEADER_HEADER */
