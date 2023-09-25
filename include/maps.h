/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MAPS_HEADER
#define MAPS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>

/*--------------------------------------------------------------------------
 * StrArray struct
 *--------------------------------------------------------------------------*/

typedef struct StrArray_struct
{
   const char  **array;
   size_t        size;
} StrArray;

#define STR_ARRAY_CREATE(str) \
   (StrArray){.array = str, .size = sizeof(str) / sizeof(str[0])}

bool StrArrayEntryExists(const StrArray, const char*);
void StrToIntArray(const char*, int*, int**);

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
#define STR_INT_MAP_ARRAY_CREATE_ON_OFF() \
   (StrIntMapArray) { \
      .array = (const StrIntMap[]){{"on",  1}, {"yes", 1}, {"true",  1},\
                                   {"off", 0}, {"no",  0}, {"false", 0}}, \
      .size = 6 \
   }
#define STR_INT_MAP_ARRAY_VOID() \
   (StrIntMapArray){.array = NULL, .size = 0}

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

#endif /* MAPS_HEADER */
