/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "maps.h"

#if 0
/*--------------------------------------------------------------------------
 * StrIntMapFind
 *--------------------------------------------------------------------------*/

static inline int
StrIntMapFind(const StrIntMap* valid, const char *string, size_t size)
{
   char     *end_ptr;
   long int  string_num = strtol(string, &end_ptr, 10);
   size_t    i;

   if (*end_ptr == '\0')
   {
      /* valid number string */
      for (i = 0; i < size; i++)
      {
         if (valid[i].num == string_num)
         {
            return (int) string_num;
         }
      }
   }
   else
   {
      /* not a number string */
      for (i = 0; i < size; i++)
      {
         if (!strcmp(valid[i].str, string))
         {
            return valid[i].num;
         }
      }
   }

   return INT_MIN;
}
#endif

/*--------------------------------------------------------------------------
 * StrArrayEntryExists
 *--------------------------------------------------------------------------*/

bool
StrArrayEntryExists(const StrArray valid, const char *string)
{
   size_t  i;

   for (i = 0; i < valid.size; i++)
   {
      if (!strcmp(valid.array[i], string))
      {
         return true;
      }
   }

   return false;
}

/*--------------------------------------------------------------------------
 * StrIntMapArrayGetImage
 *--------------------------------------------------------------------------*/

int
StrIntMapArrayGetImage(const StrIntMapArray valid, const char *string)
{
   char     *end_ptr;
   long int  string_num = strtol(string, &end_ptr, 10);
   size_t    i;

   if (*end_ptr == '\0')
   {
      /* valid number string */
      for (i = 0; i < valid.size; i++)
      {
         if (valid.array[i].num == string_num)
         {
            return valid.array[i].num;
         }
      }
   }
   else
   {
      /* not a number string */
      for (i = 0; i < valid.size; i++)
      {
         if (!strcmp(valid.array[i].str, string))
         {
            return valid.array[i].num;
         }
      }
   }

   return INT_MIN;
}

/*--------------------------------------------------------------------------
 * StrIntMapArrayDomainEntryExists
 *--------------------------------------------------------------------------*/

bool
StrIntMapArrayDomainEntryExists(const StrIntMapArray valid, const char *string)
{
   if (StrIntMapArrayGetImage(valid, string) > INT_MIN)
   {
      return true;
   }
   else
   {
      return false;
   }
}
