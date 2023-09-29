/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "containers.h"

/*--------------------------------------------------------------------------
 * IntArrayCreate
 *--------------------------------------------------------------------------*/

IntArray*
IntArrayCreate(size_t size)
{
   IntArray* int_array;

   int_array = (IntArray*) malloc(sizeof(IntArray));
   int_array->array = (int*) malloc(size * sizeof(int));
   int_array->size  = size;

   return int_array;
}

/*--------------------------------------------------------------------------
 * IntArrayDestroy
 *--------------------------------------------------------------------------*/

void
IntArrayDestroy(IntArray **int_array_ptr)
{
   free((*int_array_ptr)->array);
   free(*int_array_ptr);
   *int_array_ptr = NULL;
}

/*-----------------------------------------------------------------------------
 * StrToIntArray
 *-----------------------------------------------------------------------------*/

void
StrToIntArray(const char* string, IntArray **int_array_ptr)
{
   char      *buffer;
   char      *token;
   int       *array;
   int        count;
   IntArray  *int_array;

   /* Find number of elements in array */
   buffer = strdup(string);
   token  = strtok(buffer, "[], ");
   count  = 0;
   while (token)
   {
      count++;
      token = strtok(NULL, "[], ");
   }
   free(buffer);

   /* Create IntArray */
   int_array = IntArrayCreate(count);

   /* Build array */
   buffer = strdup(string);
   token  = strtok(buffer, "[], ");
   count  = 0;
   while (token)
   {
      int_array->array[count] = atoi(token);
      count++;
      token = strtok(NULL, "[], ");
   }
   free(buffer);

   /* Set output pointer */
   *int_array_ptr = int_array;
}

/*--------------------------------------------------------------------------
 * OnOffMapArray
 *--------------------------------------------------------------------------*/

const StrIntMapArray OnOffMapArray =
{
   .array = (const StrIntMap[]){{"on",  1}, {"yes", 1}, {"true",  1},
                                {"off", 0}, {"no",  0}, {"false", 0}},
   .size = 6
};

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
