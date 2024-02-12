/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "containers.h"

/*--------------------------------------------------------------------------
 * IntArrayCreate
 *--------------------------------------------------------------------------*/

IntArray*
IntArrayCreate(size_t size)
{
   IntArray* int_array;

   int_array = malloc(sizeof(IntArray));
   int_array->data = malloc(size * sizeof(int));
   int_array->size = size;
   int_array->num_unique_entries = 0;

   return int_array;
}

/*--------------------------------------------------------------------------
 * IntArrayClone
 *--------------------------------------------------------------------------*/

IntArray*
IntArrayClone(IntArray* int_array_other)
{
   IntArray* int_array;

   int_array = IntArrayCreate(int_array_other->size);
   memcpy(int_array->data, int_array_other->data, int_array->size * sizeof(int));

   return int_array;
}

/*--------------------------------------------------------------------------
 * IntArrayDestroy
 *--------------------------------------------------------------------------*/

void
IntArrayDestroy(IntArray **int_array_ptr)
{
   if (*int_array_ptr)
   {
      free((*int_array_ptr)->data);
      free(*int_array_ptr);
      *int_array_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * StrToIntArray
 *-----------------------------------------------------------------------------*/

void
StrToIntArray(const char* string, IntArray **int_array_ptr)
{
   char      *buffer;
   char      *token;
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
      int_array->data[count] = atoi(token);
      count++;
      token = strtok(NULL, "[], ");
   }
   free(buffer);

   /* Set output pointer */
   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * IntArrayReadASCII
 *-----------------------------------------------------------------------------*/

void
IntArrayReadASCII(int id, const char* prefix, IntArray **array_ptr)
{
   char       filename[MAX_FILENAME_LENGTH];
   int        num_entries;
   int       *data;
   IntArray  *array;
   FILE      *fp;

   *array_ptr = NULL;
   sprintf(filename, "%s.%05d", prefix, id);
   if ((fp = fopen(filename, "r")) == NULL)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(filename);
      return;
   }

   if (fscanf(fp, "%d", &num_entries) != 1) return;
   array = IntArrayCreate(num_entries);
   for (int i = 0; i < num_entries; i++)
   {
      if (fscanf(fp, "%d", &array->data[i]) != 1) return;
   }
   fclose(fp);

   *array_ptr = array;
}

/*-----------------------------------------------------------------------------
 * IntArrayReadBinary
 *-----------------------------------------------------------------------------*/

void
IntArrayReadBinary(int id, const char* prefix, IntArray **array_ptr)
{
   char        filename[MAX_FILENAME_LENGTH];
   size_t      num_entries;
   IntArray   *array;
   FILE       *fp;

   *array_ptr = NULL;
   sprintf(filename, "%s.%05d.bin", prefix, id);
   if ((fp = fopen(filename, "rb")) == NULL)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(filename);
      return;
   }

   if (fscanf(fp, "%ld", &num_entries) != 1) return;
   array = IntArrayCreate(num_entries);
   if (fread(array->data, sizeof(int), num_entries, fp) != num_entries) return;
   fclose(fp);

   *array_ptr = array;
}

/*-----------------------------------------------------------------------------
 * IntArrayCompare
 *-----------------------------------------------------------------------------*/

int
IntArrayCompare(const void *a, const void *b)
{
   return (*(int *)a - *(int *)b);
}

/*-----------------------------------------------------------------------------
 * IntArraySort
 *-----------------------------------------------------------------------------*/

void
IntArraySort(IntArray *int_array)
{
   qsort(int_array->data, int_array->size, sizeof(int), IntArrayCompare);
}

/*-----------------------------------------------------------------------------
 * IntArrayParRead
 *-----------------------------------------------------------------------------*/

void
IntArrayParRead(MPI_Comm comm, const char* prefix, IntArray **int_array_ptr)
{
   int        myid;
   int        num_unique_entries;
   IntArray  *int_array, *tmp_array;

   MPI_Comm_rank(comm, &myid);

   if (CheckBinaryDataExists(prefix))
   {
      IntArrayReadBinary(myid, prefix, &int_array);
   }
   else
   {
      IntArrayReadASCII(myid, prefix, &int_array);
   }

   /* Find largest entry */
   tmp_array = IntArrayClone(int_array);
   IntArraySort(tmp_array);
   num_unique_entries = 1;
   for (int i = 1; i < int_array->size; i++)
   {
      if (tmp_array->data[i] != tmp_array->data[i - 1])
      {
         num_unique_entries++;
      }
   }
   IntArrayDestroy(&tmp_array);
   MPI_Allreduce(&num_unique_entries, &(int_array->num_unique_entries),
                 1, MPI_INT, MPI_MAX, comm);

   /* TODO - Fix num_unique_entries computation */
   *int_array_ptr = int_array;
}

/*--------------------------------------------------------------------------
 * OnOffMapArray
 *--------------------------------------------------------------------------*/

const StrIntMapArray OnOffMapArray =
{
   .data = (const StrIntMap[]){{"on",  1}, {"yes", 1}, {"true",  1},
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
      if (!strcmp(valid.data[i], string))
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
         if (valid.data[i].num == string_num)
         {
            return valid.data[i].num;
         }
      }
   }
   else
   {
      /* not a number string */
      for (i = 0; i < valid.size; i++)
      {
         if (!strcmp(valid.data[i].str, string))
         {
            return valid.data[i].num;
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
   return (StrIntMapArrayGetImage(valid, string) > INT_MIN) ? true : false;
}
