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

IntArray *
IntArrayCreate(size_t size)
{
   IntArray *int_array = NULL;

   int_array       = malloc(sizeof(IntArray));
   int_array->data = malloc(size * sizeof(int));
   int_array->size = size;

   int_array->unique_size = 0;
   int_array->unique_data = NULL;

   int_array->g_unique_size = 0;
   int_array->g_unique_data = NULL;

   return int_array;
}

/*--------------------------------------------------------------------------
 * IntArrayClone
 *--------------------------------------------------------------------------*/

IntArray *
IntArrayClone(const IntArray *other)
{
   IntArray *this = NULL;

   this = IntArrayCreate(other->size);
   memcpy(this->data, other->data, other->size * sizeof(int));

   if (this->unique_data)
   {
      memcpy(this->unique_data, other->unique_data, other->unique_size * sizeof(int));
      this->unique_size = other->unique_size;
   }

   if (this->g_unique_data)
   {
      memcpy(this->g_unique_data, other->g_unique_data,
             other->g_unique_size * sizeof(int));
      this->g_unique_size = other->g_unique_size;
   }

   return this;
}

/*--------------------------------------------------------------------------
 * IntArrayDestroy
 *--------------------------------------------------------------------------*/

void
IntArrayDestroy(IntArray **int_array_ptr)
{
   IntArray *this = *int_array_ptr;

   if (this)
   {
      free(this->data);
      if (this->unique_data)
      {
         free(this->unique_data);
      }
      if (this->g_unique_data)
      {
         free(this->g_unique_data);
      }
      free(this);
      *int_array_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * StrToIntArray
 *-----------------------------------------------------------------------------*/

void
StrToIntArray(const char *string, IntArray **int_array_ptr)
{
   char       *buffer    = NULL;
   const char *token     = NULL;
   int         count     = 0;
   IntArray   *int_array = NULL;

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
 * StrToStackIntArray
 *-----------------------------------------------------------------------------*/

void
StrToStackIntArray(const char *string, StackIntArray *int_array)
{
   char       *buffer = NULL;
   const char *token  = NULL;
   int         count  = 0;

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

   /* Set StackIntArray size */
   int_array->size =
      (count < MAX_STACK_ARRAY_LENGTH) ? count : MAX_STACK_ARRAY_LENGTH - 1;

   /* Build array */
   buffer = strdup(string);
   token  = strtok(buffer, "[], ");
   count  = 0;
   while (token)
   {
      if (count < MAX_STACK_ARRAY_LENGTH)
      {
         int_array->data[count] = atoi(token);
      }
      count++;
      token = strtok(NULL, "[], ");
   }
   free(buffer);
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
 * IntArrayUnique
 *-----------------------------------------------------------------------------*/

void
IntArrayUnique(MPI_Comm comm, IntArray *int_array)
{
   IntArray *tmp_array         = NULL;
   int       num_entries_int   = 0;
   int       total_num_entries = 0;
   int      *all_num_entries   = NULL;
   int      *displs            = NULL;
   int      *all_data          = NULL;
   int       myid = 0, nprocs = 0;

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   /* Sort input array */
   tmp_array = IntArrayClone((const IntArray *)int_array);
   IntArraySort(tmp_array);

   /* Find number of unique entries locally */
   int_array->unique_size = 1;
   for (size_t i = 1; i < int_array->size; i++)
   {
      if (tmp_array->data[i] != tmp_array->data[i - 1])
      {
         int_array->unique_size++;
      }
   }

   /* Compute local unique array */
   int_array->unique_data    = (int *)calloc(int_array->unique_size, sizeof(int));
   int_array->unique_data[0] = tmp_array->data[0];
   for (size_t i = 1, k = 0; i < int_array->size; i++)
   {
      if (tmp_array->data[i] != tmp_array->data[i - 1])
      {
         int_array->unique_data[++k] = tmp_array->data[i];
      }
   }
   IntArrayDestroy(&tmp_array);

   /* Gather sizes of local unique arrays */
   if (!myid)
   {
      all_num_entries = (int *)malloc(nprocs * sizeof(int));
   }
   num_entries_int = (int)int_array->unique_size;
   MPI_Gather(&num_entries_int, 1, MPI_INT, all_num_entries, 1, MPI_INT, 0, comm);

   /* Gather local unique arrays */
   if (!myid)
   {
      displs            = (int *)calloc(nprocs, sizeof(int));
      total_num_entries = all_num_entries[0];
      for (int i = 1; i < nprocs; i++)
      {
         displs[i] = displs[i - 1] + all_num_entries[i - 1];
         total_num_entries += all_num_entries[i];
      }
      all_data = (int *)malloc(total_num_entries * sizeof(int));
   }
   MPI_Gatherv(int_array->unique_data, (int)int_array->unique_size, MPI_INT, all_data,
               all_num_entries, displs, MPI_INT, 0, comm);

   /* Compute global number of unique entries */
   if (!myid)
   {
      /* Sort input array */
      qsort(all_data, total_num_entries, sizeof(int), IntArrayCompare);

      /* Find number of unique entries */
      int_array->g_unique_size = 1;
      for (int i = 1; i < total_num_entries; i++)
      {
         if (all_data[i] != all_data[i - 1])
         {
            int_array->g_unique_size++;
         }
      }
   }
   MPI_Bcast(&int_array->g_unique_size, 1, MPI_UNSIGNED_LONG, 0, comm);
   int_array->g_unique_data = (int *)calloc(int_array->g_unique_size, sizeof(int));

   /* Compute global unique data */
   if (!myid)
   {
      int_array->g_unique_data[0] = all_data[0];
      for (size_t i = 1, k = 0; i < int_array->g_unique_size; i++)
      {
         if (all_data[i] != all_data[i - 1])
         {
            int_array->g_unique_data[++k] = all_data[i];
         }
      }
      free(all_data);
      free(all_num_entries);
      free(displs);
   }
   MPI_Bcast(int_array->g_unique_data, int_array->g_unique_size, MPI_INT, 0, comm);
}

/*-----------------------------------------------------------------------------
 * IntArrayParRead
 *-----------------------------------------------------------------------------*/

void
IntArrayParRead(MPI_Comm comm, const char *prefix, IntArray **int_array_ptr)
{
   char      filename[MAX_FILENAME_LENGTH];
   char      suffix[5], code[3];
   size_t    num_entries = 0, num_entries_all = 0, count = 0;
   IntArray *int_array = NULL;
   FILE     *fp        = NULL;
   int       myid = 0, nprocs = 0, nparts = 0, g_nparts = 0, offset = 0;
   int      *partids   = NULL;
   bool      is_binary = false;

   *int_array_ptr = NULL;

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   g_nparts = CountNumberOfPartitions(prefix);
   nparts   = g_nparts / nprocs;
   nparts += (myid < (g_nparts % nprocs)) ? 1 : 0;
   if (g_nparts < nprocs)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid number of parts!");
      return;
   }

   /* 1b) Compute partids array */
   partids = malloc(nparts * sizeof(int));
   offset  = myid * nparts;
   offset += (myid < (g_nparts % nprocs)) ? myid : (g_nparts % nprocs);
   for (int part = 0; part < nparts; part++)
   {
      partids[part] = offset + part;
   }

   /* Set file suffix */
   if (CheckBinaryDataExists(prefix))
   {
      is_binary = true;
      strcpy(suffix, ".bin");
      strcpy(code, "rb");
   }
   else
   {
      is_binary = false;
      strcpy(code, "r");
      suffix[0] = '\0';
   }

   /* Compute total number of entries considering all parts */
   num_entries_all = 0;
   for (int part = 0; part < nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, partids[part], suffix);
      if ((fp = fopen(filename, code)) == NULL)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(filename);
         return;
      }

      count = (is_binary) ? fread(&num_entries, sizeof(size_t), 1, fp)
                          : (size_t)fscanf(fp, "%zu", &num_entries);
      if (count != 1)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid number of header entries!");
         return;
      }
      fclose(fp);
      num_entries_all += num_entries;
   }
   int_array = IntArrayCreate(num_entries_all);

   /* Fill entries */
   for (size_t part = 0, idx = 0; part < (size_t)nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, partids[part], suffix);
      if ((fp = fopen(filename, code)) == NULL)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(filename);
         return;
      }

      count = (is_binary) ? fread(&num_entries, sizeof(size_t), 1, fp)
                          : (size_t)fscanf(fp, "%zu", &num_entries);
      if (count != 1)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid number of header entries!");
         return;
      }

      if (is_binary)
      {
         count = fread(&int_array->data[idx], sizeof(size_t), num_entries, fp);
      }
      else
      {
         count = 0;
         while (count < num_entries)
         {
            if (fscanf(fp, "%d", &int_array->data[idx + count]) != 1)
            {
               break;
            }
            count++;
         }
      }
      if (count != num_entries)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Expected %d, but found %ld coefficients!", num_entries, count);
         return;
      }

      fclose(fp);
      idx += num_entries;
   }
   free(partids);

   /* Compute unique varibales */
   IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * IntArrayBuild
 *-----------------------------------------------------------------------------*/

void
IntArrayBuild(MPI_Comm comm, int size, const int *dofmap, IntArray **int_array_ptr)
{
   IntArray *int_array = NULL;

   int_array = IntArrayCreate(size);
   memcpy(int_array->data, dofmap, size * sizeof(int));
   IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * IntArrayBuildInterleaved
 *-----------------------------------------------------------------------------*/

void
IntArrayBuildInterleaved(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                         IntArray **int_array_ptr)
{
   IntArray *int_array = NULL;
   int       size      = num_dof_types * num_local_blocks; // TODO: check overflow

   int_array = IntArrayCreate(size);
   for (int i = 0; i < num_local_blocks; i++)
   {
      for (int j = 0; j < num_dof_types; j++)
      {
         int_array->data[(i * num_dof_types) + j] = j;
      }
   }
   IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * IntArrayBuildContiguous
 *-----------------------------------------------------------------------------*/

void
IntArrayBuildContiguous(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                        IntArray **int_array_ptr)
{
   IntArray *int_array = NULL;
   int       size      = num_dof_types * num_local_blocks; // TODO: check overflow

   int_array = IntArrayCreate(size);
   for (int i = 0; i < num_dof_types; i++)
   {
      for (int j = 0; j < num_local_blocks; j++)
      {
         int_array->data[(i * num_local_blocks) + j] = j;
      }
   }
   IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*--------------------------------------------------------------------------
 * OnOffMapArray
 *--------------------------------------------------------------------------*/

const StrIntMapArray OnOffMapArray = {
   .data =
      (const StrIntMap[]){
         {"on", 1}, {"yes", 1}, {"true", 1}, {"off", 0}, {"no", 0}, {"false", 0}},
   .size = 6};

/*--------------------------------------------------------------------------
 * StrArrayEntryExists
 *--------------------------------------------------------------------------*/

bool
StrArrayEntryExists(const StrArray valid, const char *string)
{
   size_t i = 0;

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
   char    *end_ptr    = NULL;
   long int string_num = strtol(string, &end_ptr, 10);
   size_t   i          = 0;

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
