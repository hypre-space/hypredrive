/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/containers.h"
#include "internal/error.h"
#include "internal/utils.h"
/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayWriteAsciiByRank
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayWriteAsciiByRank(MPI_Comm comm, const IntArray *ia, const char *filename)
{
   int   myid = 0, nprocs = 0;
   FILE *fp = NULL;
   char  fname[MAX_FILENAME_LENGTH];

   if (!ia || !ia->data) return;

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   snprintf(fname, sizeof(fname), "%s.%05d", filename, myid);
   fp = hypredrv_FopenCreateRestricted(fname, 0, 0);
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(fname);
      return;
   }

   fprintf(fp, "%zu\n", ia->size);
   for (size_t i = 0; i < ia->size; i++)
   {
      fprintf(fp, "%d\n", ia->data[i]);
   }
   fclose(fp);
}

/*--------------------------------------------------------------------------
 * hypredrv_IntArrayCreate
 *--------------------------------------------------------------------------*/

IntArray *
hypredrv_IntArrayCreate(size_t size)
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
 * hypredrv_IntArrayClone
 *--------------------------------------------------------------------------*/

IntArray *
hypredrv_IntArrayClone(const IntArray *other)
{
   IntArray *this = NULL;

   this = hypredrv_IntArrayCreate(other->size);
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
 * hypredrv_IntArrayDestroy
 *--------------------------------------------------------------------------*/

void
hypredrv_IntArrayDestroy(IntArray **int_array_ptr)
{
   IntArray *this = *int_array_ptr;

   if (this)
   {
      free(this->data);
      free(this->unique_data);
      free(this->g_unique_data);
      free(this);
      *int_array_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_StrToIntArray
 *-----------------------------------------------------------------------------*/

void
hypredrv_StrToIntArray(const char *string, IntArray **int_array_ptr)
{
   char       *buffer    = NULL;
   const char *token     = NULL;
   char       *saveptr   = NULL;
   int         count     = 0;
   IntArray   *int_array = NULL;

   /* Find number of elements in array */
   buffer = strdup(string);
   token  = strtok_r(buffer, "[], ", &saveptr);
   count  = 0;
   while (token)
   {
      count++;
      token = strtok_r(NULL, "[], ", &saveptr);
   }
   free(buffer);

   /* Create IntArray */
   int_array = hypredrv_IntArrayCreate((size_t)count);

   /* Build array */
   buffer = strdup(string);
   token  = strtok_r(buffer, "[], ", &saveptr);
   count  = 0;
   while (token)
   {
      int_array->data[count] = atoi(token);
      count++;
      token = strtok_r(NULL, "[], ", &saveptr);
   }
   free(buffer);

   /* Set output pointer */
   *int_array_ptr = int_array;
}

/*--------------------------------------------------------------------------
 * hypredrv_DoubleArrayCreate
 *--------------------------------------------------------------------------*/

DoubleArray *
hypredrv_DoubleArrayCreate(size_t size)
{
   DoubleArray *double_array = NULL;

   double_array       = malloc(sizeof(DoubleArray));
   double_array->data = malloc(size * sizeof(double));
   double_array->size = size;

   return double_array;
}

/*--------------------------------------------------------------------------
 * hypredrv_DoubleArrayDestroy
 *--------------------------------------------------------------------------*/

void
hypredrv_DoubleArrayDestroy(DoubleArray **double_array_ptr)
{
   DoubleArray *this = *double_array_ptr;

   if (this)
   {
      free(this->data);
      free(this);
      *double_array_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_StrToDoubleArray
 *-----------------------------------------------------------------------------*/

void
hypredrv_StrToDoubleArray(const char *string, DoubleArray **double_array_ptr)
{
   char        *buffer       = NULL;
   const char  *token        = NULL;
   char        *saveptr      = NULL;
   int          count        = 0;
   DoubleArray *double_array = NULL;

   /* Find number of elements in array */
   buffer = strdup(string);
   token  = strtok_r(buffer, "[], ", &saveptr);
   count  = 0;
   while (token)
   {
      count++;
      token = strtok_r(NULL, "[], ", &saveptr);
   }
   free(buffer);

   /* Create DoubleArray */
   double_array = hypredrv_DoubleArrayCreate((size_t)count);

   /* Build array */
   buffer = strdup(string);
   token  = strtok_r(buffer, "[], ", &saveptr);
   count  = 0;
   while (token)
   {
      sscanf(token, "%lf", &double_array->data[count]);
      count++;
      token = strtok_r(NULL, "[], ", &saveptr);
   }
   free(buffer);

   /* Set output pointer */
   *double_array_ptr = double_array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_StrToStackIntArray
 *-----------------------------------------------------------------------------*/

void
hypredrv_StrToStackIntArray(const char *string, StackIntArray *int_array)
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
      (count < MAX_STACK_ARRAY_LENGTH) ? (size_t)count : MAX_STACK_ARRAY_LENGTH - 1;

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
 * hypredrv_IntArrayCompare
 *-----------------------------------------------------------------------------*/

int
hypredrv_IntArrayCompare(const void *a, const void *b)
{
   return (*(int *)a - *(int *)b);
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArraySort
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArraySort(IntArray *int_array)
{
   qsort(int_array->data, int_array->size, sizeof(int), hypredrv_IntArrayCompare);
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayUnique
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayUnique(MPI_Comm comm, IntArray *int_array)
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
   tmp_array = hypredrv_IntArrayClone((const IntArray *)int_array);
   if (int_array->size > 0)
   {
      hypredrv_IntArraySort(tmp_array);

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
   }
   else
   {
      int_array->unique_size = 0;
      int_array->unique_data = NULL;
   }
   hypredrv_IntArrayDestroy(&tmp_array);

   /* Gather sizes of local unique arrays */
   if (!myid)
   {
      all_num_entries = (int *)malloc((size_t)nprocs * sizeof(int));
   }
   num_entries_int = (int)int_array->unique_size;
   MPI_Gather(&num_entries_int, 1, MPI_INT, all_num_entries, 1, MPI_INT, 0, comm);

   /* Gather local unique arrays */
   if (!myid)
   {
      displs            = (int *)calloc((size_t)nprocs, sizeof(int));
      total_num_entries = all_num_entries[0];
      for (int i = 1; i < nprocs; i++)
      {
         displs[i] = displs[i - 1] + all_num_entries[i - 1];
         total_num_entries += all_num_entries[i];
      }
      all_data = (total_num_entries > 0)
                    ? (int *)malloc((size_t)total_num_entries * sizeof(int))
                    : NULL;
   }
   MPI_Gatherv(int_array->unique_data, (int)int_array->unique_size, MPI_INT, all_data,
               all_num_entries, displs, MPI_INT, 0, comm);

   /* Compute global number of unique entries */
   if (!myid)
   {
      if (total_num_entries > 0)
      {
         /* Sort input array */
         qsort(all_data, (size_t)total_num_entries, sizeof(int),
               hypredrv_IntArrayCompare);

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
      else
      {
         int_array->g_unique_size = 0;
      }
   }
   MPI_Bcast(&int_array->g_unique_size, 1, MPI_UNSIGNED_LONG, 0, comm);
   int_array->g_unique_data = (int_array->g_unique_size > 0)
                                 ? (int *)calloc(int_array->g_unique_size, sizeof(int))
                                 : NULL;

   /* Compute global unique data */
   if (!myid && int_array->g_unique_size > 0)
   {
      int_array->g_unique_data[0] = all_data[0];
      for (size_t i = 1, k = 0; i < (size_t)total_num_entries; i++)
      {
         if (all_data[i] != all_data[i - 1])
         {
            int_array->g_unique_data[++k] = all_data[i];
         }
      }
   }
   if (!myid)
   {
      free(all_data);
      free(all_num_entries);
      free(displs);
   }
   MPI_Bcast(int_array->g_unique_data, (int)int_array->g_unique_size, MPI_INT, 0, comm);
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayParRead
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayParRead(MPI_Comm comm, const char *prefix, IntArray **int_array_ptr)
{
   char      filename[MAX_FILENAME_LENGTH];
   char      suffix[5], code[3];
   size_t    num_entries = 0, num_entries_all = 0;
   size_t    count;
   IntArray *int_array = NULL;
   FILE     *fp        = NULL;
   int       myid = 0, nprocs = 0, nparts = 0, g_nparts = 0, offset = 0;
   int      *partids   = NULL;
   bool      is_binary = false;

   *int_array_ptr = NULL;

   if (!prefix || !hypredrv_BinaryPathPrefixIsSafe(prefix))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid dofmap path prefix");
      return;
   }

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   g_nparts = hypredrv_CountNumberOfPartitions(prefix);
   nparts   = g_nparts / nprocs;
   nparts += (myid < (g_nparts % nprocs)) ? 1 : 0;
   if (g_nparts < nprocs)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid dofmap filename \"%s\" or invalid number of parts!",
                           prefix);
      return;
   }

   /* 1b) Compute partids array */
   partids = malloc((size_t)nparts * sizeof(int));
   offset  = myid * nparts;
   offset += (myid < (g_nparts % nprocs)) ? myid : (g_nparts % nprocs);
   for (int part = 0; part < nparts; part++)
   {
      partids[part] = offset + part;
   }

   /* Set file suffix */
   if (hypredrv_CheckBinaryDataExists(prefix))
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
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(filename);
         return;
      }

      count = ((int)is_binary) ? fread(&num_entries, sizeof(size_t), 1, fp)
                               : (size_t)fscanf(fp, "%zu", &num_entries);
      if (count != 1)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid number of header entries!");
         return;
      }
      fclose(fp);
      num_entries_all += num_entries;
   }
   int_array = hypredrv_IntArrayCreate(num_entries_all);

   /* Fill entries */
   for (size_t part = 0, idx = 0; part < (size_t)nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, partids[part], suffix);
      if ((fp = fopen(filename, code)) == NULL)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(filename);
         return;
      }

      count = ((int)is_binary) ? fread(&num_entries, sizeof(size_t), 1, fp)
                               : (size_t)fscanf(fp, "%zu", &num_entries);
      if (count != 1)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid number of header entries!");
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
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Expected %d, but found %ld coefficients!", num_entries,
                              count);
         return;
      }

      fclose(fp);
      idx += num_entries;
   }
   free(partids);

   /* Compute unique varibales */
   hypredrv_IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayBuild
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayBuild(MPI_Comm comm, int size, const int *dofmap,
                       IntArray **int_array_ptr)
{
   IntArray *int_array = NULL;

   int_array = hypredrv_IntArrayCreate((size_t)size);
   memcpy(int_array->data, dofmap, (size_t)size * sizeof(int));
   hypredrv_IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayBuildInterleaved
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayBuildInterleaved(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                                  IntArray **int_array_ptr)
{
   IntArray *int_array = NULL;
   int       size      = num_dof_types * num_local_blocks; // TODO: check overflow

   int_array = hypredrv_IntArrayCreate((size_t)size);
   for (int i = 0; i < num_local_blocks; i++)
   {
      for (int j = 0; j < num_dof_types; j++)
      {
         int_array->data[(i * num_dof_types) + j] = j;
      }
   }
   hypredrv_IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayBuildContiguous
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayBuildContiguous(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                                 IntArray **int_array_ptr)
{
   IntArray *int_array = NULL;
   int       size      = num_dof_types * num_local_blocks; // TODO: check overflow

   int_array = hypredrv_IntArrayCreate((size_t)size);
   for (int i = 0; i < num_dof_types; i++)
   {
      for (int j = 0; j < num_local_blocks; j++)
      {
         int_array->data[(i * num_local_blocks) + j] = j;
      }
   }
   hypredrv_IntArrayUnique(comm, int_array);

   *int_array_ptr = int_array;
}

/*--------------------------------------------------------------------------
 * hypredrv_OnOffMapArray
 *--------------------------------------------------------------------------*/

const StrIntMapArray hypredrv_OnOffMapArray = {
   .data =
      (const StrIntMap[]){
         {"on", 1},
         {"yes", 1},
         {"true", 1},
         {"off", 0},
         {"no", 0},
         {"false", 0},
      },
   .size = 6,
};

/*--------------------------------------------------------------------------
 * hypredrv_IntArrayEntryExists
 *--------------------------------------------------------------------------*/

bool
hypredrv_IntArrayEntryExists(const IntArray *arr, int value)
{
   if (!arr || !arr->data)
   {
      return false;
   }

   for (size_t i = 0; i < arr->size; i++)
   {
      if (arr->data[i] == value)
      {
         return true;
      }
   }

   return false;
}

/*--------------------------------------------------------------------------
 * hypredrv_StrArrayEntryExists
 *--------------------------------------------------------------------------*/

bool
hypredrv_StrArrayEntryExists(const StrArray valid, const char *string)
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
 * hypredrv_StrIntMapArrayGetImage
 *--------------------------------------------------------------------------*/

int
hypredrv_StrIntMapArrayGetImage(const StrIntMapArray valid, const char *string)
{
   char    *end_ptr    = NULL;
   long int string_num = 0;
   size_t   i          = 0;
   bool     is_integer = false;

   if (!string)
   {
      return INT_MIN;
   }

   string_num = strtol(string, &end_ptr, 10);

   /* Empty strings are valid YAML scalars in some schemas (for example nested MGR
    * relaxation blocks). Treat them as strings, not as the number 0. */
   is_integer = ((string[0] != '\0' && end_ptr != string && *end_ptr == '\0') != 0);

   if (is_integer)
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
 * hypredrv_StrIntMapArrayDomainEntryExists
 *--------------------------------------------------------------------------*/

bool
hypredrv_StrIntMapArrayDomainEntryExists(const StrIntMapArray valid, const char *string)
{
   return (hypredrv_StrIntMapArrayGetImage(valid, string) > INT_MIN) != 0;
}

/*-----------------------------------------------------------------------------
 * hypredrv_DofLabelMapCreate
 *-----------------------------------------------------------------------------*/

DofLabelMap *
hypredrv_DofLabelMapCreate(void)
{
   DofLabelMap *map = (DofLabelMap *)malloc(sizeof(DofLabelMap));
   if (!map)
   {
      return NULL;
   }
   map->capacity = 8;
   map->size     = 0;
   map->data     = (DofLabelEntry *)malloc(map->capacity * sizeof(DofLabelEntry));
   if (!map->data)
   {
      free(map);
      return NULL;
   }
   return map;
}

/*-----------------------------------------------------------------------------
 * hypredrv_DofLabelMapAdd
 *-----------------------------------------------------------------------------*/

void
hypredrv_DofLabelMapAdd(DofLabelMap *map, const char *name, int value)
{
   if (!map || !name)
   {
      return;
   }

   if (map->size >= map->capacity)
   {
      size_t         new_capacity = map->capacity * 2;
      DofLabelEntry *new_data =
         (DofLabelEntry *)realloc(map->data, new_capacity * sizeof(DofLabelEntry));
      if (!new_data)
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         return;
      }
      map->data     = new_data;
      map->capacity = new_capacity;
   }

   strncpy(map->data[map->size].name, name, sizeof(map->data[map->size].name) - 1);
   map->data[map->size].name[sizeof(map->data[map->size].name) - 1] = '\0';
   map->data[map->size].value                                       = value;
   map->size++;
}

/*-----------------------------------------------------------------------------
 * hypredrv_DofLabelMapLookup
 *-----------------------------------------------------------------------------*/

int
hypredrv_DofLabelMapLookup(const DofLabelMap *map, const char *name)
{
   if (!map || !name)
   {
      return -1;
   }

   for (size_t i = 0; i < map->size; i++)
   {
      if (!strcmp(map->data[i].name, name))
      {
         return map->data[i].value;
      }
   }
   return -1;
}

/*-----------------------------------------------------------------------------
 * hypredrv_DofLabelMapDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_DofLabelMapDestroy(DofLabelMap **map_ptr)
{
   if (!map_ptr || !*map_ptr)
   {
      return;
   }
   free((*map_ptr)->data);
   free(*map_ptr);
   *map_ptr = NULL;
}
