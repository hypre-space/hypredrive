/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/containers.h"
#include <limits.h>
#include <stdarg.h>
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

   /* An empty array (size 0) is valid and still produces a header-only file; only
    * a non-empty array with no backing storage is malformed. */
   if (!ia || (ia->size > 0 && !ia->data)) return;

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

   if (size > SIZE_MAX / sizeof(int))
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      return NULL;
   }
   int_array = malloc(sizeof(IntArray));
   if (!int_array)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      return NULL;
   }
   /* Allocate at least one element so data is never NULL for a valid array; this
    * preserves the historical invariant that consumers rely on (an empty dofmap
    * still has a non-NULL data pointer). */
   int_array->data = malloc((size > 0 ? size : 1) * sizeof(int));
   if (!int_array->data)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      free(int_array);
      return NULL;
   }
   int_array->size = size;

   int_array->unique_size = 0;
   int_array->unique_data = NULL;

   int_array->g_unique_size = 0;
   int_array->g_unique_data = NULL;

   return int_array;
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

static const char array_delimiters[] = "[], \t\r\n";

/* Count tokens without copying, then make one writable copy for conversion. */
static char *
ArrayTokensCreate(const char *string, size_t *count)
{
   *count = 0;
   if (!string)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return NULL;
   }

   const char *cursor = string + strspn(string, array_delimiters);
   while (*cursor)
   {
      (*count)++;
      cursor += strcspn(cursor, array_delimiters);
      cursor += strspn(cursor, array_delimiters);
   }

   char *buffer = strdup(string);
   if (!buffer)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
   }
   return buffer;
}

static bool
IntArrayParseTokens(char *buffer, int *values)
{
   char  *saveptr = NULL;
   size_t index   = 0;
   for (const char *token = strtok_r(buffer, array_delimiters, &saveptr); token;
        token             = strtok_r(NULL, array_delimiters, &saveptr))
   {
      if (!hypredrv_ParseInt(token, &values[index++]))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid integer array entry '%s'", token);
         return false;
      }
   }
   return true;
}

void
hypredrv_StrToIntArray(const char *string, IntArray **int_array_ptr)
{
   size_t count   = 0;
   char  *buffer  = ArrayTokensCreate(string, &count);
   *int_array_ptr = NULL;
   if (!buffer)
   {
      return;
   }
   IntArray *array = hypredrv_IntArrayCreate(count);
   if (array && !IntArrayParseTokens(buffer, array->data))
   {
      hypredrv_IntArrayDestroy(&array);
   }
   free(buffer);
   *int_array_ptr = array;
}

/*--------------------------------------------------------------------------
 * hypredrv_DoubleArrayCreate
 *--------------------------------------------------------------------------*/

DoubleArray *
hypredrv_DoubleArrayCreate(size_t size)
{
   DoubleArray *double_array = NULL;

   if (size > SIZE_MAX / sizeof(double))
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      return NULL;
   }
   double_array = malloc(sizeof(DoubleArray));
   if (!double_array)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      return NULL;
   }
   double_array->data = malloc((size > 0 ? size : 1) * sizeof(double));
   if (!double_array->data)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      free(double_array);
      return NULL;
   }
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
   size_t count      = 0;
   char  *buffer     = ArrayTokensCreate(string, &count);
   char  *saveptr    = NULL;
   *double_array_ptr = NULL;
   if (!buffer)
   {
      return;
   }
   DoubleArray *array = hypredrv_DoubleArrayCreate(count);
   if (array)
   {
      size_t index = 0;
      for (const char *token = strtok_r(buffer, array_delimiters, &saveptr); token;
           token             = strtok_r(NULL, array_delimiters, &saveptr))
      {
         if (!hypredrv_ParseDouble(token, &array->data[index++]))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid floating-point array entry '%s'", token);
            hypredrv_DoubleArrayDestroy(&array);
            break;
         }
      }
   }
   free(buffer);
   *double_array_ptr = array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_StrToStackIntArray
 *-----------------------------------------------------------------------------*/

void
hypredrv_StrToStackIntArray(const char *string, StackIntArray *int_array)
{
   size_t count  = 0;
   char  *buffer = ArrayTokensCreate(string, &count);
   if (!buffer)
   {
      return;
   }
   if (count > MAX_STACK_ARRAY_LENGTH)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Integer array exceeds %d entries", MAX_STACK_ARRAY_LENGTH);
   }
   else
   {
      StackIntArray parsed = STACK_INTARRAY_CREATE();
      if (IntArrayParseTokens(buffer, parsed.data))
      {
         parsed.size = count;
         *int_array  = parsed;
      }
   }
   free(buffer);
}

/* Sort and compact a scratch array, returning its number of distinct values. */
static int
IntArrayCompare(const void *a, const void *b)
{
   int lhs = *(const int *)a;
   int rhs = *(const int *)b;
   return (lhs > rhs) - (lhs < rhs);
}

static size_t
IntArrayCompact(int *data, size_t size)
{
   size_t count = 0;
   qsort(data, size, sizeof(int), IntArrayCompare);
   for (size_t i = 0; i < size; i++)
   {
      if (count == 0 || data[i] != data[count - 1])
      {
         data[count++] = data[i];
      }
   }
   return count;
}

/* Keep every rank on the same path when validation or allocation fails. */
static bool
IntArrayCollectiveCheck(MPI_Comm comm, uint32_t local_error)
{
   const uint32_t send_error   = local_error;
   uint32_t       global_error = ERROR_NONE;
   MPI_Allreduce(&send_error, &global_error, 1, MPI_UINT32_T, MPI_BOR, comm);
   global_error |= local_error;
   if (global_error)
   {
      hypredrv_ErrorCodeSet(global_error);
   }
   return global_error == ERROR_NONE;
}

/* Compute root's Gatherv layout and allocate its concatenated label buffer. */
static uint32_t
IntArrayGatherBuffer(int nprocs, const int *counts, int *displs, int **gathered,
                     int *total)
{
   *total = 0;
   for (int i = 0; i < nprocs; i++)
   {
      if (counts[i] > INT_MAX - *total)
      {
         return ERROR_OUT_OF_BOUNDS;
      }
      displs[i] = *total;
      *total += counts[i];
   }
   if ((size_t)*total > SIZE_MAX / sizeof(int))
   {
      return ERROR_ALLOCATION;
   }
   *gathered = malloc((size_t)(*total ? *total : 1) * sizeof(int));
   return *gathered ? ERROR_NONE : ERROR_ALLOCATION;
}

static bool
IntArrayUnique(MPI_Comm comm, IntArray *array)
{
   int       rank = 0, nprocs = 0;
   int       local_count = 0, global_count = 0, total = 0;
   int      *counts = NULL, *displs = NULL, *gathered = NULL;
   IntArray *scratch = hypredrv_IntArrayCreate(array->size);
   bool      success = false;
   uint32_t  error   = ERROR_NONE;

   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);
   if (!IntArrayCollectiveCheck(comm, scratch ? ERROR_NONE : ERROR_ALLOCATION))
   {
      goto cleanup;
   }
   memcpy(scratch->data, array->data, array->size * sizeof(int));
   array->unique_size = IntArrayCompact(scratch->data, array->size);
   array->unique_data =
      malloc((array->unique_size ? array->unique_size : 1) * sizeof(int));
   if (rank == 0)
   {
      counts = malloc((size_t)nprocs * sizeof(int));
      displs = malloc((size_t)nprocs * sizeof(int));
   }
   if (!IntArrayCollectiveCheck(
          comm, (!array->unique_data || (rank == 0 && (!counts || !displs)))
                   ? ERROR_ALLOCATION
                   : ERROR_NONE))
   {
      goto cleanup;
   }
   memcpy(array->unique_data, scratch->data, array->unique_size * sizeof(int));
   hypredrv_IntArrayDestroy(&scratch);
   if (!IntArrayCollectiveCheck(comm, array->unique_size > INT_MAX ? ERROR_OUT_OF_BOUNDS
                                                                   : ERROR_NONE))
   {
      goto cleanup;
   }
   local_count = (int)array->unique_size;
   MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, comm);

   if (rank == 0)
   {
      error = IntArrayGatherBuffer(nprocs, counts, displs, &gathered, &total);
   }
   if (!IntArrayCollectiveCheck(comm, error))
   {
      goto cleanup;
   }
   MPI_Gatherv(array->unique_data, local_count, MPI_INT, gathered, counts, displs,
               MPI_INT, 0, comm);
   if (rank == 0)
   {
      global_count = (int)IntArrayCompact(gathered, (size_t)total);
   }
   MPI_Bcast(&global_count, 1, MPI_INT, 0, comm);
   array->g_unique_size = (size_t)global_count;
   array->g_unique_data = malloc((size_t)(global_count ? global_count : 1) * sizeof(int));
   if (!IntArrayCollectiveCheck(comm,
                                array->g_unique_data ? ERROR_NONE : ERROR_ALLOCATION))
   {
      goto cleanup;
   }
   if (rank == 0)
   {
      memcpy(array->g_unique_data, gathered, (size_t)global_count * sizeof(int));
   }
   MPI_Bcast(array->g_unique_data, global_count, MPI_INT, 0, comm);
   success = true;

cleanup:
   hypredrv_IntArrayDestroy(&scratch);
   free(counts);
   free(displs);
   free(gathered);
   return success;
}

/* Read one part, either its size alone or its entries into the remaining buffer. */
static bool
IntArrayReadPart(const char *prefix, int part, bool binary, int *data, size_t capacity,
                 size_t *size)
{
   char filename[MAX_FILENAME_LENGTH];
   int  written = snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, part,
                          binary ? ".bin" : "");
   if (written < 0 || (size_t)written >= sizeof(filename))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Dofmap filename is too long");
      return false;
   }
   FILE *fp = fopen(filename, binary ? "rb" : "r");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(filename);
      return false;
   }

   bool   success = false;
   size_t count =
      binary ? fread(size, sizeof(size_t), 1, fp) : (size_t)fscanf(fp, "%zu", size);
   if (count != 1 || *size > capacity)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid or oversized dofmap header in '%s'", filename);
      goto cleanup;
   }
   if (data)
   {
      if (binary)
      {
         count = fread(data, sizeof(int), *size, fp);
      }
      else
      {
         count = 0;
         while (count < *size && fscanf(fp, "%d", &data[count]) == 1)
         {
            count++;
         }
      }
      if (count != *size)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Expected %zu, but found %zu coefficients in '%s'", *size,
                              count, filename);
         goto cleanup;
      }
   }
   success = true;
cleanup:
   fclose(fp);
   return success;
}

void
hypredrv_IntArrayParRead(MPI_Comm comm, const char *prefix, IntArray **int_array_ptr)
{
   int      rank = 0, nprocs = 0;
   uint64_t first = 0, nparts = 0;
   size_t   total = 0, size = 0, offset = 0;
   bool     success = true;
   *int_array_ptr   = NULL;

   if (!IntArrayCollectiveCheck(comm, hypredrv_BinaryPathPrefixIsSafe(prefix)
                                         ? ERROR_NONE
                                         : ERROR_FILE_UNEXPECTED_ENTRY))
   {
      return;
   }
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);
   int num_parts = hypredrv_CountNumberOfPartitions(prefix);
   if (!IntArrayCollectiveCheck(comm, num_parts < nprocs ? ERROR_FILE_UNEXPECTED_ENTRY
                                                         : ERROR_NONE))
   {
      hypredrv_ErrorMsgAdd("Invalid dofmap filename \"%s\" or invalid number of parts!",
                           prefix);
      return;
   }
   hypredrv_MultipartRange((uint64_t)num_parts, nprocs, rank, &first, &nparts);
   bool binary = hypredrv_CheckBinaryDataExists(prefix) != 0;

   for (uint64_t part = first; part < first + nparts; part++)
   {
      success = IntArrayReadPart(prefix, (int)part, binary, NULL,
                                 SIZE_MAX / sizeof(int) - total, &size);
      if (!success) break;
      total += size;
   }
   if (!IntArrayCollectiveCheck(comm, success ? ERROR_NONE : hypredrv_ErrorCodeGet()))
   {
      return;
   }
   IntArray *array = hypredrv_IntArrayCreate(total);
   if (!IntArrayCollectiveCheck(comm, array ? ERROR_NONE : ERROR_ALLOCATION))
   {
      hypredrv_IntArrayDestroy(&array);
      return;
   }
   for (uint64_t part = first; part < first + nparts; part++)
   {
      success = IntArrayReadPart(prefix, (int)part, binary, array->data + offset,
                                 total - offset, &size);
      if (!success) break;
      offset += size;
   }
   if (success && offset != total)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Dofmap size changed between sizing and read passes");
      success = false;
   }
   if (!IntArrayCollectiveCheck(comm, success ? ERROR_NONE : hypredrv_ErrorCodeGet()) ||
       !IntArrayUnique(comm, array))
   {
      hypredrv_IntArrayDestroy(&array);
   }
   *int_array_ptr = array;
}

/*-----------------------------------------------------------------------------
 * hypredrv_IntArrayBuild
 *-----------------------------------------------------------------------------*/

void
hypredrv_IntArrayBuild(MPI_Comm comm, int size, const int *dofmap,
                       IntArray **int_array_ptr)
{
   *int_array_ptr = NULL;
   if (!IntArrayCollectiveCheck(
          comm, size < 0 || (size > 0 && !dofmap) ? ERROR_INVALID_VAL : ERROR_NONE))
   {
      return;
   }
   IntArray *array = hypredrv_IntArrayCreate((size_t)size);
   if (!IntArrayCollectiveCheck(comm, array ? ERROR_NONE : ERROR_ALLOCATION))
   {
      hypredrv_IntArrayDestroy(&array);
      return;
   }
   /* Empty ranks still participate in building the global label set. */
   if (size > 0)
   {
      memcpy(array->data, dofmap, (size_t)size * sizeof(int));
   }
   if (!IntArrayUnique(comm, array))
   {
      hypredrv_IntArrayDestroy(&array);
   }
   *int_array_ptr = array;
}

static void
IntArrayBuildPattern(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                     bool interleaved, IntArray **int_array_ptr)
{
   *int_array_ptr = NULL;
   uint32_t error = ERROR_NONE;
   if (num_local_blocks < 0 || num_dof_types < 0)
   {
      error = ERROR_INVALID_VAL;
   }
   else if (num_dof_types > 0 && num_local_blocks > INT_MAX / num_dof_types)
   {
      error = ERROR_OUT_OF_BOUNDS;
   }
   if (!IntArrayCollectiveCheck(comm, error))
   {
      return;
   }
   int       size  = num_local_blocks * num_dof_types;
   IntArray *array = hypredrv_IntArrayCreate((size_t)size);
   if (!IntArrayCollectiveCheck(comm, array ? ERROR_NONE : ERROR_ALLOCATION))
   {
      hypredrv_IntArrayDestroy(&array);
      return;
   }
   for (int i = 0; i < size; i++)
   {
      array->data[i] = interleaved ? i % num_dof_types : i / num_local_blocks;
   }
   if (!IntArrayUnique(comm, array))
   {
      hypredrv_IntArrayDestroy(&array);
   }
   *int_array_ptr = array;
}

void
hypredrv_IntArrayBuildInterleaved(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                                  IntArray **int_array_ptr)
{
   IntArrayBuildPattern(comm, num_local_blocks, num_dof_types, true, int_array_ptr);
}

void
hypredrv_IntArrayBuildContiguous(MPI_Comm comm, int num_local_blocks, int num_dof_types,
                                 IntArray **int_array_ptr)
{
   IntArrayBuildPattern(comm, num_local_blocks, num_dof_types, false, int_array_ptr);
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
   int number = 0;
   if (!string)
   {
      return INT_MIN;
   }
   bool is_integer = hypredrv_ParseInt(string, &number);
   for (size_t i = 0; i < valid.size; i++)
   {
      if (is_integer ? valid.data[i].num == number : !strcmp(valid.data[i].str, string))
      {
         return valid.data[i].num;
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

/*--------------------------------------------------------------------------
 * StrBufferAppend
 *--------------------------------------------------------------------------*/

static void
StrBufferAppend(char **pos, size_t *remaining, const char *format, ...)
{
   va_list args;
   int     written = 0;

   va_start(args, format);
   written = vsnprintf(*pos, *remaining, format, args);
   va_end(args);

   /* GCOVR_EXCL_BR_START */
   if (written < 0 || (size_t)written >= *remaining) /* GCOVR_EXCL_BR_STOP */
   {
      *remaining = 0; /* GCOVR_EXCL_LINE */
      return;         /* GCOVR_EXCL_LINE */
   }

   *pos += written;
   *remaining -= (size_t)written;
}

/*--------------------------------------------------------------------------
 * hypredrv_StrIntMapArrayDomainToString
 *
 * Builds a comma-separated list of the printable domain entries of a map,
 * grouping aliases that share the same numeric image with a slash, e.g.:
 * "none (-1), single/jacobi (7)". Returns a heap string owned by the
 * caller, or NULL when the map has no printable entry.
 *--------------------------------------------------------------------------*/

char *
hypredrv_StrIntMapArrayDomainToString(const StrIntMapArray valid)
{
   size_t length = 1;
   size_t count  = 0;

   for (size_t i = 0; i < valid.size; i++)
   {
      if (!valid.data[i].str || valid.data[i].str[0] == '\0')
      {
         continue;
      }
      length += strlen(valid.data[i].str) + 16; /* separators plus " (num)" */
      count++;
   }

   if (count == 0)
   {
      return NULL;
   }

   char *buffer = (char *)malloc(length);
   /* GCOVR_EXCL_BR_START */
   if (!buffer) /* GCOVR_EXCL_BR_STOP */
   {
      return NULL; /* GCOVR_EXCL_LINE */
   }

   char  *pos       = buffer;
   size_t remaining = length;
   for (size_t i = 0; i < valid.size; i++)
   {
      const char *str          = valid.data[i].str;
      int         num          = valid.data[i].num;
      bool        already_seen = false;

      if (!str || str[0] == '\0')
      {
         continue;
      }

      /* Skip entries whose alias group has already been emitted */
      for (size_t j = 0; j < i; j++)
      {
         if (valid.data[j].num == num && valid.data[j].str &&
             valid.data[j].str[0] != '\0')
         {
            already_seen = true;
            break;
         }
      }
      if (already_seen)
      {
         continue;
      }

      StrBufferAppend(&pos, &remaining, "%s%s", (pos == buffer) ? "" : ", ", str);

      /* Join the remaining aliases mapping to the same numeric image */
      for (size_t j = i + 1; j < valid.size; j++)
      {
         if (valid.data[j].num == num && valid.data[j].str &&
             valid.data[j].str[0] != '\0')
         {
            StrBufferAppend(&pos, &remaining, "/%s", valid.data[j].str);
         }
      }

      StrBufferAppend(&pos, &remaining, " (%d)", num);
   }

   return buffer;
}

/*--------------------------------------------------------------------------
 * hypredrv_StrArrayToString
 *
 * Builds a comma-separated list of the entries of a string array, e.g.:
 * "max_iter, tolerance". Returns a heap string owned by the caller, or
 * NULL when the array has no printable entry.
 *--------------------------------------------------------------------------*/

char *
hypredrv_StrArrayToString(const StrArray valid)
{
   size_t length = 1;
   size_t count  = 0;

   for (size_t i = 0; i < valid.size; i++)
   {
      if (!valid.data[i] || valid.data[i][0] == '\0')
      {
         continue;
      }
      length += strlen(valid.data[i]) + 2; /* ", " separator */
      count++;
   }

   if (count == 0)
   {
      return NULL;
   }

   char *buffer = (char *)malloc(length);
   /* GCOVR_EXCL_BR_START */
   if (!buffer) /* GCOVR_EXCL_BR_STOP */
   {
      return NULL; /* GCOVR_EXCL_LINE */
   }

   char  *pos       = buffer;
   size_t remaining = length;
   for (size_t i = 0; i < valid.size; i++)
   {
      if (!valid.data[i] || valid.data[i][0] == '\0')
      {
         continue;
      }
      StrBufferAppend(&pos, &remaining, "%s%s", (pos == buffer) ? "" : ", ",
                      valid.data[i]);
   }

   return buffer;
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
