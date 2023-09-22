/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "utils.h"

/*-----------------------------------------------------------------------------
 * CheckBinaryDataExists
 *-----------------------------------------------------------------------------*/

int
CheckBinaryDataExists(const char* prefix)
{
   char   filename[MAX_FILENAME_LENGTH];
   int    is_binary;
   FILE  *fp;

   /* Check if binary data exist */
   sprintf(filename, "%s.00000.bin", prefix);
   is_binary = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
   if (fp) fclose(fp);

   return is_binary;
}

/*-----------------------------------------------------------------------------
 * IntArrayReadASCII
 *-----------------------------------------------------------------------------*/

int
IntArrayReadASCII(int id, const char* prefix, HYPRE_Int *num_entries_ptr, HYPRE_Int **array_ptr)
{
   char    filename[MAX_FILENAME_LENGTH];
   int     num_entries;
   int    *array;
   FILE   *fp;

   sprintf(filename, "%s.%05d", prefix, id);
   if ((fp = fopen(filename, "r")) == NULL)
   {
      ErrorMsgAddInvalidFilename(filename);
      return EXIT_FAILURE;
   }

   if (sizeof(HYPRE_Int) == 8)
   {
      return EXIT_FAILURE;
   }

   if (fscanf(fp, "%d", &num_entries) != 1) return EXIT_FAILURE;
   array = (HYPRE_Int*) malloc(num_entries * sizeof(HYPRE_Int));
   for (int i = 0; i < num_entries; i++) {if (fscanf(fp, "%d", &array[i]) != 1) return EXIT_FAILURE; }
   if (fp) fclose(fp);

   *num_entries_ptr = num_entries;
   *array_ptr = array;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * IntArrayReadBinary
 *-----------------------------------------------------------------------------*/

int
IntArrayReadBinary(int id, const char* prefix, HYPRE_Int *num_entries_ptr, HYPRE_Int **array_ptr)
{
   char    filename[MAX_FILENAME_LENGTH];
   size_t  num_entries;
   int    *array;
   FILE   *fp;

   sprintf(filename, "%s.%05d.bin", prefix, id);
   if ((fp = fopen(filename, "rb")) == NULL)
   {
      ErrorMsgAddInvalidFilename(filename);
      return EXIT_FAILURE;
   }

   if (fread(&num_entries, sizeof(HYPRE_Int), 1, fp) != 1) return EXIT_FAILURE;
   array = (HYPRE_Int*) malloc(num_entries * sizeof(HYPRE_Int));
   if (fread(array, sizeof(HYPRE_Int), num_entries, fp) != num_entries) return EXIT_FAILURE;
   if (fp) fclose(fp);

   *num_entries_ptr = (HYPRE_Int) num_entries;
   *array_ptr = array;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * IntArrayCompare
 *-----------------------------------------------------------------------------*/

int
IntArrayCompare(const void *a, const void *b)
{
   return (*(HYPRE_Int *)a - *(HYPRE_Int *)b);
}

/*-----------------------------------------------------------------------------
 * IntArrayRead
 *-----------------------------------------------------------------------------*/

int
IntArrayRead(MPI_Comm comm, const char* prefix, HYPRE_IntArray *int_array)
{
   int        myid;
   HYPRE_Int  num_unique_entries;
   HYPRE_Int *temp;

   MPI_Comm_rank(comm, &myid);

   if (CheckBinaryDataExists(prefix))
   {
      IntArrayReadBinary(myid, prefix, &int_array->num_entries, &int_array->data);
   }
   else
   {
      IntArrayReadASCII(myid, prefix, &int_array->num_entries, &int_array->data);
   }

   /* Find largest entry */
   temp = (HYPRE_Int*) malloc(int_array->num_entries*sizeof(HYPRE_Int));
   memcpy(temp, int_array->data, int_array->num_entries * sizeof(HYPRE_Int));
   qsort(temp, int_array->num_entries, sizeof(HYPRE_Int), IntArrayCompare);
   num_unique_entries = 1;
   for (int i = 1; i < int_array->num_entries; i++)
   {
      if (temp[i] != temp[i-1])
      {
         num_unique_entries++;
      }
   }
   free(temp);
   MPI_Allreduce(&num_unique_entries, &(int_array->num_unique_entries),
                 1, MPI_INT, MPI_MAX, comm);

   /* TODO - Fix num_unique_entries computation */

   return EXIT_SUCCESS;
}
