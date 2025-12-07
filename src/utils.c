/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "utils.h"
#include <ctype.h>
#include <string.h>

/*-----------------------------------------------------------------------------
 * StrToLowerCase
 *-----------------------------------------------------------------------------*/

char *
StrToLowerCase(char *str)
{
   for (int i = 0; str[i]; i++)
   {
      str[i] = tolower((unsigned char)str[i]);
   }
   return str;
}

/*-----------------------------------------------------------------------------
 * StrTrim
 *-----------------------------------------------------------------------------*/

char *
StrTrim(char *str)
{
   if (!str)
   {
      return NULL;
   }

   for (int i = strlen(str) - 1; i >= 0 && str[i] == ' '; i--)
   {
      str[i] = '\0';
   }

   return str;
}

/*-----------------------------------------------------------------------------
 * CheckBinaryDataExists
 *-----------------------------------------------------------------------------*/

int
CheckBinaryDataExists(const char *prefix)
{
   char  filename[MAX_FILENAME_LENGTH];
   int   file_exists = 0;
   FILE *fp          = NULL;

   /* Check if binary data exist */
   sprintf(filename, "%*s.00000.bin", (int)strlen(prefix), prefix);
   file_exists = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
   if (fp)
   {
      fclose(fp);
   }

   return file_exists;
}

/*-----------------------------------------------------------------------------
 * CheckASCIIDataExists
 *-----------------------------------------------------------------------------*/

int
CheckASCIIDataExists(const char *prefix)
{
   char  filename[MAX_FILENAME_LENGTH];
   int   file_exists = 0;
   FILE *fp          = NULL;

   /* Check if ASCII data exist */
   sprintf(filename, "%*s.00000", (int)strlen(prefix), prefix);
   file_exists = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
   if (fp)
   {
      fclose(fp);
   }

   return file_exists;
}

/*-----------------------------------------------------------------------------
 * CountNumberOfPartitions
 *-----------------------------------------------------------------------------*/

int
CountNumberOfPartitions(const char *prefix)
{
   char filename[MAX_FILENAME_LENGTH];
   int  file_exists = 1;
   int  num_files   = 0;

   if (prefix == NULL)
   {
      return 0;
   }

   while (file_exists)
   {
      FILE *fp = NULL;

      sprintf(filename, "%*s.%05d.bin", (int)strlen(prefix), prefix, num_files);
      file_exists = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
      if (fp)
      {
         fclose(fp);
      }
      if (!file_exists)
      {
         sprintf(filename, "%*s.%05d", (int)strlen(prefix), prefix, num_files);
         file_exists = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
         if (fp)
         {
            fclose(fp);
         }
      }

      num_files++;
   }

   return --num_files;
}

/*-----------------------------------------------------------------------------
 * ComputeNumberOfDigits
 *-----------------------------------------------------------------------------*/

int
ComputeNumberOfDigits(int number)
{
   int ndigits = 0;

   while (number)
   {
      number /= 10;
      ndigits++;
   }

   return ndigits;
}

/*-----------------------------------------------------------------------------
 * SplitFilename
 *-----------------------------------------------------------------------------*/

void
SplitFilename(const char *filename, char **dirname_ptr, char **basename_ptr)
{
   const char *last_slash = strrchr(filename, '/');
   char       *dirname    = NULL;
   char       *basename   = NULL;

   if (last_slash != NULL)
   {
      /* Allocate memory and copy dirname */
      int dirname_length = last_slash - filename;

      dirname = (char *)malloc(dirname_length + 1);
      strncpy(dirname, filename, dirname_length);
      dirname[dirname_length] = '\0';

      /* Allocate memory and copy basename */
      basename = strdup(last_slash + 1);
   }
   else
   {
      /* No slash found, assume current directory and filename is the basename */
      dirname  = strdup(".");
      basename = strdup(filename);
   }

   /* Set output pointers */
   *dirname_ptr  = dirname;
   *basename_ptr = basename;
}

/*-----------------------------------------------------------------------------
 * CombineFilename
 *-----------------------------------------------------------------------------*/

void
CombineFilename(const char *dirname, const char *basename, char **filename_ptr)
{
   size_t length   = 0;
   char  *filename = NULL;

   /* Compute filename length. +2 for the slash and null terminator */
   length = strlen(dirname) + strlen(basename) + 2;

   /* Allocate space for the filename */
   filename = (char *)malloc(length);

   /* Combine dirname and basename */
   if (filename != NULL)
   {
      strcpy(filename, dirname);

      /* Add a slash only if dirname is not empty and does not already end in a slash */
      if (dirname[0] != '\0' && dirname[strlen(dirname) - 1] != '/')
      {
         strcat(filename, "/");
      }

      strcat(filename, basename);
   }

   /* Set output pointer */
   *filename_ptr = filename;
}

/*-----------------------------------------------------------------------------
 * IsYAMLFilename
 *
 * Returns true if string is a filename (no spaces) with a YAML extension
 * (.yaml or .yml)
 *-----------------------------------------------------------------------------*/

bool
IsYAMLFilename(const char *str)
{
   if (!str || *str == '\0')
   {
      return false;
   }

   /* Check for spaces - filenames should not contain spaces */
   if (strchr(str, ' ') != NULL)
   {
      return false;
   }

   const char *dot = strrchr(str, '.');
   if (!dot || dot == str)
   {
      return false;
   }

   /* Check for .yaml or .yml extension */
   const char *ext = dot + 1;
   return (strcmp(ext, "yaml") == 0 || strcmp(ext, "yml") == 0);
}
