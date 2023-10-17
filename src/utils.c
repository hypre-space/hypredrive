/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "utils.h"
#include <ctype.h>

/*-----------------------------------------------------------------------------
 * StrToLowerCase
 *-----------------------------------------------------------------------------*/

char*
StrToLowerCase(char* str)
{
   for (int i = 0; str[i]; i++)
   {
      str[i] = tolower((unsigned char) str[i]);
   }
   return str;
}

/*-----------------------------------------------------------------------------
 * StrTrim
 *-----------------------------------------------------------------------------*/

char*
StrTrim(char* str)
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
