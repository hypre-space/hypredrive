/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "info.h"

void PrintInfo(void)
{
   time_t t;
   struct tm *tm_info;
   char buffer[100];

   /* Get current time */
   time(&t);
   tm_info = localtime(&t);

   /* Format and print the date and time */
   strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
   printf("Date and time: %s\n", buffer);

   /* Print hypre info */
#if defined(HYPRE_DEVELOP_STRING) && defined(HYPRE_BRANCH_NAME)
   printf("\nUsing HYPRE_DEVELOP_STRING: %s (%s)\n\n",
           HYPRE_DEVELOP_STRING, HYPRE_BRANCH_NAME);

#elif defined(HYPRE_DEVELOP_STRING) && !defined(HYPRE_BRANCH_NAME)
   printf("\nUsing HYPRE_DEVELOP_STRING: %s\n\n",
           HYPRE_DEVELOP_STRING);

#elif defined(HYPRE_RELEASE_VERSION)
   printf("\nUsing HYPRE_RELEASE_VERSION: %s\n\n",
           HYPRE_RELEASE_VERSION);
#endif
}

void PrintExitInfo(char *argv0)
{
   time_t t;
   struct tm *tm_info;
   char buffer[100];

   printf("\%s done!\n", argv0);

   /* Get current time */
   time(&t);
   tm_info = localtime(&t);

   /* Format and print the date and time */
   strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
   printf("Date and time: %s\n", buffer);
}
