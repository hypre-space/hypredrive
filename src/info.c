/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "info.h"
#include "HYPRE_config.h"

/*--------------------------------------------------------------------------
 * PrintLibInfo
 *--------------------------------------------------------------------------*/

void
PrintLibInfo(void)
{
   time_t      t;
   struct tm  *tm_info;
   char        buffer[100];

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

/*--------------------------------------------------------------------------
 * PrintExitInfo
 *--------------------------------------------------------------------------*/

void
PrintExitInfo(const char *argv0)
{
   time_t t;
   struct tm *tm_info;
   char buffer[100];

   /* Get current time */
   time(&t);
   tm_info = localtime(&t);

   /* Format and print the date and time */
   strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
   printf("Date and time: %s\n\%s done!\n", buffer, argv0);
}
