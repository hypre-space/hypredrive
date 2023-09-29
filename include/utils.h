/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef UTILS_HEADER
#define UTILS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include "HYPRE.h"
#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "error.h"
#include "containers.h"

#define MAX_FILENAME_LENGTH 2048

/*--------------------------------------------------------------------------
 * HYPRE_Int array struct
 *--------------------------------------------------------------------------*/

typedef struct HYPRE_IntArray_struct {
   HYPRE_Int     num_entries;
   HYPRE_Int     num_unique_entries;
   HYPRE_Int    *data;
} HYPRE_IntArray;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

int CheckBinaryDataExists(const char* prefix);
int IntArrayRead(MPI_Comm, const char*, HYPRE_IntArray*);

/*******************************************************************************
 *******************************************************************************/

#define HAVE_COLORS 1

#if defined(HAVE_COLORS)
#define TEXT_RESET      "\033[0m"
#define TEXT_RED        "\033[31m"
#define TEXT_GREEN      "\033[32m"
#define TEXT_YELLOW     "\033[33m"
#define TEXT_BOLD       "\033[1m"
#define TEXT_REDBOLD    "\033[1;31m"
#define TEXT_GREENBOLD  "\033[1;32m"
#define TEXT_YELLOWBOLD "\033[1;33m"
#else
#define TEXT_RESET      ""
#define TEXT_RED        ""
#define TEXT_GREEN      ""
#define TEXT_YELLOW     ""
#define TEXT_BOLD       ""
#define TEXT_REDBOLD    ""
#define TEXT_GREENBOLD  ""
#define TEXT_YELLOWBOLD ""
#endif

#endif /* UTILS_HEADER */
