/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_COMPATIBILITY_HEADER
#define HYPREDRV_COMPATIBILITY_HEADER

#include "HYPREDRV_config.h"
#include "HYPRE_utilities.h"

/* Provide stable version macros even when older hypre headers don't define them. */
#ifndef HYPREDRV_HYPRE_RELEASE_NUMBER
#ifdef HYPRE_RELEASE_NUMBER
#define HYPREDRV_HYPRE_RELEASE_NUMBER HYPRE_RELEASE_NUMBER
#else
#define HYPREDRV_HYPRE_RELEASE_NUMBER 0
#endif
#endif

#ifndef HYPREDRV_HYPRE_DEVELOP_NUMBER
#ifdef HYPRE_DEVELOP_NUMBER
#define HYPREDRV_HYPRE_DEVELOP_NUMBER HYPRE_DEVELOP_NUMBER
#else
#define HYPREDRV_HYPRE_DEVELOP_NUMBER 0
#endif
#endif

/* Older hypre releases don't provide memory location APIs. */
#if HYPREDRV_HYPRE_RELEASE_NUMBER < 21900
typedef int HYPRE_MemoryLocation;
#define HYPRE_MEMORY_HOST 0
#define HYPRE_MEMORY_DEVICE 0
#define HYPRE_MEMORY_UNDEFINED 0
#endif

/* Older hypre releases may not define HYPRE_BigInt or HYPRE_MPI_BIG_INT. */
#ifndef HYPRE_BigInt
typedef HYPRE_Int HYPRE_BigInt;
#endif
#ifndef HYPRE_MPI_BIG_INT
#define HYPRE_MPI_BIG_INT HYPRE_MPI_INT
#endif

#endif /* HYPREDRV_COMPATIBILITY_HEADER */
