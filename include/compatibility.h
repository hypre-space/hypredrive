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

/* Feature gate: hypre v3.0.0+ APIs used by tagged GMRES residual/error support. */
#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 30000
#define HYPREDRV_HAVE_HYPRE_30000 1
#else
#define HYPREDRV_HAVE_HYPRE_30000 0
#endif

/* Older hypre releases don't provide memory location APIs. */
#if HYPREDRV_HYPRE_RELEASE_NUMBER < 21900
typedef int HYPRE_MemoryLocation;
#ifndef HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_HOST 0
#endif
#ifndef HYPRE_MEMORY_DEVICE
#define HYPRE_MEMORY_DEVICE 0
#endif
#ifndef HYPRE_MEMORY_UNDEFINED
#define HYPRE_MEMORY_UNDEFINED 0
#endif
#endif

/* Older hypre releases may not define HYPRE_MPI_BIG_INT. */
#ifndef HYPRE_MPI_BIG_INT
#define HYPRE_MPI_BIG_INT HYPRE_MPI_INT
#endif

/* Very old hypre releases may not define HYPRE_BigInt at all. */
#if HYPREDRV_HYPRE_RELEASE_NUMBER < 21900
#ifndef HYPRE_BIGINT
#ifndef HYPRE_MIXEDINT
typedef int HYPRE_BigInt;
#endif
#endif
#endif

#endif /* HYPREDRV_COMPATIBILITY_HEADER */
