/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stddef.h>

extern size_t HYPREDRV_JuliaBigIntSize(void);

int
main(void)
{
   size_t bytes = HYPREDRV_JuliaBigIntSize();
   return (bytes == 4u || bytes == 8u) ? 0 : 1;
}
