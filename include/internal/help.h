/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HELP_HEADER
#define HELP_HEADER

#include <stddef.h>
#include <stdio.h>

int hypredrv_HelpRequested(int argc, char **argv, char *topic, size_t topic_size);
int hypredrv_HelpPrint(FILE *out, const char *argv0, const char *topic);

#endif /* HELP_HEADER */
