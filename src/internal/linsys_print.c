/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>
#include "internal/error.h"
#include "internal/linsys.h"
#include "logging.h"

/* Which optional keys the YAML block actually carried; the cross-field rules at
 * the end of parsing depend on presence, not just on the parsed value. */
typedef struct
{
   int every;
   int ids;
   int ranges;
   int threshold;
   int selectors;
} PrintSystemSeenKeys;

static const char *PrintSystemStageName(int stage);

void
hypredrv_PrintSystemSetDefaultArgs(PrintSystem_args *args)
{
   if (!args)
   {
      return;
   }

   args->enabled    = 0;
   args->type       = PRINT_SYSTEM_TYPE_ALL;
   args->stage_mask = PRINT_SYSTEM_STAGE_BUILD_BIT;
   args->artifacts  = PRINT_SYSTEM_ARTIFACT_MATRIX | PRINT_SYSTEM_ARTIFACT_RHS |
                     PRINT_SYSTEM_ARTIFACT_DOFMAP;
   snprintf(args->output_dir, sizeof(args->output_dir), "%s", "hypredrive-data");
   args->overwrite          = 0;
   args->next_dump_index    = 0;
   args->overwrite_prepared = 0;

   args->every       = 1;
   args->threshold   = 0.0;
   args->ids         = NULL;
   args->ranges.data = NULL;
   args->ranges.size = 0;

   args->selectors     = NULL;
   args->num_selectors = 0;
}

static void
PrintSystemRangeArrayDestroy(IntRangeArray *ranges)
{
   /* GCOVR_EXCL_BR_START */
   if (!ranges) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   free(ranges->data);
   ranges->data = NULL;
   ranges->size = 0;
}

static void
PrintSystemSelectorDestroy(DumpSelector_args *selector)
{
   /* GCOVR_EXCL_BR_START */
   if (!selector) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   hypredrv_IntArrayDestroy(&selector->ids);
   PrintSystemRangeArrayDestroy(&selector->ranges);
   selector->every     = 0;
   selector->threshold = 0.0;
   selector->basis     = PRINT_SYSTEM_BASIS_LINEAR_SYSTEM;
   selector->level     = 0;
}

void
hypredrv_PrintSystemDestroyArgs(PrintSystem_args *args)
{
   if (!args)
   {
      return;
   }

   hypredrv_IntArrayDestroy(&args->ids);
   PrintSystemRangeArrayDestroy(&args->ranges);

   if (args->selectors)
   {
      for (size_t i = 0; i < args->num_selectors; i++)
      {
         PrintSystemSelectorDestroy(&args->selectors[i]);
      }
      free(args->selectors);
      args->selectors = NULL;
   }
   args->num_selectors = 0;
}

static int
PrintSystemParseOnOff(const char *value, int *out)
{
   /* GCOVR_EXCL_BR_START */
   if (!value || !out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (!strcasecmp(value, "on") || !strcasecmp(value, "yes") ||
       /* GCOVR_EXCL_BR_START */
       !strcasecmp(value, "true") || !strcmp(value, "1"))
   /* GCOVR_EXCL_BR_STOP */
   {
      *out = 1;
      return 1;
   }
   /* GCOVR_EXCL_BR_START */
   if (!strcasecmp(value, "off") || !strcasecmp(value, "no") ||
       /* GCOVR_EXCL_BR_STOP */
       /* GCOVR_EXCL_BR_START */
       !strcasecmp(value, "false") || !strcmp(value, "0"))
   /* GCOVR_EXCL_BR_STOP */
   {
      *out = 0;
      return 1;
   }

   return 0;
}

static int
PrintSystemParseInteger(const char *value, int *out)
{
   /* GCOVR_EXCL_BR_START */
   if (!value || !out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char *endptr = NULL;
   long  parsed = strtol(value, &endptr, 10);
   if (endptr == value)
   {
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   while (*endptr && isspace((unsigned char)*endptr)) /* GCOVR_EXCL_BR_STOP */
   {
      endptr++; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */
   if (*endptr != '\0' || parsed < INT_MIN || parsed > INT_MAX) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   *out = (int)parsed;
   return 1;
}

static int
PrintSystemParseDouble(const char *value, double *out)
{
   /* GCOVR_EXCL_BR_START */
   if (!value || !out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char  *endptr = NULL;
   double parsed = strtod(value, &endptr);
   /* GCOVR_EXCL_BR_START */
   if (endptr == value) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */
   while (*endptr && isspace((unsigned char)*endptr)) /* GCOVR_EXCL_BR_STOP */
   {
      endptr++; /* GCOVR_EXCL_LINE */
   }
   if (*endptr != '\0')
   {
      return 0;
   }

   *out = parsed;
   return 1;
}

static int
PrintSystemRangeArrayAppend(IntRangeArray *ranges, int begin, int end)
{
   /* GCOVR_EXCL_BR_START */
   if (!ranges) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (begin > end)
   {
      int tmp = begin;
      begin   = end;
      end     = tmp;
   }

   IntRange *new_data =
      (IntRange *)realloc(ranges->data, (ranges->size + 1) * sizeof(IntRange));
   /* GCOVR_EXCL_BR_START */
   if (!new_data) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);               /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate range list"); /* GCOVR_EXCL_LINE */
      return 0;                                              /* GCOVR_EXCL_LINE */
   }

   ranges->data                     = new_data;
   ranges->data[ranges->size].begin = begin;
   ranges->data[ranges->size].end   = end;
   ranges->size++;
   return 1;
}

/* Advances past any run of whitespace. */
static void
PrintSystemSkipSpaces(const char **p)
{
   /* GCOVR_EXCL_BR_START */
   while (**p && isspace((unsigned char)**p)) /* GCOVR_EXCL_BR_STOP */
   {
      (*p)++; /* GCOVR_EXCL_LINE */
   }
}

/* Accepts "[a,b]" plus the unbracketed and '-'/':'-separated spellings. The
 * input is only scanned, never modified, so no working copy is needed. */
static int
PrintSystemParseRangePair(const char *value, int *begin, int *end)
{
   const char *p      = value;
   char       *endptr = NULL;
   long        first  = 0;
   long        second = 0;

   /* GCOVR_EXCL_BR_START */
   if (!value || !begin || !end) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   PrintSystemSkipSpaces(&p);
   /* GCOVR_EXCL_BR_START */
   if (*p == '[') /* GCOVR_EXCL_BR_STOP */
   {
      p++;
   }

   first = strtol(p, &endptr, 10);
   if (endptr == p)
   {
      return 0;
   }
   p = endptr;
   PrintSystemSkipSpaces(&p);

   /* GCOVR_EXCL_BR_START */
   if (*p != ',' && *p != '-' && *p != ':') /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   p++;
   PrintSystemSkipSpaces(&p);

   second = strtol(p, &endptr, 10);
   /* GCOVR_EXCL_BR_START */
   if (endptr == p) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   p = endptr;
   PrintSystemSkipSpaces(&p);

   if (*p == ']')
   {
      p++;
   }
   PrintSystemSkipSpaces(&p);
   if (*p != '\0')
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (first < INT_MIN || first > INT_MAX || second < INT_MIN || second > INT_MAX)
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   *begin = (int)first;
   *end   = (int)second;

   return 1;
}

static int
PrintSystemParseIntArrayNode(const YAMLnode *node, IntArray **out)
{
   /* GCOVR_EXCL_BR_START */
   if (!node || !out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   hypredrv_IntArrayDestroy(out);
   if (node->children)
   {
      size_t count = 0;
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         /* GCOVR_EXCL_BR_START */
         if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
         {
            count++;
         }
      }

      IntArray *ids = hypredrv_IntArrayCreate(count);
      /* GCOVR_EXCL_BR_START */
      if (!ids) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);            /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("Failed to allocate id list"); /* GCOVR_EXCL_LINE */
         return 0;                                           /* GCOVR_EXCL_LINE */
      }

      size_t idx = 0;
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         /* GCOVR_EXCL_BR_START */
         if (strcmp(item->key, "-") != 0) /* GCOVR_EXCL_BR_STOP */
         {
            continue; /* GCOVR_EXCL_LINE */
         }

         int parsed = 0;
         /* GCOVR_EXCL_BR_START */
         const char *value = item->mapped_val ? item->mapped_val : item->val;
         /* GCOVR_EXCL_BR_STOP */
         if (!PrintSystemParseInteger(value, &parsed))
         {
            hypredrv_IntArrayDestroy(&ids);
            return 0;
         }
         ids->data[idx++] = parsed;
      }

      *out = ids;
      return 1;
   }

   /* GCOVR_EXCL_BR_START */
   const char *value = node->mapped_val ? node->mapped_val : node->val;
   /* GCOVR_EXCL_BR_STOP */
   /* GCOVR_EXCL_BR_START */
   if (!value) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   hypredrv_StrToIntArray(value, out);
   return (*out != NULL);
}

static int
PrintSystemParseRangesNode(const YAMLnode *node, IntRangeArray *ranges)
{
   /* GCOVR_EXCL_BR_START */
   if (!node || !ranges) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   PrintSystemRangeArrayDestroy(ranges);
   if (node->children)
   {
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         /* GCOVR_EXCL_BR_START */
         if (strcmp(item->key, "-") != 0) /* GCOVR_EXCL_BR_STOP */
         {
            continue; /* GCOVR_EXCL_LINE */
         }
         /* GCOVR_EXCL_BR_START */
         const char *value = item->mapped_val ? item->mapped_val : item->val;
         /* GCOVR_EXCL_BR_STOP */
         int begin = 0;
         int end   = 0;
         if (!PrintSystemParseRangePair(value, &begin, &end))
         {
            return 0;
         }
         /* GCOVR_EXCL_BR_START */
         if (!PrintSystemRangeArrayAppend(ranges, begin, end)) /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
      }
      return ranges->size > 0;
   }

   /* GCOVR_EXCL_BR_START */
   const char *value = node->mapped_val ? node->mapped_val : node->val;
   /* GCOVR_EXCL_BR_STOP */
   int begin = 0;
   int end   = 0;
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemParseRangePair(value, &begin, &end)) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   return PrintSystemRangeArrayAppend(ranges, begin, end);
}

static int
PrintSystemArtifactBitFromName(const char *token, int *bit_out)
{
   /* GCOVR_EXCL_BR_START */
   if (!token || !bit_out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (!strcasecmp(token, "matrix"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_MATRIX;
      return 1;
   }
   if (!strcasecmp(token, "precmat"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_PRECMAT;
      return 1;
   }
   if (!strcasecmp(token, "rhs"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_RHS;
      return 1;
   }
   if (!strcasecmp(token, "x0"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_X0;
      return 1;
   }
   if (!strcasecmp(token, "xref"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_XREF;
      return 1;
   }
   if (!strcasecmp(token, "solution"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_SOLUTION;
      return 1;
   }
   if (!strcasecmp(token, "dofmap"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_DOFMAP;
      return 1;
   }
   if (!strcasecmp(token, "metadata"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_METADATA;
      return 1;
   }
   if (!strcasecmp(token, "all"))
   {
      *bit_out = PRINT_SYSTEM_ARTIFACT_MATRIX | PRINT_SYSTEM_ARTIFACT_PRECMAT |
                 PRINT_SYSTEM_ARTIFACT_RHS | PRINT_SYSTEM_ARTIFACT_X0 |
                 PRINT_SYSTEM_ARTIFACT_XREF | PRINT_SYSTEM_ARTIFACT_SOLUTION |
                 PRINT_SYSTEM_ARTIFACT_DOFMAP | PRINT_SYSTEM_ARTIFACT_METADATA;
      return 1;
   }

   return 0;
}

static int
PrintSystemParseArtifactsNode(const YAMLnode *node, int *artifacts_out)
{
   /* GCOVR_EXCL_BR_START */
   if (!node || !artifacts_out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   int artifacts = 0;
   if (node->children)
   {
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         if (strcmp(item->key, "-") != 0)
         {
            continue;
         }
         /* GCOVR_EXCL_BR_START */
         const char *token = item->mapped_val ? item->mapped_val : item->val;
         /* GCOVR_EXCL_BR_STOP */
         int bit = 0;
         /* GCOVR_EXCL_BR_START */
         if (!PrintSystemArtifactBitFromName(token, &bit)) /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         artifacts |= bit;
      }
   }
   else
   {
      /* GCOVR_EXCL_BR_START */
      const char *value = node->mapped_val ? node->mapped_val : node->val;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      if (!value) /* GCOVR_EXCL_BR_STOP */
      {
         return 0;
      }

      char *buffer = strdup(value);
      /* GCOVR_EXCL_BR_START */
      if (!buffer) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate artifact parser buffer"); /* GCOVR_EXCL_LINE */
         return 0;                                        /* GCOVR_EXCL_LINE */
      }
      char       *saveptr = NULL;
      const char *token   = strtok_r(buffer, "[], ", &saveptr);
      while (token)
      {
         int bit = 0;
         if (!PrintSystemArtifactBitFromName(token, &bit))
         {
            free(buffer);
            return 0;
         }
         artifacts |= bit;
         token = strtok_r(NULL, "[], ", &saveptr);
      }
      free(buffer);
   }

   if (!artifacts)
   {
      return 0;
   }

   *artifacts_out = artifacts;
   return 1;
}

static int
PrintSystemParseBasis(const char *value, int *basis_out)
{
   /* GCOVR_EXCL_BR_START */
   if (!value || !basis_out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */
   if (!strcasecmp(value, "ids") || !strcasecmp(value, "linear_system"))
   /* GCOVR_EXCL_BR_STOP */
   {
      *basis_out = PRINT_SYSTEM_BASIS_LINEAR_SYSTEM;
      return 1;
   }
   if (!strcasecmp(value, "timestep"))
   {
      *basis_out = PRINT_SYSTEM_BASIS_TIMESTEP;
      return 1;
   }
   if (!strcasecmp(value, "level"))
   {
      *basis_out = PRINT_SYSTEM_BASIS_LEVEL;
      return 1;
   }
   if (!strcasecmp(value, "iterations"))
   {
      *basis_out = PRINT_SYSTEM_BASIS_ITERATIONS;
      return 1;
   }
   if (!strcasecmp(value, "setup_time"))
   {
      *basis_out = PRINT_SYSTEM_BASIS_SETUP_TIME;
      return 1;
   }
   /* GCOVR_EXCL_BR_START */
   if (!strcasecmp(value, "solve_time")) /* GCOVR_EXCL_BR_STOP */
   {
      *basis_out = PRINT_SYSTEM_BASIS_SOLVE_TIME;
      return 1;
   }

   return 0; /* GCOVR_EXCL_LINE */
}

static int
PrintSystemBasisUsesThreshold(int basis)
{
   return basis == PRINT_SYSTEM_BASIS_ITERATIONS ||
          basis == PRINT_SYSTEM_BASIS_SETUP_TIME ||
          basis == PRINT_SYSTEM_BASIS_SOLVE_TIME;
}

/* Applies one key of a selector mapping. Returns zero on an invalid key or
 * value; `seen` records which optional keys were present. */
static int
PrintSystemApplySelectorKey(DumpSelector_args *selector_out, const YAMLnode *child,
                            PrintSystemSeenKeys *seen)
{
   /* GCOVR_EXCL_BR_START */
   const char *value = child->mapped_val ? child->mapped_val : child->val;
   /* GCOVR_EXCL_BR_STOP */

   if (!strcmp(child->key, "basis"))
   {
      /* GCOVR_EXCL_BR_START */
      return PrintSystemParseBasis(value, &selector_out->basis); /* GCOVR_EXCL_BR_STOP */
   }
   if (!strcmp(child->key, "level"))
   {
      /* GCOVR_EXCL_BR_START */
      return (PrintSystemParseInteger(value, &selector_out->level) &&
              /* GCOVR_EXCL_BR_STOP */
              selector_out->level >= 0 && selector_out->level < STATS_MAX_LEVELS);
   }
   if (!strcmp(child->key, "every"))
   {
      /* GCOVR_EXCL_BR_START */
      if (!PrintSystemParseInteger(value, &selector_out->every) ||
          /* GCOVR_EXCL_BR_STOP */
          selector_out->every <= 0)
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      seen->every = 1;
      return 1;
   }
   if (!strcmp(child->key, "ids"))
   {
      /* GCOVR_EXCL_BR_START */
      if (!PrintSystemParseIntArrayNode(child, &selector_out->ids) ||
          /* GCOVR_EXCL_BR_STOP */
          !selector_out->ids || selector_out->ids->size == 0)
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      seen->ids = 1;
      return 1;
   }
   /* GCOVR_EXCL_BR_START */
   if (!strcmp(child->key, "ranges")) /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */
      if (!PrintSystemParseRangesNode(child,
                                      &selector_out->ranges) || /* GCOVR_EXCL_LINE */
          selector_out->ranges.size == 0)                       /* GCOVR_EXCL_LINE */
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      seen->ranges = 1; /* GCOVR_EXCL_LINE */
      return 1;         /* GCOVR_EXCL_LINE */
   }
   if (!strcmp(child->key, "threshold"))
   {
      /* GCOVR_EXCL_BR_START */
      if (!PrintSystemParseDouble(value, &selector_out->threshold) ||
          /* GCOVR_EXCL_BR_STOP */
          selector_out->threshold < 0.0)
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      seen->threshold = 1;
      return 1;
   }

   return 0;
}

/* A threshold basis is driven purely by its threshold; every other basis needs
 * at least one index-based key and must not carry a threshold. */
static int
PrintSystemSelectorKeysConsistent(const DumpSelector_args   *selector_out,
                                  const PrintSystemSeenKeys *seen)
{
   if (PrintSystemBasisUsesThreshold(selector_out->basis))
   {
      /* GCOVR_EXCL_BR_START */
      return seen->threshold && !seen->every && !seen->ids && !seen->ranges;
      /* GCOVR_EXCL_BR_STOP */
   }

   if (seen->threshold)
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   return (seen->every || seen->ids || seen->ranges); /* GCOVR_EXCL_BR_STOP */
}

static int
PrintSystemParseSelectorNode(const YAMLnode *node, DumpSelector_args *selector_out)
{
   PrintSystemSeenKeys seen = {0, 0, 0, 0, 0};

   /* GCOVR_EXCL_BR_START */
   if (!node || !selector_out || !node->children) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   selector_out->basis       = PRINT_SYSTEM_BASIS_LINEAR_SYSTEM;
   selector_out->level       = 0;
   selector_out->every       = 0;
   selector_out->threshold   = 0.0;
   selector_out->ids         = NULL;
   selector_out->ranges.data = NULL;
   selector_out->ranges.size = 0;

   for (const YAMLnode *child = node->children; child != NULL; child = child->next)
   {
      if (!PrintSystemApplySelectorKey(selector_out, child, &seen))
      {
         return 0;
      }
   }

   return PrintSystemSelectorKeysConsistent(selector_out, &seen);
}

typedef struct
{
   const char *name;
   int         value;
} PrintSystemNameMap;

static const PrintSystemNameMap kPrintSystemTypes[] = {
   {"all", PRINT_SYSTEM_TYPE_ALL},
   {"every_n_systems", PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS},
   {"every_n_timesteps", PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS},
   {"ids", PRINT_SYSTEM_TYPE_IDS},
   {"ranges", PRINT_SYSTEM_TYPE_RANGES},
   {"iterations_over", PRINT_SYSTEM_TYPE_ITERATIONS_OVER},
   {"setup_time_over", PRINT_SYSTEM_TYPE_SETUP_TIME_OVER},
   {"solve_time_over", PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER},
   {"selectors", PRINT_SYSTEM_TYPE_SELECTORS},
};

static const PrintSystemNameMap kPrintSystemStages[] = {
   {"build", PRINT_SYSTEM_STAGE_BUILD_BIT},
   {"setup", PRINT_SYSTEM_STAGE_SETUP_BIT},
   {"apply", PRINT_SYSTEM_STAGE_APPLY_BIT},
   {"all", PRINT_SYSTEM_STAGE_BUILD_BIT | PRINT_SYSTEM_STAGE_SETUP_BIT |
              PRINT_SYSTEM_STAGE_APPLY_BIT},
};

/* Case-insensitive table lookup; returns nonzero and stores the mapped value. */
static int
PrintSystemLookupName(const PrintSystemNameMap *table, size_t count, const char *name,
                      int *value_out)
{
   for (size_t i = 0; i < count; i++)
   {
      if (!strcasecmp(name, table[i].name))
      {
         *value_out = table[i].value;
         return 1;
      }
   }

   return 0;
}

/* Marks every "-" item of a sequence node as consumed. */
static void
PrintSystemMarkSequenceItemsValid(const YAMLnode *child)
{
   for (const YAMLnode *item = child->children; item != NULL; item = item->next)
   {
      /* GCOVR_EXCL_BR_START */
      if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
      {
         ((YAMLnode *)item)->valid = YAML_NODE_VALID;
      }
   }
}

/* Each key handler below returns nonzero on success, or zero with the error
 * state already populated. */

static int
PrintSystemApplyEnabled(PrintSystem_args *args, const YAMLnode *child, const char *value,
                        PrintSystemSeenKeys *seen)
{
   (void)child;
   (void)seen;
   if (!PrintSystemParseOnOff(value, &args->enabled))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      /* GCOVR_EXCL_BR_START */
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.enabled: '%s'",
                           /* GCOVR_EXCL_BR_STOP */
                           value ? value : "");
      return 0;
   }

   return 1;
}

static int
PrintSystemApplyType(PrintSystem_args *args, const YAMLnode *child, const char *value,
                     PrintSystemSeenKeys *seen)
{
   int type = 0;

   (void)child;
   (void)seen;
   if (!value)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing linear_system.print_system.type value");
      return 0;
   }
   if (!PrintSystemLookupName(kPrintSystemTypes,
                              sizeof(kPrintSystemTypes) / sizeof(kPrintSystemTypes[0]),
                              value, &type))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.type: '%s'", value);
      return 0;
   }
   args->type = type;

   return 1;
}

static int
PrintSystemApplyStage(PrintSystem_args *args, const YAMLnode *child, const char *value,
                      PrintSystemSeenKeys *seen)
{
   int mask = 0;

   (void)child;
   (void)seen;
   if (!value)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing linear_system.print_system.stage value");
      return 0;
   }
   if (!PrintSystemLookupName(kPrintSystemStages,
                              sizeof(kPrintSystemStages) / sizeof(kPrintSystemStages[0]),
                              value, &mask))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.stage: '%s'", value);
      return 0;
   }
   args->stage_mask = mask;

   return 1;
}

static int
PrintSystemApplyArtifacts(PrintSystem_args *args, const YAMLnode *child,
                          const char *value, PrintSystemSeenKeys *seen)
{
   (void)value;
   (void)seen;
   if (!PrintSystemParseArtifactsNode(child, &args->artifacts))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.artifacts");
      return 0;
   }
   PrintSystemMarkSequenceItemsValid(child);

   return 1;
}

static int
PrintSystemApplyOutputDir(PrintSystem_args *args, const YAMLnode *child,
                          const char *value, PrintSystemSeenKeys *seen)
{
   (void)child;
   (void)seen;
   if (!value)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing linear_system.print_system.output_dir value");
      return 0;
   }
   snprintf(args->output_dir, sizeof(args->output_dir), "%s", value);

   return 1;
}

static int
PrintSystemApplyOverwrite(PrintSystem_args *args, const YAMLnode *child,
                          const char *value, PrintSystemSeenKeys *seen)
{
   (void)child;
   (void)seen;
   if (!PrintSystemParseOnOff(value, &args->overwrite))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      /* GCOVR_EXCL_BR_START */
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.overwrite: '%s'",
                           /* GCOVR_EXCL_BR_STOP */
                           value ? value : "");
      return 0;
   }

   return 1;
}

static int
PrintSystemApplyEvery(PrintSystem_args *args, const YAMLnode *child, const char *value,
                      PrintSystemSeenKeys *seen)
{
   (void)child;
   if (!PrintSystemParseInteger(value, &args->every) || args->every <= 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      /* GCOVR_EXCL_BR_START */
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.every: '%s'",
                           /* GCOVR_EXCL_BR_STOP */
                           value ? value : "");
      return 0;
   }
   seen->every = 1;

   return 1;
}

static int
PrintSystemApplyIds(PrintSystem_args *args, const YAMLnode *child, const char *value,
                    PrintSystemSeenKeys *seen)
{
   (void)value;
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemParseIntArrayNode(child, &args->ids) || !args->ids ||
       /* GCOVR_EXCL_BR_STOP */
       /* GCOVR_EXCL_BR_START */
       args->ids->size == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.ids");
      return 0;
   }
   seen->ids = 1;
   PrintSystemMarkSequenceItemsValid(child);

   return 1;
}

static int
PrintSystemApplyRanges(PrintSystem_args *args, const YAMLnode *child, const char *value,
                       PrintSystemSeenKeys *seen)
{
   (void)value;
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemParseRangesNode(child, &args->ranges) || args->ranges.size == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.ranges");
      return 0;
   }
   seen->ranges = 1;
   PrintSystemMarkSequenceItemsValid(child);

   return 1;
}

static int
PrintSystemApplyThreshold(PrintSystem_args *args, const YAMLnode *child,
                          const char *value, PrintSystemSeenKeys *seen)
{
   (void)child;
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemParseDouble(value, &args->threshold) || args->threshold < 0.0)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      /* GCOVR_EXCL_BR_START */
      hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.threshold: '%s'",
                           /* GCOVR_EXCL_BR_STOP */
                           value ? value : "");
      return 0;
   }
   seen->threshold = 1;

   return 1;
}

/* Counts the "-" entries of a selector sequence. */
static size_t
PrintSystemCountSequenceItems(const YAMLnode *child)
{
   size_t count = 0;

   for (const YAMLnode *item = child->children; item != NULL; item = item->next)
   {
      /* GCOVR_EXCL_BR_START */
      if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
      {
         count++;
      }
   }

   return count;
}

/* Parses each selector into `selectors`; on failure every entry built so far is
 * destroyed and the array freed. */
static int
PrintSystemParseSelectorList(const YAMLnode *child, DumpSelector_args *selectors)
{
   size_t selector_idx = 0;

   for (const YAMLnode *item = child->children; item != NULL; item = item->next)
   {
      /* GCOVR_EXCL_BR_START */
      if (strcmp(item->key, "-") != 0) /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }

      if (!PrintSystemParseSelectorNode(item, &selectors[selector_idx]))
      {
         for (size_t cleanup_idx = 0; cleanup_idx <= selector_idx; cleanup_idx++)
         {
            PrintSystemSelectorDestroy(&selectors[cleanup_idx]);
         }
         free(selectors);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Invalid linear_system.print_system.selectors entry at index %d",
            (int)selector_idx);
         return 0;
      }
      ((YAMLnode *)item)->valid = YAML_NODE_VALID;
      selector_idx++;
   }

   return 1;
}

static int
PrintSystemApplySelectors(PrintSystem_args *args, const YAMLnode *child,
                          const char *value, PrintSystemSeenKeys *seen)
{
   DumpSelector_args *selectors      = NULL;
   size_t             selector_count = 0;

   (void)value;
   if (!child->children)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system.print_system.selectors must be a sequence");
      return 0;
   }

   selector_count = PrintSystemCountSequenceItems(child);
   /* GCOVR_EXCL_BR_START */
   if (selector_count == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system.print_system.selectors cannot be empty");
      return 0;
   }

   selectors = (DumpSelector_args *)calloc(selector_count, sizeof(DumpSelector_args));
   /* GCOVR_EXCL_BR_START */
   if (!selectors) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate selector list"); /* GCOVR_EXCL_LINE */
      return 0;                                                 /* GCOVR_EXCL_LINE */
   }

   if (!PrintSystemParseSelectorList(child, selectors))
   {
      return 0;
   }

   args->selectors     = selectors;
   args->num_selectors = selector_count;
   seen->selectors     = 1;

   return 1;
}

typedef struct
{
   const char *key;
   int (*apply)(PrintSystem_args *args, const YAMLnode *child, const char *value,
                PrintSystemSeenKeys *seen);
} PrintSystemKeyHandler;

static const PrintSystemKeyHandler kPrintSystemKeys[] = {
   {"enabled", PrintSystemApplyEnabled},      {"type", PrintSystemApplyType},
   {"stage", PrintSystemApplyStage},          {"artifacts", PrintSystemApplyArtifacts},
   {"output_dir", PrintSystemApplyOutputDir}, {"overwrite", PrintSystemApplyOverwrite},
   {"every", PrintSystemApplyEvery},          {"ids", PrintSystemApplyIds},
   {"ranges", PrintSystemApplyRanges},        {"threshold", PrintSystemApplyThreshold},
   {"selectors", PrintSystemApplySelectors},
};

/* Applies one child node of the print_system mapping. */
static int
PrintSystemApplyChild(PrintSystem_args *args, const YAMLnode *child,
                      PrintSystemSeenKeys *seen)
{
   /* GCOVR_EXCL_BR_START */
   const char *value = child->mapped_val ? child->mapped_val : child->val;
   /* GCOVR_EXCL_BR_STOP */

   for (size_t i = 0; i < sizeof(kPrintSystemKeys) / sizeof(kPrintSystemKeys[0]); i++)
   {
      if (strcmp(child->key, kPrintSystemKeys[i].key) != 0)
      {
         continue;
      }
      if (!kPrintSystemKeys[i].apply(args, child, value, seen))
      {
         return 0;
      }
      ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      return 1;
   }

   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorMsgAdd("Unknown key under linear_system.print_system: '%s'", child->key);

   return 0;
}

/* Threshold-based selection only makes sense once the relevant stage has run. */
static int
PrintSystemTypeIsThresholdBased(int type)
{
   return (type == PRINT_SYSTEM_TYPE_ITERATIONS_OVER ||
           type == PRINT_SYSTEM_TYPE_SETUP_TIME_OVER ||
           type == PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER);
}

/* The selected type dictates which companion keys are required and which are
 * forbidden. `every` has a documented default, so it is filled in rather than
 * demanded. */
static int
PrintSystemValidateTypeKeys(PrintSystem_args *args, const PrintSystemSeenKeys *seen)
{
   if (args->type == PRINT_SYSTEM_TYPE_ALL &&
       /* GCOVR_EXCL_BR_START */
       (seen->every || seen->ids || seen->ranges || seen->selectors))
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=all cannot be combined with selectors");
      return 0;
   }
   if ((args->type == PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS ||
        args->type == PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS) &&
       !seen->every)
   {
      args->every = 1;
   }
   if (args->type == PRINT_SYSTEM_TYPE_IDS && !seen->ids)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system.print_system.type=ids requires ids");
      return 0;
   }
   if (args->type == PRINT_SYSTEM_TYPE_RANGES && !seen->ranges)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system.print_system.type=ranges requires ranges");
      return 0;
   }
   if (PrintSystemTypeIsThresholdBased(args->type) && !seen->threshold)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system threshold type requires threshold");
      return 0;
   }
   if (args->type == PRINT_SYSTEM_TYPE_SELECTORS && !seen->selectors)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=selectors requires selectors");
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   if (args->type != PRINT_SYSTEM_TYPE_SELECTORS && seen->selectors)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.selectors requires type=selectors");
      return 0;
   }
   if (!PrintSystemTypeIsThresholdBased(args->type) && seen->threshold)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.threshold requires a threshold-based type");
      return 0;
   }
   return 1;
}

/* Threshold types can only fire once the stage that produces their metric has
 * actually run. */
static int
PrintSystemValidateStageMask(const PrintSystem_args *args)
{
   if (args->type == PRINT_SYSTEM_TYPE_ITERATIONS_OVER &&
       (args->stage_mask & PRINT_SYSTEM_STAGE_APPLY_BIT) == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=iterations_over requires stage apply");
      return 0;
   }
   if (args->type == PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER &&
       (args->stage_mask & PRINT_SYSTEM_STAGE_APPLY_BIT) == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=solve_time_over requires stage apply");
      return 0;
   }
   if (args->type == PRINT_SYSTEM_TYPE_SETUP_TIME_OVER &&
       (args->stage_mask &
        (PRINT_SYSTEM_STAGE_SETUP_BIT | PRINT_SYSTEM_STAGE_APPLY_BIT)) == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=setup_time_over requires stage setup or apply");
      return 0;
   }

   return 1;
}

/* Cross-field rules applied once every child key has been parsed. */
static int
PrintSystemValidateCombination(PrintSystem_args *args, const PrintSystemSeenKeys *seen)
{
   return (PrintSystemValidateTypeKeys(args, seen) && PrintSystemValidateStageMask(args));
}

/* A scalar print_system node is shorthand for the enabled flag alone. */
static void
PrintSystemSetArgsFromScalar(PrintSystem_args *args, const YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */
   const char *value = node->mapped_val ? node->mapped_val : node->val;
   /* GCOVR_EXCL_BR_STOP */

   /* GCOVR_EXCL_BR_START */
   if (value && value[0] != '\0') /* GCOVR_EXCL_BR_STOP */
   {
      if (!PrintSystemParseOnOff(value, &args->enabled))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid linear_system.print_system value: '%s'", value);
      }
   }
}

void
hypredrv_PrintSystemSetArgs(void *field, const YAMLnode *node)
{
   PrintSystem_args   *args = (PrintSystem_args *)field;
   PrintSystemSeenKeys seen = {0, 0, 0, 0, 0};

   if (!args || !node)
   {
      return;
   }

   hypredrv_PrintSystemDestroyArgs(args);
   hypredrv_PrintSystemSetDefaultArgs(args);

   if (!node->children)
   {
      PrintSystemSetArgsFromScalar(args, node);
      return;
   }

   for (const YAMLnode *child = node->children; child != NULL; child = child->next)
   {
      if (!PrintSystemApplyChild(args, child, &seen))
      {
         return;
      }
   }

   (void)PrintSystemValidateCombination(args, &seen);
}

void
hypredrv_LinearSystemPrintData(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat_A,
                               HYPRE_IJVector vec_b, const IntArray *dofmap)
{
   const char *A_base =
      (args && args->matrix_basename[0] != '\0') ? args->matrix_basename : "IJ.out.A";
   const char *b_base =
      (args && args->rhs_basename[0] != '\0') ? args->rhs_basename : "IJ.out.b";
   const char *d_base =
      (args && args->dofmap_basename[0] != '\0') ? args->dofmap_basename : "dofmap";

   char A_name[MAX_FILENAME_LENGTH];
   char b_name[MAX_FILENAME_LENGTH];
   char d_name[MAX_FILENAME_LENGTH];

   {
      size_t max_base = sizeof(A_name) - 1u - 4u;
      int    prec     = (int)max_base;
      (void)snprintf(A_name, sizeof(A_name), "%.*s.out", prec, A_base);
   }
   {
      size_t max_base = sizeof(b_name) - 1u - 4u;
      int    prec     = (int)max_base;
      (void)snprintf(b_name, sizeof(b_name), "%.*s.out", prec, b_base);
   }
   {
      size_t max_base = sizeof(d_name) - 1u - 4u;
      int    prec     = (int)max_base;
      (void)snprintf(d_name, sizeof(d_name), "%.*s.out", prec, d_base);
   }

   int use_series_dir = 1;
   if (args)
   {
      const int has_mat = args->matrix_basename[0] != '\0';
      const int has_rhs = args->rhs_basename[0] != '\0';
      const int has_dmf = args->dofmap_basename[0] != '\0';
      /* GCOVR_EXCL_BR_START */
      use_series_dir = !(has_mat || has_rhs || has_dmf);
      /* GCOVR_EXCL_BR_STOP */
   }

   char A_path[2 * MAX_FILENAME_LENGTH];
   char b_path[2 * MAX_FILENAME_LENGTH];
   char d_path[2 * MAX_FILENAME_LENGTH];

   if (use_series_dir)
   {
      const char *root = "hypre-data";
      struct stat st;
      if (stat(root, &st) != 0)
      {
         (void)mkdir(root, 0775);
      }

      int  max_idx = -1;
      DIR *dir     = opendir(root);
      /* GCOVR_EXCL_BR_START */
      if (dir) /* GCOVR_EXCL_BR_STOP */
      {
         const struct dirent *ent = NULL;
         while ((ent = readdir(dir)) != NULL)
         {
            /* GCOVR_EXCL_BR_START */
            if (ent->d_name[0] == 'l' && ent->d_name[1] == 's' && ent->d_name[2] == '_')
            /* GCOVR_EXCL_BR_STOP */
            {
               int idx = (int)strtol(ent->d_name + 3, NULL, 10);
               if (idx > max_idx)
               {
                  max_idx = idx;
               }
            }
         }
         closedir(dir);
      }
      int  next_idx = max_idx + 1;
      char run_dir[256];
      snprintf(run_dir, sizeof(run_dir), "%s/ls_%05d", root, next_idx);
      /* GCOVR_EXCL_BR_START */
      if (stat(run_dir, &st) != 0) /* GCOVR_EXCL_BR_STOP */
      {
         (void)mkdir(run_dir, 0775);
      }

      snprintf(A_path, sizeof(A_path), "%s/%s", run_dir, A_name);
      snprintf(b_path, sizeof(b_path), "%s/%s", run_dir, b_name);
      snprintf(d_path, sizeof(d_path), "%s/%s", run_dir, d_name);
   }
   else
   {
      snprintf(A_path, sizeof(A_path), "%s", A_name);
      snprintf(b_path, sizeof(b_path), "%s", b_name);
      snprintf(d_path, sizeof(d_path), "%s", d_name);
   }

   if (mat_A)
   {
      HYPRE_IJMatrixPrint(mat_A, A_path);
   }
   else
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Matrix not set; skipping matrix print.");
   }

   if (vec_b)
   {
      HYPRE_IJVectorPrint(vec_b, b_path);
   }
   else
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("RHS not set; skipping vector print.");
   }

   /* GCOVR_EXCL_BR_START */
   if (dofmap && dofmap->data) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_IntArrayWriteAsciiByRank(comm, dofmap, d_path);
   }
}

static int
PrintSystemContainsID(const IntArray *ids, int value)
{
   /* GCOVR_EXCL_BR_START */
   if (!ids || !ids->data) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   for (size_t i = 0; i < ids->size; i++)
   {
      if (ids->data[i] == value)
      {
         return 1;
      }
   }

   return 0;
}

static int
PrintSystemContainsRange(const IntRangeArray *ranges, int value)
{
   /* GCOVR_EXCL_BR_START */
   if (!ranges || !ranges->data) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   for (size_t i = 0; i < ranges->size; i++)
   {
      /* GCOVR_EXCL_BR_START */
      if (value >= ranges->data[i].begin && value <= ranges->data[i].end)
      /* GCOVR_EXCL_BR_STOP */
      {
         return 1;
      }
   }

   return 0;
}

static int
PrintSystemSelectorBasisValueGet(const DumpSelector_args  *selector,
                                 const PrintSystemContext *ctx)
{
   /* GCOVR_EXCL_BR_START */
   if (!selector || !ctx) /* GCOVR_EXCL_BR_STOP */
   {
      return -1; /* GCOVR_EXCL_LINE */
   }

   if (selector->basis == PRINT_SYSTEM_BASIS_TIMESTEP)
   {
      return ctx->timestep_index;
   }
   if (selector->basis == PRINT_SYSTEM_BASIS_LEVEL)
   {
      /* GCOVR_EXCL_BR_START */
      if (selector->level < 0 || selector->level >= STATS_MAX_LEVELS)
      /* GCOVR_EXCL_BR_STOP */
      {
         return -1;
      }
      return ctx->level_ids[selector->level];
   }

   return ctx->system_index;
}

static double
PrintSystemMetricValueGet(int basis, const PrintSystemContext *ctx)
{
   /* GCOVR_EXCL_BR_START */
   if (!ctx) /* GCOVR_EXCL_BR_STOP */
   {
      return -1.0; /* GCOVR_EXCL_LINE */
   }

   if (basis == PRINT_SYSTEM_BASIS_ITERATIONS)
   {
      /* GCOVR_EXCL_BR_START */
      return (ctx->last_iter >= 0) ? (double)ctx->last_iter : -1.0;
      /* GCOVR_EXCL_BR_STOP */
   }
   /* GCOVR_EXCL_BR_START */
   if (basis == PRINT_SYSTEM_BASIS_SETUP_TIME) /* GCOVR_EXCL_BR_STOP */
   {
      return ctx->last_setup_time;
   }
   if (basis == PRINT_SYSTEM_BASIS_SOLVE_TIME)
   {
      return ctx->last_solve_time;
   }

   return -1.0;
}

static int
PrintSystemSelectorMatches(const DumpSelector_args  *selector,
                           const PrintSystemContext *ctx)
{
   /* GCOVR_EXCL_BR_START */
   if (!selector || !ctx) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (PrintSystemBasisUsesThreshold(selector->basis))
   {
      double metric_value = PrintSystemMetricValueGet(selector->basis, ctx);
      /* GCOVR_EXCL_BR_START */
      return metric_value >= 0.0 && metric_value >= selector->threshold;
      /* GCOVR_EXCL_BR_STOP */
   }

   int basis_value = PrintSystemSelectorBasisValueGet(selector, ctx);
   if (basis_value < 0)
   {
      return 0;
   }

   int matched = 0;
   /* GCOVR_EXCL_BR_START */
   if (selector->every > 0 && (basis_value % selector->every) == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      matched = 1;
   }
   /* GCOVR_EXCL_BR_START */
   if (!matched && selector->ids && selector->ids->size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      matched = PrintSystemContainsID(selector->ids, basis_value); /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */
   if (!matched && selector->ranges.size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      matched =
         PrintSystemContainsRange(&selector->ranges, basis_value); /* GCOVR_EXCL_LINE */
   }

   return matched;
}

static const char *
PrintSystemTypeName(int type)
{
   switch (type)
   {
      case PRINT_SYSTEM_TYPE_ALL:
         return "all";
      case PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS:
         return "every_n_systems";
      case PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS:
         return "every_n_timesteps";
      case PRINT_SYSTEM_TYPE_IDS:
         return "ids";
      case PRINT_SYSTEM_TYPE_RANGES:
         return "ranges";
      case PRINT_SYSTEM_TYPE_ITERATIONS_OVER:
         return "iterations_over";
      case PRINT_SYSTEM_TYPE_SETUP_TIME_OVER:
         return "setup_time_over";
      case PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER:
         return "solve_time_over";
      case PRINT_SYSTEM_TYPE_SELECTORS:
         return "selectors";
      default:
         return "unknown";
   }
}

static const char *
PrintSystemBasisName(int basis)
{
   /* GCOVR_EXCL_BR_START */
   switch (basis)
   /* GCOVR_EXCL_BR_STOP */
   {
      case PRINT_SYSTEM_BASIS_LINEAR_SYSTEM:
         return "ids";
      case PRINT_SYSTEM_BASIS_TIMESTEP:
         return "timestep";
      case PRINT_SYSTEM_BASIS_LEVEL:
         return "level";
      case PRINT_SYSTEM_BASIS_ITERATIONS:
         return "iterations";
      case PRINT_SYSTEM_BASIS_SETUP_TIME:
         return "setup_time";
      case PRINT_SYSTEM_BASIS_SOLVE_TIME:
         return "solve_time";
      default:
         return "unknown";
   }
}

static int
PrintSystemStageEnabled(const PrintSystem_args *cfg, int stage)
{
   /* GCOVR_EXCL_BR_START */
   if (!cfg) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   int stage_bit = 0;
   if (stage == PRINT_SYSTEM_STAGE_BUILD)
   {
      stage_bit = PRINT_SYSTEM_STAGE_BUILD_BIT;
   }
   else if (stage == PRINT_SYSTEM_STAGE_SETUP)
   {
      stage_bit = PRINT_SYSTEM_STAGE_SETUP_BIT;
   }
   /* GCOVR_EXCL_BR_START */
   else if (stage == PRINT_SYSTEM_STAGE_APPLY) /* GCOVR_EXCL_BR_STOP */
   {
      stage_bit = PRINT_SYSTEM_STAGE_APPLY_BIT;
   }

   /* GCOVR_EXCL_BR_START */
   return (stage_bit != 0) && ((cfg->stage_mask & stage_bit) != 0);
   /* GCOVR_EXCL_BR_STOP */
}

/* Records why a dump was (or was not) scheduled. Safe to call with no sink, so
 * callers do not have to guard every diagnostic. */
static void
PrintSystemSetReason(char *reason, size_t reason_size, const char *fmt, ...)
{
   va_list ap;

   /* GCOVR_EXCL_BR_START */
   if (!reason || reason_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   va_start(ap, fmt);
   vsnprintf(reason, reason_size, fmt, ap);
   va_end(ap);
}

/* Preconditions common to every selection type: a usable config and context,
 * with the current stage enabled. */
static int
PrintSystemDumpGateOpen(const PrintSystem_args *cfg, const PrintSystemContext *ctx,
                        char *reason, size_t reason_size)
{
   /* GCOVR_EXCL_BR_START */
   if (!cfg) /* GCOVR_EXCL_BR_STOP */
   {
      PrintSystemSetReason(reason, reason_size, "%s",
                           "missing configuration"); /* GCOVR_EXCL_LINE */
      return 0;                                      /* GCOVR_EXCL_LINE */
   }
   if (!cfg->enabled)
   {
      PrintSystemSetReason(reason, reason_size, "%s", "print_system disabled");
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   if (!ctx) /* GCOVR_EXCL_BR_STOP */
   {
      PrintSystemSetReason(reason, reason_size, "%s",
                           "missing context"); /* GCOVR_EXCL_LINE */
      return 0;                                /* GCOVR_EXCL_LINE */
   }
   if (!PrintSystemStageEnabled(cfg, ctx->stage))
   {
      PrintSystemSetReason(reason, reason_size, "stage '%s' not selected",
                           PrintSystemStageName(ctx->stage));
      return 0;
   }

   return 1;
}

/* Describes the selector that matched, using the wording appropriate to its
 * basis (a metric threshold or a per-level value). */
static void
PrintSystemDescribeMatchedSelector(const DumpSelector_args *selector, size_t index,
                                   int basis_value, double metric_value, char *reason,
                                   size_t reason_size)
{
   if (PrintSystemBasisUsesThreshold(selector->basis))
   {
      PrintSystemSetReason(
         reason, reason_size, "selector[%zu] basis=%s metric_value=%.2e threshold=%.2e",
         index, PrintSystemBasisName(selector->basis), metric_value, selector->threshold);
   }
   else
   {
      PrintSystemSetReason(
         reason, reason_size, "selector[%zu] basis=%s level=%d basis_value=%d", index,
         PrintSystemBasisName(selector->basis), selector->level, basis_value);
   }
}

/* Selector lists match on the first entry that accepts the context. */
static int
PrintSystemAnySelectorMatches(const PrintSystem_args *cfg, const PrintSystemContext *ctx,
                              char *reason, size_t reason_size)
{
   /* GCOVR_EXCL_BR_START */
   if (!cfg->selectors || cfg->num_selectors == 0) /* GCOVR_EXCL_BR_STOP */
   {
      PrintSystemSetReason(reason, reason_size, "%s", "selectors list is empty");
      return 0;
   }

   for (size_t i = 0; i < cfg->num_selectors; i++)
   {
      const DumpSelector_args *selector = &cfg->selectors[i];
      int    basis_value                = PrintSystemSelectorBasisValueGet(selector, ctx);
      double metric_value               = PrintSystemMetricValueGet(selector->basis, ctx);

      if (PrintSystemSelectorMatches(selector, ctx))
      {
         PrintSystemDescribeMatchedSelector(selector, i, basis_value, metric_value,
                                            reason, reason_size);
         return 1;
      }
   }

   PrintSystemSetReason(reason, reason_size, "no selector matched (count=%zu)",
                        cfg->num_selectors);

   return 0;
}

static int
PrintSystemShouldDumpDetailed(const PrintSystem_args *cfg, const PrintSystemContext *ctx,
                              char *reason, size_t reason_size)
{
   /* GCOVR_EXCL_BR_START */
   if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      reason[0] = '\0';
   }

   if (!PrintSystemDumpGateOpen(cfg, ctx, reason, reason_size))
   {
      return 0;
   }

   switch (cfg->type)
   {
      case PRINT_SYSTEM_TYPE_ALL:
         PrintSystemSetReason(reason, reason_size, "%s", "type=all");
         return 1;

      case PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS:
      {
         /* GCOVR_EXCL_BR_START */
         int matched = (ctx->system_index >= 0) && (cfg->every > 0) &&
                       /* GCOVR_EXCL_BR_STOP */
                       ((ctx->system_index % cfg->every) == 0);

         PrintSystemSetReason(reason, reason_size, "system_index=%d every=%d",
                              ctx->system_index, cfg->every);
         return matched;
      }

      case PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS:
      {
         /* GCOVR_EXCL_BR_START */
         int matched = (ctx->timestep_index >= 0) && (cfg->every > 0) &&
                       /* GCOVR_EXCL_BR_STOP */
                       ((ctx->timestep_index % cfg->every) == 0);

         PrintSystemSetReason(reason, reason_size, "timestep_index=%d every=%d",
                              ctx->timestep_index, cfg->every);
         return matched;
      }

      case PRINT_SYSTEM_TYPE_IDS:
      {
         int matched = PrintSystemContainsID(cfg->ids, ctx->system_index);

         PrintSystemSetReason(reason, reason_size, "system_index=%d ids_size=%zu",
                              ctx->system_index,
                              /* GCOVR_EXCL_BR_START */
                              cfg->ids ? cfg->ids->size : 0);
         /* GCOVR_EXCL_BR_STOP */
         return matched;
      }

      case PRINT_SYSTEM_TYPE_RANGES:
      {
         int matched = PrintSystemContainsRange(&cfg->ranges, ctx->system_index);

         PrintSystemSetReason(reason, reason_size, "system_index=%d ranges_size=%zu",
                              ctx->system_index, cfg->ranges.size);
         return matched;
      }

      case PRINT_SYSTEM_TYPE_ITERATIONS_OVER:
      {
         /* GCOVR_EXCL_BR_START */
         int matched =
            (ctx->last_iter >= 0) && ((double)ctx->last_iter >= cfg->threshold);
         /* GCOVR_EXCL_BR_STOP */

         PrintSystemSetReason(reason, reason_size, "last_iter=%d threshold=%.3e",
                              ctx->last_iter, cfg->threshold);
         return matched;
      }

      case PRINT_SYSTEM_TYPE_SETUP_TIME_OVER:
      {
         /* GCOVR_EXCL_BR_START */
         int matched =
            (ctx->last_setup_time >= 0.0) && (ctx->last_setup_time >= cfg->threshold);
         /* GCOVR_EXCL_BR_STOP */

         PrintSystemSetReason(reason, reason_size, "last_setup_time=%.3e threshold=%.3e",
                              ctx->last_setup_time, cfg->threshold);
         return matched;
      }

      case PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER:
      {
         /* GCOVR_EXCL_BR_START */
         int matched =
            (ctx->last_solve_time >= 0.0) && (ctx->last_solve_time >= cfg->threshold);
         /* GCOVR_EXCL_BR_STOP */

         PrintSystemSetReason(reason, reason_size, "last_solve_time=%.3e threshold=%.3e",
                              ctx->last_solve_time, cfg->threshold);
         return matched;
      }

      case PRINT_SYSTEM_TYPE_SELECTORS:
         return PrintSystemAnySelectorMatches(cfg, ctx, reason, reason_size);

      default:
         break;
   }

   PrintSystemSetReason(reason, reason_size, "unknown type=%d", cfg->type);

   return 0;
}

static const char *
PrintSystemStageName(int stage)
{
   if (stage == PRINT_SYSTEM_STAGE_SETUP)
   {
      return "setup";
   }
   if (stage == PRINT_SYSTEM_STAGE_APPLY)
   {
      return "apply";
   }
   return "build";
}

static void
PrintSystemSanitizeToken(const char *src, char *dst, size_t dst_size)
{
   /* GCOVR_EXCL_BR_START */
   if (!dst || dst_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   size_t di = 0;
   if (src)
   {
      /* GCOVR_EXCL_BR_START */
      for (size_t si = 0; src[si] != '\0' && di + 1 < dst_size; si++)
      /* GCOVR_EXCL_BR_STOP */
      {
         unsigned char ch = (unsigned char)src[si];
         /* GCOVR_EXCL_BR_START */
         if (isalnum(ch) || ch == '_' || ch == '-') /* GCOVR_EXCL_BR_STOP */
         {
            dst[di++] = (char)ch;
         }
         else
         {
            dst[di++] = '_';
         }
      }
   }

   if (di == 0)
   {
      snprintf(dst, dst_size, "%s", "unnamed");
      return;
   }
   dst[di] = '\0';
}

static int
PrintSystemEnsureDir(const char *path)
{
   /* GCOVR_EXCL_BR_START */
   if (!path || path[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char current[2 * MAX_FILENAME_LENGTH];
   snprintf(current, sizeof(current), "%s", path);

   size_t len = strlen(current);
   /* GCOVR_EXCL_BR_START */
   if (len == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   for (size_t i = 1; i <= len; i++)
   {
      if (current[i] == '/' || current[i] == '\0')
      {
         char saved = current[i];
         current[i] = '\0';

         /* GCOVR_EXCL_BR_START */
         if (current[0] != '\0') /* GCOVR_EXCL_BR_STOP */
         {
            struct stat st;
            if (stat(current, &st) != 0)
            {
               /* GCOVR_EXCL_BR_START */
               if (mkdir(current, 0775) != 0 && errno != EEXIST) /* GCOVR_EXCL_BR_STOP */
               {
                  return 0;
               }
            }
            else if (!S_ISDIR(st.st_mode))
            {
               return 0;
            }
         }

         current[i] = saved;
      }
   }

   return 1;
}

static int
PrintSystemPathExists(const char *path)
{
   struct stat st;
   /* GCOVR_EXCL_BR_START */
   return (path && stat(path, &st) == 0);
   /* GCOVR_EXCL_BR_STOP */
}

static int
PrintSystemPathCopy(char *dst, size_t dst_size, const char *src)
{
   /* GCOVR_EXCL_BR_START */
   if (!dst || dst_size == 0 || !src) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   size_t src_len = strlen(src);
   /* GCOVR_EXCL_BR_START */
   if (src_len + 1 > dst_size) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   memcpy(dst, src, src_len + 1);
   return 1;
}

static int
PrintSystemPathJoin(char *dst, size_t dst_size, const char *base_path,
                    const char *path_component)
{
   /* GCOVR_EXCL_BR_START */
   if (!dst || dst_size == 0 || !base_path || !path_component) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   size_t base_len      = strlen(base_path);
   size_t component_len = strlen(path_component);
   /* GCOVR_EXCL_BR_START */
   int add_sep = (base_len > 0 && base_path[base_len - 1] != '/');
   /* GCOVR_EXCL_BR_STOP */
   size_t total_len = base_len + (size_t)add_sep + component_len + 1;

   /* GCOVR_EXCL_BR_START */
   if (total_len > dst_size) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   memcpy(dst, base_path, base_len);
   size_t pos = base_len;
   /* GCOVR_EXCL_BR_START */
   if (add_sep) /* GCOVR_EXCL_BR_STOP */
   {
      dst[pos++] = '/';
   }
   memcpy(dst + pos, path_component, component_len);
   dst[pos + component_len] = '\0';
   return 1;
}

static int
PrintSystemArtifactPathBuild(const char *dump_dir, const char *artifact_name,
                             char *artifact_path, size_t artifact_path_size)
{
   /* GCOVR_EXCL_BR_START */
   if (!dump_dir || !artifact_name || !artifact_path || artifact_path_size == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   return PrintSystemPathJoin(artifact_path, artifact_path_size, dump_dir, artifact_name);
}

static int
PrintSystemFindMaxDumpIndex(const char *base_dir)
{
   /* GCOVR_EXCL_BR_START */
   if (!base_dir) /* GCOVR_EXCL_BR_STOP */
   {
      return -1; /* GCOVR_EXCL_LINE */
   }

   DIR *dir = opendir(base_dir);
   /* GCOVR_EXCL_BR_START */
   if (!dir) /* GCOVR_EXCL_BR_STOP */
   {
      return -1; /* GCOVR_EXCL_LINE */
   }

   int                  max_idx = -1;
   const struct dirent *entry   = NULL;
   while ((entry = readdir(dir)) != NULL)
   {
      if (strncmp(entry->d_name, "ls_", 3) != 0)
      {
         continue;
      }

      const char *digits = entry->d_name + 3;
      /* GCOVR_EXCL_BR_START */
      if (*digits == '\0') /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }

      bool all_digits = true;
      for (const char *p = digits; *p != '\0'; p++)
      {
         /* GCOVR_EXCL_BR_START */
         if (!isdigit((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
         {
            all_digits = false; /* GCOVR_EXCL_LINE */
            break;              /* GCOVR_EXCL_LINE */
         }
      }
      /* GCOVR_EXCL_BR_START */
      if (!all_digits) /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }

      long idx_long = strtol(digits, NULL, 10);
      /* GCOVR_EXCL_BR_START */
      if (idx_long < 0 || idx_long > INT_MAX) /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }
      int idx = (int)idx_long;
      /* GCOVR_EXCL_BR_START */
      if (idx > max_idx) /* GCOVR_EXCL_BR_STOP */
      {
         max_idx = idx;
      }
   }

   closedir(dir);
   return max_idx;
}

static int
PrintSystemRemoveTree(const char *path)
{
   /* GCOVR_EXCL_BR_START */
   if (!path || path[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* Open directory without following symlinks (avoids lstat+unlink TOCTOU). */
   int dfd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
   /* GCOVR_EXCL_BR_START */
   if (dfd < 0) /* GCOVR_EXCL_BR_STOP */
   {
      if (errno == ENOENT)
      {
         return 1; /* GCOVR_EXCL_LINE */
      }
      /* Not a directory, or symlink: remove by path (unlink removes symlinks). */
      return (unlink(path) == 0) || (errno == ENOENT);
   }

   DIR *dir = fdopendir(dfd);
   /* GCOVR_EXCL_BR_START */
   if (!dir) /* GCOVR_EXCL_BR_STOP */
   {
      close(dfd);
      return 0; /* GCOVR_EXCL_LINE */
   }

   int                  ok    = 1;
   const struct dirent *entry = NULL;
   /* GCOVR_EXCL_BR_START */
   while (ok && (entry = readdir(dir)) != NULL) /* GCOVR_EXCL_BR_STOP */
   {
      if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
      {
         continue;
      }

      char child[2 * MAX_FILENAME_LENGTH];
      /* GCOVR_EXCL_BR_START */
      if (!PrintSystemPathJoin(child, sizeof(child), path, entry->d_name))
      /* GCOVR_EXCL_BR_STOP */
      {
         ok = 0; /* GCOVR_EXCL_LINE */
         break;  /* GCOVR_EXCL_LINE */
      }

      ok = PrintSystemRemoveTree(child);
   }

   closedir(dir);
   /* GCOVR_EXCL_BR_START */
   if (!ok) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */
   return (rmdir(path) == 0) || (errno == ENOENT);
   /* GCOVR_EXCL_BR_STOP */
}

static int
PrintSystemChooseDumpDirLocal(const PrintSystem_args *cfg, const PrintSystemContext *ctx,
                              const char *object_name, char *dump_dir,
                              size_t dump_dir_size)
{
   /* GCOVR_EXCL_BR_START */
   if (!cfg || !ctx || !dump_dir || dump_dir_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   (void)ctx;

   char object_token[MAX_FILENAME_LENGTH];
   PrintSystemSanitizeToken(object_name, object_token, sizeof(object_token));

   /* GCOVR_EXCL_BR_START */
   const char *root = (cfg->output_dir[0] != '\0') ? cfg->output_dir : "hypre-dumps";
   /* GCOVR_EXCL_BR_STOP */
   char base_dir[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemPathJoin(base_dir, sizeof(base_dir), root, object_token))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   PrintSystem_args *cfg_state = (PrintSystem_args *)cfg;
   if (cfg->overwrite && !cfg_state->overwrite_prepared)
   {
      /* GCOVR_EXCL_BR_START */
      if (PrintSystemPathExists(base_dir) && !PrintSystemRemoveTree(base_dir))
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      cfg_state->next_dump_index    = 0;
      cfg_state->overwrite_prepared = 1;
   }

   if (!PrintSystemEnsureDir(base_dir))
   {
      return 0;
   }

   char candidate[2 * MAX_FILENAME_LENGTH];
   char leaf[32];
   if (cfg->overwrite)
   {
      /* GCOVR_EXCL_BR_START */
      if (cfg_state->next_dump_index < 0) /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }

      snprintf(leaf, sizeof(leaf), "ls_%05d", cfg_state->next_dump_index);
      /* GCOVR_EXCL_BR_START */
      if (!PrintSystemPathJoin(candidate, sizeof(candidate), base_dir, leaf))
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      cfg_state->next_dump_index++;

      /* GCOVR_EXCL_BR_START */
      if (PrintSystemPathExists(candidate) && !PrintSystemRemoveTree(candidate))
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
   }
   else
   {
      int next_idx = PrintSystemFindMaxDumpIndex(base_dir) + 1;
      /* GCOVR_EXCL_BR_START */
      if (next_idx < 0) /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }

      do
      {
         snprintf(leaf, sizeof(leaf), "ls_%05d", next_idx);
         /* GCOVR_EXCL_BR_START */
         if (!PrintSystemPathJoin(candidate, sizeof(candidate), base_dir, leaf))
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         next_idx++;
         /* GCOVR_EXCL_BR_START */
      } while (PrintSystemPathExists(candidate));
      /* GCOVR_EXCL_BR_STOP */
   }

   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemPathCopy(dump_dir, dump_dir_size, candidate)) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   return PrintSystemEnsureDir(dump_dir);
}

static int
PrintSystemChooseDumpDir(MPI_Comm comm, const PrintSystem_args *cfg,
                         const PrintSystemContext *ctx, const char *object_name,
                         char *dump_dir, size_t dump_dir_size)
{
   int mypid = 0;
   int ok    = 0;

   /* GCOVR_EXCL_BR_START */
   if (!cfg || !ctx || !dump_dir || dump_dir_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   MPI_Comm_rank(comm, &mypid);

   if (mypid == 0)
   {
      ok = PrintSystemChooseDumpDirLocal(cfg, ctx, object_name, dump_dir, dump_dir_size);
   }

   MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
   if (!ok)
   {
      dump_dir[0] = '\0';
      return 0;
   }

   MPI_Bcast(dump_dir, (int)dump_dir_size, MPI_CHAR, 0, comm);

   return PrintSystemEnsureDir(dump_dir);
}

static void
PrintSystemWriteMetadata(const char *dump_dir, const PrintSystemContext *ctx,
                         const char *object_name, int artifacts)
{
   /* GCOVR_EXCL_BR_START */
   if (!dump_dir || !ctx) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   char metadata_path[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemPathJoin(metadata_path, sizeof(metadata_path), dump_dir,
                            /* GCOVR_EXCL_BR_STOP */
                            "metadata.txt"))
   {
      return; /* GCOVR_EXCL_LINE */
   }
   int fd = open(metadata_path, O_WRONLY | O_CREAT | O_TRUNC, (mode_t)0600);
   /* GCOVR_EXCL_BR_START */
   if (fd < 0) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }
   FILE *fp = fdopen(fd, "w");
   /* GCOVR_EXCL_BR_START */
   if (!fp) /* GCOVR_EXCL_BR_STOP */
   {
      close(fd);
      return; /* GCOVR_EXCL_LINE */
   }

   fprintf(fp, "object_name=%s\n", object_name ? object_name : "unnamed");
   fprintf(fp, "stage=%s\n", PrintSystemStageName(ctx->stage));
   fprintf(fp, "system_index=%d\n", ctx->system_index);
   fprintf(fp, "stats_ls_id=%d\n", ctx->stats_ls_id);
   fprintf(fp, "timestep_index=%d\n", ctx->timestep_index);
   fprintf(fp, "last_iter=%d\n", ctx->last_iter);
   fprintf(fp, "last_setup_time=%.17g\n", ctx->last_setup_time);
   fprintf(fp, "last_solve_time=%.17g\n", ctx->last_solve_time);
   fprintf(fp, "variant_index=%d\n", ctx->variant_index);
   fprintf(fp, "repetition_index=%d\n", ctx->repetition_index);
   fprintf(fp, "artifacts_mask=%d\n", artifacts);
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      fprintf(fp, "level_%d_id=%d\n", level, ctx->level_ids[level]);
   }

   fclose(fp);
}

static void
PrintSystemAppendStageIndex(const char *dump_dir, const PrintSystemContext *ctx,
                            const char *object_name)
{
   /* GCOVR_EXCL_BR_START */
   if (!dump_dir || !ctx) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   char path_buf[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemPathCopy(path_buf, sizeof(path_buf), dump_dir))
   /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   char *ls_name = strrchr(path_buf, '/');
   /* GCOVR_EXCL_BR_START */
   if (!ls_name || ls_name == path_buf) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }
   *ls_name = '\0';
   ls_name++;

   char index_path[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemPathJoin(index_path, sizeof(index_path), path_buf,
                            /* GCOVR_EXCL_BR_STOP */
                            "systems_index.txt"))
   {
      return; /* GCOVR_EXCL_LINE */
   }

   int fd = open(index_path, O_WRONLY | O_APPEND | O_CREAT, (mode_t)0600);
   /* GCOVR_EXCL_BR_START */
   if (fd < 0) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }
   FILE *fp = fdopen(fd, "a");
   /* GCOVR_EXCL_BR_START */
   if (!fp) /* GCOVR_EXCL_BR_STOP */
   {
      close(fd);
      return; /* GCOVR_EXCL_LINE */
   }

   fprintf(fp,
           "%s (object=%s stage=%s system=%d stats_ls=%d timestep=%d last_iter=%d "
           "last_setup=%.2e last_solve=%.2e variant=%d repetition=%d)\n",
           ls_name, object_name ? object_name : "unnamed",
           PrintSystemStageName(ctx->stage), ctx->system_index, ctx->stats_ls_id,
           ctx->timestep_index, ctx->last_iter, ctx->last_setup_time,
           ctx->last_solve_time, ctx->variant_index, ctx->repetition_index);
   fclose(fp);
}

/* One schedulable artifact. Exactly one of the object handles is set; a null
 * handle means the artifact was requested but the object does not exist. */
typedef struct
{
   unsigned        bit;
   const char     *filename;
   const char     *label;
   const char     *null_note;
   HYPRE_IJMatrix  matrix;
   HYPRE_IJVector  vector;
   const IntArray *dofmap;
} PrintSystemArtifact;

/* Writes one artifact when it is both selected and available, logging why it
 * was skipped otherwise. Returns zero only when the output path could not be
 * built, which is a hard failure for the whole dump. */
static int
PrintSystemWriteArtifact(MPI_Comm comm, const PrintSystemArtifact *artifact,
                         unsigned artifacts, const char *dump_dir,
                         const char *object_name, int ls_id_for_log)
{
   char path[2 * MAX_FILENAME_LENGTH];

   if ((artifacts & artifact->bit) == 0)
   {
      return 1;
   }

   if (!artifact->matrix && !artifact->vector && !artifact->dofmap)
   {
      /* GCOVR_EXCL_BR_START */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip %s: %s", artifact->label,
                         artifact->null_note);
      return 1;
   }

   /* GCOVR_EXCL_BR_START */
   if (!PrintSystemArtifactPathBuild(dump_dir, artifact->filename, path, sizeof(path)))
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("print_system path too long for %s artifact",
                           artifact->label); /* GCOVR_EXCL_LINE */
      return 0;                              /* GCOVR_EXCL_LINE */
   }

   HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log, "print_system write %s: %s",
                      artifact->label, path);

   if (artifact->matrix)
   {
      HYPRE_IJMatrixPrint(artifact->matrix, path);
   }
   else if (artifact->vector)
   {
      HYPRE_IJVectorPrint(artifact->vector, path);
   }
   else
   {
      hypredrv_IntArrayWriteAsciiByRank(comm, artifact->dofmap, path);
   }

   return 1;
}

/* Rank 0 owns the per-dump metadata file and the cumulative stage index. */
static void
PrintSystemWriteDumpIndex(MPI_Comm comm, const PrintSystem_args *cfg,
                          const PrintSystemContext *ctx, const char *dump_dir,
                          const char *object_name, int ls_id_for_log)
{
   int mypid = 0;

   MPI_Comm_rank(comm, &mypid);
   if (mypid != 0)
   {
      return;
   }

   /* GCOVR_EXCL_BR_START */
   if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_METADATA) /* GCOVR_EXCL_BR_STOP */
   {
      PrintSystemWriteMetadata(dump_dir, ctx, object_name, cfg->artifacts);
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system write metadata: %s/metadata.txt", dump_dir);
   }

   PrintSystemAppendStageIndex(dump_dir, ctx, object_name);
   HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                      "print_system append systems index: %s", dump_dir);
}

uint32_t
hypredrv_LinearSystemDumpScheduled(MPI_Comm comm, const LS_args *args,
                                   HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                                   HYPRE_IJVector vec_b, HYPRE_IJVector vec_x0,
                                   HYPRE_IJVector vec_xref, HYPRE_IJVector vec_x,
                                   const IntArray *dofmap, const PrintSystemContext *ctx,
                                   const char *object_name)
{
   const PrintSystem_args *cfg = NULL;
   char                    decision_reason[160];
   char                    dump_dir[2 * MAX_FILENAME_LENGTH];
   int                     ls_id_for_log = 0;

   if (!args || !ctx)
   {
      return hypredrv_ErrorCodeGet();
   }

   cfg           = &args->print_system;
   ls_id_for_log = (ctx->stats_ls_id >= 0) ? ctx->stats_ls_id : ctx->system_index;

   if (!PrintSystemShouldDumpDetailed(cfg, ctx, decision_reason, sizeof(decision_reason)))
   {
      HYPREDRV_LOG_COMMF(
         3, comm, object_name, ls_id_for_log,
         "print_system evaluate: stage=%s type=%s artifacts=0x%x "
         "system_index=%d timestep_index=%d variant=%d repetition=%d (%s)",
         PrintSystemStageName(ctx->stage), PrintSystemTypeName(cfg->type), cfg->artifacts,
         ctx->system_index, ctx->timestep_index, ctx->variant_index,
         ctx->repetition_index, decision_reason);
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system skip: selection did not match");
      return hypredrv_ErrorCodeGet();
   }

   HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                      "print_system evaluate: stage=%s type=%s artifacts=0x%x "
                      "system_index=%d timestep_index=%d variant=%d repetition=%d (%s)",
                      PrintSystemStageName(ctx->stage), PrintSystemTypeName(cfg->type),
                      cfg->artifacts, ctx->system_index, ctx->timestep_index,
                      ctx->variant_index, ctx->repetition_index, decision_reason);

   if (!PrintSystemChooseDumpDir(comm, cfg, ctx, object_name, dump_dir, sizeof(dump_dir)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd(
         "Failed to create dump directory for linear_system.print_system");
      /* GCOVR_EXCL_BR_START */
      HYPREDRV_LOG_COMMF(2, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system failed: cannot create dump directory");
      return hypredrv_ErrorCodeGet();
   }
   HYPREDRV_LOG_COMMF(2, comm, object_name, ls_id_for_log, "print_system dump dir: %s",
                      dump_dir);

   {
      const PrintSystemArtifact artifacts[] = {
         {PRINT_SYSTEM_ARTIFACT_MATRIX, "matrix.out", "matrix", "matrix object is NULL",
          mat_A, NULL, NULL},
         {PRINT_SYSTEM_ARTIFACT_PRECMAT, "precmat.out", "precmat",
          "preconditioner matrix is NULL", mat_M, NULL, NULL},
         {PRINT_SYSTEM_ARTIFACT_RHS, "rhs.out", "rhs", "rhs vector is NULL", NULL, vec_b,
          NULL},
         {PRINT_SYSTEM_ARTIFACT_X0, "x0.out", "x0", "initial guess vector is NULL", NULL,
          vec_x0, NULL},
         {PRINT_SYSTEM_ARTIFACT_XREF, "xref.out", "xref", "reference solution is NULL",
          NULL, vec_xref, NULL},
         {PRINT_SYSTEM_ARTIFACT_SOLUTION, "solution.out", "solution",
          "solution vector is NULL", NULL, vec_x, NULL},
         {PRINT_SYSTEM_ARTIFACT_DOFMAP, "dofmap.out", "dofmap", "dofmap is NULL", NULL,
          NULL, (dofmap && dofmap->data) ? dofmap : NULL},
      };

      for (size_t i = 0; i < sizeof(artifacts) / sizeof(artifacts[0]); i++)
      {
         if (!PrintSystemWriteArtifact(comm, &artifacts[i], cfg->artifacts, dump_dir,
                                       object_name, ls_id_for_log))
         {
            return hypredrv_ErrorCodeGet();
         }
      }
   }

   PrintSystemWriteDumpIndex(comm, cfg, ctx, dump_dir, object_name, ls_id_for_log);

   HYPREDRV_LOG_COMMF(2, comm, object_name, ls_id_for_log, "print_system dump complete");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Report whether the print-system config requires a timestep schedule
 *-----------------------------------------------------------------------------*/

int
hypredrv_PrintSystemNeedsTimestepSchedule(const PrintSystem_args *cfg)
{
   if (!cfg || !cfg->enabled)
   {
      return 0;
   }

   if (cfg->type == PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS)
   {
      return 1;
   }

   if (cfg->type != PRINT_SYSTEM_TYPE_SELECTORS ||
       !cfg->selectors) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }

   for (size_t i = 0; i < cfg->num_selectors; i++)
   {
      if (cfg->selectors[i].basis == PRINT_SYSTEM_BASIS_TIMESTEP)
      {
         return 1;
      }
   }

   return 0;
}
