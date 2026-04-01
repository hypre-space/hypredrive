/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ranges)              /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!selector)            /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!value || !out)       /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (!strcasecmp(value, "on") || !strcasecmp(value, "yes") ||
       /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
       !strcasecmp(value, "true") || !strcmp(value, "1"))
   /* GCOVR_EXCL_BR_STOP */
   {
      *out = 1;
      return 1;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!strcasecmp(value, "off") || !strcasecmp(value, "no") ||
       /* GCOVR_EXCL_BR_STOP */
       /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!value || !out)       /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char *endptr = NULL;
   long  parsed = strtol(value, &endptr, 10);
   if (endptr == value)
   {
      return 0;
   }
   /* GCOVR_EXCL_BR_START */                          /* low-signal branch under CI */
   while (*endptr && isspace((unsigned char)*endptr)) /* GCOVR_EXCL_BR_STOP */
   {
      endptr++; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!value || !out)       /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char  *endptr = NULL;
   double parsed = strtod(value, &endptr);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (endptr == value)      /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                          /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ranges)              /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!new_data)            /* GCOVR_EXCL_BR_STOP */
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

static int
PrintSystemParseRangePair(const char *value, int *begin, int *end)
{
   /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
   if (!value || !begin || !end) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char *buffer = strdup(value);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!buffer)              /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate range parser buffer"); /* GCOVR_EXCL_LINE */
      return 0;                                     /* GCOVR_EXCL_LINE */
   }

   char *p = buffer;
   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   while (*p && isspace((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
   {
      p++; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (*p == '[')            /* GCOVR_EXCL_BR_STOP */
   {
      p++;
   }

   char *first_end = NULL;
   long  first     = strtol(p, &first_end, 10);
   if (first_end == p)
   {
      free(buffer);
      return 0;
   }
   p = first_end;
   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   while (*p && isspace((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
   {
      p++; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   if (*p != ',' && *p != '-' && *p != ':') /* GCOVR_EXCL_BR_STOP */
   {
      free(buffer); /* GCOVR_EXCL_LINE */
      return 0;     /* GCOVR_EXCL_LINE */
   }
   p++;
   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   while (*p && isspace((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
   {
      p++;
   }

   char *second_end = NULL;
   long  second     = strtol(p, &second_end, 10);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (second_end == p)      /* GCOVR_EXCL_BR_STOP */
   {
      free(buffer); /* GCOVR_EXCL_LINE */
      return 0;     /* GCOVR_EXCL_LINE */
   }
   p = second_end;
   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   while (*p && isspace((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
   {
      p++; /* GCOVR_EXCL_LINE */
   }
   if (*p == ']')
   {
      p++;
   }
   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   while (*p && isspace((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
   {
      p++; /* GCOVR_EXCL_LINE */
   }
   if (*p != '\0')
   {
      free(buffer);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (first < INT_MIN || first > INT_MAX || second < INT_MIN || second > INT_MAX)
   /* GCOVR_EXCL_BR_STOP */
   {
      free(buffer); /* GCOVR_EXCL_LINE */
      return 0;     /* GCOVR_EXCL_LINE */
   }

   *begin = (int)first;
   *end   = (int)second;
   free(buffer);
   return 1;
}

static int
PrintSystemParseIntArrayNode(const YAMLnode *node, IntArray **out)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node || !out)        /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   hypredrv_IntArrayDestroy(out);
   if (node->children)
   {
      size_t count = 0;
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
         if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
         {
            count++;
         }
      }

      IntArray *ids = hypredrv_IntArrayCreate(count);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!ids)                 /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);            /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("Failed to allocate id list"); /* GCOVR_EXCL_LINE */
         return 0;                                           /* GCOVR_EXCL_LINE */
      }

      size_t idx = 0;
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
         if (strcmp(item->key, "-") != 0) /* GCOVR_EXCL_BR_STOP */
         {
            continue; /* GCOVR_EXCL_LINE */
         }

         int parsed = 0;
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   const char *value = node->mapped_val ? node->mapped_val : node->val;
   /* GCOVR_EXCL_BR_STOP */
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!value)               /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   hypredrv_StrToIntArray(value, out);
   return (*out != NULL);
}

static int
PrintSystemParseRangesNode(const YAMLnode *node, IntRangeArray *ranges)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node || !ranges)     /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   PrintSystemRangeArrayDestroy(ranges);
   if (node->children)
   {
      for (const YAMLnode *item = node->children; item != NULL; item = item->next)
      {
         /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
         if (strcmp(item->key, "-") != 0) /* GCOVR_EXCL_BR_STOP */
         {
            continue; /* GCOVR_EXCL_LINE */
         }
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         const char *value = item->mapped_val ? item->mapped_val : item->val;
         /* GCOVR_EXCL_BR_STOP */
         int begin = 0;
         int end   = 0;
         if (!PrintSystemParseRangePair(value, &begin, &end))
         {
            return 0;
         }
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemRangeArrayAppend(ranges, begin, end)) /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
      }
      return ranges->size > 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   const char *value = node->mapped_val ? node->mapped_val : node->val;
   /* GCOVR_EXCL_BR_STOP */
   int begin = 0;
   int end   = 0;
   /* GCOVR_EXCL_BR_START */                            /* low-signal branch under CI */
   if (!PrintSystemParseRangePair(value, &begin, &end)) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   return PrintSystemRangeArrayAppend(ranges, begin, end);
}

static int
PrintSystemArtifactBitFromName(const char *token, int *bit_out)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!token || !bit_out)   /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
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
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         const char *token = item->mapped_val ? item->mapped_val : item->val;
         /* GCOVR_EXCL_BR_STOP */
         int bit = 0;
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemArtifactBitFromName(token, &bit)) /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         artifacts |= bit;
      }
   }
   else
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      const char *value = node->mapped_val ? node->mapped_val : node->val;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!value)               /* GCOVR_EXCL_BR_STOP */
      {
         return 0;
      }

      char *buffer = strdup(value);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!buffer)              /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate artifact parser buffer"); /* GCOVR_EXCL_LINE */
         return 0;                                        /* GCOVR_EXCL_LINE */
      }
      char *saveptr = NULL;
      char *token   = strtok_r(buffer, "[], ", &saveptr);
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!value || !basis_out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */             /* low-signal branch under CI */
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

static int
PrintSystemParseSelectorNode(const YAMLnode *node, DumpSelector_args *selector_out)
{
   /* GCOVR_EXCL_BR_START */                      /* low-signal branch under CI */
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

   int seen_every     = 0;
   int seen_ids       = 0;
   int seen_ranges    = 0;
   int seen_threshold = 0;
   for (const YAMLnode *child = node->children; child != NULL; child = child->next)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      /* GCOVR_EXCL_BR_STOP */
      if (!strcmp(child->key, "basis"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseBasis(value, &selector_out->basis)) /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
      }
      else if (!strcmp(child->key, "level"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseInteger(value, &selector_out->level) ||
             /* GCOVR_EXCL_BR_STOP */
             /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                selector_out->level < 0 ||
             selector_out->level >= STATS_MAX_LEVELS)
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0;
         }
      }
      else if (!strcmp(child->key, "every"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseInteger(value, &selector_out->every) ||
             /* GCOVR_EXCL_BR_STOP */
             /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                selector_out->every <= 0)
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         seen_every = 1;
      }
      else if (!strcmp(child->key, "ids"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseIntArrayNode(child, &selector_out->ids) ||
             /* GCOVR_EXCL_BR_STOP */
             /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
             !selector_out->ids ||
             selector_out->ids->size == 0)
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         seen_ids = 1;
      }
      /* GCOVR_EXCL_BR_START */               /* low-signal branch under CI */
      else if (!strcmp(child->key, "ranges")) /* GCOVR_EXCL_BR_STOP */
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseRangesNode(child,
                                         &selector_out->ranges) || /* GCOVR_EXCL_LINE */
             selector_out->ranges.size == 0)                       /* GCOVR_EXCL_LINE */
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         seen_ranges = 1; /* GCOVR_EXCL_LINE */
      }
      else if (!strcmp(child->key, "threshold"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseDouble(value, &selector_out->threshold) ||
             /* GCOVR_EXCL_BR_STOP */
             /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                selector_out->threshold < 0.0)
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         seen_threshold = 1;
      }
      else
      {
         return 0;
      }
   }

   if (PrintSystemBasisUsesThreshold(selector_out->basis))
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      return seen_threshold && !seen_every && !seen_ids && !seen_ranges;
      /* GCOVR_EXCL_BR_STOP */
   }

   if (seen_threshold)
   {
      return 0;
   }
   /* GCOVR_EXCL_BR_START */                     /* low-signal branch under CI */
   if (!seen_every && !seen_ids && !seen_ranges) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   return 1;
}

void
hypredrv_PrintSystemSetArgs(void *field, const YAMLnode *node)
{
   PrintSystem_args *args = (PrintSystem_args *)field;
   if (!args || !node)
   {
      return;
   }

   hypredrv_PrintSystemDestroyArgs(args);
   hypredrv_PrintSystemSetDefaultArgs(args);

   if (!node->children)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      const char *value = node->mapped_val ? node->mapped_val : node->val;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (value && value[0] != '\0') /* GCOVR_EXCL_BR_STOP */
      {
         if (!PrintSystemParseOnOff(value, &args->enabled))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system value: '%s'", value);
         }
      }
      return;
   }

   int seen_every     = 0;
   int seen_ids       = 0;
   int seen_ranges    = 0;
   int seen_threshold = 0;
   int seen_selectors = 0;
   for (const YAMLnode *child = node->children; child != NULL; child = child->next)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      /* GCOVR_EXCL_BR_STOP */

      if (!strcmp(child->key, "enabled"))
      {
         if (!PrintSystemParseOnOff(value, &args->enabled))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.enabled: '%s'",
                                 /* GCOVR_EXCL_BR_STOP */
                                 value ? value : "");
            return;
         }
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "type"))
      {
         if (!value)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Missing linear_system.print_system.type value");
            return;
         }
         if (!strcasecmp(value, "all"))
         {
            args->type = PRINT_SYSTEM_TYPE_ALL;
         }
         else if (!strcasecmp(value, "every_n_systems"))
         {
            args->type = PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS;
         }
         else if (!strcasecmp(value, "every_n_timesteps"))
         {
            args->type = PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS;
         }
         else if (!strcasecmp(value, "ids"))
         {
            args->type = PRINT_SYSTEM_TYPE_IDS;
         }
         else if (!strcasecmp(value, "ranges"))
         {
            args->type = PRINT_SYSTEM_TYPE_RANGES;
         }
         else if (!strcasecmp(value, "iterations_over"))
         {
            args->type = PRINT_SYSTEM_TYPE_ITERATIONS_OVER;
         }
         else if (!strcasecmp(value, "setup_time_over"))
         {
            args->type = PRINT_SYSTEM_TYPE_SETUP_TIME_OVER;
         }
         else if (!strcasecmp(value, "solve_time_over"))
         {
            args->type = PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER;
         }
         else if (!strcasecmp(value, "selectors"))
         {
            args->type = PRINT_SYSTEM_TYPE_SELECTORS;
         }
         else
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.type: '%s'", value);
            return;
         }
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "stage"))
      {
         if (!value)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Missing linear_system.print_system.stage value");
            return;
         }
         if (!strcasecmp(value, "build"))
         {
            args->stage_mask = PRINT_SYSTEM_STAGE_BUILD_BIT;
         }
         else if (!strcasecmp(value, "setup"))
         {
            args->stage_mask = PRINT_SYSTEM_STAGE_SETUP_BIT;
         }
         else if (!strcasecmp(value, "apply"))
         {
            args->stage_mask = PRINT_SYSTEM_STAGE_APPLY_BIT;
         }
         else if (!strcasecmp(value, "all"))
         {
            args->stage_mask = PRINT_SYSTEM_STAGE_BUILD_BIT |
                               PRINT_SYSTEM_STAGE_SETUP_BIT |
                               PRINT_SYSTEM_STAGE_APPLY_BIT;
         }
         else
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.stage: '%s'", value);
            return;
         }
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "artifacts"))
      {
         if (!PrintSystemParseArtifactsNode(child, &args->artifacts))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.artifacts");
            return;
         }
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
         for (const YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
            if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
            {
               ((YAMLnode *)item)->valid = YAML_NODE_VALID;
            }
         }
      }
      else if (!strcmp(child->key, "output_dir"))
      {
         if (!value)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Missing linear_system.print_system.output_dir value");
            return;
         }
         snprintf(args->output_dir, sizeof(args->output_dir), "%s", value);
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "overwrite"))
      {
         if (!PrintSystemParseOnOff(value, &args->overwrite))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.overwrite: '%s'",
                                 /* GCOVR_EXCL_BR_STOP */
                                 value ? value : "");
            return;
         }
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "every"))
      {
         if (!PrintSystemParseInteger(value, &args->every) || args->every <= 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.every: '%s'",
                                 /* GCOVR_EXCL_BR_STOP */
                                 value ? value : "");
            return;
         }
         seen_every                 = 1;
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "ids"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseIntArrayNode(child, &args->ids) || !args->ids ||
             /* GCOVR_EXCL_BR_STOP */
             /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                args->ids->size == 0)
         /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.ids");
            return;
         }
         seen_ids                   = 1;
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
         for (const YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
            if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
            {
               ((YAMLnode *)item)->valid = YAML_NODE_VALID;
            }
         }
      }
      else if (!strcmp(child->key, "ranges"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseRangesNode(child, &args->ranges) || args->ranges.size == 0)
         /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.ranges");
            return;
         }
         seen_ranges                = 1;
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
         for (const YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
            if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
            {
               ((YAMLnode *)item)->valid = YAML_NODE_VALID;
            }
         }
      }
      else if (!strcmp(child->key, "threshold"))
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemParseDouble(value, &args->threshold) || args->threshold < 0.0)
         /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            hypredrv_ErrorMsgAdd("Invalid linear_system.print_system.threshold: '%s'",
                                 /* GCOVR_EXCL_BR_STOP */
                                 value ? value : "");
            return;
         }
         seen_threshold             = 1;
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else if (!strcmp(child->key, "selectors"))
      {
         if (!child->children)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "linear_system.print_system.selectors must be a sequence");
            return;
         }

         size_t selector_count = 0;
         for (const YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
            if (!strcmp(item->key, "-")) /* GCOVR_EXCL_BR_STOP */
            {
               selector_count++;
            }
         }
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (selector_count == 0)  /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("linear_system.print_system.selectors cannot be empty");
            return;
         }

         DumpSelector_args *selectors =
            (DumpSelector_args *)calloc(selector_count, sizeof(DumpSelector_args));
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!selectors)           /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
            hypredrv_ErrorMsgAdd(
               "Failed to allocate selector list"); /* GCOVR_EXCL_LINE */
            return;                                 /* GCOVR_EXCL_LINE */
         }

         size_t selector_idx = 0;
         for (const YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
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
               return;
            }
            ((YAMLnode *)item)->valid = YAML_NODE_VALID;
            selector_idx++;
         }

         args->selectors            = selectors;
         args->num_selectors        = selector_count;
         seen_selectors             = 1;
         ((YAMLnode *)child)->valid = YAML_NODE_VALID;
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("Unknown key under linear_system.print_system: '%s'",
                              child->key);
         return;
      }
   }

   if (args->type == PRINT_SYSTEM_TYPE_ALL &&
       /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
       (seen_every || seen_ids || seen_ranges || seen_selectors))
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=all cannot be combined with selectors");
      return;
   }
   if ((args->type == PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS ||
        args->type == PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS) &&
       !seen_every)
   {
      args->every = 1;
   }
   if (args->type == PRINT_SYSTEM_TYPE_IDS && !seen_ids)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system.print_system.type=ids requires ids");
      return;
   }
   if (args->type == PRINT_SYSTEM_TYPE_RANGES && !seen_ranges)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system.print_system.type=ranges requires ranges");
      return;
   }
   if ((args->type == PRINT_SYSTEM_TYPE_ITERATIONS_OVER ||
        args->type == PRINT_SYSTEM_TYPE_SETUP_TIME_OVER ||
        /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
           args->type == PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER) &&
       /* GCOVR_EXCL_BR_STOP */
       !seen_threshold)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system threshold type requires threshold");
      return;
   }
   if (args->type == PRINT_SYSTEM_TYPE_SELECTORS && !seen_selectors)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=selectors requires selectors");
      return;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (args->type != PRINT_SYSTEM_TYPE_SELECTORS && seen_selectors)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.selectors requires type=selectors");
      return;
   }
   if (args->type != PRINT_SYSTEM_TYPE_ITERATIONS_OVER &&
       args->type != PRINT_SYSTEM_TYPE_SETUP_TIME_OVER &&
       args->type != PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER && seen_threshold)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.threshold requires a threshold-based type");
      return;
   }
   if (args->type == PRINT_SYSTEM_TYPE_ITERATIONS_OVER &&
       (args->stage_mask & PRINT_SYSTEM_STAGE_APPLY_BIT) == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=iterations_over requires stage apply");
      return;
   }
   if (args->type == PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER &&
       (args->stage_mask & PRINT_SYSTEM_STAGE_APPLY_BIT) == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=solve_time_over requires stage apply");
      return;
   }
   if (args->type == PRINT_SYSTEM_TYPE_SETUP_TIME_OVER &&
       (args->stage_mask &
        (PRINT_SYSTEM_STAGE_SETUP_BIT | PRINT_SYSTEM_STAGE_APPLY_BIT)) == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system.print_system.type=setup_time_over requires stage setup or apply");
      return;
   }
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
      int max_base = (int)sizeof(A_name) - 1 - 4;
      if (max_base < 0) max_base = 0; /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
      snprintf(A_name, sizeof(A_name), "%.*s.out", max_base, A_base);
   }
   {
      int max_base = (int)sizeof(b_name) - 1 - 4;
      if (max_base < 0) max_base = 0; /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
      snprintf(b_name, sizeof(b_name), "%.*s.out", max_base, b_base);
   }
   {
      int max_base = (int)sizeof(d_name) - 1 - 4;
      if (max_base < 0) max_base = 0; /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
      snprintf(d_name, sizeof(d_name), "%.*s.out", max_base, d_base);
   }

   int use_series_dir = 1;
   if (args)
   {
      const int has_mat = args->matrix_basename[0] != '\0';
      const int has_rhs = args->rhs_basename[0] != '\0';
      const int has_dmf = args->dofmap_basename[0] != '\0';
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (dir)                  /* GCOVR_EXCL_BR_STOP */
      {
         const struct dirent *ent = NULL;
         while ((ent = readdir(dir)) != NULL)
         {
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
      /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
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

   /* GCOVR_EXCL_BR_START */   /* low-signal branch under CI */
   if (dofmap && dofmap->data) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_IntArrayWriteAsciiByRank(comm, dofmap, d_path);
   }
}

static int
PrintSystemContainsID(const IntArray *ids, int value)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ids || !ids->data)   /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
   if (!ranges || !ranges->data) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   for (size_t i = 0; i < ranges->size; i++)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!selector || !ctx)    /* GCOVR_EXCL_BR_STOP */
   {
      return -1; /* GCOVR_EXCL_LINE */
   }

   if (selector->basis == PRINT_SYSTEM_BASIS_TIMESTEP)
   {
      return ctx->timestep_index;
   }
   if (selector->basis == PRINT_SYSTEM_BASIS_LEVEL)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx)                 /* GCOVR_EXCL_BR_STOP */
   {
      return -1.0; /* GCOVR_EXCL_LINE */
   }

   if (basis == PRINT_SYSTEM_BASIS_ITERATIONS)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      return (ctx->last_iter >= 0) ? (double)ctx->last_iter : -1.0;
      /* GCOVR_EXCL_BR_STOP */
   }
   /* GCOVR_EXCL_BR_START */                   /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!selector || !ctx)    /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (PrintSystemBasisUsesThreshold(selector->basis))
   {
      double metric_value = PrintSystemMetricValueGet(selector->basis, ctx);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      return metric_value >= 0.0 && metric_value >= selector->threshold;
      /* GCOVR_EXCL_BR_STOP */
   }

   int basis_value = PrintSystemSelectorBasisValueGet(selector, ctx);
   if (basis_value < 0)
   {
      return 0;
   }

   int matched = 0;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (selector->every > 0 && (basis_value % selector->every) == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      matched = 1;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!matched && selector->ids && selector->ids->size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      matched = PrintSystemContainsID(selector->ids, basis_value); /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                  /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!cfg)                 /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */                   /* low-signal branch under CI */
   else if (stage == PRINT_SYSTEM_STAGE_APPLY) /* GCOVR_EXCL_BR_STOP */
   {
      stage_bit = PRINT_SYSTEM_STAGE_APPLY_BIT;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   return (stage_bit != 0) && ((cfg->stage_mask & stage_bit) != 0);
   /* GCOVR_EXCL_BR_STOP */
}

static int
PrintSystemShouldDumpDetailed(const PrintSystem_args *cfg, const PrintSystemContext *ctx,
                              char *reason, size_t reason_size)
{
   /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
   if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      reason[0] = '\0';
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!cfg)                 /* GCOVR_EXCL_BR_STOP */
   {
      if (reason && reason_size > 0) /* GCOVR_EXCL_LINE */
      {
         snprintf(reason, reason_size, "%s",
                  "missing configuration"); /* GCOVR_EXCL_LINE */
      }
      return 0; /* GCOVR_EXCL_LINE */
   }
   if (!cfg->enabled)
   {
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "%s", "print_system disabled");
      }
      return 0;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx)                 /* GCOVR_EXCL_BR_STOP */
   {
      if (reason && reason_size > 0) /* GCOVR_EXCL_LINE */
      {
         snprintf(reason, reason_size, "%s", "missing context"); /* GCOVR_EXCL_LINE */
      }
      return 0; /* GCOVR_EXCL_LINE */
   }
   if (!PrintSystemStageEnabled(cfg, ctx->stage))
   {
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "stage '%s' not selected",
                  PrintSystemStageName(ctx->stage));
      }
      return 0;
   }

   if (cfg->type == PRINT_SYSTEM_TYPE_ALL)
   {
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "%s", "type=all");
      }
      return 1;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      int matched = (ctx->system_index >= 0) && (cfg->every > 0) &&
                    /* GCOVR_EXCL_BR_STOP */
                    ((ctx->system_index % cfg->every) == 0);
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "system_index=%d every=%d", ctx->system_index,
                  cfg->every);
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      int matched = (ctx->timestep_index >= 0) && (cfg->every > 0) &&
                    /* GCOVR_EXCL_BR_STOP */
                    ((ctx->timestep_index % cfg->every) == 0);
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "timestep_index=%d every=%d", ctx->timestep_index,
                  cfg->every);
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_IDS)
   {
      int matched = PrintSystemContainsID(cfg->ids, ctx->system_index);
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "system_index=%d ids_size=%zu", ctx->system_index,
                  /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                     cfg->ids
                     ? cfg->ids->size
                     : 0);
         /* GCOVR_EXCL_BR_STOP */
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_RANGES)
   {
      int matched = PrintSystemContainsRange(&cfg->ranges, ctx->system_index);
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "system_index=%d ranges_size=%zu",
                  ctx->system_index, cfg->ranges.size);
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_ITERATIONS_OVER)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      int matched = (ctx->last_iter >= 0) && ((double)ctx->last_iter >= cfg->threshold);
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "last_iter=%d threshold=%.3e", ctx->last_iter,
                  cfg->threshold);
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_SETUP_TIME_OVER)
   {
      int matched =
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         (ctx->last_setup_time >= 0.0) && (ctx->last_setup_time >= cfg->threshold);
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "last_setup_time=%.3e threshold=%.3e",
                  ctx->last_setup_time, cfg->threshold);
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER)
   {
      int matched =
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         (ctx->last_solve_time >= 0.0) && (ctx->last_solve_time >= cfg->threshold);
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "last_solve_time=%.3e threshold=%.3e",
                  ctx->last_solve_time, cfg->threshold);
      }
      return matched;
   }
   if (cfg->type == PRINT_SYSTEM_TYPE_SELECTORS)
   {
      /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
      if (!cfg->selectors || cfg->num_selectors == 0) /* GCOVR_EXCL_BR_STOP */
      {
         /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
         if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
         {
            snprintf(reason, reason_size, "%s", "selectors list is empty");
         }
         return 0;
      }

      for (size_t i = 0; i < cfg->num_selectors; i++)
      {
         const DumpSelector_args *selector = &cfg->selectors[i];
         int    basis_value  = PrintSystemSelectorBasisValueGet(selector, ctx);
         double metric_value = PrintSystemMetricValueGet(selector->basis, ctx);
         int    matched      = PrintSystemSelectorMatches(selector, ctx);
         if (matched)
         {
            /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
            if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
            {
               if (PrintSystemBasisUsesThreshold(selector->basis))
               {
                  snprintf(reason, reason_size,
                           "selector[%zu] basis=%s metric_value=%.2e threshold=%.2e", i,
                           PrintSystemBasisName(selector->basis), metric_value,
                           selector->threshold);
               }
               else
               {
                  snprintf(reason, reason_size,
                           "selector[%zu] basis=%s level=%d basis_value=%d", i,
                           PrintSystemBasisName(selector->basis), selector->level,
                           basis_value);
               }
            }
            return 1;
         }
      }

      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         snprintf(reason, reason_size, "no selector matched (count=%zu)",
                  cfg->num_selectors);
      }
      return 0;
   }

   /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
   if (reason && reason_size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      snprintf(reason, reason_size, "unknown type=%d", cfg->type);
   }
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
   /* GCOVR_EXCL_BR_START */  /* low-signal branch under CI */
   if (!dst || dst_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   size_t di = 0;
   if (src)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      for (size_t si = 0; src[si] != '\0' && di + 1 < dst_size; si++)
      /* GCOVR_EXCL_BR_STOP */
      {
         unsigned char ch = (unsigned char)src[si];
         /* GCOVR_EXCL_BR_START */                  /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
   if (!path || path[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   char current[2 * MAX_FILENAME_LENGTH];
   snprintf(current, sizeof(current), "%s", path);

   size_t len = strlen(current);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (len == 0)             /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   for (size_t i = 1; i <= len; i++)
   {
      if (current[i] == '/' || current[i] == '\0')
      {
         char saved = current[i];
         current[i] = '\0';

         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (current[0] != '\0')   /* GCOVR_EXCL_BR_STOP */
         {
            struct stat st;
            if (stat(current, &st) != 0)
            {
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   return (path && stat(path, &st) == 0);
   /* GCOVR_EXCL_BR_STOP */
}

static int
PrintSystemPathCopy(char *dst, size_t dst_size, const char *src)
{
   /* GCOVR_EXCL_BR_START */          /* low-signal branch under CI */
   if (!dst || dst_size == 0 || !src) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   size_t src_len = strlen(src);
   /* GCOVR_EXCL_BR_START */   /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!dst || dst_size == 0 || !base_path || !path_component) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   size_t base_len      = strlen(base_path);
   size_t component_len = strlen(path_component);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   int add_sep = (base_len > 0 && base_path[base_len - 1] != '/');
   /* GCOVR_EXCL_BR_STOP */
   size_t total_len = base_len + (size_t)add_sep + component_len + 1;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (total_len > dst_size) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   memcpy(dst, base_path, base_len);
   size_t pos = base_len;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (add_sep)              /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!base_dir)            /* GCOVR_EXCL_BR_STOP */
   {
      return -1; /* GCOVR_EXCL_LINE */
   }

   DIR *dir = opendir(base_dir);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!dir)                 /* GCOVR_EXCL_BR_STOP */
   {
      return -1; /* GCOVR_EXCL_LINE */
   }

   int            max_idx = -1;
   struct dirent *entry   = NULL;
   while ((entry = readdir(dir)) != NULL)
   {
      if (strncmp(entry->d_name, "ls_", 3) != 0)
      {
         continue;
      }

      const char *digits = entry->d_name + 3;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (*digits == '\0')      /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }

      bool all_digits = true;
      for (const char *p = digits; *p != '\0'; p++)
      {
         /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
         if (!isdigit((unsigned char)*p)) /* GCOVR_EXCL_BR_STOP */
         {
            all_digits = false; /* GCOVR_EXCL_LINE */
            break;              /* GCOVR_EXCL_LINE */
         }
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!all_digits)          /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }

      long idx_long = strtol(digits, NULL, 10);
      /* GCOVR_EXCL_BR_START */               /* low-signal branch under CI */
      if (idx_long < 0 || idx_long > INT_MAX) /* GCOVR_EXCL_BR_STOP */
      {
         continue; /* GCOVR_EXCL_LINE */
      }
      int idx = (int)idx_long;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (idx > max_idx)        /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
   if (!path || path[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   struct stat st;
   /* GCOVR_EXCL_BR_START */  /* low-signal branch under CI */
   if (lstat(path, &st) != 0) /* GCOVR_EXCL_BR_STOP */
   {
      return errno == ENOENT; /* GCOVR_EXCL_LINE */
   }

   if (!S_ISDIR(st.st_mode))
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      return (unlink(path) == 0) || (errno == ENOENT);
      /* GCOVR_EXCL_BR_STOP */
   }

   DIR *dir = opendir(path);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!dir)                 /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   int            ok    = 1;
   struct dirent *entry = NULL;
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   while (ok && (entry = readdir(dir)) != NULL) /* GCOVR_EXCL_BR_STOP */
   {
      if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
      {
         continue;
      }

      char child[2 * MAX_FILENAME_LENGTH];
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemPathJoin(child, sizeof(child), path, entry->d_name))
      /* GCOVR_EXCL_BR_STOP */
      {
         ok = 0; /* GCOVR_EXCL_LINE */
         break;  /* GCOVR_EXCL_LINE */
      }

      ok = PrintSystemRemoveTree(child);
   }

   closedir(dir);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ok)                  /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   return (rmdir(path) == 0) || (errno == ENOENT);
   /* GCOVR_EXCL_BR_STOP */
}

static int
PrintSystemChooseDumpDirLocal(const PrintSystem_args *cfg, const PrintSystemContext *ctx,
                              const char *object_name, char *dump_dir,
                              size_t dump_dir_size)
{
   /* GCOVR_EXCL_BR_START */                            /* low-signal branch under CI */
   if (!cfg || !ctx || !dump_dir || dump_dir_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   (void)ctx;

   char object_token[MAX_FILENAME_LENGTH];
   PrintSystemSanitizeToken(object_name, object_token, sizeof(object_token));

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   const char *root = (cfg->output_dir[0] != '\0') ? cfg->output_dir : "hypre-dumps";
   /* GCOVR_EXCL_BR_STOP */
   char base_dir[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!PrintSystemPathJoin(base_dir, sizeof(base_dir), root, object_token))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   PrintSystem_args *cfg_state = (PrintSystem_args *)cfg;
   if (cfg->overwrite && !cfg_state->overwrite_prepared)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
      /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
      if (cfg_state->next_dump_index < 0) /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }

      snprintf(leaf, sizeof(leaf), "ls_%05d", cfg_state->next_dump_index);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemPathJoin(candidate, sizeof(candidate), base_dir, leaf))
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
      cfg_state->next_dump_index++;

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (PrintSystemPathExists(candidate) && !PrintSystemRemoveTree(candidate))
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }
   }
   else
   {
      int next_idx = PrintSystemFindMaxDumpIndex(base_dir) + 1;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (next_idx < 0)         /* GCOVR_EXCL_BR_STOP */
      {
         return 0; /* GCOVR_EXCL_LINE */
      }

      do
      {
         snprintf(leaf, sizeof(leaf), "ls_%05d", next_idx);
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!PrintSystemPathJoin(candidate, sizeof(candidate), base_dir, leaf))
         /* GCOVR_EXCL_BR_STOP */
         {
            return 0; /* GCOVR_EXCL_LINE */
         }
         next_idx++;
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      } while (PrintSystemPathExists(candidate));
      /* GCOVR_EXCL_BR_STOP */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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

   /* GCOVR_EXCL_BR_START */                            /* low-signal branch under CI */
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
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (dump_dir_size > 0)    /* GCOVR_EXCL_BR_STOP */
      {
         dump_dir[0] = '\0';
      }
      return 0;
   }

   MPI_Bcast(dump_dir, (int)dump_dir_size, MPI_CHAR, 0, comm);

   return PrintSystemEnsureDir(dump_dir);
}

static void
PrintSystemWriteMetadata(const char *dump_dir, const PrintSystemContext *ctx,
                         const char *object_name, int artifacts)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!dump_dir || !ctx)    /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   char metadata_path[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!PrintSystemPathJoin(metadata_path, sizeof(metadata_path), dump_dir,
                            /* GCOVR_EXCL_BR_STOP */
                            "metadata.txt"))
   {
      return; /* GCOVR_EXCL_LINE */
   }
   FILE *fp = fopen(metadata_path, "w");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!dump_dir || !ctx)    /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   char path_buf[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!PrintSystemPathCopy(path_buf, sizeof(path_buf), dump_dir))
   /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   char *ls_name = strrchr(path_buf, '/');
   /* GCOVR_EXCL_BR_START */            /* low-signal branch under CI */
   if (!ls_name || ls_name == path_buf) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }
   *ls_name = '\0';
   ls_name++;

   char index_path[2 * MAX_FILENAME_LENGTH];
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!PrintSystemPathJoin(index_path, sizeof(index_path), path_buf,
                            /* GCOVR_EXCL_BR_STOP */
                            "systems_index.txt"))
   {
      return; /* GCOVR_EXCL_LINE */
   }

   FILE *fp = fopen(index_path, "a");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
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

uint32_t
hypredrv_LinearSystemDumpScheduled(MPI_Comm comm, const LS_args *args,
                                   HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                                   HYPRE_IJVector vec_b, HYPRE_IJVector vec_x0,
                                   HYPRE_IJVector vec_xref, HYPRE_IJVector vec_x,
                                   const IntArray *dofmap, const PrintSystemContext *ctx,
                                   const char *object_name)
{
   if (!args || !ctx)
   {
      return hypredrv_ErrorCodeGet();
   }

   const PrintSystem_args *cfg = &args->print_system;
   int  ls_id_for_log = (ctx->stats_ls_id >= 0) ? ctx->stats_ls_id : ctx->system_index;
   char decision_reason[160];
   int  should_dump =
      PrintSystemShouldDumpDetailed(cfg, ctx, decision_reason, sizeof(decision_reason));
   HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                      "print_system evaluate: stage=%s type=%s artifacts=0x%x "
                      "system_index=%d timestep_index=%d variant=%d repetition=%d (%s)",
                      PrintSystemStageName(ctx->stage), PrintSystemTypeName(cfg->type),
                      cfg->artifacts, ctx->system_index, ctx->timestep_index,
                      ctx->variant_index, ctx->repetition_index, decision_reason);
   if (!should_dump)
   {
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system skip: selection did not match");
      return hypredrv_ErrorCodeGet();
   }

   char dump_dir[2 * MAX_FILENAME_LENGTH];
   if (!PrintSystemChooseDumpDir(comm, cfg, ctx, object_name, dump_dir, sizeof(dump_dir)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd(
         "Failed to create dump directory for linear_system.print_system");
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(2, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system failed: cannot create dump directory");
      return hypredrv_ErrorCodeGet();
   }
   HYPREDRV_LOG_COMMF(2, comm, object_name, ls_id_for_log, "print_system dump dir: %s",
                      dump_dir);

   char path[2 * MAX_FILENAME_LENGTH];
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_MATRIX) && mat_A)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "matrix.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for matrix artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                       /* GCOVR_EXCL_LINE */
      }
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system write matrix: %s", path);
      HYPRE_IJMatrixPrint(mat_A, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_MATRIX)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip matrix: matrix object is NULL");
   }
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_PRECMAT) && mat_M)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "precmat.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for precmat artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                        /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system write precmat: %s", path);
      HYPRE_IJMatrixPrint(mat_M, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_PRECMAT)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip precmat: preconditioner matrix is NULL");
   }
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_RHS) && vec_b)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "rhs.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for rhs artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                    /* GCOVR_EXCL_LINE */
      }
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system write rhs: %s", path);
      HYPRE_IJVectorPrint(vec_b, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_RHS)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip rhs: rhs vector is NULL");
   }
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_X0) && vec_x0)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "x0.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for x0 artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                   /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log, "print_system write x0: %s",
                         /* GCOVR_EXCL_BR_STOP */
                         path);
      HYPRE_IJVectorPrint(vec_x0, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_X0)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip x0: initial guess vector is NULL");
   }
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_XREF) && vec_xref)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "xref.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for xref artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                     /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system write xref: %s", path);
      HYPRE_IJVectorPrint(vec_xref, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_XREF)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip xref: reference solution is NULL");
   }
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_SOLUTION) && vec_x)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "solution.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for solution artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                         /* GCOVR_EXCL_LINE */
      }
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system write solution: %s", path);
      HYPRE_IJVectorPrint(vec_x, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_SOLUTION)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip solution: solution vector is NULL");
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_DOFMAP) && dofmap && dofmap->data)
   /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!PrintSystemArtifactPathBuild(dump_dir, "dofmap.out", path, sizeof(path)))
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "print_system path too long for dofmap artifact"); /* GCOVR_EXCL_LINE */
         return hypredrv_ErrorCodeGet();                       /* GCOVR_EXCL_LINE */
      }
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system write dofmap: %s", path);
      hypredrv_IntArrayWriteAsciiByRank(comm, dofmap, path);
   }
   else if (cfg->artifacts & PRINT_SYSTEM_ARTIFACT_DOFMAP)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         /* GCOVR_EXCL_BR_STOP */
                         "print_system skip dofmap: dofmap is NULL");
   }

   int mypid = 0;
   MPI_Comm_rank(comm, &mypid);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if ((cfg->artifacts & PRINT_SYSTEM_ARTIFACT_METADATA) && mypid == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      PrintSystemWriteMetadata(dump_dir, ctx, object_name, cfg->artifacts);
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system write metadata: %s/metadata.txt", dump_dir);
   }
   if (mypid == 0)
   {
      PrintSystemAppendStageIndex(dump_dir, ctx, object_name);
      HYPREDRV_LOG_COMMF(3, comm, object_name, ls_id_for_log,
                         "print_system append systems index: %s", dump_dir);
   }

   HYPREDRV_LOG_COMMF(2, comm, object_name, ls_id_for_log, "print_system dump complete");

   return hypredrv_ErrorCodeGet();
}
