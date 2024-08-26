/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "precon.h"

static const FieldOffsetMap precon_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(precon_args, amg, AMGSetArgs),
   FIELD_OFFSET_MAP_ENTRY(precon_args, mgr, MGRSetArgs),
   FIELD_OFFSET_MAP_ENTRY(precon_args, ilu, ILUSetArgs),
   FIELD_OFFSET_MAP_ENTRY(precon_args, fsai, FSAISetArgs),
   FIELD_OFFSET_MAP_ENTRY(precon_args, reuse, FieldTypeIntSet)
};

#define PRECON_NUM_FIELDS (sizeof(precon_field_offset_map) / sizeof(precon_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * PreconSetFieldByName
 *-----------------------------------------------------------------------------*/

void
PreconSetFieldByName(precon_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < PRECON_NUM_FIELDS; i++)
   {
      /* Which union type are we trying to set? */
      if (!strcmp(precon_field_offset_map[i].name, node->key))
      {
         precon_field_offset_map[i].setter(
            (void*)((char*) args + precon_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * PreconGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
PreconGetValidKeys(void)
{
   static const char* keys[PRECON_NUM_FIELDS];

   for (size_t i = 0; i < PRECON_NUM_FIELDS; i++)
   {
      keys[i] = precon_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * PreconGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PreconGetValidValues(const char* key)
{
   /* The "preconditioner" entry does not hold values, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * PreconGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PreconGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {{"amg",  (int) PRECON_BOOMERAMG},
                             {"mgr",  (int) PRECON_MGR},
                             {"ilu",  (int) PRECON_ILU},
                             {"fsai", (int) PRECON_FSAI}};

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * PreconSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
PreconSetDefaultArgs(precon_args *args)
{
   args->reuse = 0;
}

/*-----------------------------------------------------------------------------
 * PreconSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
PreconSetArgsFromYAML(precon_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         PreconGetValidKeys,
                         PreconGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          PreconSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * PreconCreate
 *-----------------------------------------------------------------------------*/

void
PreconCreate(precon_t         precon_method,
             precon_args     *args,
             IntArray        *dofmap,
             HYPRE_Precon    *precon_ptr)
{
   HYPRE_Precon precon = malloc(sizeof(hypre_Precon));

   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         AMGCreate(&args->amg, &precon->main);
         break;

      case PRECON_MGR:
         MGRSetDofmap(&args->mgr, dofmap);
         MGRCreate(&args->mgr, &precon->main);
         break;

      case PRECON_ILU:
         ILUCreate(&args->ilu, &precon->main);
         break;

      case PRECON_FSAI:
         FSAICreate(&args->fsai, &precon->main);
         break;

      default:
         precon->main = NULL;
   }

   *precon_ptr = precon;
}

/*-----------------------------------------------------------------------------
 * PreconDestroy
 *-----------------------------------------------------------------------------*/

void
PreconDestroy(precon_t       precon_method,
              precon_args   *args,
              HYPRE_Precon  *precon_ptr)
{
   HYPRE_Precon precon = *precon_ptr;

   if (!precon)
   {
      return;
   }

   if (precon->main)
   {
      switch (precon_method)
      {
         case PRECON_BOOMERAMG:
            HYPRE_BoomerAMGDestroy(precon->main);
            break;

         case PRECON_MGR:
            HYPRE_MGRDestroy(precon->main);

            /* TODO: should MGR free these internally? */
            if (args->mgr.coarsest_level.type == 0)
            {
               HYPRE_BoomerAMGDestroy(args->mgr.csolver);
               args->mgr.csolver = NULL;
            }
            if (args->mgr.level[0].f_relaxation.type == 2)
            {
               HYPRE_BoomerAMGDestroy(args->mgr.frelax[0]);
               args->mgr.frelax[0] = NULL;
            }
            if (args->mgr.level[0].g_relaxation.type == 16)
            {
               HYPRE_ILUDestroy(args->mgr.grelax[0]);
               args->mgr.grelax[0] = NULL;
            }
            break;

         case PRECON_ILU:
            HYPRE_ILUDestroy(precon->main);
            break;

         case PRECON_FSAI:
            HYPRE_FSAIDestroy(precon->main);
            break;

         default:
            return;
      }

      precon->main = NULL;
   }

   free(*precon_ptr);
   *precon_ptr = NULL;
}
