/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "precon.h"

static const FieldOffsetMap precon_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(precon_args, amg, FIELD_TYPE_STRUCT, AMGSetArgs),
   FIELD_OFFSET_MAP_ENTRY(precon_args, mgr, FIELD_TYPE_STRUCT, MGRSetArgs),
   FIELD_OFFSET_MAP_ENTRY(precon_args, ilu, FIELD_TYPE_STRUCT, ILUSetArgs),
};

#define PRECON_NUM_FIELDS (sizeof(precon_field_offset_map) / sizeof(precon_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * PreconSetFieldByName
 *-----------------------------------------------------------------------------*/

void
PreconSetFieldByName(precon_args *args, const char *name, YAMLnode *node)
{
   for (size_t i = 0; i < PRECON_NUM_FIELDS; i++)
   {
      /* Which union type are we trying to set? */
      if (!strcmp(precon_field_offset_map[i].name, name))
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

   for(size_t i = 0; i < PRECON_NUM_FIELDS; i++)
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
   /* The "preconditioner" enry does not hold values, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * PreconGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PreconGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {{"amg", (int) PRECON_BOOMERAMG},
                             {"mgr", (int) PRECON_MGR},
                             {"ilu", (int) PRECON_ILU}};

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * PreconSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
PreconSetArgsFromYAML(precon_args *args, YAMLnode *parent)
{
   YAMLnode    *child;

   child = parent->children;
   while (child)
   {
      YAML_VALIDATE_NODE(child,
                         PreconGetValidKeys,
                         PreconGetValidValues);

      YAML_SET_ARG_STRUCT(child,
                          args,
                          PreconSetFieldByName);

      child = child->next;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * PreconCreate
 *-----------------------------------------------------------------------------*/

int
PreconCreate(precon_t         precon_method,
             precon_args     *args,
             HYPRE_IntArray  *dofmap,
             HYPRE_Solver    *precon_ptr)
{
   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         AMGCreate(&args->amg, precon_ptr);
         break;

      case PRECON_MGR:
         MGRCreate(&args->mgr, dofmap, precon_ptr);
         break;

      case PRECON_ILU:
         ILUCreate(&args->ilu, precon_ptr);
         break;

      default:
         *precon_ptr = NULL;
         ErrorMsgAddInvalidPreconOption((int) precon_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * PreconDestroy
 *-----------------------------------------------------------------------------*/

int
PreconDestroy(precon_t      precon_method,
              HYPRE_Solver *precon_ptr)
{
   if (*precon_ptr)
   {
      switch (precon_method)
      {
         case PRECON_BOOMERAMG:
            HYPRE_BoomerAMGDestroy(*precon_ptr);
            break;

         case PRECON_MGR:
            HYPRE_MGRDestroy(*precon_ptr);
            break;

         case PRECON_ILU:
            HYPRE_ILUDestroy(*precon_ptr);
            break;

         default:
            *precon_ptr = NULL;
            ErrorMsgAddInvalidPreconOption((int) precon_method);
            return EXIT_FAILURE;
      }

      *precon_ptr = NULL;
   }

   return EXIT_SUCCESS;
}
