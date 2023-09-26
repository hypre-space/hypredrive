/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "cheby.h"

static const FieldOffsetMap cheby_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(Cheby_args, order, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(Cheby_args, eig_est, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(Cheby_args, variant, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(Cheby_args, scale, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(Cheby_args, fraction, FieldTypeDoubleSet)
};

#define CHEBY_NUM_FIELDS (sizeof(cheby_field_offset_map) / sizeof(cheby_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * ChebySetFieldByName
 *-----------------------------------------------------------------------------*/

void
ChebySetFieldByName(Cheby_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < CHEBY_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(cheby_field_offset_map[i].name, node->key))
      {
         cheby_field_offset_map[i].setter(
            (void*)((char*) args + cheby_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * ChebyGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
ChebyGetValidKeys(void)
{
   static const char* keys[CHEBY_NUM_FIELDS];

   for (size_t i = 0; i < CHEBY_NUM_FIELDS; i++)
   {
      keys[i] = cheby_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * ChebyGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
ChebyGetValidValues(const char* key)
{
   /* Don't impose any restrictions, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * ChebySetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
ChebySetDefaultArgs(Cheby_args *args)
{
   args->order    = 2;
   args->eig_est  = 10;
   args->variant  = 0;
   args->scale    = 1;
   args->fraction = 0.3;
}

/*-----------------------------------------------------------------------------
 * ChebySetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
ChebySetArgsFromYAML(Cheby_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         ChebyGetValidKeys,
                         ChebyGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          ChebySetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * ChebySetArgs
 *-----------------------------------------------------------------------------*/

void
ChebySetArgs(void *vargs, YAMLnode *parent)
{
   Cheby_args *args = (Cheby_args*) vargs;

   ChebySetDefaultArgs(args);
   ChebySetArgsFromYAML(args, parent);
}
