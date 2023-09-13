/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "precon.h"

/*-----------------------------------------------------------------------------
 * PreconSetDefaultArgs
 *-----------------------------------------------------------------------------*/

int
PreconSetDefaultArgs(precon_t     precon_method,
                     precon_args *args)
{
   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         AMGSetDefaultArgs(&args->amg);
         break;

      case PRECON_MGR:
         MGRSetDefaultArgs(&args->mgr);
         break;

      case PRECON_ILU:
         ILUSetDefaultArgs(&args->ilu);
         break;

      default:
         ErrorMsgAddInvalidPreconOption((int) precon_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * PreconSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
PreconSetArgsFromYAML(precon_t      precon_method,
                      precon_args  *args,
                      YAMLnode     *node)
{
   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         AMGSetArgsFromYAML(&args->amg, node);
         break;

      case PRECON_MGR:
         MGRSetArgsFromYAML(&args->mgr, node);
         break;

      case PRECON_ILU:
         ILUSetArgsFromYAML(&args->ilu, node);
         break;

      default:
         ErrorMsgAddInvalidPreconOption((int) precon_method);
         return EXIT_FAILURE;
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
