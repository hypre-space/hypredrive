/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef MGR_HEADER
#define MGR_HEADER

#include "amg.h"
#include "ilu.h"
#include "utils.h"

enum
{
   MAX_MGR_LEVELS = 32
};

/*--------------------------------------------------------------------------
 * Coarsest level solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRcls_args_struct
{
   HYPRE_Int type;

   /* Only one coarsest solver is active at a time; store its args in a union.
    * Note: anonymous union is a GNU C extension. */
   union
   {
      AMG_args amg;
      ILU_args ilu;
   };
} MGRcls_args;

/*--------------------------------------------------------------------------
 * F-Relaxation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRfrlx_args_struct
{
   HYPRE_Int type;
   HYPRE_Int num_sweeps;

   /* Only one fine-relaxation solver is active at a time. */
   union
   {
      AMG_args amg;
      ILU_args ilu;
   };
} MGRfrlx_args;

/*--------------------------------------------------------------------------
 * Global-Relaxation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRgrlx_args_struct
{
   HYPRE_Int type;
   HYPRE_Int num_sweeps;

   /* TODO: Ideally, these should be inside a union */
   ILU_args ilu;
} MGRgrlx_args;

/*--------------------------------------------------------------------------
 * MGR level arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRlvl_args_struct
{
   StackIntArray f_dofs;

   HYPRE_Int prolongation_type;
   HYPRE_Int restriction_type;
   HYPRE_Int coarse_level_type;

   MGRfrlx_args f_relaxation;
   MGRgrlx_args g_relaxation;
} MGRlvl_args;

/*--------------------------------------------------------------------------
 * MGR preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGR_args_struct
{
   IntArray *dofmap;

   HYPRE_Int  non_c_to_f;
   HYPRE_Int  pmax;
   HYPRE_Int  max_iter;
   HYPRE_Int  num_levels;
   HYPRE_Int  relax_type; /* TODO: we shouldn't need this */
   HYPRE_Int  print_level;
   HYPRE_Int  nonglk_max_elmts;
   HYPRE_Real tolerance;
   HYPRE_Real coarse_th;

   MGRlvl_args level[MAX_MGR_LEVELS - 1];
   MGRcls_args coarsest_level;

   HYPRE_Solver csolver;
   HYPRE_Solver frelax[MAX_MGR_LEVELS - 1];
   HYPRE_Solver grelax[MAX_MGR_LEVELS - 1];
} MGR_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void MGRSetArgs(void *, const YAMLnode *);
void MGRSetDofmap(MGR_args *, IntArray *);
void MGRCreate(MGR_args *, HYPRE_Solver *);

/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

/**
 * @brief A macro to handle level attributes based on their names and types.
 *
 * This macro simplifies the process of comparing attribute names and accessing the
 * corresponding attributes from a structure. It takes a buffer and a type, generates the
 * attribute string by replacing '.' with ':', and compares it with a string variable
 * name, assumed to be defined in the caller function. If they match, the buffer is
 * populated with the values from the structure's attributes and it is returned. This
 * macro is used for setting up the input parameters of MGR.
 *
 * @details Here is an example of how to use this macro (assuming that ibuf and rbuf are
 * statically allocated variables defined in the caller function):
 *
 * @code
 *    HANDLE_MGR_LEVEL_ATTRIBUTE(ibuf, f_relaxation.type)
 *    HANDLE_MGR_LEVEL_ATTRIBUTE(ibuf, f_relaxation.num_sweeps)
 *    HANDLE_MGR_LEVEL_ATTRIBUTE(rbuf, f_relaxation.weights)
 * @endcode
 *
 * In the above example, if name is "f_relaxation:type", "f_relaxation:num_sweeps", or
 * "f_relaxation:weights", the corresponding attribute values will be stored in the
 * buffers (ibuf or rbuf).
 *
 * @param _buffer The buffer to store the attribute values. It should be of the correct
 * type to hold the attribute values.
 * @param _type   The type and name of the attribute in the structure, using '.' notation
 * to access nested attributes, e.g., struct_name.attribute_name.
 * @return        A buffer containing the attribute values if the names match, NULL
 * otherwise.
 */
#define HANDLE_MGR_LEVEL_ATTRIBUTE(_buffer, _type)                                    \
   {                                                                                  \
      char str[] = #_type;                                                            \
      for (size_t i = 0; i < sizeof(str); i++)                                        \
         if (str[i] == '.') str[i] = ':';                                             \
      if (!strcmp(str, name))                                                         \
      {                                                                               \
         if (!strcmp(#_type, "f_relaxation.num_sweeps"))                              \
         {                                                                            \
            for (size_t i = 0; i < (size_t)(args->num_levels - 1); i++)               \
            {                                                                         \
               _buffer[i] =                                                           \
                  (args->level[i].f_relaxation.type >= 0) ? args->level[i]._type : 0; \
            }                                                                         \
         }                                                                            \
         else if (!strcmp(#_type, "g_relaxation.num_sweeps"))                         \
         {                                                                            \
            for (size_t i = 0; i < (size_t)(args->num_levels - 1); i++)               \
            {                                                                         \
               _buffer[i] =                                                           \
                  (args->level[i].g_relaxation.type >= 0) ? args->level[i]._type : 0; \
            }                                                                         \
         }                                                                            \
         else                                                                         \
         {                                                                            \
            for (size_t i = 0; i < (size_t)(args->num_levels - 1); i++)               \
            {                                                                         \
               _buffer[i] = args->level[i]._type;                                     \
            }                                                                         \
         }                                                                            \
         return _buffer;                                                              \
      }                                                                               \
   }

/*-----------------------------------------------------------------------------
 * Type-setting wrapper macro for union fields.
 *
 * For structs with unions (MGRcls, MGRfrlx, MGRgrlx), when we parse a nested
 * block like `ilu: {...}`, we need to also set the `type` field. This macro
 * generates a wrapper that recovers the parent struct and sets type before
 * calling the real setter.
 *
 * @param _func_name Desired generated wrapper name (e.g., MGRclsILUSetArgs)
 * @param _parent    Parent struct type (e.g., MGRcls_args)
 * @param _field     Union field name (e.g., amg, ilu)
 * @param _type      Type value to set (e.g., 0, 32)
 * @param _setter    Real setter function (e.g., AMGSetArgs)
 *-----------------------------------------------------------------------------*/
#define DEFINE_TYPED_SETTER(_func_name, _parent, _field, _type, _setter)    \
static void _func_name(void *v, const YAMLnode *n)                       \
{                                                                        \
   ((_parent *)((char *)v - offsetof(_parent, _field)))->type = (_type); \
   _setter(v, n);                                                        \
}

#endif /* MGR_HEADER */
