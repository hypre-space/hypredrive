#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "amg.h"
#include "bicgstab.h"
#include "cheby.h"
#include "containers.h"
#include "error.h"
#include "fgmres.h"
#include "fsai.h"
#include "gmres.h"
#include "ilu.h"
#include "mgr.h"
#include "krylov.h"
#include "pcg.h"
#include "test_helpers.h"
#include "yaml.h"

/* Internal/generated functions not in public headers */
void           hypredrv_AMGSetArgsFromYAML(void *, YAMLnode *);
void           hypredrv_AMGintSetArgsFromYAML(void *, YAMLnode *);
void           hypredrv_AMGaggSetArgsFromYAML(void *, YAMLnode *);
void           hypredrv_AMGcsnSetArgsFromYAML(void *, YAMLnode *);
void           hypredrv_AMGrlxSetArgsFromYAML(void *, YAMLnode *);
void           hypredrv_AMGsmtSetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_AMGGetValidKeys(void);
StrIntMapArray hypredrv_AMGGetValidValues(const char *);
StrArray       hypredrv_AMGintGetValidKeys(void);
StrIntMapArray hypredrv_AMGintGetValidValues(const char *);
StrArray       hypredrv_AMGaggGetValidKeys(void);
StrIntMapArray hypredrv_AMGaggGetValidValues(const char *);
StrArray       hypredrv_AMGcsnGetValidKeys(void);
StrIntMapArray hypredrv_AMGcsnGetValidValues(const char *);
StrArray       hypredrv_AMGrlxGetValidKeys(void);
StrIntMapArray hypredrv_AMGrlxGetValidValues(const char *);
StrArray       hypredrv_AMGsmtGetValidKeys(void);
StrIntMapArray hypredrv_AMGsmtGetValidValues(const char *);
void           hypredrv_FSAISetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_FSAIGetValidKeys(void);
StrIntMapArray hypredrv_FSAIGetValidValues(const char *);
void           hypredrv_ILUSetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_ILUGetValidKeys(void);
StrIntMapArray hypredrv_ILUGetValidValues(const char *);
void           hypredrv_ILUSetDefaultArgs(ILU_args *);
void           hypredrv_FSAISetDefaultArgs(FSAI_args *);
void           hypredrv_PCGSetDefaultArgs(PCG_args *);
void           hypredrv_PCGSetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_PCGGetValidKeys(void);
StrIntMapArray hypredrv_PCGGetValidValues(const char *);
void           hypredrv_GMRESSetDefaultArgs(GMRES_args *);
void           hypredrv_GMRESSetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_GMRESGetValidKeys(void);
StrIntMapArray hypredrv_GMRESGetValidValues(const char *);
void           hypredrv_FGMRESSetDefaultArgs(FGMRES_args *);
void           hypredrv_FGMRESSetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_FGMRESGetValidKeys(void);
StrIntMapArray hypredrv_FGMRESGetValidValues(const char *);
void           hypredrv_BiCGSTABSetDefaultArgs(BiCGSTAB_args *);
void           hypredrv_BiCGSTABSetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_BiCGSTABGetValidKeys(void);
StrIntMapArray hypredrv_BiCGSTABGetValidValues(const char *);
void           hypredrv_ChebySetArgsFromYAML(void *, YAMLnode *);
StrArray       hypredrv_ChebyGetValidKeys(void);
StrIntMapArray hypredrv_ChebyGetValidValues(const char *);

/* MGR (custom YAML parser + generated sub-components) */
void           hypredrv_MGRSetDefaultArgs(MGR_args *);
void           hypredrv_MGRSetArgsFromYAML(void *, YAMLnode *);
StrIntMapArray hypredrv_MGRGetValidValues(const char *);
StrIntMapArray hypredrv_MGRlvlGetValidValues(const char *);
HYPRE_Int     *hypredrv_MGRConvertArgInt(MGR_args *, const char *);

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

static const char *
pick_value(StrIntMapArray map)
{
   if (map.size > 0 && map.data && map.data[0].str)
   {
      return map.data[0].str;
   }

   return "1";
}

/* Forward declarations for recursive population helpers */
static void populate_cheby(YAMLnode *parent, int level);
static void populate_fsai(YAMLnode *parent, int level);
static void populate_ilu(YAMLnode *parent, int level);
static void populate_amgint(YAMLnode *parent, int level);
static void populate_amgagg(YAMLnode *parent, int level);
static void populate_amgcsn(YAMLnode *parent, int level);
static void populate_amgrlx(YAMLnode *parent, int level);
static void populate_amgsmt(YAMLnode *parent, int level);

static void
populate_cheby(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_ChebyGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_ChebyGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_fsai(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_FSAIGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_FSAIGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_ilu(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_ILUGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_ILUGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgint(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_AMGintGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_AMGintGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgagg(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_AMGaggGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_AMGaggGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgcsn(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_AMGcsnGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_AMGcsnGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgrlx(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_AMGrlxGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_AMGrlxGetValidValues(key);

      if (!strcmp(key, "chebyshev"))
      {
         YAMLnode *cheby = add_child(parent, key, "", level);
         populate_cheby(cheby, level + 1);
      }
      else
      {
         const char *val = pick_value(map);
         add_child(parent, key, val, level);
      }
   }
}

static void
populate_amgsmt(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_AMGsmtGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_AMGsmtGetValidValues(key);

      if (!strcmp(key, "fsai"))
      {
         YAMLnode *fsai = add_child(parent, key, "", level);
         populate_fsai(fsai, level + 1);
      }
      else if (!strcmp(key, "ilu"))
      {
         YAMLnode *ilu = add_child(parent, key, "", level);
         populate_ilu(ilu, level + 1);
      }
      else
      {
         const char *val = pick_value(map);
         add_child(parent, key, val, level);
      }
   }
}

static void
populate_amg(YAMLnode *parent, int level)
{
   StrArray keys = hypredrv_AMGGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = hypredrv_AMGGetValidValues(key);

      if (!strcmp(key, "interpolation"))
      {
         YAMLnode *node = add_child(parent, key, "", level);
         populate_amgint(node, level + 1);
      }
      else if (!strcmp(key, "aggressive"))
      {
         YAMLnode *node = add_child(parent, key, "", level);
         populate_amgagg(node, level + 1);
      }
      else if (!strcmp(key, "coarsening"))
      {
         YAMLnode *node = add_child(parent, key, "", level);
         populate_amgcsn(node, level + 1);
      }
      else if (!strcmp(key, "relaxation"))
      {
         YAMLnode *node = add_child(parent, key, "", level);
         populate_amgrlx(node, level + 1);
      }
      else if (!strcmp(key, "smoother"))
      {
         YAMLnode *node = add_child(parent, key, "", level);
         populate_amgsmt(node, level + 1);
      }
      else
      {
         const char *val = pick_value(map);
         add_child(parent, key, val, level);
      }
   }
}

static YAMLnode *
build_scalar_children(StrArray keys, StrIntMapArray (*get_vals)(const char *), int level)
{
   YAMLnode *parent = hypredrv_YAMLnodeCreate("root", "", level - 1);
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = get_vals(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
   return parent;
}

static void
exercise_solver_component(void (*set_args)(void *, YAMLnode *), StrArray (*get_keys)(void),
                          StrIntMapArray (*get_vals)(const char *), void *args)
{
   YAMLnode *parent = build_scalar_children(get_keys(), get_vals, 1);
   hypredrv_ErrorCodeResetAll();
   set_args(args, parent);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(parent);
}

static void
exercise_component_flat(void (*set_args)(void *, YAMLnode *), void *args, const char *key,
                        const char *val)
{
   /* Intentionally build a *flat* YAML node (no children) to hit the
    * flat-value parsing branch in macro-generated SetArgsFromYAML helpers. */
   YAMLnode *parent = hypredrv_YAMLnodeCreate(key, val, 0);
   hypredrv_ErrorCodeResetAll();
   set_args(args, parent);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_exhaustive_solver_parsers(void)
{
   PCG_args pcg;
   GMRES_args gmres;
   FGMRES_args fgmres;
   BiCGSTAB_args bicg;
   Cheby_args cheby;

   hypredrv_PCGSetDefaultArgs(&pcg);
   hypredrv_GMRESSetDefaultArgs(&gmres);
   hypredrv_FGMRESSetDefaultArgs(&fgmres);
   hypredrv_BiCGSTABSetDefaultArgs(&bicg);
   hypredrv_ChebySetDefaultArgs(&cheby);

   exercise_solver_component(hypredrv_PCGSetArgsFromYAML, hypredrv_PCGGetValidKeys, hypredrv_PCGGetValidValues, &pcg);
   exercise_solver_component(hypredrv_GMRESSetArgsFromYAML, hypredrv_GMRESGetValidKeys, hypredrv_GMRESGetValidValues,
                             &gmres);
   exercise_solver_component(hypredrv_FGMRESSetArgsFromYAML, hypredrv_FGMRESGetValidKeys, hypredrv_FGMRESGetValidValues,
                             &fgmres);
   exercise_solver_component(hypredrv_BiCGSTABSetArgsFromYAML, hypredrv_BiCGSTABGetValidKeys,
                             hypredrv_BiCGSTABGetValidValues, &bicg);
   exercise_solver_component(hypredrv_ChebySetArgsFromYAML, hypredrv_ChebyGetValidKeys, hypredrv_ChebyGetValidValues,
                             &cheby);

   /* Also exercise the flat-value SetArgsFromYAML branch (no children) */
   exercise_component_flat(hypredrv_PCGSetArgsFromYAML, &pcg, "pcg", "pcg");
   exercise_component_flat(hypredrv_GMRESSetArgsFromYAML, &gmres, "gmres", "gmres");
   exercise_component_flat(hypredrv_FGMRESSetArgsFromYAML, &fgmres, "fgmres", "fgmres");
   exercise_component_flat(hypredrv_BiCGSTABSetArgsFromYAML, &bicg, "bicgstab", "bicgstab");
   exercise_component_flat(hypredrv_ChebySetArgsFromYAML, &cheby, "cheby", "cheby");
}

static void
test_exhaustive_ilu_fsai_parsers(void)
{
   ILU_args  ilu;
   FSAI_args fsai;
   hypredrv_ILUSetDefaultArgs(&ilu);
   hypredrv_FSAISetDefaultArgs(&fsai);

   YAMLnode *ilu_parent = build_scalar_children(hypredrv_ILUGetValidKeys(), hypredrv_ILUGetValidValues, 1);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ILUSetArgsFromYAML(&ilu, ilu_parent);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(ilu_parent);

   YAMLnode *fsai_parent = build_scalar_children(hypredrv_FSAIGetValidKeys(), hypredrv_FSAIGetValidValues, 1);
   hypredrv_ErrorCodeResetAll();
   hypredrv_FSAISetArgsFromYAML(&fsai, fsai_parent);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(fsai_parent);

   exercise_component_flat(hypredrv_ILUSetArgsFromYAML, &ilu, "ilu", "ilu");
   exercise_component_flat(hypredrv_FSAISetArgsFromYAML, &fsai, "fsai", "fsai");
}

static void
test_exhaustive_amg_parser(void)
{
   AMG_args args;
   hypredrv_AMGSetDefaultArgs(&args);

   YAMLnode *root = hypredrv_YAMLnodeCreate("amg", "", 0);
   populate_amg(root, 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_AMGSetArgsFromYAML(&args, root);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_YAMLnodeDestroy(root);

   /* Flat-value branch (no children) */
   exercise_component_flat((void (*)(void *, YAMLnode *))hypredrv_AMGSetArgsFromYAML, &args, "amg", "amg");
}

static void
test_exhaustive_mgr_parser(void)
{
   MGR_args args;
   hypredrv_MGRSetDefaultArgs(&args);

   YAMLnode *mgr = hypredrv_YAMLnodeCreate("mgr", "", 0);

   /* Scalars (exercise YAML_NODE_VALIDATE path using MGRGetValidKeys/Values) */
   add_child(mgr, "max_iter", "2", 1);
   add_child(mgr, "num_levels", "2", 1);
   add_child(mgr, "relax_type", pick_value(hypredrv_MGRGetValidValues("relax_type")), 1);
   add_child(mgr, "print_level", "0", 1);

   /* levels (exercise custom parsing + lvl bounds checks + nested validation) */
   YAMLnode *levels = add_child(mgr, "level", "", 1);

   /* level 0 */
   YAMLnode *lvl0 = add_child(levels, "0", "", 2);
   add_child(lvl0, "f_dofs", "[0]", 3);
   add_child(lvl0, "prolongation_type", pick_value(hypredrv_MGRlvlGetValidValues("prolongation_type")), 3);
   add_child(lvl0, "restriction_type", pick_value(hypredrv_MGRlvlGetValidValues("restriction_type")), 3);
   add_child(lvl0, "coarse_level_type", pick_value(hypredrv_MGRlvlGetValidValues("coarse_level_type")), 3);
   YAMLnode *f0 = add_child(lvl0, "f_relaxation", "", 3);
   YAMLnode *f0_gmres = add_child(f0, "gmres", "", 4);
   add_child(f0_gmres, "max_iter", "2", 5);
   YAMLnode *f0_prec = add_child(f0_gmres, "preconditioner", "", 5);
   YAMLnode *f0_amg = add_child(f0_prec, "amg", "", 6);
   add_child(f0_amg, "max_iter", "1", 7);

   YAMLnode *g0 = add_child(lvl0, "g_relaxation", "", 3);
   add_child(g0, "num_sweeps", "2", 4);
   YAMLnode *g0_ilu = add_child(g0, "ilu", "", 4);
   add_child(g0_ilu, "type", "bj-ilut", 5);
   add_child(g0_ilu, "fill_level", "1", 5);

   /* level 1 */
   YAMLnode *lvl1 = add_child(levels, "1", "", 2);
   add_child(lvl1, "f_dofs", "[1]", 3);
   add_child(lvl1, "f_relaxation", "none", 3); /* triggers num_sweeps special-case logic */
   YAMLnode *g1 = add_child(lvl1, "g_relaxation", "", 3);
   YAMLnode *g1_gmres = add_child(g1, "gmres", "", 4);
   add_child(g1_gmres, "max_iter", "3", 5);

   /* invalid level index to hit out-of-range branch */
   YAMLnode *lvl_bad = add_child(levels, "99", "", 2);

   /* coarsest solver */
   YAMLnode *cls = add_child(mgr, "coarsest_level", "", 1);
   YAMLnode *cls_ilu = add_child(cls, "ilu", "", 2);
   add_child(cls_ilu, "type", "bj-iluk", 3);

   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRSetArgsFromYAML(&args, mgr);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_TRUE(args.level[0].f_relaxation.use_krylov);
   ASSERT_NOT_NULL(args.level[0].f_relaxation.krylov);
   ASSERT_EQ(args.level[0].f_relaxation.krylov->solver_method, SOLVER_GMRES);
   ASSERT_EQ(args.level[0].f_relaxation.krylov->solver.gmres.max_iter, 2);
   ASSERT_TRUE(args.level[0].f_relaxation.krylov->has_precon);
   ASSERT_EQ(args.level[0].f_relaxation.krylov->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args.level[0].f_relaxation.krylov->precon.amg.max_iter, 1);
   ASSERT_TRUE(args.level[1].g_relaxation.use_krylov);
   ASSERT_NOT_NULL(args.level[1].g_relaxation.krylov);
   ASSERT_EQ(args.level[1].g_relaxation.krylov->solver_method, SOLVER_GMRES);
   ASSERT_EQ(args.level[1].g_relaxation.krylov->solver.gmres.max_iter, 3);

   ASSERT_EQ(lvl_bad->valid, YAML_NODE_INVALID_KEY);

   /* Exercise hypredrv_MGRConvertArgInt table conversion paths (HANDLE_MGR_LEVEL_ATTRIBUTE macro) */
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "f_relaxation:type"));
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "f_relaxation:num_sweeps"));
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "g_relaxation:type"));
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "g_relaxation:num_sweeps"));
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "prolongation_type"));
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "restriction_type"));
   ASSERT_NOT_NULL(hypredrv_MGRConvertArgInt(&args, "coarse_level_type"));
   ASSERT_NULL(hypredrv_MGRConvertArgInt(&args, "unknown:name"));

   hypredrv_YAMLnodeDestroy(mgr);
   hypredrv_MGRDestroyNestedSolverArgs(&args);
}

static void
test_mgr_nested_krylov_rejects_mgr_precon(void)
{
   MGR_args args;
   hypredrv_MGRSetDefaultArgs(&args);

   YAMLnode *mgr = hypredrv_YAMLnodeCreate("mgr", "", 0);
   YAMLnode *levels = add_child(mgr, "level", "", 1);
   YAMLnode *lvl0 = add_child(levels, "0", "", 2);
   add_child(lvl0, "f_dofs", "[0]", 3);

   YAMLnode *f0 = add_child(lvl0, "f_relaxation", "", 3);
   YAMLnode *gmres = add_child(f0, "gmres", "", 4);
   YAMLnode *prec = add_child(gmres, "preconditioner", "", 5);
   YAMLnode *mgr_prec = add_child(prec, "mgr", "", 6);

   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRSetArgsFromYAML(&args, mgr);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(mgr_prec->valid, YAML_NODE_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLnodeDestroy(mgr);
   hypredrv_MGRDestroyNestedSolverArgs(&args);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_exhaustive_solver_parsers);
   RUN_TEST(test_exhaustive_ilu_fsai_parsers);
   RUN_TEST(test_exhaustive_amg_parser);
   RUN_TEST(test_exhaustive_mgr_parser);
   RUN_TEST(test_mgr_nested_krylov_rejects_mgr_precon);

   MPI_Finalize();
   return 0;
}
