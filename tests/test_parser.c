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
#include "nested_krylov.h"
#include "pcg.h"
#include "test_helpers.h"
#include "yaml.h"

/* Internal/generated functions not in public headers */
void           AMGSetArgsFromYAML(void *, YAMLnode *);
void           AMGintSetArgsFromYAML(void *, YAMLnode *);
void           AMGaggSetArgsFromYAML(void *, YAMLnode *);
void           AMGcsnSetArgsFromYAML(void *, YAMLnode *);
void           AMGrlxSetArgsFromYAML(void *, YAMLnode *);
void           AMGsmtSetArgsFromYAML(void *, YAMLnode *);
StrArray       AMGGetValidKeys(void);
StrIntMapArray AMGGetValidValues(const char *);
StrArray       AMGintGetValidKeys(void);
StrIntMapArray AMGintGetValidValues(const char *);
StrArray       AMGaggGetValidKeys(void);
StrIntMapArray AMGaggGetValidValues(const char *);
StrArray       AMGcsnGetValidKeys(void);
StrIntMapArray AMGcsnGetValidValues(const char *);
StrArray       AMGrlxGetValidKeys(void);
StrIntMapArray AMGrlxGetValidValues(const char *);
StrArray       AMGsmtGetValidKeys(void);
StrIntMapArray AMGsmtGetValidValues(const char *);
void           FSAISetArgsFromYAML(void *, YAMLnode *);
StrArray       FSAIGetValidKeys(void);
StrIntMapArray FSAIGetValidValues(const char *);
void           ILUSetArgsFromYAML(void *, YAMLnode *);
StrArray       ILUGetValidKeys(void);
StrIntMapArray ILUGetValidValues(const char *);
void           ILUSetDefaultArgs(ILU_args *);
void           FSAISetDefaultArgs(FSAI_args *);
void           PCGSetDefaultArgs(PCG_args *);
void           PCGSetArgsFromYAML(void *, YAMLnode *);
StrArray       PCGGetValidKeys(void);
StrIntMapArray PCGGetValidValues(const char *);
void           GMRESSetDefaultArgs(GMRES_args *);
void           GMRESSetArgsFromYAML(void *, YAMLnode *);
StrArray       GMRESGetValidKeys(void);
StrIntMapArray GMRESGetValidValues(const char *);
void           FGMRESSetDefaultArgs(FGMRES_args *);
void           FGMRESSetArgsFromYAML(void *, YAMLnode *);
StrArray       FGMRESGetValidKeys(void);
StrIntMapArray FGMRESGetValidValues(const char *);
void           BiCGSTABSetDefaultArgs(BiCGSTAB_args *);
void           BiCGSTABSetArgsFromYAML(void *, YAMLnode *);
StrArray       BiCGSTABGetValidKeys(void);
StrIntMapArray BiCGSTABGetValidValues(const char *);
void           ChebySetArgsFromYAML(void *, YAMLnode *);
StrArray       ChebyGetValidKeys(void);
StrIntMapArray ChebyGetValidValues(const char *);

/* MGR (custom YAML parser + generated sub-components) */
void           MGRSetDefaultArgs(MGR_args *);
void           MGRSetArgsFromYAML(void *, YAMLnode *);
StrIntMapArray MGRGetValidValues(const char *);
StrIntMapArray MGRlvlGetValidValues(const char *);
HYPRE_Int     *MGRConvertArgInt(MGR_args *, const char *);

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = YAMLnodeCreate(key, val, level);
   YAMLnodeAddChild(parent, child);
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
   StrArray keys = ChebyGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = ChebyGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_fsai(YAMLnode *parent, int level)
{
   StrArray keys = FSAIGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = FSAIGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_ilu(YAMLnode *parent, int level)
{
   StrArray keys = ILUGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = ILUGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgint(YAMLnode *parent, int level)
{
   StrArray keys = AMGintGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = AMGintGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgagg(YAMLnode *parent, int level)
{
   StrArray keys = AMGaggGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = AMGaggGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgcsn(YAMLnode *parent, int level)
{
   StrArray keys = AMGcsnGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = AMGcsnGetValidValues(key);
      const char    *val = pick_value(map);
      add_child(parent, key, val, level);
   }
}

static void
populate_amgrlx(YAMLnode *parent, int level)
{
   StrArray keys = AMGrlxGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = AMGrlxGetValidValues(key);

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
   StrArray keys = AMGsmtGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = AMGsmtGetValidValues(key);

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
   StrArray keys = AMGGetValidKeys();
   for (size_t i = 0; i < keys.size; i++)
   {
      const char    *key = keys.data[i];
      StrIntMapArray map = AMGGetValidValues(key);

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
   YAMLnode *parent = YAMLnodeCreate("root", "", level - 1);
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
   ErrorCodeResetAll();
   set_args(args, parent);
   ASSERT_FALSE(ErrorCodeActive());
   YAMLnodeDestroy(parent);
}

static void
exercise_component_flat(void (*set_args)(void *, YAMLnode *), void *args, const char *key,
                        const char *val)
{
   /* Intentionally build a *flat* YAML node (no children) to hit the
    * flat-value parsing branch in macro-generated SetArgsFromYAML helpers. */
   YAMLnode *parent = YAMLnodeCreate(key, val, 0);
   ErrorCodeResetAll();
   set_args(args, parent);
   YAMLnodeDestroy(parent);
}

static void
test_exhaustive_solver_parsers(void)
{
   PCG_args pcg;
   GMRES_args gmres;
   FGMRES_args fgmres;
   BiCGSTAB_args bicg;
   Cheby_args cheby;

   PCGSetDefaultArgs(&pcg);
   GMRESSetDefaultArgs(&gmres);
   FGMRESSetDefaultArgs(&fgmres);
   BiCGSTABSetDefaultArgs(&bicg);
   ChebySetDefaultArgs(&cheby);

   exercise_solver_component(PCGSetArgsFromYAML, PCGGetValidKeys, PCGGetValidValues, &pcg);
   exercise_solver_component(GMRESSetArgsFromYAML, GMRESGetValidKeys, GMRESGetValidValues,
                             &gmres);
   exercise_solver_component(FGMRESSetArgsFromYAML, FGMRESGetValidKeys, FGMRESGetValidValues,
                             &fgmres);
   exercise_solver_component(BiCGSTABSetArgsFromYAML, BiCGSTABGetValidKeys,
                             BiCGSTABGetValidValues, &bicg);
   exercise_solver_component(ChebySetArgsFromYAML, ChebyGetValidKeys, ChebyGetValidValues,
                             &cheby);

   /* Also exercise the flat-value SetArgsFromYAML branch (no children) */
   exercise_component_flat(PCGSetArgsFromYAML, &pcg, "pcg", "pcg");
   exercise_component_flat(GMRESSetArgsFromYAML, &gmres, "gmres", "gmres");
   exercise_component_flat(FGMRESSetArgsFromYAML, &fgmres, "fgmres", "fgmres");
   exercise_component_flat(BiCGSTABSetArgsFromYAML, &bicg, "bicgstab", "bicgstab");
   exercise_component_flat(ChebySetArgsFromYAML, &cheby, "cheby", "cheby");
}

static void
test_exhaustive_ilu_fsai_parsers(void)
{
   ILU_args  ilu;
   FSAI_args fsai;
   ILUSetDefaultArgs(&ilu);
   FSAISetDefaultArgs(&fsai);

   YAMLnode *ilu_parent = build_scalar_children(ILUGetValidKeys(), ILUGetValidValues, 1);
   ErrorCodeResetAll();
   ILUSetArgsFromYAML(&ilu, ilu_parent);
   ASSERT_FALSE(ErrorCodeActive());
   YAMLnodeDestroy(ilu_parent);

   YAMLnode *fsai_parent = build_scalar_children(FSAIGetValidKeys(), FSAIGetValidValues, 1);
   ErrorCodeResetAll();
   FSAISetArgsFromYAML(&fsai, fsai_parent);
   ASSERT_FALSE(ErrorCodeActive());
   YAMLnodeDestroy(fsai_parent);

   exercise_component_flat(ILUSetArgsFromYAML, &ilu, "ilu", "ilu");
   exercise_component_flat(FSAISetArgsFromYAML, &fsai, "fsai", "fsai");
}

static void
test_exhaustive_amg_parser(void)
{
   AMG_args args;
   AMGSetDefaultArgs(&args);

   YAMLnode *root = YAMLnodeCreate("amg", "", 0);
   populate_amg(root, 1);

   ErrorCodeResetAll();
   AMGSetArgsFromYAML(&args, root);
   ASSERT_FALSE(ErrorCodeActive());

   YAMLnodeDestroy(root);

   /* Flat-value branch (no children) */
   exercise_component_flat((void (*)(void *, YAMLnode *))AMGSetArgsFromYAML, &args, "amg", "amg");
}

static void
test_exhaustive_mgr_parser(void)
{
   MGR_args args;
   MGRSetDefaultArgs(&args);

   YAMLnode *mgr = YAMLnodeCreate("mgr", "", 0);

   /* Scalars (exercise YAML_NODE_VALIDATE path using MGRGetValidKeys/Values) */
   add_child(mgr, "max_iter", "2", 1);
   add_child(mgr, "num_levels", "2", 1);
   add_child(mgr, "relax_type", pick_value(MGRGetValidValues("relax_type")), 1);
   add_child(mgr, "print_level", "0", 1);

   /* levels (exercise custom parsing + lvl bounds checks + nested validation) */
   YAMLnode *levels = add_child(mgr, "level", "", 1);

   /* level 0 */
   YAMLnode *lvl0 = add_child(levels, "0", "", 2);
   add_child(lvl0, "f_dofs", "[0]", 3);
   add_child(lvl0, "prolongation_type", pick_value(MGRlvlGetValidValues("prolongation_type")), 3);
   add_child(lvl0, "restriction_type", pick_value(MGRlvlGetValidValues("restriction_type")), 3);
   add_child(lvl0, "coarse_level_type", pick_value(MGRlvlGetValidValues("coarse_level_type")), 3);
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

   ErrorCodeResetAll();
   MGRSetArgsFromYAML(&args, mgr);
   ASSERT_FALSE(ErrorCodeActive());

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

   /* Exercise MGRConvertArgInt table conversion paths (HANDLE_MGR_LEVEL_ATTRIBUTE macro) */
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "f_relaxation:type"));
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "f_relaxation:num_sweeps"));
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "g_relaxation:type"));
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "g_relaxation:num_sweeps"));
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "prolongation_type"));
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "restriction_type"));
   ASSERT_NOT_NULL(MGRConvertArgInt(&args, "coarse_level_type"));
   ASSERT_NULL(MGRConvertArgInt(&args, "unknown:name"));

   YAMLnodeDestroy(mgr);
   MGRDestroyNestedKrylovArgs(&args);
}

static void
test_mgr_nested_krylov_rejects_mgr_precon(void)
{
   MGR_args args;
   MGRSetDefaultArgs(&args);

   YAMLnode *mgr = YAMLnodeCreate("mgr", "", 0);
   YAMLnode *levels = add_child(mgr, "level", "", 1);
   YAMLnode *lvl0 = add_child(levels, "0", "", 2);
   add_child(lvl0, "f_dofs", "[0]", 3);

   YAMLnode *f0 = add_child(lvl0, "f_relaxation", "", 3);
   YAMLnode *gmres = add_child(f0, "gmres", "", 4);
   YAMLnode *prec = add_child(gmres, "preconditioner", "", 5);
   YAMLnode *mgr_prec = add_child(prec, "mgr", "", 6);

   ErrorCodeResetAll();
   MGRSetArgsFromYAML(&args, mgr);
   ASSERT_TRUE(ErrorCodeActive());
   ASSERT_EQ(mgr_prec->valid, YAML_NODE_INVALID_VAL);

   ErrorCodeResetAll();
   YAMLnodeDestroy(mgr);
   MGRDestroyNestedKrylovArgs(&args);
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
