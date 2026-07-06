/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/help.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "internal/compatibility.h"
#include "internal/containers.h"

typedef StrArray (*HelpGetKeysFunc)(void);
typedef StrIntMapArray (*HelpGetValuesFunc)(const char *);

typedef struct HelpNode_struct HelpNode;

typedef struct HelpChild_struct
{
   const char     *name;
   const HelpNode *node;
} HelpChild;

typedef struct HelpKeyDoc_struct
{
   const char *key;
   const char *doc;
} HelpKeyDoc;

struct HelpNode_struct
{
   const char        *name;
   const char        *title;
   const char        *value_hint;
   HelpGetKeysFunc    get_keys;
   HelpGetValuesFunc  get_values;
   const HelpChild   *children;
   size_t             num_children;
   int                allow_numeric_index;
};

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

/* Generated/internal schema getters used as read-only metadata. */
StrArray       hypredrv_GeneralGetValidKeys(void);
StrIntMapArray hypredrv_GeneralGetValidValues(const char *);
StrArray       hypredrv_LinearSystemGetValidKeys(void);
StrIntMapArray hypredrv_LinearSystemGetValidValues(const char *);
StrArray       hypredrv_EigSpecGetValidKeys(void);
StrIntMapArray hypredrv_EigSpecGetValidValues(const char *);
StrArray       hypredrv_ScalingGetValidKeys(void);
StrIntMapArray hypredrv_ScalingGetValidValues(const char *);
StrArray       hypredrv_SolverGetValidKeys(void);
StrIntMapArray hypredrv_SolverGetValidValues(const char *);
StrIntMapArray hypredrv_SolverGetValidTypeIntMap(void);
StrArray       hypredrv_PCGGetValidKeys(void);
StrIntMapArray hypredrv_PCGGetValidValues(const char *);
StrArray       hypredrv_GMRESGetValidKeys(void);
StrIntMapArray hypredrv_GMRESGetValidValues(const char *);
StrArray       hypredrv_FGMRESGetValidKeys(void);
StrIntMapArray hypredrv_FGMRESGetValidValues(const char *);
StrArray       hypredrv_BiCGSTABGetValidKeys(void);
StrIntMapArray hypredrv_BiCGSTABGetValidValues(const char *);
StrArray       hypredrv_PreconGetValidKeys(void);
StrIntMapArray hypredrv_PreconGetValidValues(const char *);
StrIntMapArray hypredrv_PreconGetValidTypeIntMap(void);
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
StrArray       hypredrv_ChebyGetValidKeys(void);
StrIntMapArray hypredrv_ChebyGetValidValues(const char *);
StrArray       hypredrv_ILUGetValidKeys(void);
StrIntMapArray hypredrv_ILUGetValidValues(const char *);
StrArray       hypredrv_FSAIGetValidKeys(void);
StrIntMapArray hypredrv_FSAIGetValidValues(const char *);
StrArray       hypredrv_AMSGetValidKeys(void);
StrIntMapArray hypredrv_AMSGetValidValues(const char *);
StrArray       hypredrv_ADSGetValidKeys(void);
StrIntMapArray hypredrv_ADSGetValidValues(const char *);
StrArray       hypredrv_MGRGetValidKeys(void);
StrIntMapArray hypredrv_MGRGetValidValues(const char *);
StrArray       hypredrv_MGRlvlGetValidKeys(void);
StrIntMapArray hypredrv_MGRlvlGetValidValues(const char *);
StrArray       hypredrv_MGRclsGetValidKeys(void);
StrIntMapArray hypredrv_MGRclsGetValidValues(const char *);
StrArray       hypredrv_MGRfrlxGetValidKeys(void);
StrIntMapArray hypredrv_MGRfrlxGetValidValues(const char *);
StrArray       hypredrv_MGRgrlxGetValidKeys(void);
StrIntMapArray hypredrv_MGRgrlxGetValidValues(const char *);
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
StrArray       hypredrv_SchwarzGetValidKeys(void);
StrIntMapArray hypredrv_SchwarzGetValidValues(const char *);
#endif

static StrIntMapArray
HelpVoidValues(const char *key)
{
   (void)key;
   return STR_INT_MAP_ARRAY_VOID();
}

static StrIntMapArray
HelpSolverTypeValues(const char *key)
{
   if (!key || !strcmp(key, "type"))
   {
      return hypredrv_SolverGetValidTypeIntMap();
   }
   return STR_INT_MAP_ARRAY_VOID();
}

static StrIntMapArray
HelpPreconTypeValues(const char *key)
{
   if (!key || !strcmp(key, "type"))
   {
      return hypredrv_PreconGetValidTypeIntMap();
   }
   return STR_INT_MAP_ARRAY_VOID();
}

static StrIntMapArray
HelpMGRCycleValues(const char *key)
{
   if (!strcmp(key, "cycle"))
   {
      static StrIntMap map[] = {
         {"v", 1},      {"w", 2},      {"v(1,0)", 1}, {"v(0,1)", 1},
         {"v(1,1)", 1}, {"w(1,0)", 2}, {"w(0,1)", 2}, {"w(1,1)", 2},
         {"1", 1},      {"2", 2},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return hypredrv_MGRGetValidValues(key);
}

static StrIntMapArray
HelpPrintSystemValues(const char *key)
{
   if (!strcmp(key, "enabled") || !strcmp(key, "overwrite"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {
         {"all", 0},             {"every_n_systems", 1}, {"every_n_timesteps", 2},
         {"ids", 3},             {"ranges", 4},          {"iterations_over", 5},
         {"setup_time_over", 6}, {"solve_time_over", 7}, {"selectors", 8},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "stage"))
   {
      static StrIntMap map[] = {
         {"build", 1}, {"setup", 2}, {"apply", 4}, {"all", 7},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "artifacts"))
   {
      static StrIntMap map[] = {
         {"matrix", 1}, {"precmat", 2}, {"rhs", 4},     {"x0", 8},
         {"xref", 16},  {"solution", 32}, {"dofmap", 64}, {"metadata", 128},
         {"all", 255},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "basis"))
   {
      static StrIntMap map[] = {
         {"linear_system", 0}, {"timestep", 1},   {"level", 2},
         {"iterations", 3},   {"setup_time", 4}, {"solve_time", 5},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return STR_INT_MAP_ARRAY_VOID();
}

static StrIntMapArray
HelpPreconReuseValues(const char *key)
{
   if (!strcmp(key, "enabled") || !strcmp(key, "per_timestep") ||
       !strcmp(key, "rebuild_on_new_timestep") ||
       !strcmp(key, "rebuild_on_solver_failure"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   if (!strcmp(key, "type") || !strcmp(key, "policy"))
   {
      static StrIntMap map[] = {
         {"static", 0}, {"adaptive", 1}, {"always", 0},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "metric"))
   {
      static StrIntMap map[] = {
         {"iterations", 0}, {"solve_time", 1}, {"setup_time", 2},
         {"total_time", 3}, {"solve_overhead_vs_setup", 4},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "direction"))
   {
      static StrIntMap map[] = {
         {"above", 0}, {"below", 1},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "baseline"))
   {
      static StrIntMap map[] = {
         {"rebuild", 0}, {"window_mean", 1},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "source"))
   {
      static StrIntMap map[] = {
         {"linear_solves", 0}, {"active_level", 1}, {"completed_level", 2},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "reduction"))
   {
      static StrIntMap map[] = {
         {"none", 0}, {"mean", 1}, {"sum", 2},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return STR_INT_MAP_ARRAY_VOID();
}

/* The "kind" selector key is shared by the mean and transform score sections but
 * maps to different enums, so each gets its own value getter that resolves "kind"
 * and defers every other key to the common reuse getter above. */
static StrIntMapArray
HelpReuseMeanValues(const char *key)
{
   if (key && !strcmp(key, "kind"))
   {
      static StrIntMap map[] = {
         {"arithmetic", 0}, {"geometric", 1}, {"harmonic", 2}, {"rms", 3},
         {"min", 4},        {"max", 5},       {"power", 6},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return HelpPreconReuseValues(key);
}

static StrIntMapArray
HelpReuseTransformValues(const char *key)
{
   if (key && !strcmp(key, "kind"))
   {
      static StrIntMap map[] = {
         {"raw", 0},               {"delta_from_baseline", 1},
         {"ratio_to_baseline", 2}, {"relative_increase", 3},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return HelpPreconReuseValues(key);
}

static StrArray HelpKeys(const char **keys, size_t size)
{
   return (StrArray){.data = keys, .size = size};
}

static StrArray HelpRootKeys(void)
{
   static const char *keys[] = {"general", "linear_system", "solver",
                                "preconditioner"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPrintSystemKeys(void)
{
   static const char *keys[] = {
      "enabled", "type", "stage", "artifacts", "output_dir", "overwrite",
      "every",   "ids",  "ranges", "threshold", "selectors",
   };
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPrintSystemSelectorKeys(void)
{
   static const char *keys[] = {"basis", "level", "every", "ids", "ranges",
                                "threshold"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseKeys(void)
{
   static const char *keys[] = {
      "enabled",           "frequency",             "linear_system_ids",
      "linear_solver_ids", "per_timestep",          "type",
      "policy",            "guards",                "adaptive",
   };
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseGuardsKeys(void)
{
   static const char *keys[] = {
      "min_reuse_solves",        "max_reuse_solves",
      "min_history_points",      "bad_decisions_to_rebuild",
      "max_iteration_ratio",     "max_solve_time_ratio",
      "rebuild_on_new_timestep", "rebuild_on_solver_failure",
      "rebuild_on_new_level",
   };
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseAdaptiveKeys(void)
{
   static const char *keys[] = {"rebuild_threshold", "positive_floor", "components"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseScoreKeys(void)
{
   static const char *keys[] = {"name", "enabled", "metric", "weight", "direction",
                                "target", "scale", "mean", "transform", "history"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseMeanKeys(void)
{
   static const char *keys[] = {"kind", "power"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseTransformKeys(void)
{
   static const char *keys[] = {"kind", "baseline", "amortization_window"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static StrArray HelpPreconReuseHistoryKeys(void)
{
   static const char *keys[] = {"source", "level", "max_points", "reduction"};
   return HelpKeys(keys, ARRAY_SIZE(keys));
}

static const HelpNode NodeRoot;
static const HelpNode NodeGeneral;
static const HelpNode NodeLinearSystem;
static const HelpNode NodePrintSystem;
static const HelpNode NodePrintSystemSelector;
static const HelpNode NodeEigSpec;
static const HelpNode NodeScaling;
static const HelpNode NodeSolver;
static const HelpNode NodePCG;
static const HelpNode NodeGMRES;
static const HelpNode NodeFGMRES;
static const HelpNode NodeBiCGSTAB;
static const HelpNode NodePrecon;
static const HelpNode NodeAMG;
static const HelpNode NodeAMGInterpolation;
static const HelpNode NodeAMGAggressive;
static const HelpNode NodeAMGCoarsening;
static const HelpNode NodeAMGRelaxation;
static const HelpNode NodeAMGSmoother;
static const HelpNode NodeCheby;
static const HelpNode NodeILU;
static const HelpNode NodeFSAI;
static const HelpNode NodeAMS;
static const HelpNode NodeADS;
static const HelpNode NodeMGR;
static const HelpNode NodeMGRLevel;
static const HelpNode NodeMGRCoarsest;
static const HelpNode NodeMGRFRelax;
static const HelpNode NodeMGRGRelax;
static const HelpNode NodeReuse;
static const HelpNode NodeReuseGuards;
static const HelpNode NodeReuseAdaptive;
static const HelpNode NodeReuseScore;
static const HelpNode NodeReuseMean;
static const HelpNode NodeReuseTransform;
static const HelpNode NodeReuseHistory;
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
static const HelpNode NodeSchwarz;
#endif

static const HelpChild RootChildren[] = {
   {"general", &NodeGeneral},
   {"linear_system", &NodeLinearSystem},
   {"solver", &NodeSolver},
   {"preconditioner", &NodePrecon},
};

static const HelpChild LinearSystemChildren[] = {
   {"print_system", &NodePrintSystem},
   {"eigspec", &NodeEigSpec},
   {"scaling", &NodeScaling},
};

static const HelpChild PrintSystemChildren[] = {
   {"selectors", &NodePrintSystemSelector},
};

static const HelpChild SolverChildren[] = {
   {"pcg", &NodePCG},
   {"gmres", &NodeGMRES},
   {"fgmres", &NodeFGMRES},
   {"bicgstab", &NodeBiCGSTAB},
};

static const HelpChild NestedKrylovPCGChildren[] = {
   {"preconditioner", &NodePrecon},
};

static const HelpChild NestedKrylovGMRESChildren[] = {
   {"preconditioner", &NodePrecon},
};

static const HelpChild NestedKrylovFGMRESChildren[] = {
   {"preconditioner", &NodePrecon},
};

static const HelpChild NestedKrylovBiCGSTABChildren[] = {
   {"preconditioner", &NodePrecon},
};

static const HelpChild PreconChildren[] = {
   {"amg", &NodeAMG},
   {"mgr", &NodeMGR},
   {"ilu", &NodeILU},
   {"fsai", &NodeFSAI},
   {"ams", &NodeAMS},
   {"ads", &NodeADS},
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   {"schwarz", &NodeSchwarz},
#endif
   {"reuse", &NodeReuse},
};

static const HelpChild AMGChildren[] = {
   {"interpolation", &NodeAMGInterpolation},
   {"aggressive", &NodeAMGAggressive},
   {"coarsening", &NodeAMGCoarsening},
   {"relaxation", &NodeAMGRelaxation},
   {"smoother", &NodeAMGSmoother},
};

static const HelpChild AMGRelaxationChildren[] = {
   {"chebyshev", &NodeCheby},
};

static const HelpChild AMGSmootherChildren[] = {
   {"fsai", &NodeFSAI},
   {"ilu", &NodeILU},
};

static const HelpChild MGRChildren[] = {
   {"level", &NodeMGRLevel},
   {"coarsest_level", &NodeMGRCoarsest},
};

static const HelpChild MGRLevelChildren[] = {
   {"f_relaxation", &NodeMGRFRelax},
   {"g_relaxation", &NodeMGRGRelax},
};

static const HelpChild MGRCoarsestChildren[] = {
   {"amg", &NodeAMG},
   {"ilu", &NodeILU},
   {"fsai", &NodeFSAI},
   {"pcg", &NodePCG},
   {"gmres", &NodeGMRES},
   {"fgmres", &NodeFGMRES},
   {"bicgstab", &NodeBiCGSTAB},
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   {"schwarz", &NodeSchwarz},
#endif
   {"reuse", &NodeReuse},
};

static const HelpChild MGRFRelaxChildren[] = {
   {"mgr", &NodeMGR},
   {"amg", &NodeAMG},
   {"ilu", &NodeILU},
   {"fsai", &NodeFSAI},
   {"pcg", &NodePCG},
   {"gmres", &NodeGMRES},
   {"fgmres", &NodeFGMRES},
   {"bicgstab", &NodeBiCGSTAB},
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   {"schwarz", &NodeSchwarz},
#endif
   {"reuse", &NodeReuse},
};

static const HelpChild MGRGRelaxChildren[] = {
   {"amg", &NodeAMG},
   {"ilu", &NodeILU},
   {"fsai", &NodeFSAI},
   {"pcg", &NodePCG},
   {"gmres", &NodeGMRES},
   {"fgmres", &NodeFGMRES},
   {"bicgstab", &NodeBiCGSTAB},
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   {"schwarz", &NodeSchwarz},
#endif
   {"reuse", &NodeReuse},
};

static const HelpChild ReuseChildren[] = {
   {"guards", &NodeReuseGuards},
   {"adaptive", &NodeReuseAdaptive},
};

static const HelpChild ReuseAdaptiveChildren[] = {
   {"components", &NodeReuseScore},
};

static const HelpChild ReuseScoreChildren[] = {
   {"mean", &NodeReuseMean},
   {"transform", &NodeReuseTransform},
   {"history", &NodeReuseHistory},
};

static const HelpNode NodeRoot = {
   "root", "YAML input", "<section>", HelpRootKeys, HelpVoidValues, RootChildren,
   ARRAY_SIZE(RootChildren), 0};
static const HelpNode NodeGeneral = {
   "general", "Global configuration settings", "<value>", hypredrv_GeneralGetValidKeys,
   hypredrv_GeneralGetValidValues, NULL, 0, 0};
static const HelpNode NodeLinearSystem = {
   "linear_system", "Linear system settings", "<value>",
   hypredrv_LinearSystemGetValidKeys, hypredrv_LinearSystemGetValidValues,
   LinearSystemChildren, ARRAY_SIZE(LinearSystemChildren), 0};
static const HelpNode NodePrintSystem = {
   "print_system", "linear_system.print_system section", "<value>",
   HelpPrintSystemKeys, HelpPrintSystemValues, PrintSystemChildren,
   ARRAY_SIZE(PrintSystemChildren), 0};
static const HelpNode NodePrintSystemSelector = {
   "selectors", "linear_system.print_system.selectors item", "<value>",
   HelpPrintSystemSelectorKeys, HelpPrintSystemValues, NULL, 0, 1};
static const HelpNode NodeEigSpec = {
   "eigspec", "linear_system.eigspec section", "<value>", hypredrv_EigSpecGetValidKeys,
   hypredrv_EigSpecGetValidValues, NULL, 0, 0};
static const HelpNode NodeScaling = {
   "scaling", "solver.scaling section", "<value>", hypredrv_ScalingGetValidKeys,
   hypredrv_ScalingGetValidValues, NULL, 0, 0};
static const HelpNode NodeSolver = {
   "solver", "Linear solver settings", "<solver-type>", hypredrv_SolverGetValidKeys,
   HelpSolverTypeValues, SolverChildren, ARRAY_SIZE(SolverChildren), 0};
static const HelpNode NodePCG = {
   "pcg", "PCG solver", "<value>", hypredrv_PCGGetValidKeys, hypredrv_PCGGetValidValues,
   NestedKrylovPCGChildren, ARRAY_SIZE(NestedKrylovPCGChildren), 0};
static const HelpNode NodeGMRES = {
   "gmres", "GMRES solver", "<value>", hypredrv_GMRESGetValidKeys,
   hypredrv_GMRESGetValidValues, NestedKrylovGMRESChildren,
   ARRAY_SIZE(NestedKrylovGMRESChildren), 0};
static const HelpNode NodeFGMRES = {
   "fgmres", "FGMRES solver", "<value>", hypredrv_FGMRESGetValidKeys,
   hypredrv_FGMRESGetValidValues, NestedKrylovFGMRESChildren,
   ARRAY_SIZE(NestedKrylovFGMRESChildren), 0};
static const HelpNode NodeBiCGSTAB = {
   "bicgstab", "BiCGSTAB solver", "<value>", hypredrv_BiCGSTABGetValidKeys,
   hypredrv_BiCGSTABGetValidValues, NestedKrylovBiCGSTABChildren,
   ARRAY_SIZE(NestedKrylovBiCGSTABChildren), 0};
static const HelpNode NodePrecon = {
   "preconditioner", "Preconditioner settings", "<preconditioner-type>",
   hypredrv_PreconGetValidKeys, HelpPreconTypeValues, PreconChildren,
   ARRAY_SIZE(PreconChildren), 0};
static const HelpNode NodeAMG = {
   "amg", "AMG preconditioner", "<value>", hypredrv_AMGGetValidKeys,
   hypredrv_AMGGetValidValues, AMGChildren, ARRAY_SIZE(AMGChildren), 0};
static const HelpNode NodeAMGInterpolation = {
   "interpolation", "AMG interpolation section", "<value>",
   hypredrv_AMGintGetValidKeys, hypredrv_AMGintGetValidValues, NULL, 0, 0};
static const HelpNode NodeAMGAggressive = {
   "aggressive", "AMG aggressive section", "<value>", hypredrv_AMGaggGetValidKeys,
   hypredrv_AMGaggGetValidValues, NULL, 0, 0};
static const HelpNode NodeAMGCoarsening = {
   "coarsening", "AMG coarsening section", "<value>", hypredrv_AMGcsnGetValidKeys,
   hypredrv_AMGcsnGetValidValues, NULL, 0, 0};
static const HelpNode NodeAMGRelaxation = {
   "relaxation", "AMG relaxation section", "<value>", hypredrv_AMGrlxGetValidKeys,
   hypredrv_AMGrlxGetValidValues, AMGRelaxationChildren,
   ARRAY_SIZE(AMGRelaxationChildren), 0};
static const HelpNode NodeAMGSmoother = {
   "smoother", "AMG smoother section", "<value>", hypredrv_AMGsmtGetValidKeys,
   hypredrv_AMGsmtGetValidValues, AMGSmootherChildren, ARRAY_SIZE(AMGSmootherChildren),
   0};
static const HelpNode NodeCheby = {
   "chebyshev", "Chebyshev smoother section", "<value>", hypredrv_ChebyGetValidKeys,
   hypredrv_ChebyGetValidValues, NULL, 0, 0};
static const HelpNode NodeILU = {
   "ilu", "ILU preconditioner", "<value>", hypredrv_ILUGetValidKeys,
   hypredrv_ILUGetValidValues, NULL, 0, 0};
static const HelpNode NodeFSAI = {
   "fsai", "FSAI preconditioner", "<value>", hypredrv_FSAIGetValidKeys,
   hypredrv_FSAIGetValidValues, NULL, 0, 0};
static const HelpNode NodeAMS = {
   "ams", "AMS preconditioner", "<value>", hypredrv_AMSGetValidKeys,
   hypredrv_AMSGetValidValues, NULL, 0, 0};
static const HelpNode NodeADS = {
   "ads", "ADS preconditioner", "<value>", hypredrv_ADSGetValidKeys,
   hypredrv_ADSGetValidValues, NULL, 0, 0};
static const HelpNode NodeMGR = {
   "mgr", "MGR preconditioner", "<value>", hypredrv_MGRGetValidKeys,
   HelpMGRCycleValues, MGRChildren, ARRAY_SIZE(MGRChildren), 0};
static const HelpNode NodeMGRLevel = {
   "level", "MGR level item", "<value>", hypredrv_MGRlvlGetValidKeys,
   hypredrv_MGRlvlGetValidValues, MGRLevelChildren, ARRAY_SIZE(MGRLevelChildren), 1};
static const HelpNode NodeMGRCoarsest = {
   "coarsest_level", "MGR coarsest_level section", "<value>",
   hypredrv_MGRclsGetValidKeys, hypredrv_MGRclsGetValidValues, MGRCoarsestChildren,
   ARRAY_SIZE(MGRCoarsestChildren), 0};
static const HelpNode NodeMGRFRelax = {
   "f_relaxation", "MGR f_relaxation section", "<value>",
   hypredrv_MGRfrlxGetValidKeys, hypredrv_MGRfrlxGetValidValues, MGRFRelaxChildren,
   ARRAY_SIZE(MGRFRelaxChildren), 0};
static const HelpNode NodeMGRGRelax = {
   "g_relaxation", "MGR g_relaxation section", "<value>",
   hypredrv_MGRgrlxGetValidKeys, hypredrv_MGRgrlxGetValidValues, MGRGRelaxChildren,
   ARRAY_SIZE(MGRGRelaxChildren), 0};
static const HelpNode NodeReuse = {
   "reuse", "preconditioner reuse section", "<value>", HelpPreconReuseKeys,
   HelpPreconReuseValues, ReuseChildren, ARRAY_SIZE(ReuseChildren), 0};
static const HelpNode NodeReuseGuards = {
   "guards", "preconditioner reuse guards section", "<value>",
   HelpPreconReuseGuardsKeys, HelpPreconReuseValues, NULL, 0, 0};
static const HelpNode NodeReuseAdaptive = {
   "adaptive", "adaptive preconditioner reuse section", "<value>",
   HelpPreconReuseAdaptiveKeys, HelpPreconReuseValues, ReuseAdaptiveChildren,
   ARRAY_SIZE(ReuseAdaptiveChildren), 0};
static const HelpNode NodeReuseScore = {
   "score", "adaptive reuse score component", "<value>", HelpPreconReuseScoreKeys,
   HelpPreconReuseValues, ReuseScoreChildren, ARRAY_SIZE(ReuseScoreChildren), 1};
static const HelpNode NodeReuseMean = {
   "mean", "adaptive reuse score mean section", "<value>", HelpPreconReuseMeanKeys,
   HelpReuseMeanValues, NULL, 0, 0};
static const HelpNode NodeReuseTransform = {
   "transform", "adaptive reuse score transform section", "<value>",
   HelpPreconReuseTransformKeys, HelpReuseTransformValues, NULL, 0, 0};
static const HelpNode NodeReuseHistory = {
   "history", "adaptive reuse score history section", "<value>",
   HelpPreconReuseHistoryKeys, HelpPreconReuseValues, NULL, 0, 0};
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
static const HelpNode NodeSchwarz = {
   "schwarz", "Schwarz preconditioner", "<value>", hypredrv_SchwarzGetValidKeys,
   hypredrv_SchwarzGetValidValues, NULL, 0, 0};
#endif

static const HelpKeyDoc CommonKeyDocs[] = {
   {"enabled", "Enable or disable this feature"},
   {"type", "Select the algorithm or mode for this section"},
   {"policy", "Select the policy used by this section"},
   {"name", "Human-readable label used in printed output"},
   {"min_iter", "Minimum number of iterations before convergence checks may stop"},
   {"max_iter", "Maximum number of iterations"},
   {"print_level", "Verbosity level for solver/preconditioner output"},
   {"logging", "Enable hypre logging for this object"},
   {"relative_tol", "Relative convergence tolerance"},
   {"absolute_tol", "Absolute convergence tolerance"},
   {"tolerance", "Convergence tolerance"},
   {"conv_fac_tol", "Convergence-factor tolerance"},
   {"residual_tol", "Residual-norm tolerance"},
   {"stop_crit", "Select the stopping criterion"},
   {"rel_change", "Require relative solution-change convergence check"},
   {"two_norm", "Use the two-norm in PCG convergence checks"},
   {"recompute_res", "Recompute the residual during PCG iterations"},
   {"skip_real_res_check", "Skip GMRES real residual verification"},
   {"krylov_dim", "Restart dimension for Krylov subspace methods"},
   {"num_levels", "Number of multilevel hierarchy levels"},
   {"max_levels", "Maximum number of multilevel hierarchy levels"},
   {"num_sweeps", "Number of relaxation or smoothing sweeps"},
   {"num_functions", "Number of nodal/vector components"},
   {"filter_functions", "Filter interpolation by function labels"},
   {"max_nnz_row", "Maximum nonzeros allowed per interpolation row"},
   {"trunc_factor", "Drop tolerance used to truncate interpolation entries"},
   {"weight", "Relaxation weight"},
   {"outer_weight", "Outer relaxation weight"},
   {"order", "Polynomial or relaxation order"},
   {"variant", "Select the algorithm variant"},
   {"overlap", "Overlap depth for Schwarz subdomains"},
   {"domain_type", "Schwarz subdomain construction mode"},
   {"use_nonsymm", "Use the nonsymmetric Schwarz setup"},
   {"local_solver_type", "Local solver used inside Schwarz"},
   {"iluk_level_of_fill", "ILU(k) fill level"},
   {"ilut_max_nnz_row", "Maximum nonzeros per row for ILUT"},
   {"ilut_droptol", "Drop tolerance for ILUT"},
};

static const HelpKeyDoc GeneralKeyDocs[] = {
   {"statistics_filename", "File receiving statistics output; empty means stdout"},
   {"warmup", "Run an untimed warmup solve before measured solves"},
   {"statistics", "Statistics reporting level"},
   {"print_config_params", "Print parsed YAML configuration after parsing"},
   {"use_millisec", "Print timing values in milliseconds"},
   {"exec_policy", "Execution policy for hypre operations"},
   {"use_vendor_spgemm", "Use vendor sparse matrix-matrix kernels when available"},
   {"use_vendor_spmv", "Use vendor sparse matrix-vector kernels when available"},
   {"num_repetitions", "Number of repeated solve runs"},
   {"dev_pool_size", "Initial device memory pool size in GB"},
   {"uvm_pool_size", "Initial unified-memory pool size in GB"},
   {"host_pool_size", "Initial host memory pool size in GB"},
   {"pinned_pool_size", "Initial pinned-host memory pool size in GB"},
};

static const HelpKeyDoc LinearSystemKeyDocs[] = {
   {"dirname", "Directory containing linear-system input files"},
   {"sequence_filename", "Compressed linear-system sequence file"},
   {"matrix_filename", "Matrix file for a single linear system"},
   {"matrix_basename", "Matrix filename prefix for a sequence"},
   {"precmat_filename", "Matrix file used to build the preconditioner"},
   {"precmat_basename", "Preconditioner-matrix filename prefix for a sequence"},
   {"rhs_filename", "Right-hand-side vector file"},
   {"rhs_basename", "Right-hand-side filename prefix for a sequence"},
   {"x0_filename", "Initial-guess vector file"},
   {"sol_filename", "Output solution vector file"},
   {"xref_filename", "Reference-solution vector file"},
   {"xref_basename", "Reference-solution filename prefix for a sequence"},
   {"timestep_filename", "File mapping linear systems to timesteps"},
   {"dofmap_filename", "Degree-of-freedom map file"},
   {"dofmap_basename", "Degree-of-freedom map filename prefix for a sequence"},
   {"digits_suffix", "Number of digits used for sequence filename suffixes"},
   {"init_suffix", "First suffix in a linear-system sequence"},
   {"last_suffix", "Last suffix in a linear-system sequence"},
   {"set_suffix", "Explicit list of suffixes to solve"},
   {"init_guess_mode", "How to initialize the solution vector"},
   {"rhs_mode", "How to provide or generate the right-hand side"},
   {"print_system", "Nested scheduled linear-system dump options"},
   {"eigspec", "Nested eigenspectrum diagnostic options"},
   {"dof_labels", "Symbolic labels for degree-of-freedom ids"},
};

static const HelpKeyDoc PrintSystemKeyDocs[] = {
   {"stage", "Lifecycle stage that may trigger a dump"},
   {"artifacts", "Linear-system artifacts to write"},
   {"output_dir", "Directory for dumped artifacts"},
   {"overwrite", "Allow reuse of existing dump output paths"},
   {"every", "Dump every N matching systems or timesteps"},
   {"ids", "Explicit zero-based ids to dump"},
   {"ranges", "Inclusive id ranges to dump"},
   {"threshold", "Metric threshold for metric-triggered dumps"},
   {"selectors", "Nested list of selector blocks"},
   {"basis", "Index or metric basis used by a selector"},
   {"level", "Multilevel solver level id used by a selector"},
};

static const HelpKeyDoc EigSpecKeyDocs[] = {
   {"enable", "Enable eigenspectrum diagnostics"},
   {"vectors", "Write eigenvectors in addition to eigenvalues"},
   {"hermitian", "Treat the operator as Hermitian/symmetric"},
   {"preconditioned", "Analyze the preconditioned operator M^{-1}A"},
   {"output_prefix", "Prefix for eigenspectrum output files"},
};

static const HelpKeyDoc ScalingKeyDocs[] = {
   {"custom_values", "Custom per-DOF scaling values"},
};

static const HelpKeyDoc SolverSectionKeyDocs[] = {
   {"pcg", "Preconditioned Conjugate Gradient solver block"},
   {"gmres", "Generalized Minimal RESidual solver block"},
   {"fgmres", "Flexible Generalized Minimal RESidual solver block"},
   {"bicgstab", "Bi-Conjugate Gradient Stabilized solver block"},
};

static const HelpKeyDoc PreconSectionKeyDocs[] = {
   {"amg", "BoomerAMG preconditioner block"},
   {"mgr", "Multigrid Reduction preconditioner block"},
   {"ilu", "Incomplete LU preconditioner block"},
   {"fsai", "Factorized Sparse Approximate Inverse preconditioner block"},
   {"ams", "Auxiliary-space Maxwell solver preconditioner block"},
   {"ads", "Auxiliary-space divergence solver preconditioner block"},
   {"schwarz", "Schwarz preconditioner block"},
   {"reuse", "Preconditioner reuse policy block"},
};

static const HelpKeyDoc AMGKeyDocs[] = {
   {"interpolation", "Nested AMG interpolation options"},
   {"aggressive", "Nested aggressive coarsening options"},
   {"coarsening", "Nested AMG coarsening options"},
   {"relaxation", "Nested AMG relaxation options"},
   {"smoother", "Nested AMG complex smoother options"},
};

static const HelpKeyDoc AMGInterpolationKeyDocs[] = {
   {"prolongation_type", "AMG interpolation/prolongation type"},
   {"restriction_type", "AMG restriction type"},
};

static const HelpKeyDoc AMGAggressiveKeyDocs[] = {
   {"num_paths", "Number of paths used by aggressive coarsening"},
   {"prolongation_type", "Interpolation type used after aggressive coarsening"},
   {"P12_max_elements", "Maximum elements in the second-stage interpolation"},
   {"P12_trunc_factor", "Truncation factor for second-stage interpolation"},
};

static const HelpKeyDoc AMGCoarseningKeyDocs[] = {
   {"rap2", "Use two-stage RAP construction"},
   {"mod_rap2", "Use modified two-stage RAP construction"},
   {"keep_transpose", "Keep transpose data structures when available"},
   {"nodal", "Use nodal coarsening mode"},
   {"seq_amg_th", "Threshold for using sequential AMG on coarse levels"},
   {"min_coarse_size", "Minimum coarse-grid size"},
   {"max_coarse_size", "Maximum coarse-grid size"},
   {"max_row_sum", "Maximum row-sum threshold"},
   {"strong_th", "Strength-of-connection threshold"},
};

static const HelpKeyDoc AMGRelaxationKeyDocs[] = {
   {"down_type", "Relaxation type on the down cycle"},
   {"up_type", "Relaxation type on the up cycle"},
   {"coarse_type", "Relaxation type on the coarsest AMG level"},
   {"down_sweeps", "Relaxation sweeps on the down cycle"},
   {"up_sweeps", "Relaxation sweeps on the up cycle"},
   {"coarse_sweeps", "Relaxation sweeps on the coarsest AMG level"},
   {"chebyshev", "Nested Chebyshev smoother options"},
};

static const HelpKeyDoc AMGSmootherKeyDocs[] = {
   {"fsai", "Nested FSAI smoother options"},
   {"ilu", "Nested ILU smoother options"},
};

static const HelpKeyDoc ChebyKeyDocs[] = {
   {"eig_est", "Eigenvalue estimate strategy for Chebyshev smoothing"},
   {"scale", "Scale Chebyshev coefficients"},
   {"fraction", "Spectral fraction used by Chebyshev smoothing"},
};

static const HelpKeyDoc ILUKeyDocs[] = {
   {"fill_level", "Level of fill for ILU(k)"},
   {"reordering", "Local reordering scheme (none or RCM)"},
   {"tri_solve", "Triangular solve mode (direct or iterative)"},
   {"lower_jac_iters", "Jacobi iterations for lower triangular solve"},
   {"upper_jac_iters", "Jacobi iterations for upper triangular solve"},
   {"max_row_nnz", "Maximum nonzeros per row for ILUT"},
   {"schur_max_iter", "Maximum iterations for Schur complement solve"},
   {"droptol", "Drop tolerance for ILUT factorization"},
   {"nsh_droptol", "Drop tolerance for NSH Schur inverse"},
};

static const HelpKeyDoc FSAIKeyDocs[] = {
   {"algo_type", "FSAI algorithm type"},
   {"ls_type", "Local dense-solve backend for FSAI (GPU only)"},
   {"max_steps", "Maximum FSAI setup steps"},
   {"max_step_size", "Maximum entries added per FSAI setup step"},
   {"max_nnz_row", "Maximum off-diagonal entries per row of G"},
   {"num_levels", "Levels for the static FSAI candidate pattern"},
   {"eig_max_iters", "Iterations for maximum-eigenvalue estimation"},
   {"threshold", "Threshold for the static FSAI candidate pattern"},
   {"kap_tolerance", "Kaporin tolerance for FSAI filtering"},
};

static const HelpKeyDoc AMSKeyDocs[] = {
   {"dimension", "Spatial dimension of the problem (2 or 3)"},
   {"cycle_type", "AMS multiplicative/additive cycle type"},
   {"proj_freq", "Subspace-projection frequency"},
   {"relax_type", "Smoother relaxation type on the original matrix"},
   {"relax_times", "Number of smoother sweeps on the original matrix"},
   {"relax_weight", "Smoother relaxation weight on the original matrix"},
   {"omega", "Smoother SSOR/omega parameter on the original matrix"},
   {"alpha_coarsen_type", "Coarsening type for the alpha-space AMG solver"},
   {"alpha_agg_levels", "Aggressive-coarsening levels for the alpha-space AMG solver"},
   {"alpha_relax_type", "Relaxation type for the alpha-space AMG solver"},
   {"alpha_strength_threshold", "Strength threshold for the alpha-space AMG solver"},
   {"alpha_interp_type", "Interpolation type for the alpha-space AMG solver"},
   {"alpha_Pmax", "Interpolation stencil cap for the alpha-space AMG solver"},
   {"alpha_coarse_relax_type", "Coarsest-level relaxation for the alpha-space AMG solver"},
   {"beta_coarsen_type", "Coarsening type for the beta-space AMG solver"},
   {"beta_agg_levels", "Aggressive-coarsening levels for the beta-space AMG solver"},
   {"beta_relax_type", "Relaxation type for the beta-space AMG solver"},
   {"beta_strength_threshold", "Strength threshold for the beta-space AMG solver"},
   {"beta_interp_type", "Interpolation type for the beta-space AMG solver"},
   {"beta_Pmax", "Interpolation stencil cap for the beta-space AMG solver"},
   {"beta_coarse_relax_type", "Coarsest-level relaxation for the beta-space AMG solver"},
};

static const HelpKeyDoc ADSKeyDocs[] = {
   {"cycle_type", "ADS auxiliary-space solver cycle type"},
   {"relax_type", "Relaxation type for the ADS smoother"},
   {"relax_times", "Number of ADS smoother sweeps"},
   {"relax_weight", "Relaxation weight for the ADS smoother"},
   {"omega", "SSOR over-relaxation factor for the ADS smoother"},
   {"cheby_order", "Chebyshev smoother order for ADS"},
   {"cheby_fraction", "Eigenvalue fraction for ADS Chebyshev smoothing"},
   {"ams_cycle_type", "Cycle type for the ADS auxiliary AMS solver"},
   {"ams_coarsen_type", "Coarsening type for the ADS auxiliary AMS solver"},
   {"ams_agg_levels", "Aggressive coarsening levels for the ADS AMS solver"},
   {"ams_relax_type", "Relaxation type for the ADS auxiliary AMS solver"},
   {"ams_strength_threshold", "Strength threshold for the ADS auxiliary AMS solver"},
   {"ams_interp_type", "Interpolation type for the ADS auxiliary AMS solver"},
   {"ams_Pmax", "Max interpolation entries per row for the ADS AMS solver"},
   {"amg_coarsen_type", "Coarsening type for the ADS auxiliary AMG solver"},
   {"amg_agg_levels", "Aggressive coarsening levels for the ADS AMG solver"},
   {"amg_relax_type", "Relaxation type for the ADS auxiliary AMG solver"},
   {"amg_strength_threshold", "Strength threshold for the ADS auxiliary AMG solver"},
   {"amg_interp_type", "Interpolation type for the ADS auxiliary AMG solver"},
   {"amg_Pmax", "Max interpolation entries per row for the ADS AMG solver"},
};

static const HelpKeyDoc MGRKeyDocs[] = {
   {"non_c_to_f", "Treat non-C-points as F-points"},
   {"pmax", "Maximum interpolation stencil size"},
   {"relax_type", "Default MGR relaxation type"},
   {"nonglk_max_elmts", "Maximum elements for non-Galerkin dropping"},
   {"coarse_th", "Coarse-grid strength threshold"},
   {"coarsest_level", "Nested coarsest-level solver options"},
   {"cycle", "MGR cycle shape and smoothing position"},
   {"level", "Nested per-level MGR options"},
};

static const HelpKeyDoc MGRLevelKeyDocs[] = {
   {"f_dofs", "Fine-grid degrees of freedom for this MGR level"},
   {"prolongation_type", "Interpolation type for this MGR level"},
   {"restriction_type", "Restriction type for this MGR level"},
   {"coarse_level_type", "Coarse-level construction type"},
   {"f_relaxation", "Nested F-relaxation options"},
   {"g_relaxation", "Nested global relaxation options"},
};

static const HelpKeyDoc MGRComponentKeyDocs[] = {
   {"amg", "Nested AMG component solver options"},
   {"ilu", "Nested ILU component solver options"},
   {"fsai", "Nested FSAI component solver options"},
   {"mgr", "Nested MGR component solver options"},
   {"pcg", "Nested PCG component solver options"},
   {"gmres", "Nested GMRES component solver options"},
   {"fgmres", "Nested FGMRES component solver options"},
   {"bicgstab", "Nested BiCGSTAB component solver options"},
   {"schwarz", "Nested Schwarz component solver options"},
   {"reuse", "Nested component reuse policy"},
};

static const HelpKeyDoc ReuseKeyDocs[] = {
   {"frequency", "Number of linear solves between preconditioner rebuilds"},
   {"linear_system_ids", "Linear-system ids where reuse is allowed"},
   {"linear_solver_ids", "Deprecated alias for linear_system_ids"},
   {"per_timestep", "Apply reuse decisions per timestep"},
   {"guards", "Nested rebuild guard options"},
   {"adaptive", "Nested adaptive reuse policy options"},
   {"min_reuse_solves", "Minimum solves to reuse before rebuilding"},
   {"max_reuse_solves", "Maximum solves to reuse before rebuilding"},
   {"min_history_points", "Minimum history samples before adaptive decisions"},
   {"bad_decisions_to_rebuild", "Bad adaptive decisions allowed before rebuild"},
   {"max_iteration_ratio", "Rebuild when iterations exceed this ratio"},
   {"max_solve_time_ratio", "Rebuild when solve time exceeds this ratio"},
   {"rebuild_on_new_timestep", "Force rebuild when timestep changes"},
   {"rebuild_on_solver_failure", "Force rebuild after solver failure"},
   {"rebuild_on_new_level", "MGR levels that force component rebuilds"},
   {"rebuild_threshold", "Adaptive score threshold for rebuilding"},
   {"positive_floor", "Small positive floor used by adaptive ratios"},
   {"components", "Nested list of adaptive score components"},
   {"kind", "Algorithm kind selected for this section"},
   {"metric", "Runtime metric used by the adaptive score"},
   {"direction", "Whether the metric is considered bad above or below target"},
   {"target", "Target metric value for the adaptive score"},
   {"scale", "Scale applied to adaptive score contribution"},
   {"mean", "Nested mean operator for adaptive history"},
   {"transform", "Nested transform for metric normalization"},
   {"history", "Nested history source and reduction options"},
   {"power", "Power used by power mean"},
   {"baseline", "Baseline used by metric transform"},
   {"amortization_window", "Setup-cost amortization window"},
   {"source", "Runtime sample source for history"},
   {"level", "Solver level id sampled by the history source"},
   {"max_points", "Maximum history samples to use"},
   {"reduction", "Reduction applied to selected history samples"},
};

static const char *
HelpFindDocIn(const HelpKeyDoc *docs, size_t num_docs, const char *key)
{
   for (size_t i = 0; i < num_docs; i++)
   {
      if (!strcmp(docs[i].key, key))
      {
         return docs[i].doc;
      }
   }
   return NULL;
}

static const char *
HelpFindDoc(const HelpNode *node, const char *key)
{
   const char *doc = NULL;

#define HELP_TRY_DOCS(_node, _docs)                                                \
   do                                                                              \
   {                                                                               \
      if ((node) == &(_node) &&                                                     \
          (doc = HelpFindDocIn((_docs), ARRAY_SIZE(_docs), (key))) != NULL)        \
      {                                                                            \
         return doc;                                                               \
      }                                                                            \
   } while (0)

   HELP_TRY_DOCS(NodeGeneral, GeneralKeyDocs);
   HELP_TRY_DOCS(NodeLinearSystem, LinearSystemKeyDocs);
   HELP_TRY_DOCS(NodePrintSystem, PrintSystemKeyDocs);
   HELP_TRY_DOCS(NodePrintSystemSelector, PrintSystemKeyDocs);
   HELP_TRY_DOCS(NodeEigSpec, EigSpecKeyDocs);
   HELP_TRY_DOCS(NodeScaling, ScalingKeyDocs);
   HELP_TRY_DOCS(NodeSolver, SolverSectionKeyDocs);
   HELP_TRY_DOCS(NodePrecon, PreconSectionKeyDocs);
   HELP_TRY_DOCS(NodeAMG, AMGKeyDocs);
   HELP_TRY_DOCS(NodeAMGInterpolation, AMGInterpolationKeyDocs);
   HELP_TRY_DOCS(NodeAMGAggressive, AMGAggressiveKeyDocs);
   HELP_TRY_DOCS(NodeAMGCoarsening, AMGCoarseningKeyDocs);
   HELP_TRY_DOCS(NodeAMGRelaxation, AMGRelaxationKeyDocs);
   HELP_TRY_DOCS(NodeAMGSmoother, AMGSmootherKeyDocs);
   HELP_TRY_DOCS(NodeCheby, ChebyKeyDocs);
   HELP_TRY_DOCS(NodeILU, ILUKeyDocs);
   HELP_TRY_DOCS(NodeFSAI, FSAIKeyDocs);
   HELP_TRY_DOCS(NodeAMS, AMSKeyDocs);
   HELP_TRY_DOCS(NodeADS, ADSKeyDocs);
   HELP_TRY_DOCS(NodeMGR, MGRKeyDocs);
   HELP_TRY_DOCS(NodeMGRLevel, MGRLevelKeyDocs);
   HELP_TRY_DOCS(NodeMGRCoarsest, MGRComponentKeyDocs);
   HELP_TRY_DOCS(NodeMGRFRelax, MGRComponentKeyDocs);
   HELP_TRY_DOCS(NodeMGRGRelax, MGRComponentKeyDocs);
   HELP_TRY_DOCS(NodeReuse, ReuseKeyDocs);
   HELP_TRY_DOCS(NodeReuseGuards, ReuseKeyDocs);
   HELP_TRY_DOCS(NodeReuseAdaptive, ReuseKeyDocs);
   HELP_TRY_DOCS(NodeReuseScore, ReuseKeyDocs);
   HELP_TRY_DOCS(NodeReuseMean, ReuseKeyDocs);
   HELP_TRY_DOCS(NodeReuseTransform, ReuseKeyDocs);
   HELP_TRY_DOCS(NodeReuseHistory, ReuseKeyDocs);

#undef HELP_TRY_DOCS

   doc = HelpFindDocIn(CommonKeyDocs, ARRAY_SIZE(CommonKeyDocs), key);
   if (doc)
   {
      return doc;
   }

   for (size_t i = 0; node && i < node->num_children; i++)
   {
      if (!strcmp(node->children[i].name, key))
      {
         return node->children[i].node->title;
      }
   }

   return "Configuration value";
}

static int
HelpIsNumeric(const char *s)
{
   if (!s || *s == '\0')
   {
      return 0;
   }
   for (const unsigned char *p = (const unsigned char *)s; *p; p++)
   {
      if (!isdigit(*p))
      {
         return 0;
      }
   }
   return 1;
}

static const HelpNode *
HelpFindChild(const HelpNode *node, const char *name)
{
   if (!node || !name)
   {
      return NULL;
   }
   if (node->allow_numeric_index && HelpIsNumeric(name))
   {
      return node;
   }
   for (size_t i = 0; i < node->num_children; i++)
   {
      if (!strcmp(node->children[i].name, name))
      {
         return node->children[i].node;
      }
   }
   return NULL;
}

static void
HelpPrintMap(FILE *out, StrIntMapArray map, int indent)
{
   for (size_t i = 0; i < map.size; i++)
   {
      int already_printed = 0;
      int num             = map.data[i].num;
      int aliases         = 0;

      for (size_t j = 0; j < i; j++)
      {
         if (map.data[j].num == num)
         {
            already_printed = 1;
            break;
         }
      }
      if (already_printed)
      {
         continue;
      }

      fprintf(out, "%*s", indent, "");
      for (size_t j = i; j < map.size; j++)
      {
         if (map.data[j].num == num)
         {
            if (!map.data[j].str || map.data[j].str[0] == '\0')
            {
               continue;
            }
            fprintf(out, "%s%s", aliases ? "/" : "", map.data[j].str);
            aliases++;
         }
      }
      if (aliases == 0)
      {
         fprintf(out, "<empty>");
      }
      fprintf(out, " (%d)\n", num);
   }
}

static void
HelpPrintKeys(FILE *out, const HelpNode *node)
{
   if (!node || !node->get_keys)
   {
      return;
   }

   StrArray keys = node->get_keys();
   if (keys.size == 0)
   {
      return;
   }

   size_t max_key_len  = 0;
   size_t max_hint_len = 0;
   for (size_t i = 0; i < keys.size; i++)
   {
      const char     *key = keys.data[i];
      StrIntMapArray vals =
         node->get_values ? node->get_values(key) : STR_INT_MAP_ARRAY_VOID();
      const char *hint = vals.size ? "<one of>" : node->value_hint;
      size_t      key_len = strlen(key);
      size_t      hint_len = strlen(hint);

      if (key_len > max_key_len)
      {
         max_key_len = key_len;
      }
      if (hint_len > max_hint_len)
      {
         max_hint_len = hint_len;
      }
   }

   fprintf(out, "\nValid keys:\n");
   for (size_t i = 0; i < keys.size; i++)
   {
      const char     *key = keys.data[i];
      StrIntMapArray vals =
         node->get_values ? node->get_values(key) : STR_INT_MAP_ARRAY_VOID();
      const char *hint = vals.size ? "<one of>" : node->value_hint;
      fprintf(out, "  %-*s  %-*s  %s\n", (int)max_key_len, key, (int)max_hint_len,
              hint, HelpFindDoc(node, key));
      if (vals.size)
      {
         HelpPrintMap(out, vals, 6);
      }
   }
}

static void
HelpPrintChildren(FILE *out, const HelpNode *node)
{
   if (!node || node->num_children == 0)
   {
      return;
   }

   fprintf(out, "\nNested topics:\n");
   for (size_t i = 0; i < node->num_children; i++)
   {
      fprintf(out, "  %s\n", node->children[i].name);
   }
   if (node->allow_numeric_index)
   {
      fprintf(out, "  <index>\n");
   }
}

static int
HelpNodeHasKey(const HelpNode *node, const char *key)
{
   if (!node || !node->get_keys || !key)
   {
      return 0;
   }
   StrArray keys = node->get_keys();
   for (size_t i = 0; i < keys.size; i++)
   {
      if (!strcmp(keys.data[i], key))
      {
         return 1;
      }
   }
   return 0;
}

static void
HelpPrintKey(FILE *out, const HelpNode *node, const char *path, const char *key)
{
   StrIntMapArray vals =
      (node && node->get_values) ? node->get_values(key) : STR_INT_MAP_ARRAY_VOID();

   fprintf(out, "Help for %s:%s\n", path && *path ? path : node->name, key);
   fprintf(out, "%s  %s  %s\n", key, vals.size ? "<one of>" : node->value_hint,
           HelpFindDoc(node, key));
   if (vals.size)
   {
      fprintf(out, "\nAccepted values:\n");
      HelpPrintMap(out, vals, 2);
   }
}

static int
HelpPrintUnknown(FILE *out, const char *topic, const HelpNode *nearest)
{
   fprintf(out, "Unknown help topic '%s'.\n", topic ? topic : "");
   if (nearest)
   {
      HelpPrintChildren(out, nearest);
   }
   return 1;
}

static int
HelpResolve(const char *topic, const HelpNode **node_out, const HelpNode **nearest_out,
            char *normalized, size_t normalized_size, const char **key_out)
{
   const HelpNode *node = &NodeRoot;
   const HelpNode *nearest = node;
   static char     key_buf[128];

   if (normalized && normalized_size > 0)
   {
      normalized[0] = '\0';
   }
   if (key_out)
   {
      *key_out = NULL;
      key_buf[0] = '\0';
   }
   if (!topic || *topic == '\0')
   {
      *node_out = node;
      if (nearest_out)
      {
         *nearest_out = nearest;
      }
      return 1;
   }

   char buf[512];
   snprintf(buf, sizeof(buf), "%s", topic);
   for (char *p = buf; *p; p++)
   {
      if (*p == '.')
      {
         *p = ':';
      }
      else
      {
         *p = (char)tolower((unsigned char)*p);
      }
   }

   char *save = NULL;
   char *tokens[64];
   int   ntokens = 0;
   for (char *tok = strtok_r(buf, ":", &save); tok && ntokens < 64;
        tok = strtok_r(NULL, ":", &save))
   {
      tokens[ntokens++] = tok;
   }

   for (int i = 0; i < ntokens; i++)
   {
      const char     *tok   = tokens[i];
      const HelpNode *child = HelpFindChild(node, tok);
      if (!child)
      {
         if (i == ntokens - 1 && HelpNodeHasKey(node, tok))
         {
            snprintf(key_buf, sizeof(key_buf), "%s", tok);
            if (key_out)
            {
               *key_out = key_buf;
            }
            *node_out = node;
            if (nearest_out)
            {
               *nearest_out = nearest;
            }
            return 1;
         }
         if (nearest_out)
         {
            *nearest_out = nearest;
         }
         return 0;
      }
      node = child;
      nearest = node;
      if (normalized && normalized_size > 0)
      {
         if (normalized[0] != '\0')
         {
            strncat(normalized, ":", normalized_size - strlen(normalized) - 1);
         }
         strncat(normalized, tok, normalized_size - strlen(normalized) - 1);
      }
   }

   *node_out = node;
   if (nearest_out)
   {
      *nearest_out = nearest;
   }
   return 1;
}

int
hypredrv_HelpRequested(int argc, char **argv, char *topic, size_t topic_size)
{
   if (topic && topic_size > 0)
   {
      topic[0] = '\0';
   }

   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         for (int j = i + 1; j < argc && topic && topic_size > 0; j++)
         {
            if (topic[0] != '\0')
            {
               strncat(topic, ":", topic_size - strlen(topic) - 1);
            }
            strncat(topic, argv[j], topic_size - strlen(topic) - 1);
         }
         return 1;
      }
   }
   return 0;
}

int
hypredrv_HelpPrint(FILE *out, const char *argv0, const char *topic)
{
   char            normalized[512];
   const HelpNode *node    = NULL;
   const HelpNode *nearest = NULL;
   const char     *key     = NULL;

   if (!out)
   {
      out = stdout;
   }
   if (!HelpResolve(topic, &node, &nearest, normalized, sizeof(normalized), &key))
   {
      return HelpPrintUnknown(out, topic, nearest);
   }
   if (key)
   {
      HelpPrintKey(out, node, normalized, key);
      return 0;
   }

   if (node == &NodeRoot)
   {
      fprintf(out, "Usage: %s [options] <input.yml>\n", argv0 ? argv0 : "hypredrive-cli");
      fprintf(out, "\nList of valid sections for input.yml:\n");
   }
   else
   {
      fprintf(out, "Help for %s\n", normalized);
      fprintf(out, "%s\n", node->title);
   }

   /* Only sections whose identity is chosen by a "type" value (solver and
    * preconditioner) advertise that type list here; for every other node the
    * per-key listing below already covers any "type" field it may have. */
   if (node->get_values == HelpSolverTypeValues ||
       node->get_values == HelpPreconTypeValues)
   {
      StrIntMapArray vals = node->get_values("type");
      if (vals.size)
      {
         fprintf(out, "\nAccepted values:\n");
         HelpPrintMap(out, vals, 2);
      }
   }

   HelpPrintKeys(out, node);
   HelpPrintChildren(out, node);

   fprintf(out, "\nExamples (use \":\" for nested options):\n");
   fprintf(out, "  %s --help solver:gmres\n", argv0 ? argv0 : "hypredrive-cli");
   fprintf(out, "  %s --help preconditioner:mgr:level:f_relaxation\n",
           argv0 ? argv0 : "hypredrive-cli");

   return 0;
}
