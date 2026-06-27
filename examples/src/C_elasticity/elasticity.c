/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "HYPREDRV.h"
#include "HYPREDRV_utils.h"

/* HYPREDRV_HYPRE_RELEASE_NUMBER is provided by the internal hypredrive headers
   only; examples see hypre's own HYPRE_RELEASE_NUMBER through HYPREDRV.h */
#ifndef HYPREDRV_HYPRE_RELEASE_NUMBER
#ifdef HYPRE_RELEASE_NUMBER
#define HYPREDRV_HYPRE_RELEASE_NUMBER HYPRE_RELEASE_NUMBER
#else
#define HYPREDRV_HYPRE_RELEASE_NUMBER 0
#endif
#endif

#if defined(HYPRE_RELEASE_NUMBER) && HYPRE_RELEASE_NUMBER >= 21900
#define HYPREDRV_IJ_MATRIX_INIT_HOST(mat) \
   HYPRE_IJMatrixInitialize_v2((mat), HYPRE_MEMORY_HOST)
#define HYPREDRV_IJ_VECTOR_INIT_HOST(vec) \
   HYPRE_IJVectorInitialize_v2((vec), HYPRE_MEMORY_HOST)
#else
#define HYPREDRV_IJ_MATRIX_INIT_HOST(mat) HYPRE_IJMatrixInitialize((mat))
#define HYPREDRV_IJ_VECTOR_INIT_HOST(vec) HYPRE_IJVectorInitialize((vec))
#endif

/*==========================================================================
 *   3D Linear Elasticity (Q1 Hexahedra) Example Driver
 *==========================================================================
 *
 *   Grid Partitioning:
 *   ------------------
 *                     P[1]
 *                      ↑      Processor Grid: P[0] × P[1] × P[2]
 *                      |   /  Each proc owns a n[0] × n[1] × n[2] block
 *                      |  /
 *                      | /
 *         P[2] ←———————C
 *                     /
 *                    /
 *                   /
 *                  ↙ P[0]
 *
 *   Geometry and Axes:
 *   ------------------
 *   Ω = [0, Lx] × [0, Ly] × [0, Lz]
 *     - x: beam length (clamped at x = 0)
 *     - y: vertical axis (gravity and optional top traction)
 *     - z: out-of-plane/depth
 *
 *   Physics (small-strain linear elasticity):
 *   -----------------------------------------
 *     - PDE:  −∇·σ(u) = f  in Ω
 *     - Kinematics: ε(u) = sym(∇u)
 *     - Constitutive: σ = D ε with isotropic D(E, ν)
 *
 *   Boundary Conditions:
 *   --------------------
 *     - Clamp: u = (0,0,0) on x = 0 plane (all three displacement DOFs).
 *     - Optional traction: t applied on the top surface y = Ly.
 *     - Body force: f = ρ g (default g = (0, −9.81, 0)).
 *
 *   Discretization:
 *   ---------------
 *     - Q1 hexahedra, 3 DOFs per node (ux, uy, uz), interleaved per node.
 *     - Element stiffness: Ke = ∫Ωe Bᵀ D B dV; 2×2×2 Gauss quadrature.
 *     - Top traction: 2×2 Gauss on the y = Ly face.
 *     - Assembly yields a symmetric (SPD with clamp) global matrix.
 *
 *   Notes:
 *   ------
 *     - Dirichlet rows are imposed as identity with zero RHS.
 *     - Columns associated with clamped DOFs are kept zero by skipping those
 *       columns during element insertion.
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Default solver configuration for the two-material bar. The coupled u-p system
 * is symmetric indefinite, so it uses FGMRES with a 2-level MGR preconditioner:
 *   F-block  : displacements (labels 0-5), one unknown-based (num_functions = 3)
 *              BoomerAMG V-cycle (Korn-coercive, lambda-independent block).
 *   C-block  : pressure (label 6). Instead of the Galerkin RAP Schur, MGR is
 *              handed the scaled pressure-mass Schur Mhat = (1/(2mu)+1/lambda) M_p
 *              via coarse_level_type: user (HYPRE_MGRSetCoarseGridMatrixAtLevel). Mhat is
 *              spectrally equivalent to the exact pressure Schur uniformly in h and
 *              lambda (Mardal-Winther / ESW; the Bochev-Dohrmann pressure
 *              stabilization makes this hold for equal-order Q1-Q1), so the outer
 *              FGMRES count stays bounded as nu_top -> 1/2. The mass matrix is well
 *              conditioned, so a single coarse cycle suffices.
 * The driver assembles Mhat (over MGR's compressed coarse numbering) and supplies
 * it automatically for the built-in recipe; see BuildMixedTwoMaterialSystem and
 * HYPREDRV_LinearSystemSetCoarseSchur. Targets hypre >= 3.1.0 with MGR
 * coarse_grid_method 6 (user); older releases can run a user file via -i.
 *--------------------------------------------------------------------------*/
static const char *default_config_mixed =
   "solver:\n"
   "  fgmres:\n"
   "    krylov_dim: 100\n"
   "    max_iter: 300\n"
   "    print_level: 0\n"
   "    relative_tol: 1.0e-6\n"
   "preconditioner:\n"
   "  mgr:\n"
   "    max_iter: 1\n"
   "    tolerance: 0.0\n"
   "    print_level: 0\n"
   "    coarse_th: 0.0\n"
   "    level:\n"
   "      0:\n"
   "        f_dofs: [0, 1, 2, 3, 4, 5]\n"
   "        f_relaxation:\n"
   "          amg:\n"
   /* Two V-cycles of a rigid-body-aware (GM2) displacement AMG. Nodal coarsening
      (nodal_type 2) keeps the hierarchy cheap (operator complexity ~2.0) while the
      three rotational rigid-body modes -- injected automatically by MGR, restricted
      to this level's F-points -- make each cycle resolve the elasticity rotation
      modes. The result is nearly mesh-independent outer iterations at below-baseline
      cost (see reports/mixed_up_preconditioner.tex, Sec. 11.2). */
   "            max_iter: 2\n"
   "            tolerance: 0.0\n"
   "            interpolation:\n"
   "              prolongation_type: extended+i\n"
   "            coarsening:\n"
   "              type: hmis\n"
   "              strong_th: 0.25\n"
   "              num_functions: 3\n"
   "              nodal: 1\n"
   "              nodal_type: 2\n"
   "              interp_vec_variant: 2\n"
   "              smooth_interp_vecs: 0\n"
   "            relaxation:\n"
   "              down_type: forward-hl1gs\n"
   "              up_type: backward-hl1gs\n"
   "              coarse_type: ge\n"
   "        g_relaxation: none\n"
   "        restriction_type: injection\n"
   "        prolongation_type: jacobi\n"
   "        coarse_level_type: user\n"
   "    coarsest_level:\n"
   "      gmres:\n"
   "        max_iter: 1\n"
   "        print_level: 0\n"
   "        preconditioner:\n"
   "          amg:\n"
   "            max_iter: 1\n"
   "            tolerance: 0.0\n"
   "            interpolation:\n"
   "              prolongation_type: extended+i\n"
   "            coarsening:\n"
   "              type: hmis\n"
   "              strong_th: 0.3\n"
   "              num_functions: 1\n"
   "            relaxation:\n"
   "              down_type: forward-hl1gs\n"
   "              up_type: backward-hl1gs\n"
   "              coarse_type: ge\n";

/*--------------------------------------------------------------------------
 * Problem parameters struct
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int  visualize;         /* Visualize displacement in VTK format */
   HYPRE_Int  verbose;           /* Verbosity level bitset */
   HYPRE_Int  nsolve;            /* Number of solves */
   HYPRE_Int  N[3];              /* Grid dimensions (nodes) */
   HYPRE_Int  P[3];              /* Processor grid dims */
   HYPRE_Real L[3];              /* Physical dimensions (Lx, Ly, Lz) */
   HYPRE_Int  max_rows_per_call; /* Max rows per IJ add/set call */
   HYPRE_Real E;                 /* Young's modulus */
   HYPRE_Real nu;                /* Poisson ratio */
   HYPRE_Real rho;               /* Density */
   HYPRE_Real g[3];              /* Gravity vector */
   HYPRE_Int  traction_top;      /* Apply traction on y = Ly */
   HYPRE_Real traction[3];       /* Traction vector on top surface */
   char      *yaml_file;         /* YAML configuration file */
   char      *solver_preset;     /* Example solver preset selector */
   HYPRE_Int  problem;           /* 0 = single material, 1 = two-material bar */
   HYPRE_Int  discretization;    /* 0 = mixed u-p top, 1 = standard CG, 2 = B-bar (Q1-P0) */
   HYPRE_Real E_top;             /* Top-material Young's modulus (two-material) */
   HYPRE_Real nu_top;            /* Top-material Poisson ratio (two-material) */
   HYPRE_Int  prec_mass_schur;   /* mixed: provide scaled pressure-mass Schur prec matrix */
   HYPRE_Int  coarse_schur;      /* mixed: provide pressure-mass Schur as MGR coarse operator */
} ElasticParams;

/*--------------------------------------------------------------------------
 * Distributed Mesh struct (reused)
 *--------------------------------------------------------------------------*/
typedef struct
{
   MPI_Comm      cart_comm;
   HYPRE_Int     mypid;
   HYPRE_Int     gdims[3];
   HYPRE_Int     pdims[3];
   HYPRE_Int     coords[3];
   HYPRE_Int     nlocal[3];
   HYPRE_Int     nbrs[14];
   HYPRE_Int     local_size; /* local number of nodes */
   HYPRE_BigInt  ilower;     /* first local node gid (scalar) */
   HYPRE_BigInt  iupper;     /* last  local node gid (scalar) */
   HYPRE_BigInt *pstarts[3];
   HYPRE_Real    gsizes[3];
} DistMesh;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/
static inline HYPRE_BigInt grid2idx(const HYPRE_BigInt gcoords[3],
                                    const HYPRE_Int bcoords[3], const HYPRE_Int gdims[3],
                                    HYPRE_BigInt **pstarts);

int      PrintUsage(void);
int      CreateDistMesh(MPI_Comm, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int,
                        HYPRE_Int, DistMesh **);
int      DestroyDistMesh(DistMesh **);
int      ParseArguments(int, char **, ElasticParams *, int, int);
int      BuildElasticitySystem_Q1Hex(DistMesh *, ElasticParams *, HYPRE_IJMatrix *,
                                     HYPRE_IJVector *, MPI_Comm);
int      BuildMixedTwoMaterialSystem(DistMesh *, ElasticParams *, HYPRE_IJMatrix *,
                                     HYPRE_IJVector *, int **, HYPRE_Int *, HYPRE_Real **,
                                     HYPRE_IJMatrix *, HYPRE_IJMatrix *, MPI_Comm);
int      WriteVTKsolutionVector(DistMesh *, ElasticParams *, HYPRE_Real *);
int      WriteVTKMixedSolution(DistMesh *, ElasticParams *, HYPRE_Real *, MPI_Comm);
int      ComputeRigidBodyModes(DistMesh *, ElasticParams *, HYPRE_Real **);
uint32_t RegisterExamplePreconPresets(void);
int      ValidateSolverPreset(const char *, int);

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/
int
PrintUsage(void)
{
   printf("\n");
   printf("Usage: ${MPIEXEC_COMMAND} <np> ./elasticity [options]\n");
   printf("\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML configuration file for solver settings (Opt.)\n");
   printf("  -n <nx> <ny> <nz> : Global grid dimensions in nodes (30 10 10)\n");
   printf("  -P <Px> <Py> <Pz> : Processor grid dimensions (1 1 1)\n");
   printf("  -L <Lx> <Ly> <Lz> : Physical dimensions (3 1 1)\n");
   printf("  -g <gx> <gy> <gz> : Gravity vector g (0 -9.81 0)\n");
   printf("  -T <tx> <ty> <tz> : Uniform traction on top surface y=Ly (0 -100 0)\n");
   printf("  -br <n>           : Batch rows for matrix assembly (128)\n");
   printf("  -E <val>          : Young's modulus E (1.0e5)\n");
   printf("  -nu <val>         : Poisson ratio nu (0.3)\n");
   printf("  -rho <val>        : Density rho (1.0)\n");
   printf("  --solver-preset <name>\n");
   printf("                    : Solver preset selector (elasticity_3D | ");
   printf("elasticity_sdc_3D | elasticity_nodal_3D)\n");
   printf("                      (ignored when --problem two-material)\n");
   printf("  --problem <name>  : Problem configuration: single | two-material (single)\n");
   printf("                      two-material splits the bar at y=Ly/2 into a bottom\n");
   printf("                      half and a near-incompressible top half;\n");
   printf("                      requires (ny-1) even so a node layer sits at y=Ly/2\n");
   printf("  --discretization <name>\n");
   printf("                    : discretization: mixed | standard | bbar (mixed)\n");
   printf("                      mixed    = u-p (Q1-Q1) top + CG bottom (saddle point)\n");
   printf("                      standard = standard CG Q1 displacement everywhere\n");
   printf("                                 (locks near nu=0.5; pass -i amg-pcg.yml)\n");
   printf("                      bbar     = Q1-P0 mean-dilatation (B-bar) everywhere:\n");
   printf("                                 displacement-only, locking-free, SPD\n");
   printf("                                 (PCG + AMG, e.g. -i amg-pcg.yml)\n");
   printf("  --E-top <val>     : Top-material Young's modulus (defaults to -E)\n");
   printf("  --nu-top <val>    : Top-material Poisson ratio (0.4999)\n");
   printf("  -ns|--nsolve <n>  : Number of solves (5)\n");
   printf("  -vis <m>          : Visualization mode (0)\n");
   printf("                         0: none\n");
   printf("                         1: ASCII VTK\n");
   printf("                         2: binary VTK\n");
   printf("  -v|--verbose <n>  : Verbosity bitset (0)\n");
   printf("                         0x1: Library info and linear solver statistics\n");
   printf("                         0x2: System info\n");
   printf("                         0x4: Print linear system matrices\n");
   printf("  -h|--help         : Print this message\n");
   printf("\n");

   return 0;
}

/*--------------------------------------------------------------------------
 * Parse command line arguments
 *--------------------------------------------------------------------------*/
int
ParseArguments(int argc, char *argv[], ElasticParams *params, int myid, int num_procs)
{
   /* Defaults */
   params->visualize = 0;
   params->verbose   = 3;
   params->nsolve    = 5;
   for (int i = 0; i < 3; i++)
   {
      params->N[i] = 10;
      params->P[i] = 1;
      params->L[i] = 1.0;
   }
   params->N[0]              = 30;
   params->L[0]              = 3.0;
   params->max_rows_per_call = 128;
   params->E                 = 1.0e5;
   params->nu                = 0.3;
   params->rho               = 1.0;
   params->g[0]              = 0.0;
   params->g[1]              = -9.81;
   params->g[2]              = 0.0;
   params->traction_top      = 1;
   params->traction[0]       = 0.0;
   params->traction[1]       = -100.0;
   params->traction[2]       = 0.0;
   params->yaml_file         = NULL;
   params->solver_preset     = "elasticity_3D";
   params->problem           = 0;
   params->discretization    = 0; /* mixed u-p top (default for two-material) */
   params->E_top             = -1.0; /* sentinel: inherit E after parsing */
   params->nu_top            = 0.4999;
   params->prec_mass_schur   = 0;
   params->coarse_schur      = 0;

   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))
      {
         if (++i < argc) params->yaml_file = argv[i];
      }
      else if (!strcmp(argv[i], "--solver-preset"))
      {
         if (++i >= argc)
         {
            if (!myid) printf("Error: --solver-preset requires one value\n");
            return 1;
         }
         params->solver_preset = argv[i];
      }
      else if (!strcmp(argv[i], "--problem") || !strcmp(argv[i], "-prob"))
      {
         if (++i >= argc)
         {
            if (!myid) printf("Error: --problem requires one value\n");
            return 1;
         }
         if (!strcmp(argv[i], "single"))
         {
            params->problem = 0;
         }
         else if (!strcmp(argv[i], "two-material"))
         {
            params->problem = 1;
         }
         else
         {
            if (!myid)
               printf("Error: --problem must be 'single' or 'two-material'\n");
            return 1;
         }
      }
      else if (!strcmp(argv[i], "--discretization") || !strcmp(argv[i], "-disc"))
      {
         if (++i >= argc)
         {
            if (!myid) printf("Error: --discretization requires one value\n");
            return 1;
         }
         if (!strcmp(argv[i], "mixed"))
         {
            params->discretization = 0;
         }
         else if (!strcmp(argv[i], "standard"))
         {
            params->discretization = 1;
         }
         else if (!strcmp(argv[i], "bbar") || !strcmp(argv[i], "condensed") ||
                  !strcmp(argv[i], "mean-dilatation"))
         {
            params->discretization = 2;
         }
         else
         {
            if (!myid)
               printf("Error: --discretization must be 'mixed', 'standard', or "
                      "'bbar'\n");
            return 1;
         }
      }
      else if (!strcmp(argv[i], "--E-top"))
      {
         if (++i < argc) params->E_top = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "--nu-top"))
      {
         if (++i < argc) params->nu_top = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "--prec-mass-schur"))
      {
         params->prec_mass_schur = 1;
      }
      else if (!strcmp(argv[i], "--coarse-schur"))
      {
         params->coarse_schur = 1;
      }
      else if (!strcmp(argv[i], "-n"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -n requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++)
         {
            params->N[j] = atoi(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-P"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -P requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++)
         {
            params->P[j] = atoi(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-L"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -L requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++)
         {
            params->L[j] = atof(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-E"))
      {
         if (++i < argc) params->E = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-br") || !strcmp(argv[i], "--batch-rows"))
      {
         if (++i < argc) params->max_rows_per_call = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-nu"))
      {
         if (++i < argc) params->nu = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-rho"))
      {
         if (++i < argc) params->rho = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-g"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -g requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++) params->g[j] = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-T"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -T requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++) params->traction[j] = atof(argv[++i]);
         params->traction_top = 1;
      }
      else if (!strcmp(argv[i], "-ns") || !strcmp(argv[i], "--nsolve"))
      {
         if (++i < argc) params->nsolve = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-vis") || !strcmp(argv[i], "--visualize"))
      {
         if (i + 1 < argc && argv[i + 1][0] != '-')
         {
            params->visualize = atoi(argv[++i]);
         }
         else
         {
            /* No argument means ASCII */
            params->visualize = 1;
         }
      }
      else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose"))
      {
         if (++i < argc) params->verbose = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         if (!myid) PrintUsage();
         return 2;
      }
   }

   /* Checks */
   if (params->P[0] * params->P[1] * params->P[2] != num_procs)
   {
      if (!myid)
      {
         printf("Error: Number of processes (%d) must match processor grid "
                "dimensions (%d x %d x %d = %d)\n",
                num_procs, params->P[0], params->P[1], params->P[2],
                params->P[0] * params->P[1] * params->P[2]);
      }
      return 1;
   }

   for (int d = 0; d < 3; d++)
   {
      char *name[] = {"First", "Second", "Third"};
      if (params->P[d] > params->N[d])
      {
         if (!myid)
         {
            printf("Error: %s grid dimension (N = %d) must be larger than the "
                   "number of ranks (P = %d)\n",
                   name[d], params->N[d], params->P[d]);
         }
         return 1;
      }
   }

   if (params->nu <= -1.0 || params->nu >= 0.5)
   {
      if (!myid) printf("Error: Poisson ratio must be in (-1, 0.5)\n");
      return 1;
   }

   /* Resolve the top-material modulus sentinel (inherit the bottom modulus) */
   if (params->E_top < 0.0) params->E_top = params->E;

   if (params->problem)
   {
      if ((params->N[1] - 1) % 2 != 0)
      {
         if (!myid)
            printf("Error: --problem two-material requires (ny-1) even so a node "
                   "layer sits at the y=Ly/2 interface\n");
         return 1;
      }
      if (params->nu_top <= -1.0 || params->nu_top >= 0.5)
      {
         if (!myid) printf("Error: top Poisson ratio must be in (-1, 0.5)\n");
         return 1;
      }
      /* The mixed (flipped-Y) layout requires each half to reach every rank.
         The standard CG discretization uses the ordinary Cartesian partition. */
      if (params->discretization == 0 && params->P[1] > (params->N[1] - 1) / 2)
      {
         if (!myid)
            printf("Error: --problem two-material --discretization mixed requires "
                   "Py <= (ny-1)/2 = %d so each half (split at y=Ly/2) reaches every "
                   "rank (Py=%d)\n",
                   (params->N[1] - 1) / 2, params->P[1]);
         return 1;
      }
   }

   if (ValidateSolverPreset(params->solver_preset, myid))
   {
      return 1;
   }

   /* The built-in mixed recipe (used when no -i is given) drives MGR's coarse
      solve with the scaled pressure-mass Schur (coarse_level_type: user), so the
      driver must assemble and supply that operator. Enable it automatically.
      With a user-provided config, this stays opt-in via --coarse-schur. */
   if (params->problem && params->discretization == 0 && !params->yaml_file)
   {
      params->coarse_schur = 1;
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Register application-local preconditioner presets
 *--------------------------------------------------------------------------*/
uint32_t
RegisterExamplePreconPresets(void)
{
   static const char sdc_preset_yaml[]   = "amg:\n"
                                           "  coarsening:\n"
                                           "    num_functions: 3\n"
                                           "    strong_th: 0.8\n"
                                           "    filter_functions: on";
   /* Nodal coarsening + GM2 rigid-body-mode interpolation. Two settings are
      essential and were wrong before: (1) strong_th must be LOW (0.25, not the 0.8
      used by the scalar/unknown presets) -- the nodal (block-norm) strength matrix
      is far denser than the pointwise one, so 0.8 barely coarsens (first level
      18432->17220) and bloats the hierarchy to operator complexity ~6.5; (2)
      nodal_type 2 (block sum-of-abs) coarsens cleanly to complexity ~2.0. With the
      six rigid-body modes the driver supplies, GM2 (interp_vec_variant 2, the
      hypredrive default) then resolves the rotational near-null space, making this
      preset BEAT the plain unknown-based AMG in both iterations and time. */
   static const char nodal_preset_yaml[] = "amg:\n"
                                           "  coarsening:\n"
                                           "    num_functions: 3\n"
                                           "    strong_th: 0.25\n"
                                           "    nodal: 1\n"
                                           "    nodal_type: 2\n"
                                           "    interp_vec_variant: 2";

   uint32_t err = HYPREDRV_PreconPresetRegister(
      "elasticity_sdc_3D", sdc_preset_yaml, "Elasticity 3D AMG with function filtering");
   if (err)
   {
      return err;
   }

   return HYPREDRV_PreconPresetRegister("elasticity_nodal_3D", nodal_preset_yaml,
                                        "Elasticity 3D AMG with nodal coarsening");
}

/*--------------------------------------------------------------------------
 * Validate supported solver preset aliases for this example
 *--------------------------------------------------------------------------*/
int
ValidateSolverPreset(const char *preset, int myid)
{
   if (!preset)
   {
      if (!myid) printf("Error: --solver-preset cannot be empty\n");
      return 1;
   }

   if (!strcmp(preset, "elasticity_3D") || !strcmp(preset, "elasticity_sdc_3D") ||
       !strcmp(preset, "elasticity_nodal_3D"))
   {
      return 0;
   }

   if (!myid)
   {
      printf("Error: Unknown --solver-preset '%s'\n", preset);
      printf(
         "       Valid options: elasticity_3D, elasticity_sdc_3D, elasticity_nodal_3D\n");
   }
   return 1;
}

/*--------------------------------------------------------------------------
 * Compute global index from global grid coordinates
 *--------------------------------------------------------------------------*/
static inline HYPRE_BigInt
grid2idx(const HYPRE_BigInt gcoords[3], const HYPRE_Int bcoords[3],
         const HYPRE_Int gdims[3], HYPRE_BigInt **pstarts)
{
   return pstarts[2][bcoords[2]] * (HYPRE_BigInt)gdims[0] * (HYPRE_BigInt)gdims[1] +
          pstarts[1][bcoords[1]] * (HYPRE_BigInt)gdims[0] *
             (pstarts[2][bcoords[2] + 1] - pstarts[2][bcoords[2]]) +
          pstarts[0][bcoords[0]] * (pstarts[1][bcoords[1] + 1] - pstarts[1][bcoords[1]]) *
             (pstarts[2][bcoords[2] + 1] - pstarts[2][bcoords[2]]) +
          ((gcoords[2] - pstarts[2][bcoords[2]]) *
              (pstarts[1][bcoords[1] + 1] - pstarts[1][bcoords[1]]) +
           (gcoords[1] - pstarts[1][bcoords[1]])) *
             (pstarts[0][bcoords[0] + 1] - pstarts[0][bcoords[0]]) +
          (gcoords[0] - pstarts[0][bcoords[0]]);
}

/*--------------------------------------------------------------------------
 * Create mesh partition information (reused from Laplacian example)
 *--------------------------------------------------------------------------*/
int
CreateDistMesh(MPI_Comm comm, HYPRE_Int Nx, HYPRE_Int Ny, HYPRE_Int Nz, HYPRE_Int Px,
               HYPRE_Int Py, HYPRE_Int Pz, DistMesh **mesh_ptr)
{
   DistMesh *mesh = (DistMesh *)malloc(sizeof(DistMesh));
   int       myid, num_procs;
   (void)num_procs; /* Reserved for future use */

   mesh->gdims[0] = Nx;
   mesh->gdims[1] = Ny;
   mesh->gdims[2] = Nz;
   mesh->pdims[0] = Px;
   mesh->pdims[1] = Py;
   mesh->pdims[2] = Pz;

   /* Use int arrays for MPI calls to avoid type mismatch if HYPRE_Int != int */
   int mpi_dims[3]   = {(int)Px, (int)Py, (int)Pz};
   int mpi_coords[3] = {0, 0, 0};

   /* Keep cart ranks consistent with MPI_COMM_WORLD ranks (avoid reorder=1). */
   MPI_Cart_create(comm, 3, mpi_dims, (int[]){0, 0, 0}, 0, &(mesh->cart_comm));
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Cart_coords(mesh->cart_comm, myid, 3, mpi_coords);

   /* Copy to HYPRE_Int members */
   mesh->coords[0] = mpi_coords[0];
   mesh->coords[1] = mpi_coords[1];
   mesh->coords[2] = mpi_coords[2];

   mesh->mypid = (HYPRE_Int)myid;

   for (int i = 0; i < 3; i++)
   {
      HYPRE_Int size = mesh->gdims[i] / mesh->pdims[i];
      HYPRE_Int rest = mesh->gdims[i] - size * mesh->pdims[i];

      mesh->pstarts[i] = calloc((size_t)(mesh->pdims[i] + 1), sizeof(HYPRE_BigInt));
      for (int j = 0; j < mesh->pdims[i] + 1; j++)
      {
         mesh->pstarts[i][j] = (HYPRE_BigInt)(size * j + (j < rest ? j : rest));
      }
      mesh->nlocal[i] = (HYPRE_Int)(mesh->pstarts[i][mesh->coords[i] + 1] -
                                    mesh->pstarts[i][mesh->coords[i]]);
      mesh->gsizes[i] = 1.0 / (mesh->gdims[i] - 1);
   }

   HYPRE_BigInt gcoords[3] = {mesh->pstarts[0][mesh->coords[0]],
                              mesh->pstarts[1][mesh->coords[1]],
                              mesh->pstarts[2][mesh->coords[2]]};
   mesh->ilower            = grid2idx(gcoords, mesh->coords, mesh->gdims, mesh->pstarts);
   mesh->local_size        = mesh->nlocal[0] * mesh->nlocal[1] * mesh->nlocal[2];
   mesh->iupper            = mesh->ilower + mesh->local_size - 1;

   MPI_Cart_shift(mesh->cart_comm, 0, 1, &mesh->nbrs[0], &mesh->nbrs[1]);
   MPI_Cart_shift(mesh->cart_comm, 1, 1, &mesh->nbrs[2], &mesh->nbrs[3]);
   MPI_Cart_shift(mesh->cart_comm, 2, 1, &mesh->nbrs[4], &mesh->nbrs[5]);

   if (mesh->coords[0] > 0 && mesh->coords[1] > 0)
      MPI_Cart_rank(mesh->cart_comm,
                    (int[]){mesh->coords[0] - 1, mesh->coords[1] - 1, mesh->coords[2]},
                    &mesh->nbrs[6]);
   else mesh->nbrs[6] = MPI_PROC_NULL;

   if (mesh->coords[0] < mesh->pdims[0] - 1 && mesh->coords[1] < mesh->pdims[1] - 1)
      MPI_Cart_rank(mesh->cart_comm,
                    (int[]){mesh->coords[0] + 1, mesh->coords[1] + 1, mesh->coords[2]},
                    &mesh->nbrs[7]);
   else mesh->nbrs[7] = MPI_PROC_NULL;

   if (mesh->coords[0] > 0 && mesh->coords[2] > 0)
      MPI_Cart_rank(mesh->cart_comm,
                    (int[]){mesh->coords[0] - 1, mesh->coords[1], mesh->coords[2] - 1},
                    &mesh->nbrs[8]);
   else mesh->nbrs[8] = MPI_PROC_NULL;

   if (mesh->coords[0] < mesh->pdims[0] - 1 && mesh->coords[2] < mesh->pdims[2] - 1)
      MPI_Cart_rank(mesh->cart_comm,
                    (int[]){mesh->coords[0] + 1, mesh->coords[1], mesh->coords[2] + 1},
                    &mesh->nbrs[9]);
   else mesh->nbrs[9] = MPI_PROC_NULL;

   if (mesh->coords[1] > 0 && mesh->coords[2] > 0)
      MPI_Cart_rank(mesh->cart_comm,
                    (int[]){mesh->coords[0], mesh->coords[1] - 1, mesh->coords[2] - 1},
                    &mesh->nbrs[10]);
   else mesh->nbrs[10] = MPI_PROC_NULL;

   if (mesh->coords[1] < mesh->pdims[1] - 1 && mesh->coords[2] < mesh->pdims[2] - 1)
      MPI_Cart_rank(mesh->cart_comm,
                    (int[]){mesh->coords[0], mesh->coords[1] + 1, mesh->coords[2] + 1},
                    &mesh->nbrs[11]);
   else mesh->nbrs[11] = MPI_PROC_NULL;

   if (mesh->coords[0] > 0 && mesh->coords[1] > 0 && mesh->coords[2] > 0)
      MPI_Cart_rank(
         mesh->cart_comm,
         (int[]){mesh->coords[0] - 1, mesh->coords[1] - 1, mesh->coords[2] - 1},
         &mesh->nbrs[12]);
   else mesh->nbrs[12] = MPI_PROC_NULL;

   if (mesh->coords[0] < mesh->pdims[0] - 1 && mesh->coords[1] < mesh->pdims[1] - 1 &&
       mesh->coords[2] < mesh->pdims[2] - 1)
      MPI_Cart_rank(
         mesh->cart_comm,
         (int[]){mesh->coords[0] + 1, mesh->coords[1] + 1, mesh->coords[2] + 1},
         &mesh->nbrs[13]);
   else mesh->nbrs[13] = MPI_PROC_NULL;

   *mesh_ptr = mesh;
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy distributed mesh structure
 *--------------------------------------------------------------------------*/
int
DestroyDistMesh(DistMesh **mesh_ptr)
{
   DistMesh *mesh = *mesh_ptr;
   if (!mesh) return 0;

   MPI_Comm_free(&(mesh->cart_comm));
   for (int i = 0; i < 3; i++) free(mesh->pstarts[i]);
   free(mesh);
   *mesh_ptr = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * Elasticity assembly helpers
 *--------------------------------------------------------------------------*/
/* Isotropic Voigt constitutive matrix from shear modulus G and first Lamé
   parameter lam. Passing lam = 0 yields the pure shear ("2G epsilon") operator
   used as the (1,1) block of the mixed u-p formulation, where the volumetric
   response is carried by the pressure unknown instead. */
static void
constitutive_matrix_3d_GL(const HYPRE_Real G, const HYPRE_Real lam, HYPRE_Real D[6][6])
{
   for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++) D[i][j] = 0.0;

   D[0][0] = lam + 2.0 * G;
   D[0][1] = lam;
   D[0][2] = lam;
   D[1][0] = lam;
   D[1][1] = lam + 2.0 * G;
   D[1][2] = lam;
   D[2][0] = lam;
   D[2][1] = lam;
   D[2][2] = lam + 2.0 * G;
   D[3][3] = G; /* yz */
   D[4][4] = G; /* xz */
   D[5][5] = G; /* xy */
}

static void
constitutive_matrix_3d(const HYPRE_Real E, const HYPRE_Real nu, HYPRE_Real D[6][6])
{
   HYPRE_Real G   = E / (2.0 * (1.0 + nu));
   HYPRE_Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

   constitutive_matrix_3d_GL(G, lam, D);
}

/* Eight-node signs in reference element (-1 or +1) */
static const int hex_sgn[8][3] = {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
                                  {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

/* 2-pt Gauss abscissae and weights */
static const HYPRE_Real gauss_x[2] = {-0.5773502691896257, 0.5773502691896257};
static const HYPRE_Real gauss_w[2] = {1.0, 1.0};

/* Compute shape functions and reference derivatives at (xi,eta,zeta) */
static void
q1_shape_ref(const HYPRE_Real xi, const HYPRE_Real eta, const HYPRE_Real zeta,
             HYPRE_Real N[8], HYPRE_Real dN_dxi[8], HYPRE_Real dN_deta[8],
             HYPRE_Real dN_dzeta[8])
{
   for (int a = 0; a < 8; a++)
   {
      HYPRE_Real sx = hex_sgn[a][0];
      HYPRE_Real sy = hex_sgn[a][1];
      HYPRE_Real sz = hex_sgn[a][2];
      HYPRE_Real f1 = (1.0 + sx * xi);
      HYPRE_Real f2 = (1.0 + sy * eta);
      HYPRE_Real f3 = (1.0 + sz * zeta);
      N[a]          = 0.125 * f1 * f2 * f3;
      dN_dxi[a]     = 0.125 * sx * f2 * f3;
      dN_deta[a]    = 0.125 * sy * f1 * f3;
      dN_dzeta[a]   = 0.125 * sz * f1 * f2;
   }
}

/*--------------------------------------------------------------------------
 * Precompute Q1 hex templates (Ke, volume N integral, top-face N integral)
 *--------------------------------------------------------------------------*/
static void
PrecomputeQ1HexTemplates(const HYPRE_Real hx, const HYPRE_Real hy, const HYPRE_Real hz,
                         const HYPRE_Real D[6][6], HYPRE_Real Ke_t[24][24],
                         HYPRE_Real Nvol_t[8], HYPRE_Real NfaceTop_t[8])
{
   for (int i = 0; i < 24; i++)
   {
      for (int j = 0; j < 24; j++)
      {
         Ke_t[i][j] = 0.0;
      }
   }

   for (int a = 0; a < 8; a++)
   {
      Nvol_t[a]     = 0.0;
      NfaceTop_t[a] = 0.0;
   }

   const HYPRE_Real Jinv[3] = {2.0 / hx, 2.0 / hy, 2.0 / hz};
   const HYPRE_Real detJ    = (hx * hy * hz) / 8.0;

   /* Volume: 2x2x2 Gauss */
   for (int iz = 0; iz < 2; iz++)
   {
      HYPRE_Real zeta = gauss_x[iz], wz = gauss_w[iz];
      for (int iy = 0; iy < 2; iy++)
      {
         HYPRE_Real eta = gauss_x[iy], wy = gauss_w[iy];
         for (int ix = 0; ix < 2; ix++)
         {
            HYPRE_Real xi = gauss_x[ix], wx = gauss_w[ix];
            HYPRE_Real w = wx * wy * wz * detJ;

            HYPRE_Real N[8], dxi[8], deta[8], dzeta[8];
            q1_shape_ref(xi, eta, zeta, N, dxi, deta, dzeta);

            HYPRE_Real dNx[8], dNy[8], dNz[8];
            for (int a = 0; a < 8; a++)
            {
               dNx[a] = dxi[a] * Jinv[0];
               dNy[a] = deta[a] * Jinv[1];
               dNz[a] = dzeta[a] * Jinv[2];
               Nvol_t[a] += N[a] * w;
            }

            for (int a = 0; a < 8; a++)
            {
               HYPRE_Real Ba[6][3] = {{dNx[a], 0.0, 0.0},    {0.0, dNy[a], 0.0},
                                      {0.0, 0.0, dNz[a]},    {0.0, dNz[a], dNy[a]},
                                      {dNz[a], 0.0, dNx[a]}, {dNy[a], dNx[a], 0.0}};
               for (int b = 0; b < 8; b++)
               {
                  HYPRE_Real Bb[6][3]  = {{dNx[b], 0.0, 0.0},    {0.0, dNy[b], 0.0},
                                          {0.0, 0.0, dNz[b]},    {0.0, dNz[b], dNy[b]},
                                          {dNz[b], 0.0, dNx[b]}, {dNy[b], dNx[b], 0.0}};
                  HYPRE_Real Kab[3][3] = {{0}};
                  for (int i = 0; i < 3; i++)
                  {
                     for (int j = 0; j < 3; j++)
                     {
                        HYPRE_Real sum = 0.0;
                        for (int alpha = 0; alpha < 6; alpha++)
                        {
                           for (int beta = 0; beta < 6; beta++)
                           {
                              sum += Ba[alpha][i] * D[alpha][beta] * Bb[beta][j];
                           }
                        }
                        Kab[i][j] = sum * w;
                     }
                  }

                  int ia = 3 * a, ib = 3 * b;
                  for (int i = 0; i < 3; i++)
                  {
                     for (int j = 0; j < 3; j++)
                     {
                        Ke_t[ia + i][ib + j] += Kab[i][j];
                     }
                  }
               }
            }
         }
      }
   }

   /* Top face (y = +1): 2x2 Gauss over (xi,zeta) */
   const HYPRE_Real detJs = (hx * hz) / 4.0;
   const HYPRE_Real eta   = 1.0;
   for (int iz = 0; iz < 2; iz++)
   {
      HYPRE_Real zeta = gauss_x[iz], wz = gauss_w[iz];
      for (int ix = 0; ix < 2; ix++)
      {
         HYPRE_Real xi = gauss_x[ix], wx = gauss_w[ix];
         HYPRE_Real w = wx * wz * detJs;
         HYPRE_Real N[8], dxi[8], deta_[8], dzeta[8];
         q1_shape_ref(xi, eta, zeta, N, dxi, deta_, dzeta);
         for (int a = 0; a < 8; a++)
         {
            if (hex_sgn[a][1] == +1)
            {
               NfaceTop_t[a] += N[a] * w;
            }
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * Q1-P0 mean-dilatation (B-bar) element stiffness.
 *
 * Locking-free, displacement-only counterpart of PrecomputeQ1HexTemplates' Ke.
 * The volumetric (dilatational) part of the strain-displacement operator B is
 * replaced element-wide by its volume average, while the deviatoric part is kept
 * pointwise. This is the mean-dilatation method (Nagtegaal-Parks-Rice 1974;
 * Hughes, FEM, the B-bar method) -- algebraically the Q1/P0 mixed element with
 * its element-constant pressure statically condensed out. Relaxing the
 * volumetric constraint to one per element keeps the operator well conditioned
 * as nu -> 1/2, so the stiffness stays SPD and is solved with the same PCG + AMG
 * as the ordinary CG path. Same uniform-hex geometry/material inputs as
 * PrecomputeQ1HexTemplates; only the stiffness changes (the RHS templates
 * Nvol/Nface are geometric and identical, so they are not recomputed here).
 *
 * Bbar keeps the pointwise deviatoric rows and replaces the dilatational (trace)
 * part of the three normal-strain rows by the averaged gradient: for node a,
 * displacement component j and normal-strain row i (Voigt order [xx,yy,zz,...]),
 *    Bbar[i][j] = B[i][j] + (1/3) ( gbar[a][j] - g[a][j] ),
 * with g[a] the pointwise gradient and gbar[a] = (1/Ve) integral grad N_a.
 *--------------------------------------------------------------------------*/
static void
PrecomputeQ1HexBbar(const HYPRE_Real hx, const HYPRE_Real hy, const HYPRE_Real hz,
                    const HYPRE_Real D[6][6], HYPRE_Real Ke_t[24][24])
{
   for (int i = 0; i < 24; i++)
      for (int j = 0; j < 24; j++) Ke_t[i][j] = 0.0;

   const HYPRE_Real Jinv[3] = {2.0 / hx, 2.0 / hy, 2.0 / hz};
   const HYPRE_Real detJ    = (hx * hy * hz) / 8.0;

   /* Pass 1: volume-averaged nodal gradients gbar[a] = (1/Ve) integral grad N_a */
   HYPRE_Real gbar[8][3] = {{0}};
   HYPRE_Real Ve         = 0.0;
   for (int iz = 0; iz < 2; iz++)
      for (int iy = 0; iy < 2; iy++)
         for (int ix = 0; ix < 2; ix++)
         {
            HYPRE_Real w = gauss_w[ix] * gauss_w[iy] * gauss_w[iz] * detJ;
            HYPRE_Real N[8], dxi[8], deta[8], dzeta[8];
            q1_shape_ref(gauss_x[ix], gauss_x[iy], gauss_x[iz], N, dxi, deta, dzeta);
            Ve += w;
            for (int a = 0; a < 8; a++)
            {
               gbar[a][0] += dxi[a] * Jinv[0] * w;
               gbar[a][1] += deta[a] * Jinv[1] * w;
               gbar[a][2] += dzeta[a] * Jinv[2] * w;
            }
         }
   for (int a = 0; a < 8; a++)
      for (int d = 0; d < 3; d++) gbar[a][d] /= Ve;

   /* Pass 2: integrate Kbar = integral Bbar^T D Bbar */
   for (int iz = 0; iz < 2; iz++)
      for (int iy = 0; iy < 2; iy++)
         for (int ix = 0; ix < 2; ix++)
         {
            HYPRE_Real w = gauss_w[ix] * gauss_w[iy] * gauss_w[iz] * detJ;
            HYPRE_Real N[8], dxi[8], deta[8], dzeta[8];
            q1_shape_ref(gauss_x[ix], gauss_x[iy], gauss_x[iz], N, dxi, deta, dzeta);

            /* Bbar[a]: 6x3 block (Voigt rows x displacement components) */
            HYPRE_Real Bbar[8][6][3];
            for (int a = 0; a < 8; a++)
            {
               HYPRE_Real gx = dxi[a] * Jinv[0];
               HYPRE_Real gy = deta[a] * Jinv[1];
               HYPRE_Real gz = dzeta[a] * Jinv[2];
               HYPRE_Real c0 = (gbar[a][0] - gx) / 3.0;
               HYPRE_Real c1 = (gbar[a][1] - gy) / 3.0;
               HYPRE_Real c2 = (gbar[a][2] - gz) / 3.0;
               /* normal rows xx,yy,zz: pointwise normal grad + averaged trace fix */
               Bbar[a][0][0] = gx + c0; Bbar[a][0][1] = c1;      Bbar[a][0][2] = c2;
               Bbar[a][1][0] = c0;      Bbar[a][1][1] = gy + c1; Bbar[a][1][2] = c2;
               Bbar[a][2][0] = c0;      Bbar[a][2][1] = c1;      Bbar[a][2][2] = gz + c2;
               /* shear rows yz,xz,xy: unchanged (match PrecomputeQ1HexTemplates) */
               Bbar[a][3][0] = 0.0;     Bbar[a][3][1] = gz;      Bbar[a][3][2] = gy;
               Bbar[a][4][0] = gz;      Bbar[a][4][1] = 0.0;     Bbar[a][4][2] = gx;
               Bbar[a][5][0] = gy;      Bbar[a][5][1] = gx;      Bbar[a][5][2] = 0.0;
            }

            for (int a = 0; a < 8; a++)
               for (int b = 0; b < 8; b++)
               {
                  int ia = 3 * a, ib = 3 * b;
                  for (int i = 0; i < 3; i++)
                     for (int j = 0; j < 3; j++)
                     {
                        HYPRE_Real sum = 0.0;
                        for (int al = 0; al < 6; al++)
                           for (int be = 0; be < 6; be++)
                              sum += Bbar[a][al][i] * D[al][be] * Bbar[b][be][j];
                        Ke_t[ia + i][ib + j] += sum * w;
                     }
               }
         }
}

/*==========================================================================
 *  Two-material bar: mixed u-p (top) over CG (bottom) -- parallel layout
 *==========================================================================
 *  The bar is one continuous Q1 displacement mesh (Nx x Ny x Nz nodes) split
 *  at the y = Ly/2 plane (global node layer Jsplit = (Ny-1)/2). The bottom half
 *  (gy in [0,Jsplit]) is standard CG elasticity; the top half (gy in
 *  [Jsplit,Ny-1]) additionally carries one scalar pressure unknown per node.
 *
 *  Each rank owns a CONTIGUOUS global row range ordered as
 *      [ bottom displacement dofs ][ top displacement dofs ][ pressure dofs ].
 *  The Y partition is FLIPPED between the two halves so that, for fixed (cx,cz),
 *  the rank with cy = 0 owns the node slabs of BOTH halves adjacent to the
 *  interface. Hence the shared interface nodes and the interface coupling are
 *  rank-local (no communication across the material interface), and every rank
 *  owns part of both halves (load balanced).
 *
 *  Internal rank id: lr = cx + Px*(cy + Py*cz). The layout is deterministic, so
 *  every rank builds it identically and can compute global dof ids for off-rank
 *  nodes too (needed for IJ column ids).
 *==========================================================================*/
typedef struct
{
   HYPRE_Int     Px, Py, Pz;
   HYPRE_Int     Nx, Ny, Nz;
   HYPRE_Int     Jsplit;     /* interface global node layer = (Ny-1)/2 */
   HYPRE_Int     nprocs;
   HYPRE_BigInt *xstart;     /* [Px+1] node prefix in x */
   HYPRE_BigInt *zstart;     /* [Pz+1] node prefix in z */
   HYPRE_BigInt *dbot_lo;    /* [Py+1] bottom-half band prefix (dist. from interface) */
   HYPRE_BigInt *dtop_lo;    /* [Py+1] top-half band prefix   (dist. from interface) */
   HYPRE_BigInt *count_ubot; /* [nprocs] owned bottom-disp dofs */
   HYPRE_BigInt *count_utop; /* [nprocs] owned top-disp dofs */
   HYPRE_BigInt *count_p;    /* [nprocs] owned pressure dofs */
   HYPRE_BigInt *offset;     /* [nprocs+1] rank-contiguous prefix sum */
   HYPRE_BigInt  N;          /* total global dofs */
} MixedLayout;

static inline HYPRE_Int
ml_lrank(const MixedLayout *L, HYPRE_Int cx, HYPRE_Int cy, HYPRE_Int cz)
{
   return cx + L->Px * (cy + L->Py * cz);
}

/* Largest block c in [0,P-1] with starts[c] <= idx */
static inline HYPRE_Int
ml_find_block(const HYPRE_BigInt *starts, HYPRE_Int P, HYPRE_BigInt idx)
{
   HYPRE_Int c = 0;
   while (c + 1 < P && idx >= starts[c + 1]) c++;
   return c;
}

/* Balanced prefix split of n consecutive items into P parts */
static void
ml_split_prefix(HYPRE_BigInt n, HYPRE_Int P, HYPRE_BigInt *prefix)
{
   HYPRE_BigInt base = (P > 0) ? n / P : 0;
   HYPRE_BigInt rest = n - base * (HYPRE_BigInt)P;
   prefix[0]         = 0;
   for (HYPRE_Int c = 0; c < P; c++)
   {
      prefix[c + 1] = prefix[c] + base + ((HYPRE_BigInt)c < rest ? 1 : 0);
   }
}

/* Owner cy of a bottom-half node layer gy in [0,Jsplit] (interface included) */
static inline HYPRE_Int
ml_owner_bot(const MixedLayout *L, HYPRE_BigInt gy)
{
   return ml_find_block(L->dbot_lo, L->Py, (HYPRE_BigInt)L->Jsplit - gy);
}

/* Owner cy of a top-half displacement node layer gy in [Jsplit+1,Ny-1] */
static inline HYPRE_Int
ml_owner_top(const MixedLayout *L, HYPRE_BigInt gy)
{
   return ml_find_block(L->dtop_lo, L->Py, gy - (HYPRE_BigInt)(L->Jsplit + 1));
}

/* Map a pressure dof (global id in the FULL system numbering) to its index in
   MGR's COMPRESSED coarse (C-point) space: per rank, the pressure dofs renumbered
   contiguously in rank order. coarse_start is the prefix sum of count_p. */
static inline HYPRE_BigInt
ml_pdof_to_coarse(const MixedLayout *L, const HYPRE_BigInt *coarse_start, HYPRE_BigInt pgid)
{
   HYPRE_Int    owner  = ml_find_block(L->offset, L->nprocs, pgid);
   HYPRE_BigInt pstart = L->offset[owner] + L->count_ubot[owner] + L->count_utop[owner];
   return coarse_start[owner] + (pgid - pstart);
}

static int
MixedLayoutCreate(DistMesh *mesh, MixedLayout **L_ptr)
{
   MixedLayout *L = (MixedLayout *)calloc(1, sizeof(MixedLayout));
   L->Px          = mesh->pdims[0];
   L->Py          = mesh->pdims[1];
   L->Pz          = mesh->pdims[2];
   L->Nx          = mesh->gdims[0];
   L->Ny          = mesh->gdims[1];
   L->Nz          = mesh->gdims[2];
   L->Jsplit      = (L->Ny - 1) / 2;
   L->nprocs      = L->Px * L->Py * L->Pz;

   L->xstart = (HYPRE_BigInt *)malloc((size_t)(L->Px + 1) * sizeof(HYPRE_BigInt));
   L->zstart = (HYPRE_BigInt *)malloc((size_t)(L->Pz + 1) * sizeof(HYPRE_BigInt));
   ml_split_prefix(L->Nx, L->Px, L->xstart);
   ml_split_prefix(L->Nz, L->Pz, L->zstart);

   /* Flipped Y: bottom band has Jsplit+1 node layers, top (displacement,
      excluding the shared interface) has Ny-1-Jsplit. cy = 0 sits at the
      interface in both halves. */
   L->dbot_lo = (HYPRE_BigInt *)malloc((size_t)(L->Py + 1) * sizeof(HYPRE_BigInt));
   L->dtop_lo = (HYPRE_BigInt *)malloc((size_t)(L->Py + 1) * sizeof(HYPRE_BigInt));
   ml_split_prefix((HYPRE_BigInt)L->Jsplit + 1, L->Py, L->dbot_lo);
   ml_split_prefix((HYPRE_BigInt)(L->Ny - 1 - L->Jsplit), L->Py, L->dtop_lo);

   L->count_ubot = (HYPRE_BigInt *)calloc((size_t)L->nprocs, sizeof(HYPRE_BigInt));
   L->count_utop = (HYPRE_BigInt *)calloc((size_t)L->nprocs, sizeof(HYPRE_BigInt));
   L->count_p    = (HYPRE_BigInt *)calloc((size_t)L->nprocs, sizeof(HYPRE_BigInt));
   L->offset     = (HYPRE_BigInt *)calloc((size_t)(L->nprocs + 1), sizeof(HYPRE_BigInt));

   for (HYPRE_Int lr = 0; lr < L->nprocs; lr++)
   {
      HYPRE_Int    cx  = lr % L->Px;
      HYPRE_Int    cy  = (lr / L->Px) % L->Py;
      HYPRE_Int    cz  = lr / (L->Px * L->Py);
      HYPRE_BigInt nx  = L->xstart[cx + 1] - L->xstart[cx];
      HYPRE_BigInt nz  = L->zstart[cz + 1] - L->zstart[cz];
      HYPRE_BigInt nyb = L->dbot_lo[cy + 1] - L->dbot_lo[cy];
      HYPRE_BigInt nyt = L->dtop_lo[cy + 1] - L->dtop_lo[cy];
      HYPRE_BigInt nyp = nyt + (cy == 0 ? 1 : 0); /* + interface pressure row */

      L->count_ubot[lr] = 3 * nx * nyb * nz;
      L->count_utop[lr] = 3 * nx * nyt * nz;
      L->count_p[lr]    = nx * nyp * nz;
      L->offset[lr + 1] =
         L->offset[lr] + L->count_ubot[lr] + L->count_utop[lr] + L->count_p[lr];
   }
   L->N = L->offset[L->nprocs];

   HYPRE_BigInt expect = (HYPRE_BigInt)3 * L->Nx * L->Ny * L->Nz +
                         (HYPRE_BigInt)L->Nx * (L->Ny - L->Jsplit) * L->Nz;
   if (L->N != expect)
   {
      printf("Error: mixed layout dof count %lld != expected %lld\n", (long long)L->N,
             (long long)expect);
      return 1;
   }

   *L_ptr = L;
   return 0;
}

static void
MixedLayoutDestroy(MixedLayout **L_ptr)
{
   MixedLayout *L = *L_ptr;
   if (!L) return;
   free(L->xstart);
   free(L->zstart);
   free(L->dbot_lo);
   free(L->dtop_lo);
   free(L->count_ubot);
   free(L->count_utop);
   free(L->count_p);
   free(L->offset);
   free(L);
   *L_ptr = NULL;
}

/* Global displacement dof id for node (gx,gy,gz), component comp in {0,1,2}.
   Interface nodes (gy == Jsplit) belong to the bottom block by construction. */
static HYPRE_BigInt
ml_udof(const MixedLayout *L, HYPRE_BigInt gx, HYPRE_BigInt gy, HYPRE_BigInt gz, int comp)
{
   HYPRE_Int    cx     = ml_find_block(L->xstart, L->Px, gx);
   HYPRE_Int    cz     = ml_find_block(L->zstart, L->Pz, gz);
   int          is_top = (gy > L->Jsplit);
   HYPRE_Int    cy;
   HYPRE_BigInt gy_lo, nyslab;

   if (!is_top)
   {
      cy     = ml_owner_bot(L, gy);
      gy_lo  = (HYPRE_BigInt)L->Jsplit - L->dbot_lo[cy + 1] + 1;
      nyslab = L->dbot_lo[cy + 1] - L->dbot_lo[cy];
   }
   else
   {
      cy     = ml_owner_top(L, gy);
      gy_lo  = (HYPRE_BigInt)L->Jsplit + 1 + L->dtop_lo[cy];
      nyslab = L->dtop_lo[cy + 1] - L->dtop_lo[cy];
   }

   HYPRE_Int    lr   = ml_lrank(L, cx, cy, cz);
   HYPRE_BigInt nx   = L->xstart[cx + 1] - L->xstart[cx];
   HYPRE_BigInt nl   = (gx - L->xstart[cx]) +
                     nx * ((gy - gy_lo) + nyslab * (gz - L->zstart[cz]));
   HYPRE_BigInt base = L->offset[lr] + (is_top ? L->count_ubot[lr] : 0);
   return base + 3 * nl + comp;
}

/* Global pressure dof id for a top-material node (gx,gy,gz), gy in [Jsplit,Ny-1].
   The interface pressure (gy == Jsplit) is owned by the cy = 0 rank, co-located
   with the bottom-half interface displacement nodes. */
static HYPRE_BigInt
ml_pdof(const MixedLayout *L, HYPRE_BigInt gx, HYPRE_BigInt gy, HYPRE_BigInt gz)
{
   HYPRE_Int    cx = ml_find_block(L->xstart, L->Px, gx);
   HYPRE_Int    cz = ml_find_block(L->zstart, L->Pz, gz);
   HYPRE_Int    cy = (gy == L->Jsplit) ? 0 : ml_owner_top(L, gy);
   HYPRE_BigInt gy_lo, nyslab;

   if (cy == 0)
   {
      gy_lo  = L->Jsplit; /* pressure slab on cy=0 starts at the interface */
      nyslab = (L->dtop_lo[1] - L->dtop_lo[0]) + 1;
   }
   else
   {
      gy_lo  = (HYPRE_BigInt)L->Jsplit + 1 + L->dtop_lo[cy];
      nyslab = L->dtop_lo[cy + 1] - L->dtop_lo[cy];
   }

   HYPRE_Int    lr   = ml_lrank(L, cx, cy, cz);
   HYPRE_BigInt nx   = L->xstart[cx + 1] - L->xstart[cx];
   HYPRE_BigInt nl   = (gx - L->xstart[cx]) +
                     nx * ((gy - gy_lo) + nyslab * (gz - L->zstart[cz]));
   HYPRE_BigInt base = L->offset[lr] + L->count_ubot[lr] + L->count_utop[lr];
   return base + nl;
}

/*--------------------------------------------------------------------------
 * Precompute mixed top-material element templates (uniform hex). Emits:
 *   A_t[24][24] : deviatoric stiffness  int 2G eps(u):eps(v)  (lambda = 0)
 *   B_t[8][24]  : pressure-displacement coupling  B[b][3a+i] = int N_b dN_a/dx_i
 *   C_t[8][8]   : pressure block  (1/lambda) M + S  (Bochev-Dohrmann PSPP)
 * The (2,2) block is inserted with a minus sign (symmetric saddle point).
 * As nu_top -> 0.5, lambda -> inf so (1/lambda) M -> 0 and C -> S (still SPD),
 * which is what makes the formulation locking-free.
 *--------------------------------------------------------------------------*/
static void
PrecomputeMixedTopTemplates(const HYPRE_Real hx, const HYPRE_Real hy, const HYPRE_Real hz,
                            const HYPRE_Real E_top, const HYPRE_Real nu_top,
                            HYPRE_Real A_t[24][24], HYPRE_Real B_t[8][24],
                            HYPRE_Real C_t[8][8], HYPRE_Real Mhat_t[8][8])
{
   const HYPRE_Real G   = E_top / (2.0 * (1.0 + nu_top));
   const HYPRE_Real lam = E_top * nu_top / ((1.0 + nu_top) * (1.0 - 2.0 * nu_top));

   HYPRE_Real D_A[6][6];
   constitutive_matrix_3d_GL(G, 0.0, D_A); /* pure 2G eps:eps operator */

   for (int i = 0; i < 24; i++)
      for (int j = 0; j < 24; j++) A_t[i][j] = 0.0;
   for (int b = 0; b < 8; b++)
      for (int j = 0; j < 24; j++) B_t[b][j] = 0.0;

   HYPRE_Real M[8][8], m[8];
   for (int a = 0; a < 8; a++)
   {
      m[a] = 0.0;
      for (int b = 0; b < 8; b++) M[a][b] = 0.0;
   }

   const HYPRE_Real Jinv[3] = {2.0 / hx, 2.0 / hy, 2.0 / hz};
   const HYPRE_Real detJ    = (hx * hy * hz) / 8.0;
   const HYPRE_Real vol     = hx * hy * hz;

   for (int iz = 0; iz < 2; iz++)
   {
      HYPRE_Real zeta = gauss_x[iz], wz = gauss_w[iz];
      for (int iy = 0; iy < 2; iy++)
      {
         HYPRE_Real eta = gauss_x[iy], wy = gauss_w[iy];
         for (int ix = 0; ix < 2; ix++)
         {
            HYPRE_Real xi = gauss_x[ix], wx = gauss_w[ix];
            HYPRE_Real w  = wx * wy * wz * detJ;

            HYPRE_Real N[8], dxi[8], deta[8], dzeta[8];
            q1_shape_ref(xi, eta, zeta, N, dxi, deta, dzeta);

            HYPRE_Real dNx[8], dNy[8], dNz[8];
            for (int a = 0; a < 8; a++)
            {
               dNx[a] = dxi[a] * Jinv[0];
               dNy[a] = deta[a] * Jinv[1];
               dNz[a] = dzeta[a] * Jinv[2];
               m[a] += N[a] * w;
            }

            /* A = int B^T D_A B */
            for (int a = 0; a < 8; a++)
            {
               HYPRE_Real Ba[6][3] = {{dNx[a], 0.0, 0.0},    {0.0, dNy[a], 0.0},
                                      {0.0, 0.0, dNz[a]},    {0.0, dNz[a], dNy[a]},
                                      {dNz[a], 0.0, dNx[a]}, {dNy[a], dNx[a], 0.0}};
               for (int b = 0; b < 8; b++)
               {
                  HYPRE_Real Bb[6][3] = {{dNx[b], 0.0, 0.0},    {0.0, dNy[b], 0.0},
                                         {0.0, 0.0, dNz[b]},    {0.0, dNz[b], dNy[b]},
                                         {dNz[b], 0.0, dNx[b]}, {dNy[b], dNx[b], 0.0}};
                  for (int i = 0; i < 3; i++)
                  {
                     for (int j = 0; j < 3; j++)
                     {
                        HYPRE_Real sum = 0.0;
                        for (int al = 0; al < 6; al++)
                           for (int be = 0; be < 6; be++)
                              sum += Ba[al][i] * D_A[al][be] * Bb[be][j];
                        A_t[3 * a + i][3 * b + j] += sum * w;
                     }
                  }
               }
            }

            /* B[b][3a+i] = int N_b dN_a/dx_i ;  M[b][a] = int N_b N_a */
            for (int b = 0; b < 8; b++)
            {
               for (int a = 0; a < 8; a++)
               {
                  B_t[b][3 * a + 0] += N[b] * dNx[a] * w;
                  B_t[b][3 * a + 1] += N[b] * dNy[a] * w;
                  B_t[b][3 * a + 2] += N[b] * dNz[a] * w;
                  M[b][a] += N[b] * N[a] * w;
               }
            }
         }
      }
   }

   /* C = (1/lambda) M + S, with PSPP stabilization
      S[a][b] = (1/(2G)) ( M[a][b] - m_a m_b / vol ). */
   const HYPRE_Real inv_lam = (lam > 0.0) ? 1.0 / lam : 0.0;
   for (int a = 0; a < 8; a++)
      for (int b = 0; b < 8; b++)
         C_t[a][b] =
            inv_lam * M[a][b] + (1.0 / (2.0 * G)) * (M[a][b] - m[a] * m[b] / vol);

   /* Scaled pressure-mass Schur approximation Mhat = (1/(2mu) + 1/lambda) M_p.
      This is the canonical lambda-robust Schur-complement preconditioner for
      nearly-incompressible elasticity. Unlike C above (which carries the PSPP
      mean-projection -m m^T/vol and becomes singular on the per-element constant
      pressure as lambda -> inf), the FULL mass matrix stays SPD and well
      conditioned, so it is the correct operator to use as an external coarse
      (C-block) operator in MGR. Optional output (skipped when Mhat_t is NULL). */
   if (Mhat_t)
   {
      for (int a = 0; a < 8; a++)
         for (int b = 0; b < 8; b++)
            Mhat_t[a][b] = (1.0 / (2.0 * G) + inv_lam) * M[a][b];
   }
}

/*--------------------------------------------------------------------------
 * Build Q1 Hex elasticity system (3 DOF/node, interleaved)
 *--------------------------------------------------------------------------*/
int
BuildElasticitySystem_Q1Hex(DistMesh *mesh, ElasticParams *params, HYPRE_IJMatrix *A_ptr,
                            HYPRE_IJVector *b_ptr, MPI_Comm solver_comm)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;

   const HYPRE_Int *p       = &mesh->coords[0];
   const HYPRE_Int *gdims   = &mesh->gdims[0];
   HYPRE_BigInt   **pstarts = &mesh->pstarts[0];

   HYPRE_BigInt node_ilower = mesh->ilower;
   HYPRE_BigInt node_iupper = mesh->iupper;
   HYPRE_BigInt dof_ilower  = node_ilower * 3;
   HYPRE_BigInt dof_iupper  = node_iupper * 3 + 2;

   /* Create IJ objects */
   HYPRE_IJMatrixCreate(solver_comm, dof_ilower, dof_iupper, dof_ilower, dof_iupper, &A);
   HYPRE_IJVectorCreate(solver_comm, dof_ilower, dof_iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   /* Set row sizes: conservative upper bound 81 per row (27 nodes x 3 dof) */
   HYPRE_Int  local_dofs = 3 * mesh->local_size;
   HYPRE_Int *nnzrow     = (HYPRE_Int *)calloc((size_t)local_dofs, sizeof(HYPRE_Int));
   for (int r = 0; r < local_dofs; r++) nnzrow[r] = 81;
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   free(nnzrow);

   HYPREDRV_IJ_MATRIX_INIT_HOST(A);
   HYPREDRV_IJ_VECTOR_INIT_HOST(b);

   /* Element sizes (physical domain) */
   HYPRE_Real hx = params->L[0] / (gdims[0] - 1);
   HYPRE_Real hy = params->L[1] / (gdims[1] - 1);
   HYPRE_Real hz = params->L[2] / (gdims[2] - 1);

   /* Constitutive matrix and body force */
   HYPRE_Real D[6][6];
   constitutive_matrix_3d(params->E, params->nu, D);
   HYPRE_Real bforce[3] = {params->rho * params->g[0], params->rho * params->g[1],
                           params->rho * params->g[2]};

   /* Precompute element templates */
   HYPRE_Real Ke_t[24][24];
   HYPRE_Real Nvol_t[8];
   HYPRE_Real NfaceTop_t[8];
   PrecomputeQ1HexTemplates(hx, hy, hz, D, Ke_t, Nvol_t, NfaceTop_t);

   /* B-bar (Q1-P0 mean-dilatation): displacement-only, locking-free stiffness.
      Reuses the entire CG assembly path -- only the element stiffness changes,
      so overwrite Ke_t (the RHS templates Nvol/Nface above are geometric and
      unchanged). Applies to both single-material and two-material bars. */
   const int use_bbar = (params->discretization == 2);
   if (use_bbar) PrecomputeQ1HexBbar(hx, hy, hz, D, Ke_t);

   /* Two-material STANDARD CG / B-bar: a second stiffness template for the top
      half. The element material is selected by y-region in the loop below.
      (Reached only for --problem two-material with --discretization standard or
      bbar; single-material and mixed runs never set params->problem here.) */
   HYPRE_Real         Ke_top[24][24];
   const HYPRE_BigInt Jsplit = (gdims[1] - 1) / 2;
   if (params->problem)
   {
      HYPRE_Real Dtop[6][6], nvol_tmp[8], nface_tmp[8];
      constitutive_matrix_3d(params->E_top, params->nu_top, Dtop);
      PrecomputeQ1HexTemplates(hx, hy, hz, Dtop, Ke_top, nvol_tmp, nface_tmp);
      if (use_bbar) PrecomputeQ1HexBbar(hx, hy, hz, Dtop, Ke_top);
   }

   /* Loop over elements intersecting this rank's node set:
    * gx in [max(start-1,0), min(end, Nx-1)) and similarly for gy,gz. */
   HYPRE_BigInt gz_lo = (pstarts[2][p[2]] > 0) ? (pstarts[2][p[2]] - 1) : 0;
   HYPRE_BigInt gz_hi =
      (pstarts[2][p[2] + 1] < gdims[2]) ? pstarts[2][p[2] + 1] : (gdims[2] - 1);
   HYPRE_BigInt gy_lo = (pstarts[1][p[1]] > 0) ? (pstarts[1][p[1]] - 1) : 0;
   HYPRE_BigInt gy_hi =
      (pstarts[1][p[1] + 1] < gdims[1]) ? pstarts[1][p[1] + 1] : (gdims[1] - 1);
   HYPRE_BigInt gx_lo = (pstarts[0][p[0]] > 0) ? (pstarts[0][p[0]] - 1) : 0;
   HYPRE_BigInt gx_hi =
      (pstarts[0][p[0] + 1] < gdims[0]) ? pstarts[0][p[0] + 1] : (gdims[0] - 1);

   /* Parallelize element loop - each thread processes all elements but only
    * adds contributions for rows in its partition */
   HYPRE_Int num_elements_z = (HYPRE_Int)(gz_hi - gz_lo);
   HYPRE_Int num_elements_y = (HYPRE_Int)(gy_hi - gy_lo);
   HYPRE_Int num_elements_x = (HYPRE_Int)(gx_hi - gx_lo);
   HYPRE_Int total_elements = num_elements_z * num_elements_y * num_elements_x;

#ifdef _OPENMP
#pragma omp parallel
#endif
   {
#ifdef _OPENMP
      HYPRE_Int tid         = omp_get_thread_num();
      HYPRE_Int num_threads = omp_get_num_threads();
#else
      HYPRE_Int tid         = 0;
      HYPRE_Int num_threads = 1;
#endif
      /* Partition owned DOF range among OpenMP threads to avoid data races.
       * Each thread only adds contributions for rows in its partition. */
      HYPRE_BigInt dof_range        = dof_iupper - dof_ilower + 1;
      HYPRE_BigInt dofs_per_thread  = (dof_range + num_threads - 1) / num_threads;
      HYPRE_BigInt thread_dof_lower = dof_ilower + tid * dofs_per_thread;
      HYPRE_BigInt thread_dof_upper = (tid == num_threads - 1)
                                         ? dof_iupper
                                         : (dof_ilower + (tid + 1) * dofs_per_thread - 1);

      /* Process all elements, but only add contributions for rows in this thread's
       * partition. Each thread processes ALL elements to ensure that when an element
       * contributes to rows in multiple partitions, all contributions are added.
       * This avoids the need for critical sections while maintaining correctness. */
      for (HYPRE_Int elem_idx = 0; elem_idx < total_elements; elem_idx++)
      {
         /* Convert linear index back to 3D coordinates */
         HYPRE_Int idx_z = elem_idx / (num_elements_y * num_elements_x);
         HYPRE_Int idx_y =
            (elem_idx % (num_elements_y * num_elements_x)) / num_elements_x;
         HYPRE_Int    idx_x = elem_idx % num_elements_x;
         HYPRE_BigInt gz    = gz_lo + idx_z;
         HYPRE_BigInt gy    = gy_lo + idx_y;
         HYPRE_BigInt gx    = gx_lo + idx_x;

         /* Select the element material template (two-material: top vs bottom) */
         const HYPRE_Real(*Ke_e)[24] =
            (params->problem && gy >= Jsplit) ? Ke_top : Ke_t;

         /* Global node coords of the 8 vertices (lexicograph - x fastest) */
         HYPRE_BigInt ng[8][3] = {{gx, gy, gz},
                                  {gx + 1, gy, gz},
                                  {gx + 1, gy + 1, gz},
                                  {gx, gy + 1, gz},
                                  {gx, gy, gz + 1},
                                  {gx + 1, gy, gz + 1},
                                  {gx + 1, gy + 1, gz + 1},
                                  {gx, gy + 1, gz + 1}};

         /* Global scalar node ids and DOF ids */
         HYPRE_BigInt node_gid[8];
         HYPRE_BigInt dof_gid[24];
         for (int a = 0; a < 8; a++)
         {
            HYPRE_Int owner_bc[3];
            for (int d = 0; d < 3; d++)
            {
               HYPRE_Int     pd = mesh->pdims[d];
               HYPRE_BigInt *ps = mesh->pstarts[d];
               HYPRE_BigInt  cg = ng[a][d];
               HYPRE_Int     bd = 0;
               while ((bd + 1) < pd && cg >= ps[bd + 1]) bd++;
               owner_bc[d] = bd;
            }
            node_gid[a]        = grid2idx(ng[a], owner_bc, gdims, pstarts);
            dof_gid[3 * a + 0] = 3 * node_gid[a] + 0;
            dof_gid[3 * a + 1] = 3 * node_gid[a] + 1;
            dof_gid[3 * a + 2] = 3 * node_gid[a] + 2;
         }

         /* Top-face flag for this element (y = Ly plane) */
         HYPRE_Int on_top_face = (gy + 1 == gdims[1] - 1) ? 1 : 0;

         /* Local element vectors (matrix is reused template) */
         HYPRE_Real fe[24];
         for (int a = 0; a < 8; a++)
         {
            for (int i3 = 0; i3 < 3; i3++)
            {
               fe[3 * a + i3] = bforce[i3] * Nvol_t[a];
            }
         }
         if (on_top_face && params->traction_top)
         {
            for (int a = 0; a < 8; a++)
            {
               for (int i3 = 0; i3 < 3; i3++)
               {
                  fe[3 * a + i3] += params->traction[i3] * NfaceTop_t[a];
               }
            }
         }

         /* Apply clamped BC at x=0: skip any coupling to DOFs at nodes with
          * gx==0 */
         HYPRE_Int is_dirichlet[24];
         for (int a = 0; a < 8; a++)
         {
            HYPRE_Int on_clamped    = (ng[a][0] == 0) ? 1 : 0;
            is_dirichlet[3 * a + 0] = on_clamped;
            is_dirichlet[3 * a + 1] = on_clamped;
            is_dirichlet[3 * a + 2] = on_clamped;
         }

         /* Insert element contributions batched by rows (skip Dirichlet DOFs).
          * Only add contributions for rows in this thread's partition. */
         HYPRE_Int    nrows_elem = 0;
         HYPRE_Int    ncols_row[24];
         HYPRE_BigInt rows_elem[24];
         HYPRE_BigInt cols_flat[24 * 24];
         HYPRE_Real   vals_flat[24 * 24];
         HYPRE_Int    off = 0;
         for (int i = 0; i < 24; i++)
         {
            if (is_dirichlet[i]) continue; /* row suppressed */
            if (dof_gid[i] < dof_ilower || dof_gid[i] > dof_iupper)
               continue; /* only own rows */
            /* Only process rows in this thread's partition */
            if (dof_gid[i] < thread_dof_lower || dof_gid[i] > thread_dof_upper) continue;
            rows_elem[nrows_elem] = dof_gid[i];
            HYPRE_Int ncols       = 0;
            for (int j = 0; j < 24; j++)
            {
               if (is_dirichlet[j]) continue; /* column suppressed */
               cols_flat[off + ncols] = dof_gid[j];
               vals_flat[off + ncols] = Ke_e[i][j];
               ncols++;
            }
            ncols_row[nrows_elem] = ncols;
            off += ncols;
            nrows_elem++;
         }
         if (nrows_elem > 0)
         {
            HYPRE_Int max_batch =
               (params->max_rows_per_call > 0) ? params->max_rows_per_call : nrows_elem;
            for (HYPRE_Int start = 0; start < nrows_elem; start += max_batch)
            {
               HYPRE_Int count =
                  (start + max_batch <= nrows_elem) ? max_batch : (nrows_elem - start);
               /* compute offset into flat cols/vals for this chunk */
               HYPRE_Int start_off = 0;
               for (HYPRE_Int rr = 0; rr < start; rr++) start_off += ncols_row[rr];
               /* No critical section needed - each thread only touches its own rows */
               HYPRE_IJMatrixAddToValues(A, count, &ncols_row[start], &rows_elem[start],
                                         &cols_flat[start_off], &vals_flat[start_off]);
            }
         }

         /* Add force vector batched (skip Dirichlet dofs).
          * Only add contributions for rows in this thread's partition. */
         HYPRE_Int    nvals_elem = 0;
         HYPRE_BigInt vec_rows[24];
         HYPRE_Real   vec_vals[24];
         for (int i = 0; i < 24; i++)
         {
            if (is_dirichlet[i]) continue;
            if (dof_gid[i] < dof_ilower || dof_gid[i] > dof_iupper)
               continue; /* only own rows */
            /* Only process rows in this thread's partition */
            if (dof_gid[i] < thread_dof_lower || dof_gid[i] > thread_dof_upper) continue;
            vec_rows[nvals_elem] = dof_gid[i];
            vec_vals[nvals_elem] = fe[i];
            nvals_elem++;
         }
         if (nvals_elem > 0)
         {
            HYPRE_Int max_batch =
               (params->max_rows_per_call > 0) ? params->max_rows_per_call : nvals_elem;
            for (HYPRE_Int start = 0; start < nvals_elem; start += max_batch)
            {
               HYPRE_Int count =
                  (start + max_batch <= nvals_elem) ? max_batch : (nvals_elem - start);
               /* No critical section needed - each thread only touches its own rows */
               HYPRE_IJVectorAddToValues(b, count, &vec_rows[start], &vec_vals[start]);
            }
         }
      }
   }

   /* Set Dirichlet rows to identity with zero RHS (x=0 plane) BEFORE assembly.
      We skipped inserting any couplings to these DOFs above, so their rows/cols
      are empty so far. */
   if (pstarts[0][p[0]] == 0)
   {
      HYPRE_Int     ny_plane  = (HYPRE_Int)(pstarts[1][p[1] + 1] - pstarts[1][p[1]]);
      HYPRE_Int     nz_plane  = (HYPRE_Int)(pstarts[2][p[2] + 1] - pstarts[2][p[2]]);
      HYPRE_Int     num_nodes = ny_plane * nz_plane;
      HYPRE_Int     num_rows  = 3 * num_nodes;
      HYPRE_Int    *ncols_arr = (HYPRE_Int *)malloc((size_t)num_rows * sizeof(HYPRE_Int));
      HYPRE_BigInt *rows_arr =
         (HYPRE_BigInt *)malloc((size_t)num_rows * sizeof(HYPRE_BigInt));
      HYPRE_BigInt *cols_arr =
         (HYPRE_BigInt *)malloc((size_t)num_rows * sizeof(HYPRE_BigInt));
      HYPRE_Real *vals_arr = (HYPRE_Real *)malloc((size_t)num_rows * sizeof(HYPRE_Real));
      HYPRE_Real *rhs_arr  = (HYPRE_Real *)malloc((size_t)num_rows * sizeof(HYPRE_Real));

      HYPRE_Int idx          = 0;
      HYPRE_Int num_nodes_gy = (HYPRE_Int)(pstarts[1][p[1] + 1] - pstarts[1][p[1]]);
      HYPRE_Int num_nodes_gz = (HYPRE_Int)(pstarts[2][p[2] + 1] - pstarts[2][p[2]]);
      HYPRE_Int total_nodes  = num_nodes_gy * num_nodes_gz;

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (HYPRE_Int node_idx = 0; node_idx < total_nodes; node_idx++)
      {
         HYPRE_Int    idx_y      = node_idx / num_nodes_gz;
         HYPRE_Int    idx_z      = node_idx % num_nodes_gz;
         HYPRE_BigInt gy         = pstarts[1][p[1]] + idx_y;
         HYPRE_BigInt gz         = pstarts[2][p[2]] + idx_z;
         HYPRE_BigInt ncoords[3] = {0, gy, gz};
         HYPRE_Int    owner_bc[3];
         for (int d = 0; d < 3; d++)
         {
            HYPRE_Int     pd = mesh->pdims[d];
            HYPRE_BigInt *ps = mesh->pstarts[d];
            HYPRE_BigInt  cg = ncoords[d];
            HYPRE_Int     bd = 0;
            while ((bd + 1) < pd && cg >= ps[bd + 1]) bd++;
            owner_bc[d] = bd;
         }
         HYPRE_BigInt n_gid     = grid2idx(ncoords, owner_bc, gdims, pstarts);
         HYPRE_Int    local_idx = node_idx * 3;
         for (int c = 0; c < 3; c++)
         {
            HYPRE_BigInt r           = 3 * n_gid + c;
            rows_arr[local_idx + c]  = r;
            cols_arr[local_idx + c]  = r;
            vals_arr[local_idx + c]  = 1.0;
            rhs_arr[local_idx + c]   = 0.0;
            ncols_arr[local_idx + c] = 1;
         }
      }
      idx = num_rows;
      if (idx > 0)
      {
         HYPRE_Int max_batch =
            (params->max_rows_per_call > 0) ? params->max_rows_per_call : idx;
         for (HYPRE_Int start = 0; start < idx; start += max_batch)
         {
            HYPRE_Int count = (start + max_batch <= idx) ? max_batch : (idx - start);
            HYPRE_IJMatrixSetValues(A, count, &ncols_arr[start], &rows_arr[start],
                                    &cols_arr[start], &vals_arr[start]);
            HYPRE_IJVectorSetValues(b, count, &rows_arr[start], &rhs_arr[start]);
         }
      }
      free(ncols_arr);
      free(rows_arr);
      free(cols_arr);
      free(vals_arr);
      free(rhs_arr);
   }

   /* Finalize assembly after all values (including Dirichlet rows) are set */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);

   *A_ptr = A;
   *b_ptr = b;
   return 0;
}

/*--------------------------------------------------------------------------
 * Build the two-material bar system: CG displacement on the bottom half, mixed
 * u-p (Q1-Q1, PSPP-stabilized) on the top half. Returns an IJ matrix/vector,
 * the 7-label dofmap, and the number of local rows. Displacement is continuous
 * across the y = Ly/2 interface (shared nodes); pressure lives on top nodes.
 *--------------------------------------------------------------------------*/
int
BuildMixedTwoMaterialSystem(DistMesh *mesh, ElasticParams *params, HYPRE_IJMatrix *A_ptr,
                            HYPRE_IJVector *b_ptr, int **dofmap_ptr,
                            HYPRE_Int *local_rows_ptr, HYPRE_Real **rbm_ptr,
                            HYPRE_IJMatrix *M_ptr, HYPRE_IJMatrix *Schur_ptr,
                            MPI_Comm solver_comm)
{
   MixedLayout *L = NULL;
   if (MixedLayoutCreate(mesh, &L))
   {
      MPI_Abort(solver_comm, 1);
   }

   const HYPRE_Int    cx     = mesh->coords[0];
   const HYPRE_Int    cy     = mesh->coords[1];
   const HYPRE_Int    cz     = mesh->coords[2];
   const HYPRE_Int    lr     = ml_lrank(L, cx, cy, cz);
   const HYPRE_Int    Jsplit = L->Jsplit;
   const HYPRE_Int    Nx = L->Nx, Ny = L->Ny, Nz = L->Nz;
   const HYPRE_BigInt ilower = L->offset[lr];
   const HYPRE_BigInt iupper = L->offset[lr + 1] - 1;
   const HYPRE_Int    local_rows = (HYPRE_Int)(L->offset[lr + 1] - L->offset[lr]);

   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJMatrixCreate(solver_comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(solver_comm, ilower, iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPREDRV_IJ_MATRIX_INIT_HOST(A);
   HYPREDRV_IJ_VECTOR_INIT_HOST(b);

   /* Optional preconditioning matrix M: identical to A except the pressure (2,2)
      block uses the SPD scaled mass Schur -Mhat instead of the PSPP block -C.
      Used (with coarse_level_type: acc) to drive MGR's coarse solve with the
      mass-matrix Schur preconditioner Mhat = (1/(2mu)+1/lambda) M_p. */
   const int      build_prec = (M_ptr != NULL);
   HYPRE_IJMatrix M          = NULL;
   if (build_prec)
   {
      HYPRE_IJMatrixCreate(solver_comm, ilower, iupper, ilower, iupper, &M);
      HYPRE_IJMatrixSetObjectType(M, HYPRE_PARCSR);
      HYPREDRV_IJ_MATRIX_INIT_HOST(M);
   }

   /* Optional coarse Schur operator Mhat = (1/(2mu)+1/lambda) M_p, assembled over
      the COMPRESSED coarse (pressure C-point) numbering MGR expects for
      coarse_level_type: user. Provided to MGR via HYPRE_MGRSetCoarseGridMatrixAtLevel so
      MGR skips the Galerkin RAP product on the coarse level. */
   const int      build_schur  = (Schur_ptr != NULL);
   HYPRE_IJMatrix Schur        = NULL;
   HYPRE_BigInt  *coarse_start = NULL; /* [nprocs+1] prefix sum of count_p */
   if (build_schur)
   {
      coarse_start = (HYPRE_BigInt *)calloc((size_t)(L->nprocs + 1), sizeof(HYPRE_BigInt));
      for (HYPRE_Int r = 0; r < L->nprocs; r++)
      {
         coarse_start[r + 1] = coarse_start[r] + L->count_p[r];
      }
      HYPRE_BigInt cilower = coarse_start[lr];
      HYPRE_BigInt ciupper = coarse_start[lr] + L->count_p[lr] - 1;
      HYPRE_IJMatrixCreate(solver_comm, cilower, ciupper, cilower, ciupper, &Schur);
      HYPRE_IJMatrixSetObjectType(Schur, HYPRE_PARCSR);
      HYPREDRV_IJ_MATRIX_INIT_HOST(Schur);
   }

   const HYPRE_Real hx = params->L[0] / (Nx - 1);
   const HYPRE_Real hy = params->L[1] / (Ny - 1);
   const HYPRE_Real hz = params->L[2] / (Nz - 1);

   HYPRE_Real D_bot[6][6];
   constitutive_matrix_3d(params->E, params->nu, D_bot);

   HYPRE_Real Ke_bot[24][24], Nvol_t[8], NfaceTop_t[8];
   PrecomputeQ1HexTemplates(hx, hy, hz, D_bot, Ke_bot, Nvol_t, NfaceTop_t);

   HYPRE_Real A_t[24][24], B_t[8][24], C_t[8][8], Mhat_t[8][8];
   PrecomputeMixedTopTemplates(hx, hy, hz, params->E_top, params->nu_top, A_t, B_t, C_t,
                               Mhat_t);

   const HYPRE_Real bforce[3] = {params->rho * params->g[0], params->rho * params->g[1],
                                 params->rho * params->g[2]};

   /* Element loop bounds touching this rank's owned x/z nodes */
   HYPRE_BigInt gx_lo = (L->xstart[cx] > 0) ? L->xstart[cx] - 1 : 0;
   HYPRE_BigInt gx_hi =
      (L->xstart[cx + 1] - 1 < Nx - 1) ? L->xstart[cx + 1] - 1 : (HYPRE_BigInt)Nx - 2;
   HYPRE_BigInt gz_lo = (L->zstart[cz] > 0) ? L->zstart[cz] - 1 : 0;
   HYPRE_BigInt gz_hi =
      (L->zstart[cz + 1] - 1 < Nz - 1) ? L->zstart[cz + 1] - 1 : (HYPRE_BigInt)Nz - 2;

   for (HYPRE_BigInt gz = gz_lo; gz <= gz_hi; gz++)
   {
      for (HYPRE_BigInt gy = 0; gy < Ny - 1; gy++)
      {
         /* Skip element layers whose y-nodes this rank does not own */
         int own0 =
            (gy <= Jsplit) ? (ml_owner_bot(L, gy) == cy) : (ml_owner_top(L, gy) == cy);
         int own1 = ((gy + 1) <= Jsplit) ? (ml_owner_bot(L, gy + 1) == cy)
                                         : (ml_owner_top(L, gy + 1) == cy);
         if (!own0 && !own1) continue;

         int is_top = (gy >= Jsplit);

         for (HYPRE_BigInt gx = gx_lo; gx <= gx_hi; gx++)
         {
            HYPRE_BigInt ng[8][3] = {{gx, gy, gz},
                                     {gx + 1, gy, gz},
                                     {gx + 1, gy + 1, gz},
                                     {gx, gy + 1, gz},
                                     {gx, gy, gz + 1},
                                     {gx + 1, gy, gz + 1},
                                     {gx + 1, gy + 1, gz + 1},
                                     {gx, gy + 1, gz + 1}};

            HYPRE_Int  ndof = is_top ? 32 : 24;
            HYPRE_BigInt gid[32];
            int          is_dir[32];
            HYPRE_Real   fe[32];
            HYPRE_Real   Ke[32][32];

            for (int i = 0; i < ndof; i++)
            {
               fe[i] = 0.0;
               for (int j = 0; j < ndof; j++) Ke[i][j] = 0.0;
            }

            /* Displacement dofs (present in both materials) + body force */
            for (int a = 0; a < 8; a++)
            {
               int clamped = (ng[a][0] == 0);
               for (int i = 0; i < 3; i++)
               {
                  gid[3 * a + i]    = ml_udof(L, ng[a][0], ng[a][1], ng[a][2], i);
                  is_dir[3 * a + i] = clamped;
                  fe[3 * a + i]     = bforce[i] * Nvol_t[a];
               }
            }

            int on_top_face = (gy + 1 == Ny - 1);
            if (is_top && on_top_face && params->traction_top)
            {
               for (int a = 0; a < 8; a++)
                  for (int i = 0; i < 3; i++)
                     fe[3 * a + i] += params->traction[i] * NfaceTop_t[a];
            }

            if (!is_top)
            {
               for (int i = 0; i < 24; i++)
                  for (int j = 0; j < 24; j++) Ke[i][j] = Ke_bot[i][j];
            }
            else
            {
               for (int bb = 0; bb < 8; bb++)
               {
                  gid[24 + bb]    = ml_pdof(L, ng[bb][0], ng[bb][1], ng[bb][2]);
                  is_dir[24 + bb] = 0;
                  fe[24 + bb]     = 0.0;
               }
               /* [ A    B^T ]
                  [ B   -C   ]  (symmetric saddle point) */
               for (int i = 0; i < 24; i++)
                  for (int j = 0; j < 24; j++) Ke[i][j] = A_t[i][j];
               for (int bb = 0; bb < 8; bb++)
               {
                  for (int a = 0; a < 8; a++)
                     for (int i = 0; i < 3; i++)
                     {
                        Ke[3 * a + i][24 + bb] = B_t[bb][3 * a + i]; /* u-row, p-col */
                        Ke[24 + bb][3 * a + i] = B_t[bb][3 * a + i]; /* p-row, u-col */
                     }
                  for (int cc = 0; cc < 8; cc++) Ke[24 + bb][24 + cc] = -C_t[bb][cc];
               }
            }

            /* Coarse Schur Mhat: accumulate the element pressure-mass block into
               the compressed coarse numbering, owned pressure rows only -- exactly
               mirroring how A's -C pressure block is assembled above. */
            if (build_schur && is_top)
            {
               for (int bb = 0; bb < 8; bb++)
               {
                  HYPRE_BigInt prow = gid[24 + bb];
                  if (prow < ilower || prow > iupper) continue; /* owned pressure row */

                  HYPRE_BigInt crow = ml_pdof_to_coarse(L, coarse_start, prow);
                  HYPRE_BigInt ccols[8];
                  HYPRE_Real   cvals[8];
                  HYPRE_Int    cn = 8;
                  for (int cc = 0; cc < 8; cc++)
                  {
                     ccols[cc] = ml_pdof_to_coarse(L, coarse_start, gid[24 + cc]);
                     /* -Mhat: MGR's coarse operator is the (negative) Schur, to
                        match the -C sign of the assembled (2,2) block. */
                     cvals[cc] = -Mhat_t[bb][cc];
                  }
                  HYPRE_IJMatrixAddToValues(Schur, 1, &cn, &crow, ccols, cvals);
               }
            }

            /* Insert owned, non-Dirichlet rows; skip Dirichlet columns (the
               clamp value is zero, so no RHS correction is needed). */
            for (int i = 0; i < ndof; i++)
            {
               if (is_dir[i]) continue;
               HYPRE_BigInt row = gid[i];
               if (row < ilower || row > iupper) continue;

               HYPRE_Int    ncols = 0;
               HYPRE_BigInt cols[32];
               HYPRE_Real   vals[32];
               HYPRE_Real   vals_M[32];
               for (int j = 0; j < ndof; j++)
               {
                  if (is_dir[j]) continue;
                  cols[ncols] = gid[j];
                  vals[ncols] = Ke[i][j];
                  /* Prec matrix M differs from A only in the pressure (2,2)
                     block: -Mhat (SPD scaled mass) instead of -C (PSPP). */
                  vals_M[ncols] = (is_top && i >= 24 && j >= 24)
                                     ? -Mhat_t[i - 24][j - 24]
                                     : Ke[i][j];
                  ncols++;
               }
               HYPRE_IJMatrixAddToValues(A, 1, &ncols, &row, cols, vals);
               HYPRE_IJVectorAddToValues(b, 1, &row, &fe[i]);
               if (build_prec)
               {
                  HYPRE_IJMatrixAddToValues(M, 1, &ncols, &row, cols, vals_M);
               }
            }
         }
      }
   }

   /* Dirichlet identity rows on the clamped x = 0 plane (owned rows only) */
   if (cx == 0)
   {
      for (HYPRE_BigInt gz = L->zstart[cz]; gz < L->zstart[cz + 1]; gz++)
      {
         for (HYPRE_BigInt gy = 0; gy < Ny; gy++)
         {
            for (int i = 0; i < 3; i++)
            {
               HYPRE_BigInt row = ml_udof(L, 0, gy, gz, i);
               if (row < ilower || row > iupper) continue;
               HYPRE_Int  one  = 1;
               HYPRE_Real diag = 1.0;
               HYPRE_Real zero = 0.0;
               HYPRE_IJMatrixSetValues(A, 1, &one, &row, &row, &diag);
               HYPRE_IJVectorSetValues(b, 1, &row, &zero);
               if (build_prec)
               {
                  HYPRE_IJMatrixSetValues(M, 1, &one, &row, &row, &diag);
               }
            }
         }
      }
   }

   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);
   if (build_prec)
   {
      HYPRE_IJMatrixAssemble(M);
      *M_ptr = M;
   }
   if (build_schur)
   {
      HYPRE_IJMatrixAssemble(Schur);
      *Schur_ptr = Schur;
      free(coarse_start);
   }

   /* 7-label dofmap (one entry per local row, in [bottom-u][top-u][p] order):
      0,1,2 = bottom ux/uy/uz; 3,4,5 = top ux/uy/uz; 6 = top pressure. */
   int         *dofmap = (int *)malloc((size_t)(local_rows > 0 ? local_rows : 1) * sizeof(int));
   HYPRE_BigInt nub    = L->count_ubot[lr];
   HYPRE_BigInt nut    = L->count_utop[lr];
   for (HYPRE_Int i = 0; i < local_rows; i++)
   {
      if ((HYPRE_BigInt)i < nub)
         dofmap[i] = i % 3;
      else if ((HYPRE_BigInt)i < nub + nut)
         dofmap[i] = 3 + (int)(((HYPRE_BigInt)i - nub) % 3);
      else
         dofmap[i] = 6;
   }

   /* Rigid-body near-null-space modes (6) in the combined row ordering: the
      displacement rows carry the translation/rotation values, pressure rows are
      zero, and clamped (x=0) displacement rows are zero. As in the single-
      material case these are consumed by BoomerAMG only under nodal coarsening. */
   HYPRE_Real *rbm =
      (HYPRE_Real *)calloc((size_t)6 * (size_t)(local_rows > 0 ? local_rows : 1),
                           sizeof(HYPRE_Real));
   {
      const HYPRE_Real ccx = 0.5 * params->L[0];
      const HYPRE_Real ccy = 0.5 * params->L[1];
      const HYPRE_Real ccz = 0.5 * params->L[2];
      for (HYPRE_BigInt gz = L->zstart[cz]; gz < L->zstart[cz + 1]; gz++)
      {
         HYPRE_Real rzc = (HYPRE_Real)gz * hz - ccz;
         for (HYPRE_BigInt gx = L->xstart[cx]; gx < L->xstart[cx + 1]; gx++)
         {
            HYPRE_Real rxc = (HYPRE_Real)gx * hx - ccx;
            int        clamped = (gx == 0);
            for (HYPRE_BigInt gy = 0; gy < Ny; gy++)
            {
               HYPRE_BigInt r0 = ml_udof(L, gx, gy, gz, 0);
               if (r0 < ilower || r0 > iupper) continue; /* not an owned disp node */
               if (clamped) continue;                    /* modes stay zero */
               HYPRE_Int  loc = (HYPRE_Int)(r0 - ilower);
               HYPRE_Real ryc = (HYPRE_Real)gy * hy - ccy;
               rbm[0 * local_rows + loc + 0] = 1.0;  /* Tx */
               rbm[1 * local_rows + loc + 1] = 1.0;  /* Ty */
               rbm[2 * local_rows + loc + 2] = 1.0;  /* Tz */
               rbm[3 * local_rows + loc + 1] = -rzc; /* Rx: (0,-rz, ry) */
               rbm[3 * local_rows + loc + 2] = ryc;
               rbm[4 * local_rows + loc + 0] = rzc;  /* Ry: (rz, 0,-rx) */
               rbm[4 * local_rows + loc + 2] = -rxc;
               rbm[5 * local_rows + loc + 0] = -ryc; /* Rz: (-ry, rx, 0) */
               rbm[5 * local_rows + loc + 1] = rxc;
            }
         }
      }
   }

   MixedLayoutDestroy(&L);

   *A_ptr          = A;
   *b_ptr          = b;
   *dofmap_ptr     = dofmap;
   *local_rows_ptr = local_rows;
   *rbm_ptr        = rbm;
   return 0;
}

/*--------------------------------------------------------------------------
 * Write vector VTK (RectilinearGrid, point vectors named "displacement")
 *   - Simpler than Laplacian's: no ghost exchanges, non-overlapping pieces
 *--------------------------------------------------------------------------*/
static void
WriteCoordArray(FILE *fp, const char *name, double start, double delta, int count)
{
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n",
           name);
   fprintf(fp, "          ");
   for (int i = 0; i < count; i++)
   {
      fprintf(fp, "%.15g ", start + i * delta);
      if ((i + 1) % 6 == 0) fprintf(fp, "\n          ");
   }
   fprintf(fp, "\n        </DataArray>\n");
}

int
WriteVTKsolutionVector(DistMesh *mesh, ElasticParams *params, HYPRE_Real *sol_data)
{
   int myid, num_procs;
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Comm_size(mesh->cart_comm, &num_procs);

   /* Local sizes and offsets */
   HYPRE_Int     *p       = &mesh->coords[0];
   HYPRE_Int     *nlocal  = &mesh->nlocal[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_Real    *gsizes  = &mesh->gsizes[0];

   int nx       = (int)nlocal[0];
   int ny       = (int)nlocal[1];
   int nz       = (int)nlocal[2];
   int ix_start = (int)pstarts[0][p[0]];
   int iy_start = (int)pstarts[1][p[1]];
   int iz_start = (int)pstarts[2][p[2]];

   /* Overlap on negative faces (like Laplacian VTK writer) */
   int ofi = (p[0] > 0) ? 1 : 0; /* left  ghost layer */
   int ofj = (p[1] > 0) ? 1 : 0; /* down  ghost layer */
   int ofk = (p[2] > 0) ? 1 : 0; /* back  ghost layer */
   int nxg = nx + ofi;
   int nyg = ny + ofj;
   int nzg = nz + ofk;

   /* Allocate ghost buffers (vectors: 3 doubles per point) */
   double *ghost[14];
   for (int i = 0; i < 14; i++) ghost[i] = NULL;
   if (mesh->nbrs[0] != MPI_PROC_NULL)
      ghost[0] = (double *)calloc((size_t)(ny * nz * 3), sizeof(double)); /* left  face */
   if (mesh->nbrs[1] != MPI_PROC_NULL)
      ghost[1] =
         (double *)calloc((size_t)(ny * nz * 3), sizeof(double)); /* right face send */
   if (mesh->nbrs[2] != MPI_PROC_NULL)
      ghost[2] = (double *)calloc((size_t)(nx * nz * 3), sizeof(double)); /* down  face */
   if (mesh->nbrs[3] != MPI_PROC_NULL)
      ghost[3] =
         (double *)calloc((size_t)(nx * nz * 3), sizeof(double)); /* up    face send */
   if (mesh->nbrs[4] != MPI_PROC_NULL)
      ghost[4] = (double *)calloc((size_t)(nx * ny * 3), sizeof(double)); /* back  face */
   if (mesh->nbrs[5] != MPI_PROC_NULL)
      ghost[5] =
         (double *)calloc((size_t)(nx * ny * 3), sizeof(double)); /* front face send */
   if (mesh->nbrs[6] != MPI_PROC_NULL)
      ghost[6] = (double *)calloc((size_t)(nz * 3), sizeof(double)); /* left-down edge */
   if (mesh->nbrs[7] != MPI_PROC_NULL)
      ghost[7] =
         (double *)calloc((size_t)(nz * 3), sizeof(double)); /* right-up  edge send */
   if (mesh->nbrs[8] != MPI_PROC_NULL)
      ghost[8] = (double *)calloc((size_t)(ny * 3), sizeof(double)); /* left-back edge */
   if (mesh->nbrs[9] != MPI_PROC_NULL)
      ghost[9] =
         (double *)calloc((size_t)(ny * 3), sizeof(double)); /* right-front edge send */
   if (mesh->nbrs[10] != MPI_PROC_NULL)
      ghost[10] = (double *)calloc((size_t)(nx * 3), sizeof(double)); /* down-back edge */
   if (mesh->nbrs[11] != MPI_PROC_NULL)
      ghost[11] =
         (double *)calloc((size_t)(nx * 3), sizeof(double)); /* up-front  edge send */
   if (mesh->nbrs[12] != MPI_PROC_NULL)
      ghost[12] = (double *)calloc(3, sizeof(double)); /* left-down-back corner */
   if (mesh->nbrs[13] != MPI_PROC_NULL)
      ghost[13] = (double *)calloc(3, sizeof(double)); /* right-up-front corner send */

   /* Fill send buffers and post Irecvs */
   MPI_Request reqs[28];
   int         reqc = 0;

   /* X-direction faces */
   if (mesh->nbrs[1] != MPI_PROC_NULL)
   {
      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            int nidx          = (k * ny + j) * nx + (nx - 1);
            int off           = (k * ny + j) * 3;
            ghost[1][off + 0] = sol_data[3 * nidx + 0];
            ghost[1][off + 1] = sol_data[3 * nidx + 1];
            ghost[1][off + 2] = sol_data[3 * nidx + 2];
         }
      }
      MPI_Isend(ghost[1], ny * nz * 3, MPI_DOUBLE, mesh->nbrs[1], 0, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[0] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[0], ny * nz * 3, MPI_DOUBLE, mesh->nbrs[0], 0, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* Y-direction faces */
   if (mesh->nbrs[3] != MPI_PROC_NULL)
   {
      for (int k = 0; k < nz; k++)
      {
         for (int i = 0; i < nx; i++)
         {
            int nidx          = (k * ny + (ny - 1)) * nx + i;
            int off           = (k * nx + i) * 3;
            ghost[3][off + 0] = sol_data[3 * nidx + 0];
            ghost[3][off + 1] = sol_data[3 * nidx + 1];
            ghost[3][off + 2] = sol_data[3 * nidx + 2];
         }
      }
      MPI_Isend(ghost[3], nx * nz * 3, MPI_DOUBLE, mesh->nbrs[3], 1, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[2], nx * nz * 3, MPI_DOUBLE, mesh->nbrs[2], 1, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* Z-direction faces */
   if (mesh->nbrs[5] != MPI_PROC_NULL)
   {
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            int nidx          = ((nz - 1) * ny + j) * nx + i;
            int off           = (j * nx + i) * 3;
            ghost[5][off + 0] = sol_data[3 * nidx + 0];
            ghost[5][off + 1] = sol_data[3 * nidx + 1];
            ghost[5][off + 2] = sol_data[3 * nidx + 2];
         }
      }
      MPI_Isend(ghost[5], nx * ny * 3, MPI_DOUBLE, mesh->nbrs[5], 2, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[4] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[4], nx * ny * 3, MPI_DOUBLE, mesh->nbrs[4], 2, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* XY edge */
   if (mesh->nbrs[7] != MPI_PROC_NULL)
   {
      for (int k = 0; k < nz; k++)
      {
         int nidx            = (k * ny + (ny - 1)) * nx + (nx - 1);
         ghost[7][3 * k + 0] = sol_data[3 * nidx + 0];
         ghost[7][3 * k + 1] = sol_data[3 * nidx + 1];
         ghost[7][3 * k + 2] = sol_data[3 * nidx + 2];
      }
      MPI_Isend(ghost[7], nz * 3, MPI_DOUBLE, mesh->nbrs[7], 3, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[6] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[6], nz * 3, MPI_DOUBLE, mesh->nbrs[6], 3, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* XZ edge */
   if (mesh->nbrs[9] != MPI_PROC_NULL)
   {
      for (int j = 0; j < ny; j++)
      {
         int nidx            = ((nz - 1) * ny + j) * nx + (nx - 1);
         ghost[9][3 * j + 0] = sol_data[3 * nidx + 0];
         ghost[9][3 * j + 1] = sol_data[3 * nidx + 1];
         ghost[9][3 * j + 2] = sol_data[3 * nidx + 2];
      }
      MPI_Isend(ghost[9], ny * 3, MPI_DOUBLE, mesh->nbrs[9], 4, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[8] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[8], ny * 3, MPI_DOUBLE, mesh->nbrs[8], 4, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* YZ edge */
   if (mesh->nbrs[11] != MPI_PROC_NULL)
   {
      for (int i = 0; i < nx; i++)
      {
         int nidx             = ((nz - 1) * ny + (ny - 1)) * nx + i;
         ghost[11][3 * i + 0] = sol_data[3 * nidx + 0];
         ghost[11][3 * i + 1] = sol_data[3 * nidx + 1];
         ghost[11][3 * i + 2] = sol_data[3 * nidx + 2];
      }
      MPI_Isend(ghost[11], nx * 3, MPI_DOUBLE, mesh->nbrs[11], 5, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[10] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[10], nx * 3, MPI_DOUBLE, mesh->nbrs[10], 5, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* XYZ corner */
   if (mesh->nbrs[13] != MPI_PROC_NULL)
   {
      int nidx     = ((nz - 1) * ny + (ny - 1)) * nx + (nx - 1);
      ghost[13][0] = sol_data[3 * nidx + 0];
      ghost[13][1] = sol_data[3 * nidx + 1];
      ghost[13][2] = sol_data[3 * nidx + 2];
      MPI_Isend(ghost[13], 3, MPI_DOUBLE, mesh->nbrs[13], 6, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[12] != MPI_PROC_NULL)
   {
      MPI_Irecv(ghost[12], 3, MPI_DOUBLE, mesh->nbrs[12], 6, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* Build extended data including negative-face ghosts */
   double *ext = (double *)calloc((size_t)(nxg * nyg * nzg * 3), sizeof(double));
   for (int k = 0; k < nz; k++)
   {
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            int ig = i + ofi, jg = j + ofj, kg = k + ofk;
            int idx_e      = ((kg * nyg + jg) * nxg + ig) * 3;
            int nidx       = (k * ny + j) * nx + i;
            ext[idx_e + 0] = sol_data[3 * nidx + 0];
            ext[idx_e + 1] = sol_data[3 * nidx + 1];
            ext[idx_e + 2] = sol_data[3 * nidx + 2];
         }
      }
   }

   /* Finish comms before using ghost buffers */
   if (reqc > 0)
   {
      MPI_Status *statuses = (MPI_Status *)malloc((size_t)reqc * sizeof(MPI_Status));
      MPI_Waitall(reqc, reqs, statuses);
      free(statuses);
   }

   /* Insert faces */
   if (mesh->nbrs[0] != MPI_PROC_NULL)
   {
      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            int kg = k + ofk, jg = j + ofj;
            int src      = (k * ny + j) * 3;
            int dst      = ((kg * nyg + jg) * nxg + 0) * 3;
            ext[dst + 0] = ghost[0][src + 0];
            ext[dst + 1] = ghost[0][src + 1];
            ext[dst + 2] = ghost[0][src + 2];
         }
      }
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL)
   {
      for (int k = 0; k < nz; k++)
      {
         for (int i = 0; i < nx; i++)
         {
            int kg = k + ofk, ig = i + ofi;
            int src      = (k * nx + i) * 3;
            int dst      = ((kg * nyg + 0) * nxg + ig) * 3;
            ext[dst + 0] = ghost[2][src + 0];
            ext[dst + 1] = ghost[2][src + 1];
            ext[dst + 2] = ghost[2][src + 2];
         }
      }
   }
   if (mesh->nbrs[4] != MPI_PROC_NULL)
   {
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            int jg = j + ofj, ig = i + ofi;
            int src      = (j * nx + i) * 3;
            int dst      = ((0 * nyg + jg) * nxg + ig) * 3;
            ext[dst + 0] = ghost[4][src + 0];
            ext[dst + 1] = ghost[4][src + 1];
            ext[dst + 2] = ghost[4][src + 2];
         }
      }
   }

   /* Insert edges */
   if (mesh->nbrs[6] != MPI_PROC_NULL) /* left-down */
   {
      for (int k = 0; k < nz; k++)
      {
         int kg       = k + ofk;
         int dst      = ((kg * nyg + 0) * nxg + 0) * 3;
         ext[dst + 0] = ghost[6][3 * k + 0];
         ext[dst + 1] = ghost[6][3 * k + 1];
         ext[dst + 2] = ghost[6][3 * k + 2];
      }
   }
   if (mesh->nbrs[8] != MPI_PROC_NULL) /* left-back */
   {
      for (int j = 0; j < ny; j++)
      {
         int jg       = j + ofj;
         int dst      = ((0 * nyg + jg) * nxg + 0) * 3;
         ext[dst + 0] = ghost[8][3 * j + 0];
         ext[dst + 1] = ghost[8][3 * j + 1];
         ext[dst + 2] = ghost[8][3 * j + 2];
      }
   }
   if (mesh->nbrs[10] != MPI_PROC_NULL) /* down-back */
   {
      for (int i = 0; i < nx; i++)
      {
         int ig       = i + ofi;
         int dst      = ((0 * nyg + 0) * nxg + ig) * 3;
         ext[dst + 0] = ghost[10][3 * i + 0];
         ext[dst + 1] = ghost[10][3 * i + 1];
         ext[dst + 2] = ghost[10][3 * i + 2];
      }
   }

   /* Insert corner */
   if (mesh->nbrs[12] != MPI_PROC_NULL)
   {
      int dst      = ((0 * nyg + 0) * nxg + 0) * 3;
      ext[dst + 0] = ghost[12][0];
      ext[dst + 1] = ghost[12][1];
      ext[dst + 2] = ghost[12][2];
   }

   /* Filenames */
   char filename[256];
   snprintf(filename, sizeof(filename), "elasticity_%dx%dx%d_%dx%dx%d_%d.vtr",
            (int)params->N[0], (int)params->N[1], (int)params->N[2], (int)params->P[0],
            (int)params->P[1], (int)params->P[2], myid);
   FILE *fp = fopen(filename, "w");
   if (!fp)
   {
      printf("Error: Cannot open file %s\n", filename);
      MPI_Abort(mesh->cart_comm, -1);
   }

   /* Coordinates (scale by physical dimensions) and extents with overlap */
   double x_step  = params->L[0] * gsizes[0];
   double y_step  = params->L[1] * gsizes[1];
   double z_step  = params->L[2] * gsizes[2];
   double x_start = (ix_start - ofi) * x_step;
   double y_start = (iy_start - ofj) * y_step;
   double z_start = (iz_start - ofk) * z_step;

   fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
   fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\">\n");
   fprintf(fp, "  <RectilinearGrid WholeExtent=\"%d %d %d %d %d %d\">\n", ix_start - ofi,
           ix_start + nx - 1, iy_start - ofj, iy_start + ny - 1, iz_start - ofk,
           iz_start + nz - 1);
   fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", ix_start - ofi,
           ix_start + nx - 1, iy_start - ofj, iy_start + ny - 1, iz_start - ofk,
           iz_start + nz - 1);

   fprintf(fp, "      <Coordinates>\n");
   WriteCoordArray(fp, "x", x_start, x_step, nxg);
   WriteCoordArray(fp, "y", y_start, y_step, nyg);
   WriteCoordArray(fp, "z", z_start, z_step, nzg);
   fprintf(fp, "      </Coordinates>\n");

   /* Point data: vectors */
   fprintf(fp, "      <PointData Vectors=\"displacement\">\n");
   if (params->visualize == 1)
   {
      /* ASCII */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"displacement\" "
                  "NumberOfComponents=\"3\" format=\"ascii\">\n");
      fprintf(fp, "          ");
      for (int k = 0; k < nzg; k++)
      {
         for (int j = 0; j < nyg; j++)
         {
            for (int i = 0; i < nxg; i++)
            {
               int idx = ((k * nyg + j) * nxg + i) * 3;
               fprintf(fp, "%.15g %.15g %.15g ", ext[idx + 0], ext[idx + 1],
                       ext[idx + 2]);
               if (((k * nyg + j) * nxg + i + 1) % 2 == 0) fprintf(fp, "\n          ");
            }
         }
      }
      fprintf(fp, "\n        </DataArray>\n");
      fprintf(fp, "      </PointData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </RectilinearGrid>\n");
   }
   else if (params->visualize == 2)
   {
      /* Appended binary */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"displacement\" "
                  "NumberOfComponents=\"3\" format=\"appended\" offset=\"0\">\n");
      fprintf(fp, "        </DataArray>\n");
      fprintf(fp, "      </PointData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </RectilinearGrid>\n");
      fprintf(fp, "  <AppendedData encoding=\"raw\">\n   _");
      int    npts          = nxg * nyg * nzg;
      size_t data_size     = (size_t)(npts * 3) * sizeof(double);
      int    data_size_int = (int)data_size;
      fwrite(&data_size_int, sizeof(int), 1, fp);
      fwrite(ext, sizeof(double), (size_t)(npts * 3), fp);
      fprintf(fp, "\n  </AppendedData>\n");
   }
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);

   /* PVD file (rank 0) */
   if (!myid)
   {
      snprintf(filename, sizeof(filename), "elasticity_%dx%dx%d_%dx%dx%d.pvd",
               (int)params->N[0], (int)params->N[1], (int)params->N[2], (int)params->P[0],
               (int)params->P[1], (int)params->P[2]);
      fp = fopen(filename, "w");
      if (!fp)
      {
         printf("Error: Cannot open %s\n", filename);
         MPI_Abort(mesh->cart_comm, -1);
      }
      fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
      fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n");
      fprintf(fp, "  <Collection>\n");
      for (int i = 0; i < num_procs; i++)
         fprintf(fp,
                 "    <DataSet part=\"%d\" "
                 "file=\"elasticity_%dx%dx%d_%dx%dx%d_%d.vtr\"/>\n",
                 i, (int)params->N[0], (int)params->N[1], (int)params->N[2],
                 (int)params->P[0], (int)params->P[1], (int)params->P[2], i);
      fprintf(fp, "  </Collection>\n");
      fprintf(fp, "</VTKFile>\n");
      fclose(fp);
   }

   /* Cleanup */
   for (int i = 0; i < 14; i++)
      if (ghost[i]) free(ghost[i]);
   free(ext);

   return 0;
}

/*--------------------------------------------------------------------------
 * Visualize the mixed two-material solution. The mixed solution vector is in the
 * combined, flipped-Y layout, so it cannot be indexed on the structured mesh
 * directly. We scatter the displacement components into a standard interleaved
 * IJ vector (keyed by the ordinary node numbering, letting HYPRE redistribute to
 * the standard partition) and then reuse the structured-grid VTK writer. Pressure
 * is not visualized.
 *--------------------------------------------------------------------------*/
int
WriteVTKMixedSolution(DistMesh *mesh, ElasticParams *params, HYPRE_Real *mixed_sol,
                      MPI_Comm comm)
{
   MixedLayout *L = NULL;
   if (MixedLayoutCreate(mesh, &L)) return 1;

   const HYPRE_Int    cx = mesh->coords[0];
   const HYPRE_Int    cy = mesh->coords[1];
   const HYPRE_Int    cz = mesh->coords[2];
   const HYPRE_Int    lr = ml_lrank(L, cx, cy, cz);
   const HYPRE_BigInt ilower_m = L->offset[lr];
   const HYPRE_BigInt iupper_m = L->offset[lr + 1] - 1;
   const HYPRE_Int   *gdims    = &mesh->gdims[0];
   HYPRE_BigInt     **pstarts  = &mesh->pstarts[0];

   /* Standard interleaved displacement vector over this rank's node range */
   const HYPRE_BigInt d_lo = 3 * mesh->ilower;
   const HYPRE_BigInt d_hi = 3 * mesh->iupper + 2;
   HYPRE_IJVector     u;
   HYPRE_IJVectorCreate(comm, d_lo, d_hi, &u);
   HYPRE_IJVectorSetObjectType(u, HYPRE_PARCSR);
   HYPREDRV_IJ_VECTOR_INIT_HOST(u);

   /* Scatter each mixed-owned node's displacement into the standard numbering */
   for (HYPRE_BigInt gz = L->zstart[cz]; gz < L->zstart[cz + 1]; gz++)
   {
      for (HYPRE_BigInt gx = L->xstart[cx]; gx < L->xstart[cx + 1]; gx++)
      {
         for (HYPRE_BigInt gy = 0; gy < L->Ny; gy++)
         {
            HYPRE_BigInt r0 = ml_udof(L, gx, gy, gz, 0);
            if (r0 < ilower_m || r0 > iupper_m) continue; /* not owned here */

            HYPRE_BigInt ng[3] = {gx, gy, gz};
            HYPRE_Int    bc[3];
            for (int d = 0; d < 3; d++)
            {
               HYPRE_Int     pd = mesh->pdims[d];
               HYPRE_BigInt *ps = mesh->pstarts[d];
               HYPRE_Int     bd = 0;
               while ((bd + 1) < pd && ng[d] >= ps[bd + 1]) bd++;
               bc[d] = bd;
            }
            HYPRE_BigInt nid = grid2idx(ng, bc, gdims, pstarts);
            for (int i = 0; i < 3; i++)
            {
               HYPRE_BigInt sg  = 3 * nid + i;
               HYPRE_Real   val = mixed_sol[r0 - ilower_m + i];
               /* AddToValues (not SetValues): a node's mixed owner (flipped-Y
                  layout) often differs from its standard owner, so this is an
                  off-process contribution. IJ stages and communicates AddTo
                  contributions on Assemble; the vector is zero-initialized and
                  each node is contributed exactly once, so add == set. */
               HYPRE_IJVectorAddToValues(u, 1, &sg, &val);
            }
         }
      }
   }
   HYPRE_IJVectorAssemble(u);

   /* Pull the standard-local 3-interleaved displacement and reuse the writer */
   HYPRE_Int     nloc    = 3 * mesh->local_size;
   HYPRE_Real   *std_sol = (HYPRE_Real *)malloc((size_t)(nloc > 0 ? nloc : 1) *
                                              sizeof(HYPRE_Real));
   HYPRE_BigInt *rows =
      (HYPRE_BigInt *)malloc((size_t)(nloc > 0 ? nloc : 1) * sizeof(HYPRE_BigInt));
   for (HYPRE_Int k = 0; k < nloc; k++) rows[k] = d_lo + k;
   HYPRE_IJVectorGetValues(u, nloc, rows, std_sol);

   WriteVTKsolutionVector(mesh, params, std_sol);

   free(std_sol);
   free(rows);
   HYPRE_IJVectorDestroy(u);
   MixedLayoutDestroy(&L);
   return 0;
}

/*--------------------------------------------------------------------------
 * Near-nullspace rigid body modes
 *--------------------------------------------------------------------------*/
int
ComputeRigidBodyModes(DistMesh *mesh, ElasticParams *params, HYPRE_Real **rbm_ptr)
{
   /* Local sizes and global starts */
   HYPRE_Int     *p       = &mesh->coords[0];
   HYPRE_Int     *nlocal  = &mesh->nlocal[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_Int     *gdims   = &mesh->gdims[0];
   (void)gdims; /* Reserved for future use */
   HYPRE_Real *gsizes = &mesh->gsizes[0];

   const HYPRE_Int    nx  = nlocal[0];
   const HYPRE_Int    ny  = nlocal[1];
   const HYPRE_Int    nz  = nlocal[2];
   const HYPRE_BigInt ix0 = pstarts[0][p[0]];
   const HYPRE_BigInt iy0 = pstarts[1][p[1]];
   const HYPRE_BigInt iz0 = pstarts[2][p[2]];

   /* Physical spacing and domain center */
   const HYPRE_Real dx = params->L[0] * gsizes[0];
   const HYPRE_Real dy = params->L[1] * gsizes[1];
   const HYPRE_Real dz = params->L[2] * gsizes[2];
   const HYPRE_Real cx = 0.5 * params->L[0];
   const HYPRE_Real cy = 0.5 * params->L[1];
   const HYPRE_Real cz = 0.5 * params->L[2];

   const HYPRE_Int num_nodes_local = nx * ny * nz;
   const HYPRE_Int dofs_per_node   = 3;
   const HYPRE_Int num_modes       = 6; /* 3 translations + 3 rotations */
   const HYPRE_Int stride_mode     = dofs_per_node * num_nodes_local;

   HYPRE_Real *rbm =
      (HYPRE_Real *)calloc((size_t)num_modes * (size_t)stride_mode, sizeof(HYPRE_Real));
   for (HYPRE_Int k = 0; k < nz; k++)
   {
      const HYPRE_BigInt gz = iz0 + k;
      const HYPRE_Real   z  = (HYPRE_Real)gz * dz;
      for (HYPRE_Int j = 0; j < ny; j++)
      {
         const HYPRE_BigInt gy = iy0 + j;
         const HYPRE_Real   y  = (HYPRE_Real)gy * dy;
         for (HYPRE_Int i = 0; i < nx; i++)
         {
            const HYPRE_BigInt gx = ix0 + i;
            const HYPRE_Real   x  = (HYPRE_Real)gx * dx;

            const HYPRE_Int nidx   = (k * ny + j) * nx + i; /* local node index */
            const HYPRE_Int base   = dofs_per_node * nidx;  /* dof base within a mode */
            const HYPRE_Int off_Tx = 0 * stride_mode;
            const HYPRE_Int off_Ty = 1 * stride_mode;
            const HYPRE_Int off_Tz = 2 * stride_mode;
            const HYPRE_Int off_Rx = 3 * stride_mode;
            const HYPRE_Int off_Ry = 4 * stride_mode;
            const HYPRE_Int off_Rz = 5 * stride_mode;

            /* Relative position to domain center */
            const HYPRE_Real rx = x - cx;
            const HYPRE_Real ry = y - cy;
            const HYPRE_Real rz = z - cz;

            /* Translations */
            rbm[off_Tx + base + 0] = 1.0;
            rbm[off_Tx + base + 1] = 0.0;
            rbm[off_Tx + base + 2] = 0.0;
            rbm[off_Ty + base + 0] = 0.0;
            rbm[off_Ty + base + 1] = 1.0;
            rbm[off_Ty + base + 2] = 0.0;
            rbm[off_Tz + base + 0] = 0.0;
            rbm[off_Tz + base + 1] = 0.0;
            rbm[off_Tz + base + 2] = 1.0;

            /* Rotations: u = w x r, with r = (rx, ry, rz) about the domain center */
            /* Rx: w=(1,0,0) -> (0, -rz,  ry) */
            rbm[off_Rx + base + 0] = 0.0;
            rbm[off_Rx + base + 1] = -rz;
            rbm[off_Rx + base + 2] = ry;

            /* Ry: w=(0,1,0) -> ( rz, 0, -rx) */
            rbm[off_Ry + base + 0] = rz;
            rbm[off_Ry + base + 1] = 0.0;
            rbm[off_Ry + base + 2] = -rx;

            /* Rz: w=(0,0,1) -> (-ry, rx, 0) */
            rbm[off_Rz + base + 0] = -ry;
            rbm[off_Rz + base + 1] = rx;
            rbm[off_Rz + base + 2] = 0.0;

            /* Enforce clamp at x=0 by zeroing modes on clamped plane */
            if (gx == 0)
            {
               for (HYPRE_Int m = 0; m < num_modes; m++)
               {
                  HYPRE_Int moff = m * stride_mode + base;
                  rbm[moff + 0]  = 0.0;
                  rbm[moff + 1]  = 0.0;
                  rbm[moff + 2]  = 0.0;
               }
            }
         }
      }
   }

   *rbm_ptr = rbm;
   return 0;
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/
int
main(int argc, char *argv[])
{
   MPI_Comm       comm = MPI_COMM_WORLD;
   HYPREDRV_t     hypredrv;
   int            myid, num_procs;
   ElasticParams  params;
   DistMesh      *mesh;
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJMatrix M_prec     = NULL;
   HYPRE_IJMatrix S_coarse   = NULL;
   HYPRE_Real    *sol_data;
   HYPRE_Real    *rbms       = NULL;
   int           *dofmap     = NULL;
   HYPRE_Int      local_rows = 0;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   if (ParseArguments(argc, argv, &params, myid, num_procs))
   {
      MPI_Finalize();
      return 1;
   }


   /* Initialize hypredrive */
   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
   HYPREDRV_SAFE_CALL(RegisterExamplePreconPresets());

   /* Print library info if requested */
   if (params.verbose & 0x1)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   }

   /* Print system info if requested */
   if (params.verbose & 0x2)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   /* Create hypredrive object */
   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

   /* Configure solver:
       - two-material MIXED: symmetric indefinite saddle point -> built-in
         FGMRES + MGR configuration (or a user file via -i).
       - two-material STANDARD CG: SPD; the config file (e.g. amg-gmres.yml) is
         authoritative, or PCG + AMG preset if none is given.
       - single material: SPD; PCG + AMG preset, optionally tweaked by -i. */
   if (params.problem && params.discretization == 0)
   {
#if HYPREDRV_HYPRE_RELEASE_NUMBER < 30100
      if (!params.yaml_file)
      {
         if (!myid)
            printf("Error: the built-in two-material solver configuration requires "
                   "hypre >= 3.1.0; provide a configuration file with -i instead\n");
         HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
         HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
         MPI_Finalize();
         return 1;
      }
#endif
      char *args[2] = {
         params.yaml_file ? params.yaml_file : (char *)default_config_mixed, NULL};
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));
   }
   else if (params.problem && params.yaml_file)
   {
      char *args[2] = {params.yaml_file, NULL};
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));
   }
   else
   {
      if (params.yaml_file)
      {
         /* A user config fully specifies solver+preconditioner; do not also apply
            the built-in precon preset (it would override e.g. nodal coarsening). */
         char *args[2] = {params.yaml_file, NULL};
         HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));
      }
      else
      {
         HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetSolverPreset(hypredrv, "pcg"));
         HYPREDRV_SAFE_CALL(
            HYPREDRV_InputArgsSetPreconPreset(hypredrv, params.solver_preset));
      }
   }

   /* Set HYPREDRV global options */

   if (!myid)
   {
      printf("\n");
      printf("=====================================================\n");
      printf("          Linear Elasticity Problem Setup\n");
      printf("=====================================================\n");
      printf("Physical dimensions:     %d x %d x %d\n", (int)params.L[0],
             (int)params.L[1], (int)params.L[2]);
      printf("Grid dimensions (nodes): %d x %d x %d\n", (int)params.N[0],
             (int)params.N[1], (int)params.N[2]);
      printf("Processor topology:      %d x %d x %d\n", (int)params.P[0],
             (int)params.P[1], (int)params.P[2]);
      if (params.problem)
      {
         const char *disc_name =
            (params.discretization == 0) ? "mixed u-p (Q1-Q1) top + CG bottom"
            : (params.discretization == 2)
               ? "B-bar Q1-P0 mean-dilatation (displacement only)"
               : "standard CG Q1 (displacement only)";
         printf("Problem:                 Two-Material Bar (interface at y=Ly/2)\n");
         printf("Discretization:          %s\n", disc_name);
         printf("Bottom material:         E=%.1e, nu=%.4f\n", params.E, params.nu);
         printf("Top material:            E=%.1e, nu=%.4f\n", params.E_top,
                params.nu_top);
      }
      else
      {
         printf("Material:                E=%.1e, nu=%.3f\n", params.E, params.nu);
      }
      printf("Body force:              rho=%.1e, g=(%.1f, %.1f, %.1f)\n", params.rho,
             params.g[0], params.g[1], params.g[2]);
      if (params.problem && params.discretization == 0)
         printf("Solver:                  FGMRES + MGR (u-p saddle point)\n");
      else if (params.yaml_file)
         printf("Solver:                  from %s\n", params.yaml_file);
      else printf("Solver preset:           %s\n", params.solver_preset);
      if (params.traction_top)
         printf("Top traction:            t=(%.1e, %.1e, %.1e)\n", params.traction[0],
                params.traction[1], params.traction[2]);
      else printf("Top traction:            none\n");
      printf("Visualization:           %s\n", params.visualize ? "true" : "false");
      printf("Verbosity level:         0x%x\n", params.verbose);
      printf("Number of solves:        %d\n", params.nsolve);
      printf("=====================================================\n\n");
   }

   CreateDistMesh(comm, params.N[0], params.N[1], params.N[2], params.P[0], params.P[1],
                  params.P[2], &mesh);

   /* Mixed u-p path (saddle point) vs standard CG path (SPD, interleaved dofs).
      The standard CG path serves both the single-material problem and the
      two-material bar with --discretization standard. */
   const int use_mixed = (params.problem && params.discretization == 0);

   if (!myid && (params.verbose > 0)) printf("Assembling linear system...");
   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(hypredrv, "system", -1));
   if (use_mixed)
   {
      BuildMixedTwoMaterialSystem(mesh, &params, &A, &b, &dofmap, &local_rows, &rbms,
                                  params.prec_mass_schur ? &M_prec : NULL,
                                  params.coarse_schur ? &S_coarse : NULL, comm);
   }
   else
   {
      BuildElasticitySystem_Q1Hex(mesh, &params, &A, &b, comm);
      ComputeRigidBodyModes(mesh, &params, &rbms);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(hypredrv, "system", -1));
   if (!myid && (params.verbose > 0)) printf(" Done!\n");

#if defined(HYPRE_USING_GPU)
   if (!myid && (params.verbose > 0)) printf("Migrating linear system to GPU...");
   HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   HYPRE_IJVectorMigrate(b, HYPRE_MEMORY_DEVICE);
   if (!myid && (params.verbose > 0)) printf(" Done!\n");
#endif

   if (use_mixed)
   {
      /* Mixed u-p: explicit 7-label dofmap drives the MGR field split. The
         rigid-body modes (zero on pressure rows) are supplied for the
         displacement block, consumed by AMG under nodal coarsening. */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
      HYPREDRV_SAFE_CALL(
         HYPREDRV_LinearSystemSetDofmap(hypredrv, (int)local_rows, dofmap));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(
         hypredrv, params.prec_mass_schur ? (HYPRE_Matrix)M_prec : NULL));
      if (params.coarse_schur)
      {
         /* Provide the level-0 coarse (Schur) operator. Ownership transfers to
            hypredrive, so S_coarse must NOT be destroyed by the application. */
         HYPREDRV_SAFE_CALL(
            HYPREDRV_LinearSystemSetCoarseSchur(hypredrv, 0, (HYPRE_Matrix)S_coarse));
         S_coarse = NULL;
      }
      HYPREDRV_SAFE_CALL(
         HYPREDRV_LinearSystemSetNearNullSpace(hypredrv, (int)local_rows, 6, rbms));
   }
   else
   {
      /* Single material: 3 interleaved dofs per node + rigid-body modes */
      HYPREDRV_SAFE_CALL(
         HYPREDRV_LinearSystemSetInterleavedDofmap(hypredrv, mesh->local_size, 3));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
      HYPREDRV_SAFE_CALL(
         HYPREDRV_LinearSystemSetNearNullSpace(hypredrv, 3 * mesh->local_size, 6, rbms));
   }

   if (params.verbose & 0x4)
   {
      if (!myid) printf("Printing linear system...");
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemPrint(hypredrv));
      if (!myid) printf(" Done!\n");
   }

   for (int isolve = 0; isolve < params.nsolve; isolve++)
   {
      if (!myid) printf("Solve %d/%d...\n", isolve + 1, params.nsolve);

      /* (Optional) Annotate the entire solve iteration */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(hypredrv, "Run", isolve));

      /* Build and apply linear solver */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));

      /* (Optional) Annotate the entire solve iteration */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(hypredrv, "Run", isolve));
   }

   if (!myid && (params.verbose & 0x1)) HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));

   if (params.visualize)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol_data));
      if (use_mixed)
         WriteVTKMixedSolution(mesh, &params, sol_data, comm);
      else
         WriteVTKsolutionVector(mesh, &params, sol_data);
   }

   /* Free memory */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   if (M_prec) HYPRE_IJMatrixDestroy(M_prec);
   if (S_coarse) HYPRE_IJMatrixDestroy(S_coarse);
   DestroyDistMesh(&mesh);
   free(rbms);
   free(dofmap);

   if (params.verbose & 0x2) HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}
