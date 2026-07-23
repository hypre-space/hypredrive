/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <math.h> // fabs
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "HYPREDRV.h"
#include "HYPREDRV_utils.h"

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
 *   3D Convection-Diffusion in a Duct (upwind finite volumes)
 *==========================================================================
 *
 *   Grid Partitioning:
 *   ------------------
 *                   P[2]
 *                    ↑     Processor Grid: P[0] × P[1] × P[2]
 *                    |   /  Each proc owns a local block
 *                    |  /   of size n[0] × n[1] × n[2] cells
 *                    | /
 *            P[1] ←--C
 *                   /
 *                  /
 *                 /
 *                ↙ P[0]
 *
 *   PDE Problem:
 *   ------------
 *   ∂c/∂t + ∇·(v c) - κ ∇²c = 0    in Ω = [0,L] × [0,H] × [0,H]
 *
 *   with a prescribed, divergence-free duct flow along the x-axis
 *
 *      v(y,z) = (u(y,z), -∂ψ/∂z, ∂ψ/∂y),
 *      u(y,z) = 16 u_max (y/H) (1 - y/H) (z/H) (1 - z/H)
 *
 *   The optional cross-plane stream function
 *
 *      ψ(y,z) = a G(y/H) G(z/H),   G(t) = t² (1 - t)²
 *
 *   superposes a wall-tangential swirl (-w) on the axial flow, turning the
 *   straight duct into a helical one. Since G vanishes at the walls, the
 *   transverse flow is divergence-free *and* leaves the no-flux boundaries
 *   intact. The swirl closes the characteristics in the y-z plane and makes
 *   the linear systems harder for every preconditioner (single-level ones
 *   most of all).
 *
 *   Boundary and initial conditions:
 *
 *      c = 1                 on x = 0        (inlet)
 *      κ ∂c/∂x = 0           on x = L        (convective outflow)
 *      κ ∂c/∂n = 0           on the walls    (no flux)
 *      c = 0                 at t = 0
 *
 *   Discretization:
 *   ---------------
 *   Cell-centered finite volumes on a uniform Cartesian grid:
 *     - diffusion:  two-point central fluxes (7-point stencil)
 *     - convection: first-order upwind fluxes (nonsymmetric couplings)
 *     - time:       backward Euler, geometrically growing time steps
 *
 *   Each step assembles a *new* linear system, since the diagonal carries
 *   V/Δt with Δt_n = Δt_0 g^n. The resulting operator is a nonsymmetric
 *   M-matrix whose diagonal dominance weakens as the time step grows, which
 *   makes it a compact showcase for preconditioners tailored to convection-
 *   dominated problems (AMG with approximate ideal restriction, ILU).
 *
 *   Usage: ${MPIEXEC_COMMAND} <np> ./convdif [options]
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Problem parameters struct
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int  visualize;     /* Visualize solution in VTK format */
   HYPRE_Int  verbose;       /* Verbosity level bitset */
   HYPRE_Int  N[3];          /* Grid dimensions (number of cells) */
   HYPRE_Int  P[3];          /* Processor grid dimensions */
   HYPRE_Real dom[3];        /* Domain lengths (L, H, H) */
   HYPRE_Real kappa;         /* Diffusion coefficient */
   HYPRE_Real umax;          /* Peak axial velocity */
   HYPRE_Real wmax;          /* Peak transverse (swirl) velocity */
   HYPRE_Int  nt;            /* Number of time steps */
   HYPRE_Real dt0;           /* Initial time step size */
   HYPRE_Real dt_growth;     /* Time step growth factor */
   char      *yaml_file;     /* YAML configuration file */
   HYPRE_Int  hypredrv_argc; /* Number of hypredrive override args (incl. -a) */
   char     **hypredrv_argv; /* Hypredrive override args, starting at -a */
} ProblemParams;

/*--------------------------------------------------------------------------
 * Distributed Mesh struct
 *--------------------------------------------------------------------------*/
typedef struct
{
   MPI_Comm      cart_comm;  /* Cartesian communicator */
   HYPRE_Int     mypid;      /* Local partition ID (rank ID) */
   HYPRE_Int     gdims[3];   /* Global dimensions (nx, ny, nz) */
   HYPRE_Int     pdims[3];   /* Processor grid dimensions (Px, Py, Pz) */
   HYPRE_Int     coords[3];  /* Process coordinates in processor grid */
   HYPRE_Int     nlocal[3];  /* Local dimensions */
   HYPRE_Int     local_size; /* Local problem size */
   HYPRE_BigInt  ilower;     /* Lower bound of local rows */
   HYPRE_BigInt  iupper;     /* Upper bound of local rows */
   HYPRE_BigInt *pstarts[3]; /* Partition prefix sums for each dimension */
   HYPRE_Real    hsizes[3];  /* Cell sizes (hx, hy, hz) */
} DistMesh;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

int PrintUsage(void);
int CreateDistMesh(MPI_Comm, ProblemParams *, DistMesh **);
int DestroyDistMesh(DistMesh **);
int ParseArguments(int, char **, ProblemParams *, int, int);
int BuildSystem(DistMesh *, ProblemParams *, HYPRE_Real *, HYPRE_Real, HYPRE_IJMatrix *,
                HYPRE_IJVector *, HYPRE_IJVector *);
int WriteVTKsolution(DistMesh *, ProblemParams *, HYPRE_Real *, HYPRE_Int);
int WriteVTKseries(DistMesh *, ProblemParams *, HYPRE_Int, const HYPRE_Real *);

/*--------------------------------------------------------------------------
 * Axial velocity of the prescribed duct flow at a point (y, z)
 *--------------------------------------------------------------------------*/
static inline HYPRE_Real
AxialVelocity(HYPRE_Real y, HYPRE_Real z, HYPRE_Real Hy, HYPRE_Real Hz, HYPRE_Real umax)
{
   return 16.0 * umax * (y / Hy) * (1.0 - y / Hy) * (z / Hz) * (1.0 - z / Hz);
}

/*--------------------------------------------------------------------------
 * Shape function G(t) = t² (1 - t)² of the cross-plane stream function, and
 * the amplitude that turns a requested peak transverse velocity into the
 * stream function scale: max|∇ψ| = max(G) max(G') |a| / H, with
 * max(G) = 1/16 at t = 1/2 and max|G'| = 2 t (1-t) (1-2t) at t = (3-√3)/6.
 *--------------------------------------------------------------------------*/
static inline HYPRE_Real
SwirlShape(HYPRE_Real s, HYPRE_Real len)
{
   const HYPRE_Real t = s / len;
   return t * t * (1.0 - t) * (1.0 - t);
}

static const HYPRE_Real SWIRL_GRAD_MAX = 0.0120281306; /* max(G) * max|G'| = 1/(48*sqrt(3)) */

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/
int
PrintUsage(void)
{
   printf("\n");
   printf("Usage: ${MPIEXEC_COMMAND} <np> ./convdif [options]\n");
   printf("\n");
   printf("Options:\n");
   printf(
      "  -i <file>          : YAML configuration file for solver settings (Optional)\n");
   printf("  -a|--args ...      : Hypredrive YAML overrides, e.g. -a "
          "--solver:gmres:max_iter 100\n");
   printf("                       (requires -i; must come last)\n");
   printf("  -n <nx> <ny> <nz>  : Global grid dimensions in cells (64 16 16)\n");
   printf("  -P <Px> <Py> <Pz>  : Processor grid dimensions (1 1 1)\n");
   printf("  -L <val>           : Duct length along x (4.0)\n");
   printf("  -H <val>           : Duct height/width along y and z (1.0)\n");
   printf("  -k|--kappa <val>   : Diffusion coefficient (1.0e-3)\n");
   printf("  -u|--umax <val>    : Peak axial velocity (1.0)\n");
   printf("  -w|--swirl <val>   : Peak transverse (swirl) velocity (0.0)\n");
   printf("  -nt <n>            : Number of time steps (10)\n");
   printf("  -dt <val>          : Initial time step size (0.1)\n");
   printf("  -dtg|--dt-growth <val> : Time step growth factor (1.5)\n");
   printf("  -vis [<m>]         : Write the solution in VTK format (1)\n");
   printf("                          1: final state only\n");
   printf("                          2: every time step (time series .pvd)\n");
   printf("  -v|--verbose <n>   : Verbosity bitset (3)\n");
   printf("                          0x1: Library info and linear solver statistics\n");
   printf("                          0x2: System information\n");
   printf("                          0x4: Linear system printing\n");
   printf("  -h|--help          : Print this message\n");
   printf("\n");

   return 0;
}

/*--------------------------------------------------------------------------
 * Parse command line arguments
 *--------------------------------------------------------------------------*/
int
ParseArguments(int argc, char *argv[], ProblemParams *params, int myid, int num_procs)
{
   /* Set defaults */
   params->visualize = 0;
   params->verbose   = 3;
   params->N[0]      = 64;
   params->N[1]      = 16;
   params->N[2]      = 16;
   for (int i = 0; i < 3; i++)
   {
      params->P[i] = 1;
   }
   params->dom[0]        = 4.0;
   params->dom[1]        = 1.0;
   params->dom[2]        = 1.0;
   params->kappa         = 1.0e-3;
   params->umax          = 1.0;
   params->wmax          = 0.0;
   params->nt            = 10;
   params->dt0           = 0.1;
   params->dt_growth     = 1.5;
   params->yaml_file     = NULL;
   params->hypredrv_argc = 0;
   params->hypredrv_argv = NULL;

   /* Parse command line */
   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))
      {
         if (++i < argc) params->yaml_file = argv[i];
      }
      else if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--args"))
      {
         params->hypredrv_argc = argc - i;
         params->hypredrv_argv = argv + i;
         break;
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
         if (++i < argc) params->dom[0] = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-H"))
      {
         if (++i < argc)
         {
            params->dom[1] = params->dom[2] = atof(argv[i]);
         }
      }
      else if (!strcmp(argv[i], "-k") || !strcmp(argv[i], "--kappa"))
      {
         if (++i < argc) params->kappa = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-u") || !strcmp(argv[i], "--umax"))
      {
         if (++i < argc) params->umax = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--swirl"))
      {
         if (++i < argc) params->wmax = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-nt"))
      {
         if (++i < argc) params->nt = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-dt"))
      {
         if (++i < argc) params->dt0 = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-dtg") || !strcmp(argv[i], "--dt-growth"))
      {
         if (++i < argc) params->dt_growth = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-vis") || !strcmp(argv[i], "--visualize"))
      {
         params->visualize = 1;
         if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
         {
            params->visualize = atoi(argv[++i]);
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

   /* Overrides need a YAML configuration to apply to */
   if (params->hypredrv_argc && !params->yaml_file)
   {
      if (!myid) printf("Error: -a/--args requires a YAML configuration file (-i)\n");
      return 1;
   }

   /* Verify processor grid entries are positive */
   if (params->P[0] < 1 || params->P[1] < 1 || params->P[2] < 1)
   {
      if (!myid)
      {
         printf("Error: -P entries must be >= 1 (got %d x %d x %d)\n",
                params->P[0], params->P[1], params->P[2]);
      }
      return 1;
   }

   /* Verify processor grid matches total number of processes */
   if (params->P[0] * params->P[1] * params->P[2] != num_procs)
   {
      if (!myid)
      {
         printf("Error: Number of processes (%d) must match processor grid dimensions "
                "(%d x %d x %d = %d)\n",
                num_procs, params->P[0], params->P[1], params->P[2],
                params->P[0] * params->P[1] * params->P[2]);
      }
      return 1;
   }

   /* Verify the global grid can be partitioned */
   for (int d = 0; d < 3; d++)
   {
      static const char *name[] = {"First", "Second", "Third"};
      if (params->P[d] > params->N[d])
      {
         if (!myid)
         {
            printf("Error: %s grid dimension (N = %d) must be at least as large as the "
                   "number of ranks (P = %d)\n",
                   name[d], params->N[d], params->P[d]);
         }
         return 1;
      }
   }

   /* Verify domain and physical parameters */
   if (params->dom[0] <= 0.0 || params->dom[1] <= 0.0 || params->dom[2] <= 0.0 ||
       params->kappa < 0.0)
   {
      if (!myid)
      {
         printf("Error: require -L > 0, -H > 0, and -k >= 0\n");
      }
      return 1;
   }

   /* Verify time stepping parameters */
   if (params->nt < 1 || params->dt0 <= 0.0 || params->dt_growth < 1.0)
   {
      if (!myid)
      {
         printf("Error: require -nt >= 1, -dt > 0, and -dtg >= 1\n");
      }
      return 1;
   }

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
   ProblemParams  params;
   DistMesh      *mesh;
   HYPRE_IJMatrix A;
   HYPRE_IJVector b, x0;
   HYPRE_Real    *sol_data;
   HYPRE_Real    *c;
   HYPRE_Real     time = 0.0, dt;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   /* Parse command line arguments */
   if (ParseArguments(argc, argv, &params, myid, num_procs))
   {
      MPI_Finalize();
      return 1;
   }

   /* Initialize hypredrive */
   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());

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

   /* Configure solver using YAML input or default presets */
   if (params.yaml_file)
   {
      HYPRE_Int hypredrv_argc = 1 + params.hypredrv_argc;
      char     *hypredrv_argv[hypredrv_argc];
      hypredrv_argv[0] = params.yaml_file;
      for (HYPRE_Int k = 0; k < params.hypredrv_argc; k++)
      {
         hypredrv_argv[k + 1] = params.hypredrv_argv[k];
      }
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(hypredrv_argc, hypredrv_argv, hypredrv));
   }
   else
   {
      /* The discrete operator is nonsymmetric, hence the Krylov default is GMRES */
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetSolverPreset(hypredrv, "gmres"));
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(hypredrv, "poisson"));
   }

   /* Create distributed mesh object */
   CreateDistMesh(comm, &params, &mesh);

   /* Print problem parameters */
   if (!myid && (params.verbose & 0x1))
   {
      printf("\n");
      printf("=====================================================\n");
      printf("        Convection-Diffusion Problem Setup\n");
      printf("=====================================================\n");
      printf("Grid dimensions:      %d x %d x %d cells\n", (int)params.N[0],
             (int)params.N[1], (int)params.N[2]);
      printf("Processor topology:   %d x %d x %d\n", (int)params.P[0], (int)params.P[1],
             (int)params.P[2]);
      printf("Domain sizes:         %.2f x %.2f x %.2f\n", params.dom[0], params.dom[1],
             params.dom[2]);
      printf("Diffusion coeff:      %.2e\n", params.kappa);
      printf("Peak axial velocity:  %.2e\n", params.umax);
      printf("Peak swirl velocity:  %.2e\n", params.wmax);
      printf("Cell Peclet number:   %.2e\n",
             params.umax * mesh->hsizes[0] / params.kappa);
      printf("Initial CFL number:   %.2e\n", params.umax * params.dt0 / mesh->hsizes[0]);
      printf("Time steps:           %d\n", (int)params.nt);
      printf("Initial step size:    %.2e\n", params.dt0);
      printf("Step growth factor:   %.2f\n", params.dt_growth);
      printf("Visualization:        %s\n", params.visualize ? "true" : "false");
      printf("Verbosity level:      0x%x\n", params.verbose);
      printf("=====================================================\n\n");
   }

   /* Initial condition: c = 0 everywhere */
   c = (HYPRE_Real *)calloc((size_t)mesh->local_size, sizeof(HYPRE_Real));

   /* In time-series mode, record frame times and write the initial state */
   HYPRE_Real *frame_times = NULL;
   if (params.visualize > 1)
   {
      frame_times    = (HYPRE_Real *)calloc((size_t)params.nt + 1, sizeof(HYPRE_Real));
      frame_times[0] = 0.0;
      WriteVTKsolution(mesh, &params, c, 0);
   }

   /* Time stepping loop. Each step assembles and solves a new linear system */
   dt = params.dt0;
   for (HYPRE_Int t_step = 0; t_step < params.nt; t_step++)
   {
      if (t_step > 0)
      {
         dt *= params.dt_growth;
      }
      time += dt;

      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(hypredrv, 0, "timestep", t_step));

      /* Assemble the backward Euler system for this time step. The "system"
       * annotation tells hypredrive that a new linear system starts here */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(hypredrv, "system", -1));
      BuildSystem(mesh, &params, c, dt, &A, &b, &x0);
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(hypredrv, "system", -1));

      /* Transfer data to GPU memory */
#if defined(HYPRE_USING_GPU)
      HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
      HYPRE_IJVectorMigrate(b, HYPRE_MEMORY_DEVICE);
      HYPRE_IJVectorMigrate(x0, HYPRE_MEMORY_DEVICE);
#endif

      /* Associate the matrix and vectors with hypredrive. The solution at the
       * previous time step is used as initial guess for the current one */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
      HYPREDRV_SAFE_CALL(
         HYPREDRV_LinearSystemSetInitialGuess(hypredrv, (HYPRE_Vector)x0));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));

      /* Print linear system if requested */
      if (params.verbose & 0x4)
      {
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemPrint(hypredrv));
      }

      /* Build and apply linear solver */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));

      /* Update the concentration field with the new solution */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol_data));
      memcpy(c, sol_data, (size_t)mesh->local_size * sizeof(HYPRE_Real));

      /* Report time step information */
      if (params.verbose & 0x1)
      {
         HYPRE_Real cell_volume = mesh->hsizes[0] * mesh->hsizes[1] * mesh->hsizes[2];
         HYPRE_Real local[3]    = {c[0], c[0], 0.0};
         HYPRE_Real global[3];
         int        num_iterations = 0;

         for (HYPRE_Int i = 0; i < mesh->local_size; i++)
         {
            local[0] = (c[i] < local[0]) ? c[i] : local[0];
            local[1] = (c[i] > local[1]) ? c[i] : local[1];
            local[2] += c[i] * cell_volume;
         }
         MPI_Allreduce(&local[0], &global[0], 1, MPI_DOUBLE, MPI_MIN, mesh->cart_comm);
         MPI_Allreduce(&local[1], &global[1], 1, MPI_DOUBLE, MPI_MAX, mesh->cart_comm);
         MPI_Allreduce(&local[2], &global[2], 1, MPI_DOUBLE, MPI_SUM, mesh->cart_comm);
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverGetNumIter(hypredrv, &num_iterations));

         if (!myid)
         {
            printf("Time step: %3d | Time: %.4e | dt: %.4e | CFL: %8.2f | Lin: %3d | "
                   "min(c)=%9.2e max(c)=%9.2e mass=%.6e\n",
                   (int)t_step + 1, time, dt, params.umax * dt / mesh->hsizes[0],
                   num_iterations, global[0], global[1], global[2]);
         }
      }

      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));

      /* Free the linear system of this time step */
      HYPRE_IJMatrixDestroy(A);
      HYPRE_IJVectorDestroy(b);
      HYPRE_IJVectorDestroy(x0);

      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(hypredrv, 0, "timestep", t_step));

      /* Write the current state as a time-series frame */
      if (params.visualize > 1)
      {
         frame_times[t_step + 1] = time;
         WriteVTKsolution(mesh, &params, c, t_step + 1);
      }
   }

   /* Print per-timestep and per-solve statistics if requested */
   if (!myid && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsLevelPrint(hypredrv, 0));
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
   }

   /* Visualization phase: single snapshot of the final state, or the time
    * series collection tying together the per-step frames */
   if (params.visualize == 1)
   {
      WriteVTKsolution(mesh, &params, c, -1);
   }
   else if (params.visualize > 1)
   {
      WriteVTKseries(mesh, &params, params.nt, frame_times);
   }

   /* Clean up */
   free(frame_times);
   free(c);
   DestroyDistMesh(&mesh);

   /* Print exit info if requested */
   if (params.verbose & 0x2)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));
   }

   /* Finalize */
   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}

/*--------------------------------------------------------------------------
 * Compute global index from global grid coordinates
 *   - gcoords: global cell coordinates
 *   - bcoords: block partition coordinates
 *   - gdims:   global grid dimensions
 *   - pstarts: block partition prefixed sums for each dimension
 *--------------------------------------------------------------------------*/
static inline HYPRE_BigInt
grid2idx(const HYPRE_BigInt gcoords[3], const HYPRE_Int bcoords[3],
         const HYPRE_Int gdims[3], HYPRE_BigInt **pstarts)
{
   return pstarts[2][bcoords[2]] * gdims[0] * gdims[1] +
          pstarts[1][bcoords[1]] * gdims[0] *
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
 * Create mesh partition information
 *--------------------------------------------------------------------------*/
int
CreateDistMesh(MPI_Comm comm, ProblemParams *params, DistMesh **mesh_ptr)
{
   DistMesh *mesh = (DistMesh *)malloc(sizeof(DistMesh));
   int       myid;

   /* Store dimensions */
   for (int i = 0; i < 3; i++)
   {
      mesh->gdims[i] = params->N[i];
      mesh->pdims[i] = params->P[i];
   }

   /* Use int arrays for MPI calls to avoid type mismatch if HYPRE_Int != int */
   int mpi_dims[3]   = {(int)mesh->pdims[0], (int)mesh->pdims[1], (int)mesh->pdims[2]};
   int mpi_coords[3] = {0, 0, 0};

   /* Create cartesian communicator */
   /* Keep cart ranks consistent with MPI_COMM_WORLD ranks (avoid reorder=1). */
   MPI_Cart_create(comm, 3, mpi_dims, (int[]){0, 0, 0}, 0, &(mesh->cart_comm));
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Cart_coords(mesh->cart_comm, myid, 3, mpi_coords);

   /* Copy to HYPRE_Int members */
   mesh->coords[0] = mpi_coords[0];
   mesh->coords[1] = mpi_coords[1];
   mesh->coords[2] = mpi_coords[2];

   /* Store process rank */
   mesh->mypid = (HYPRE_Int)myid;

   /* Compute partition prefix sums for each dimension, local dimensions, and cell
    * sizes. Note that the unknowns live at cell centers, hence h = domain / cells */
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
      mesh->hsizes[i] = params->dom[i] / mesh->gdims[i];
   }

   /* Compute local matrix bounds */
   HYPRE_BigInt gcoords[3] = {mesh->pstarts[0][mesh->coords[0]],
                              mesh->pstarts[1][mesh->coords[1]],
                              mesh->pstarts[2][mesh->coords[2]]};
   mesh->ilower            = grid2idx(gcoords, mesh->coords, mesh->gdims, mesh->pstarts);
   mesh->local_size        = mesh->nlocal[0] * mesh->nlocal[1] * mesh->nlocal[2];
   mesh->iupper            = mesh->ilower + mesh->local_size - 1;

   /* Set output pointer */
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
   for (int i = 0; i < 3; i++)
   {
      free(mesh->pstarts[i]);
   }
   free(mesh);
   *mesh_ptr = NULL;

   return 0;
}

/*--------------------------------------------------------------------------
 * Assemble the backward Euler system for one time step
 *
 * Integrating the PDE over a cell of volume V and applying the divergence
 * theorem gives, for every face f of the cell,
 *
 *    V (c_P - c_P^n) / dt + Σ_f [ (m_f)^+ c_P - (m_f)^- c_N ] +
 *                           Σ_f [ D_f (c_P - c_N) ] = 0
 *
 * where m_f is the outward mass flux through f, (·)^± = max(±·, 0) selects
 * the upwind cell, and D_f = κ A_f / h_f is the two-point diffusive
 * conductance. The axial flow is x-independent, so both x-faces share the
 * same coefficient Cf = u(y,z) A_x. The transverse fluxes are evaluated as
 * differences of the stream function rather than by sampling the velocity,
 *
 *    m_y(j+1/2) = -a h_x G(y_{j+1/2}) [G(z_{k+1/2}) - G(z_{k-1/2})]
 *    m_z(k+1/2) = +a h_x [G(y_{j+1/2}) - G(y_{j-1/2})] G(z_{k+1/2})
 *
 * which makes the discrete face fluxes sum to zero in every cell *exactly*,
 * so the swirl neither creates nor destroys mass at the discrete level.
 *
 * Boundary faces:
 *   - inlet  (x = 0): Dirichlet c = 1 imposed through a half-cell distance,
 *     i.e. a conductance 2 D_x, plus the incoming convective flux
 *   - outlet (x = L): zero diffusive flux, purely convective outflow
 *   - walls:          no flux, i.e. the face is simply skipped
 *
 * Interior rows sum to V/dt, which makes the operator a nonsymmetric
 * M-matrix that conserves mass up to boundary fluxes.
 *--------------------------------------------------------------------------*/
int
BuildSystem(DistMesh *mesh, ProblemParams *params, HYPRE_Real *c_old, HYPRE_Real dt,
            HYPRE_IJMatrix *A_ptr, HYPRE_IJVector *b_ptr, HYPRE_IJVector *x0_ptr)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b, x0;
   HYPRE_Int      local_size = mesh->local_size;
   HYPRE_Int     *n          = &mesh->nlocal[0];
   HYPRE_Int     *p          = &mesh->coords[0];
   HYPRE_Int     *gdims      = &mesh->gdims[0];
   HYPRE_BigInt **pstarts    = &mesh->pstarts[0];
   HYPRE_BigInt   ilower     = mesh->ilower;
   HYPRE_BigInt   iupper     = mesh->iupper;

   /* Cell metrics and constant face coefficients */
   const HYPRE_Real hx   = mesh->hsizes[0];
   const HYPRE_Real hy   = mesh->hsizes[1];
   const HYPRE_Real hz   = mesh->hsizes[2];
   const HYPRE_Real vol  = hx * hy * hz;
   const HYPRE_Real Dx   = params->kappa * hy * hz / hx;
   const HYPRE_Real Dy   = params->kappa * hx * hz / hy;
   const HYPRE_Real Dz   = params->kappa * hx * hy / hz;
   const HYPRE_Real Ax   = hy * hz;
   const HYPRE_Real c_in = 1.0;

   /* Stream function amplitude realizing the requested peak swirl velocity.
      The calibration assumes a square cross-section (dom[1] == dom[2], which
      -H enforces); it needs a per-direction length if that ever changes. */
   const HYPRE_Real swirl_amp = params->wmax * params->dom[1] / SWIRL_GRAD_MAX * hx;

   /* Create the matrix and vectors */
   HYPRE_IJMatrixCreate(mesh->cart_comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(mesh->cart_comm, ilower, iupper, &b);
   HYPRE_IJVectorCreate(mesh->cart_comm, ilower, iupper, &x0);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(x0, HYPRE_PARCSR);

   /* Set sizes for diagonal and off-diagonal entries per row */
   HYPRE_Int *nnzrow = (HYPRE_Int *)calloc((size_t)local_size, sizeof(HYPRE_Int));
   for (int row = 0; row < local_size; row++)
   {
      nnzrow[row] = 7; /* Maximum stencil size */
   }
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   HYPREDRV_IJ_MATRIX_INIT_HOST(A);
   free(nnzrow);

   /* Initialize the RHS and initial guess vectors */
   HYPREDRV_IJ_VECTOR_INIT_HOST(b);
   HYPREDRV_IJ_VECTOR_INIT_HOST(x0);

   /* Set matrix coefficients and vector values. Every cell writes only its own
    * row, so the z-slabs can be processed independently by different threads */
#ifdef _OPENMP
#pragma omp parallel for
#endif
   for (HYPRE_BigInt gz = pstarts[2][p[2]]; gz < pstarts[2][p[2] + 1]; gz++)
   {
      HYPRE_Int    nentries;
      HYPRE_BigInt row, cols[7];
      HYPRE_Real   values[7], diag, rhs_value, x0_value;

      const HYPRE_Real z = ((HYPRE_Real)gz + 0.5) * hz;

      /* Stream function at the two z-faces bounding this slab */
      const HYPRE_Real Gz_lo = SwirlShape((HYPRE_Real)gz * hz, params->dom[2]);
      const HYPRE_Real Gz_hi = SwirlShape((HYPRE_Real)(gz + 1) * hz, params->dom[2]);

      for (HYPRE_BigInt gy = pstarts[1][p[1]]; gy < pstarts[1][p[1] + 1]; gy++)
      {
         const HYPRE_Real y = ((HYPRE_Real)gy + 0.5) * hy;

         /* Axial mass flux through both x-faces of this row of cells */
         const HYPRE_Real Cf =
            AxialVelocity(y, z, params->dom[1], params->dom[2], params->umax) * Ax;
         const HYPRE_Real Cf_pos = (Cf > 0.0) ? Cf : 0.0;
         const HYPRE_Real Cf_neg = (Cf < 0.0) ? -Cf : 0.0;

         /* Transverse mass fluxes from the cross-plane stream function. They
          * vanish on the walls because the shape function does */
         const HYPRE_Real Gy_lo = SwirlShape((HYPRE_Real)gy * hy, params->dom[1]);
         const HYPRE_Real Gy_hi = SwirlShape((HYPRE_Real)(gy + 1) * hy, params->dom[1]);

         const HYPRE_Real My_lo = -swirl_amp * Gy_lo * (Gz_hi - Gz_lo);
         const HYPRE_Real My_hi = -swirl_amp * Gy_hi * (Gz_hi - Gz_lo);
         const HYPRE_Real Mz_lo = swirl_amp * (Gy_hi - Gy_lo) * Gz_lo;
         const HYPRE_Real Mz_hi = swirl_amp * (Gy_hi - Gy_lo) * Gz_hi;

         /* Outward fluxes: the low faces point along the negative axes */
         const HYPRE_Real Sy_lo = -My_lo, Sy_hi = My_hi;
         const HYPRE_Real Sz_lo = -Mz_lo, Sz_hi = Mz_hi;

         for (HYPRE_BigInt gx = pstarts[0][p[0]]; gx < pstarts[0][p[0] + 1]; gx++)
         {
            /* Global index in block-partitioned ordering */
            row = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, p, gdims, pstarts);

            /* Reserve the first slot for the diagonal entry */
            cols[0]  = row;
            nentries = 1;

            /* Backward Euler contributions */
            diag      = vol / dt;
            rhs_value = (vol / dt) * c_old[row - ilower];
            x0_value  = c_old[row - ilower];

            /* West face: inlet boundary or upwind/diffusive coupling */
            if (gx > 0)
            {
               diag += Dx + Cf_neg;
               cols[nentries] =
                  (gx > pstarts[0][p[0]])
                     ? row - 1
                     : grid2idx((HYPRE_BigInt[]){gx - 1, gy, gz},
                                (HYPRE_Int[]){p[0] - 1, p[1], p[2]}, gdims, pstarts);
               values[nentries++] = -(Dx + Cf_pos);
            }
            else
            {
               diag += 2.0 * Dx + Cf_neg;
               rhs_value += (2.0 * Dx + Cf_pos) * c_in;
            }

            /* East face: outflow boundary or upwind/diffusive coupling.
             * Backflow at the outlet would draw in c = 0 */
            if (gx < gdims[0] - 1)
            {
               diag += Dx + Cf_pos;
               cols[nentries] =
                  (gx < pstarts[0][p[0] + 1] - 1)
                     ? row + 1
                     : grid2idx((HYPRE_BigInt[]){gx + 1, gy, gz},
                                (HYPRE_Int[]){p[0] + 1, p[1], p[2]}, gdims, pstarts);
               values[nentries++] = -(Dx + Cf_neg);
            }
            else
            {
               diag += Cf_pos;
            }

            /* Y-faces: diffusion plus upwinded swirl, no flux through the walls */
            if (gy > 0)
            {
               diag += Dy + ((Sy_lo > 0.0) ? Sy_lo : 0.0);
               cols[nentries] =
                  (gy > pstarts[1][p[1]])
                     ? row - n[0]
                     : grid2idx((HYPRE_BigInt[]){gx, gy - 1, gz},
                                (HYPRE_Int[]){p[0], p[1] - 1, p[2]}, gdims, pstarts);
               values[nentries++] = -(Dy + ((Sy_lo < 0.0) ? -Sy_lo : 0.0));
            }
            if (gy < gdims[1] - 1)
            {
               diag += Dy + ((Sy_hi > 0.0) ? Sy_hi : 0.0);
               cols[nentries] =
                  (gy < pstarts[1][p[1] + 1] - 1)
                     ? row + n[0]
                     : grid2idx((HYPRE_BigInt[]){gx, gy + 1, gz},
                                (HYPRE_Int[]){p[0], p[1] + 1, p[2]}, gdims, pstarts);
               values[nentries++] = -(Dy + ((Sy_hi < 0.0) ? -Sy_hi : 0.0));
            }

            /* Z-faces: diffusion plus upwinded swirl, no flux through the walls */
            if (gz > 0)
            {
               diag += Dz + ((Sz_lo > 0.0) ? Sz_lo : 0.0);
               cols[nentries] =
                  (gz > pstarts[2][p[2]])
                     ? row - n[0] * n[1]
                     : grid2idx((HYPRE_BigInt[]){gx, gy, gz - 1},
                                (HYPRE_Int[]){p[0], p[1], p[2] - 1}, gdims, pstarts);
               values[nentries++] = -(Dz + ((Sz_lo < 0.0) ? -Sz_lo : 0.0));
            }
            if (gz < gdims[2] - 1)
            {
               diag += Dz + ((Sz_hi > 0.0) ? Sz_hi : 0.0);
               cols[nentries] =
                  (gz < pstarts[2][p[2] + 1] - 1)
                     ? row + n[0] * n[1]
                     : grid2idx((HYPRE_BigInt[]){gx, gy, gz + 1},
                                (HYPRE_Int[]){p[0], p[1], p[2] + 1}, gdims, pstarts);
               values[nentries++] = -(Dz + ((Sz_hi < 0.0) ? -Sz_hi : 0.0));
            }

            /* Set matrix and vector values */
            values[0] = diag;
            HYPRE_IJMatrixSetValues(A, 1, &nentries, &row, cols, values);
            HYPRE_IJVectorSetValues(b, 1, &row, &rhs_value);
            HYPRE_IJVectorSetValues(x0, 1, &row, &x0_value);
         }
      }
   }

   /* Assemble the matrix and vectors */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorAssemble(x0);

   /* Return matrix and vectors through pointers */
   *A_ptr  = A;
   *b_ptr  = b;
   *x0_ptr = x0;

   return 0;
}

/*--------------------------------------------------------------------------
 * Write an array of equally spaced coordinates to a VTK file
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

/*--------------------------------------------------------------------------
 * Write VTK solution to file
 *
 * The unknowns live at cell centers, so each rank writes its own block as a
 * rectilinear grid whose coordinates are the *face* positions and attaches
 * the concentration as cell data. Unlike node-centered output, this needs no
 * ghost exchange: neighboring blocks share faces, not values.
 *
 * A negative step index writes a standalone snapshot (plus its collection
 * file); a non-negative one writes frame files for a time series, which is
 * later assembled into a single collection by WriteVTKseries.
 *--------------------------------------------------------------------------*/
int
WriteVTKsolution(DistMesh *mesh, ProblemParams *params, HYPRE_Real *c, HYPRE_Int step)
{
   int    myid, num_procs;
   char   filename[256];
   double t0 = MPI_Wtime();

   /* Mesh info */
   HYPRE_Int     *p       = &mesh->coords[0];
   HYPRE_Int     *nlocal  = &mesh->nlocal[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_Real    *hsizes  = &mesh->hsizes[0];

   /* Get process info */
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Comm_size(mesh->cart_comm, &num_procs);

   /* Set short-hand variables */
   int nx       = (int)(nlocal[0]);
   int ny       = (int)(nlocal[1]);
   int nz       = (int)(nlocal[2]);
   int ix_start = (int)pstarts[0][p[0]];
   int iy_start = (int)pstarts[1][p[1]];
   int iz_start = (int)pstarts[2][p[2]];

   /* Write the local piece */
   if (step < 0)
   {
      snprintf(filename, sizeof(filename), "convdif_%dx%dx%d_%dx%dx%d_%d.vtr",
               params->N[0], params->N[1], params->N[2], params->P[0], params->P[1],
               params->P[2], myid);
   }
   else
   {
      snprintf(filename, sizeof(filename), "convdif_%dx%dx%d_%dx%dx%d_s%04d_%d.vtr",
               params->N[0], params->N[1], params->N[2], params->P[0], params->P[1],
               params->P[2], (int)step, myid);
   }
   FILE *fp = fopen(filename, "w");
   if (!fp)
   {
      printf("Error: Cannot open file %s\n", filename);
      MPI_Abort(mesh->cart_comm, -1);
   }

   fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
   fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\">\n");
   fprintf(fp, "  <RectilinearGrid WholeExtent=\"%d %d %d %d %d %d\">\n", ix_start,
           ix_start + nx, iy_start, iy_start + ny, iz_start, iz_start + nz);
   fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", ix_start, ix_start + nx,
           iy_start, iy_start + ny, iz_start, iz_start + nz);

   /* Write face coordinates */
   fprintf(fp, "      <Coordinates>\n");
   WriteCoordArray(fp, "x", ix_start * hsizes[0], hsizes[0], nx + 1);
   WriteCoordArray(fp, "y", iy_start * hsizes[1], hsizes[1], ny + 1);
   WriteCoordArray(fp, "z", iz_start * hsizes[2], hsizes[2], nz + 1);
   fprintf(fp, "      </Coordinates>\n");

   /* Write cell-centered solution data */
   fprintf(fp, "      <CellData Scalars=\"concentration\">\n");
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"concentration\" "
               "format=\"appended\" offset=\"0\">\n");
   fprintf(fp, "        </DataArray>\n");
   fprintf(fp, "      </CellData>\n");
   fprintf(fp, "    </Piece>\n");
   fprintf(fp, "  </RectilinearGrid>\n");

   /* Write appended binary data */
   fprintf(fp, "  <AppendedData encoding=\"raw\">\n   _");
   int data_size_int = (int)((size_t)mesh->local_size * sizeof(double));
   fwrite(&data_size_int, sizeof(int), 1, fp);
   fwrite(c, sizeof(double), (size_t)mesh->local_size, fp);
   fprintf(fp, "\n  </AppendedData>\n");
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);

   /* Create PVD file (rank 0 only, standalone-snapshot mode) */
   if (!myid && step < 0)
   {
      snprintf(filename, sizeof(filename), "convdif_%dx%dx%d_%dx%dx%d.pvd", params->N[0],
               params->N[1], params->N[2], params->P[0], params->P[1], params->P[2]);
      fp = fopen(filename, "w");
      if (!fp)
      {
         printf("Error: Cannot open %s\n", filename);
         MPI_Abort(mesh->cart_comm, -1);
      }
      printf("Writing PVD file %s ...", filename);

      fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
      fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n");
      fprintf(fp, "  <Collection>\n");
      for (int i = 0; i < num_procs; i++)
      {
         fprintf(fp,
                 "    <DataSet part=\"%d\" file=\"convdif_%dx%dx%d_%dx%dx%d_%d.vtr\"/>\n",
                 i, params->N[0], params->N[1], params->N[2], params->P[0], params->P[1],
                 params->P[2], i);
      }
      fprintf(fp, "  </Collection>\n");
      fprintf(fp, "</VTKFile>\n");
      fclose(fp);
      printf(" Time elapsed: %.4f s\n", MPI_Wtime() - t0);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Write the PVD collection tying all per-step frame files into a single
 * time series (rank 0 only). times[s] holds the physical time of frame s,
 * for s = 0 (initial condition) through nsteps.
 *--------------------------------------------------------------------------*/
int
WriteVTKseries(DistMesh *mesh, ProblemParams *params, HYPRE_Int nsteps,
               const HYPRE_Real *times)
{
   int  myid, num_procs;
   char filename[256];

   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Comm_size(mesh->cart_comm, &num_procs);
   if (myid) return 0;

   snprintf(filename, sizeof(filename), "convdif_%dx%dx%d_%dx%dx%d.pvd", params->N[0],
            params->N[1], params->N[2], params->P[0], params->P[1], params->P[2]);
   FILE *fp = fopen(filename, "w");
   if (!fp)
   {
      printf("Error: Cannot open %s\n", filename);
      MPI_Abort(mesh->cart_comm, -1);
   }
   printf("Writing PVD file %s (%d frames)\n", filename, (int)nsteps + 1);

   fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
   fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n");
   fprintf(fp, "  <Collection>\n");
   for (HYPRE_Int s = 0; s <= nsteps; s++)
   {
      for (int i = 0; i < num_procs; i++)
      {
         fprintf(fp,
                 "    <DataSet timestep=\"%g\" part=\"%d\" "
                 "file=\"convdif_%dx%dx%d_%dx%dx%d_s%04d_%d.vtr\"/>\n",
                 times[s], i, params->N[0], params->N[1], params->N[2], params->P[0],
                 params->P[1], params->P[2], (int)s, i);
      }
   }
   fprintf(fp, "  </Collection>\n");
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);

   return 0;
}
