/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // fabs
#include "HYPREDRV.h"
//#define VTK_USE_ASCII 0

/*==========================================================================
 *   3D Laplacian PDE Solver with Multiple Stencil Options
 *==========================================================================
 *
 *   Grid Partitioning:
 *   ------------------
 *                   P[2]
 *                    ↑     Processor Grid: P[0] × P[1] × P[2]
 *                    |   /  Each proc owns a local block
 *                    |  /   of size n[0] × n[1] × n[2]
 *                    | /
 *            P[1] ←--C
 *                   /
 *                  /
 *                 /
 *                ↙ P[0]
 *
 *   PDE Problem:
 *   ------------
 *   -∇·(c∇u) = f    in Ω = [0,1]³
 *        u = 0      on ∂Ω, except
 *        u = 1      on y = 0
 *
 *   Available Stencils:
 *   ------------------
 *     7-point: (C) + 6 face neighbors
 *                    2nd order, M-matrix
 *
 *    19-point: (C) + 6 face + 12 edge neighbors
 *                    2nd order, M-matrix
 *
 *    27-point: (C) + all neighbors in 3×3×3 cube
 *                    2nd order, M-matrix
 *
 *   125-point: (C) + all neighbors in 5×5×5 cube
 *                    2nd order, M-matrix
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Default solver configuration
 *--------------------------------------------------------------------------*/
static const char *default_config = "solver: pcg\n"
                                    "preconditioner: amg\n";

/*--------------------------------------------------------------------------
 * Problem parameters struct
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int   visualize;   /* Visualize solution in VTK format */
   HYPRE_Int   print;       /* Print matrices/vectors to file */
   HYPRE_Int   verbose;     /* Verbosity level bitset */
   HYPRE_Int   nsolve;      /* Number of times to solve the system */
   HYPRE_Int   N[3];        /* Grid dimensions */
   HYPRE_Int   P[3];        /* Processor grid dimensions */
   HYPRE_Real  c[6];        /* Diffusion coefficients */
   HYPRE_Int   stencil;     /* Stencil type (7 or 27) */
   char       *yaml_file;   /* YAML configuration file */
} ProblemParams;

/*--------------------------------------------------------------------------
 * Distributed Mesh struct
 *--------------------------------------------------------------------------*/
typedef struct {
   MPI_Comm      cart_comm;     /* Cartesian communicator */
   HYPRE_Int     mypid;         /* Local partition ID (rank ID) */
   HYPRE_Int     gdims[3];      /* Global dimensions (nx, ny, nz) */
   HYPRE_Int     pdims[3];      /* Processor grid dimensions (Px, Py, Pz) */
   HYPRE_Int     coords[3];     /* Process coordinates in processor grid */
   HYPRE_Int     nlocal[3];     /* Local dimensions */
   HYPRE_Int     nbrs[14];      /* Neighbor rank indices */
   HYPRE_Int     local_size;    /* Local problem size */
   HYPRE_BigInt  ilower;        /* Lower bound of local rows */
   HYPRE_BigInt  iupper;        /* Upper bound of local rows */
   HYPRE_BigInt *pstarts[3];    /* Partition prefix sums for each dimension */
   HYPRE_Real    gsizes[3];     /* Global grid sizes */
} DistMesh;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

int PrintUsage(void);
int CreateDistMesh(MPI_Comm, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, DistMesh**);
int DestroyDistMesh(DistMesh**);
int ParseArguments(int, char**, ProblemParams*, int);
int BuildLaplacianSystem_7pt(DistMesh*, ProblemParams*, HYPRE_IJMatrix*, HYPRE_IJVector*);
int BuildLaplacianSystem_19pt(DistMesh*, ProblemParams*, HYPRE_IJMatrix*, HYPRE_IJVector*);
int BuildLaplacianSystem_27pt(DistMesh*, ProblemParams*, HYPRE_IJMatrix*, HYPRE_IJVector*);
int BuildLaplacianSystem_125pt(DistMesh*, ProblemParams*, HYPRE_IJMatrix*, HYPRE_IJVector*);
int WriteVTKsolution(DistMesh*, ProblemParams*, HYPRE_Real*);

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/
int
PrintUsage(void)
{
   printf("\n");
   printf("Usage: ${MPIEXEC_COMMAND} ${MPIEXEC_NUMPROC_FLAG} <np> ./laplacian [options]\n");
   printf("\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML configuration file for solver settings\n");
   printf("  -n <nx> <ny> <nz> : Grid dimensions (default: 10 10 10)\n");
   printf("  -c <cx> <cy> <cz> : Diffusion coefficients (default: 1.0 1.0 1.0)\n");
   printf("  -P <Px> <Py> <Pz> : Processor grid dimensions (default: 1 1 1)\n");
   printf("  -s <val>          : Stencil type: 7 19 27 125 (default: 7)\n");
   printf("  -ns|--nsolve <n>  : Number of times to solve the system (default: 5)\n");
   printf("  -vis|--visualize  : Output solution in VTK format (default: false)\n");
   printf("  -p|--print        : Print matrices/vectors to file (default: false)\n");
   printf("  -v|--verbose <n>  : Verbosity level (bitset):\n");
   printf("                      0x1: Print solver statistics\n");
   printf("                      0x2: Print library info\n");
   printf("                      0x4: Print system info\n");
   printf("  -h|--help         : Print this message\n");
   printf("\n");

   return 0;
}

/*--------------------------------------------------------------------------
 * Parse command line arguments
 *--------------------------------------------------------------------------*/
int
ParseArguments(int argc, char *argv[], ProblemParams *params, int myid)
{
   /* Set defaults */
   params->visualize = 0;
   params->print = 0;
   params->verbose = 7;
   params->nsolve = 5;
   for (int i = 0; i < 3; i++)
   {
      params->N[i] = 10;
      params->P[i] = 1;
      params->c[i] = 1.0;
      params->c[i+3] = 1.0;
   }
   params->stencil = 7;
   params->yaml_file = NULL;

   /* Parse command line */
   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))
      {
         if (++i < argc) params->yaml_file = argv[i];
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
      else if (!strcmp(argv[i], "-c"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -c requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++)
         {
            params->c[j] = atof(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-c2"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -c2 requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++)
         {
            params->c[j+3] = atof(argv[++i]);
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
      else if (!strcmp(argv[i], "-s"))
      {
         if (++i < argc) params->stencil = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-ns") || !strcmp(argv[i], "--nsolve"))
      {
         if (++i < argc) params->nsolve = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-vis") || !strcmp(argv[i], "--visualize"))
      {
         params->visualize = 1;
      }
      else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--print"))
      {
         params->print = 1;
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

   return 0;
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
   MPI_Comm        comm = MPI_COMM_WORLD;
   HYPREDRV_t      hypredrv;
   int             myid, num_procs;
   ProblemParams   params;
   DistMesh*       mesh;
   HYPRE_IJMatrix  A;
   HYPRE_IJVector  b;
   HYPRE_Real     *sol_data;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   /* Parse command line arguments */
   if (ParseArguments(argc, argv, &params, myid))
   {
      MPI_Finalize();
      return 1;
   }

   /* Verify processor grid matches total number of processes */
   if (params.P[0] * params.P[1] * params.P[2] != num_procs)
   {
      if (!myid)
      {
         printf("Error: Number of processes (%d) must match processor grid dimensions (%d x %d x %d = %d)\n",
                num_procs, params.P[0], params.P[1], params.P[2],
                params.P[0] * params.P[1] * params.P[2]);
      }
      MPI_Finalize();
      return 1;
   }

   /* Initialize hypredrive */
   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());

   /* Print library info if requested */
   if (params.verbose & 0x2)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm));
   }

   /* Print system info if requested */
   if (params.verbose & 0x4)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   /* Create hypredrive object */
   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));

   /* Configure solver using YAML input or default configuration */
   char *args[2];
   args[0] = params.yaml_file ? params.yaml_file : (char *)default_config;
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));

   /* Set HYPREDRV options */
   HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

   /* Print problem parameters */
   if (!myid)
   {
      printf("\n");
      printf("=====================================================\n");
      printf("              Laplacian Problem Setup\n");
      printf("=====================================================\n");
      printf("Grid dimensions:      %d x %d x %d\n", (int)params.N[0], (int)params.N[1], (int)params.N[2]);
      printf("Processor topology:   %d x %d x %d\n", (int)params.P[0], (int)params.P[1], (int)params.P[2]);
      printf("Diffusion coeffs:     (%.2e, %.2e, %.2e)\n", params.c[0], params.c[1], params.c[2]);
      printf("Discretization:       %d-point stencil\n", (int)params.stencil);
      printf("Visualization:        %s\n", params.visualize ? "true" : "false");
      printf("Print system:         %s\n", params.print ? "true" : "false");
      printf("Verbosity level:      0x%x\n", params.verbose);
      printf("Number of solves:     %d\n", params.nsolve);
      printf("=====================================================\n\n");
   }

   /* Create distributed mesh object */
   CreateDistMesh(comm,
                  params.N[0], params.N[1], params.N[2],
                  params.P[0], params.P[1], params.P[2],
                  &mesh);

   /* Create the linear system */
   HYPREDRV_SAFE_CALL(HYPREDRV_TimerStart("system"));
   if (params.stencil == 7)
   {
      BuildLaplacianSystem_7pt(mesh, &params, &A, &b);
   }
   else if (params.stencil == 19)
   {
      BuildLaplacianSystem_19pt(mesh, &params, &A, &b);
   }
   else if (params.stencil == 27)
   {
      BuildLaplacianSystem_27pt(mesh, &params, &A, &b);
   }
   else if (params.stencil == 125)
   {
      BuildLaplacianSystem_125pt(mesh, &params, &A, &b);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_TimerStop("system"));

   /* Transfer data to GPU memory */
#if defined(HYPRE_USING_GPU)
   HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   HYPRE_IJVectorMigrate(b, HYPRE_MEMORY_DEVICE);
#endif

   /* Associate the matrix and vectors with hypredrive */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix) A));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector) b));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv));

   /* Solve multiple times if requested */
   for (int isolve = 0; isolve < params.nsolve; isolve++)
   {
      if (!myid) printf("Solve %d/%d...\n", isolve + 1, params.nsolve);

      /* Reset initial guess to zero before each solve */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));

      /* Create phase */
      HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));

      /* Setup and solve */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));

      /* Destroy phase */
      HYPREDRV_SAFE_CALL(HYPREDRV_PreconDestroy(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));
   }

   /* Print solver statistics if requested */
   if (!myid && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
   }

   /* Visualization phase */
   if (params.visualize)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol_data));
      WriteVTKsolution(mesh, &params, sol_data);
   }

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
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
 *   - gcoords: global grid point coordinates
 *   - bcoords: block partition coordinates
 *   - gdims:   global grid dimensions
 *   - pstarts: block partition prefixed sums for each dimension
 *--------------------------------------------------------------------------*/
static inline HYPRE_BigInt
grid2idx(const HYPRE_BigInt   gcoords[3],
         const HYPRE_Int      bcoords[3],
         const HYPRE_BigInt   gdims[3],
         HYPRE_BigInt       **pstarts)
{
   return pstarts[2][bcoords[2]] * gdims[0] * gdims[1] +
          pstarts[1][bcoords[1]] * gdims[0] * (pstarts[2][bcoords[2] + 1] - pstarts[2][bcoords[2]]) +
          pstarts[0][bcoords[0]] * (pstarts[1][bcoords[1] + 1] - pstarts[1][bcoords[1]]) *
          (pstarts[2][bcoords[2] + 1] - pstarts[2][bcoords[2]]) +
          ((gcoords[2] - pstarts[2][bcoords[2]]) * (pstarts[1][bcoords[1] + 1] - pstarts[1][bcoords[1]]) +
          (gcoords[1] - pstarts[1][bcoords[1]])) * (pstarts[0][bcoords[0] + 1] - pstarts[0][bcoords[0]]) +
          (gcoords[0] - pstarts[0][bcoords[0]]);
}

/*--------------------------------------------------------------------------
 * Create mesh partition information
 *--------------------------------------------------------------------------*/
int
CreateDistMesh(MPI_Comm   comm,
               HYPRE_Int  Nx, HYPRE_Int Ny, HYPRE_Int Nz,
               HYPRE_Int  Px, HYPRE_Int Py, HYPRE_Int Pz,
               DistMesh **mesh_ptr)
{
   DistMesh *mesh = (DistMesh *) malloc(sizeof(DistMesh));
   int       myid, num_procs;

   /* Store dimensions */
   mesh->gdims[0] = Nx; mesh->gdims[1] = Ny; mesh->gdims[2] = Nz;
   mesh->pdims[0] = Px; mesh->pdims[1] = Py; mesh->pdims[2] = Pz;

   /* Create cartesian communicator */
   MPI_Cart_create(comm, 3, mesh->pdims, (int[]){0, 0, 0}, 1, &(mesh->cart_comm));
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Cart_coords(mesh->cart_comm, myid, 3, mesh->coords);

   /* Store process rank */
   mesh->mypid = (HYPRE_Int) myid;

   /* Compute partition prefix sums for each dimension, local dimensions, and global grid sizes */
   for (int i = 0; i < 3; i++)
   {
      HYPRE_Int size = mesh->gdims[i] / mesh->pdims[i];
      HYPRE_Int rest = mesh->gdims[i] - size * mesh->pdims[i];

      mesh->pstarts[i] = calloc(mesh->pdims[i] + 1, sizeof(HYPRE_BigInt));
      for (int j = 0; j < mesh->pdims[i] + 1; j++)
      {
         mesh->pstarts[i][j] = (HYPRE_BigInt)(size * j + (j < rest ? j : rest));
      }
      mesh->nlocal[i] = (HYPRE_Int)(mesh->pstarts[i][mesh->coords[i] + 1] - mesh->pstarts[i][mesh->coords[i]]);
      mesh->gsizes[i] = 1.0 / (mesh->gdims[i] - 1);
   }

   /* Compute local matrix bounds */
   HYPRE_BigInt gcoords[3] = {mesh->pstarts[0][mesh->coords[0]],
                              mesh->pstarts[1][mesh->coords[1]],
                              mesh->pstarts[2][mesh->coords[2]]};
   mesh->ilower = grid2idx(gcoords, mesh->coords, mesh->gdims, mesh->pstarts);
   mesh->local_size = mesh->nlocal[0] * mesh->nlocal[1] * mesh->nlocal[2];
   mesh->iupper = mesh->ilower + mesh->local_size - 1;

   /* Get first-order neighbors */
   MPI_Cart_shift(mesh->cart_comm, 0, 1, &mesh->nbrs[0], &mesh->nbrs[1]); // x: left/right
   MPI_Cart_shift(mesh->cart_comm, 1, 1, &mesh->nbrs[2], &mesh->nbrs[3]); // y: down/up
   MPI_Cart_shift(mesh->cart_comm, 2, 1, &mesh->nbrs[4], &mesh->nbrs[5]); // z: back/front

   /* Left-down neighbor */
   if (mesh->coords[0] > 0 &&
       mesh->coords[1] > 0)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0] - 1, mesh->coords[1] - 1, mesh->coords[2]}, &mesh->nbrs[6]);
   }
   else
   {
      mesh->nbrs[6] = MPI_PROC_NULL;
   }

   /* Right-up neighbor */
   if (mesh->coords[0] < mesh->pdims[0] - 1 &&
       mesh->coords[1] < mesh->pdims[1] - 1)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0] + 1, mesh->coords[1] + 1, mesh->coords[2]}, &mesh->nbrs[7]);
   }
   else
   {
      mesh->nbrs[7] = MPI_PROC_NULL;
   }

   /* Left-back neighbor*/
   if (mesh->coords[0] > 0 &&
       mesh->coords[2] > 0)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0] - 1, mesh->coords[1], mesh->coords[2] - 1}, &mesh->nbrs[8]);
   }
   else
   {
      mesh->nbrs[8] = MPI_PROC_NULL;
   }

   /* Right-front neighbor */
   if (mesh->coords[0] < mesh->pdims[0] - 1 &&
       mesh->coords[2] < mesh->pdims[2] - 1)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0] + 1, mesh->coords[1], mesh->coords[2] + 1}, &mesh->nbrs[9]);
   }
   else
   {
      mesh->nbrs[9] = MPI_PROC_NULL;
   }

   /* Down-back neighbor */
   if (mesh->coords[1] > 0 &&
       mesh->coords[2] > 0)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0], mesh->coords[1] - 1, mesh->coords[2] - 1}, &mesh->nbrs[10]);
   }
   else
   {
      mesh->nbrs[10] = MPI_PROC_NULL;
   }

   /* Up-front neighbor */
   if (mesh->coords[1] < mesh->pdims[1] - 1 &&
       mesh->coords[2] < mesh->pdims[2] - 1)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0], mesh->coords[1] + 1, mesh->coords[2] + 1}, &mesh->nbrs[11]);
   }
   else
   {
      mesh->nbrs[11] = MPI_PROC_NULL;
   }

   /* Left-down-back neighbor */
   if (mesh->coords[0] > 0 &&
       mesh->coords[1] > 0 &&
       mesh->coords[2] > 0)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0] - 1, mesh->coords[1] - 1, mesh->coords[2] - 1}, &mesh->nbrs[12]);
   }
   else
   {
      mesh->nbrs[12] = MPI_PROC_NULL;
   }

   /* Right-up-front neighbor */
   if (mesh->coords[0] < mesh->pdims[0] - 1 &&
       mesh->coords[1] < mesh->pdims[1] - 1 &&
       mesh->coords[2] < mesh->pdims[2] - 1)
   {
      MPI_Cart_rank(mesh->cart_comm, (int[]){mesh->coords[0] + 1, mesh->coords[1] + 1, mesh->coords[2] + 1}, &mesh->nbrs[13]);
   }
   else
   {
      mesh->nbrs[13] = MPI_PROC_NULL;
   }

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
 * Create 7-point Laplacian stencil using finite differences with Dirichlet BCs
 * All boundaries = 0, except for back y-direction, where the boundary is 1
 *--------------------------------------------------------------------------*/
int
BuildLaplacianSystem_7pt(DistMesh        *mesh,
                         ProblemParams   *params,
                         HYPRE_IJMatrix  *A_ptr,
                         HYPRE_IJVector  *b_ptr)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_Int      local_size = mesh->local_size;
   HYPRE_Int     *n = &mesh->nlocal[0];
   HYPRE_Int     *p = &mesh->coords[0];
   HYPRE_Int     *gdims = &mesh->gdims[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_BigInt   ilower = mesh->ilower;
   HYPRE_BigInt   iupper = mesh->iupper;

   /* Create the matrix and vector */
   HYPRE_IJMatrixCreate(mesh->cart_comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(mesh->cart_comm, ilower, iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   /* Set sizes for diagonal and off-diagonal entries per row */
   HYPRE_Int *nnzrow = (HYPRE_Int *) calloc(local_size, sizeof(HYPRE_Int));
   for (int row = 0; row < local_size; row++)
   {
      nnzrow[row] = 7; /* Maximum stencil size */
   }
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   free(nnzrow);

   /* Initialize the RHS vector */
   HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST);

   /* Set matrix coefficients and RHS values */
   HYPRE_Int     nentries;
   HYPRE_BigInt  row, cols[7];
   HYPRE_Real    values[7], rhs_value;

   for (HYPRE_BigInt gz = pstarts[2][p[2]]; gz < pstarts[2][p[2] + 1]; gz++)
   {
      for (HYPRE_BigInt gy = pstarts[1][p[1]]; gy < pstarts[1][p[1] + 1]; gy++)
      {
         for (HYPRE_BigInt gx = pstarts[0][p[0]]; gx < pstarts[0][p[0] + 1]; gx++)
         {
            /* Global index in block-partitioned ordering */
            row = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, p, gdims, pstarts);
            nentries = 0;

            /* Center point */
            cols[nentries] = row;
            values[nentries++] = 2.0 * (params->c[0] + params->c[1] + params->c[2]);

            /* Z-direction negative connections */
            if (gz > pstarts[2][p[2]])
            {
               cols[nentries] = row - n[0] * n[1];
               values[nentries++] = -params->c[2];
            }
            else if (gz == pstarts[2][p[2]] && p[2] > 0)
            {
               /* Neighboring process coordinates */
               HYPRE_BigInt ncoords[3] = {gx, gy, gz - 1};
               HYPRE_Int    np[3]      = {p[0], p[1], p[2] - 1};

               cols[nentries] = grid2idx(ncoords, np, gdims, pstarts);
               values[nentries++] = -params->c[2];
            }

            /* Y-direction negative connections */
            if (gy > pstarts[1][p[1]])
            {
               cols[nentries] = row - n[0];
               values[nentries++] = -params->c[1];
            }
            else if (gy == pstarts[1][p[1]] && p[1] > 0)
            {
               /* Neighboring process coordinates */
               HYPRE_BigInt ncoords[3] = {gx, gy - 1, gz};
               HYPRE_Int    np[3]      = {p[0], p[1] - 1, p[2]};

               cols[nentries] = grid2idx(ncoords, np, gdims, pstarts);
               values[nentries++] = -params->c[1];
            }

            /* X-direction negative connections */
            if (gx > pstarts[0][p[0]])
            {
               cols[nentries] = row - 1;
               values[nentries++] = -params->c[0];
            }
            else if (gx == pstarts[0][p[0]] && p[0] > 0)
            {
               /* Neighboring process coordinates */
               HYPRE_BigInt ncoords[3] = {gx - 1, gy, gz};
               HYPRE_Int    np[3]      = {p[0] - 1, p[1], p[2]};

               cols[nentries] = grid2idx(ncoords, np, gdims, pstarts);
               values[nentries++] = -params->c[0];
            }

            /* X-direction positive connections */
            if (gx < pstarts[0][p[0] + 1] - 1)
            {
               cols[nentries] = row + 1;
               values[nentries++] = -params->c[0];
            }
            else if (gx == pstarts[0][p[0] + 1] - 1 && p[0] < mesh->pdims[0] - 1)
            {
               /* Neighboring process coordinates */
               HYPRE_BigInt ncoords[3] = {gx + 1, gy, gz};
               HYPRE_Int    np[3]      = {p[0] + 1, p[1], p[2]};

               cols[nentries] = grid2idx(ncoords, np, gdims, pstarts);
               values[nentries++] = -params->c[0];
            }

            /* Y-direction positive connections */
            if (gy < pstarts[1][p[1] + 1] - 1)
            {
               cols[nentries] = row + n[0];
               values[nentries++] = -params->c[1];
            }
            else if (gy == pstarts[1][p[1] + 1] - 1 && p[1] < mesh->pdims[1] - 1)
            {
               /* Neighboring process coordinates */
               HYPRE_BigInt ncoords[3] = {gx, gy + 1, gz};
               HYPRE_Int    np[3]      = {p[0], p[1] + 1, p[2]};

               cols[nentries] = grid2idx(ncoords, np, gdims, pstarts);
               values[nentries++] = -params->c[1];
            }

            /* Z-direction positive connections */
            if (gz < pstarts[2][p[2] + 1] - 1)
            {
               cols[nentries] = row + n[0] * n[1];
               values[nentries++] = -params->c[2];
            }
            else if (gz == pstarts[2][p[2] + 1] - 1 && p[2] < mesh->pdims[2] - 1)
            {
               /* Neighboring process coordinates */
               HYPRE_BigInt ncoords[3] = {gx, gy, gz + 1};
               HYPRE_Int    np[3]      = {p[0], p[1], p[2] + 1};

               cols[nentries] = grid2idx(ncoords, np, gdims, pstarts);
               values[nentries++] = -params->c[2];
            }

            /* Set matrix values */
            HYPRE_IJMatrixSetValues(A, 1, &nentries, &row, cols, values);

            /* Set RHS value - zero everywhere except back boundary (y = 0) */
            if (gy == pstarts[1][p[1]] && p[1] == 0)
            {
               rhs_value = 1.0;
            }
            else
            {
               rhs_value = 0.0;
            }
            HYPRE_IJVectorSetValues(b, 1, &row, &rhs_value);
         }
      }
   }

   /* Assemble the matrix and vector */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);
   if (params->print)
   {
      if (!mesh->mypid)
      {
         printf("Printing A_7pt.out and b_7pt.out...\n");
      }
      HYPRE_IJMatrixPrint(A, "A_7pt.out");
      HYPRE_IJVectorPrint(b, "b_7pt.out");
   }

   /* Return matrix and vector through pointers */
   *A_ptr = A;
   *b_ptr = b;

   return 0;
}

/*--------------------------------------------------------------------------
 * Build 19-point Laplacian stencil using finite differences with Dirichlet BCs
 * This stencil includes:
 * - 6 face neighbors (±1 in each direction)
 * - 12 edge neighbors (±1 in two directions)
 * - 1 center point
 * All boundaries = 0, except for back y-direction, where the boundary is 1
 *--------------------------------------------------------------------------*/
int
BuildLaplacianSystem_19pt(DistMesh        *mesh,
                         ProblemParams    *params,
                         HYPRE_IJMatrix   *A_ptr,
                         HYPRE_IJVector   *b_ptr)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_Int      local_size = mesh->local_size;
   HYPRE_Int     *n = &mesh->nlocal[0];
   HYPRE_Int     *p = &mesh->coords[0];
   HYPRE_Int     *gdims = &mesh->gdims[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_BigInt   ilower = mesh->ilower;
   HYPRE_BigInt   iupper = mesh->iupper;

   /* Create matrix and vector */
   HYPRE_IJMatrixCreate(mesh->cart_comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(mesh->cart_comm, ilower, iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   /* Set sizes for diagonal and off-diagonal entries per row */
   HYPRE_Int *nnzrow = (HYPRE_Int *) calloc(local_size, sizeof(HYPRE_Int));
   for (int row = 0; row < local_size; row++)
   {
      nnzrow[row] = 19; /* Maximum stencil size */
   }
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   free(nnzrow);

   /* Initialize the RHS vector */
   HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST);

   /* Set matrix coefficients and RHS values */
   for (HYPRE_BigInt gz = pstarts[2][p[2]]; gz < pstarts[2][p[2] + 1]; gz++)
   {
      for (HYPRE_BigInt gy = pstarts[1][p[1]]; gy < pstarts[1][p[1] + 1]; gy++)
      {
         for (HYPRE_BigInt gx = pstarts[0][p[0]]; gx < pstarts[0][p[0] + 1]; gx++)
         {
            HYPRE_BigInt row = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, p, gdims, pstarts);
            HYPRE_Int nentries = 0;
            HYPRE_BigInt cols[19];
            HYPRE_Real values[19];
            HYPRE_Real center_val = 0.0;
            HYPRE_Real rhs_val = 0.0;

            /* Coefficients for the stencil */
            HYPRE_Real face_coeff = -1.0;  /* For direct neighbors */
            HYPRE_Real edge_coeff = -0.5;  /* For edge neighbors */

            /* Loop over potential neighbors */
            for (int dz = -1; dz <= 1; dz++)
            {
               for (int dy = -1; dy <= 1; dy++)
               {
                  for (int dx = -1; dx <= 1; dx++)
                  {
                     /* Skip center point and corner points */
                     if ((dx == 0 && dy == 0 && dz == 0) ||
                         (abs(dx) + abs(dy) + abs(dz) == 3))
                        continue;

                     HYPRE_BigInt nx = gx + dx;
                     HYPRE_BigInt ny = gy + dy;
                     HYPRE_BigInt nz = gz + dz;

                     /* Determine coefficient based on neighbor type */
                     HYPRE_Real coeff;
                     if (abs(dx) + abs(dy) + abs(dz) == 1)
                     {
                        /* Face neighbor */
                        coeff = face_coeff *
                               ((dx != 0) ? params->c[0] :
                                (dy != 0) ? params->c[1] : params->c[2]);
                     }
                     else
                     {
                        /* Edge neighbor */
                        coeff = edge_coeff *
                               ((dx == 0) ? params->c[5] :  /* yz */
                                (dy == 0) ? params->c[4] :  /* xz */
                                           params->c[3]);   /* xy */
                     }

                     /* Check if neighbor is within global domain */
                     if (nx >= 0 && nx < gdims[0] &&
                         ny >= 0 && ny < gdims[1] &&
                         nz >= 0 && nz < gdims[2])
                     {
                        /* Determine processor coordinates */
                        int neighbor_proc[3] = {p[0], p[1], p[2]};

                        /* Check x */
                        if (nx < pstarts[0][p[0]] && p[0] > 0)
                           neighbor_proc[0] = p[0] - 1;
                        else if (nx >= pstarts[0][p[0] + 1] && p[0] < mesh->pdims[0] - 1)
                           neighbor_proc[0] = p[0] + 1;

                        /* Check y */
                        if (ny < pstarts[1][p[1]] && p[1] > 0)
                           neighbor_proc[1] = p[1] - 1;
                        else if (ny >= pstarts[1][p[1] + 1] && p[1] < mesh->pdims[1] - 1)
                           neighbor_proc[1] = p[1] + 1;

                        /* Check z */
                        if (nz < pstarts[2][p[2]] && p[2] > 0)
                           neighbor_proc[2] = p[2] - 1;
                        else if (nz >= pstarts[2][p[2] + 1] && p[2] < mesh->pdims[2] - 1)
                           neighbor_proc[2] = p[2] + 1;

                        /* Add entry if neighbor is in valid partition */
                        if (neighbor_proc[0] >= 0 && neighbor_proc[0] < mesh->pdims[0] &&
                            neighbor_proc[1] >= 0 && neighbor_proc[1] < mesh->pdims[1] &&
                            neighbor_proc[2] >= 0 && neighbor_proc[2] < mesh->pdims[2])
                        {
                           HYPRE_BigInt nrow = grid2idx((HYPRE_BigInt[]){nx, ny, nz},
                                                       neighbor_proc, gdims, pstarts);
                           cols[nentries] = nrow;
                           values[nentries] = coeff;
                           nentries++;
                           center_val -= coeff;
                        }
                     }
                     else
                     {
                        /* Out-of-domain => Dirichlet boundary = 0, except y=0 => 1
                         * This affects both the diagonal and possibly the RHS */
                        if (ny == -1 || (ny == 0 && gy == 0))  // Check if this is the y=0 boundary
                        {
                           rhs_val += coeff;  // Add to RHS for y=0 boundary condition
                        }
                        center_val -= coeff;  // Always add to diagonal for all boundaries
                        continue;
                     }
                  }
               }
            }

            /* Add center point */
            cols[nentries] = row;
            values[nentries] = center_val;
            nentries++;

            /* Set RHS value - zero everywhere except back boundary (y = 0) */
            if (gy == pstarts[1][p[1]] && p[1] == 0)
            {
               rhs_val = 1.0;
            }

            /* Set matrix and vector values */
            HYPRE_IJMatrixSetValues(A, 1, &nentries, &row, cols, values);
            HYPRE_IJVectorSetValues(b, 1, &row, &rhs_val);
         }
      }
   }

   /* Assemble matrix and vector */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);
   if (params->print)
   {
      if (!mesh->mypid)
      {
         printf("Printing A_19pt.out and b_19pt.out...\n");
      }
      HYPRE_IJMatrixPrint(A, "A_19pt.out");
      HYPRE_IJVectorPrint(b, "b_19pt.out");
   }

   *A_ptr = A;
   *b_ptr = b;

   return 0;
}

/*--------------------------------------------------------------------------
 * Create 27-point Laplacian stencil using finite differences with Dirichlet BCs
 * in a more "classic" second-order style. This variant assigns smaller weights
 * to edge and corner neighbors. We assume uniform spacing = 1 in x, y, z, and
 * the PDE is:
 *   cx * d²u/dx² + cy * d²u/dy² + cz * d²u/dz² = f
 * All boundaries = 0, except for the "back boundary" at y=0, where the boundary is 1.
 *--------------------------------------------------------------------------*/
int
BuildLaplacianSystem_27pt(DistMesh        *mesh,
                          ProblemParams   *params,
                          HYPRE_IJMatrix  *A_ptr,
                          HYPRE_IJVector  *b_ptr)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_Int      local_size = mesh->local_size;
   HYPRE_Int     *p = &mesh->coords[0];
   HYPRE_Int     *gdims = &mesh->gdims[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_BigInt   ilower = mesh->ilower;
   HYPRE_BigInt   iupper = mesh->iupper;

   /* Create the matrix and vector objects */
   HYPRE_IJMatrixCreate(mesh->cart_comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(mesh->cart_comm, ilower, iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   /* Row nnz sizes: up to 27 entries per row for a 27-point stencil */
   HYPRE_Int *nnzrow = (HYPRE_Int *) calloc(local_size, sizeof(HYPRE_Int));
   for (int i = 0; i < local_size; i++)
   {
      nnzrow[i] = 27;
   }
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   free(nnzrow);

   /* Initialize the RHS */
   HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST);

   /*
    * Fill matrix and RHS with a "classic" style 27-pt discretization.
    *
    * We treat:
    *   face neighbors (1D offset) with full coefficient c[i],
    *   edge neighbors (2D offset) with half of c[i],
    *   corner neighbors (3D offset) with a third of c[i].
    *
    * This is a common simple approach to shrink cross terms (edges/corners).
    * Then each neighbor's off-diagonal gets (-adj), and the center diagonal
    * picks up the sum of all adjacency as a positive value.
    */
   for (HYPRE_BigInt gz = pstarts[2][p[2]]; gz < pstarts[2][p[2] + 1]; gz++)
   {
      for (HYPRE_BigInt gy = pstarts[1][p[1]]; gy < pstarts[1][p[1] + 1]; gy++)
      {
         for (HYPRE_BigInt gx = pstarts[0][p[0]]; gx < pstarts[0][p[0] + 1]; gx++)
         {
            /* Row index in global numbering */
            HYPRE_BigInt row = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, p, params->N, pstarts);

            /* We'll store neighbor columns and values here */
            HYPRE_BigInt  cols[27];
            HYPRE_Real    vals[27];
            HYPRE_Int     nentries = 0;

            /* We'll build the center diagonal by summing neighbor adjacency. */
            HYPRE_Real center_val = 0.0;
            HYPRE_Real rhs_val    = 0.0; /* accumulate any boundary offsets here */

            /* Loop over neighbor offsets (dx,dy,dz) ∈ [-1..1], excluding (0,0,0) */
            for (int dz = -1; dz <= 1; dz++)
            {
               for (int dy = -1; dy <= 1; dy++)
               {
                  for (int dx = -1; dx <= 1; dx++)
                  {
                     if (dx == 0 && dy == 0 && dz == 0)
                        continue; /* skip the center itself in this neighbor pass */

                     /* Count how many dimensions differ to scale the adjacency properly */
                     int ndiff = (dx != 0) + (dy != 0) + (dz != 0);

                     /* Proposed neighbor in global coords */
                     HYPRE_BigInt nx = gx + dx;
                     HYPRE_BigInt ny = gy + dy;
                     HYPRE_BigInt nz = gz + dz;

                     /* For each dimension in which we differ, we add c[0], c[1], or c[2],
                        scaled by 1/ndiff to reduce cross-terms. */
                     HYPRE_Real adj = 0.0;
                     if (dx != 0) adj += params->c[0] / ndiff;
                     if (dy != 0) adj += params->c[1] / ndiff;
                     if (dz != 0) adj += params->c[2] / ndiff;

                     /* Check if neighbor is within the global domain */
                     if (nx >= 0 && nx < gdims[0] &&
                         ny >= 0 && ny < gdims[1] &&
                         nz >= 0 && nz < gdims[2])
                     {
                        /* Figure out which rank this neighbor belongs to */
                        int neighbor_proc[3] = {p[0], p[1], p[2]};

                        /* Check if it's outside this proc's local range in x */
                        if (nx < pstarts[0][p[0]] && p[0] > 0)
                           neighbor_proc[0] = p[0] - 1;
                        else if (nx >= pstarts[0][p[0] + 1] && p[0] < mesh->pdims[0] - 1)
                           neighbor_proc[0] = p[0] + 1;

                        /* Check y */
                        if (ny < pstarts[1][p[1]] && p[1] > 0)
                           neighbor_proc[1] = p[1] - 1;
                        else if (ny >= pstarts[1][p[1] + 1] && p[1] < mesh->pdims[1] - 1)
                           neighbor_proc[1] = p[1] + 1;

                        /* Check z */
                        if (nz < pstarts[2][p[2]] && p[2] > 0)
                           neighbor_proc[2] = p[2] - 1;
                        else if (nz >= pstarts[2][p[2] + 1] && p[2] < mesh->pdims[2] - 1)
                           neighbor_proc[2] = p[2] + 1;

                        /* If still valid, compute global index for that neighbor */
                        if (neighbor_proc[0] >= 0 && neighbor_proc[0] < mesh->pdims[0] &&
                            neighbor_proc[1] >= 0 && neighbor_proc[1] < mesh->pdims[1] &&
                            neighbor_proc[2] >= 0 && neighbor_proc[2] < mesh->pdims[2])
                        {
                           HYPRE_BigInt nrow = grid2idx((HYPRE_BigInt[]){nx, ny, nz},
                                                        neighbor_proc, gdims, pstarts);
                           /* Off-diagonal entry gets -adj */
                           cols[nentries]   = nrow;
                           vals[nentries++] = -adj;

                           /* The center picks up +adj */
                           center_val += adj;
                        }
                        else
                        {
                           /* Out-of-domain => Dirichlet boundary = 0, except y=0 => 1
                            * This affects both the diagonal and possibly the RHS */
                           if (ny == -1 || (ny == 0 && gy == 0))  // Check if this is the y=0 boundary
                           {
                              rhs_val += adj;  // Add to RHS for y=0 boundary condition
                           }
                           center_val -= adj;  // Always add to diagonal for all boundaries
                           continue;
                        }
                     }
                     else
                     {
                        /* Global out-of-domain => Dirichlet boundary = 0 for everything
                           except if ny=0 => boundary=1 => add to center + RHS. */
                        if (ny == 0 && ny >= 0 && ny < params->N[1])
                        {
                           center_val += adj;
                           rhs_val    += adj;
                        }
                        else
                        {
                           center_val += adj;
                        }
                     }
                  }
               }
            }

            /* Put the center entry at the end */
            cols[nentries]   = row;
            vals[nentries++] = center_val;

            /* Insert row into the matrix */
            HYPRE_IJMatrixSetValues(A, 1, &nentries, &row, cols, vals);

            /* Set the RHS:
             * Baseline approach matches the 7pt code: if global y=0 => RHS=1.0
             * BUT we've already added "adj" contributions above if out-of-domain.
             * So here we do the same boundary condition as the 7pt code. */
            if (gy == pstarts[1][p[1]] && p[1] == 0)
            {
               rhs_val = 1.0;
            }

            HYPRE_IJVectorSetValues(b, 1, &row, &rhs_val);
         }
      }
   }

   /* Assemble and output debug info if desired */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);
   if (params->print)
   {
      if (!mesh->mypid)
      {
         printf("Printing A_27pt.out and b_27pt.out...\n");
      }
      HYPRE_IJMatrixPrint(A, "A_27pt.out");
      HYPRE_IJVectorPrint(b, "b_27pt.out");
   }

   *A_ptr = A;
   *b_ptr = b;

   return 0;
}

/******************************************************************************
 * BuildLaplacianSystem_125pt:
 *
 * Creates a large-stencil (up to 125-pt) negative-offdiagonal Laplacian-like
 * operator in 3D, yielding an M-matrix for −∇²u = f. We enforce Dirichlet=0
 * except y=0 => 1. This is a low-order approach but uses a wide neighborhood.
 *
 * All interior off-diagonal entries are ≤ 0; row sum is 0 so that diagonal > 0,
 * hence an M-matrix. We have two uniform negative weights:
 *   W1 = -1.0  for "face-adjacent" neighbors (|dx|+|dy|+|dz|=1),
 *   W2 = -0.01 for other neighbors with 1 < |dx|+|dy|+|dz| ≤ 3 (up to ±2 away).
 ******************************************************************************/
int
BuildLaplacianSystem_125pt(DistMesh        *mesh,
                           ProblemParams   *params,
                           HYPRE_IJMatrix  *A_ptr,
                           HYPRE_IJVector  *b_ptr)
{
   MPI_Comm       comm = mesh->cart_comm;
   HYPRE_Int      local_size = mesh->local_size;
   HYPRE_Int     *p = &mesh->coords[0];
   HYPRE_Int     *gdims = &mesh->gdims[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_BigInt   ilower = mesh->ilower;
   HYPRE_BigInt   iupper = mesh->iupper;

   /* Create and initialize the matrix/vector */
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(comm, ilower, iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   /* We allow up to 125 neighbors (3^3=27 for radius=1, 5^3=125 for radius=2) */
   HYPRE_Int *nnzrow = (HYPRE_Int *) calloc(local_size, sizeof(HYPRE_Int));
   for (int i = 0; i < local_size; i++)
   {
      nnzrow[i] = 125; /* maximum possible neighbors including center */
   }
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   free(nnzrow);

   HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST);

   /* Some uniform negative weights for the off-diagonals. */
   const HYPRE_Real W_FACE = -1.0;   /* neighbors with |dx|+|dy|+|dz| = 1 */
   const HYPRE_Real W_OTHER = -0.01; /* further neighbors with sum of offsets ≥ 2 */

   /* Loop over local grid points */
   for (HYPRE_BigInt gz = pstarts[2][p[2]]; gz < pstarts[2][p[2] + 1]; gz++)
   {
      for (HYPRE_BigInt gy = pstarts[1][p[1]]; gy < pstarts[1][p[1] + 1]; gy++)
      {
         for (HYPRE_BigInt gx = pstarts[0][p[0]]; gx < pstarts[0][p[0] + 1]; gx++)
         {
            /* Global row index for (gx, gy, gz) */
            HYPRE_BigInt row = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, p, gdims, pstarts);

            /* We'll collect column indices & values here */
            HYPRE_BigInt cols[125];
            HYPRE_Real   vals[125];
            HYPRE_Int    nentries = 0;

            /* Baseline RHS = 0, may get overridden if boundary y=0 */
            HYPRE_Real rhs_val = 0.0;

            /* We'll sum the negative off-diagonals. Our diagonal entry will be
               diag = - ( sum_of_off_diag ), so the row sums to zero. */
            HYPRE_Real sum_offdiag = 0.0;

            /* For each offset (dx,dy,dz) in [-2..2]^3 except (0,0,0) */
            for (int dz = -2; dz <= 2; dz++)
            {
               for (int dy = -2; dy <= 2; dy++)
               {
                  for (int dx = -2; dx <= 2; dx++)
                  {
                     if (dx == 0 && dy == 0 && dz == 0) continue; /* skip center */

                     HYPRE_BigInt nx = gx + dx;
                     HYPRE_BigInt ny = gy + dy;
                     HYPRE_BigInt nz = gz + dz;

                     /* Decide the weight. This is always negative or zero. */
                     int dist1 = abs(dx) + abs(dy) + abs(dz);
                     HYPRE_Real w = 0.0;

                     if (dist1 == 1)
                     {
                        w = W_FACE;      /* face neighbors => e.g. -1.0 */
                     }
                     else
                     {
                        /* radius-2 edge/corner => -0.01 in this example */
                        w = W_OTHER;
                     }

                     if (w == 0.0)
                     {
                        continue; /* skip if zero weight */
                     }

                     /* Check if neighbor is out-of-domain => Dirichlet boundary = 0
                      * except y=0 => 1. This means we do not insert a column for
                      * out-of-domain, but we add w to the diagonal, and if that boundary
                      * is y=0 => add w to the RHS as well. */
                     if (nx < 0 || nx >= gdims[0] ||
                         ny < 0 || ny >= gdims[1] ||
                         nz < 0 || nz >= gdims[2])
                     {
                        /* If the boundary is y=0 for this row, add w to RHS: */
                        if ((gy == 0) && (dy == 0))
                        {
                           rhs_val += w;  /* boundary=1 => +w on RHS */
                        }
                        sum_offdiag += w; /* we add w to the diagonal. */
                        continue;
                     }

                     /* If neighbor is inside the global domain, figure out which process
                      * subdomain it's on (like the other stencils do). */
                     int neighbor_proc[3] = {p[0], p[1], p[2]};

                     /* Check x partition */
                     if (nx < pstarts[0][p[0]] && p[0] > 0)
                        neighbor_proc[0] = p[0] - 1;
                     else if (nx >= pstarts[0][p[0] + 1] && p[0] < mesh->pdims[0] - 1)
                        neighbor_proc[0] = p[0] + 1;

                     /* Check y partition */
                     if (ny < pstarts[1][p[1]] && p[1] > 0)
                        neighbor_proc[1] = p[1] - 1;
                     else if (ny >= pstarts[1][p[1] + 1] && p[1] < mesh->pdims[1] - 1)
                        neighbor_proc[1] = p[1] + 1;

                     /* Check z partition */
                     if (nz < pstarts[2][p[2]] && p[2] > 0)
                        neighbor_proc[2] = p[2] - 1;
                     else if (nz >= pstarts[2][p[2] + 1] && p[2] < mesh->pdims[2] - 1)
                        neighbor_proc[2] = p[2] + 1;

                     /* Possibly the neighbor is out-of-partition => treated like boundary=0 except y=0 => add w to RHS. */
                     if (   neighbor_proc[0] < 0 || neighbor_proc[0] >= mesh->pdims[0]
                         || neighbor_proc[1] < 0 || neighbor_proc[1] >= mesh->pdims[1]
                         || neighbor_proc[2] < 0 || neighbor_proc[2] >= mesh->pdims[2])
                     {
                        if ((gy == 0) && (dy == 0))
                        {
                           rhs_val += w;
                        }
                        sum_offdiag += w;
                        continue;
                     }

                     /* The neighbor is valid => we add an off-diagonal entry. */
                     HYPRE_BigInt nrow = grid2idx((HYPRE_BigInt[]){nx, ny, nz},
                                                  neighbor_proc, gdims, pstarts);

                     cols[nentries] = nrow;
                     vals[nentries] = w; /* negative or zero => M-matrix requirement */
                     nentries++;

                     /* Accumulate for the diagonal. Row sum is 0 => diagonal = - sum(offdiag). */
                     sum_offdiag += w;
                  } /* dx */
               } /* dy */
            } /* dz */

            /* Diagonal entry to ensure row sum is zero => diag = - sum_offdiag */
            HYPRE_Real diag = -sum_offdiag;

            /* Insert diagonal last */
            cols[nentries] = row;
            vals[nentries] = diag;
            nentries++;

            /* If y=0 => user wants boundary=1 on entire plane. Overwrite RHS=1. */
            if (gy == pstarts[1][p[1]] && p[1] == 0)
            {
               rhs_val = 1.0;
            }

            /* Fill matrix row and b vector. */
            HYPRE_IJMatrixSetValues(A, 1, &nentries, &row, cols, vals);
            HYPRE_IJVectorSetValues(b, 1, &row, &rhs_val);
         }
      }
   }

   /* Assemble matrix and vector */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);

   /* Optional output */
   if (params->print)
   {
      int mypid = 0;
      MPI_Comm_rank(comm, &mypid);
      if (!mypid)
      {
         printf("Printing matrix and vector: A_125pt.out and b_125pt.out...\n");
      }
      HYPRE_IJMatrixPrint(A, "A_125pt.out");
      HYPRE_IJVectorPrint(b, "b_125pt.out");
   }

   /* Return pointers to the matrix and vector */
   *A_ptr = A;
   *b_ptr = b;

   return 0;
}

/*--------------------------------------------------------------------------
 * Helper function to write coordinate arrays
 *--------------------------------------------------------------------------*/
static void
WriteCoordArray(FILE       *fp,
                const char *name,
                double      start,
                double      delta,
                int         count)
{
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", name);
   fprintf(fp, "          ");
   for (int i = 0; i < count; i++)
   {
      fprintf(fp, "%.15g ", start + i*delta);
      if ((i+1) % 6 == 0) fprintf(fp, "\n          ");
   }
   fprintf(fp, "\n        </DataArray>\n");
}

/*--------------------------------------------------------------------------
 * Write VTK solution to file
 *--------------------------------------------------------------------------*/

int
WriteVTKsolution(DistMesh       *mesh,
                 ProblemParams  *params,
                 HYPRE_Real     *sol_data)
{
   int    myid, num_procs;
   char   filename[256];
   char   stype[][10] = {"7pt", "19pt", "27pt", "125pt"};
   int    stmap = 0;
   double t0 = MPI_Wtime();

   /* Mesh info */
   HYPRE_Int     *p = &mesh->coords[0];
   HYPRE_Int     *gdims = &mesh->gdims[0];
   HYPRE_Int     *nlocal = &mesh->nlocal[0];
   HYPRE_BigInt **pstarts = &mesh->pstarts[0];
   HYPRE_Real    *gsizes = &mesh->gsizes[0];

   /* Get stencil type */
   if      (params->stencil == 7)   stmap = 0;
   else if (params->stencil == 19)  stmap = 1;
   else if (params->stencil == 27)  stmap = 2;
   else if (params->stencil == 125) stmap = 3;

   /* Get process info */
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Comm_size(mesh->cart_comm, &num_procs);

   /* Set short-hand variables */
   int nx  = (int)(nlocal[0]);
   int ny  = (int)(nlocal[1]);
   int nz  = (int)(nlocal[2]);
   int ofi = (int)(p[0] > 0 ? 1 : 0);
   int ofj = (int)(p[1] > 0 ? 1 : 0);
   int ofk = (int)(p[2] > 0 ? 1 : 0);
   int nxg = nx + ofi;
   int nyg = ny + ofj;
   int nzg = nz + ofk;
   int ix_start = (int)pstarts[0][p[0]];
   int iy_start = (int)pstarts[1][p[1]];
   int iz_start = (int)pstarts[2][p[2]];
   double x_start = (ix_start - ofi) * gsizes[0];
   double y_start = (iy_start - ofj) * gsizes[1];
   double z_start = (iz_start - ofk) * gsizes[2];

   /* Allocate buffers for ghost data */
   double *ghost_data[14];
   for (int i =  0; i <  2; i++) ghost_data[i] = calloc(ny * nz, sizeof(double)); // x-faces
   for (int i =  2; i <  4; i++) ghost_data[i] = calloc(nx * nz, sizeof(double)); // y-faces
   for (int i =  4; i <  6; i++) ghost_data[i] = calloc(nx * ny, sizeof(double)); // z-faces
   for (int i =  6; i <  8; i++) ghost_data[i] = calloc(nz, sizeof(double));      // xy-edges
   for (int i =  8; i < 10; i++) ghost_data[i] = calloc(ny, sizeof(double));      // xz-edges
   for (int i = 10; i < 12; i++) ghost_data[i] = calloc(nx, sizeof(double));      // yz-edges
   for (int i = 12; i < 14; i++) ghost_data[i] = calloc(1, sizeof(double));       // xyz-corner

   /* Exchange ghost layers using non-blocking communication */
   MPI_Request requests[28];
   int req_count = 0;

   /* X-direction exchange */
   if (mesh->nbrs[1] != MPI_PROC_NULL) { // Send right
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++)
            ghost_data[1][k*ny + j] = sol_data[k*ny*nx + j*nx + (nx-1)];
      MPI_Isend(ghost_data[1], ny*nz, MPI_DOUBLE, mesh->nbrs[1], 0, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[0] != MPI_PROC_NULL) // Receive left
      MPI_Irecv(ghost_data[0], ny*nz, MPI_DOUBLE, mesh->nbrs[0], 0, mesh->cart_comm, &requests[req_count++]);

   /* Y-direction exchange */
   if (mesh->nbrs[3] != MPI_PROC_NULL) { // Send up
      for (int k = 0; k < nz; k++)
         for (int i = 0; i < nx; i++)
            ghost_data[3][k*nx + i] = sol_data[k*ny*nx + (ny-1)*nx + i];
      MPI_Isend(ghost_data[3], nx*nz, MPI_DOUBLE, mesh->nbrs[3], 1, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL) // Receive down
      MPI_Irecv(ghost_data[2], nx*nz, MPI_DOUBLE, mesh->nbrs[2], 1, mesh->cart_comm, &requests[req_count++]);

   /* Z-direction exchange */
   if (mesh->nbrs[5] != MPI_PROC_NULL) { // Send front
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++)
            ghost_data[5][j*nx + i] = sol_data[(nz-1)*ny*nx + j*nx + i];
      MPI_Isend(ghost_data[5], nx*ny, MPI_DOUBLE, mesh->nbrs[5], 2, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[4] != MPI_PROC_NULL) // Receive back
      MPI_Irecv(ghost_data[4], nx*ny, MPI_DOUBLE, mesh->nbrs[4], 2, mesh->cart_comm, &requests[req_count++]);

   /* XY-direction exchange */
   if (mesh->nbrs[7] != MPI_PROC_NULL) { // Send right-up
      for (int k = 0; k < nz; k++)
         ghost_data[7][k] = sol_data[k*ny*nx + (ny-1)*nx + (nx-1)];
      MPI_Isend(ghost_data[7], nz, MPI_DOUBLE, mesh->nbrs[7], 3, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[6] != MPI_PROC_NULL) // Receive left-down
      MPI_Irecv(ghost_data[6], nz, MPI_DOUBLE, mesh->nbrs[6], 3, mesh->cart_comm, &requests[req_count++]);

   /* XZ-direction exchange */
   if (mesh->nbrs[9] != MPI_PROC_NULL) { // Send right-front
      for (int j = 0; j < ny; j++)
         ghost_data[9][j] = sol_data[(nz-1)*ny*nx + j*nx + (nx-1)];
      MPI_Isend(ghost_data[9], ny, MPI_DOUBLE, mesh->nbrs[9], 4, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[8] != MPI_PROC_NULL) // Receive left-back
      MPI_Irecv(ghost_data[8], ny, MPI_DOUBLE, mesh->nbrs[8], 4, mesh->cart_comm, &requests[req_count++]);

   /* YZ-direction exchange */
   if (mesh->nbrs[11] != MPI_PROC_NULL) { // Send up-front
      for (int i = 0; i < nx; i++)
         ghost_data[11][i] = sol_data[(nz-1)*ny*nx + (ny-1)*nx + i];
      MPI_Isend(ghost_data[11], nx, MPI_DOUBLE, mesh->nbrs[11], 5, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[10] != MPI_PROC_NULL) // Receive down-back
      MPI_Irecv(ghost_data[10], nx, MPI_DOUBLE, mesh->nbrs[10], 5, mesh->cart_comm, &requests[req_count++]);

   /* XYZ-direction exchange */
   if (mesh->nbrs[13] != MPI_PROC_NULL) { // Send right-up-front
      ghost_data[13][0] = sol_data[(nz-1)*ny*nx + (ny-1)*nx + (nx-1)];
      MPI_Isend(ghost_data[13], 1, MPI_DOUBLE, mesh->nbrs[13], 6, mesh->cart_comm, &requests[req_count++]);
   }
   if (mesh->nbrs[12] != MPI_PROC_NULL) // Receive left-down-back
      MPI_Irecv(ghost_data[12], 1, MPI_DOUBLE, mesh->nbrs[12], 6, mesh->cart_comm, &requests[req_count++]);

   /* Create extended solution array including ghost points */
   double *extended_data = calloc(nxg * nyg * nzg, sizeof(double));

   /* Copy interior solution data */
   for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++)
         {
            int ig = i + ofi;
            int jg = j + ofj;
            int kg = k + ofk;
            extended_data[(kg*nyg + jg)*nxg + ig] = sol_data[(k*ny + j)*nx + i];
         }

   /* Wait for all communications to complete */
   MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

   /* Copy ghost data */
   if (mesh->nbrs[0] != MPI_PROC_NULL) // Left face (x = 0)
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++)
         {
            int jg = j + ofj;
            int kg = k + ofk;
            extended_data[(kg*nyg + jg)*nxg + 0] = ghost_data[0][k*ny + j];
         }

   if (mesh->nbrs[2] != MPI_PROC_NULL) // Bottom face (y = 0)
      for (int k = 0; k < nz; k++)
         for (int i = 0; i < nx; i++)
         {
            int ig = i + ofi;
            int kg = k + ofk;
            extended_data[(kg*nyg + 0)*nxg + ig] = ghost_data[2][k*nx + i];
         }

   if (mesh->nbrs[4] != MPI_PROC_NULL) // Back face (z = 0)
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++)
         {
            int ig = i + ofi;
            int jg = j + ofj;
            extended_data[(0*nyg + jg)*nxg + ig] = ghost_data[4][j*nx + i];
         }

   if (mesh->nbrs[6] != MPI_PROC_NULL) // Left-down (x- y-)
     for (int k = 0; k < nz; k++)
     {
        int kg = k + ofk;
        extended_data[(kg*nyg + 0)*nxg + 0] = ghost_data[6][k];
     }

   if (mesh->nbrs[8] != MPI_PROC_NULL) // Left-back (x- z-)
     for (int j = 0; j < ny; j++)
     {
        int jg = j + ofj;
        extended_data[(0*nyg + jg)*nxg + 0] = ghost_data[8][j];
     }

   if (mesh->nbrs[10] != MPI_PROC_NULL) // Down-back (y- z-)
     for (int i = 0; i < nx; i++)
     {
        int ig = i + ofi;
        extended_data[(0*nyg + 0)*nxg + ig] = ghost_data[10][i];
     }

   if (mesh->nbrs[12] != MPI_PROC_NULL) // Left-down-back (x- y- z-)
     extended_data[(0*nyg + 0)*nxg + 0] = ghost_data[12][0];

   /* Write VTK file */
   snprintf(filename, sizeof(filename), "laplacian_%s_%dx%dx%d_%dx%dx%d_%d.vtr",
            stype[stmap],
            params->N[0], params->N[1], params->N[2],
            params->P[0], params->P[1], params->P[2], myid);
   FILE *fp = fopen(filename, "w");
   if (!fp) { printf("Error: Cannot open file %s\n", filename); MPI_Abort(mesh->cart_comm, -1); }

   fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
   fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\">\n");
   fprintf(fp, "  <RectilinearGrid WholeExtent=\"%d %d %d %d %d %d\">\n",
           ix_start - ofi, ix_start + nx - 1,
           iy_start - ofj, iy_start + ny - 1,
           iz_start - ofk, iz_start + nz - 1);
   fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n",
           ix_start - ofi, ix_start + nx - 1,
           iy_start - ofj, iy_start + ny - 1,
           iz_start - ofk, iz_start + nz - 1);

   /* Write coordinates */
   fprintf(fp, "      <Coordinates>\n");
   WriteCoordArray(fp, "x", x_start, gsizes[0], nxg);
   WriteCoordArray(fp, "y", y_start, gsizes[1], nyg);
   WriteCoordArray(fp, "z", z_start, gsizes[2], nzg);
   fprintf(fp, "      </Coordinates>\n");

   /* Write solution data */
   fprintf(fp, "      <PointData Scalars=\"solution\">\n");
#ifdef VTK_USE_ASCII
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"solution\" format=\"ascii\">\n");
   fprintf(fp, "          ");
   for (int k = 0; k < nzg; k++)
      for (int j = 0; j < nyg; j++)
         for (int i = 0; i < nxg; i++)
         {
            fprintf(fp, "%.15g ", extended_data[(k*nyg + j)*nxg + i]);
            if (((k*nyg + j)*nxg + i + 1) % 6 == 0) fprintf(fp, "\n          ");
         }
   fprintf(fp, "\n        </DataArray>\n");
   fprintf(fp, "      </PointData>\n");
   fprintf(fp, "    </Piece>\n");
   fprintf(fp, "  </RectilinearGrid>\n");
#else
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"solution\" format=\"appended\" offset=\"0\">\n");
   fprintf(fp, "        </DataArray>\n");
   fprintf(fp, "      </PointData>\n");
   fprintf(fp, "    </Piece>\n");
   fprintf(fp, "  </RectilinearGrid>\n");

   /* Write appended binary data */
   fprintf(fp, "  <AppendedData encoding=\"raw\">\n   _");
   int data_size = nxg * nyg * nzg * sizeof(double);
   fwrite(&data_size, sizeof(int), 1, fp);
   fwrite(extended_data, sizeof(double), nxg * nyg * nzg, fp);
   fprintf(fp, "\n  </AppendedData>\n");
#endif
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);

   /* Create PVD file (rank 0 only) */
   if (!myid)
   {
      snprintf(filename, sizeof(filename), "laplacian_%s_%dx%dx%d_%dx%dx%d.pvd",
               stype[stmap],
               params->N[0], params->N[1], params->N[2],
               params->P[0], params->P[1], params->P[2]);
      fp = fopen(filename, "w");
      if (!fp) { printf("Error: Cannot open %s\n", filename); MPI_Abort(mesh->cart_comm, -1); }
      printf("Writing PVD file %s ...", filename);

      fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
      fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n");
      fprintf(fp, "  <Collection>\n");
      for (int i = 0; i < num_procs; i++)
         fprintf(fp, "    <DataSet part=\"%d\" file=\"laplacian_%s_%dx%dx%d_%dx%dx%d_%d.vtr\"/>\n", i,
                 stype[stmap],
                 params->N[0], params->N[1], params->N[2],
                 params->P[0], params->P[1], params->P[2], i);
      fprintf(fp, "  </Collection>\n");
      fprintf(fp, "</VTKFile>\n");
      fclose(fp);
      printf(" Time elapsed: %.4f s\n", MPI_Wtime() - t0);
   }

   /* Cleanup */
   for (int i = 0; i < 14; i++) free(ghost_data[i]);
   free(extended_data);

   return 0;
}
