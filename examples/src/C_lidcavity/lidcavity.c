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
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "HYPREDRV.h"
#include "compatibility.h"

#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 21900
#define HYPREDRV_IJ_MATRIX_INIT_HOST(mat) \
   HYPRE_IJMatrixInitialize_v2((mat), HYPRE_MEMORY_HOST)
#define HYPREDRV_IJ_VECTOR_INIT_HOST(vec) \
   HYPRE_IJVectorInitialize_v2((vec), HYPRE_MEMORY_HOST)
#else
#define HYPREDRV_IJ_MATRIX_INIT_HOST(mat) HYPRE_IJMatrixInitialize((mat))
#define HYPREDRV_IJ_VECTOR_INIT_HOST(vec) HYPRE_IJVectorInitialize((vec))
#endif

/*==========================================================================
 *   2D Lid Driven Cavity (Q1-Q1 Stabilized) Example Driver
 *==========================================================================
 *
 *   Geometry and Axes:
 *   ------------------
 *
 *        y=L  +-------lid (u=1)-------+
 *             |                       |
 *             |                       |
 *             |                       |
 *        wall |         Domain        | wall
 *       (u=0) |       [0,L]x[0,L]     | (u=0)
 *             |                       |          P[1]
 *             |                       |          ^
 *             |                       |          |
 *             |                       |          |
 *        y=0  +---------wall----------+          +-----> P[0]
 *            x=0                     x=L    Grid Partitioning
 *
 *   Physics (Unsteady Incompressible Navier-Stokes):
 *   ------------------------------------------------
 *
 *     du/dt + (U . grad) U - ν * lap(U) + grad(p) = 0   (momentum)
 *     div(U) = 0                                        (continuity)
 *
 *   where:
 *     U = (u, v)  velocity vector
 *     p           pressure (normalized by density)
 *     Re          Reynolds number
 *     ν = 1/Re    kinematic viscosity
 *
 *   Boundary Conditions:
 *   --------------------
 *   - Lid (y = L, excluding corners): u = 1, v = 0
 *       Classic:     u = 1 (discontinuous at corners)
 *       Regularized: u = [1 - (2x/L - 1)^16]^2 (smooth, zero at corners)
 *   - Walls (x = 0, x = L, y = 0, and corners): u = 0, v = 0 (no-slip)
 *   - Pressure: Fixed at one node (e.g., bottom-left corner, p = 0)
 *
 *   Discretization:
 *   ---------------
 *   - Spatial:  Q1-Q1 finite elements (bilinear velocity and pressure)
 *   - Temporal: Backward Euler (implicit, first-order)
 *   - Stabilization: SUPG (momentum) + PSPG (pressure)
 *   - Nonlinear: Newton iteration with analytical Jacobian
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Default linear solver configuration (FGMRES+AMG+ILU)
 *--------------------------------------------------------------------------*/
static const char *default_config = "solver:\n"
                                    "  fgmres:\n"
                                    "    krylov_dim: 100\n"
                                    "    max_iter: 300\n"
                                    "    print_level: 0\n"
                                    "    relative_tol: 1.0e-6\n"
                                    " preconditioner:\n"
                                    "  amg:\n"
                                    "    print_level: 0\n"
                                    "    coarsening:\n"
                                    "      strong_th: 0.6\n"
                                    "      num_functions: 3\n"
                                    "    smoother:\n"
                                    "      type: ilu\n"
                                    "      num_levels: 5\n"
                                    "      num_sweeps: 1\n"
                                    "      ilu:\n"
                                    "        type: bj-ilut\n"
                                    "        fill_level: 0\n"
                                    "        droptol: 1e-2\n";

/*--------------------------------------------------------------------------
 * Problem parameters struct
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int  visualize;         /* Visualize displacement in VTK format */
   HYPRE_Int  verbose;           /* Verbosity level bitset */
   HYPRE_Real final_time;        /* Final simulation time */
   HYPRE_Int  N[2];              /* Grid dimensions (nodes) */
   HYPRE_Int  P[2];              /* Processor grid dims */
   HYPRE_Real L[2];              /* Physical dimensions (Lx, Ly) */
   HYPRE_Int  max_rows_per_call; /* Max rows per IJ add/set call */
   HYPRE_Real Re;                /* Reynolds number */
   HYPRE_Real dt;                /* Time step */
   HYPRE_Real newton_tol;        /* Non-linear solver tolerance */
   HYPRE_Int  adaptive_dt;       /* Adaptive time stepping flag */
   HYPRE_Real max_cfl;           /* Maximum CFL for adaptive time stepping (0=no limit) */
   HYPRE_Int  regularize_bc;     /* Use regularized lid BC (smooth corners) */
   char      *yaml_file;         /* YAML configuration file */
} LidCavityParams;

/*--------------------------------------------------------------------------
 * Distributed Mesh struct (reused)
 *--------------------------------------------------------------------------*/
typedef struct
{
   MPI_Comm      cart_comm;
   HYPRE_Int     mypid;
   HYPRE_Int     gdims[2];
   HYPRE_Int     pdims[2];
   HYPRE_Int     coords[2];
   HYPRE_Int     nlocal[2];
   HYPRE_Int     nbrs[8];
   HYPRE_Real    h[3];       /* local length array */
   HYPRE_Int     local_size; /* local number of nodes */
   HYPRE_BigInt  ilower;     /* first local node gid (scalar) */
   HYPRE_BigInt  iupper;     /* last  local node gid (scalar) */
   HYPRE_BigInt  dof_ilower; /* first local row (in global numbering) */
   HYPRE_BigInt  dof_iupper; /* last  local row (in global numbering) */
   HYPRE_BigInt *pstarts[2];
   HYPRE_Real    gsizes[2];
} DistMesh2D;

/*--------------------------------------------------------------------------
 * Ghost Data struct
 *--------------------------------------------------------------------------*/
typedef struct
{
   double     *recv_bufs[8]; /* 0:L, 1:R, 2:B, 3:T, 4:BL, 5:TR, 6:TL, 7:BR */
   double     *send_bufs[8];
   MPI_Request reqs[16];
} GhostData;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/
static inline HYPRE_BigInt grid2idx(const HYPRE_BigInt gcoords[2],
                                    const HYPRE_Int bcoords[2], const HYPRE_Int gdims[2],
                                    HYPRE_BigInt **pstarts);

int PrintUsage(void);
int CreateDistMesh2D(MPI_Comm, HYPRE_Real, HYPRE_Real, HYPRE_Int, HYPRE_Int, HYPRE_Int,
                     HYPRE_Int, DistMesh2D **);
int DestroyDistMesh2D(DistMesh2D **);
int CreateGhostData(DistMesh2D *, GhostData **);
int DestroyGhostData(GhostData **);
int ExchangeVectorGhosts(DistMesh2D *, double *, GhostData *);
int ParseArguments(int, char **, LidCavityParams *, int, int);
int BuildNewtonSystem(DistMesh2D *, LidCavityParams *, HYPRE_Real *, HYPRE_Real *,
                      HYPRE_IJMatrix *, HYPRE_IJVector *, HYPRE_Real *, GhostData *,
                      GhostData *);
double *ComputeDivergence(DistMesh2D *, LidCavityParams *, double *, GhostData *);
int     WriteVTKsolutionVector(DistMesh2D *, LidCavityParams *, HYPRE_Real *, GhostData *,
                               int);
void    GetVTKBaseName(LidCavityParams *, char *, size_t);
void    GetVTKDataDir(LidCavityParams *, char *, size_t);
void    WritePVDCollectionFromStats(LidCavityParams *, int, double);
void    UpdateTimeStep(LidCavityParams *, int, HYPRE_Real);

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/
int
PrintUsage(void)
{
   printf("\n");
   printf("Usage: ${MPIEXEC_COMMAND} <np> ./lidcavity [options]\n");
   printf("\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML configuration file for solver settings (Opt.)\n");
   printf("  -n <nx> <ny>      : Global grid dimensions in nodes (32 32)\n");
   printf("  -P <Px> <Py>      : Processor grid dimensions (1 1)\n");
   printf("  -L <Lx> <Ly>      : Physical dimensions (1 1)\n");
   printf("  -Re <val>         : Reynolds number (100.0)\n");
   printf("  -dt <val>         : Initial time step size (1)\n");
   printf("  -tf <val>         : Final simulation time (50)\n");
   printf("  -ntol <val>       : Non-linear solver tolerance (1.0e-6)\n");
   printf("  -adt              : Enable simple adaptive time stepping\n");
   printf("  -cfl <val>        : Maximum CFL for adaptive time stepping (0=no limit)\n");
   printf("  -reg              : Use regularized lid BC (smooth corners)\n");
   printf("  -br <n>           : Batch rows for matrix assembly (128)\n");
   printf("  -vis <m>          : Visualization mode bitset (0)\n");
   printf("                         Any nonzero value enables visualization\n");
   printf("                         Bit 2 (0x2): ASCII (1) or binary (0)\n");
   printf("                         Bit 3 (0x4): All timesteps (1) or last only (0)\n");
   printf("                         Examples: 1=binary last, 4=binary all\n");
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
ParseArguments(int argc, char *argv[], LidCavityParams *params, int myid, int num_procs)
{
   /* Defaults */
   params->visualize  = 0;
   params->verbose    = 3;
   params->final_time = 50;
   for (int i = 0; i < 2; i++)
   {
      params->N[i] = 32;
      params->P[i] = 1;
      params->L[i] = 1.0;
   }
   params->max_rows_per_call = 128;
   params->Re                = 100.0;
   params->dt                = 1.0;
   params->newton_tol        = 1.0e-6;
   params->adaptive_dt       = 0;
   params->max_cfl           = 0.0; /* 0 means no limit */
   params->regularize_bc     = 0;
   params->yaml_file         = NULL;

   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))
      {
         if (++i < argc) params->yaml_file = argv[i];
      }
      else if (!strcmp(argv[i], "-n"))
      {
         if (i + 2 >= argc)
         {
            if (!myid) printf("Error: -n requires two values\n");
            return 1;
         }
         for (int j = 0; j < 2; j++)
         {
            params->N[j] = atoi(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-P"))
      {
         if (i + 2 >= argc)
         {
            if (!myid) printf("Error: -P requires two values\n");
            return 1;
         }
         for (int j = 0; j < 2; j++)
         {
            params->P[j] = atoi(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-L"))
      {
         if (i + 2 >= argc)
         {
            if (!myid) printf("Error: -L requires two values\n");
            return 1;
         }
         for (int j = 0; j < 2; j++)
         {
            params->L[j] = atof(argv[++i]);
         }
      }
      else if (!strcmp(argv[i], "-Re"))
      {
         if (++i < argc) params->Re = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-dt"))
      {
         if (++i < argc) params->dt = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-adt"))
      {
         params->adaptive_dt = 1;
      }
      else if (!strcmp(argv[i], "-cfl") || !strcmp(argv[i], "--max-cfl"))
      {
         if (++i < argc) params->max_cfl = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-reg") || !strcmp(argv[i], "--regularize"))
      {
         params->regularize_bc = 1;
      }
      else if (!strcmp(argv[i], "-br") || !strcmp(argv[i], "--batch-rows"))
      {
         if (++i < argc) params->max_rows_per_call = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-tf") || !strcmp(argv[i], "--final-time"))
      {
         if (++i < argc) params->final_time = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-ntol") || !strcmp(argv[i], "--newton-tol"))
      {
         if (++i < argc) params->newton_tol = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-vis") || !strcmp(argv[i], "--visualize"))
      {
         if (i + 1 < argc && argv[i + 1][0] != '-')
         {
            params->visualize = atoi(argv[++i]);
         }
         else
         {
            /* No argument: default is 0 (disabled) */
            params->visualize = 0;
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
   if (params->P[0] * params->P[1] != num_procs)
   {
      if (!myid)
      {
         printf("Error: Number of processes (%d) must match processor grid "
                "dimensions (%d x %d = %d)\n",
                num_procs, params->P[0], params->P[1], params->P[0] * params->P[1]);
      }
      return 1;
   }

   for (int d = 0; d < 2; d++)
   {
      char *name[] = {"First", "Second"};
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

   return 0;
}

/*--------------------------------------------------------------------------
 * Compute global index from global grid coordinates
 *--------------------------------------------------------------------------*/
static inline HYPRE_BigInt
grid2idx(const HYPRE_BigInt gcoords[2], const HYPRE_Int bcoords[2],
         const HYPRE_Int gdims[2], HYPRE_BigInt **pstarts)
{
   return pstarts[1][bcoords[1]] * (HYPRE_BigInt)gdims[0] +
          pstarts[0][bcoords[0]] * (pstarts[1][bcoords[1] + 1] - pstarts[1][bcoords[1]]) +
          (gcoords[1] - pstarts[1][bcoords[1]]) *
             (pstarts[0][bcoords[0] + 1] - pstarts[0][bcoords[0]]) +
          (gcoords[0] - pstarts[0][bcoords[0]]);
}

/*--------------------------------------------------------------------------
 * Create mesh partition information (reused from Laplacian example)
 *--------------------------------------------------------------------------*/
int
CreateDistMesh2D(MPI_Comm comm, HYPRE_Real Lx, HYPRE_Real Ly, HYPRE_Int Nx, HYPRE_Int Ny,
                 HYPRE_Int Px, HYPRE_Int Py, DistMesh2D **mesh_ptr)
{
   DistMesh2D *mesh = (DistMesh2D *)malloc(sizeof(DistMesh2D));
   int         myid;

   mesh->gdims[0] = Nx;
   mesh->gdims[1] = Ny;
   mesh->pdims[0] = Px;
   mesh->pdims[1] = Py;
   mesh->h[0]     = Lx / (Nx - 1);
   mesh->h[1]     = Ly / (Ny - 1);
   mesh->h[2]     = (mesh->h[0] < mesh->h[1]) ? mesh->h[0] : mesh->h[1];

   /* Use int arrays for MPI calls to avoid type mismatch if HYPRE_Int != int */
   int mpi_dims[2]   = {(int)Px, (int)Py};
   int mpi_coords[2] = {0, 0};
   int mpi_nbrs[8]   = {0};

   /* IMPORTANT: keep cart ranks consistent with MPI_COMM_WORLD ranks.
    * Hypre/HYPREDRV objects in this example are built on MPI_COMM_WORLD, so if MPI
    * reorders ranks here (reorder=1), the global row partition can become inconsistent
    * with the communicator rank ordering and can trigger failures in ParCSR comm pkg. */
   MPI_Cart_create(comm, 2, mpi_dims, (int[]){0, 0}, 0, &(mesh->cart_comm));
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Cart_coords(mesh->cart_comm, myid, 2, mpi_coords);

   /* Copy to HYPRE_Int members */
   mesh->coords[0] = mpi_coords[0];
   mesh->coords[1] = mpi_coords[1];
   mesh->mypid     = (HYPRE_Int)myid;

   for (int i = 0; i < 2; i++)
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

   HYPRE_BigInt gcoords[2] = {mesh->pstarts[0][mesh->coords[0]],
                              mesh->pstarts[1][mesh->coords[1]]};
   mesh->ilower            = grid2idx(gcoords, mesh->coords, mesh->gdims, mesh->pstarts);
   mesh->local_size        = mesh->nlocal[0] * mesh->nlocal[1];
   mesh->iupper            = mesh->ilower + mesh->local_size - 1;
   mesh->dof_ilower        = mesh->ilower * 3;
   mesh->dof_iupper        = mesh->iupper * 3 + 2;

   /* Get face neighbors using int temporaries */
   MPI_Cart_shift(mesh->cart_comm, 0, 1, &mpi_nbrs[0], &mpi_nbrs[1]);
   MPI_Cart_shift(mesh->cart_comm, 1, 1, &mpi_nbrs[2], &mpi_nbrs[3]);

   /* Get diagonal neighbors */
   if (mpi_coords[0] > 0 && mpi_coords[1] > 0)
      MPI_Cart_rank(mesh->cart_comm, (int[]){mpi_coords[0] - 1, mpi_coords[1] - 1},
                    &mpi_nbrs[4]);
   else mpi_nbrs[4] = MPI_PROC_NULL;

   if (mpi_coords[0] < mpi_dims[0] - 1 && mpi_coords[1] < mpi_dims[1] - 1)
      MPI_Cart_rank(mesh->cart_comm, (int[]){mpi_coords[0] + 1, mpi_coords[1] + 1},
                    &mpi_nbrs[5]);
   else mpi_nbrs[5] = MPI_PROC_NULL;

   if (mpi_coords[0] > 0 && mpi_coords[1] < mpi_dims[1] - 1)
      MPI_Cart_rank(mesh->cart_comm, (int[]){mpi_coords[0] - 1, mpi_coords[1] + 1},
                    &mpi_nbrs[6]);
   else mpi_nbrs[6] = MPI_PROC_NULL;

   if (mpi_coords[0] < mpi_dims[0] - 1 && mpi_coords[1] > 0)
      MPI_Cart_rank(mesh->cart_comm, (int[]){mpi_coords[0] + 1, mpi_coords[1] - 1},
                    &mpi_nbrs[7]);
   else mpi_nbrs[7] = MPI_PROC_NULL;

   /* Copy neighbors to HYPRE_Int array */
   for (int i = 0; i < 8; i++)
   {
      mesh->nbrs[i] = mpi_nbrs[i];
   }

   *mesh_ptr = mesh;
   return 0;
}

/*--------------------------------------------------------------------------
 * Ghost Data struct and functions
 *--------------------------------------------------------------------------*/

int
CreateGhostData(DistMesh2D *mesh, GhostData **ghost_ptr)
{
   GhostData *g  = (GhostData *)calloc(1, sizeof(GhostData));
   int        nx = mesh->nlocal[0];
   int        ny = mesh->nlocal[1];

   if (mesh->nbrs[0] != MPI_PROC_NULL)
   {
      g->recv_bufs[0] = (double *)calloc((size_t)(ny * 3), sizeof(double));
      g->send_bufs[0] = (double *)calloc((size_t)(ny * 3), sizeof(double));
   }
   if (mesh->nbrs[1] != MPI_PROC_NULL)
   {
      g->recv_bufs[1] = (double *)calloc((size_t)(ny * 3), sizeof(double));
      g->send_bufs[1] = (double *)calloc((size_t)(ny * 3), sizeof(double));
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL)
   {
      g->recv_bufs[2] = (double *)calloc((size_t)(nx * 3), sizeof(double));
      g->send_bufs[2] = (double *)calloc((size_t)(nx * 3), sizeof(double));
   }
   if (mesh->nbrs[3] != MPI_PROC_NULL)
   {
      g->recv_bufs[3] = (double *)calloc((size_t)(nx * 3), sizeof(double));
      g->send_bufs[3] = (double *)calloc((size_t)(nx * 3), sizeof(double));
   }
   for (int i = 4; i < 8; i++)
      if (mesh->nbrs[i] != MPI_PROC_NULL)
      {
         g->recv_bufs[i] = (double *)calloc(3, sizeof(double));
         g->send_bufs[i] = (double *)calloc(3, sizeof(double));
      }

   *ghost_ptr = g;
   return 0;
}

int
DestroyGhostData(GhostData **ghost_ptr)
{
   GhostData *g = *ghost_ptr;
   if (!g) return 0;
   for (int i = 0; i < 8; i++)
   {
      if (g->recv_bufs[i]) free(g->recv_bufs[i]);
      if (g->send_bufs[i]) free(g->send_bufs[i]);
   }
   free(g);
   *ghost_ptr = NULL;
   return 0;
}

int
ExchangeVectorGhosts(DistMesh2D *mesh, double *vec, GhostData *g)
{
   int nx   = mesh->nlocal[0];
   int ny   = mesh->nlocal[1];
   int reqc = 0;

   // Post Recvs (Tag = my_dir_index)
   // We expect neighbor to send with Tag = my_dir_index (their dest index relative to
   // them? No). P0 recv from P1 (Right). P1 sends to P0 (Left). Scheme: Send Tag =
   // Dest_Direction. P1 sends to P0 (Left=0). Tag=0. P0 recvs from P1 (Right=1). Expect
   // Tag=0.

   // Recv logic:
   if (mesh->nbrs[0] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[0], ny * 3, MPI_DOUBLE, mesh->nbrs[0], 0, mesh->cart_comm,
                &g->reqs[reqc++]);
   if (mesh->nbrs[1] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[1], ny * 3, MPI_DOUBLE, mesh->nbrs[1], 1, mesh->cart_comm,
                &g->reqs[reqc++]);
   if (mesh->nbrs[2] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[2], nx * 3, MPI_DOUBLE, mesh->nbrs[2], 2, mesh->cart_comm,
                &g->reqs[reqc++]);
   if (mesh->nbrs[3] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[3], nx * 3, MPI_DOUBLE, mesh->nbrs[3], 3, mesh->cart_comm,
                &g->reqs[reqc++]);

   if (mesh->nbrs[4] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[4], 3, MPI_DOUBLE, mesh->nbrs[4], 4, mesh->cart_comm,
                &g->reqs[reqc++]);
   if (mesh->nbrs[5] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[5], 3, MPI_DOUBLE, mesh->nbrs[5], 5, mesh->cart_comm,
                &g->reqs[reqc++]);
   if (mesh->nbrs[6] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[6], 3, MPI_DOUBLE, mesh->nbrs[6], 6, mesh->cart_comm,
                &g->reqs[reqc++]);
   if (mesh->nbrs[7] != MPI_PROC_NULL)
      MPI_Irecv(g->recv_bufs[7], 3, MPI_DOUBLE, mesh->nbrs[7], 7, mesh->cart_comm,
                &g->reqs[reqc++]);

   /* Pack and Send (Tag = My Direction relative to neighbor) */

   // Send logic:
   if (mesh->nbrs[0] != MPI_PROC_NULL)
   {
      for (int j = 0; j < ny; j++)
      {
         int idx                    = (j * nx + 0) * 3;
         g->send_bufs[0][3 * j + 0] = vec[idx + 0];
         g->send_bufs[0][3 * j + 1] = vec[idx + 1];
         g->send_bufs[0][3 * j + 2] = vec[idx + 2];
      }
      // Send to Left (0). But neighbor P_left receives from Right (1).
      // If neighbor expects Tag 1, I must send Tag 1.
      // Neighbor P_left: MPI_Irecv(..., P_me, 1) -> Expects Tag 1.
      MPI_Isend(g->send_bufs[0], ny * 3, MPI_DOUBLE, mesh->nbrs[0], 1, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (mesh->nbrs[1] != MPI_PROC_NULL)
   {
      for (int j = 0; j < ny; j++)
      {
         int idx                    = (j * nx + (nx - 1)) * 3;
         g->send_bufs[1][3 * j + 0] = vec[idx + 0];
         g->send_bufs[1][3 * j + 1] = vec[idx + 1];
         g->send_bufs[1][3 * j + 2] = vec[idx + 2];
      }
      // Send to Right (1). Neighbor P_right receives from Left (0). Expects Tag 0.
      MPI_Isend(g->send_bufs[1], ny * 3, MPI_DOUBLE, mesh->nbrs[1], 0, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL)
   {
      for (int i = 0; i < nx; i++)
      {
         int idx                    = (0 * nx + i) * 3;
         g->send_bufs[2][3 * i + 0] = vec[idx + 0];
         g->send_bufs[2][3 * i + 1] = vec[idx + 1];
         g->send_bufs[2][3 * i + 2] = vec[idx + 2];
      }
      // Send to Bottom (2). Neighbor P_down receives from Top (3). Expects Tag 3.
      MPI_Isend(g->send_bufs[2], nx * 3, MPI_DOUBLE, mesh->nbrs[2], 3, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (mesh->nbrs[3] != MPI_PROC_NULL)
   {
      for (int i = 0; i < nx; i++)
      {
         int idx                    = ((ny - 1) * nx + i) * 3;
         g->send_bufs[3][3 * i + 0] = vec[idx + 0];
         g->send_bufs[3][3 * i + 1] = vec[idx + 1];
         g->send_bufs[3][3 * i + 2] = vec[idx + 2];
      }
      // Send to Top (3). Neighbor P_up receives from Bottom (2). Expects Tag 2.
      MPI_Isend(g->send_bufs[3], nx * 3, MPI_DOUBLE, mesh->nbrs[3], 2, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

   // Diagonals
   // 4: Down-Left. Opp: Up-Right (5). Send Tag 5.
   if (mesh->nbrs[4] != MPI_PROC_NULL)
   {
      int idx            = (0 * nx + 0) * 3;
      g->send_bufs[4][0] = vec[idx + 0];
      g->send_bufs[4][1] = vec[idx + 1];
      g->send_bufs[4][2] = vec[idx + 2];
      MPI_Isend(g->send_bufs[4], 3, MPI_DOUBLE, mesh->nbrs[4], 5, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   // 5: Up-Right. Opp: Down-Left (4). Send Tag 4.
   if (mesh->nbrs[5] != MPI_PROC_NULL)
   {
      int idx            = ((ny - 1) * nx + (nx - 1)) * 3;
      g->send_bufs[5][0] = vec[idx + 0];
      g->send_bufs[5][1] = vec[idx + 1];
      g->send_bufs[5][2] = vec[idx + 2];
      MPI_Isend(g->send_bufs[5], 3, MPI_DOUBLE, mesh->nbrs[5], 4, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   // 6: Up-Left. Opp: Down-Right (7). Send Tag 7.
   if (mesh->nbrs[6] != MPI_PROC_NULL)
   {
      int idx            = ((ny - 1) * nx + 0) * 3;
      g->send_bufs[6][0] = vec[idx + 0];
      g->send_bufs[6][1] = vec[idx + 1];
      g->send_bufs[6][2] = vec[idx + 2];
      MPI_Isend(g->send_bufs[6], 3, MPI_DOUBLE, mesh->nbrs[6], 7, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   // 7: Down-Right. Opp: Up-Left (6). Send Tag 6.
   if (mesh->nbrs[7] != MPI_PROC_NULL)
   {
      int idx            = (0 * nx + (nx - 1)) * 3;
      g->send_bufs[7][0] = vec[idx + 0];
      g->send_bufs[7][1] = vec[idx + 1];
      g->send_bufs[7][2] = vec[idx + 2];
      MPI_Isend(g->send_bufs[7], 3, MPI_DOUBLE, mesh->nbrs[7], 6, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

   if (reqc > 0)
   {
      MPI_Status statuses[16];
      MPI_Waitall(reqc, g->reqs, statuses);
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy distributed mesh structure
 *--------------------------------------------------------------------------*/
int
DestroyDistMesh2D(DistMesh2D **mesh_ptr)
{
   DistMesh2D *mesh = *mesh_ptr;
   if (!mesh) return 0;

   MPI_Comm_free(&(mesh->cart_comm));
   for (int i = 0; i < 2; i++) free(mesh->pstarts[i]);
   free(mesh);
   *mesh_ptr = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * 2D Q1 shape functions
 *--------------------------------------------------------------------------*/
static const int        quad_sgn[4][2] = {{-1, -1}, {+1, -1}, {+1, +1}, {-1, +1}};
static const HYPRE_Real gauss_x[2]     = {-0.5773502691896257, 0.5773502691896257};
static const HYPRE_Real gauss_w[2]     = {1.0, 1.0};

static void
q1_shape_ref_2d(const HYPRE_Real xi, const HYPRE_Real eta, HYPRE_Real N[4],
                HYPRE_Real dN_dxi[4], HYPRE_Real dN_deta[4])
{
   for (int a = 0; a < 4; a++)
   {
      HYPRE_Real sx = quad_sgn[a][0];
      HYPRE_Real sy = quad_sgn[a][1];
      HYPRE_Real f1 = (1.0 + sx * xi);
      HYPRE_Real f2 = (1.0 + sy * eta);
      N[a]          = 0.25 * f1 * f2;
      dN_dxi[a]     = 0.25 * sx * f2;
      dN_deta[a]    = 0.25 * sy * f1;
   }
}

static void
get_global_coords(HYPRE_BigInt global_node_idx, DistMesh2D *mesh, HYPRE_BigInt gcoords[2])
{
   // Calculate local index on this processor
   HYPRE_BigInt local_node_idx_on_proc = global_node_idx - mesh->ilower;
   // Calculate local x and y coordinates on this processor
   HYPRE_BigInt local_x = local_node_idx_on_proc % mesh->nlocal[0];
   HYPRE_BigInt local_y = local_node_idx_on_proc / mesh->nlocal[0];
   // Convert local processor coordinates to global coordinates
   gcoords[0] = local_x + mesh->pstarts[0][mesh->coords[0]];
   gcoords[1] = local_y + mesh->pstarts[1][mesh->coords[1]];
}

void
ComputeComponentNorms(HYPRE_IJVector b, HYPRE_Int local_size, double *norm_u,
                      double *norm_v, double *norm_p)
{
   double         local_sum[3] = {0.0, 0.0, 0.0};
   double         global_sum[3];
   HYPRE_Complex *val =
      (HYPRE_Complex *)malloc((size_t)(3 * local_size) * sizeof(HYPRE_Complex));

   /* Gather local values */
   HYPRE_IJVectorGetValues(b, 3 * local_size, NULL, val);

   /* Compute local dot products */
   for (int i = 0; i < local_size; i++)
   {
      local_sum[0] += val[3 * i + 0] * val[3 * i + 0];
      local_sum[1] += val[3 * i + 1] * val[3 * i + 1];
      local_sum[2] += val[3 * i + 2] * val[3 * i + 2];
   }
   free(val);

   /* Compute global dot products */
   MPI_Allreduce(local_sum, global_sum, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   /* Compute L2-norms */
   *norm_u = sqrt(global_sum[0]);
   *norm_v = sqrt(global_sum[1]);
   *norm_p = sqrt(global_sum[2]);
}

/*--------------------------------------------------------------------------
 * Compute divergence of velocity field: div(u) = du/dx + dv/dy
 * Uses central differences with ghost data from all neighbors.
 * Returns divergence at local nodes (caller must free the returned array).
 *--------------------------------------------------------------------------*/
double *
ComputeDivergence(DistMesh2D *mesh, LidCavityParams *params, double *u_vec,
                  GhostData *ghost)
{
   int            nx      = mesh->nlocal[0];
   int            ny      = mesh->nlocal[1];
   int           *gdims   = mesh->gdims;
   int           *p       = mesh->coords;
   HYPRE_BigInt **pstarts = mesh->pstarts;

   double hx = params->L[0] / (gdims[0] - 1);
   double hy = params->L[1] / (gdims[1] - 1);

   /* Exchange ghost data with all neighbors */
   ExchangeVectorGhosts(mesh, u_vec, ghost);

   /* Allocate divergence array for local nodes */
   double *div_u = (double *)calloc((size_t)(nx * ny), sizeof(double));

   for (int ly = 0; ly < ny; ly++)
   {
      for (int lx = 0; lx < nx; lx++)
      {
         /* Global coordinates */
         int gx = (int)pstarts[0][p[0]] + lx;
         int gy = (int)pstarts[1][p[1]] + ly;

         /* Skip global boundary nodes - div(u) not enforced there */
         if (gx == 0 || gx == gdims[0] - 1 || gy == 0 || gy == gdims[1] - 1)
         {
            div_u[ly * nx + lx] = 0.0;
            continue;
         }

         double u_l, u_r, v_b, v_t;
         int    idx_c = (ly * nx + lx) * 3;

         /* Get u values for du/dx (left and right neighbors) */
         if (lx > 0)
         {
            u_l = u_vec[(ly * nx + (lx - 1)) * 3 + 0];
         }
         else if (ghost->recv_bufs[0]) /* Left neighbor */
         {
            u_l = ghost->recv_bufs[0][ly * 3 + 0];
         }
         else
         {
            u_l = u_vec[idx_c + 0]; /* Fallback (shouldn't happen for interior) */
         }

         if (lx < nx - 1)
         {
            u_r = u_vec[(ly * nx + (lx + 1)) * 3 + 0];
         }
         else if (ghost->recv_bufs[1]) /* Right neighbor */
         {
            u_r = ghost->recv_bufs[1][ly * 3 + 0];
         }
         else
         {
            u_r = u_vec[idx_c + 0]; /* Fallback */
         }

         /* Get v values for dv/dy (bottom and top neighbors) */
         if (ly > 0)
         {
            v_b = u_vec[((ly - 1) * nx + lx) * 3 + 1];
         }
         else if (ghost->recv_bufs[2]) /* Bottom neighbor */
         {
            v_b = ghost->recv_bufs[2][lx * 3 + 1];
         }
         else
         {
            v_b = u_vec[idx_c + 1]; /* Fallback */
         }

         if (ly < ny - 1)
         {
            v_t = u_vec[((ly + 1) * nx + lx) * 3 + 1];
         }
         else if (ghost->recv_bufs[3]) /* Top neighbor */
         {
            v_t = ghost->recv_bufs[3][lx * 3 + 1];
         }
         else
         {
            v_t = u_vec[idx_c + 1]; /* Fallback */
         }

         /* Central differences */
         double du_dx = (u_r - u_l) / (2.0 * hx);
         double dv_dy = (v_t - v_b) / (2.0 * hy);

         div_u[ly * nx + lx] = du_dx + dv_dy;
      }
   }

   return div_u;
}

int
BuildNewtonSystem(DistMesh2D *mesh, LidCavityParams *params,
                  HYPRE_Real     *u_prev_time,    /* u at n */
                  HYPRE_Real     *u_current_iter, /* u at k */
                  HYPRE_IJMatrix *J_ptr, HYPRE_IJVector *R_ptr, HYPRE_Real *res_norm,
                  GhostData *g_u_n, GhostData *g_u_k)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;

   const HYPRE_Int *p       = &mesh->coords[0];
   const HYPRE_Int *gdims   = &mesh->gdims[0];
   HYPRE_BigInt   **pstarts = &mesh->pstarts[0];

   HYPRE_BigInt       node_ilower      = mesh->ilower;
   HYPRE_BigInt       node_iupper      = mesh->iupper;
   HYPRE_BigInt       dof_ilower       = mesh->dof_ilower;
   HYPRE_BigInt       dof_iupper       = mesh->dof_iupper;
   const HYPRE_BigInt global_num_nodes = (HYPRE_BigInt)gdims[0] * (HYPRE_BigInt)gdims[1];
   const HYPRE_BigInt global_num_dofs  = 3 * global_num_nodes;

   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin("system", -1));

   /* Create IJ objects */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, dof_ilower, dof_iupper, dof_ilower, dof_iupper,
                        &A);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, dof_ilower, dof_iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   /* Set row sizes: conservative upper bound 27 per row (9 nodes x 3 dof) */
   HYPRE_Int  local_dofs = 3 * mesh->local_size;
   HYPRE_Int *nnzrow     = (HYPRE_Int *)calloc((size_t)local_dofs, sizeof(HYPRE_Int));
   for (int r = 0; r < local_dofs; r++) nnzrow[r] = 27;
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   free(nnzrow);

   HYPREDRV_IJ_MATRIX_INIT_HOST(A);
   HYPREDRV_IJ_VECTOR_INIT_HOST(b);

   /* Element sizes (physical domain) */
   HYPRE_Real       hx      = params->L[0] / (gdims[0] - 1);
   HYPRE_Real       hy      = params->L[1] / (gdims[1] - 1);
   const HYPRE_Real Jinv[2] = {2.0 / hx, 2.0 / hy};
   const HYPRE_Real detJ    = (hx * hy) / 4.0;

   HYPRE_BigInt gx_lo = (pstarts[0][p[0]] > 0) ? (pstarts[0][p[0]] - 1) : 0;
   HYPRE_BigInt gx_hi =
      (pstarts[0][p[0] + 1] < gdims[0]) ? pstarts[0][p[0] + 1] : (gdims[0] - 1);
   HYPRE_BigInt gy_lo = (pstarts[1][p[1]] > 0) ? (pstarts[1][p[1]] - 1) : 0;
   HYPRE_BigInt gy_hi =
      (pstarts[1][p[1] + 1] < gdims[1]) ? pstarts[1][p[1] + 1] : (gdims[1] - 1);

   /* Batch assembly buffers - accumulate across elements, flush when full */
   const int     max_rows = params->max_rows_per_call;
   const int     max_nnz  = max_rows * 27; /* Conservative: 27 nnz per row max */
   HYPRE_BigInt *batch_rows =
      (HYPRE_BigInt *)malloc((size_t)max_rows * sizeof(HYPRE_BigInt));
   HYPRE_Int    *batch_ncols = (HYPRE_Int *)malloc((size_t)max_rows * sizeof(HYPRE_Int));
   HYPRE_BigInt *batch_cols =
      (HYPRE_BigInt *)malloc((size_t)max_nnz * sizeof(HYPRE_BigInt));
   HYPRE_Real   *batch_vals = (HYPRE_Real *)malloc((size_t)max_nnz * sizeof(HYPRE_Real));
   HYPRE_BigInt *rhs_rows =
      (HYPRE_BigInt *)malloc((size_t)max_rows * sizeof(HYPRE_BigInt));
   HYPRE_Real *rhs_vals    = (HYPRE_Real *)malloc((size_t)max_rows * sizeof(HYPRE_Real));
   int         batch_nrows = 0;
   int         batch_nnz   = 0;
   int         rhs_count   = 0;

   for (HYPRE_BigInt gy = gy_lo; gy < gy_hi; gy++)
   {
      for (HYPRE_BigInt gx = gx_lo; gx < gx_hi; gx++)
      {
         // Element loop
         HYPRE_Real K_elem[12][12] = {{0}}; /* Full element Jacobian */
         HYPRE_Real R_elem[12]     = {0};   /* Full element residual */

         HYPRE_Real u_k_e[12]; /* u and p at current Newton iter (element) */
         HYPRE_Real u_n_e[12]; /* u and p at previous time step (element) */
         HYPRE_Real u_quad[2]; /* interpolated velocity at quadrature point */
         (void)u_quad;         /* Reserved for future use */
         HYPRE_BigInt node_gid[4];
         HYPRE_BigInt dof_gid[12];

         HYPRE_BigInt ng[4][2] = {{gx, gy}, {gx + 1, gy}, {gx + 1, gy + 1}, {gx, gy + 1}};
         for (int a = 0; a < 4; a++)
         {
            HYPRE_Int owner_bc[2];
            for (int d = 0; d < 2; d++)
            {
               HYPRE_Int     pd = mesh->pdims[d];
               HYPRE_BigInt *ps = mesh->pstarts[d];
               HYPRE_BigInt  cg = ng[a][d];
               HYPRE_Int     bd = 0;
               while ((bd + 1) < pd && cg >= ps[bd + 1]) bd++;
               owner_bc[d] = bd;
            }
            node_gid[a] = grid2idx(ng[a], owner_bc, gdims, pstarts);
            /* Sanity check: global IDs must be within [0, global_num_nodes) */
            if (node_gid[a] < 0 || node_gid[a] >= global_num_nodes)
            {
               printf(
                  "[rank %d coords %d,%d] ERROR: node_gid out of range: node_gid=%lld "
                  "global_num_nodes=%lld (gx=%lld gy=%lld owner_bc=%d,%d)\n",
                  (int)mesh->mypid, (int)mesh->coords[0], (int)mesh->coords[1],
                  (long long)node_gid[a], (long long)global_num_nodes,
                  (long long)ng[a][0], (long long)ng[a][1], (int)owner_bc[0],
                  (int)owner_bc[1]);
               MPI_Abort(mesh->cart_comm, -1);
            }
            dof_gid[3 * a + 0] = 3 * node_gid[a] + 0;
            dof_gid[3 * a + 1] = 3 * node_gid[a] + 1;
            dof_gid[3 * a + 2] = 3 * node_gid[a] + 2;
            /* Sanity check: global DOF IDs must be within [0, global_num_dofs) */
            for (int d = 0; d < 3; d++)
            {
               HYPRE_BigInt gid = dof_gid[3 * a + d];
               if (gid < 0 || gid >= global_num_dofs)
               {
                  printf(
                     "[rank %d coords %d,%d] ERROR: dof_gid out of range: dof_gid=%lld "
                     "global_num_dofs=%lld (node_gid=%lld)\n",
                     (int)mesh->mypid, (int)mesh->coords[0], (int)mesh->coords[1],
                     (long long)gid, (long long)global_num_dofs, (long long)node_gid[a]);
                  MPI_Abort(mesh->cart_comm, -1);
               }
            }

            /* Gather current solution DOFs for element */
            // Note: node_gid is global, but u_current_iter is local
            // We can only access nodes owned by this process
            // For now, assume all nodes in elements we process are owned (or use 0 for
            // ghost nodes)
            HYPRE_BigInt local_node_idx = node_gid[a] - node_ilower;
            if (local_node_idx >= 0 && local_node_idx < (HYPRE_BigInt)mesh->local_size)
            {
               u_k_e[3 * a + 0] = u_current_iter[3 * local_node_idx + 0];
               u_k_e[3 * a + 1] = u_current_iter[3 * local_node_idx + 1];
               u_k_e[3 * a + 2] = u_current_iter[3 * local_node_idx + 2];

               u_n_e[3 * a + 0] = u_prev_time[3 * local_node_idx + 0];
               u_n_e[3 * a + 1] = u_prev_time[3 * local_node_idx + 1];
               u_n_e[3 * a + 2] = u_prev_time[3 * local_node_idx + 2];
            }
            else
            {
               // Ghost node - lookup in ghost data
               int     lx    = (int)(ng[a][0] - pstarts[0][p[0]]);
               int     ly    = (int)(ng[a][1] - pstarts[1][p[1]]);
               int     nx    = (int)mesh->nlocal[0];
               int     ny    = (int)mesh->nlocal[1];
               double *src_k = NULL;
               double *src_n = NULL;
               int     off   = 0;

               if (lx == -1 && ly >= 0 && ly < ny)
               { // Left
                  src_k = g_u_k->recv_bufs[0];
                  src_n = g_u_n->recv_bufs[0];
                  off   = ly * 3;
               }
               else if (lx == nx && ly >= 0 && ly < ny)
               { // Right
                  src_k = g_u_k->recv_bufs[1];
                  src_n = g_u_n->recv_bufs[1];
                  off   = ly * 3;
               }
               else if (ly == -1 && lx >= 0 && lx < nx)
               { // Bottom
                  src_k = g_u_k->recv_bufs[2];
                  src_n = g_u_n->recv_bufs[2];
                  off   = lx * 3;
               }
               else if (ly == ny && lx >= 0 && lx < nx)
               { // Top
                  src_k = g_u_k->recv_bufs[3];
                  src_n = g_u_n->recv_bufs[3];
                  off   = lx * 3;
               }
               else if (lx == -1 && ly == -1)
               { // DL
                  src_k = g_u_k->recv_bufs[4];
                  src_n = g_u_n->recv_bufs[4];
                  off   = 0;
               }
               else if (lx == nx && ly == ny)
               { // TR
                  src_k = g_u_k->recv_bufs[5];
                  src_n = g_u_n->recv_bufs[5];
                  off   = 0;
               }
               else if (lx == -1 && ly == ny)
               { // TL
                  src_k = g_u_k->recv_bufs[6];
                  src_n = g_u_n->recv_bufs[6];
                  off   = 0;
               }
               else if (lx == nx && ly == -1)
               { // BR
                  src_k = g_u_k->recv_bufs[7];
                  src_n = g_u_n->recv_bufs[7];
                  off   = 0;
               }

               if (src_k)
               {
                  u_k_e[3 * a + 0] = src_k[off + 0];
                  u_k_e[3 * a + 1] = src_k[off + 1];
                  u_k_e[3 * a + 2] = src_k[off + 2];
                  u_n_e[3 * a + 0] = src_n[off + 0];
                  u_n_e[3 * a + 1] = src_n[off + 1];
                  u_n_e[3 * a + 2] = src_n[off + 2];
               }
               else
               {
                  u_k_e[3 * a + 0] = 0.0;
                  u_k_e[3 * a + 1] = 0.0;
                  u_k_e[3 * a + 2] = 0.0;
                  u_n_e[3 * a + 0] = 0.0;
                  u_n_e[3 * a + 1] = 0.0;
                  u_n_e[3 * a + 2] = 0.0;
               }
            }
         }

         /* Calculate element mean velocity magnitude and stabilization parameter tau_e */
         HYPRE_Real u_mean_e = 0.0, v_mean_e = 0.0;
         for (int a = 0; a < 4; a++)
         {
            u_mean_e += u_k_e[3 * a + 0];
            v_mean_e += u_k_e[3 * a + 1];
         }
         HYPRE_Real u_mag_e = sqrt(u_mean_e * u_mean_e + v_mean_e * v_mean_e) / 4.0;
         HYPRE_Real he      = sqrt(hx * hx + hy * hy); // Element characteristic length
         HYPRE_Real nu      = 1.0 / params->Re;
         HYPRE_Real tau_e_term1 = (2.0 / params->dt) * (2.0 / params->dt);
         HYPRE_Real tau_e_term2 = (2.0 * u_mag_e / he) * (2.0 * u_mag_e / he);
         HYPRE_Real tau_e_term3 = (4.0 * nu / (he * he)) * (4.0 * nu / (he * he));
         HYPRE_Real tau_e       = 1.0 / sqrt(tau_e_term1 + tau_e_term2 + tau_e_term3);

         for (int iy = 0; iy < 2; iy++)
         {
            for (int ix = 0; ix < 2; ix++)
            {
               // Quadrature loop
               HYPRE_Real xi  = gauss_x[ix];
               HYPRE_Real eta = gauss_x[iy];
               HYPRE_Real w   = gauss_w[ix] * gauss_w[iy] * detJ;

               HYPRE_Real N[4], dN_dxi[4], dN_deta[4];
               q1_shape_ref_2d(xi, eta, N, dN_dxi, dN_deta);

               HYPRE_Real dN_dx[4], dN_dy[4];
               for (int a = 0; a < 4; a++)
               {
                  dN_dx[a] = dN_dxi[a] * Jinv[0];
                  dN_dy[a] = dN_deta[a] * Jinv[1];
               }
               /* Interpolate solution values at quadrature point */
               HYPRE_Real u_k = 0.0, v_k = 0.0, p_k = 0.0;
               HYPRE_Real u_n = 0.0, v_n = 0.0;
               HYPRE_Real du_dx_k = 0.0, du_dy_k = 0.0, dv_dx_k = 0.0, dv_dy_k = 0.0;
               HYPRE_Real dp_dx_k = 0.0, dp_dy_k = 0.0;

               for (int a = 0; a < 4; a++)
               {
                  u_k += N[a] * u_k_e[3 * a + 0];
                  v_k += N[a] * u_k_e[3 * a + 1];
                  p_k += N[a] * u_k_e[3 * a + 2];

                  u_n += N[a] * u_n_e[3 * a + 0];
                  v_n += N[a] * u_n_e[3 * a + 1];

                  du_dx_k += dN_dx[a] * u_k_e[3 * a + 0];
                  du_dy_k += dN_dy[a] * u_k_e[3 * a + 0];
                  dv_dx_k += dN_dx[a] * u_k_e[3 * a + 1];
                  dv_dy_k += dN_dy[a] * u_k_e[3 * a + 1];
                  dp_dx_k += dN_dx[a] * u_k_e[3 * a + 2];
                  dp_dy_k += dN_dy[a] * u_k_e[3 * a + 2];
               }

               // Store current velocity at quadrature point for convenience
               u_quad[0] = u_k;
               u_quad[1] = v_k;

               /* Strong form residual components (for stabilization) */
               HYPRE_Real Res_x =
                  (u_k - u_n) / params->dt + (u_k * du_dx_k + v_k * du_dy_k) + dp_dx_k;
               HYPRE_Real Res_y =
                  (v_k - v_n) / params->dt + (u_k * dv_dx_k + v_k * dv_dy_k) + dp_dy_k;

               for (int a = 0; a < 4; a++)
               {
                  for (int b_idx = 0; b_idx < 4; b_idx++)
                  {
                     // Galerkin Jacobian contributions
                     // Time derivative: (1/dt) * M
                     K_elem[3 * a + 0][3 * b_idx + 0] += w * N[a] * N[b_idx] / params->dt;
                     K_elem[3 * a + 1][3 * b_idx + 1] += w * N[a] * N[b_idx] / params->dt;

                     // Viscous term: nu * grad(u) : grad(v)
                     K_elem[3 * a + 0][3 * b_idx + 0] +=
                        w * nu * (dN_dx[a] * dN_dx[b_idx] + dN_dy[a] * dN_dy[b_idx]);
                     K_elem[3 * a + 1][3 * b_idx + 1] +=
                        w * nu * (dN_dx[a] * dN_dx[b_idx] + dN_dy[a] * dN_dy[b_idx]);

                     // Convection term linearization (Newton)
                     // Part 1: (u_k * grad(delta u)) * test_func  [Picard part]
                     K_elem[3 * a + 0][3 * b_idx + 0] +=
                        w * (u_k * dN_dx[b_idx] + v_k * dN_dy[b_idx]) * N[a];
                     K_elem[3 * a + 1][3 * b_idx + 1] +=
                        w * (u_k * dN_dx[b_idx] + v_k * dN_dy[b_idx]) * N[a];

                     // Part 2: (delta u * grad(u_k)) * test_func  [Newton part]
                     // X-Momentum: (delta_u * du/dx + delta_v * du/dy) * test_func
                     K_elem[3 * a + 0][3 * b_idx + 0] +=
                        w * (N[b_idx] * du_dx_k) * N[a]; // d(conv)/du
                     K_elem[3 * a + 0][3 * b_idx + 1] +=
                        w * (N[b_idx] * du_dy_k) * N[a]; // d(conv)/dv

                     // Y-Momentum: (delta_u * dv/dx + delta_v * dv/dy) * test_func
                     K_elem[3 * a + 1][3 * b_idx + 0] +=
                        w * (N[b_idx] * dv_dx_k) * N[a]; // d(conv)/du
                     K_elem[3 * a + 1][3 * b_idx + 1] +=
                        w * (N[b_idx] * dv_dy_k) * N[a]; // d(conv)/dv

                     // Pressure-Velocity coupling (from -p div v and q div u)
                     K_elem[3 * a + 0][3 * b_idx + 2] -= w * dN_dx[a] * N[b_idx]; // du/dp
                     K_elem[3 * a + 1][3 * b_idx + 2] -= w * dN_dy[a] * N[b_idx]; // dv/dp
                     K_elem[3 * a + 2][3 * b_idx + 0] += w * N[a] * dN_dx[b_idx]; // dp/du
                     K_elem[3 * a + 2][3 * b_idx + 1] += w * N[a] * dN_dy[b_idx]; // dp/dv

                     // Stabilization Jacobian contributions (frozen tau_e and conv_vel in
                     // test func pert) d(R_mom_x)/d(u_b) * SUPG_test_func_x
                     K_elem[3 * a + 0][3 * b_idx + 0] +=
                        w * tau_e *
                        (N[b_idx] / params->dt + u_k * dN_dx[b_idx] +
                         v_k * dN_dy[b_idx]) *
                        (u_k * dN_dx[a] + v_k * dN_dy[a]);
                     // d(R_mom_x)/d(p_b) * SUPG_test_func_x
                     K_elem[3 * a + 0][3 * b_idx + 2] +=
                        w * tau_e * dN_dx[b_idx] * (u_k * dN_dx[a] + v_k * dN_dy[a]);

                     // d(R_mom_y)/d(v_b) * SUPG_test_func_y
                     K_elem[3 * a + 1][3 * b_idx + 1] +=
                        w * tau_e *
                        (N[b_idx] / params->dt + u_k * dN_dx[b_idx] +
                         v_k * dN_dy[b_idx]) *
                        (u_k * dN_dx[a] + v_k * dN_dy[a]);
                     // d(R_mom_y)/d(p_b) * SUPG_test_func_y
                     K_elem[3 * a + 1][3 * b_idx + 2] +=
                        w * tau_e * dN_dy[b_idx] * (u_k * dN_dx[a] + v_k * dN_dy[a]);

                     // d(R_mom_x)/d(u_b) * PSPG_test_func_x
                     K_elem[3 * a + 2][3 * b_idx + 0] +=
                        w * tau_e *
                        (N[b_idx] / params->dt + u_k * dN_dx[b_idx] +
                         v_k * dN_dy[b_idx]) *
                        dN_dx[a];
                     // d(R_mom_y)/d(v_b) * PSPG_test_func_y
                     K_elem[3 * a + 2][3 * b_idx + 1] +=
                        w * tau_e *
                        (N[b_idx] / params->dt + u_k * dN_dx[b_idx] +
                         v_k * dN_dy[b_idx]) *
                        dN_dy[a];
                     // d(R_mom_x)/d(p_b) * PSPG_test_func_x + d(R_mom_y)/d(p_b) *
                     // PSPG_test_func_y
                     K_elem[3 * a + 2][3 * b_idx + 2] +=
                        w * tau_e * (dN_dx[b_idx] * dN_dx[a] + dN_dy[b_idx] * dN_dy[a]);
                  }

                  // Galerkin Residual contributions
                  // Momentum x
                  R_elem[3 * a + 0] +=
                     w *
                     (N[a] * (u_k - u_n) / params->dt +
                      N[a] * (u_k * du_dx_k + v_k * du_dy_k) +
                      nu * (dN_dx[a] * du_dx_k + dN_dy[a] * du_dy_k) - dN_dx[a] * p_k);
                  // Momentum y
                  R_elem[3 * a + 1] +=
                     w *
                     (N[a] * (v_k - v_n) / params->dt +
                      N[a] * (u_k * dv_dx_k + v_k * dv_dy_k) +
                      nu * (dN_dx[a] * dv_dx_k + dN_dy[a] * dv_dy_k) - dN_dy[a] * p_k);
                  // Continuity
                  R_elem[3 * a + 2] += w * (N[a] * (du_dx_k + dv_dy_k));

                  // Stabilization Residual contributions (SUPG + PSPG)
                  // SUPG for x-momentum
                  R_elem[3 * a + 0] +=
                     w * tau_e * (Res_x * (u_k * dN_dx[a] + v_k * dN_dy[a]));
                  // SUPG for y-momentum
                  R_elem[3 * a + 1] +=
                     w * tau_e * (Res_y * (u_k * dN_dx[a] + v_k * dN_dy[a]));
                  // PSPG for pressure
                  R_elem[3 * a + 2] += w * tau_e * (Res_x * dN_dx[a] + Res_y * dN_dy[a]);
               }
            }
         }

         /* Identify Dirichlet DOFs for this element */
         HYPRE_Int is_dirichlet[12] = {0};
         for (int a = 0; a < 4; a++)
         {
            HYPRE_BigInt gx_local    = ng[a][0];
            HYPRE_BigInt gy_local    = ng[a][1];
            HYPRE_Int    on_boundary = 0;
            HYPRE_Real   u_bc        = 0.0;
            HYPRE_Real   v_bc        = 0.0;
            (void)u_bc; /* Reserved for future use */
            (void)v_bc; /* Reserved for future use */

            if (gy_local == gdims[1] - 1 && gx_local != 0 && gx_local != gdims[0] - 1)
            {
               on_boundary = 1; // Lid: u = 1, v = 0
               u_bc        = 1.0;
            }
            else if (gx_local == 0 || gx_local == gdims[0] - 1 || gy_local == 0)
            {
               on_boundary = 1; // Wall: u = 0, v = 0
            }

            if (gx_local == 0 && gy_local == 0)
            {
               // Pressure reference node: p is Dirichlet-like (constrained)
               is_dirichlet[3 * a + 2] = 1;
            }

            if (on_boundary)
            {
               is_dirichlet[3 * a + 0] = 1; // u is Dirichlet
               is_dirichlet[3 * a + 1] = 1; // v is Dirichlet
            }
         }

         /* Accumulate element contributions into batch buffers */
         for (int i = 0; i < 12; i++)
         {
            if (is_dirichlet[i]) continue; /* Skip Dirichlet rows */
            if (dof_gid[i] < dof_ilower || dof_gid[i] > dof_iupper)
               continue; /* Only own rows */

            /* Check if batch is full - flush before adding */
            int row_nnz = 0;
            for (int j = 0; j < 12; j++)
               if (K_elem[i][j] != 0.0) row_nnz++;

            if (batch_nrows >= max_rows || batch_nnz + row_nnz > max_nnz)
            {
               /* Flush matrix batch */
               if (batch_nrows > 0)
               {
                  HYPRE_IJMatrixAddToValues(A, batch_nrows, batch_ncols, batch_rows,
                                            batch_cols, batch_vals);
               }
               /* Flush RHS batch */
               if (rhs_count > 0)
               {
                  HYPRE_IJVectorAddToValues(b, rhs_count, rhs_rows, rhs_vals);
               }
               batch_nrows = 0;
               batch_nnz   = 0;
               rhs_count   = 0;
            }

            /* Add row to batch */
            batch_rows[batch_nrows] = dof_gid[i];
            int row_ncols           = 0;
            for (int j = 0; j < 12; j++)
            {
               if (K_elem[i][j] != 0.0)
               {
                  batch_cols[batch_nnz] = dof_gid[j];
                  batch_vals[batch_nnz] = K_elem[i][j];
                  batch_nnz++;
                  row_ncols++;
               }
            }
            batch_ncols[batch_nrows] = row_ncols;
            batch_nrows++;

            if (R_elem[i] != 0.0)
            {
               rhs_rows[rhs_count] = dof_gid[i];
               rhs_vals[rhs_count] = -R_elem[i]; /* Newton RHS is -Residual */
               rhs_count++;
            }
         }
      }
   }

   /* Flush remaining batch */
   if (batch_nrows > 0)
   {
      HYPRE_IJMatrixAddToValues(A, batch_nrows, batch_ncols, batch_rows, batch_cols,
                                batch_vals);
   }
   if (rhs_count > 0)
   {
      HYPRE_IJVectorAddToValues(b, rhs_count, rhs_rows, rhs_vals);
   }

   free(batch_rows);
   free(batch_ncols);
   free(batch_cols);
   free(batch_vals);
   free(rhs_rows);
   free(rhs_vals);

   // Apply boundary conditions: Set Dirichlet rows to identity with prescribed RHS
   for (HYPRE_BigInt i = node_ilower; i <= node_iupper; i++)
   {
      HYPRE_BigInt gcoords[2];
      get_global_coords(i, mesh, gcoords);

      HYPRE_Int boundary_type = 0; // 0: interior, 1: wall, 2: lid

      if (gcoords[1] == gdims[1] - 1 && gcoords[0] != 0 && gcoords[0] != gdims[0] - 1)
      {
         boundary_type = 2; // Lid (top, excluding corners): u = profile, v = 0
      }
      else if (gcoords[0] == 0 || gcoords[0] == gdims[0] - 1 || gcoords[1] == 0)
      {
         boundary_type = 1; // Wall (left, right, bottom, top corners): u = 0, v = 0
      }

      // Fix pressure at a reference node (e.g., global (0,0))
      if (gcoords[0] == 0 && gcoords[1] == 0)
      {
         HYPRE_BigInt row        = 3 * i + 2; // Pressure DOF
         HYPRE_Real   val_matrix = 1.0;
         HYPRE_Real   val_vector = 0.0; // p = 0
         HYPRE_Int    ncols      = 1;
         HYPRE_IJMatrixSetValues(A, 1, &ncols, &row, &row, &val_matrix);
         HYPRE_IJVectorSetValues(b, 1, &row, &val_vector);
      }

      if (boundary_type > 0)
      {
         for (int j = 0; j < 2; j++) // Apply to u and v components
         {
            HYPRE_BigInt row        = 3 * i + j; // u or v DOF (global DOF ID)
            HYPRE_Real   val_matrix = 1.0;
            HYPRE_Real   u_bc       = 0.0; // Default to 0 for walls

            if (boundary_type == 2 && j == 0) // Lid, u-component
            {
               if (params->regularize_bc)
               {
                  // Regularized lid BC: u(x) = [1 - (2x-1)^(2n)]^2 with n=8
                  // Flatter profile in center, steeper near corners
                  // u(0) = 0, u(0.5) = 1, u(1) = 0
                  HYPRE_Real x_norm = (HYPRE_Real)gcoords[0] / (gdims[0] - 1);
                  HYPRE_Real xi     = 2.0 * x_norm - 1.0; // Map [0,1] to [-1,1]
                  HYPRE_Real xi16   = xi * xi;            // xi^2
                  xi16 *= xi16;                           // xi^4
                  xi16 *= xi16;                           // xi^8
                  xi16 *= xi16;                           // xi^16
                  u_bc = (1.0 - xi16) * (1.0 - xi16);
               }
               else
               {
                  u_bc = 1.0; // Classic BC: u = 1 on entire lid
               }
            }

            // Convert global node ID to local index for accessing u_current_iter
            HYPRE_BigInt local_node_idx = i - node_ilower;
            HYPRE_BigInt local_dof_idx  = 3 * local_node_idx + j;

            // Compute residual: R = u_BC - u_current
            HYPRE_Real u_current  = u_current_iter[local_dof_idx];
            HYPRE_Real val_vector = u_bc - u_current;

            // Set diagonal of Jacobian to 1 and RHS to u_BC - u_current
            HYPRE_Int ncols = 1;
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row, &row, &val_matrix);
            HYPRE_IJVectorSetValues(b, 1, &row, &val_vector);
         }
      }
   }

   /* Finalize assembly after all values (including Dirichlet rows) are set */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);

   /* Compute current residual */
#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 22600
   HYPRE_IJVectorInnerProd(b, b, res_norm);
#else
   void           *b_obj = NULL;
   HYPRE_ParVector b_par = NULL;
   HYPRE_IJVectorGetObject(b, &b_obj);
   b_par = (HYPRE_ParVector)b_obj;
   HYPRE_ParVectorInnerProd(b_par, b_par, res_norm);
#endif
   *res_norm = sqrt(*res_norm);

   *J_ptr = A;
   *R_ptr = b;

   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd("system", -1));

   return 0;
}

/*--------------------------------------------------------------------------
 * Write vector VTK (RectilinearGrid, point vectors named "velocity")
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
WriteVTKsolutionVector(DistMesh2D *mesh, LidCavityParams *params, HYPRE_Real *sol_data,
                       GhostData *ghost, int time_step_idx)
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
   int ix_start = (int)pstarts[0][p[0]];
   int iy_start = (int)pstarts[1][p[1]];

   /* Overlap on negative faces */
   int ofi = (p[0] > 0) ? 1 : 0; /* left  ghost layer */
   int ofj = (p[1] > 0) ? 1 : 0; /* down  ghost layer */
   int nxg = nx + ofi;
   int nyg = ny + ofj;

   /* Allocate VTK ghost buffers (vectors: 3 doubles per point) */
   double *vtk_ghost[8];
   for (int i = 0; i < 8; i++) vtk_ghost[i] = NULL;
   if (mesh->nbrs[0] != MPI_PROC_NULL)
      vtk_ghost[0] = (double *)calloc((size_t)(ny * 3), sizeof(double)); /* left  face */
   if (mesh->nbrs[1] != MPI_PROC_NULL)
      vtk_ghost[1] =
         (double *)calloc((size_t)(ny * 3), sizeof(double)); /* right face send */
   if (mesh->nbrs[2] != MPI_PROC_NULL)
      vtk_ghost[2] = (double *)calloc((size_t)(nx * 3), sizeof(double)); /* down  face */
   if (mesh->nbrs[3] != MPI_PROC_NULL)
      vtk_ghost[3] =
         (double *)calloc((size_t)(nx * 3), sizeof(double)); /* up    face send */
   if (mesh->nbrs[4] != MPI_PROC_NULL)
      vtk_ghost[4] = (double *)calloc(3, sizeof(double)); /* left-down corner */
   if (mesh->nbrs[5] != MPI_PROC_NULL)
      vtk_ghost[5] = (double *)calloc(3, sizeof(double)); /* right-up  corner send */
   if (mesh->nbrs[6] != MPI_PROC_NULL)
      vtk_ghost[6] = (double *)calloc(3, sizeof(double)); /* left-up corner */
   if (mesh->nbrs[7] != MPI_PROC_NULL)
      vtk_ghost[7] = (double *)calloc(3, sizeof(double)); /* right-down  corner send */

   /* Fill send buffers and post Irecvs */
   MPI_Request reqs[16];
   int         reqc = 0;

   /* X-direction faces */
   if (mesh->nbrs[1] != MPI_PROC_NULL)
   {
      for (int j = 0; j < ny; j++)
      {
         int nidx              = j * nx + (nx - 1);
         int off               = j * 3;
         vtk_ghost[1][off + 0] = sol_data[3 * nidx + 0];
         vtk_ghost[1][off + 1] = sol_data[3 * nidx + 1];
         vtk_ghost[1][off + 2] = sol_data[3 * nidx + 2];
      }
      MPI_Isend(vtk_ghost[1], ny * 3, MPI_DOUBLE, mesh->nbrs[1], 0, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[0] != MPI_PROC_NULL)
   {
      MPI_Irecv(vtk_ghost[0], ny * 3, MPI_DOUBLE, mesh->nbrs[0], 0, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* Y-direction faces */
   if (mesh->nbrs[3] != MPI_PROC_NULL)
   {
      for (int i = 0; i < nx; i++)
      {
         int nidx              = (ny - 1) * nx + i;
         int off               = i * 3;
         vtk_ghost[3][off + 0] = sol_data[3 * nidx + 0];
         vtk_ghost[3][off + 1] = sol_data[3 * nidx + 1];
         vtk_ghost[3][off + 2] = sol_data[3 * nidx + 2];
      }
      MPI_Isend(vtk_ghost[3], nx * 3, MPI_DOUBLE, mesh->nbrs[3], 1, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL)
   {
      MPI_Irecv(vtk_ghost[2], nx * 3, MPI_DOUBLE, mesh->nbrs[2], 1, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* Corners */
   if (mesh->nbrs[5] != MPI_PROC_NULL)
   {
      int nidx        = (ny - 1) * nx + (nx - 1);
      vtk_ghost[5][0] = sol_data[3 * nidx + 0];
      vtk_ghost[5][1] = sol_data[3 * nidx + 1];
      vtk_ghost[5][2] = sol_data[3 * nidx + 2];
      MPI_Isend(vtk_ghost[5], 3, MPI_DOUBLE, mesh->nbrs[5], 2, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[4] != MPI_PROC_NULL)
   {
      MPI_Irecv(vtk_ghost[4], 3, MPI_DOUBLE, mesh->nbrs[4], 2, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[6] != MPI_PROC_NULL)
   {
      int nidx        = (ny - 1) * nx;
      vtk_ghost[6][0] = sol_data[3 * nidx + 0];
      vtk_ghost[6][1] = sol_data[3 * nidx + 1];
      vtk_ghost[6][2] = sol_data[3 * nidx + 2];
      MPI_Isend(vtk_ghost[6], 3, MPI_DOUBLE, mesh->nbrs[6], 3, mesh->cart_comm,
                &reqs[reqc++]);
   }
   if (mesh->nbrs[7] != MPI_PROC_NULL)
   {
      MPI_Irecv(vtk_ghost[7], 3, MPI_DOUBLE, mesh->nbrs[7], 3, mesh->cart_comm,
                &reqs[reqc++]);
   }

   /* Build extended data including negative-face ghosts */
   double *ext = (double *)calloc((size_t)(nxg * nyg * 3), sizeof(double));
   for (int j = 0; j < ny; j++)
   {
      for (int i = 0; i < nx; i++)
      {
         int ig = i + ofi, jg = j + ofj;
         int idx_e      = (jg * nxg + ig) * 3;
         int nidx       = j * nx + i;
         ext[idx_e + 0] = sol_data[3 * nidx + 0];
         ext[idx_e + 1] = sol_data[3 * nidx + 1];
         ext[idx_e + 2] = sol_data[3 * nidx + 2];
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
      for (int j = 0; j < ny; j++)
      {
         int jg       = j + ofj;
         int src      = j * 3;
         int dst      = (jg * nxg + 0) * 3;
         ext[dst + 0] = vtk_ghost[0][src + 0];
         ext[dst + 1] = vtk_ghost[0][src + 1];
         ext[dst + 2] = vtk_ghost[0][src + 2];
      }
   }
   if (mesh->nbrs[2] != MPI_PROC_NULL)
   {
      for (int i = 0; i < nx; i++)
      {
         int ig       = i + ofi;
         int src      = i * 3;
         int dst      = (0 * nxg + ig) * 3;
         ext[dst + 0] = vtk_ghost[2][src + 0];
         ext[dst + 1] = vtk_ghost[2][src + 1];
         ext[dst + 2] = vtk_ghost[2][src + 2];
      }
   }

   /* Insert corners */
   if (mesh->nbrs[4] != MPI_PROC_NULL) /* left-down */
   {
      int dst      = (0 * nxg + 0) * 3;
      ext[dst + 0] = vtk_ghost[4][0];
      ext[dst + 1] = vtk_ghost[4][1];
      ext[dst + 2] = vtk_ghost[4][2];
   }

   /* Create data directory if it doesn't exist (all ranks) */
   char data_dir[256];
   GetVTKDataDir(params, data_dir, sizeof(data_dir));
   struct stat st = {0};
   if (stat(data_dir, &st) == -1)
   {
      mkdir(data_dir, 0755);
   }
   MPI_Barrier(mesh->cart_comm); /* Ensure directory exists before writing */

   /* Filenames: step_XXX_rank.vtr in data directory (0-indexed steps) */
   char filename[512];
   snprintf(filename, sizeof(filename), "%s/step_%05d_%d.vtr", data_dir,
            time_step_idx - 1, myid);
   FILE *fp = fopen(filename, "w");
   if (!fp)
   {
      printf("Error: Cannot open file %s\n", filename);
      MPI_Abort(mesh->cart_comm, -1);
   }

   /* Coordinates (scale by physical dimensions) and extents with overlap */
   double x_step  = params->L[0] * gsizes[0];
   double y_step  = params->L[1] * gsizes[1];
   double x_start = (ix_start - ofi) * x_step;
   double y_start = (iy_start - ofj) * y_step;

   fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
   fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\">\n");
   fprintf(fp, "  <RectilinearGrid WholeExtent=\"%d %d %d %d 0 0\">\n", ix_start - ofi,
           ix_start + nx - 1, iy_start - ofj, iy_start + ny - 1);
   fprintf(fp, "    <Piece Extent=\"%d %d %d %d 0 0\">\n", ix_start - ofi,
           ix_start + nx - 1, iy_start - ofj, iy_start + ny - 1);

   fprintf(fp, "      <Coordinates>\n");
   WriteCoordArray(fp, "x", x_start, x_step, nxg);
   WriteCoordArray(fp, "y", y_start, y_step, nyg);
   WriteCoordArray(fp, "z", 0.0, 0.0, 1);
   fprintf(fp, "      </Coordinates>\n");

   /* Compute divergence on local nodes using proper MPI ghost exchange */
   double *div_local = ComputeDivergence(mesh, params, sol_data, ghost);

   /* Expand divergence to extended grid (including VTK overlap ghosts) */
   double *div_u = (double *)calloc((size_t)(nxg * nyg), sizeof(double));
   for (int j = 0; j < ny; j++)
   {
      for (int i = 0; i < nx; i++)
      {
         int ig = i + ofi, jg = j + ofj;
         div_u[jg * nxg + ig] = div_local[j * nx + i];
      }
   }
   free(div_local);

   /* Fill ghost regions of divergence with zeros (boundary/overlap regions) */
   /* Left ghost column (if present) */
   if (ofi > 0)
   {
      for (int j = 0; j < nyg; j++) div_u[j * nxg + 0] = 0.0;
   }
   /* Bottom ghost row (if present) */
   if (ofj > 0)
   {
      for (int i = 0; i < nxg; i++) div_u[0 * nxg + i] = 0.0;
   }

   /* Point data: vectors and scalars */
   fprintf(fp, "      <PointData Vectors=\"velocity\" Scalars=\"pressure\">\n");
   if (params->visualize & 0x2)
   {
      /* ASCII */
      /* Velocity */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" "
                  "NumberOfComponents=\"3\" format=\"ascii\">\n");
      fprintf(fp, "          ");
      for (int j = 0; j < nyg; j++)
      {
         for (int i = 0; i < nxg; i++)
         {
            int idx = (j * nxg + i) * 3;
            fprintf(fp, "%.15g %.15g 0.0 ", ext[idx + 0], ext[idx + 1]);
            if ((j * nxg + i + 1) % 2 == 0) fprintf(fp, "\n          ");
         }
      }
      fprintf(fp, "\n        </DataArray>\n");

      /* Pressure */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"pressure\" "
                  "NumberOfComponents=\"1\" format=\"ascii\">\n");
      fprintf(fp, "          ");
      for (int j = 0; j < nyg; j++)
      {
         for (int i = 0; i < nxg; i++)
         {
            int idx = (j * nxg + i) * 3;
            fprintf(fp, "%.15g ", ext[idx + 2]);
            if ((j * nxg + i + 1) % 6 == 0) fprintf(fp, "\n          ");
         }
      }
      fprintf(fp, "\n        </DataArray>\n");

      /* Divergence of velocity */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"div_velocity\" "
                  "NumberOfComponents=\"1\" format=\"ascii\">\n");
      fprintf(fp, "          ");
      for (int j = 0; j < nyg; j++)
      {
         for (int i = 0; i < nxg; i++)
         {
            fprintf(fp, "%.15g ", div_u[j * nxg + i]);
            if ((j * nxg + i + 1) % 6 == 0) fprintf(fp, "\n          ");
         }
      }
      fprintf(fp, "\n        </DataArray>\n");

      fprintf(fp, "      </PointData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </RectilinearGrid>\n");
   }
   else
   {
      /* Binary (bit 1 = 0 means binary) */
      int    npts      = nxg * nyg;
      size_t vel_size  = (size_t)(npts * 3) * sizeof(double);
      size_t pres_size = (size_t)(npts * 1) * sizeof(double);
      size_t div_size  = (size_t)(npts * 1) * sizeof(double);

      double *vel_buf  = (double *)malloc(vel_size);
      double *pres_buf = (double *)malloc(pres_size);

      for (int k = 0; k < npts; k++)
      {
         vel_buf[3 * k + 0] = ext[3 * k + 0];
         vel_buf[3 * k + 1] = ext[3 * k + 1];
         vel_buf[3 * k + 2] = 0.0;
         pres_buf[k]        = ext[3 * k + 2];
      }

      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" "
                  "NumberOfComponents=\"3\" format=\"appended\" offset=\"0\">\n");
      fprintf(fp, "        </DataArray>\n");

      fprintf(fp,
              "        <DataArray type=\"Float64\" Name=\"pressure\" "
              "NumberOfComponents=\"1\" format=\"appended\" offset=\"%lu\">\n",
              sizeof(int) + (unsigned long)vel_size);
      fprintf(fp, "        </DataArray>\n");

      fprintf(fp,
              "        <DataArray type=\"Float64\" Name=\"div_velocity\" "
              "NumberOfComponents=\"1\" format=\"appended\" offset=\"%lu\">\n",
              2 * sizeof(int) + (unsigned long)vel_size + (unsigned long)pres_size);
      fprintf(fp, "        </DataArray>\n");

      fprintf(fp, "      </PointData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </RectilinearGrid>\n");
      fprintf(fp, "  <AppendedData encoding=\"raw\">\n   _");

      fwrite(&vel_size, sizeof(int), 1, fp);
      fwrite(vel_buf, sizeof(double), (size_t)(npts * 3), fp);

      fwrite(&pres_size, sizeof(int), 1, fp);
      fwrite(pres_buf, sizeof(double), (size_t)npts, fp);

      fwrite(&div_size, sizeof(int), 1, fp);
      fwrite(div_u, sizeof(double), (size_t)npts, fp);

      fprintf(fp, "\n  </AppendedData>\n");

      free(vel_buf);
      free(pres_buf);
   }
   free(div_u);
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);

   /* PVD file (rank 0) - handled by WritePVDCollection at the end */

   /* Cleanup */
   for (int i = 0; i < 8; i++)
      if (vtk_ghost[i]) free(vtk_ghost[i]);
   free(ext);

   return 0;
}

/*--------------------------------------------------------------------------
 * Get VTK base name (for .pvd file)
 *--------------------------------------------------------------------------*/
void
GetVTKBaseName(LidCavityParams *params, char *buf, size_t bufsize)
{
   snprintf(buf, bufsize, "lidcavity_Re%d_%dx%d_%dx%d", (int)params->Re,
            (int)params->N[0], (int)params->N[1], (int)params->P[0], (int)params->P[1]);
}

/*--------------------------------------------------------------------------
 * Get VTK data directory name
 *--------------------------------------------------------------------------*/
void
GetVTKDataDir(LidCavityParams *params, char *buf, size_t bufsize)
{
   char base[256];
   GetVTKBaseName(params, base, sizeof(base));
   /* Limit base length to leave room for "-data" (5 bytes) */
   size_t base_len     = strlen(base);
   size_t max_base_len = (bufsize > 5) ? bufsize - 5 : 0;
   if (base_len > max_base_len) base_len = max_base_len;
   snprintf(buf, bufsize, "%.*s-data", (int)base_len, base);
}

/*--------------------------------------------------------------------------
 * Write PVD collection file (using internal stats system)
 *--------------------------------------------------------------------------*/
void
WritePVDCollectionFromStats(LidCavityParams *params, int num_procs, double final_time)
{
   int num_steps = HYPREDRV_StatsLevelGetCount(0);
   if (num_steps == 0) return;

   char filename[256];
   GetVTKBaseName(params, filename, sizeof(filename));
   strcat(filename, ".pvd");

   FILE *fp = fopen(filename, "w");
   if (!fp) return;

   char data_dir[256];
   GetVTKDataDir(params, data_dir, sizeof(data_dir));

   fprintf(fp, "<?xml version=\"1.0\"?>\n");
   fprintf(fp,
           "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
   fprintf(fp, "  <Collection>\n");

   /* Check if all timesteps should be included (bit 3: 0x4) */
   int write_all = (params->visualize & 0x4) != 0;

   /* Compute time per step (approximate - assumes uniform dt) */
   double dt_approx = final_time / num_steps;

   /* Determine which timesteps to include in PVD */
   int start_idx = write_all ? 0 : (num_steps - 1); /* All timesteps or only last */
   int end_idx   = num_steps;

   for (int i = start_idx; i < end_idx; i++)
   {
      int timestep_id;
      HYPREDRV_StatsLevelGetEntry(0, i, &timestep_id, NULL, NULL, NULL, NULL);

      double sim_time = timestep_id * dt_approx;

      for (int r = 0; r < num_procs; r++)
      {
         char vtr_file[512];
         snprintf(vtr_file, sizeof(vtr_file), "%s/step_%05d_%d.vtr", data_dir,
                  timestep_id - 1, r);

         fprintf(fp,
                 "    <DataSet timestep=\"%g\" group=\"\" part=\"%d\" file=\"%s\"/>\n",
                 sim_time, r, vtr_file);
      }
   }

   fprintf(fp, "  </Collection>\n");
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);
}

/*--------------------------------------------------------------------------
 * Update time step based on Newton iteration convergence
 *
 * Args:
 *   params: Problem parameters (dt, adaptive_dt, max_cfl flags are used/modified)
 *   newton_iter_count: Number of Newton iterations performed (0-indexed)
 *                      After the loop, this is the last iteration index
 *   h_min: Minimum mesh spacing (for CFL limiting)
 *--------------------------------------------------------------------------*/
void
UpdateTimeStep(LidCavityParams *params, int newton_iter_count, HYPRE_Real h_min)
{
   /* Simple adaptive stepping: double dt if Newton iteration count is 1 */
   if (newton_iter_count == 1)
   {
      params->dt *= 2.0;
   }

   /* More sophisticated adaptive stepping if enabled */
   if (params->adaptive_dt)
   {
      if (newton_iter_count <= 3)
      {
         params->dt *= 1.2;
      }
      else if (newton_iter_count >= 6)
      {
         params->dt *= 0.8;
      }
   }

   /* Enforce maximum CFL constraint if specified */
   if (params->max_cfl > 0.0)
   {
      HYPRE_Real dt_max = params->max_cfl * h_min;
      if (params->dt > dt_max)
      {
         params->dt = dt_max;
      }
   }
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/
int
main(int argc, char *argv[])
{
   MPI_Comm        comm = MPI_COMM_WORLD;
   char           *hypredrv_args[2];
   HYPREDRV_t      hypredrv;
   int             myid, num_procs;
   HYPRE_Real      res_norm;
   LidCavityParams params;
   DistMesh2D     *mesh;
   GhostData      *g_u_k = NULL, *g_u_n = NULL;
   HYPRE_IJMatrix  A;
   HYPRE_IJVector  b;
   HYPRE_IJVector  vec_s[2];
   HYPRE_Real     *u_old;
   HYPRE_Real     *u_now;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   if (ParseArguments(argc, argv, &params, myid, num_procs))
   {
      MPI_Finalize();
      return 1;
   }
   hypredrv_args[0] = params.yaml_file ? params.yaml_file : (char *)default_config;

   if (!myid)
   {
      printf("\n");
      printf("=====================================================\n");
      printf("          Lid-Driven Cavity Problem Setup\n");
      printf("=====================================================\n");
      printf("Physical dimensions:     %d x %d\n", (int)params.L[0], (int)params.L[1]);
      printf("Grid dimensions (nodes): %d x %d\n", (int)params.N[0], (int)params.N[1]);
      printf("Processor topology:      %d x %d\n", (int)params.P[0], (int)params.P[1]);
      printf("Reynolds number:         %.1e\n", params.Re);
      printf("Initial time step:       %.1e\n", params.dt);
      printf("Adaptive time stepping:  %s\n", params.adaptive_dt ? "true" : "false");
      if (params.max_cfl > 0.0)
      {
         printf("Maximum CFL:             %.1f\n", params.max_cfl);
      }
      printf("Regularized lid BC:      %s\n", params.regularize_bc ? "true" : "false");
      printf("Final time:              %.1e\n", params.final_time);
      if (params.visualize)
      {
         const char *format    = (params.visualize & 0x2) ? "ASCII" : "binary";
         const char *timesteps = (params.visualize & 0x4) ? "all" : "last only";
         printf("Visualization:           enabled (%s, %s timesteps)\n", format,
                timesteps);
      }
      else
      {
         printf("Visualization:           disabled\n");
      }
      printf("Verbosity level:         0x%x\n", params.verbose);
      printf("=====================================================\n\n");
   }

   /* Initialize hypredrive library and create main object */
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
   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, hypredrv_args, hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

   /* Create distributed mesh object */
   CreateDistMesh2D(comm, params.L[0], params.L[1], params.N[0], params.N[1], params.P[0],
                    params.P[1], &mesh);

   /* Create state vectors */
   for (int i = 0; i < 2; i++)
   {
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, mesh->dof_ilower, mesh->dof_iupper, &vec_s[i]);
      HYPRE_IJVectorSetObjectType(vec_s[i], HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(vec_s[i]);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorSet(hypredrv, 2, vec_s));

   /* Create ghost exchange objects */
   CreateGhostData(mesh, &g_u_k);
   CreateGhostData(mesh, &g_u_n);

   double current_time = 0.0;
   int    newton_iter, t_step = 0;

   /* Time solver loop */
   for (t_step = 0; current_time < params.final_time; ++t_step)
   {
      if (current_time + params.dt > params.final_time)
      {
         params.dt = params.final_time - current_time;
      }
      current_time += params.dt;

      /* Begin timestep annotation (level 0) - stats accumulated automatically */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(0, "timestep", t_step));

      /* Initialize u_now with the solution from the previous time step */
      HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorCopy(hypredrv, 1, 0));

      /* Retrieve solution values from the old state (U^{n - 1}) and exchange ghost data
       */
      HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 1, &u_old));
      ExchangeVectorGhosts(mesh, u_old, g_u_n);

      /* Non-linear solver loop */
      for (newton_iter = 0; newton_iter < 20; newton_iter++)
      {
         /* Begin Newton iteration annotation (level 1) */
         HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(1, "newton", newton_iter));

         /* Retrieve solution values from the current state (U^{n}) and exchange ghost
          * data */
         HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 0, &u_now));
         ExchangeVectorGhosts(mesh, u_now, g_u_k);

         /* Assemble linear system: J ΔU = -R */
         BuildNewtonSystem(mesh, &params, u_old, u_now, &A, &b, &res_norm, g_u_n, g_u_k);

         /* Check Newton convergence */
         if (newton_iter > 0 && res_norm < params.newton_tol)
         {
            /* Free memory */
            HYPRE_IJMatrixDestroy(A);
            HYPRE_IJVectorDestroy(b);

            HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(1, "newton", newton_iter));
            break;
         }

#if defined(HYPRE_USING_GPU)
         if (!myid && (params.verbose > 0)) printf("Migrating linear system to GPU...");
         HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
         HYPRE_IJVectorMigrate(b, HYPRE_MEMORY_DEVICE);
         if (!myid && (params.verbose > 0)) printf(" Done!\n");
#endif

         /* Tell hypredrv we have 3 interleaved dofs per node */
         HYPREDRV_SAFE_CALL(
            HYPREDRV_LinearSystemSetInterleavedDofmap(hypredrv, mesh->local_size, 3));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));

         if (params.verbose & 0x4)
         {
            if (!myid) printf("Printing linear system...");
            HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemPrint(hypredrv));
            if (!myid) printf(" Done!\n");
         }

         /* Build and apply linear solver */
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));

         /* U = U + ΔU */
         HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorApplyCorrection(hypredrv));

         /* Report running log */
         if (params.verbose > 0)
         {
            int    num_iterations;
            double cfl = params.dt / mesh->h[2];
            double norm_u, norm_v, norm_p;

            HYPREDRV_SAFE_CALL(HYPREDRV_GetLastStat(hypredrv, "iter", &num_iterations));
            ComputeComponentNorms(b, mesh->local_size, &norm_u, &norm_v, &norm_p);
            if (!myid)
            {
               printf(
                  "Time step: %3d | Time: %.4e [s] | CFL: %8.2f | NL: %2d | Lin: %3d | ",
                  t_step + 1, current_time, cfl, newton_iter + 1, num_iterations);
               printf("L2(u)=%.2e  L2(v)=%.2e  L2(p)=%.2e\n", norm_u, norm_v, norm_p);
            }
         }

         /* Free memory */
         HYPRE_IJMatrixDestroy(A);
         HYPRE_IJVectorDestroy(b);

         /* End Newton iteration annotation (level 1) */
         HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(1, "newton", newton_iter));
      }
      if (!myid && params.verbose > 0 && newton_iter > 1) printf("\n");

      /* Update timestep based on Newton convergence */
      UpdateTimeStep(&params, newton_iter, mesh->h[2]);

      /* End timestep annotation (level 0) - stats finalized automatically */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(0, "timestep", t_step));

      /* Save output */
      if (params.visualize)
      {
         int is_last_timestep = (current_time >= params.final_time - 1e-10);
         int write_all        = (params.visualize & 0x4) != 0;
         if (write_all || is_last_timestep)
         {
            WriteVTKsolutionVector(mesh, &params, u_now, g_u_k, t_step + 1);
         }
      }

      /* Update "old" and "new" state vectors (U^{n} = U^{n + 1})*/
      HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorUpdateAll(hypredrv));
   }

   /* Free memory */
   DestroyDistMesh2D(&mesh);
   DestroyGhostData(&g_u_k);
   DestroyGhostData(&g_u_n);
   for (int i = 0; i < 2; i++)
   {
      HYPRE_IJVectorDestroy(vec_s[i]);
   }

   /* Print timestep statistics summary (uses internal stats system) */
   if (!myid && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsLevelPrint(0));
   }

   if (!myid && params.visualize)
   {
      WritePVDCollectionFromStats(&params, num_procs, params.final_time);
   }

   if (!myid && (params.verbose & 0x1)) HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
   if (params.verbose & 0x2) HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}
