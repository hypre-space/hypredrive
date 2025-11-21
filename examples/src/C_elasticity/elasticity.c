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
 * Default solver configuration
 *--------------------------------------------------------------------------*/
static const char *default_config = "solver: pcg\n"
                                    "preconditioner:\n"
                                    "  amg:\n"
                                    "    coarsening:\n"
                                    "      strong_th: 0.6\n"
                                    "      num_functions: 3\n";

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

int PrintUsage(void);
int CreateDistMesh(MPI_Comm, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int,
                   HYPRE_Int, DistMesh **);
int DestroyDistMesh(DistMesh **);
int ParseArguments(int, char **, ElasticParams *, int, int);
int BuildElasticitySystem_Q1Hex(DistMesh *, ElasticParams *, HYPRE_IJMatrix *,
                                HYPRE_IJVector *, MPI_Comm);
int WriteVTKsolutionVector(DistMesh *, ElasticParams *, HYPRE_Real *);
int ComputeRigidBodyModes(DistMesh *, ElasticParams *, HYPRE_Real **);

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
   printf("  -ns|--nsolve <n>  : Number of solves (5)\n");
   printf("  -vis <m>          : Visualization mode (0)\n");
   printf("                         0: none\n");
   printf("                         1: ASCII VTK\n");
   printf("                         2: binary VTK\n");
   printf("  -v|--verbose <n>  : Verbosity bitset (0)\n");
   printf("                         0x1: Linear solver statistics\n");
   printf("                         0x2: Library information\n");
   printf("                         0x4: Linear system printing\n");
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

   return 0;
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

   mesh->gdims[0] = Nx;
   mesh->gdims[1] = Ny;
   mesh->gdims[2] = Nz;
   mesh->pdims[0] = Px;
   mesh->pdims[1] = Py;
   mesh->pdims[2] = Pz;

   MPI_Cart_create(comm, 3, mesh->pdims, (int[]){0, 0, 0}, 1, &(mesh->cart_comm));
   MPI_Comm_rank(mesh->cart_comm, &myid);
   MPI_Cart_coords(mesh->cart_comm, myid, 3, mesh->coords);
   mesh->mypid = (HYPRE_Int)myid;

   for (int i = 0; i < 3; i++)
   {
      HYPRE_Int size = mesh->gdims[i] / mesh->pdims[i];
      HYPRE_Int rest = mesh->gdims[i] - size * mesh->pdims[i];

      mesh->pstarts[i] = calloc(mesh->pdims[i] + 1, sizeof(HYPRE_BigInt));
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
static void
constitutive_matrix_3d(const HYPRE_Real E, const HYPRE_Real nu, HYPRE_Real D[6][6])
{
   HYPRE_Real G   = E / (2.0 * (1.0 + nu));
   HYPRE_Real lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

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
   HYPRE_Int *nnzrow     = (HYPRE_Int *)calloc(local_dofs, sizeof(HYPRE_Int));
   for (int r = 0; r < local_dofs; r++) nnzrow[r] = 81;
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   free(nnzrow);

   HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST);

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
               vals_flat[off + ncols] = Ke_t[i][j];
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
      HYPRE_Int    *ncols_arr = (HYPRE_Int *)malloc(num_rows * sizeof(HYPRE_Int));
      HYPRE_BigInt *rows_arr  = (HYPRE_BigInt *)malloc(num_rows * sizeof(HYPRE_BigInt));
      HYPRE_BigInt *cols_arr  = (HYPRE_BigInt *)malloc(num_rows * sizeof(HYPRE_BigInt));
      HYPRE_Real   *vals_arr  = (HYPRE_Real *)malloc(num_rows * sizeof(HYPRE_Real));
      HYPRE_Real   *rhs_arr   = (HYPRE_Real *)malloc(num_rows * sizeof(HYPRE_Real));

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
      ghost[0] = (double *)calloc(ny * nz * 3, sizeof(double)); /* left  face */
   if (mesh->nbrs[1] != MPI_PROC_NULL)
      ghost[1] = (double *)calloc(ny * nz * 3, sizeof(double)); /* right face send */
   if (mesh->nbrs[2] != MPI_PROC_NULL)
      ghost[2] = (double *)calloc(nx * nz * 3, sizeof(double)); /* down  face */
   if (mesh->nbrs[3] != MPI_PROC_NULL)
      ghost[3] = (double *)calloc(nx * nz * 3, sizeof(double)); /* up    face send */
   if (mesh->nbrs[4] != MPI_PROC_NULL)
      ghost[4] = (double *)calloc(nx * ny * 3, sizeof(double)); /* back  face */
   if (mesh->nbrs[5] != MPI_PROC_NULL)
      ghost[5] = (double *)calloc(nx * ny * 3, sizeof(double)); /* front face send */
   if (mesh->nbrs[6] != MPI_PROC_NULL)
      ghost[6] = (double *)calloc(nz * 3, sizeof(double)); /* left-down edge */
   if (mesh->nbrs[7] != MPI_PROC_NULL)
      ghost[7] = (double *)calloc(nz * 3, sizeof(double)); /* right-up  edge send */
   if (mesh->nbrs[8] != MPI_PROC_NULL)
      ghost[8] = (double *)calloc(ny * 3, sizeof(double)); /* left-back edge */
   if (mesh->nbrs[9] != MPI_PROC_NULL)
      ghost[9] = (double *)calloc(ny * 3, sizeof(double)); /* right-front edge send */
   if (mesh->nbrs[10] != MPI_PROC_NULL)
      ghost[10] = (double *)calloc(nx * 3, sizeof(double)); /* down-back edge */
   if (mesh->nbrs[11] != MPI_PROC_NULL)
      ghost[11] = (double *)calloc(nx * 3, sizeof(double)); /* up-front  edge send */
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
   double *ext = (double *)calloc(nxg * nyg * nzg * 3, sizeof(double));
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
   if (reqc > 0) MPI_Waitall(reqc, reqs, MPI_STATUSES_IGNORE);

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
      int npts      = nxg * nyg * nzg;
      int data_size = npts * 3 * sizeof(double);
      fwrite(&data_size, sizeof(int), 1, fp);
      fwrite(ext, sizeof(double), npts * 3, fp);
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
   HYPRE_Real    *gsizes  = &mesh->gsizes[0];

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
      (HYPRE_Real *)calloc((size_t)num_modes * stride_mode, sizeof(HYPRE_Real));
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
   HYPRE_Real    *sol_data;
   HYPRE_Real    *rbms;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   if (ParseArguments(argc, argv, &params, myid, num_procs))
   {
      MPI_Finalize();
      return 1;
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());

   if (params.verbose & 0x2)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm));
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));

   char *args[2];
   args[0] = params.yaml_file ? params.yaml_file : (char *)default_config;
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));

   HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

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
      printf("Material:                E=%.1e, nu=%.3f\n", params.E, params.nu);
      printf("Body force:              rho=%.1e, g=(%.1f, %.1f, %.1f)\n", params.rho,
             params.g[0], params.g[1], params.g[2]);
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

   if (!myid && (params.verbose > 0)) printf("Assembling linear system...");
   HYPREDRV_SAFE_CALL(HYPREDRV_TimerStart("system"));
   BuildElasticitySystem_Q1Hex(mesh, &params, &A, &b, comm);
   ComputeRigidBodyModes(mesh, &params, &rbms);
   HYPREDRV_SAFE_CALL(HYPREDRV_TimerStop("system"));
   if (!myid && (params.verbose > 0)) printf(" Done!\n");

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
   HYPREDRV_SAFE_CALL(
      HYPREDRV_LinearSystemSetNearNullSpace(hypredrv, 3 * mesh->local_size, 6, rbms));

   if (params.verbose & 0x4)
   {
      if (!myid) printf("Printing linear system...");
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemPrint(hypredrv));
      if (!myid) printf(" Done!\n");
   }

   for (int isolve = 0; isolve < params.nsolve; isolve++)
   {
      if (!myid) printf("Solve %d/%d...\n", isolve + 1, params.nsolve);
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_PreconDestroy(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));
   }

   if (!myid && (params.verbose & 0x1)) HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));

   if (params.visualize)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol_data));
      WriteVTKsolutionVector(mesh, &params, sol_data);
   }

   /* Free memory */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   DestroyDistMesh(&mesh);
   free(rbms);

   if (params.verbose & 0x2) HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}
