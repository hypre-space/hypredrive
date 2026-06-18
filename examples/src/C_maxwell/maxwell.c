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
 *   Definite Maxwell (curl-curl + mass) Example Driver -- AMS preconditioner
 *==========================================================================
 *
 *   PDE (definite Maxwell):
 *   -----------------------
 *      curl (muinv curl E) + sigma E = f   in  Omega = [0,Lx]x[0,Ly]x[0,Lz]
 *      E x n = prescribed                  on  the boundary
 *
 *   In this example muinv = sigma = 1, so the operator reduces to
 *   curl-curl + mass. The mass term (sigma > 0) makes the problem definite,
 *   which is the ideal target for the Auxiliary-space Maxwell Solver (AMS).
 *
 *   Discretization:
 *   ---------------
 *      Lowest order Nedelec ("edge") elements in H(curl) on a structured
 *      hexahedral grid. Each edge degree of freedom is the integral of the
 *      tangential component of E along the (axis-aligned, +axis oriented) edge.
 *
 *   Manufactured solution (benchmark):
 *   ----------------------------------
 *      E = (sin(k y), sin(k z), sin(k x)),  k = freq * pi.
 *      It satisfies curl curl E = k^2 E, so f = (1 + k^2) E. This gives an
 *      exact reference field used to measure the discretization error and to
 *      verify the solver converges to the right answer (not just a small
 *      residual). The reference edge DOFs have the closed form
 *         x-edge: hx * sin(k * y_node),   y-edge: hy * sin(k * z_node),
 *         z-edge: hz * sin(k * x_node).
 *
 *   AMS operator inputs (set through HYPREDRV):
 *   -------------------------------------------
 *      - the discrete gradient G (edge x node incidence, -1 tail / +1 head),
 *      - the vertex coordinate vectors (xcoord, ycoord, zcoord).
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Problem parameters
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int  N[3];          /* grid dimensions in nodes */
   HYPRE_Int  P[3];          /* processor grid dimensions */
   HYPRE_Real L[3];          /* physical domain dimensions */
   HYPRE_Real freq;          /* manufactured-solution frequency (kappa = freq*pi) */
   HYPRE_Int  nsolve;        /* number of solves */
   HYPRE_Int  verbose;       /* verbosity bitset */
   char      *yaml_file;     /* YAML configuration file */
   char      *solver_preset; /* preset selector when no YAML file is given */
} MaxwellParams;

/*--------------------------------------------------------------------------
 * Distributed structured mesh (node partitioning, shared by all ranks)
 *--------------------------------------------------------------------------*/
typedef struct
{
   MPI_Comm      cart_comm;
   HYPRE_Int     mypid;
   HYPRE_Int     nprocs;
   HYPRE_Int     gdims[3]; /* global node dimensions */
   HYPRE_Int     pdims[3]; /* processor grid dimensions */
   HYPRE_Int     coords[3];
   HYPRE_Int     nlocal[3];
   HYPRE_BigInt *pstarts[3]; /* node partition boundaries per dimension */
   HYPRE_Real    h[3];       /* mesh spacing per dimension */

   /* Node ownership (scalar) */
   HYPRE_BigInt node_ilower;
   HYPRE_BigInt node_iupper;
   HYPRE_Int    num_nodes_local;

   /* Edge ownership (rank-contiguous numbering derived from pstarts) */
   HYPRE_BigInt  edge_ilower;
   HYPRE_BigInt  edge_iupper;
   HYPRE_Int     num_edges_local;
   HYPRE_BigInt  num_edges_global;
   HYPRE_BigInt *rank_edge_lower; /* global edge start per MPI rank (size nprocs+1) */
   HYPRE_BigInt *rank_node_lower; /* global node start per MPI rank (size nprocs+1) */
   HYPRE_Int   (*rank_coords)[3]; /* Cartesian coords per MPI rank */
} MaxwellMesh;

enum
{
   EDGE_X = 0,
   EDGE_Y = 1,
   EDGE_Z = 2
};

/* Find the processor-grid block index that owns global index g in dimension d. */
static inline HYPRE_Int
block_of(HYPRE_BigInt g, const HYPRE_BigInt *ps, HYPRE_Int P)
{
   HYPRE_Int b = 0;
   while ((b + 1) < P && g >= ps[b + 1]) b++;
   return b;
}

/* Global node id of node (gi,gj,gk). Rank-monotonic: rank r owns a contiguous
 * range starting at rank_node_lower[r]; within a rank the nodes are ordered
 * lexicographically (x fastest). This mirrors the edge numbering so the node
 * (column/coordinate) partition stays consistent with the edge partition, which
 * AMS requires when forming G^T A G. */
static HYPRE_BigInt
node_gid(const MaxwellMesh *m, HYPRE_BigInt gi, HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   HYPRE_Int px = block_of(gi, m->pstarts[0], m->pdims[0]);
   HYPRE_Int py = block_of(gj, m->pstarts[1], m->pdims[1]);
   HYPRE_Int pz = block_of(gk, m->pstarts[2], m->pdims[2]);
   int       owner;
   MPI_Cart_rank(m->cart_comm, (int[]){(int)px, (int)py, (int)pz}, &owner);

   HYPRE_Int nx = (HYPRE_Int)(m->pstarts[0][px + 1] - m->pstarts[0][px]);
   HYPRE_Int ny = (HYPRE_Int)(m->pstarts[1][py + 1] - m->pstarts[1][py]);
   HYPRE_Int li = (HYPRE_Int)(gi - m->pstarts[0][px]);
   HYPRE_Int lj = (HYPRE_Int)(gj - m->pstarts[1][py]);
   HYPRE_Int lk = (HYPRE_Int)(gk - m->pstarts[2][pz]);

   return m->rank_node_lower[owner] + (HYPRE_BigInt)((lk * ny + lj) * nx + li);
}

/* Number of edges of the given family owned by the rank whose block is given.
 * An edge is owned by the rank owning its lower-index node; it exists only if
 * the upper node is inside the domain (lower index < N-1 in the edge direction). */
static HYPRE_Int
block_edge_count(const MaxwellMesh *m, int family, HYPRE_Int px, HYPRE_Int py, HYPRE_Int pz)
{
   HYPRE_Int nx = (HYPRE_Int)(m->pstarts[0][px + 1] - m->pstarts[0][px]);
   HYPRE_Int ny = (HYPRE_Int)(m->pstarts[1][py + 1] - m->pstarts[1][py]);
   HYPRE_Int nz = (HYPRE_Int)(m->pstarts[2][pz + 1] - m->pstarts[2][pz]);

   /* Whether the block touches the far domain boundary in each dimension. */
   HYPRE_Int top_x = (m->pstarts[0][px + 1] == m->gdims[0]);
   HYPRE_Int top_y = (m->pstarts[1][py + 1] == m->gdims[1]);
   HYPRE_Int top_z = (m->pstarts[2][pz + 1] == m->gdims[2]);

   HYPRE_Int bx = top_x ? (nx - 1) : nx; /* valid lower-i count for x-edges */
   HYPRE_Int by = top_y ? (ny - 1) : ny;
   HYPRE_Int bz = top_z ? (nz - 1) : nz;

   if (family == EDGE_X) return bx * ny * nz;
   if (family == EDGE_Y) return nx * by * nz;
   return nx * ny * bz; /* EDGE_Z */
}

/* Global edge id of the family-edge whose lower node is (gi,gj,gk). */
static HYPRE_BigInt
edge_gid(const MaxwellMesh *m, int family, HYPRE_BigInt gi, HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   HYPRE_Int px = block_of(gi, m->pstarts[0], m->pdims[0]);
   HYPRE_Int py = block_of(gj, m->pstarts[1], m->pdims[1]);
   HYPRE_Int pz = block_of(gk, m->pstarts[2], m->pdims[2]);
   int       owner;
   MPI_Cart_rank(m->cart_comm, (int[]){(int)px, (int)py, (int)pz}, &owner);

   HYPRE_BigInt base = m->rank_edge_lower[owner];

   HYPRE_Int x0 = (HYPRE_Int)m->pstarts[0][px], x1 = (HYPRE_Int)m->pstarts[0][px + 1];
   HYPRE_Int y0 = (HYPRE_Int)m->pstarts[1][py], y1 = (HYPRE_Int)m->pstarts[1][py + 1];
   HYPRE_Int z0 = (HYPRE_Int)m->pstarts[2][pz], z1 = (HYPRE_Int)m->pstarts[2][pz + 1];
   HYPRE_Int nx = x1 - x0, ny = y1 - y0, nz = z1 - z0;
   HYPRE_Int bx = (x1 == m->gdims[0]) ? (nx - 1) : nx;
   HYPRE_Int by = (y1 == m->gdims[1]) ? (ny - 1) : ny;

   HYPRE_Int cntX = bx * ny * nz;
   HYPRE_Int cntY = nx * by * nz;

   HYPRE_Int li = (HYPRE_Int)(gi - x0);
   HYPRE_Int lj = (HYPRE_Int)(gj - y0);
   HYPRE_Int lk = (HYPRE_Int)(gk - z0);

   HYPRE_BigInt local;
   if (family == EDGE_X)
   {
      local = (HYPRE_BigInt)((lk * ny + lj) * bx + li);
   }
   else if (family == EDGE_Y)
   {
      local = (HYPRE_BigInt)cntX + (HYPRE_BigInt)((lk * by + lj) * nx + li);
   }
   else /* EDGE_Z */
   {
      local = (HYPRE_BigInt)(cntX + cntY) + (HYPRE_BigInt)((lk * ny + lj) * nx + li);
   }

   return base + local;
}

/* Is the family-edge with lower node (gi,gj,gk) on the domain boundary? */
static int
edge_is_boundary(const MaxwellMesh *m, int family, HYPRE_BigInt gi, HYPRE_BigInt gj,
                 HYPRE_BigInt gk)
{
   HYPRE_BigInt Nx = m->gdims[0] - 1, Ny = m->gdims[1] - 1, Nz = m->gdims[2] - 1;
   if (family == EDGE_X) return (gj == 0 || gj == Ny || gk == 0 || gk == Nz);
   if (family == EDGE_Y) return (gi == 0 || gi == Nx || gk == 0 || gk == Nz);
   return (gi == 0 || gi == Nx || gj == 0 || gj == Ny); /* EDGE_Z */
}

/* Exact (manufactured) reference value of a family-edge DOF: integral of the
 * tangential component of E along the edge. */
static HYPRE_Real
edge_exact(const MaxwellMesh *m, HYPRE_Real kappa, int family, HYPRE_BigInt gi,
           HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   HYPRE_Real x = gi * m->h[0], y = gj * m->h[1], z = gk * m->h[2];
   if (family == EDGE_X) return m->h[0] * sin(kappa * y); /* E_x = sin(k y) */
   if (family == EDGE_Y) return m->h[1] * sin(kappa * z); /* E_y = sin(k z) */
   return m->h[2] * sin(kappa * x);                       /* E_z = sin(k x) */
}

/*--------------------------------------------------------------------------
 * Print usage
 *--------------------------------------------------------------------------*/
static int
PrintUsage(void)
{
   printf("\n");
   printf("Usage: ${MPIEXEC_COMMAND} <np> ./maxwell [options]\n\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML configuration file (Opt.)\n");
   printf("  -n <nx> <ny> <nz> : Global grid dimensions in nodes (17 17 17)\n");
   printf("  -P <Px> <Py> <Pz> : Processor grid dimensions (1 1 1)\n");
   printf("  -L <Lx> <Ly> <Lz> : Physical dimensions (1 1 1)\n");
   printf("  -freq <f>         : Manufactured frequency, kappa = f*pi (1.0)\n");
   printf("  -ns|--nsolve <n>  : Number of solves (1)\n");
   printf("  -v|--verbose <n>  : Verbosity bitset (1)\n");
   printf("  -h|--help         : Print this message\n\n");
   return 0;
}

/*--------------------------------------------------------------------------
 * Parse command line arguments
 *--------------------------------------------------------------------------*/
static int
ParseArguments(int argc, char *argv[], MaxwellParams *params, int myid, int num_procs)
{
   for (int i = 0; i < 3; i++)
   {
      params->N[i] = 17;
      params->P[i] = 1;
      params->L[i] = 1.0;
   }
   params->freq          = 1.0;
   params->nsolve        = 1;
   params->verbose       = 1;
   params->yaml_file     = NULL;
   params->solver_preset = "pcg";

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
         for (int j = 0; j < 3; j++) params->N[j] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-P"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -P requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++) params->P[j] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-L"))
      {
         if (i + 3 >= argc)
         {
            if (!myid) printf("Error: -L requires three values\n");
            return 1;
         }
         for (int j = 0; j < 3; j++) params->L[j] = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-freq"))
      {
         if (++i < argc) params->freq = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-ns") || !strcmp(argv[i], "--nsolve"))
      {
         if (++i < argc) params->nsolve = atoi(argv[i]);
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

   if (params->P[0] * params->P[1] * params->P[2] != num_procs)
   {
      if (!myid)
      {
         printf("Error: number of processes (%d) must equal Px*Py*Pz (%d)\n", num_procs,
                params->P[0] * params->P[1] * params->P[2]);
      }
      return 1;
   }
   for (int d = 0; d < 3; d++)
   {
      if (params->N[d] < 2)
      {
         if (!myid) printf("Error: each grid dimension must be >= 2 nodes\n");
         return 1;
      }
      if (params->P[d] > params->N[d] - 1)
      {
         if (!myid) printf("Error: too many ranks in dimension %d\n", d);
         return 1;
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * Create the distributed mesh (node partitioning + edge numbering)
 *--------------------------------------------------------------------------*/
static int
CreateMesh(MPI_Comm comm, MaxwellParams *params, MaxwellMesh **mesh_ptr)
{
   MaxwellMesh *m = (MaxwellMesh *)calloc(1, sizeof(MaxwellMesh));
   int          myid;

   for (int d = 0; d < 3; d++)
   {
      m->gdims[d] = params->N[d];
      m->pdims[d] = params->P[d];
      m->h[d]     = params->L[d] / (params->N[d] - 1);
   }

   int mpi_dims[3]   = {(int)m->pdims[0], (int)m->pdims[1], (int)m->pdims[2]};
   int mpi_coords[3] = {0, 0, 0};
   MPI_Cart_create(comm, 3, mpi_dims, (int[]){0, 0, 0}, 0, &m->cart_comm);
   MPI_Comm_rank(m->cart_comm, &myid);
   MPI_Comm_size(m->cart_comm, &m->nprocs);
   MPI_Cart_coords(m->cart_comm, myid, 3, mpi_coords);
   m->mypid     = myid;
   m->coords[0] = mpi_coords[0];
   m->coords[1] = mpi_coords[1];
   m->coords[2] = mpi_coords[2];

   for (int d = 0; d < 3; d++)
   {
      HYPRE_Int size  = m->gdims[d] / m->pdims[d];
      HYPRE_Int rest  = m->gdims[d] - size * m->pdims[d];
      m->pstarts[d]   = (HYPRE_BigInt *)calloc((size_t)(m->pdims[d] + 1), sizeof(HYPRE_BigInt));
      for (int j = 0; j < m->pdims[d] + 1; j++)
      {
         m->pstarts[d][j] = (HYPRE_BigInt)(size * j + (j < rest ? j : rest));
      }
      m->nlocal[d] = (HYPRE_Int)(m->pstarts[d][m->coords[d] + 1] - m->pstarts[d][m->coords[d]]);
   }

   m->num_nodes_local = m->nlocal[0] * m->nlocal[1] * m->nlocal[2];

   /* Rank-monotonic node and edge numbering. Gather each rank's Cartesian coords
    * then prefix-sum the per-rank node and edge counts (in MPI-rank order) so
    * that both partitions are contiguous and ordered by rank. Using the same
    * scheme for nodes and edges keeps G's column/coordinate partition consistent
    * with its edge row partition, as required by AMS. */
   m->rank_coords     = malloc((size_t)m->nprocs * sizeof(*m->rank_coords));
   m->rank_edge_lower = malloc((size_t)(m->nprocs + 1) * sizeof(HYPRE_BigInt));
   m->rank_node_lower = malloc((size_t)(m->nprocs + 1) * sizeof(HYPRE_BigInt));
   for (int r = 0; r < m->nprocs; r++)
   {
      int rc[3];
      MPI_Cart_coords(m->cart_comm, r, 3, rc);
      m->rank_coords[r][0] = rc[0];
      m->rank_coords[r][1] = rc[1];
      m->rank_coords[r][2] = rc[2];
   }
   m->rank_edge_lower[0] = 0;
   m->rank_node_lower[0] = 0;
   for (int r = 0; r < m->nprocs; r++)
   {
      HYPRE_Int px = m->rank_coords[r][0], py = m->rank_coords[r][1], pz = m->rank_coords[r][2];
      HYPRE_Int rnx = (HYPRE_Int)(m->pstarts[0][px + 1] - m->pstarts[0][px]);
      HYPRE_Int rny = (HYPRE_Int)(m->pstarts[1][py + 1] - m->pstarts[1][py]);
      HYPRE_Int rnz = (HYPRE_Int)(m->pstarts[2][pz + 1] - m->pstarts[2][pz]);
      HYPRE_Int ecnt = block_edge_count(m, EDGE_X, px, py, pz) +
                       block_edge_count(m, EDGE_Y, px, py, pz) +
                       block_edge_count(m, EDGE_Z, px, py, pz);
      m->rank_edge_lower[r + 1] = m->rank_edge_lower[r] + ecnt;
      m->rank_node_lower[r + 1] = m->rank_node_lower[r] + (HYPRE_BigInt)(rnx * rny * rnz);
   }
   m->num_edges_global = m->rank_edge_lower[m->nprocs];
   m->num_edges_local  = (HYPRE_Int)(m->rank_edge_lower[m->mypid + 1] - m->rank_edge_lower[m->mypid]);
   m->edge_ilower      = m->rank_edge_lower[m->mypid];
   m->edge_iupper      = m->edge_ilower + m->num_edges_local - 1;
   m->node_ilower      = m->rank_node_lower[m->mypid];
   m->node_iupper      = m->node_ilower + m->num_nodes_local - 1;

   *mesh_ptr = m;
   return 0;
}

static void
DestroyMesh(MaxwellMesh **mesh_ptr)
{
   MaxwellMesh *m = *mesh_ptr;
   if (!m) return;
   for (int d = 0; d < 3; d++) free(m->pstarts[d]);
   free(m->rank_coords);
   free(m->rank_edge_lower);
   free(m->rank_node_lower);
   MPI_Comm_free(&m->cart_comm);
   free(m);
   *mesh_ptr = NULL;
}

/*--------------------------------------------------------------------------
 * Lowest-order Nedelec edge basis on a brick element [0,hx]x[0,hy]x[0,hz].
 *
 * Local edge ordering (lower node -> upper node, +axis oriented):
 *   0..3 : x-edges at (y-corner, z-corner) = (0,0),(1,0),(0,1),(1,1)
 *   4..7 : y-edges at (x-corner, z-corner) = (0,0),(1,0),(0,1),(1,1)
 *   8..11: z-edges at (x-corner, y-corner) = (0,0),(1,0),(0,1),(1,1)
 *--------------------------------------------------------------------------*/
static const int edge_dir[12] = {EDGE_X, EDGE_X, EDGE_X, EDGE_X, EDGE_Y, EDGE_Y,
                                 EDGE_Y, EDGE_Y, EDGE_Z, EDGE_Z, EDGE_Z, EDGE_Z};
/* transverse corners (c1,c2) per edge in (lo..hi) order described above */
static const int edge_c1[12] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
static const int edge_c2[12] = {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};

/* Evaluate Whitney edge basis function 'e' and its curl at reference-mapped
 * physical point with local coordinates (lx,ly,lz) in [0,h]. */
static void
edge_basis(int e, HYPRE_Real lx, HYPRE_Real ly, HYPRE_Real lz, const HYPRE_Real h[3],
           HYPRE_Real W[3], HYPRE_Real curlW[3])
{
   HYPRE_Real ax[2] = {1.0 - lx / h[0], lx / h[0]}, dax[2] = {-1.0 / h[0], 1.0 / h[0]};
   HYPRE_Real ay[2] = {1.0 - ly / h[1], ly / h[1]}, day[2] = {-1.0 / h[1], 1.0 / h[1]};
   HYPRE_Real az[2] = {1.0 - lz / h[2], lz / h[2]}, daz[2] = {-1.0 / h[2], 1.0 / h[2]};

   W[0] = W[1] = W[2] = 0.0;
   curlW[0] = curlW[1] = curlW[2] = 0.0;

   int c1 = edge_c1[e], c2 = edge_c2[e];
   if (edge_dir[e] == EDGE_X) /* W = ay[c1] az[c2] / hx in x */
   {
      W[0]     = ay[c1] * az[c2] / h[0];
      curlW[1] = ay[c1] * daz[c2] / h[0];  /* dWx/dz */
      curlW[2] = -day[c1] * az[c2] / h[0]; /* -dWx/dy */
   }
   else if (edge_dir[e] == EDGE_Y) /* W = ax[c1] az[c2] / hy in y */
   {
      W[1]     = ax[c1] * az[c2] / h[1];
      curlW[0] = -ax[c1] * daz[c2] / h[1]; /* -dWy/dz */
      curlW[2] = dax[c1] * az[c2] / h[1];  /* dWy/dx */
   }
   else /* EDGE_Z: W = ax[c1] ay[c2] / hz in z */
   {
      W[2]     = ax[c1] * ay[c2] / h[2];
      curlW[0] = ax[c1] * day[c2] / h[2];  /* dWz/dy */
      curlW[1] = -dax[c1] * ay[c2] / h[2]; /* -dWz/dx */
   }
}

/* Lower-node offset (a,b,c in {0,1}) of edge 'e' within a cell. */
static void
edge_lower_offset(int e, int off[3])
{
   int c1 = edge_c1[e], c2 = edge_c2[e];
   if (edge_dir[e] == EDGE_X) { off[0] = 0; off[1] = c1; off[2] = c2; }
   else if (edge_dir[e] == EDGE_Y) { off[0] = c1; off[1] = 0; off[2] = c2; }
   else { off[0] = c1; off[1] = c2; off[2] = 0; }
}

/* 3-point Gauss rule on [0,1] (mapped to [0,h]). */
static const HYPRE_Real g3_x[3] = {0.1127016653792583, 0.5, 0.8872983346207417};
static const HYPRE_Real g3_w[3] = {0.2777777777777778, 0.4444444444444444,
                                   0.2777777777777778};

/* Precompute the (constant) 12x12 element matrix S = curl-curl + mass. */
static void
ComputeElementMatrix(const HYPRE_Real h[3], HYPRE_Real S[12][12])
{
   HYPRE_Real detJ = h[0] * h[1] * h[2];
   for (int a = 0; a < 12; a++)
      for (int b = 0; b < 12; b++) S[a][b] = 0.0;

   for (int qz = 0; qz < 3; qz++)
      for (int qy = 0; qy < 3; qy++)
         for (int qx = 0; qx < 3; qx++)
         {
            HYPRE_Real lx = g3_x[qx] * h[0], ly = g3_x[qy] * h[1], lz = g3_x[qz] * h[2];
            HYPRE_Real w  = g3_w[qx] * g3_w[qy] * g3_w[qz] * detJ;
            HYPRE_Real W[12][3], C[12][3];
            for (int e = 0; e < 12; e++) edge_basis(e, lx, ly, lz, h, W[e], C[e]);
            for (int a = 0; a < 12; a++)
               for (int b = 0; b < 12; b++)
               {
                  HYPRE_Real m = W[a][0] * W[b][0] + W[a][1] * W[b][1] + W[a][2] * W[b][2];
                  HYPRE_Real s = C[a][0] * C[b][0] + C[a][1] * C[b][1] + C[a][2] * C[b][2];
                  S[a][b] += w * (s + m); /* muinv = sigma = 1 */
               }
         }
}

/*--------------------------------------------------------------------------
 * Build the Maxwell system: matrix A, rhs b, discrete gradient G, vertex
 * coordinates, and the reference (exact) edge DOFs.
 *--------------------------------------------------------------------------*/
static int
BuildMaxwellSystem(MaxwellMesh *m, MaxwellParams *params, MPI_Comm comm, HYPRE_IJMatrix *A_ptr,
                   HYPRE_IJVector *b_ptr, HYPRE_IJMatrix *G_ptr, HYPRE_IJVector coord_ptr[3],
                   HYPRE_Real **xref_ptr)
{
   const HYPRE_Real kappa = params->freq * M_PI;
   const HYPRE_Real fcoef = 1.0 + kappa * kappa;
   const HYPRE_Int  Nx = m->gdims[0], Ny = m->gdims[1], Nz = m->gdims[2];

   HYPRE_IJMatrix A, G;
   HYPRE_IJVector b, cx, cy, cz;

   HYPRE_IJMatrixCreate(comm, m->edge_ilower, m->edge_iupper, m->edge_ilower, m->edge_iupper, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);

   HYPRE_IJMatrixCreate(comm, m->edge_ilower, m->edge_iupper, m->node_ilower, m->node_iupper, &G);
   HYPRE_IJMatrixSetObjectType(G, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(G);

   HYPRE_IJVectorCreate(comm, m->edge_ilower, m->edge_iupper, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(comm, m->node_ilower, m->node_iupper, &cx);
   HYPRE_IJVectorCreate(comm, m->node_ilower, m->node_iupper, &cy);
   HYPRE_IJVectorCreate(comm, m->node_ilower, m->node_iupper, &cz);
   HYPRE_IJVectorSetObjectType(cx, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(cy, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(cz, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(cx);
   HYPRE_IJVectorInitialize(cy);
   HYPRE_IJVectorInitialize(cz);

   HYPRE_Real *xref = (HYPRE_Real *)calloc((size_t)(m->num_edges_local > 0 ? m->num_edges_local : 1),
                                           sizeof(HYPRE_Real));

   HYPRE_Real S[12][12];
   ComputeElementMatrix(m->h, S);

   /* ---- Assemble owned interior edge rows of A and the load b ----
    * Loop cells whose lower corner lies in an extended (ghost) range so that
    * every owned edge receives contributions from all incident cells. Only
    * rows owned by this rank are added (AddToValues accumulates correctly). */
   HYPRE_Int cx_lo = (m->pstarts[0][m->coords[0]] > 0) ? (HYPRE_Int)m->pstarts[0][m->coords[0]] - 1 : 0;
   HYPRE_Int cy_lo = (m->pstarts[1][m->coords[1]] > 0) ? (HYPRE_Int)m->pstarts[1][m->coords[1]] - 1 : 0;
   HYPRE_Int cz_lo = (m->pstarts[2][m->coords[2]] > 0) ? (HYPRE_Int)m->pstarts[2][m->coords[2]] - 1 : 0;
   HYPRE_Int cx_hi = (HYPRE_Int)((m->pstarts[0][m->coords[0] + 1] < Nx) ? m->pstarts[0][m->coords[0] + 1] : Nx - 1);
   HYPRE_Int cy_hi = (HYPRE_Int)((m->pstarts[1][m->coords[1] + 1] < Ny) ? m->pstarts[1][m->coords[1] + 1] : Ny - 1);
   HYPRE_Int cz_hi = (HYPRE_Int)((m->pstarts[2][m->coords[2] + 1] < Nz) ? m->pstarts[2][m->coords[2] + 1] : Nz - 1);

   for (HYPRE_Int cz = cz_lo; cz < cz_hi; cz++)
   for (HYPRE_Int cy = cy_lo; cy < cy_hi; cy++)
   for (HYPRE_Int cx = cx_lo; cx < cx_hi; cx++)
   {
      /* Global edge ids, boundary flags, exact values, and element load. */
      HYPRE_BigInt egid[12];
      int          ebnd[12];
      HYPRE_Real   eval[12];
      HYPRE_Real   F[12];
      for (int e = 0; e < 12; e++)
      {
         int off[3];
         edge_lower_offset(e, off);
         HYPRE_BigInt gi = cx + off[0], gj = cy + off[1], gk = cz + off[2];
         egid[e] = edge_gid(m, edge_dir[e], gi, gj, gk);
         ebnd[e] = edge_is_boundary(m, edge_dir[e], gi, gj, gk);
         eval[e] = edge_exact(m, kappa, edge_dir[e], gi, gj, gk);
         F[e]    = 0.0;
      }

      /* Element load F_e = integral of f . W_e over the cell, f = fcoef * E. */
      for (int qz = 0; qz < 3; qz++)
      for (int qy = 0; qy < 3; qy++)
      for (int qx = 0; qx < 3; qx++)
      {
         HYPRE_Real lx = g3_x[qx] * m->h[0], ly = g3_x[qy] * m->h[1], lz = g3_x[qz] * m->h[2];
         HYPRE_Real px = cx * m->h[0] + lx, py = cy * m->h[1] + ly, pz = cz * m->h[2] + lz;
         HYPRE_Real wq = g3_w[qx] * g3_w[qy] * g3_w[qz] * (m->h[0] * m->h[1] * m->h[2]);
         HYPRE_Real fx = fcoef * sin(kappa * py); /* f = (1+k^2) E */
         HYPRE_Real fy = fcoef * sin(kappa * pz);
         HYPRE_Real fz = fcoef * sin(kappa * px);
         for (int e = 0; e < 12; e++)
         {
            HYPRE_Real W[3], C[3];
            edge_basis(e, lx, ly, lz, m->h, W, C);
            F[e] += wq * (fx * W[0] + fy * W[1] + fz * W[2]);
         }
      }

      /* Add owned interior rows; lift inhomogeneous Dirichlet columns to RHS. */
      for (int a = 0; a < 12; a++)
      {
         if (ebnd[a]) continue; /* boundary rows handled separately */
         /* Only add contributions to rows this rank owns. */
         if (egid[a] < m->edge_ilower || egid[a] > m->edge_iupper) continue;

         HYPRE_BigInt row = egid[a];
         HYPRE_Int    ncols = 0;
         HYPRE_BigInt cols[12];
         HYPRE_Real   vals[12];
         HYPRE_Real   rhs = F[a];
         for (int bcol = 0; bcol < 12; bcol++)
         {
            if (ebnd[bcol])
            {
               rhs -= S[a][bcol] * eval[bcol]; /* move known boundary value to RHS */
            }
            else
            {
               cols[ncols] = egid[bcol];
               vals[ncols] = S[a][bcol];
               ncols++;
            }
         }
         HYPRE_IJMatrixAddToValues(A, 1, &ncols, &row, cols, vals);
         HYPRE_IJVectorAddToValues(b, 1, &row, &rhs);
      }
   }

   /* ---- Owned-edge pass: boundary identity rows, G rows, xref ---- */
   for (HYPRE_Int lk = 0; lk < m->nlocal[2]; lk++)
   for (HYPRE_Int lj = 0; lj < m->nlocal[1]; lj++)
   for (HYPRE_Int li = 0; li < m->nlocal[0]; li++)
   {
      HYPRE_BigInt gi = m->pstarts[0][m->coords[0]] + li;
      HYPRE_BigInt gj = m->pstarts[1][m->coords[1]] + lj;
      HYPRE_BigInt gk = m->pstarts[2][m->coords[2]] + lk;

      for (int fam = 0; fam < 3; fam++)
      {
         /* Edge exists if the upper node is inside the domain. */
         if (fam == EDGE_X && gi >= Nx - 1) continue;
         if (fam == EDGE_Y && gj >= Ny - 1) continue;
         if (fam == EDGE_Z && gk >= Nz - 1) continue;

         HYPRE_BigInt row = edge_gid(m, fam, gi, gj, gk);
         if (row < m->edge_ilower || row > m->edge_iupper) continue; /* owned by us */

         HYPRE_Real exact = edge_exact(m, kappa, fam, gi, gj, gk);
         xref[row - m->edge_ilower] = exact;

         /* Discrete gradient row: -1 at lower node, +1 at upper node. */
         HYPRE_BigInt up[3] = {gi, gj, gk};
         up[fam] += 1;
         HYPRE_BigInt gcols[2] = {node_gid(m, gi, gj, gk), node_gid(m, up[0], up[1], up[2])};
         HYPRE_Real   gvals[2] = {-1.0, 1.0};
         HYPRE_Int    gnc      = 2;
         HYPRE_IJMatrixSetValues(G, 1, &gnc, &row, gcols, gvals);

         if (edge_is_boundary(m, fam, gi, gj, gk))
         {
            HYPRE_Int  one = 1;
            HYPRE_Real diag = 1.0;
            HYPRE_IJMatrixSetValues(A, 1, &one, &row, &row, &diag);
            HYPRE_IJVectorSetValues(b, 1, &row, &exact);
         }
      }
   }

   /* ---- Nodal coordinate vectors ---- */
   for (HYPRE_Int lk = 0; lk < m->nlocal[2]; lk++)
   for (HYPRE_Int lj = 0; lj < m->nlocal[1]; lj++)
   for (HYPRE_Int li = 0; li < m->nlocal[0]; li++)
   {
      HYPRE_BigInt gi  = m->pstarts[0][m->coords[0]] + li;
      HYPRE_BigInt gj  = m->pstarts[1][m->coords[1]] + lj;
      HYPRE_BigInt gk  = m->pstarts[2][m->coords[2]] + lk;
      HYPRE_BigInt nid = node_gid(m, gi, gj, gk);
      HYPRE_Real   vx = gi * m->h[0], vy = gj * m->h[1], vz = gk * m->h[2];
      HYPRE_IJVectorSetValues(cx, 1, &nid, &vx);
      HYPRE_IJVectorSetValues(cy, 1, &nid, &vy);
      HYPRE_IJVectorSetValues(cz, 1, &nid, &vz);
   }

   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJMatrixAssemble(G);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorAssemble(cx);
   HYPRE_IJVectorAssemble(cy);
   HYPRE_IJVectorAssemble(cz);

   *A_ptr        = A;
   *b_ptr        = b;
   *G_ptr        = G;
   coord_ptr[0]  = cx;
   coord_ptr[1]  = cy;
   coord_ptr[2]  = cz;
   *xref_ptr     = xref;
   return 0;
}

/*--------------------------------------------------------------------------
 * main
 *--------------------------------------------------------------------------*/
int
main(int argc, char *argv[])
{
   MPI_Comm      comm = MPI_COMM_WORLD;
   int           myid, num_procs;
   MaxwellParams params;
   MaxwellMesh  *mesh = NULL;
   HYPREDRV_t    hypredrv;
   HYPRE_IJMatrix A, G;
   HYPRE_IJVector b, coord[3];
   HYPRE_Real    *xref = NULL;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   int parse = ParseArguments(argc, argv, &params, myid, num_procs);
   if (parse)
   {
      MPI_Finalize();
      return (parse == 2) ? 0 : 1;
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());

   /* Register an "ams" preconditioner preset for the default (no-YAML) path. */
   HYPREDRV_SAFE_CALL(
      HYPREDRV_PreconPresetRegister("ams", "ams", "AMS preconditioner defaults"));

   if (!myid && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

   if (params.yaml_file)
   {
      char *args[2] = {params.yaml_file, NULL};
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));
   }
   else
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetSolverPreset(hypredrv, params.solver_preset));
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(hypredrv, "ams"));
   }

   CreateMesh(comm, &params, &mesh);

   if (!myid && (params.verbose & 0x1))
   {
      printf("\nDefinite Maxwell (curl-curl + mass), Nedelec edge elements + AMS\n");
      printf("  grid (nodes) : %d x %d x %d\n", (int)params.N[0], (int)params.N[1],
             (int)params.N[2]);
      printf("  procs        : %d x %d x %d\n", (int)params.P[0], (int)params.P[1],
             (int)params.P[2]);
      printf("  global edges : %lld\n", (long long)mesh->num_edges_global);
      printf("  frequency    : kappa = %g * pi\n\n", params.freq);
   }

   BuildMaxwellSystem(mesh, &params, comm, &A, &b, &G, coord, &xref);

   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetDiscreteGradient(hypredrv, (HYPRE_Matrix)G));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetCoordinates(
      hypredrv, (HYPRE_Vector)coord[0], (HYPRE_Vector)coord[1], (HYPRE_Vector)coord[2]));

   for (int isolve = 0; isolve < params.nsolve; isolve++)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));
   }

   if (!myid && (params.verbose & 0x1)) HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));

   /* Discretization error: compare the computed edge DOFs to the exact
    * reference field (manufactured solution). */
   {
      /* Borrowed pointer into the solution vector data; do not free. */
      HYPRE_Complex *sol = NULL;
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol));

      HYPRE_Real loc[2] = {0.0, 0.0}, glob[2] = {0.0, 0.0};
      for (HYPRE_Int i = 0; i < mesh->num_edges_local; i++)
      {
         HYPRE_Real d = (HYPRE_Real)sol[i] - xref[i];
         loc[0] += d * d;
         loc[1] += xref[i] * xref[i];
      }
      MPI_Allreduce(loc, glob, 2, HYPRE_MPI_REAL, MPI_SUM, comm);
      HYPRE_Real rel = (glob[1] > 0.0) ? sqrt(glob[0] / glob[1]) : sqrt(glob[0]);
      if (!myid)
      {
         printf("\nDiscretization error (relative l2 over edge DOFs): %.6e\n", rel);
      }
   }

   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(G);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(coord[0]);
   HYPRE_IJVectorDestroy(coord[1]);
   HYPRE_IJVectorDestroy(coord[2]);
   free(xref);
   DestroyMesh(&mesh);

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();
   return 0;
}
