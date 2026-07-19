/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HYPREDRV.h"
#include "HYPREDRV_utils.h"

/*==========================================================================
 *   Definite grad-div (grad-div + mass) Example Driver -- ADS preconditioner
 *==========================================================================
 *
 *   PDE (definite H(div) problem):
 *   ------------------------------
 *      -alpha grad(div u) + beta u = f   in  Omega = [0,Lx]x[0,Ly]x[0,Lz]
 *      u . n = prescribed                on  the boundary
 *
 *   In this example alpha = beta = 1, so the operator reduces to grad-div +
 *   mass. The mass term (beta > 0) makes the problem definite, which is the
 *   ideal target for the Auxiliary-space Divergence Solver (ADS).
 *
 *   Discretization:
 *   ---------------
 *      Lowest order Raviart-Thomas ("face") elements (RT0) in H(div). Each face
 *      degree of freedom is the integral of the normal component of u through
 *      the (axis-aligned, +axis oriented) face.
 *
 *   Manufactured solution (benchmark):
 *   ----------------------------------
 *      u = (sin(k x), sin(k y), sin(k z)),  k = freq * pi.
 *      Then div u = k (cos kx + cos ky + cos kz) and grad(div u) = -k^2 u, so
 *      f = -grad(div u) + u = (1 + k^2) u. This gives an exact reference field
 *      to measure the discretization error and verify the solver converges to
 *      the right answer. The reference face DOFs have the closed form
 *         x-face: hy*hz * sin(k * x_node),
 *         y-face: hx*hz * sin(k * y_node),
 *         z-face: hx*hy * sin(k * z_node).
 *
 *   ADS operator inputs (set through HYPREDRV):
 *   -------------------------------------------
 *      - the discrete gradient G (edge x node incidence),
 *      - the discrete curl     C (face x edge incidence),
 *      - the vertex coordinate vectors (xcoord, ycoord, zcoord).
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Problem parameters
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int  N[3];
   HYPRE_Int  P[3];
   HYPRE_Real L[3];
   HYPRE_Real freq;
   HYPRE_Real alpha; /* grad-div coefficient */
   HYPRE_Real beta;  /* mass coefficient (beta >= 0 for ADS) */
   HYPRE_Int  nsolve;
   HYPRE_Int  verbose;
   char      *yaml_file;
   char      *solver_preset;
   char      *name;          /* optional object name (labels the statistics table) */
   char      *vtk_file;      /* optional VTK output base name (parallel: .pvti + .vti) */
   HYPRE_Int  hypredrv_argc; /* Number of hypredrive override args (incl. -a) */
   char     **hypredrv_argv; /* Hypredrive override args, starting at -a */
} GradDivParams;

/*--------------------------------------------------------------------------
 * Distributed structured mesh with rank-monotonic node/edge/face numbering.
 *--------------------------------------------------------------------------*/
typedef struct
{
   MPI_Comm      cart_comm;
   HYPRE_Int     mypid;
   HYPRE_Int     nprocs;
   HYPRE_Int     gdims[3];
   HYPRE_Int     pdims[3];
   HYPRE_Int     coords[3];
   HYPRE_Int     nlocal[3];
   HYPRE_BigInt *pstarts[3];
   HYPRE_Real    h[3];

   HYPRE_Int    num_nodes_local;
   HYPRE_Int    num_edges_local;
   HYPRE_Int    num_faces_local;
   HYPRE_BigInt node_ilower, node_iupper;
   HYPRE_BigInt edge_ilower, edge_iupper;
   HYPRE_BigInt face_ilower, face_iupper;
   HYPRE_BigInt num_faces_global;

   HYPRE_BigInt *rank_node_lower; /* size nprocs+1 */
   HYPRE_BigInt *rank_edge_lower;
   HYPRE_BigInt *rank_face_lower;
   HYPRE_Int (*rank_coords)[3];
} GradDivMesh;

enum
{
   DIR_X = 0,
   DIR_Y = 1,
   DIR_Z = 2
};

/* Find the processor-grid block index that owns global index g in dimension d. */
static inline HYPRE_Int
block_of(HYPRE_BigInt g, const HYPRE_BigInt *ps, HYPRE_Int P)
{
   HYPRE_Int b = 0;
   while ((b + 1) < P && g >= ps[b + 1]) b++;
   return b;
}

/* Owner MPI rank of node/entity anchored at node (gi,gj,gk). */
static int
owner_rank(const GradDivMesh *m, HYPRE_BigInt gi, HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   int owner;
   MPI_Cart_rank(m->cart_comm,
                 (int[]){(int)block_of(gi, m->pstarts[0], m->pdims[0]),
                         (int)block_of(gj, m->pstarts[1], m->pdims[1]),
                         (int)block_of(gk, m->pstarts[2], m->pdims[2])},
                 &owner);
   return owner;
}

/* Owner block dimensions (node counts nx,ny,nz) and per-dimension "caps"
 * bx,by,bz = node count minus one when the block touches the far boundary
 * (used for entities that have one fewer DOF per boundary-touching dimension). */
static void
block_dims(const GradDivMesh *m, HYPRE_Int px, HYPRE_Int py, HYPRE_Int pz, HYPRE_Int *nx,
           HYPRE_Int *ny, HYPRE_Int *nz, HYPRE_Int *bx, HYPRE_Int *by, HYPRE_Int *bz)
{
   *nx = (HYPRE_Int)(m->pstarts[0][px + 1] - m->pstarts[0][px]);
   *ny = (HYPRE_Int)(m->pstarts[1][py + 1] - m->pstarts[1][py]);
   *nz = (HYPRE_Int)(m->pstarts[2][pz + 1] - m->pstarts[2][pz]);
   *bx = (m->pstarts[0][px + 1] == m->gdims[0]) ? (*nx - 1) : *nx;
   *by = (m->pstarts[1][py + 1] == m->gdims[1]) ? (*ny - 1) : *ny;
   *bz = (m->pstarts[2][pz + 1] == m->gdims[2]) ? (*nz - 1) : *nz;
}

/* ---- Per-block entity counts ---- */
static HYPRE_Int
node_count(const GradDivMesh *m, HYPRE_Int px, HYPRE_Int py, HYPRE_Int pz)
{
   HYPRE_Int nx, ny, nz, bx, by, bz;
   block_dims(m, px, py, pz, &nx, &ny, &nz, &bx, &by, &bz);
   return nx * ny * nz;
}

/* Edges are 1D along their direction (capped in that direction). */
static HYPRE_Int
edge_count(const GradDivMesh *m, int dir, HYPRE_Int px, HYPRE_Int py, HYPRE_Int pz)
{
   HYPRE_Int nx, ny, nz, bx, by, bz;
   block_dims(m, px, py, pz, &nx, &ny, &nz, &bx, &by, &bz);
   if (dir == DIR_X) return bx * ny * nz;
   if (dir == DIR_Y) return nx * by * nz;
   return nx * ny * bz;
}

/* Faces are 2D in their tangential plane (capped in the two tangential dims). */
static HYPRE_Int
face_count(const GradDivMesh *m, int dir, HYPRE_Int px, HYPRE_Int py, HYPRE_Int pz)
{
   HYPRE_Int nx, ny, nz, bx, by, bz;
   block_dims(m, px, py, pz, &nx, &ny, &nz, &bx, &by, &bz);
   if (dir == DIR_X) return nx * by * bz;
   if (dir == DIR_Y) return bx * ny * bz;
   return bx * by * nz;
}

/* ---- Global ids (rank-monotonic; deterministic from pstarts on every rank) ---- */
static HYPRE_BigInt
node_gid(const GradDivMesh *m, HYPRE_BigInt gi, HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   int       o  = owner_rank(m, gi, gj, gk);
   HYPRE_Int px = block_of(gi, m->pstarts[0], m->pdims[0]);
   HYPRE_Int py = block_of(gj, m->pstarts[1], m->pdims[1]);
   HYPRE_Int pz = block_of(gk, m->pstarts[2], m->pdims[2]);
   HYPRE_Int nx, ny, nz, bx, by, bz;
   block_dims(m, px, py, pz, &nx, &ny, &nz, &bx, &by, &bz);
   HYPRE_Int li = (HYPRE_Int)(gi - m->pstarts[0][px]);
   HYPRE_Int lj = (HYPRE_Int)(gj - m->pstarts[1][py]);
   HYPRE_Int lk = (HYPRE_Int)(gk - m->pstarts[2][pz]);
   return m->rank_node_lower[o] + (HYPRE_BigInt)((lk * ny + lj) * nx + li);
}

/* Global edge id (lower node (gi,gj,gk), direction = edge tangent). */
static HYPRE_BigInt
edge_gid(const GradDivMesh *m, int dir, HYPRE_BigInt gi, HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   int       o  = owner_rank(m, gi, gj, gk);
   HYPRE_Int px = block_of(gi, m->pstarts[0], m->pdims[0]);
   HYPRE_Int py = block_of(gj, m->pstarts[1], m->pdims[1]);
   HYPRE_Int pz = block_of(gk, m->pstarts[2], m->pdims[2]);
   HYPRE_Int nx, ny, nz, bx, by, bz;
   block_dims(m, px, py, pz, &nx, &ny, &nz, &bx, &by, &bz);
   HYPRE_Int    li   = (HYPRE_Int)(gi - m->pstarts[0][px]);
   HYPRE_Int    lj   = (HYPRE_Int)(gj - m->pstarts[1][py]);
   HYPRE_Int    lk   = (HYPRE_Int)(gk - m->pstarts[2][pz]);
   HYPRE_Int    cntX = bx * ny * nz, cntY = nx * by * nz;
   HYPRE_BigInt local;
   if (dir == DIR_X) local = (HYPRE_BigInt)((lk * ny + lj) * bx + li);
   else if (dir == DIR_Y)
      local = (HYPRE_BigInt)cntX + (HYPRE_BigInt)((lk * by + lj) * nx + li);
   else local = (HYPRE_BigInt)(cntX + cntY) + (HYPRE_BigInt)((lk * ny + lj) * nx + li);
   return m->rank_edge_lower[o] + local;
}

/* Global face id (lower node (gi,gj,gk), direction = face normal). */
static HYPRE_BigInt
face_gid(const GradDivMesh *m, int dir, HYPRE_BigInt gi, HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   int       o  = owner_rank(m, gi, gj, gk);
   HYPRE_Int px = block_of(gi, m->pstarts[0], m->pdims[0]);
   HYPRE_Int py = block_of(gj, m->pstarts[1], m->pdims[1]);
   HYPRE_Int pz = block_of(gk, m->pstarts[2], m->pdims[2]);
   HYPRE_Int nx, ny, nz, bx, by, bz;
   block_dims(m, px, py, pz, &nx, &ny, &nz, &bx, &by, &bz);
   HYPRE_Int    li   = (HYPRE_Int)(gi - m->pstarts[0][px]);
   HYPRE_Int    lj   = (HYPRE_Int)(gj - m->pstarts[1][py]);
   HYPRE_Int    lk   = (HYPRE_Int)(gk - m->pstarts[2][pz]);
   HYPRE_Int    cntX = nx * by * bz, cntY = bx * ny * bz;
   HYPRE_BigInt local;
   if (dir == DIR_X) local = (HYPRE_BigInt)((lk * by + lj) * nx + li);
   else if (dir == DIR_Y)
      local = (HYPRE_BigInt)cntX + (HYPRE_BigInt)((lk * ny + lj) * bx + li);
   else local = (HYPRE_BigInt)(cntX + cntY) + (HYPRE_BigInt)((lk * by + lj) * bx + li);
   return m->rank_face_lower[o] + local;
}

/* Is the face (normal direction `dir`, lower node (gi,gj,gk)) on the boundary? */
static int
face_is_boundary(const GradDivMesh *m, int dir, HYPRE_BigInt gi, HYPRE_BigInt gj,
                 HYPRE_BigInt gk)
{
   if (dir == DIR_X) return (gi == 0 || gi == m->gdims[0] - 1);
   if (dir == DIR_Y) return (gj == 0 || gj == m->gdims[1] - 1);
   return (gk == 0 || gk == m->gdims[2] - 1);
}

/* Exact (manufactured) reference value of a face DOF: integral of the normal
 * component of u through the face. */
static HYPRE_Real
face_exact(const GradDivMesh *m, HYPRE_Real kappa, int dir, HYPRE_BigInt gi,
           HYPRE_BigInt gj, HYPRE_BigInt gk)
{
   HYPRE_Real x = gi * m->h[0], y = gj * m->h[1], z = gk * m->h[2];
   if (dir == DIR_X) return m->h[1] * m->h[2] * sin(kappa * x); /* u_x = sin(k x) */
   if (dir == DIR_Y) return m->h[0] * m->h[2] * sin(kappa * y); /* u_y = sin(k y) */
   return m->h[0] * m->h[1] * sin(kappa * z);                   /* u_z = sin(k z) */
}

/*--------------------------------------------------------------------------
 * Usage / argument parsing
 *--------------------------------------------------------------------------*/
static int
PrintUsage(void)
{
   printf("\nUsage: ${MPIEXEC_COMMAND} <np> ./graddiv [options]\n\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML configuration file (Opt.)\n");
   printf("  -a|--args ...     : Hypredrive YAML overrides, e.g. -a "
          "--solver:pcg:max_iter 100\n");
   printf("                      (requires -i; must come last)\n");
   printf("  --name <str>      : Object name (labels the statistics table) (Opt.)\n");
   printf("  -n <nx> <ny> <nz> : Global grid dimensions in nodes (17 17 17)\n");
   printf("  -P <Px> <Py> <Pz> : Processor grid dimensions (1 1 1)\n");
   printf("  -L <Lx> <Ly> <Lz> : Physical dimensions (1 1 1)\n");
   printf("  -freq <f>         : Manufactured frequency, kappa = f*pi (1.0)\n");
   printf("  -alpha <val>      : grad-div coefficient (1.0)\n");
   printf("  -beta <val>       : mass coefficient, >= 0 (1.0)\n");
   printf("  -ns|--nsolve <n>  : Number of solves (1)\n");
   printf("  -v|--verbose <n>  : Verbosity bitset (1)\n");
   printf(
      "  -vtk <base>       : Write the solution as VTK ImageData (cell-centered u +\n");
   printf("                      magnitude); serial -> <base>.vti, parallel -> "
          "<base>.pvti\n");
   printf("  -h|--help         : Print this message\n\n");
   return 0;
}

static int
ParseArguments(int argc, char *argv[], GradDivParams *params, int myid, int num_procs)
{
   for (int i = 0; i < 3; i++)
   {
      params->N[i] = 17;
      params->P[i] = 1;
      params->L[i] = 1.0;
   }
   params->freq          = 1.0;
   params->alpha         = 1.0;
   params->beta          = 1.0;
   params->nsolve        = 1;
   params->verbose       = 1;
   params->yaml_file     = NULL;
   params->solver_preset = "pcg";
   params->name          = NULL;
   params->vtk_file      = NULL;
   params->hypredrv_argc = 0;
   params->hypredrv_argv = NULL;

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
      else if (!strcmp(argv[i], "--name"))
      {
         if (++i < argc) params->name = argv[i];
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
      else if (!strcmp(argv[i], "-alpha"))
      {
         if (++i < argc) params->alpha = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-beta"))
      {
         if (++i < argc) params->beta = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-ns") || !strcmp(argv[i], "--nsolve"))
      {
         if (++i < argc) params->nsolve = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose"))
      {
         if (++i < argc) params->verbose = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-vtk") || !strcmp(argv[i], "--vtk"))
      {
         if (++i < argc) params->vtk_file = argv[i];
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
         printf("Error: number of processes (%d) must equal Px*Py*Pz (%d)\n", num_procs,
                params->P[0] * params->P[1] * params->P[2]);
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
   if (params->hypredrv_argc && !params->yaml_file)
   {
      if (!myid) printf("Error: -a/--args requires a YAML configuration file (-i)\n");
      return 1;
   }
   if (params->alpha <= 0.0)
   {
      if (!myid) printf("Error: alpha must be positive\n");
      return 1;
   }
   if (params->beta < 0.0)
   {
      if (!myid) printf("Error: beta must be >= 0 (required for ADS)\n");
      return 1;
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * Mesh creation: rank-monotonic node, edge, and face numbering.
 *--------------------------------------------------------------------------*/
static int
CreateMesh(MPI_Comm comm, GradDivParams *params, GradDivMesh **mesh_ptr)
{
   GradDivMesh *m = (GradDivMesh *)calloc(1, sizeof(GradDivMesh));
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
      HYPRE_Int size = m->gdims[d] / m->pdims[d];
      HYPRE_Int rest = m->gdims[d] - size * m->pdims[d];
      m->pstarts[d] =
         (HYPRE_BigInt *)calloc((size_t)(m->pdims[d] + 1), sizeof(HYPRE_BigInt));
      for (int j = 0; j < m->pdims[d] + 1; j++)
         m->pstarts[d][j] = (HYPRE_BigInt)(size * j + (j < rest ? j : rest));
      m->nlocal[d] =
         (HYPRE_Int)(m->pstarts[d][m->coords[d] + 1] - m->pstarts[d][m->coords[d]]);
   }

   m->rank_coords     = malloc((size_t)m->nprocs * sizeof(*m->rank_coords));
   m->rank_node_lower = malloc((size_t)(m->nprocs + 1) * sizeof(HYPRE_BigInt));
   m->rank_edge_lower = malloc((size_t)(m->nprocs + 1) * sizeof(HYPRE_BigInt));
   m->rank_face_lower = malloc((size_t)(m->nprocs + 1) * sizeof(HYPRE_BigInt));
   for (int r = 0; r < m->nprocs; r++)
   {
      int rc[3];
      MPI_Cart_coords(m->cart_comm, r, 3, rc);
      m->rank_coords[r][0] = rc[0];
      m->rank_coords[r][1] = rc[1];
      m->rank_coords[r][2] = rc[2];
   }
   m->rank_node_lower[0] = m->rank_edge_lower[0] = m->rank_face_lower[0] = 0;
   for (int r = 0; r < m->nprocs; r++)
   {
      HYPRE_Int px = m->rank_coords[r][0], py = m->rank_coords[r][1],
                pz              = m->rank_coords[r][2];
      m->rank_node_lower[r + 1] = m->rank_node_lower[r] + node_count(m, px, py, pz);
      m->rank_edge_lower[r + 1] =
         m->rank_edge_lower[r] + edge_count(m, DIR_X, px, py, pz) +
         edge_count(m, DIR_Y, px, py, pz) + edge_count(m, DIR_Z, px, py, pz);
      m->rank_face_lower[r + 1] =
         m->rank_face_lower[r] + face_count(m, DIR_X, px, py, pz) +
         face_count(m, DIR_Y, px, py, pz) + face_count(m, DIR_Z, px, py, pz);
   }
   m->num_nodes_local = m->nlocal[0] * m->nlocal[1] * m->nlocal[2];
   m->num_edges_local =
      (HYPRE_Int)(m->rank_edge_lower[myid + 1] - m->rank_edge_lower[myid]);
   m->num_faces_local =
      (HYPRE_Int)(m->rank_face_lower[myid + 1] - m->rank_face_lower[myid]);
   m->node_ilower      = m->rank_node_lower[myid];
   m->node_iupper      = m->node_ilower + m->num_nodes_local - 1;
   m->edge_ilower      = m->rank_edge_lower[myid];
   m->edge_iupper      = m->edge_ilower + m->num_edges_local - 1;
   m->face_ilower      = m->rank_face_lower[myid];
   m->face_iupper      = m->face_ilower + m->num_faces_local - 1;
   m->num_faces_global = m->rank_face_lower[m->nprocs];

   *mesh_ptr = m;
   return 0;
}

static void
DestroyMesh(GradDivMesh **mesh_ptr)
{
   GradDivMesh *m = *mesh_ptr;
   if (!m) return;
   for (int d = 0; d < 3; d++) free(m->pstarts[d]);
   free(m->rank_coords);
   free(m->rank_node_lower);
   free(m->rank_edge_lower);
   free(m->rank_face_lower);
   MPI_Comm_free(&m->cart_comm);
   free(m);
   *mesh_ptr = NULL;
}

/*--------------------------------------------------------------------------
 * Lowest-order Raviart-Thomas (RT0) face basis on a brick [0,hx]x[0,hy]x[0,hz].
 *
 * Local face ordering (all DOFs oriented along +axis normal):
 *   0: lo-x face (normal x)   1: hi-x face
 *   2: lo-y face (normal y)   3: hi-y face
 *   4: lo-z face (normal z)   5: hi-z face
 *--------------------------------------------------------------------------*/
static const int face_dir6[6] = {DIR_X, DIR_X, DIR_Y, DIR_Y, DIR_Z, DIR_Z};
/* lower-node offset of each local face within a cell */
static const int face_off6[6][3] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0},
                                    {0, 1, 0}, {0, 0, 0}, {0, 0, 1}};

static void
rt0_basis(int a, HYPRE_Real lx, HYPRE_Real ly, HYPRE_Real lz, const HYPRE_Real h[3],
          HYPRE_Real v[3], HYPRE_Real *div)
{
   HYPRE_Real V = h[0] * h[1] * h[2];
   v[0] = v[1] = v[2] = 0.0;
   *div               = 0.0;
   switch (a)
   {
      case 0:
         v[0] = (1.0 - lx / h[0]) / (h[1] * h[2]);
         *div = -1.0 / V;
         break;
      case 1:
         v[0] = (lx / h[0]) / (h[1] * h[2]);
         *div = 1.0 / V;
         break;
      case 2:
         v[1] = (1.0 - ly / h[1]) / (h[0] * h[2]);
         *div = -1.0 / V;
         break;
      case 3:
         v[1] = (ly / h[1]) / (h[0] * h[2]);
         *div = 1.0 / V;
         break;
      case 4:
         v[2] = (1.0 - lz / h[2]) / (h[0] * h[1]);
         *div = -1.0 / V;
         break;
      case 5:
         v[2] = (lz / h[2]) / (h[0] * h[1]);
         *div = 1.0 / V;
         break;
   }
}

/* 3-point Gauss rule on [0,1]. */
static const HYPRE_Real g3_x[3] = {0.1127016653792583, 0.5, 0.8872983346207417};
static const HYPRE_Real g3_w[3] = {0.2777777777777778, 0.4444444444444444,
                                   0.2777777777777778};

/* Precompute the (constant) 6x6 RT0 element matrix S = alpha*(grad-div) + beta*(mass). */
static void
ComputeElementMatrix(const HYPRE_Real h[3], HYPRE_Real alpha, HYPRE_Real beta,
                     HYPRE_Real S[6][6])
{
   HYPRE_Real detJ = h[0] * h[1] * h[2];
   for (int a = 0; a < 6; a++)
      for (int b = 0; b < 6; b++) S[a][b] = 0.0;

   for (int qz = 0; qz < 3; qz++)
      for (int qy = 0; qy < 3; qy++)
         for (int qx = 0; qx < 3; qx++)
         {
            HYPRE_Real lx = g3_x[qx] * h[0], ly = g3_x[qy] * h[1], lz = g3_x[qz] * h[2];
            HYPRE_Real w = g3_w[qx] * g3_w[qy] * g3_w[qz] * detJ;
            HYPRE_Real v[6][3], dv[6];
            for (int a = 0; a < 6; a++) rt0_basis(a, lx, ly, lz, h, v[a], &dv[a]);
            for (int a = 0; a < 6; a++)
               for (int b = 0; b < 6; b++)
               {
                  HYPRE_Real mass =
                     v[a][0] * v[b][0] + v[a][1] * v[b][1] + v[a][2] * v[b][2];
                  S[a][b] += w * (alpha * dv[a] * dv[b] + beta * mass);
               }
         }
}

/*--------------------------------------------------------------------------
 * Build the grad-div system: A (face), b, discrete gradient G (edge x node),
 * discrete curl C (face x edge), vertex coordinates, and reference face DOFs.
 *--------------------------------------------------------------------------*/
static int
BuildGradDivSystem(GradDivMesh *m, GradDivParams *params, MPI_Comm comm,
                   HYPRE_IJMatrix *A_ptr, HYPRE_IJVector *b_ptr, HYPRE_IJMatrix *G_ptr,
                   HYPRE_IJMatrix *C_ptr, HYPRE_IJVector coord_ptr[3],
                   HYPRE_Real **xref_ptr)
{
   const HYPRE_Real kappa = params->freq * M_PI;
   /* grad(div u) = -kappa^2 u, so f = alpha*kappa^2*u + beta*u = (alpha*kappa^2 + beta) u
    */
   const HYPRE_Real fcoef = params->alpha * kappa * kappa + params->beta;
   const HYPRE_Int  Nx = m->gdims[0], Ny = m->gdims[1], Nz = m->gdims[2];

   HYPRE_IJMatrix A, G, C;
   HYPRE_IJVector b, cx, cy, cz;

   HYPRE_IJMatrixCreate(comm, m->face_ilower, m->face_iupper, m->face_ilower,
                        m->face_iupper, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);
   HYPRE_IJMatrixCreate(comm, m->edge_ilower, m->edge_iupper, m->node_ilower,
                        m->node_iupper, &G);
   HYPRE_IJMatrixSetObjectType(G, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(G);
   HYPRE_IJMatrixCreate(comm, m->face_ilower, m->face_iupper, m->edge_ilower,
                        m->edge_iupper, &C);
   HYPRE_IJMatrixSetObjectType(C, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(C);
   HYPRE_IJVectorCreate(comm, m->face_ilower, m->face_iupper, &b);
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

   HYPRE_Real *xref = (HYPRE_Real *)calloc(
      (size_t)(m->num_faces_local > 0 ? m->num_faces_local : 1), sizeof(HYPRE_Real));
   HYPRE_Real S[6][6];
   ComputeElementMatrix(m->h, params->alpha, params->beta, S);

   /* ---- Assemble owned interior face rows of A and load b (ghost-cell loop) ---- */
   HYPRE_Int cx_lo =
      (m->pstarts[0][m->coords[0]] > 0) ? (HYPRE_Int)m->pstarts[0][m->coords[0]] - 1 : 0;
   HYPRE_Int cy_lo =
      (m->pstarts[1][m->coords[1]] > 0) ? (HYPRE_Int)m->pstarts[1][m->coords[1]] - 1 : 0;
   HYPRE_Int cz_lo =
      (m->pstarts[2][m->coords[2]] > 0) ? (HYPRE_Int)m->pstarts[2][m->coords[2]] - 1 : 0;
   HYPRE_Int cx_hi =
      (HYPRE_Int)((m->pstarts[0][m->coords[0] + 1] < Nx) ? m->pstarts[0][m->coords[0] + 1]
                                                         : Nx - 1);
   HYPRE_Int cy_hi =
      (HYPRE_Int)((m->pstarts[1][m->coords[1] + 1] < Ny) ? m->pstarts[1][m->coords[1] + 1]
                                                         : Ny - 1);
   HYPRE_Int cz_hi =
      (HYPRE_Int)((m->pstarts[2][m->coords[2] + 1] < Nz) ? m->pstarts[2][m->coords[2] + 1]
                                                         : Nz - 1);

   for (HYPRE_Int cz = cz_lo; cz < cz_hi; cz++)
      for (HYPRE_Int cy = cy_lo; cy < cy_hi; cy++)
         for (HYPRE_Int cx = cx_lo; cx < cx_hi; cx++)
         {
            HYPRE_BigInt fgid[6];
            int          fbnd[6];
            HYPRE_Real   fval[6], F[6];
            for (int a = 0; a < 6; a++)
            {
               HYPRE_BigInt gi = cx + face_off6[a][0], gj = cy + face_off6[a][1],
                            gk = cz + face_off6[a][2];
               fgid[a]         = face_gid(m, face_dir6[a], gi, gj, gk);
               fbnd[a]         = face_is_boundary(m, face_dir6[a], gi, gj, gk);
               fval[a]         = face_exact(m, kappa, face_dir6[a], gi, gj, gk);
               F[a]            = 0.0;
            }

            for (int qz = 0; qz < 3; qz++)
               for (int qy = 0; qy < 3; qy++)
                  for (int qx = 0; qx < 3; qx++)
                  {
                     HYPRE_Real lx = g3_x[qx] * m->h[0], ly = g3_x[qy] * m->h[1],
                                lz = g3_x[qz] * m->h[2];
                     HYPRE_Real px = cx * m->h[0] + lx, py = cy * m->h[1] + ly,
                                pz = cz * m->h[2] + lz;
                     HYPRE_Real wq =
                        g3_w[qx] * g3_w[qy] * g3_w[qz] * (m->h[0] * m->h[1] * m->h[2]);
                     HYPRE_Real fx = fcoef * sin(kappa * px); /* f = (1+k^2) u */
                     HYPRE_Real fy = fcoef * sin(kappa * py);
                     HYPRE_Real fz = fcoef * sin(kappa * pz);
                     for (int a = 0; a < 6; a++)
                     {
                        HYPRE_Real v[3], dv;
                        rt0_basis(a, lx, ly, lz, m->h, v, &dv);
                        F[a] += wq * (fx * v[0] + fy * v[1] + fz * v[2]);
                     }
                  }

            for (int a = 0; a < 6; a++)
            {
               if (fbnd[a]) continue;
               if (fgid[a] < m->face_ilower || fgid[a] > m->face_iupper)
                  continue; /* owned rows only */

               HYPRE_BigInt row   = fgid[a];
               HYPRE_Int    ncols = 0;
               HYPRE_BigInt cols[6];
               HYPRE_Real   vals[6];
               HYPRE_Real   rhs = F[a];
               for (int bcol = 0; bcol < 6; bcol++)
               {
                  if (fbnd[bcol])
                  {
                     rhs -= S[a][bcol] * fval[bcol];
                  }
                  else
                  {
                     cols[ncols] = fgid[bcol];
                     vals[ncols] = S[a][bcol];
                     ncols++;
                  }
               }
               HYPRE_IJMatrixAddToValues(A, 1, &ncols, &row, cols, vals);
               HYPRE_IJVectorAddToValues(b, 1, &row, &rhs);
            }
         }

   /* ---- Owned-face pass: boundary identity rows, discrete-curl rows, xref ----
    * C[face][edge] is the signed incidence of the 4 boundary edges of the face,
    * oriented by the right-hand rule about the +axis face normal so that the
    * fundamental relation curl-of-grad (C G = 0) holds exactly. */
   for (HYPRE_Int lk = 0; lk < m->nlocal[2]; lk++)
      for (HYPRE_Int lj = 0; lj < m->nlocal[1]; lj++)
         for (HYPRE_Int li = 0; li < m->nlocal[0]; li++)
         {
            HYPRE_BigInt gi = m->pstarts[0][m->coords[0]] + li;
            HYPRE_BigInt gj = m->pstarts[1][m->coords[1]] + lj;
            HYPRE_BigInt gk = m->pstarts[2][m->coords[2]] + lk;

            for (int dir = 0; dir < 3; dir++)
            {
               /* A face's lower node may sit on the far boundary plane of its normal
                * direction; it is invalid only when a tangential index hits N-1. */
               if (dir == DIR_X && (gj >= Ny - 1 || gk >= Nz - 1)) continue;
               if (dir == DIR_Y && (gi >= Nx - 1 || gk >= Nz - 1)) continue;
               if (dir == DIR_Z && (gi >= Nx - 1 || gj >= Ny - 1)) continue;

               HYPRE_BigInt row = face_gid(m, dir, gi, gj, gk);
               if (row < m->face_ilower || row > m->face_iupper) continue;

               xref[row - m->face_ilower] = face_exact(m, kappa, dir, gi, gj, gk);

               /* Four boundary edges of the face with right-hand-rule signs. */
               HYPRE_BigInt ec[4];
               HYPRE_Real   es[4];
               if (dir == DIR_X)
               {
                  ec[0] = edge_gid(m, DIR_Y, gi, gj, gk);
                  es[0] = 1.0;
                  ec[1] = edge_gid(m, DIR_Z, gi, gj + 1, gk);
                  es[1] = 1.0;
                  ec[2] = edge_gid(m, DIR_Y, gi, gj, gk + 1);
                  es[2] = -1.0;
                  ec[3] = edge_gid(m, DIR_Z, gi, gj, gk);
                  es[3] = -1.0;
               }
               else if (dir == DIR_Y)
               {
                  ec[0] = edge_gid(m, DIR_Z, gi, gj, gk);
                  es[0] = 1.0;
                  ec[1] = edge_gid(m, DIR_X, gi, gj, gk + 1);
                  es[1] = 1.0;
                  ec[2] = edge_gid(m, DIR_Z, gi + 1, gj, gk);
                  es[2] = -1.0;
                  ec[3] = edge_gid(m, DIR_X, gi, gj, gk);
                  es[3] = -1.0;
               }
               else
               {
                  ec[0] = edge_gid(m, DIR_X, gi, gj, gk);
                  es[0] = 1.0;
                  ec[1] = edge_gid(m, DIR_Y, gi + 1, gj, gk);
                  es[1] = 1.0;
                  ec[2] = edge_gid(m, DIR_X, gi, gj + 1, gk);
                  es[2] = -1.0;
                  ec[3] = edge_gid(m, DIR_Y, gi, gj, gk);
                  es[3] = -1.0;
               }
               HYPRE_Int cnc = 4;
               HYPRE_IJMatrixSetValues(C, 1, &cnc, &row, ec, es);

               if (face_is_boundary(m, dir, gi, gj, gk))
               {
                  HYPRE_Int  one  = 1;
                  HYPRE_Real diag = 1.0, val = xref[row - m->face_ilower];
                  HYPRE_IJMatrixSetValues(A, 1, &one, &row, &row, &diag);
                  HYPRE_IJVectorSetValues(b, 1, &row, &val);
               }
            }
         }

   /* ---- Owned-edge pass: discrete gradient G (-1 lower node, +1 upper node) ---- */
   for (HYPRE_Int lk = 0; lk < m->nlocal[2]; lk++)
      for (HYPRE_Int lj = 0; lj < m->nlocal[1]; lj++)
         for (HYPRE_Int li = 0; li < m->nlocal[0]; li++)
         {
            HYPRE_BigInt gi = m->pstarts[0][m->coords[0]] + li;
            HYPRE_BigInt gj = m->pstarts[1][m->coords[1]] + lj;
            HYPRE_BigInt gk = m->pstarts[2][m->coords[2]] + lk;
            for (int dir = 0; dir < 3; dir++)
            {
               if (dir == DIR_X && gi >= Nx - 1) continue;
               if (dir == DIR_Y && gj >= Ny - 1) continue;
               if (dir == DIR_Z && gk >= Nz - 1) continue;
               HYPRE_BigInt row = edge_gid(m, dir, gi, gj, gk);
               if (row < m->edge_ilower || row > m->edge_iupper) continue;
               HYPRE_BigInt up[3] = {gi, gj, gk};
               up[dir] += 1;
               HYPRE_BigInt gcols[2] = {node_gid(m, gi, gj, gk),
                                        node_gid(m, up[0], up[1], up[2])};
               HYPRE_Real   gvals[2] = {-1.0, 1.0};
               HYPRE_Int    gnc      = 2;
               HYPRE_IJMatrixSetValues(G, 1, &gnc, &row, gcols, gvals);
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
   HYPRE_IJMatrixAssemble(C);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorAssemble(cx);
   HYPRE_IJVectorAssemble(cy);
   HYPRE_IJVectorAssemble(cz);

   *A_ptr       = A;
   *b_ptr       = b;
   *G_ptr       = G;
   *C_ptr       = C;
   coord_ptr[0] = cx;
   coord_ptr[1] = cy;
   coord_ptr[2] = cz;
   *xref_ptr    = xref;
   return 0;
}

/*--------------------------------------------------------------------------
 * VTK output: reconstruct the cell-centered field u (and its magnitude) from
 * the face DOFs and write a parallel VTK ImageData dataset (one .vti piece per
 * rank plus a .pvti master). The full face solution is gathered to every rank
 * (rank-contiguous numbering), after which each rank reconstructs its own cell
 * block locally.
 *--------------------------------------------------------------------------*/
static void
vtk_cell_extent(const GradDivMesh *m, HYPRE_Int px, HYPRE_Int py, HYPRE_Int pz,
                HYPRE_Int ext[6])
{
   HYPRE_BigInt x1 = m->pstarts[0][px + 1], y1 = m->pstarts[1][py + 1],
                z1 = m->pstarts[2][pz + 1];
   ext[0]          = (HYPRE_Int)m->pstarts[0][px];
   ext[1]          = (HYPRE_Int)(x1 < m->gdims[0] ? x1 : m->gdims[0] - 1);
   ext[2]          = (HYPRE_Int)m->pstarts[1][py];
   ext[3]          = (HYPRE_Int)(y1 < m->gdims[1] ? y1 : m->gdims[1] - 1);
   ext[4]          = (HYPRE_Int)m->pstarts[2][pz];
   ext[5]          = (HYPRE_Int)(z1 < m->gdims[2] ? z1 : m->gdims[2] - 1);
}

static void
vtk_write_piece(const char *fname, const HYPRE_Int ext[6], const HYPRE_Real sp[3],
                const double *uvec, const double *umag, size_t ncells)
{
   FILE *fp = fopen(fname, "wb");
   if (!fp) return;
   uint64_t ub = (uint64_t)ncells * 3 * sizeof(double);
   uint64_t mb = (uint64_t)ncells * sizeof(double);
   uint64_t uo = 0, mo = uo + sizeof(uint64_t) + ub;
   fprintf(fp, "<?xml version=\"1.0\"?>\n");
   fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" "
               "header_type=\"UInt64\">\n");
   fprintf(fp,
           "  <ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" "
           "Spacing=\"%.17g %.17g %.17g\">\n",
           ext[0], ext[1], ext[2], ext[3], ext[4], ext[5], sp[0], sp[1], sp[2]);
   fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", ext[0], ext[1], ext[2],
           ext[3], ext[4], ext[5]);
   fprintf(fp, "      <CellData Vectors=\"solution\" Scalars=\"magnitude\">\n");
   fprintf(fp,
           "        <DataArray type=\"Float64\" Name=\"solution\" "
           "NumberOfComponents=\"3\" format=\"appended\" offset=\"%llu\"/>\n",
           (unsigned long long)uo);
   fprintf(fp,
           "        <DataArray type=\"Float64\" Name=\"magnitude\" format=\"appended\" "
           "offset=\"%llu\"/>\n",
           (unsigned long long)mo);
   fprintf(fp, "      </CellData>\n      <PointData></PointData>\n    </Piece>\n"
               "  </ImageData>\n  <AppendedData encoding=\"raw\">\n   _");
   fwrite(&ub, sizeof(uint64_t), 1, fp);
   fwrite(uvec, 1, (size_t)ub, fp);
   fwrite(&mb, sizeof(uint64_t), 1, fp);
   fwrite(umag, 1, (size_t)mb, fp);
   fprintf(fp, "\n  </AppendedData>\n</VTKFile>\n");
   fclose(fp);
}

static void
vtk_write_master(const char *fname, const char *piece_base, const GradDivMesh *m,
                 const HYPRE_Real sp[3])
{
   FILE *fp = fopen(fname, "w");
   if (!fp) return;
   HYPRE_Int whole[6] = {0, m->gdims[0] - 1, 0, m->gdims[1] - 1, 0, m->gdims[2] - 1};
   fprintf(fp, "<?xml version=\"1.0\"?>\n<VTKFile type=\"PImageData\" version=\"1.0\" "
               "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
   fprintf(fp,
           "  <PImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" "
           "Spacing=\"%.17g %.17g %.17g\" GhostLevel=\"0\">\n",
           whole[0], whole[1], whole[2], whole[3], whole[4], whole[5], sp[0], sp[1],
           sp[2]);
   fprintf(
      fp,
      "    <PCellData Vectors=\"solution\" Scalars=\"magnitude\">\n"
      "      <PDataArray type=\"Float64\" Name=\"solution\" NumberOfComponents=\"3\"/>\n"
      "      <PDataArray type=\"Float64\" Name=\"magnitude\"/>\n    </PCellData>\n");
   for (int r = 0; r < m->nprocs; r++)
   {
      HYPRE_Int ext[6];
      vtk_cell_extent(m, m->rank_coords[r][0], m->rank_coords[r][1], m->rank_coords[r][2],
                      ext);
      fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s_p%d.vti\"/>\n",
              ext[0], ext[1], ext[2], ext[3], ext[4], ext[5], piece_base, r);
   }
   fprintf(fp, "  </PImageData>\n</VTKFile>\n");
   fclose(fp);
}

static void
WriteGradDivVTK(const GradDivMesh *m, const char *base, const HYPRE_Complex *sol_local)
{
   int *counts = (int *)malloc((size_t)m->nprocs * sizeof(int));
   int *displs = (int *)malloc((size_t)m->nprocs * sizeof(int));
   for (int r = 0; r < m->nprocs; r++)
   {
      counts[r] = (int)(m->rank_face_lower[r + 1] - m->rank_face_lower[r]);
      displs[r] = (int)m->rank_face_lower[r];
   }
   double *gsol = (double *)malloc((size_t)m->num_faces_global * sizeof(double));
   MPI_Allgatherv((const void *)sol_local, m->num_faces_local, HYPRE_MPI_REAL, gsol,
                  counts, displs, HYPRE_MPI_REAL, m->cart_comm);

   HYPRE_Int ext[6];
   vtk_cell_extent(m, m->coords[0], m->coords[1], m->coords[2], ext);
   HYPRE_Int  ncx = ext[1] - ext[0], ncy = ext[3] - ext[2], ncz = ext[5] - ext[4];
   size_t     ncells = (size_t)ncx * (size_t)ncy * (size_t)ncz;
   double    *uvec   = (double *)malloc((ncells ? ncells : 1) * 3 * sizeof(double));
   double    *umag   = (double *)malloc((ncells ? ncells : 1) * sizeof(double));
   HYPRE_Real cc[3]  = {0.5 * m->h[0], 0.5 * m->h[1], 0.5 * m->h[2]};

   size_t idx = 0;
   for (HYPRE_Int kc = ext[4]; kc < ext[5]; kc++)
      for (HYPRE_Int jc = ext[2]; jc < ext[3]; jc++)
         for (HYPRE_Int ic = ext[0]; ic < ext[1]; ic++)
         {
            HYPRE_Real U[3] = {0.0, 0.0, 0.0};
            for (int a = 0; a < 6; a++)
            {
               HYPRE_BigInt gid = face_gid(m, face_dir6[a], ic + face_off6[a][0],
                                           jc + face_off6[a][1], kc + face_off6[a][2]);
               HYPRE_Real   v[3], div;
               rt0_basis(a, cc[0], cc[1], cc[2], m->h, v, &div);
               U[0] += gsol[gid] * v[0];
               U[1] += gsol[gid] * v[1];
               U[2] += gsol[gid] * v[2];
            }
            uvec[3 * idx + 0] = U[0];
            uvec[3 * idx + 1] = U[1];
            uvec[3 * idx + 2] = U[2];
            umag[idx]         = sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2]);
            idx++;
         }

   char        fname[512];
   const char *bn = strrchr(base, '/');
   bn             = bn ? bn + 1 : base;
   if (m->nprocs == 1)
   {
      snprintf(fname, sizeof(fname), "%s.vti", base);
   }
   else
   {
      snprintf(fname, sizeof(fname), "%s_p%d.vti", base, m->mypid);
   }
   vtk_write_piece(fname, ext, m->h, uvec, umag, ncells);
   if (m->mypid == 0 && m->nprocs > 1)
   {
      snprintf(fname, sizeof(fname), "%s.pvti", base);
      vtk_write_master(fname, bn, m, m->h);
   }

   free(uvec);
   free(umag);
   free(gsol);
   free(counts);
   free(displs);
}

/*--------------------------------------------------------------------------
 * main
 *--------------------------------------------------------------------------*/
int
main(int argc, char *argv[])
{
   MPI_Comm       comm = MPI_COMM_WORLD;
   int            myid, num_procs;
   GradDivParams  params;
   GradDivMesh   *mesh = NULL;
   HYPREDRV_t     hypredrv;
   HYPRE_IJMatrix A, G, C;
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

   /* Register an "ads" preconditioner preset for the default (no-YAML) path. */
   HYPREDRV_SAFE_CALL(
      HYPREDRV_PreconPresetRegister("ads", "ads", "ADS preconditioner defaults"));

   if (!myid && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

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
      HYPREDRV_SAFE_CALL(
         HYPREDRV_InputArgsSetSolverPreset(hypredrv, params.solver_preset));
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(hypredrv, "ads"));
   }

   /* Name the object after parsing (input parsing re-initializes the stats). */
   if (params.name)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_ObjectSetName(hypredrv, params.name));
   }

   CreateMesh(comm, &params, &mesh);

   if (!myid && (params.verbose & 0x1))
   {
      printf("\nDefinite grad-div (grad-div + mass), RT0 face elements + ADS\n");
      printf("  grid (nodes) : %d x %d x %d\n", (int)params.N[0], (int)params.N[1],
             (int)params.N[2]);
      printf("  procs        : %d x %d x %d\n", (int)params.P[0], (int)params.P[1],
             (int)params.P[2]);
      printf("  global faces : %lld\n", (long long)mesh->num_faces_global);
      printf("  frequency    : kappa = %g * pi\n", params.freq);
      printf("  coefficients : alpha = %g, beta = %g\n\n", params.alpha, params.beta);
   }

   BuildGradDivSystem(mesh, &params, comm, &A, &b, &G, &C, coord, &xref);

   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(
      HYPREDRV_LinearSystemSetDiscreteGradient(hypredrv, (HYPRE_Matrix)G));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetDiscreteCurl(hypredrv, (HYPRE_Matrix)C));
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

   {
      /* Borrowed pointer into the solution vector data; do not free. */
      HYPRE_Complex *sol = NULL;
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol));

      HYPRE_Real loc[2] = {0.0, 0.0}, glob[2] = {0.0, 0.0};
      for (HYPRE_Int i = 0; i < mesh->num_faces_local; i++)
      {
         HYPRE_Real d = (HYPRE_Real)sol[i] - xref[i];
         loc[0] += d * d;
         loc[1] += xref[i] * xref[i];
      }
      MPI_Allreduce(loc, glob, 2, HYPRE_MPI_REAL, MPI_SUM, comm);
      HYPRE_Real rel = (glob[1] > 0.0) ? sqrt(glob[0] / glob[1]) : sqrt(glob[0]);
      if (!myid)
         printf("\nDiscretization error (relative l2 over face DOFs): %.6e\n", rel);

      if (params.vtk_file)
      {
         WriteGradDivVTK(mesh, params.vtk_file, sol);
         if (!myid)
            printf("Wrote VTK solution to %s%s\n", params.vtk_file,
                   num_procs > 1 ? ".pvti" : ".vti");
      }
   }

   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(G);
   HYPRE_IJMatrixDestroy(C);
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
