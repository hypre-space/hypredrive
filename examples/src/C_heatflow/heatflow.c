#define _GNU_SOURCE
/*==========================================================================
 *   3D Transient Nonlinear Heat Conduction (Q1 Hexahedra) Example Driver
 *==========================================================================
 *
 *   Grid Partitioning:
 *   ------------------
 *                     P[1]
 *                      ^      Processor Grid: P[0] x P[1] x P[2]
 *                      |   /  Each proc owns a n[0] x n[1] x n[2] block
 *                      |  /
 *                      | /
 *         P[2] <-------C
 *                     /
 *                    /
 *                   /
 *                  v P[0]
 *
 *   Geometry and Axes:
 *   ------------------
 *   Domain: [0, Lx] x [0, Ly] x [0, Lz]
 *     - x: horizontal axis
 *     - y: vertical axis (cold base at y = 0)
 *     - z: depth axis
 *
 *   Physics (transient nonlinear heat conduction):
 *   ----------------------------------------------
 *     - PDE:  rho*cp * dT/dt - div(k(T) * grad(T)) = Q  in Omega
 *     - Constitutive: k(T) = k0 * exp(beta * T)
 *       - beta = 0: linear conductivity (k = k0)
 *       - beta > 0: conductivity increases with temperature
 *       - beta < 0: conductivity decreases with temperature
 *
 *   Boundary Conditions:
 *   --------------------
 *     - Dirichlet: T = 0 on y = 0 plane (cold base)
 *     - Neumann (insulated): dT/dn = 0 on all other faces
 *       (x = 0, x = Lx, y = Ly, z = 0, z = Lz)
 *
 *   Discretization:
 *   ---------------
 *     - Q1 hexahedra, 1 DOF per node (temperature T)
 *     - Time integration: Backward Euler (implicit)
 *     - Nonlinear solver: Full Newton with line search
 *
 *   Newton Linearization:
 *   ---------------------
 *     J[dT] = rho*cp/dt * M + K(T) + J_extra
 *     where:
 *       M[a][b]       = integral( phi_a * phi_b )
 *       K(T)[a][b]    = integral( k(T) * grad(phi_a) . grad(phi_b) )
 *       J_extra[a][b] = integral( k'(T) * phi_b * grad(phi_a) . grad(T) )
 *
 *   Validation:
 *   -----------
 *     MMS (Method of Manufactured Solutions) with non-negative 3D exact solution:
 *       T(x,y,z,t) = exp(-t) * [(1+cos(2*pi*x/Lx))/2] * sin(pi*y/(2*Ly))
 *                            * [(1+cos(2*pi*z/Lz))/2]
 *     which satisfies:
 *       - T >= 0 everywhere (physically meaningful temperature)
 *       - T = 0 at y = 0 (Dirichlet, since sin(0) = 0)
 *       - dT/dx = 0 at x = 0, Lx (Neumann, since sin(0) = sin(2*pi) = 0)
 *       - dT/dy = 0 at y = Ly (Neumann, since cos(pi/2) = 0)
 *       - dT/dz = 0 at z = 0, Lz (Neumann, since sin(0) = sin(2*pi) = 0)
 *
 *   Notes:
 *   ------
 *     - Dirichlet rows are imposed as identity with zero RHS
 *     - For beta != 0, the Jacobian is non-symmetric due to J_extra term
 *==========================================================================*/

#include <errno.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "HYPREDRV.h"

typedef struct
{
   HYPRE_Int  max_iters;
   HYPRE_Real tol;  /* absolute update tolerance on ||Δ||_inf */
   HYPRE_Real rtol; /* residual tolerance on ||R||_2 */
   HYPRE_Real damping;
   HYPRE_Real ls_min; /* minimum damping */
   HYPRE_Real ls_max; /* maximum damping */
} NewtonParams;

typedef struct
{
   HYPRE_Int    visualize;
   HYPRE_Int    verbose;
   HYPRE_Int    N[3];
   HYPRE_Int    P[3];
   HYPRE_Real   L[3];
   HYPRE_Real   alpha0; /* thermal diffusivity using k0: k0/(rho*cp) */
   HYPRE_Real   hmin;   /* minimum grid spacing */
   HYPRE_Real   rho;    /* density */
   HYPRE_Real   cp;     /* heat capacity */
   HYPRE_Real   dt;     /* time step */
   HYPRE_Real   tf;     /* final time */
   HYPRE_Real   k0;     /* base conductivity */
   HYPRE_Real   beta;   /* k(T) = k0 * exp(beta*T) */
   HYPRE_Int    max_rows_per_call;
   HYPRE_Int    adaptive_dt; /* adaptive time stepping flag */
   HYPRE_Real   max_cfl;      /* Maximum CFL for adaptive time stepping (0=no limit) */
   NewtonParams newton;
   char        *yaml_file;
} HeatParams;

/*--------------------------------------------------------------------------
 * Ghost Data struct for parallel communication (like lidcavity)
 *--------------------------------------------------------------------------*/
typedef struct
{
   double     *recv_bufs[26]; /* up to 26 neighbors in 3D */
   double     *send_bufs[26];
   MPI_Request reqs[52];
} GhostData3D;

typedef struct
{
   MPI_Comm      cart_comm;
   HYPRE_Int     mypid;
   HYPRE_Int     gdims[3];
   HYPRE_Int     pdims[3];
   HYPRE_Int     coords[3];
   HYPRE_Int     nlocal[3];
   HYPRE_Int     nbrs[14];
   HYPRE_Int     local_size;
   HYPRE_BigInt  ilower;
   HYPRE_BigInt  iupper;
   HYPRE_BigInt *pstarts[3];
   HYPRE_Real    gsizes[3];
} DistMesh;

static const HYPRE_Real gauss_x[2] = {-0.5773502691896257, 0.5773502691896257};
static const HYPRE_Real gauss_w[2] = {1.0, 1.0};

/* Prototypes */
static inline HYPRE_BigInt grid2idx(const HYPRE_BigInt g[3], const HYPRE_Int c[3],
                                    const HYPRE_Int gd[3], HYPRE_BigInt **ps);
int                        PrintUsage(void);
int                        ParseArguments(int, char **, HeatParams *, int, int);
static double ComputeTotalEnergyLumped(DistMesh *, HeatParams *, const HYPRE_Real *);
int  CreateDistMesh(MPI_Comm, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int,
                    HYPRE_Int, DistMesh **);
int  DestroyDistMesh(DistMesh **);
int  CreateGhostData3D(DistMesh *, GhostData3D **);
int  DestroyGhostData3D(GhostData3D **);
int  ExchangeScalarGhosts(DistMesh *, double *, GhostData3D *);
int  BuildNonlinearSystem_Heat(DistMesh *, HeatParams *, const HYPRE_Real *,
                               const HYPRE_Real *, HYPRE_IJMatrix *, HYPRE_IJVector *,
                               double *, double, GhostData3D *, GhostData3D *);
int  WriteVTKsolutionScalar(DistMesh *, HeatParams *, HYPRE_Real *, GhostData3D *, int,
                            double);
void GetVTKBaseName(HeatParams *, char *, size_t);
void GetVTKDataDir(HeatParams *, char *, size_t);
void WritePVDCollectionFromStats(HeatParams *, int, double);
void UpdateTimeStep(HeatParams *, int);
static HYPRE_Real MMS_ExactSolution(HYPRE_Real x, HYPRE_Real y, HYPRE_Real z,
                                    HYPRE_Real t, HeatParams *p);
static HYPRE_Real MMS_SourceTerm(HYPRE_Real x, HYPRE_Real y, HYPRE_Real z, HYPRE_Real t,
                                 HeatParams *p);
/* static const char *HeatOutDir(void); */ /* Reserved for future use */
static void EnsureDir(const char *path);
static void ProjectDirichlet(DistMesh *, HeatParams *, HYPRE_Real *);
static void InitializeTemperatureField(DistMesh *, HeatParams *, HYPRE_Real *);
static void q1_shape_ref(HYPRE_Real, HYPRE_Real, HYPRE_Real, HYPRE_Real N[8],
                         HYPRE_Real dxi[8], HYPRE_Real deta[8], HYPRE_Real dzeta[8]);
static void PrecomputeQ1ScalarTemplates(HYPRE_Real, HYPRE_Real, HYPRE_Real,
                                        HYPRE_Real M_t[8][8], HYPRE_Real K_t[8][8]);
static void ComputeMMSError(DistMesh *m, HeatParams *p, HYPRE_Real *T, double t,
                            double *L2_err, double *Linf_err);
static void ComputeHeatFlux(DistMesh *m, HeatParams *p, HYPRE_Real *T, GhostData3D *g,
                            HYPRE_Real *qx, HYPRE_Real *qy, HYPRE_Real *qz);

/* Safe exponential to avoid under/overflow in k(T) = k0 * exp(beta*T) */
static inline HYPRE_Real
safe_exp(HYPRE_Real x)
{
   /* Clamp exponent to a reasonable range for double precision */
   const HYPRE_Real xmin = -50.0; /* ~1.93e-22 */
   const HYPRE_Real xmax = 50.0;  /* ~3.86e+21 */
   if (x < xmin) x = xmin;
   if (x > xmax) x = xmax;
   return exp(x);
}

static double
ComputeTotalEnergyLumped(DistMesh *m, HeatParams *p, const HYPRE_Real *T)
{
   const HYPRE_Int *gd = &m->gdims[0];
   const HYPRE_Int *c  = &m->coords[0];
   HYPRE_BigInt   **ps = m->pstarts;
   int              nx = m->nlocal[0], ny = m->nlocal[1], nz = m->nlocal[2];
   HYPRE_Real       hx      = p->L[0] / (gd[0] - 1);
   HYPRE_Real       hy      = p->L[1] / (gd[1] - 1);
   HYPRE_Real       hz      = p->L[2] / (gd[2] - 1);
   HYPRE_Real       vcell   = hx * hy * hz;
   double           local_E = 0.0;
   for (int lz = 0; lz < nz; lz++)
      for (int ly = 0; ly < ny; ly++)
         for (int lx = 0; lx < nx; lx++)
         {
            HYPRE_BigInt gx = ps[0][c[0]] + lx;
            HYPRE_BigInt gy = ps[1][c[1]] + ly;
            HYPRE_BigInt gz = ps[2][c[2]] + lz;
            int          fx = (gx == 0 || gx == gd[0] - 1) ? 1 : 2;
            int          fy = (gy == 0 || gy == gd[1] - 1) ? 1 : 2;
            int          fz = (gz == 0 || gz == gd[2] - 1) ? 1 : 2;
            HYPRE_Real   w  = vcell * ((HYPRE_Real)(fx * fy * fz)) / 8.0;
            int          li = lz * ny * nx + ly * nx + lx;
            local_E += (double)(p->rho * p->cp) * (double)T[li] * (double)w;
         }
   double global_E = 0.0;
   MPI_Allreduce(&local_E, &global_E, 1, MPI_DOUBLE, MPI_SUM, m->cart_comm);
   return global_E;
}

int
PrintUsage(void)
{
   printf("\n");
   printf(
      "Usage: ${MPIEXEC_COMMAND} ${MPIEXEC_NUMPROC_FLAG} <np> ./heatflow [options]\n\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML configuration file for solver settings\n");
   printf("  -n <nx> <ny> <nz> : Global grid nodes (default: 17 17 17)\n");
   printf("  -P <Px> <Py> <Pz> : Processor grid (default: 1 1 1)\n");
   printf("  -L <Lx> <Ly> <Lz> : Domain lengths (default: 1 1 1)\n");
   printf("  -rho <val>        : Density (default: 1)\n");
   printf("  -cp <val>         : Heat capacity (default: 1)\n");
   printf("  -dt <val>         : Time step (default: 1e-2)\n");
   printf("  -tf <val>         : Final time (default: 1.0)\n");
   printf("  -k0 <val>         : Base conductivity (default: 1)\n");
   printf("  -beta <val>       : Conductivity exponent (default: 1)\n");
   printf("  -br <n>           : Batch rows per IJ call (default: 128)\n");
   printf("  -adt              : Enable adaptive time stepping\n");
   printf("  -cfl <val>        : Maximum CFL for adaptive time stepping (0=no limit)\n");
   printf("  -vis <m>          : Visualization mode bitset (default: 0)\n");
   printf("                         Any nonzero value enables visualization\n");
   printf("                         Bit 0x2: ASCII format (default: binary)\n");
   printf("                         Bit 0x4: All timesteps (default: last only)\n");
   printf("                         Bit 0x8: Include heat flux vectors\n");
   printf("                         Bit 0x10: Include exact MMS solution\n");
   printf("  -nw_max <n>       : Newton max iterations (default: 20)\n");
   printf(
      "  -nw_tol <t>       : Newton update tolerance ||delta||_inf (default: 1e-5)\n");
   printf("  -nw_rtol <t>      : Newton residual tolerance ||R||_2 (default: 1e-5)\n");
   printf("  -v|--verbose <n>  : Verbosity bitset (default: 7)\n");
   printf("                         0x1: Library info and Newton iteration info\n");
   printf("                         0x2: System info\n");
   printf("                         0x4: Print linear system matrices\n");
   printf("  -h|--help         : This help\n\n");
   return 0;
}

int
ParseArguments(int argc, char *argv[], HeatParams *p, int myid, int nprocs)
{
   p->visualize = 0;
   p->verbose   = 7;

   p->N[0] = 17;
   p->N[1] = 17;
   p->N[2] = 17;
   p->P[0] = 1;
   p->P[1] = 1;
   p->P[2] = 1;
   p->L[0] = 1.0;
   p->L[1] = 1.0;
   p->L[2] = 1.0;

   p->rho  = 1.0;
   p->cp   = 1.0;
   p->dt   = 1e-2;
   p->tf   = 1.0;
   p->k0   = 1.0;
   p->beta = 0.0; /* default linear for easier validation */

   p->max_rows_per_call = 128;
   p->adaptive_dt       = 0;
   p->max_cfl           = 0.0; /* 0 means no limit */
   p->newton.max_iters  = 20;
   p->newton.tol        = 1e-5;
   p->newton.rtol       = 1e-5;
   p->newton.damping    = 1.0;
   p->newton.ls_min     = 1e-2;
   p->newton.ls_max     = 1.0;
   p->yaml_file         = NULL;

   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))
      {
         if (++i < argc) p->yaml_file = argv[i];
      }
      else if (!strcmp(argv[i], "-n"))
      {
         for (int j = 0; j < 3; j++) p->N[j] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-P"))
      {
         for (int j = 0; j < 3; j++) p->P[j] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-L"))
      {
         for (int j = 0; j < 3; j++) p->L[j] = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-rho"))
      {
         p->rho = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-cp"))
      {
         p->cp = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-dt"))
      {
         p->dt = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-tf"))
      {
         p->tf = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-k0"))
      {
         p->k0 = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-beta"))
      {
         p->beta = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-adt"))
      {
         p->adaptive_dt = 1;
      }
      else if (!strcmp(argv[i], "-cfl") || !strcmp(argv[i], "--max-cfl"))
      {
         if (++i < argc) p->max_cfl = atof(argv[i]);
      }
      else if (!strcmp(argv[i], "-ls_min"))
      {
         p->newton.ls_min = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-ls_max"))
      {
         p->newton.ls_max = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-br") || !strcmp(argv[i], "--batch-rows"))
      {
         p->max_rows_per_call = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-nw_max"))
      {
         p->newton.max_iters = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-nw_tol"))
      {
         p->newton.tol = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-nw_rtol"))
      {
         p->newton.rtol = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-nw_w"))
      {
         p->newton.damping = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-vis") || !strcmp(argv[i], "--visualize"))
      {
         if (i + 1 < argc && argv[i + 1][0] != '-')
         {
            p->visualize = atoi(argv[++i]);
         }
         else
         {
            p->visualize = 1; /* default binary last only */
         }
      }
      else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose"))
      {
         p->verbose = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         if (!myid) PrintUsage();
         return 2;
      }
   }

   if (p->P[0] * p->P[1] * p->P[2] != nprocs)
   {
      if (!myid) printf("Error: P grid must match nprocs\n");
      return 1;
   }

   for (int d = 0; d < 3; d++)
   {
      if (p->P[d] > p->N[d])
      {
         if (!myid) printf("Error: P[%d] > N[%d]\n", d, d);
         return 1;
      }
   }

   /* Precompute thermal diffusivity (alpha0) and min grid spacing (hmin) */
   p->alpha0 = p->k0 / (p->rho * p->cp);
   {
      HYPRE_Real hx = p->L[0] / (p->N[0] - 1);
      HYPRE_Real hy = p->L[1] / (p->N[1] - 1);
      HYPRE_Real hz = p->L[2] / (p->N[2] - 1);
      p->hmin       = hx;
      if (hy < p->hmin) p->hmin = hy;
      if (hz < p->hmin) p->hmin = hz;
   }

   return 0;
}

/* Check if global grid coordinates (gx,gy,gz) are owned by processor with coords c */
static inline int
is_local_node(const HYPRE_BigInt g[3], const HYPRE_Int c[3], HYPRE_BigInt **ps)
{
   return (g[0] >= ps[0][c[0]] && g[0] < ps[0][c[0] + 1] && g[1] >= ps[1][c[1]] &&
           g[1] < ps[1][c[1] + 1] && g[2] >= ps[2][c[2]] && g[2] < ps[2][c[2] + 1]);
}

/* Find which processor owns grid coordinates g, return its coords in owner_c */
static inline void
find_owner(const HYPRE_BigInt g[3], const HYPRE_Int pd[3], HYPRE_BigInt **ps,
           HYPRE_Int owner_c[3])
{
   for (int d = 0; d < 3; d++)
   {
      owner_c[d] = 0;
      for (int b = 0; b < pd[d]; b++)
      {
         if (g[d] >= ps[d][b] && g[d] < ps[d][b + 1])
         {
            owner_c[d] = b;
            break;
         }
      }
   }
}

/* Compute global DOF ID for node at grid coordinates g.
 * This MUST be called with the correct owner processor coordinates. */
static inline HYPRE_BigInt
grid2idx(const HYPRE_BigInt g[3], const HYPRE_Int c[3], const HYPRE_Int gd[3],
         HYPRE_BigInt **ps)
{
   return ps[2][c[2]] * gd[0] * gd[1] +
          ps[1][c[1]] * gd[0] * (ps[2][c[2] + 1] - ps[2][c[2]]) +
          ps[0][c[0]] * (ps[1][c[1] + 1] - ps[1][c[1]]) *
             (ps[2][c[2] + 1] - ps[2][c[2]]) +
          ((g[2] - ps[2][c[2]]) * (ps[1][c[1] + 1] - ps[1][c[1]]) +
           (g[1] - ps[1][c[1]])) *
             (ps[0][c[0] + 1] - ps[0][c[0]]) +
          (g[0] - ps[0][c[0]]);
}

/* Compute global DOF ID for any node, finding the correct owner first */
static inline HYPRE_BigInt
grid2idx_any(const HYPRE_BigInt g[3], const HYPRE_Int pd[3], const HYPRE_Int gd[3],
             HYPRE_BigInt **ps)
{
   HYPRE_Int owner_c[3];
   find_owner(g, pd, ps, owner_c);
   return grid2idx(g, owner_c, gd, ps);
}

int
CreateDistMesh(MPI_Comm comm, HYPRE_Int Nx, HYPRE_Int Ny, HYPRE_Int Nz, HYPRE_Int Px,
               HYPRE_Int Py, HYPRE_Int Pz, DistMesh **msh)
{
   DistMesh *m = (DistMesh *)malloc(sizeof(DistMesh));

   m->gdims[0] = Nx;
   m->gdims[1] = Ny;
   m->gdims[2] = Nz;
   m->pdims[0] = Px;
   m->pdims[1] = Py;
   m->pdims[2] = Pz;

   MPI_Cart_create(comm, 3, m->pdims, (int[]){0, 0, 0}, 1, &(m->cart_comm));
   int myid;
   MPI_Comm_rank(m->cart_comm, &myid);
   m->mypid = (HYPRE_Int)myid;
   MPI_Cart_coords(m->cart_comm, myid, 3, m->coords);

   for (int i = 0; i < 3; i++)
   {
      HYPRE_Int size = m->gdims[i] / m->pdims[i];
      HYPRE_Int rest = m->gdims[i] - size * m->pdims[i];

      m->pstarts[i] = calloc((size_t)(m->pdims[i] + 1), sizeof(HYPRE_BigInt));
      for (int j = 0; j < m->pdims[i] + 1; j++)
      {
         m->pstarts[i][j] = (HYPRE_BigInt)(size * j + (j < rest ? j : rest));
      }
      m->nlocal[i] =
         (HYPRE_Int)(m->pstarts[i][m->coords[i] + 1] - m->pstarts[i][m->coords[i]]);
      m->gsizes[i] = 1.0 / (m->gdims[i] - 1);
   }

   HYPRE_BigInt g0[3] = {m->pstarts[0][m->coords[0]], m->pstarts[1][m->coords[1]],
                         m->pstarts[2][m->coords[2]]};
   m->ilower          = grid2idx(g0, m->coords, m->gdims, m->pstarts);
   m->local_size      = m->nlocal[0] * m->nlocal[1] * m->nlocal[2];
   m->iupper          = m->ilower + m->local_size - 1;

   *msh = m;
   return 0;
}

int
DestroyDistMesh(DistMesh **pm)
{
   DistMesh *m = *pm;
   if (!m) return 0;
   MPI_Comm_free(&(m->cart_comm));
   for (int i = 0; i < 3; i++) free(m->pstarts[i]);
   free(m);
   *pm = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * Ghost Data functions for 3D scalar fields
 * 26-neighbor exchange (Faces + Edges + Corners)
 * Buffer Index Mapping: idx = (dz+1)*9 + (dy+1)*3 + (dx+1)
 * Skips idx=13 (center).
 *--------------------------------------------------------------------------*/
#define NBR_IDX(dx, dy, dz) ((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1))

int
CreateGhostData3D(DistMesh *mesh, GhostData3D **ghost_ptr)
{
   GhostData3D *g  = (GhostData3D *)calloc(1, sizeof(GhostData3D));
   int          nx = mesh->nlocal[0];
   int          ny = mesh->nlocal[1];
   int          nz = mesh->nlocal[2];

   int *pd = mesh->pdims;
   int *cd = mesh->coords;

   for (int i = 0; i < 26; i++)
   {
      g->recv_bufs[i] = NULL;
      g->send_bufs[i] = NULL;
   }

   /* Complete 26-neighbor ghost exchange for 3D multi-axis partitioning.
    * Buffer layout:
    *   0-5:   6 faces   (x-, x+, y-, y+, z-, z+)
    *   6-17:  12 edges  (xy+, xy-, x+y-, x-y+, xz+, xz-, x+z-, x-z+, yz+, yz-, y+z-,
    * y-z+) 18-25: 8 corners (+++, ++-, +-+, +--, -++, -+-, --+, ---)
    */

   /* Faces: 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+ */
   if (cd[0] > 0)
   {
      g->recv_bufs[0] = (double *)calloc((size_t)(ny * nz), sizeof(double));
      g->send_bufs[0] = (double *)calloc((size_t)(ny * nz), sizeof(double));
   }
   if (cd[0] < pd[0] - 1)
   {
      g->recv_bufs[1] = (double *)calloc((size_t)(ny * nz), sizeof(double));
      g->send_bufs[1] = (double *)calloc((size_t)(ny * nz), sizeof(double));
   }
   if (cd[1] > 0)
   {
      g->recv_bufs[2] = (double *)calloc((size_t)(nx * nz), sizeof(double));
      g->send_bufs[2] = (double *)calloc((size_t)(nx * nz), sizeof(double));
   }
   if (cd[1] < pd[1] - 1)
   {
      g->recv_bufs[3] = (double *)calloc((size_t)(nx * nz), sizeof(double));
      g->send_bufs[3] = (double *)calloc((size_t)(nx * nz), sizeof(double));
   }
   if (cd[2] > 0)
   {
      g->recv_bufs[4] = (double *)calloc((size_t)(nx * ny), sizeof(double));
      g->send_bufs[4] = (double *)calloc((size_t)(nx * ny), sizeof(double));
   }
   if (cd[2] < pd[2] - 1)
   {
      g->recv_bufs[5] = (double *)calloc((size_t)(nx * ny), sizeof(double));
      g->send_bufs[5] = (double *)calloc((size_t)(nx * ny), sizeof(double));
   }

   /* Edges along z (xy combinations): 6=x+y+, 7=x-y-, 8=x+y-, 9=x-y+ */
   if (cd[0] < pd[0] - 1 && cd[1] < pd[1] - 1)
   {
      g->recv_bufs[6] = (double *)calloc((size_t)nz, sizeof(double));
      g->send_bufs[6] = (double *)calloc((size_t)nz, sizeof(double));
   }
   if (cd[0] > 0 && cd[1] > 0)
   {
      g->recv_bufs[7] = (double *)calloc((size_t)nz, sizeof(double));
      g->send_bufs[7] = (double *)calloc((size_t)nz, sizeof(double));
   }
   if (cd[0] < pd[0] - 1 && cd[1] > 0)
   {
      g->recv_bufs[8] = (double *)calloc((size_t)nz, sizeof(double));
      g->send_bufs[8] = (double *)calloc((size_t)nz, sizeof(double));
   }
   if (cd[0] > 0 && cd[1] < pd[1] - 1)
   {
      g->recv_bufs[9] = (double *)calloc((size_t)nz, sizeof(double));
      g->send_bufs[9] = (double *)calloc((size_t)nz, sizeof(double));
   }

   /* Edges along y (xz combinations): 10=x+z+, 11=x-z-, 12=x+z-, 13=x-z+ */
   if (cd[0] < pd[0] - 1 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[10] = (double *)calloc((size_t)ny, sizeof(double));
      g->send_bufs[10] = (double *)calloc((size_t)ny, sizeof(double));
   }
   if (cd[0] > 0 && cd[2] > 0)
   {
      g->recv_bufs[11] = (double *)calloc((size_t)ny, sizeof(double));
      g->send_bufs[11] = (double *)calloc((size_t)ny, sizeof(double));
   }
   if (cd[0] < pd[0] - 1 && cd[2] > 0)
   {
      g->recv_bufs[12] = (double *)calloc((size_t)ny, sizeof(double));
      g->send_bufs[12] = (double *)calloc((size_t)ny, sizeof(double));
   }
   if (cd[0] > 0 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[13] = (double *)calloc((size_t)ny, sizeof(double));
      g->send_bufs[13] = (double *)calloc((size_t)ny, sizeof(double));
   }

   /* Edges along x (yz combinations): 14=y+z+, 15=y-z-, 16=y+z-, 17=y-z+ */
   if (cd[1] < pd[1] - 1 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[14] = (double *)calloc((size_t)nx, sizeof(double));
      g->send_bufs[14] = (double *)calloc((size_t)nx, sizeof(double));
   }
   if (cd[1] > 0 && cd[2] > 0)
   {
      g->recv_bufs[15] = (double *)calloc((size_t)nx, sizeof(double));
      g->send_bufs[15] = (double *)calloc((size_t)nx, sizeof(double));
   }
   if (cd[1] < pd[1] - 1 && cd[2] > 0)
   {
      g->recv_bufs[16] = (double *)calloc((size_t)nx, sizeof(double));
      g->send_bufs[16] = (double *)calloc((size_t)nx, sizeof(double));
   }
   if (cd[1] > 0 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[17] = (double *)calloc((size_t)nx, sizeof(double));
      g->send_bufs[17] = (double *)calloc((size_t)nx, sizeof(double));
   }

   /* Corners: 18=+++, 19=++-, 20=+-+, 21=+--, 22=-++, 23=-+-, 24=--+, 25=--- */
   if (cd[0] < pd[0] - 1 && cd[1] < pd[1] - 1 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[18] = (double *)calloc(1, sizeof(double));
      g->send_bufs[18] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] < pd[0] - 1 && cd[1] < pd[1] - 1 && cd[2] > 0)
   {
      g->recv_bufs[19] = (double *)calloc(1, sizeof(double));
      g->send_bufs[19] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] < pd[0] - 1 && cd[1] > 0 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[20] = (double *)calloc(1, sizeof(double));
      g->send_bufs[20] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] < pd[0] - 1 && cd[1] > 0 && cd[2] > 0)
   {
      g->recv_bufs[21] = (double *)calloc(1, sizeof(double));
      g->send_bufs[21] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] > 0 && cd[1] < pd[1] - 1 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[22] = (double *)calloc(1, sizeof(double));
      g->send_bufs[22] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] > 0 && cd[1] < pd[1] - 1 && cd[2] > 0)
   {
      g->recv_bufs[23] = (double *)calloc(1, sizeof(double));
      g->send_bufs[23] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] > 0 && cd[1] > 0 && cd[2] < pd[2] - 1)
   {
      g->recv_bufs[24] = (double *)calloc(1, sizeof(double));
      g->send_bufs[24] = (double *)calloc(1, sizeof(double));
   }
   if (cd[0] > 0 && cd[1] > 0 && cd[2] > 0)
   {
      g->recv_bufs[25] = (double *)calloc(1, sizeof(double));
      g->send_bufs[25] = (double *)calloc(1, sizeof(double));
   }

   *ghost_ptr = g;
   return 0;
}

int
DestroyGhostData3D(GhostData3D **ghost_ptr)
{
   GhostData3D *g = *ghost_ptr;
   if (!g) return 0;
   for (int i = 0; i < 26; i++)
   {
      if (g->recv_bufs[i]) free(g->recv_bufs[i]);
      if (g->send_bufs[i]) free(g->send_bufs[i]);
   }
   free(g);
   *ghost_ptr = NULL;
   return 0;
}

int
ExchangeScalarGhosts(DistMesh *mesh, double *vec, GhostData3D *g)
{
   int nx   = mesh->nlocal[0];
   int ny   = mesh->nlocal[1];
   int nz   = mesh->nlocal[2];
   int reqc = 0;

   int *cd = mesh->coords;

/* Helper macro to get local index */
#define IDX(i, j, k) ((k) * ny * nx + (j) * nx + (i))

   /* ===== FACES (0-5) ===== */
   /* 0: x- face */
   if (g->recv_bufs[0])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1], cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[0], ny * nz, MPI_DOUBLE, src, 0, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[0])
   {
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++) g->send_bufs[0][k * ny + j] = vec[IDX(0, j, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1], cd[2]}, &dst);
      MPI_Isend(g->send_bufs[0], ny * nz, MPI_DOUBLE, dst, 1, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 1: x+ face */
   if (g->recv_bufs[1])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1], cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[1], ny * nz, MPI_DOUBLE, src, 1, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[1])
   {
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++)
            g->send_bufs[1][k * ny + j] = vec[IDX(nx - 1, j, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1], cd[2]}, &dst);
      MPI_Isend(g->send_bufs[1], ny * nz, MPI_DOUBLE, dst, 0, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 2: y- face */
   if (g->recv_bufs[2])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] - 1, cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[2], nx * nz, MPI_DOUBLE, src, 2, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[2])
   {
      for (int k = 0; k < nz; k++)
         for (int i = 0; i < nx; i++) g->send_bufs[2][k * nx + i] = vec[IDX(i, 0, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] - 1, cd[2]}, &dst);
      MPI_Isend(g->send_bufs[2], nx * nz, MPI_DOUBLE, dst, 3, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 3: y+ face */
   if (g->recv_bufs[3])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] + 1, cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[3], nx * nz, MPI_DOUBLE, src, 3, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[3])
   {
      for (int k = 0; k < nz; k++)
         for (int i = 0; i < nx; i++)
            g->send_bufs[3][k * nx + i] = vec[IDX(i, ny - 1, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] + 1, cd[2]}, &dst);
      MPI_Isend(g->send_bufs[3], nx * nz, MPI_DOUBLE, dst, 2, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 4: z- face */
   if (g->recv_bufs[4])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1], cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[4], nx * ny, MPI_DOUBLE, src, 4, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[4])
   {
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++) g->send_bufs[4][j * nx + i] = vec[IDX(i, j, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1], cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[4], nx * ny, MPI_DOUBLE, dst, 5, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 5: z+ face */
   if (g->recv_bufs[5])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1], cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[5], nx * ny, MPI_DOUBLE, src, 5, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[5])
   {
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++)
            g->send_bufs[5][j * nx + i] = vec[IDX(i, j, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1], cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[5], nx * ny, MPI_DOUBLE, dst, 4, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

   /* ===== EDGES ALONG Z (6-9): xy combinations ===== */
   /* 6: x+y+ edge */
   if (g->recv_bufs[6])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] + 1, cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[6], nz, MPI_DOUBLE, src, 6, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[6])
   {
      for (int k = 0; k < nz; k++) g->send_bufs[6][k] = vec[IDX(nx - 1, ny - 1, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] + 1, cd[2]}, &dst);
      MPI_Isend(g->send_bufs[6], nz, MPI_DOUBLE, dst, 7, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 7: x-y- edge */
   if (g->recv_bufs[7])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] - 1, cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[7], nz, MPI_DOUBLE, src, 7, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[7])
   {
      for (int k = 0; k < nz; k++) g->send_bufs[7][k] = vec[IDX(0, 0, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] - 1, cd[2]}, &dst);
      MPI_Isend(g->send_bufs[7], nz, MPI_DOUBLE, dst, 6, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 8: x+y- edge */
   if (g->recv_bufs[8])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] - 1, cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[8], nz, MPI_DOUBLE, src, 8, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[8])
   {
      for (int k = 0; k < nz; k++) g->send_bufs[8][k] = vec[IDX(nx - 1, 0, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] - 1, cd[2]}, &dst);
      MPI_Isend(g->send_bufs[8], nz, MPI_DOUBLE, dst, 9, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 9: x-y+ edge */
   if (g->recv_bufs[9])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] + 1, cd[2]}, &src);
      MPI_Irecv(g->recv_bufs[9], nz, MPI_DOUBLE, src, 9, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[9])
   {
      for (int k = 0; k < nz; k++) g->send_bufs[9][k] = vec[IDX(0, ny - 1, k)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] + 1, cd[2]}, &dst);
      MPI_Isend(g->send_bufs[9], nz, MPI_DOUBLE, dst, 8, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

   /* ===== EDGES ALONG Y (10-13): xz combinations ===== */
   /* 10: x+z+ edge */
   if (g->recv_bufs[10])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1], cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[10], ny, MPI_DOUBLE, src, 10, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[10])
   {
      for (int j = 0; j < ny; j++) g->send_bufs[10][j] = vec[IDX(nx - 1, j, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1], cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[10], ny, MPI_DOUBLE, dst, 11, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 11: x-z- edge */
   if (g->recv_bufs[11])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1], cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[11], ny, MPI_DOUBLE, src, 11, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[11])
   {
      for (int j = 0; j < ny; j++) g->send_bufs[11][j] = vec[IDX(0, j, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1], cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[11], ny, MPI_DOUBLE, dst, 10, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 12: x+z- edge */
   if (g->recv_bufs[12])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1], cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[12], ny, MPI_DOUBLE, src, 12, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[12])
   {
      for (int j = 0; j < ny; j++) g->send_bufs[12][j] = vec[IDX(nx - 1, j, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1], cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[12], ny, MPI_DOUBLE, dst, 13, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 13: x-z+ edge */
   if (g->recv_bufs[13])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1], cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[13], ny, MPI_DOUBLE, src, 13, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[13])
   {
      for (int j = 0; j < ny; j++) g->send_bufs[13][j] = vec[IDX(0, j, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1], cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[13], ny, MPI_DOUBLE, dst, 12, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

   /* ===== EDGES ALONG X (14-17): yz combinations ===== */
   /* 14: y+z+ edge */
   if (g->recv_bufs[14])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] + 1, cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[14], nx, MPI_DOUBLE, src, 14, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[14])
   {
      for (int i = 0; i < nx; i++) g->send_bufs[14][i] = vec[IDX(i, ny - 1, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] + 1, cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[14], nx, MPI_DOUBLE, dst, 15, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 15: y-z- edge */
   if (g->recv_bufs[15])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] - 1, cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[15], nx, MPI_DOUBLE, src, 15, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[15])
   {
      for (int i = 0; i < nx; i++) g->send_bufs[15][i] = vec[IDX(i, 0, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] - 1, cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[15], nx, MPI_DOUBLE, dst, 14, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 16: y+z- edge */
   if (g->recv_bufs[16])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] + 1, cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[16], nx, MPI_DOUBLE, src, 16, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[16])
   {
      for (int i = 0; i < nx; i++) g->send_bufs[16][i] = vec[IDX(i, ny - 1, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] + 1, cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[16], nx, MPI_DOUBLE, dst, 17, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 17: y-z+ edge */
   if (g->recv_bufs[17])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] - 1, cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[17], nx, MPI_DOUBLE, src, 17, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[17])
   {
      for (int i = 0; i < nx; i++) g->send_bufs[17][i] = vec[IDX(i, 0, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0], cd[1] - 1, cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[17], nx, MPI_DOUBLE, dst, 16, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

   /* ===== CORNERS (18-25) ===== */
   /* 18: +++ corner */
   if (g->recv_bufs[18])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] + 1, cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[18], 1, MPI_DOUBLE, src, 18, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[18])
   {
      g->send_bufs[18][0] = vec[IDX(nx - 1, ny - 1, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] + 1, cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[18], 1, MPI_DOUBLE, dst, 25, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 19: ++- corner */
   if (g->recv_bufs[19])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] + 1, cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[19], 1, MPI_DOUBLE, src, 19, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[19])
   {
      g->send_bufs[19][0] = vec[IDX(nx - 1, ny - 1, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] + 1, cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[19], 1, MPI_DOUBLE, dst, 24, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 20: +-+ corner */
   if (g->recv_bufs[20])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] - 1, cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[20], 1, MPI_DOUBLE, src, 20, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[20])
   {
      g->send_bufs[20][0] = vec[IDX(nx - 1, 0, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] - 1, cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[20], 1, MPI_DOUBLE, dst, 23, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 21: +-- corner */
   if (g->recv_bufs[21])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] - 1, cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[21], 1, MPI_DOUBLE, src, 21, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[21])
   {
      g->send_bufs[21][0] = vec[IDX(nx - 1, 0, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] + 1, cd[1] - 1, cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[21], 1, MPI_DOUBLE, dst, 22, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 22: -++ corner */
   if (g->recv_bufs[22])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] + 1, cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[22], 1, MPI_DOUBLE, src, 22, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[22])
   {
      g->send_bufs[22][0] = vec[IDX(0, ny - 1, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] + 1, cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[22], 1, MPI_DOUBLE, dst, 21, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 23: -+- corner */
   if (g->recv_bufs[23])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] + 1, cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[23], 1, MPI_DOUBLE, src, 23, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[23])
   {
      g->send_bufs[23][0] = vec[IDX(0, ny - 1, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] + 1, cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[23], 1, MPI_DOUBLE, dst, 20, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 24: --+ corner */
   if (g->recv_bufs[24])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] - 1, cd[2] + 1}, &src);
      MPI_Irecv(g->recv_bufs[24], 1, MPI_DOUBLE, src, 24, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[24])
   {
      g->send_bufs[24][0] = vec[IDX(0, 0, nz - 1)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] - 1, cd[2] + 1}, &dst);
      MPI_Isend(g->send_bufs[24], 1, MPI_DOUBLE, dst, 19, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   /* 25: --- corner */
   if (g->recv_bufs[25])
   {
      int src;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] - 1, cd[2] - 1}, &src);
      MPI_Irecv(g->recv_bufs[25], 1, MPI_DOUBLE, src, 25, mesh->cart_comm,
                &g->reqs[reqc++]);
   }
   if (g->send_bufs[25])
   {
      g->send_bufs[25][0] = vec[IDX(0, 0, 0)];
      int dst;
      MPI_Cart_rank(mesh->cart_comm, (int[]){cd[0] - 1, cd[1] - 1, cd[2] - 1}, &dst);
      MPI_Isend(g->send_bufs[25], 1, MPI_DOUBLE, dst, 18, mesh->cart_comm,
                &g->reqs[reqc++]);
   }

#undef IDX

   if (reqc > 0) MPI_Waitall(reqc, g->reqs, MPI_STATUSES_IGNORE);
   return 0;
}

static void
ProjectDirichlet(DistMesh *m, HeatParams *p, HYPRE_Real *T)
{
   /* Enforce Dirichlet BC: T = 0 at y = 0 plane.
    * This ensures the solution vector respects the BC after corrections. */
   const HYPRE_Int *c  = &m->coords[0];
   HYPRE_BigInt   **ps = m->pstarts;
   int              nx = m->nlocal[0], ny = m->nlocal[1], nz = m->nlocal[2];

   /* Only processors owning y=0 plane */
   if (ps[1][c[1]] == 0)
   {
      for (int lz = 0; lz < nz; lz++)
         for (int lx = 0; lx < nx; lx++)
         {
            int li = lz * ny * nx + 0 * nx + lx; /* ly = 0 */
            T[li]  = 0.0;
         }
   }
   (void)p;
}

static void
InitializeTemperatureField(DistMesh *mesh, HeatParams *params, HYPRE_Real *T_now)
{
   const HYPRE_Int *gd = &mesh->gdims[0];
   const HYPRE_Int *c  = &mesh->coords[0];
   HYPRE_BigInt   **ps = mesh->pstarts;
   HYPRE_Real       hx = params->L[0] / (gd[0] - 1);
   HYPRE_Real       hy = params->L[1] / (gd[1] - 1);
   HYPRE_Real       hz = params->L[2] / (gd[2] - 1);

   for (HYPRE_BigInt gz = ps[2][c[2]]; gz < ps[2][c[2] + 1]; gz++)
      for (HYPRE_BigInt gy = ps[1][c[1]]; gy < ps[1][c[1] + 1]; gy++)
         for (HYPRE_BigInt gx = ps[0][c[0]]; gx < ps[0][c[0] + 1]; gx++)
         {
            HYPRE_BigInt gid = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, c, gd, ps);
            HYPRE_BigInt li  = gid - mesh->ilower;
            T_now[li]        = MMS_ExactSolution(gx * hx, gy * hy, gz * hz, 0.0, params);
         }
}

static void __attribute__((unused))
EnsureDir(const char *path)
{
   if (!path || !*path) return;
   char   buf[512];
   size_t len = strlen(path);
   if (len >= sizeof(buf)) return;
   strcpy(buf, path);
   for (char *p = buf + 1; *p; ++p)
   {
      if (*p == '/')
      {
         *p = '\0';
         mkdir(buf, 0777);
         *p = '/';
      }
   }
   mkdir(buf, 0777);
}

/*--------------------------------------------------------------------------
 * MMS (Method of Manufactured Solutions) function
 * T(x,y,z,t) = exp(-t) * [(1+cos(2*pi*x/Lx))/2] * sin(pi*y/(2*Ly))
 *                      * [(1+cos(2*pi*z/Lz))/2]
 *
 * This non-negative 3D solution satisfies:
 *   - T >= 0 everywhere (all factors are non-negative)
 *   - T = 0 at y = 0 (Dirichlet cold base, since sin(0) = 0)
 *   - dT/dx = 0 at x = 0, Lx (insulated, since sin(0) = sin(2*pi) = 0)
 *   - dT/dy = 0 at y = Ly (insulated top, since cos(pi/2) = 0)
 *   - dT/dz = 0 at z = 0, Lz (insulated, since sin(0) = sin(2*pi) = 0)
 *--------------------------------------------------------------------------*/
static HYPRE_Real
MMS_ExactSolution(HYPRE_Real x, HYPRE_Real y, HYPRE_Real z, HYPRE_Real t, HeatParams *p)
{
   HYPRE_Real ax = 2.0 * M_PI * x / p->L[0];   /* 2*pi*x/Lx */
   HYPRE_Real ay = M_PI * y / (2.0 * p->L[1]); /* pi*y/(2*Ly) */
   HYPRE_Real az = 2.0 * M_PI * z / p->L[2];   /* 2*pi*z/Lz */

   HYPRE_Real fx = (1.0 + cos(ax)) / 2.0; /* in [0, 1] */
   HYPRE_Real fy = sin(ay);               /* in [0, 1] for y in [0, Ly] */
   HYPRE_Real fz = (1.0 + cos(az)) / 2.0; /* in [0, 1] */

   return exp(-t) * fx * fy * fz;
}

static HYPRE_Real
MMS_SourceTerm(HYPRE_Real x, HYPRE_Real y, HYPRE_Real z, HYPRE_Real t, HeatParams *p)
{
   /* Compute source Q such that T_exact satisfies the PDE:
    * rho*cp * dT/dt - div(k*grad(T)) = Q
    * For linear k (beta=0): k = k0
    * For nonlinear k: k(T) = k0 * exp(beta*T)
    *
    * MMS solution: T = exp(-t) * fx * fy * fz
    * where:
    *   fx = (1 + cos(2*pi*x/Lx))/2
    *   fy = sin(pi*y/(2*Ly))
    *   fz = (1 + cos(2*pi*z/Lz))/2
    */
   HYPRE_Real ax = 2.0 * M_PI * x / p->L[0];
   HYPRE_Real ay = M_PI * y / (2.0 * p->L[1]);
   HYPRE_Real az = 2.0 * M_PI * z / p->L[2];

   HYPRE_Real sx = sin(ax), cx = cos(ax); /* sin/cos of 2*pi*x/Lx */
   HYPRE_Real sy = sin(ay), cy = cos(ay); /* sin/cos of pi*y/(2*Ly) */
   HYPRE_Real sz = sin(az), cz = cos(az); /* sin/cos of 2*pi*z/Lz */

   /* Component functions */
   HYPRE_Real fx = (1.0 + cx) / 2.0;
   HYPRE_Real fy = sy;
   HYPRE_Real fz = (1.0 + cz) / 2.0;

   /* Derivative coefficients */
   HYPRE_Real pi_Lx = M_PI / p->L[0];
   HYPRE_Real pi_Ly = M_PI / (2.0 * p->L[1]);
   HYPRE_Real pi_Lz = M_PI / p->L[2];

   /* T and dT/dt */
   HYPRE_Real et      = exp(-t);
   HYPRE_Real T_exact = et * fx * fy * fz;
   HYPRE_Real dTdt    = -T_exact;

   HYPRE_Real k = p->k0 * safe_exp(p->beta * T_exact);

   /* Gradients:
    * dfx/dx = -(pi/Lx) * sin(2*pi*x/Lx) = -pi_Lx * sx
    * dfy/dy = (pi/(2*Ly)) * cos(pi*y/(2*Ly)) = pi_Ly * cy
    * dfz/dz = -(pi/Lz) * sin(2*pi*z/Lz) = -pi_Lz * sz
    *
    * dT/dx = et * (dfx/dx) * fy * fz = -et * pi_Lx * sx * fy * fz
    * dT/dy = et * fx * (dfy/dy) * fz = et * pi_Ly * fx * cy * fz
    * dT/dz = et * fx * fy * (dfz/dz) = -et * pi_Lz * fx * fy * sz
    */
   HYPRE_Real dTdx = -et * pi_Lx * sx * fy * fz;
   HYPRE_Real dTdy = et * pi_Ly * fx * cy * fz;
   HYPRE_Real dTdz = -et * pi_Lz * fx * fy * sz;

   /* Second derivatives:
    * d2fx/dx2 = -(2*pi^2/Lx^2) * cos(2*pi*x/Lx) = -2 * pi_Lx^2 * cx
    * d2fy/dy2 = -(pi^2/(4*Ly^2)) * sin(pi*y/(2*Ly)) = -pi_Ly^2 * sy
    * d2fz/dz2 = -(2*pi^2/Lz^2) * cos(2*pi*z/Lz) = -2 * pi_Lz^2 * cz
    *
    * d2T/dx2 = et * (d2fx/dx2) * fy * fz = -et * 2 * pi_Lx^2 * cx * fy * fz
    * d2T/dy2 = et * fx * (d2fy/dy2) * fz = -et * pi_Ly^2 * fx * sy * fz
    * d2T/dz2 = et * fx * fy * (d2fz/dz2) = -et * 2 * pi_Lz^2 * fx * fy * cz
    */
   HYPRE_Real lapT = -et * fy *
                     (2.0 * pi_Lx * pi_Lx * cx * fz + pi_Ly * pi_Ly * fx * fz +
                      2.0 * pi_Lz * pi_Lz * fx * cz);

   /* k'(T) = beta * k */
   HYPRE_Real kp = p->beta * k;

   /* -div(k*grad(T)) = -k*lap(T) - k'(T)*|grad(T)|^2 */
   HYPRE_Real grad2       = dTdx * dTdx + dTdy * dTdy + dTdz * dTdz;
   HYPRE_Real div_k_gradT = -k * lapT - kp * grad2;

   /* Q = rho*cp*dT/dt + div_k_gradT */
   return p->rho * p->cp * dTdt + div_k_gradT;
}

static void
ComputeMMSError(DistMesh *m, HeatParams *p, HYPRE_Real *T, double t, double *L2_err,
                double *Linf_err)
{
   const HYPRE_Int *gd = &m->gdims[0];
   const HYPRE_Int *c  = &m->coords[0];
   HYPRE_BigInt   **ps = m->pstarts;

   HYPRE_Real hx = p->L[0] / (gd[0] - 1);
   HYPRE_Real hy = p->L[1] / (gd[1] - 1);
   HYPRE_Real hz = p->L[2] / (gd[2] - 1);

   double local_L2  = 0.0;
   double local_inf = 0.0;

   for (HYPRE_BigInt gz = ps[2][c[2]]; gz < ps[2][c[2] + 1]; gz++)
      for (HYPRE_BigInt gy = ps[1][c[1]]; gy < ps[1][c[1] + 1]; gy++)
         for (HYPRE_BigInt gx = ps[0][c[0]]; gx < ps[0][c[0] + 1]; gx++)
         {
            HYPRE_BigInt gid = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, c, gd, ps);
            HYPRE_BigInt li  = gid - m->ilower;

            HYPRE_Real x = gx * hx;
            HYPRE_Real y = gy * hy;
            HYPRE_Real z = gz * hz;

            HYPRE_Real T_exact = MMS_ExactSolution(x, y, z, t, p);
            HYPRE_Real err     = fabs(T[li] - T_exact);

            local_L2 += err * err * hx * hy * hz;
            if (err > local_inf) local_inf = err;
         }

   double global_L2, global_inf;
   MPI_Allreduce(&local_L2, &global_L2, 1, MPI_DOUBLE, MPI_SUM, m->cart_comm);
   MPI_Allreduce(&local_inf, &global_inf, 1, MPI_DOUBLE, MPI_MAX, m->cart_comm);

   *L2_err   = sqrt(global_L2);
   *Linf_err = global_inf;
}

/*--------------------------------------------------------------------------
 * Heat flux computation: q = -k(T) * grad(T)
 *--------------------------------------------------------------------------*/
static void
ComputeHeatFlux(DistMesh *m, HeatParams *p, HYPRE_Real *T, GhostData3D *g, HYPRE_Real *qx,
                HYPRE_Real *qy, HYPRE_Real *qz)
{
   const HYPRE_Int *gd = &m->gdims[0];
   const HYPRE_Int *c  = &m->coords[0];
   HYPRE_BigInt   **ps = m->pstarts;
   int              nx = m->nlocal[0];
   int              ny = m->nlocal[1];
   int              nz = m->nlocal[2];

   HYPRE_Real hx = p->L[0] / (gd[0] - 1);
   HYPRE_Real hy = p->L[1] / (gd[1] - 1);
   HYPRE_Real hz = p->L[2] / (gd[2] - 1);

   /* Exchange ghost values */
   ExchangeScalarGhosts(m, T, g);

   for (int lz = 0; lz < nz; lz++)
      for (int ly = 0; ly < ny; ly++)
         for (int lx = 0; lx < nx; lx++)
         {
            int lidx = lz * ny * nx + ly * nx + lx;

            /* Global coordinates */
            int gx = (int)ps[0][c[0]] + lx;
            int gy = (int)ps[1][c[1]] + ly;
            int gz = (int)ps[2][c[2]] + lz;

            HYPRE_Real T_c = T[lidx];
            HYPRE_Real k   = p->k0 * safe_exp(p->beta * T_c);

            /* Compute gradients using central differences */
            HYPRE_Real dTdx = 0.0, dTdy = 0.0, dTdz = 0.0;
            HYPRE_Real T_xm, T_xp, T_ym, T_yp, T_zm, T_zp;

            /* x-direction */
            if (lx > 0) T_xm = T[lidx - 1];
            else if (g->recv_bufs[0]) T_xm = g->recv_bufs[0][lz * ny + ly];
            else T_xm = T_c; /* boundary */

            if (lx < nx - 1) T_xp = T[lidx + 1];
            else if (g->recv_bufs[1]) T_xp = g->recv_bufs[1][lz * ny + ly];
            else T_xp = T_c; /* boundary */

            /* y-direction */
            if (ly > 0) T_ym = T[lidx - nx];
            else if (g->recv_bufs[2]) T_ym = g->recv_bufs[2][lz * nx + lx];
            else T_ym = T_c; /* boundary */

            if (ly < ny - 1) T_yp = T[lidx + nx];
            else if (g->recv_bufs[3]) T_yp = g->recv_bufs[3][lz * nx + lx];
            else T_yp = T_c; /* boundary */

            /* z-direction */
            if (lz > 0) T_zm = T[lidx - ny * nx];
            else if (g->recv_bufs[4]) T_zm = g->recv_bufs[4][ly * nx + lx];
            else T_zm = T_c; /* boundary */

            if (lz < nz - 1) T_zp = T[lidx + ny * nx];
            else if (g->recv_bufs[5]) T_zp = g->recv_bufs[5][ly * nx + lx];
            else T_zp = T_c; /* boundary */

            /* Boundary handling for gradient */
            if (gx == 0 || gx == gd[0] - 1)
               dTdx = (gx == 0) ? (T_xp - T_c) / hx : (T_c - T_xm) / hx;
            else dTdx = (T_xp - T_xm) / (2.0 * hx);

            if (gy == 0 || gy == gd[1] - 1)
               dTdy = (gy == 0) ? (T_yp - T_c) / hy : (T_c - T_ym) / hy;
            else dTdy = (T_yp - T_ym) / (2.0 * hy);

            if (gz == 0 || gz == gd[2] - 1)
               dTdz = (gz == 0) ? (T_zp - T_c) / hz : (T_c - T_zm) / hz;
            else dTdz = (T_zp - T_zm) / (2.0 * hz);

            qx[lidx] = -k * dTdx;
            qy[lidx] = -k * dTdy;
            qz[lidx] = -k * dTdz;
         }
}

/*--------------------------------------------------------------------------
 * VTK helper functions (like lidcavity)
 *--------------------------------------------------------------------------*/
void
GetVTKBaseName(HeatParams *params, char *buf, size_t bufsize)
{
   snprintf(buf, bufsize, "heatflow_%dx%dx%d_%dx%dx%d", (int)params->N[0],
            (int)params->N[1], (int)params->N[2], (int)params->P[0], (int)params->P[1],
            (int)params->P[2]);
}

void
GetVTKDataDir(HeatParams *params, char *buf, size_t bufsize)
{
   char base[256];
   GetVTKBaseName(params, base, sizeof(base));
   /* Limit base length to leave room for "-data" (5 bytes) */
   size_t base_len     = strlen(base);
   size_t max_base_len = (bufsize > 5) ? bufsize - 5 : 0;
   if (base_len > max_base_len) base_len = max_base_len;
   snprintf(buf, bufsize, "%.*s-data", (int)base_len, base);
}

void
WritePVDCollectionFromStats(HeatParams *params, int num_procs, double final_time)
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

   /* Check if all timesteps should be included (bit 2: 0x4) */
   int write_all = (params->visualize & 0x4) != 0;

   /* Compute time per step (approximate - assumes uniform dt) */
   double dt_approx = final_time / num_steps;

   /* Determine which timesteps to include in PVD */
   int start_idx = write_all ? 0 : (num_steps - 1);
   int end_idx   = num_steps;

   for (int i = start_idx; i < end_idx; i++)
   {
      int timestep_id;
      HYPREDRV_StatsLevelGetEntry(0, i, &timestep_id, NULL, NULL, NULL, NULL);

      /* timestep_id is 1-indexed, file names are 0-indexed */
      int    file_idx = timestep_id - 1;
      double sim_time = timestep_id * dt_approx;

      for (int r = 0; r < num_procs; r++)
      {
         char vtr_file[512];
         snprintf(vtr_file, sizeof(vtr_file), "%s/step_%05d_%d.vtr", data_dir, file_idx,
                  r);

         fprintf(fp,
                 "    <DataSet timestep=\"%g\" group=\"\" part=\"%d\" file=\"%s\"/>\n",
                 sim_time, r, vtr_file);
      }
   }

   fprintf(fp, "  </Collection>\n");
   fprintf(fp, "</VTKFile>\n");
   fclose(fp);
}

void
UpdateTimeStep(HeatParams *params, int newton_iter_count)
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
      HYPRE_Real dt_max = params->max_cfl * params->hmin;
      if (params->dt > dt_max)
      {
         params->dt = dt_max;
      }
   }
}

static void
q1_shape_ref(HYPRE_Real xi, HYPRE_Real eta, HYPRE_Real zeta, HYPRE_Real N[8],
             HYPRE_Real dxi[8], HYPRE_Real deta[8], HYPRE_Real dzeta[8])
{
   static const int s[8][3] = {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
                               {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

   for (int a = 0; a < 8; a++)
   {
      HYPRE_Real sx = s[a][0];
      HYPRE_Real sy = s[a][1];
      HYPRE_Real sz = s[a][2];

      HYPRE_Real f1 = (1 + sx * xi);
      HYPRE_Real f2 = (1 + sy * eta);
      HYPRE_Real f3 = (1 + sz * zeta);

      N[a]     = 0.125 * f1 * f2 * f3;
      dxi[a]   = 0.125 * sx * f2 * f3;
      deta[a]  = 0.125 * sy * f1 * f3;
      dzeta[a] = 0.125 * sz * f1 * f2;
   }
}

static void
PrecomputeQ1ScalarTemplates(HYPRE_Real hx, HYPRE_Real hy, HYPRE_Real hz,
                            HYPRE_Real M_t[8][8], HYPRE_Real K_t[8][8])
{
   for (int i = 0; i < 8; i++)
   {
      for (int j = 0; j < 8; j++)
      {
         M_t[i][j] = 0.0;
         K_t[i][j] = 0.0;
      }
   }

   HYPRE_Real Jinv[3] = {2.0 / hx, 2.0 / hy, 2.0 / hz};
   HYPRE_Real detJ    = (hx * hy * hz) / 8.0;

   for (int iz = 0; iz < 2; iz++)
   {
      HYPRE_Real z = gauss_x[iz], wz = gauss_w[iz];
      for (int iy = 0; iy < 2; iy++)
      {
         HYPRE_Real e = gauss_x[iy], wy = gauss_w[iy];
         for (int ix = 0; ix < 2; ix++)
         {
            HYPRE_Real x = gauss_x[ix], wx = gauss_w[ix];
            HYPRE_Real w = wx * wy * wz * detJ;

            HYPRE_Real N[8], dxi[8], deta[8], dzeta[8];
            q1_shape_ref(x, e, z, N, dxi, deta, dzeta);

            HYPRE_Real dNx[8], dNy[8], dNz[8];
            for (int a = 0; a < 8; a++)
            {
               dNx[a] = dxi[a] * Jinv[0];
               dNy[a] = deta[a] * Jinv[1];
               dNz[a] = dzeta[a] * Jinv[2];
            }

            for (int a = 0; a < 8; a++)
            {
               for (int b_idx = 0; b_idx < 8; b_idx++)
               {
                  M_t[a][b_idx] += N[a] * N[b_idx] * w;
                  K_t[a][b_idx] +=
                     (dNx[a] * dNx[b_idx] + dNy[a] * dNy[b_idx] + dNz[a] * dNz[b_idx]) *
                     w;
               }
            }
         }
      }
   }
}

int
BuildNonlinearSystem_Heat(DistMesh *m, HeatParams *p, const HYPRE_Real *T_state,
                          const HYPRE_Real *T_prev, HYPRE_IJMatrix *A_ptr,
                          HYPRE_IJVector *b_ptr, double *rnorm_out, double current_time,
                          GhostData3D *g_T_k, GhostData3D *g_T_n)
{
   const HYPRE_Int *gd   = &m->gdims[0];
   HYPRE_BigInt   **ps   = m->pstarts;
   const HYPRE_Int *c    = &m->coords[0];
   HYPRE_BigInt     nlow = m->ilower, nhigh = m->iupper; /* node ids */
   HYPRE_BigInt     dof_low = nlow, dof_high = nhigh;

   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin("system", -1));

   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJMatrixCreate(m->cart_comm, dof_low, dof_high, dof_low, dof_high, &A);
   HYPRE_IJVectorCreate(m->cart_comm, dof_low, dof_high, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   HYPRE_Int  local_dofs = m->local_size;
   HYPRE_Int *nnzrow     = (HYPRE_Int *)calloc((size_t)local_dofs, sizeof(HYPRE_Int));
   for (int i = 0; i < local_dofs; i++) nnzrow[i] = 64;
   HYPRE_IJMatrixSetRowSizes(A, nnzrow);
   free(nnzrow);
   HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST);

   /* Exchange ghost data for T_state and T_prev */
   if (g_T_k) ExchangeScalarGhosts(m, (double *)T_state, g_T_k);
   if (g_T_n) ExchangeScalarGhosts(m, (double *)T_prev, g_T_n);

   /* Local dimensions */
   int nx = m->nlocal[0], ny = m->nlocal[1], nz = m->nlocal[2];

/* Helper function to get T value at a node (local or ghost).
 * Buffer layout:
 *   0-5:   6 faces   (x-, x+, y-, y+, z-, z+)
 *   6-9:   4 edges along z (x+y+, x-y-, x+y-, x-y+)
 *   10-13: 4 edges along y (x+z+, x-z-, x+z-, x-z+)
 *   14-17: 4 edges along x (y+z+, y-z-, y+z-, y-z+)
 *   18-25: 8 corners (+++, ++-, +-+, +--, -++, -+-, --+, ---)
 */
#define GET_T_VALUE(T_local, g_T, ng_coords)                             \
   ({                                                                    \
      HYPRE_Real _val  = 0.0;                                            \
      int        _lx   = (int)(ng_coords[0] - ps[0][c[0]]);              \
      int        _ly   = (int)(ng_coords[1] - ps[1][c[1]]);              \
      int        _lz   = (int)(ng_coords[2] - ps[2][c[2]]);              \
      int        _in_x = (_lx >= 0 && _lx < nx);                         \
      int        _in_y = (_ly >= 0 && _ly < ny);                         \
      int        _in_z = (_lz >= 0 && _lz < nz);                         \
      int        _xm = (_lx == -1), _xp = (_lx == nx);                   \
      int        _ym = (_ly == -1), _yp = (_ly == ny);                   \
      int        _zm = (_lz == -1), _zp = (_lz == nz);                   \
      if (_in_x && _in_y && _in_z)                                       \
      {                                                                  \
         _val = T_local ? T_local[_lz * ny * nx + _ly * nx + _lx] : 0.0; \
      }                                                                  \
      else if (g_T)                                                      \
      {                                                                  \
         /* Faces */                                                     \
         if (_xm && _in_y && _in_z && g_T->recv_bufs[0])                 \
         {                                                               \
            _val = g_T->recv_bufs[0][_lz * ny + _ly];                    \
         }                                                               \
         else if (_xp && _in_y && _in_z && g_T->recv_bufs[1])            \
         {                                                               \
            _val = g_T->recv_bufs[1][_lz * ny + _ly];                    \
         }                                                               \
         else if (_ym && _in_x && _in_z && g_T->recv_bufs[2])            \
         {                                                               \
            _val = g_T->recv_bufs[2][_lz * nx + _lx];                    \
         }                                                               \
         else if (_yp && _in_x && _in_z && g_T->recv_bufs[3])            \
         {                                                               \
            _val = g_T->recv_bufs[3][_lz * nx + _lx];                    \
         }                                                               \
         else if (_zm && _in_x && _in_y && g_T->recv_bufs[4])            \
         {                                                               \
            _val = g_T->recv_bufs[4][_ly * nx + _lx];                    \
         }                                                               \
         else if (_zp && _in_x && _in_y && g_T->recv_bufs[5])            \
         {                                                               \
            _val = g_T->recv_bufs[5][_ly * nx + _lx];                    \
         }                                                               \
         /* Edges along z (xy combinations) */                           \
         else if (_xp && _yp && _in_z && g_T->recv_bufs[6])              \
         {                                                               \
            _val = g_T->recv_bufs[6][_lz];                               \
         }                                                               \
         else if (_xm && _ym && _in_z && g_T->recv_bufs[7])              \
         {                                                               \
            _val = g_T->recv_bufs[7][_lz];                               \
         }                                                               \
         else if (_xp && _ym && _in_z && g_T->recv_bufs[8])              \
         {                                                               \
            _val = g_T->recv_bufs[8][_lz];                               \
         }                                                               \
         else if (_xm && _yp && _in_z && g_T->recv_bufs[9])              \
         {                                                               \
            _val = g_T->recv_bufs[9][_lz];                               \
         }                                                               \
         /* Edges along y (xz combinations) */                           \
         else if (_xp && _zp && _in_y && g_T->recv_bufs[10])             \
         {                                                               \
            _val = g_T->recv_bufs[10][_ly];                              \
         }                                                               \
         else if (_xm && _zm && _in_y && g_T->recv_bufs[11])             \
         {                                                               \
            _val = g_T->recv_bufs[11][_ly];                              \
         }                                                               \
         else if (_xp && _zm && _in_y && g_T->recv_bufs[12])             \
         {                                                               \
            _val = g_T->recv_bufs[12][_ly];                              \
         }                                                               \
         else if (_xm && _zp && _in_y && g_T->recv_bufs[13])             \
         {                                                               \
            _val = g_T->recv_bufs[13][_ly];                              \
         }                                                               \
         /* Edges along x (yz combinations) */                           \
         else if (_yp && _zp && _in_x && g_T->recv_bufs[14])             \
         {                                                               \
            _val = g_T->recv_bufs[14][_lx];                              \
         }                                                               \
         else if (_ym && _zm && _in_x && g_T->recv_bufs[15])             \
         {                                                               \
            _val = g_T->recv_bufs[15][_lx];                              \
         }                                                               \
         else if (_yp && _zm && _in_x && g_T->recv_bufs[16])             \
         {                                                               \
            _val = g_T->recv_bufs[16][_lx];                              \
         }                                                               \
         else if (_ym && _zp && _in_x && g_T->recv_bufs[17])             \
         {                                                               \
            _val = g_T->recv_bufs[17][_lx];                              \
         }                                                               \
         /* Corners */                                                   \
         else if (_xp && _yp && _zp && g_T->recv_bufs[18])               \
         {                                                               \
            _val = g_T->recv_bufs[18][0];                                \
         }                                                               \
         else if (_xp && _yp && _zm && g_T->recv_bufs[19])               \
         {                                                               \
            _val = g_T->recv_bufs[19][0];                                \
         }                                                               \
         else if (_xp && _ym && _zp && g_T->recv_bufs[20])               \
         {                                                               \
            _val = g_T->recv_bufs[20][0];                                \
         }                                                               \
         else if (_xp && _ym && _zm && g_T->recv_bufs[21])               \
         {                                                               \
            _val = g_T->recv_bufs[21][0];                                \
         }                                                               \
         else if (_xm && _yp && _zp && g_T->recv_bufs[22])               \
         {                                                               \
            _val = g_T->recv_bufs[22][0];                                \
         }                                                               \
         else if (_xm && _yp && _zm && g_T->recv_bufs[23])               \
         {                                                               \
            _val = g_T->recv_bufs[23][0];                                \
         }                                                               \
         else if (_xm && _ym && _zp && g_T->recv_bufs[24])               \
         {                                                               \
            _val = g_T->recv_bufs[24][0];                                \
         }                                                               \
         else if (_xm && _ym && _zm && g_T->recv_bufs[25])               \
         {                                                               \
            _val = g_T->recv_bufs[25][0];                                \
         }                                                               \
      }                                                                  \
      _val;                                                              \
   })

   HYPRE_Real hx = p->L[0] / (gd[0] - 1), hy = p->L[1] / (gd[1] - 1),
              hz = p->L[2] / (gd[2] - 1);
   HYPRE_Real M_t[8][8], K_unit[8][8];
   PrecomputeQ1ScalarTemplates(hx, hy, hz, M_t, K_unit);

   HYPRE_Real mass_coeff = p->rho * p->cp / p->dt;

   /* For residual norm on free rows only */
   HYPRE_Int   local_nodes = m->local_size;
   HYPRE_Real *res_accum = (HYPRE_Real *)calloc((size_t)local_nodes, sizeof(HYPRE_Real));
   HYPRE_Int  *dir_mask  = (HYPRE_Int *)calloc((size_t)local_nodes, sizeof(HYPRE_Int));

   /* Element loop: Process all elements that touch at least one local node.
    * This includes elements anchored in neighbor's domain (one element earlier
    * in each direction if we have a neighbor there).
    * Each processor assembles only its local rows, but by processing these
    * boundary-straddling elements, symmetric contributions are ensured.
    */
   HYPRE_BigInt gx_min = (c[0] > 0) ? (ps[0][c[0]] - 1) : 0;
   HYPRE_BigInt gy_min = (c[1] > 0) ? (ps[1][c[1]] - 1) : 0;
   HYPRE_BigInt gz_min = (c[2] > 0) ? (ps[2][c[2]] - 1) : 0;
   HYPRE_BigInt gx_max = (ps[0][c[0] + 1] < gd[0]) ? ps[0][c[0] + 1] : (gd[0] - 1);
   HYPRE_BigInt gy_max = (ps[1][c[1] + 1] < gd[1]) ? ps[1][c[1] + 1] : (gd[1] - 1);
   HYPRE_BigInt gz_max = (ps[2][c[2] + 1] < gd[2]) ? ps[2][c[2] + 1] : (gd[2] - 1);

   for (HYPRE_BigInt gz = gz_min; gz < gz_max; gz++)
      for (HYPRE_BigInt gy = gy_min; gy < gy_max; gy++)
         for (HYPRE_BigInt gx = gx_min; gx < gx_max; gx++)
         {
            HYPRE_BigInt ng[8][3] = {{gx, gy, gz},
                                     {gx + 1, gy, gz},
                                     {gx + 1, gy + 1, gz},
                                     {gx, gy + 1, gz},
                                     {gx, gy, gz + 1},
                                     {gx + 1, gy, gz + 1},
                                     {gx + 1, gy + 1, gz + 1},
                                     {gx, gy + 1, gz + 1}};
            HYPRE_BigInt gid[8];
            HYPRE_BigInt dof[8];
            int          is_local[8];
            for (int a = 0; a < 8; a++)
            {
               /* Check ownership based on grid coordinates (not global ID) */
               is_local[a] = is_local_node(ng[a], c, ps);
               /* Compute correct global ID using the proper owner's coordinates */
               gid[a] = grid2idx_any(ng[a], m->pdims, gd, ps);
               dof[a] = gid[a];
            }

            /* Dirichlet flags: Only y=0 plane is Dirichlet (T=0), all other faces
             * insulated */
            HYPRE_Int  is_dirichlet[8] = {0};
            HYPRE_Real rhs_dirichlet[8];
            for (int a = 0; a < 8; a++) rhs_dirichlet[a] = 0.0;
            for (int a = 0; a < 8; a++)
            {
               /* Dirichlet BC: T = 0 at y = 0 (cold base) */
               if (ng[a][1] == 0)
               {
                  is_dirichlet[a]  = 1;
                  rhs_dirichlet[a] = 0.0;
               }
               /* All other boundaries (x=0, x=Lx, y=Ly, z=0, z=Lz) are insulated
                * (Neumann) Natural BC: no special handling required in weak form */
            }

            /* Batch rows */
            HYPRE_Int    nrows = 0;
            HYPRE_BigInt rows[8];
            HYPRE_Int    ncols_row[8];
            HYPRE_BigInt cols_flat[8 * 64];
            HYPRE_Real   vals_flat[8 * 64];
            HYPRE_Int    off = 0;
            HYPRE_BigInt vec_rows[64];
            HYPRE_Real   vec_vals[64];
            HYPRE_Int    nvec = 0;

            HYPRE_Real fe_mass[8];
            HYPRE_Real fe_cond[8];
            HYPRE_Real fe_src[8];
            for (int i = 0; i < 8; i++)
            {
               fe_mass[i] = 0.0;
               fe_cond[i] = 0.0;
               fe_src[i]  = 0.0;
            }
            HYPRE_Real K_lin[8][8];
            HYPRE_Real J_extra[8][8];
            for (int a2 = 0; a2 < 8; a2++)
               for (int b2 = 0; b2 < 8; b2++)
               {
                  K_lin[a2][b2]   = 0.0;
                  J_extra[a2][b2] = 0.0;
               }

            /* Gauss-point loop for conduction residual/Jacobian */
            HYPRE_Real detJ    = (hx * hy * hz) / 8.0;
            HYPRE_Real Jinv[3] = {2.0 / hx, 2.0 / hy, 2.0 / hz};
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
                     }

                     /* Evaluate T (with Dirichlet overlay) and its gradient */
                     HYPRE_Real T_gp = 0.0;
                     HYPRE_Real dTdx = 0.0, dTdy = 0.0, dTdz = 0.0;
                     for (int bnode = 0; bnode < 8; bnode++)
                     {
                        HYPRE_Real Tbv = is_dirichlet[bnode]
                                            ? rhs_dirichlet[bnode]
                                            : GET_T_VALUE(T_state, g_T_k, ng[bnode]);
                        T_gp += N[bnode] * Tbv;
                        dTdx += dNx[bnode] * Tbv;
                        dTdy += dNy[bnode] * Tbv;
                        dTdz += dNz[bnode] * Tbv;
                     }

                     HYPRE_Real k     = p->k0 * safe_exp(p->beta * T_gp);
                     HYPRE_Real kprim = p->beta * k;

                     for (int a = 0; a < 8; a++)
                     {
                        /* Residual conduction term: ∫ k ∇φ_a · ∇T */
                        HYPRE_Real gradTa_dot =
                           dNx[a] * dTdx + dNy[a] * dTdy + dNz[a] * dTdz;
                        fe_cond[a] += k * gradTa_dot * w;

                        for (int b_idx = 0; b_idx < 8; b_idx++)
                        {
                           /* Linear stiffness part: k ∫ ∇φ_a·∇φ_b */
                           HYPRE_Real gradab = dNx[a] * dNx[b_idx] + dNy[a] * dNy[b_idx] +
                                               dNz[a] * dNz[b_idx];
                           K_lin[a][b_idx] += k * gradab * w;
                        }
                        /* Nonlinear Jacobian extra: k'(T) φ_l (∇φ_a·∇T) */
                        for (int l = 0; l < 8; l++)
                        {
                           J_extra[a][l] += kprim * N[l] * gradTa_dot * w;
                        }
                        /* Source term: spatial source + optional MMS */
                        /* Compute physical coordinates at this GP */
                        HYPRE_Real x_gp = 0.0, y_gp = 0.0, z_gp = 0.0;
                        for (int bnode = 0; bnode < 8; bnode++)
                        {
                           HYPRE_Real x_node = ng[bnode][0] * hx;
                           HYPRE_Real y_node = ng[bnode][1] * hy;
                           HYPRE_Real z_node = ng[bnode][2] * hz;
                           x_gp += N[bnode] * x_node;
                           y_gp += N[bnode] * y_node;
                           z_gp += N[bnode] * z_node;
                        }
                        HYPRE_Real Q_total =
                           MMS_SourceTerm(x_gp, y_gp, z_gp, current_time, p);
                        if (Q_total != 0.0)
                        {
                           fe_src[a] += Q_total * N[a] * w;
                        }
                     }
                  }
               }
            }

            /* Mass RHS using templates: +ρc/Δt M T^n  -  ρc/Δt M T^k */
            for (int a = 0; a < 8; a++)
            {
               HYPRE_Real acc_prev = 0.0, acc_curr = 0.0;
               for (int b_idx = 0; b_idx < 8; b_idx++)
               {
                  if (T_prev)
                     acc_prev += M_t[a][b_idx] * GET_T_VALUE(T_prev, g_T_n, ng[b_idx]);
                  if (T_state)
                     acc_curr += M_t[a][b_idx] * GET_T_VALUE(T_state, g_T_k, ng[b_idx]);
               }
               fe_mass[a] += mass_coeff * (acc_prev - acc_curr);
            }

            /* Assemble matrix rows - only LOCAL non-Dirichlet nodes.
             * Off-processor rows are handled by their owning processor. */
            for (int a = 0; a < 8; a++)
            {
               if (is_dirichlet[a]) continue; /* Dirichlet rows handled separately */
               if (!is_local[a]) continue;    /* Only assemble rows we own */

               rows[nrows]     = dof[a];
               HYPRE_Int ncols = 0;

               /* Interior row: normal assembly, but include Dirichlet columns with zero
                */
               for (int b_idx = 0; b_idx < 8; b_idx++)
               {
                  HYPRE_Real val;
                  if (is_dirichlet[b_idx])
                  {
                     /* Column is Dirichlet: still add entry but with zero value
                      * to maintain symmetric sparsity pattern for AMG */
                     val = 0.0;
                  }
                  else
                  {
                     val =
                        mass_coeff * M_t[a][b_idx] + K_lin[a][b_idx] + J_extra[a][b_idx];
                  }
                  if (val != 0.0 || is_dirichlet[b_idx])
                  {
                     cols_flat[off + ncols] = dof[b_idx];
                     vals_flat[off + ncols] = val;
                     ncols++;
                  }
               }
               ncols_row[nrows] = ncols;
               off += ncols;
               nrows++;
            }

            /* RHS: Dirichlet rows get zero RHS (no update needed at boundaries) */
            for (int a = 0; a < 8; a++)
            {
               if (is_dirichlet[a] && is_local[a])
               {
                  HYPRE_BigInt li = gid[a] - m->ilower;
                  dir_mask[li]    = 1;
               }
            }
            /* Free rows: -R = mass - cond + src (only LOCAL non-Dirichlet nodes)
             * Note: RHS vector only accepts local rows, unlike the matrix */
            for (int a = 0; a < 8; a++)
            {
               if (is_dirichlet[a]) continue;
               if (!is_local[a]) continue; /* Vector only accepts local rows */
               vec_rows[nvec] = dof[a];
               vec_vals[nvec] = fe_mass[a] - fe_cond[a] + fe_src[a];
               /* accumulate residual for norm */
               HYPRE_BigInt li = gid[a] - m->ilower;
               res_accum[li] += vec_vals[nvec];
               nvec++;
            }

            if (nrows > 0)
            {
               HYPRE_Int maxb = (p->max_rows_per_call > 0) ? p->max_rows_per_call : nrows;
               HYPRE_Int start_off = 0;
               for (HYPRE_Int s = 0; s < nrows; s += maxb)
               {
                  HYPRE_Int cnt = (s + maxb <= nrows) ? maxb : (nrows - s);
                  HYPRE_IJMatrixAddToValues(A, cnt, &ncols_row[s], &rows[s],
                                            &cols_flat[start_off], &vals_flat[start_off]);
                  for (HYPRE_Int i = 0; i < cnt; i++) start_off += ncols_row[s + i];
               }
            }
            if (nvec > 0)
            {
               HYPRE_Int maxb = (p->max_rows_per_call > 0) ? p->max_rows_per_call : nvec;
               for (HYPRE_Int s = 0; s < nvec; s += maxb)
               {
                  HYPRE_Int cnt = (s + maxb <= nvec) ? maxb : (nvec - s);
                  HYPRE_IJVectorAddToValues(b, cnt, &vec_rows[s], &vec_vals[s]);
               }
            }
         }

   /* Apply Dirichlet BCs: set identity rows with zero RHS.
    * This is done BEFORE assembly. For rows skipped during element assembly,
    * this creates the only entry (diagonal=1).
    * Only y=0 plane has Dirichlet BC (T=0); other faces are insulated (Neumann). */
   {
      HYPRE_Int  one      = 1;
      HYPRE_Real onev     = 1.0;
      HYPRE_Real zero_rhs = 0.0;

      /* Dirichlet BC on y=0 plane only */
      if (ps[1][c[1]] == 0) /* Only processors that own the y=0 plane */
      {
         HYPRE_BigInt gy = 0;
         for (HYPRE_BigInt gz = ps[2][c[2]]; gz < ps[2][c[2] + 1]; gz++)
            for (HYPRE_BigInt gx = ps[0][c[0]]; gx < ps[0][c[0] + 1]; gx++)
            {
               HYPRE_BigInt gid = grid2idx((HYPRE_BigInt[]){gx, gy, gz}, c, gd, ps);
               HYPRE_IJMatrixSetValues(A, 1, &one, &gid, &gid, &onev);
               HYPRE_IJVectorSetValues(b, 1, &gid, &zero_rhs);
            }
      }
   }

   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);

   /* Compute ||R||_2 over free rows only */
   if (rnorm_out)
   {
      double local_sum2 = 0.0;
      for (HYPRE_Int i = 0; i < local_nodes; i++)
      {
         if (!dir_mask[i])
         {
            double ri = (double)res_accum[i];
            local_sum2 += ri * ri;
         }
      }
      double global_sum2 = 0.0;
      MPI_Allreduce(&local_sum2, &global_sum2, 1, MPI_DOUBLE, MPI_SUM, m->cart_comm);
      *rnorm_out = sqrt(global_sum2);
   }
   free(res_accum);
   free(dir_mask);
#undef GET_T_VALUE

   *A_ptr = A;
   *b_ptr = b;

   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd("system", -1));

   return 0;
}

int
WriteVTKsolutionScalar(DistMesh *m, HeatParams *p, HYPRE_Real *sol, GhostData3D *ghost,
                       int step, double time)
{
   int myid, nprocs;
   MPI_Comm_rank(m->cart_comm, &myid);
   MPI_Comm_size(m->cart_comm, &nprocs);

   HYPRE_Int     *c  = &m->coords[0];
   HYPRE_Int     *n  = &m->nlocal[0];
   HYPRE_BigInt **ps = m->pstarts;
   HYPRE_Real    *gs = &m->gsizes[0];

   int nx = n[0], ny = n[1], nz = n[2];
   int ix0 = (int)ps[0][c[0]];
   int iy0 = (int)ps[1][c[1]];
   int iz0 = (int)ps[2][c[2]];

   /* Overlap on negative faces for VTK piece connectivity */
   int ofi  = (c[0] > 0) ? 1 : 0; /* left ghost layer */
   int ofj  = (c[1] > 0) ? 1 : 0; /* front ghost layer */
   int ofk  = (c[2] > 0) ? 1 : 0; /* bottom ghost layer */
   int nxg  = nx + ofi;
   int nyg  = ny + ofj;
   int nzg  = nz + ofk;
   int npts = nxg * nyg * nzg;

   /* Create data directory */
   char data_dir[256];
   GetVTKDataDir(p, data_dir, sizeof(data_dir));
   struct stat st = {0};
   if (stat(data_dir, &st) == -1)
   {
      mkdir(data_dir, 0755);
   }
   MPI_Barrier(m->cart_comm);

   char fn[512];
   snprintf(fn, sizeof(fn), "%s/step_%05d_%d.vtr", data_dir, step, myid);

   FILE *fp = fopen(fn, "w");
   if (!fp)
   {
      if (!myid) printf("Error opening %s\n", fn);
      MPI_Abort(m->cart_comm, -1);
   }

   /* Build extended data array including ghost overlap for VTK connectivity */
   HYPRE_Real *ext_T = (HYPRE_Real *)calloc((size_t)npts, sizeof(HYPRE_Real));

   /* Copy local data to extended array */
   for (int k = 0; k < nz; k++)
   {
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            int ig = i + ofi, jg = j + ofj, kg = k + ofk;
            int idx_e    = kg * nxg * nyg + jg * nxg + ig;
            int idx_l    = k * nx * ny + j * nx + i;
            ext_T[idx_e] = sol[idx_l];
         }
      }
   }

   /* Fill ghost regions from neighbor data (recv_bufs: 0=x-, 2=y-, 4=z-) */
   /* x- face ghost (left) */
   if (ofi > 0 && ghost && ghost->recv_bufs[0])
   {
      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            int kg = k + ofk, jg = j + ofj;
            int idx_e    = kg * nxg * nyg + jg * nxg + 0;
            int idx_g    = k * ny + j;
            ext_T[idx_e] = ghost->recv_bufs[0][idx_g];
         }
      }
   }
   /* y- face ghost (front) */
   if (ofj > 0 && ghost && ghost->recv_bufs[2])
   {
      for (int k = 0; k < nz; k++)
      {
         for (int i = 0; i < nx; i++)
         {
            int kg = k + ofk, ig = i + ofi;
            int idx_e    = kg * nxg * nyg + 0 * nxg + ig;
            int idx_g    = k * nx + i;
            ext_T[idx_e] = ghost->recv_bufs[2][idx_g];
         }
      }
   }
   /* z- face ghost (bottom) */
   if (ofk > 0 && ghost && ghost->recv_bufs[4])
   {
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            int jg = j + ofj, ig = i + ofi;
            int idx_e    = 0 * nxg * nyg + jg * nxg + ig;
            int idx_g    = j * nx + i;
            ext_T[idx_e] = ghost->recv_bufs[4][idx_g];
         }
      }
   }
   /* Edge ghosts (xy-, xz-, yz-) and corner ghost (xyz-) would need extra handling
    * For now, fill with nearest valid data to avoid gaps */
   if (ofi > 0 && ofj > 0)
   {
      for (int k = 0; k < nzg; k++)
         ext_T[k * nxg * nyg + 0 * nxg + 0] = ext_T[k * nxg * nyg + ofj * nxg + ofi];
   }
   if (ofi > 0 && ofk > 0)
   {
      for (int j = 0; j < nyg; j++)
         ext_T[0 * nxg * nyg + j * nxg + 0] = ext_T[ofk * nxg * nyg + j * nxg + ofi];
   }
   if (ofj > 0 && ofk > 0)
   {
      for (int i = 0; i < nxg; i++)
         ext_T[0 * nxg * nyg + 0 * nxg + i] = ext_T[ofk * nxg * nyg + ofj * nxg + i];
   }
   if (ofi > 0 && ofj > 0 && ofk > 0)
   {
      ext_T[0] = ext_T[ofk * nxg * nyg + ofj * nxg + ofi];
   }

   /* Compute heat flux if requested (bit 3: 0x8) */
   HYPRE_Real *qx = NULL, *qy = NULL, *qz = NULL;
   int         include_flux  = (p->visualize & 0x8) != 0;
   HYPRE_Real *T_exact       = NULL;
   int         include_exact = (p->visualize & 0x10) != 0;

   if (include_exact)
   {
      T_exact = (HYPRE_Real *)calloc((size_t)npts, sizeof(HYPRE_Real));
      /* Compute exact solution at all extended grid points */
      HYPRE_Real hx = p->L[0] / (m->gdims[0] - 1);
      HYPRE_Real hy = p->L[1] / (m->gdims[1] - 1);
      HYPRE_Real hz = p->L[2] / (m->gdims[2] - 1);
      for (int k = 0; k < nzg; k++)
         for (int j = 0; j < nyg; j++)
            for (int i = 0; i < nxg; i++)
            {
               int        idx = k * nyg * nxg + j * nxg + i;
               HYPRE_Real x   = (ix0 - ofi + i) * hx;
               HYPRE_Real y   = (iy0 - ofj + j) * hy;
               HYPRE_Real z   = (iz0 - ofk + k) * hz;
               T_exact[idx]   = MMS_ExactSolution(x, y, z, time, p);
            }
   }

   if (include_flux)
   {
      qx = (HYPRE_Real *)calloc((size_t)npts, sizeof(HYPRE_Real));
      qy = (HYPRE_Real *)calloc((size_t)npts, sizeof(HYPRE_Real));
      qz = (HYPRE_Real *)calloc((size_t)npts, sizeof(HYPRE_Real));
      /* Note: flux computed on local grid, not extended */
      HYPRE_Real *qx_loc =
         (HYPRE_Real *)calloc((size_t)(nx * ny * nz), sizeof(HYPRE_Real));
      HYPRE_Real *qy_loc =
         (HYPRE_Real *)calloc((size_t)(nx * ny * nz), sizeof(HYPRE_Real));
      HYPRE_Real *qz_loc =
         (HYPRE_Real *)calloc((size_t)(nx * ny * nz), sizeof(HYPRE_Real));
      ComputeHeatFlux(m, p, sol, ghost, qx_loc, qy_loc, qz_loc);
      /* Expand to extended grid */
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
            {
               int ig = i + ofi, jg = j + ofj, kg = k + ofk;
               int idx_e = kg * nxg * nyg + jg * nxg + ig;
               int idx_l = k * nx * ny + j * nx + i;
               qx[idx_e] = qx_loc[idx_l];
               qy[idx_e] = qy_loc[idx_l];
               qz[idx_e] = qz_loc[idx_l];
            }
      free(qx_loc);
      free(qy_loc);
      free(qz_loc);
   }

   /* VTK extents with overlap */
   int ext_x0 = ix0 - ofi, ext_x1 = ix0 + nx - 1;
   int ext_y0 = iy0 - ofj, ext_y1 = iy0 + ny - 1;
   int ext_z0 = iz0 - ofk, ext_z1 = iz0 + nz - 1;

   fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
   fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\">\n");
   fprintf(fp, "  <RectilinearGrid WholeExtent=\"%d %d %d %d %d %d\">\n", ext_x0, ext_x1,
           ext_y0, ext_y1, ext_z0, ext_z1);
   /* Time as FieldData */
   fprintf(fp, "    <FieldData>\n");
   fprintf(fp, "      <DataArray type=\"Float64\" Name=\"TimeValue\" "
               "NumberOfTuples=\"1\" format=\"ascii\">\n");
   fprintf(fp, "        %.15g\n", time);
   fprintf(fp, "      </DataArray>\n");
   fprintf(fp, "    </FieldData>\n");

   fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", ext_x0, ext_x1, ext_y0,
           ext_y1, ext_z0, ext_z1);
   fprintf(fp, "      <Coordinates>\n");
   /* X */
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"x\" "
               "format=\"ascii\">\n          ");
   for (int i = 0; i < nxg; i++)
   {
      fprintf(fp, "%.15g ", (ix0 - ofi + i) * p->L[0] * gs[0]);
      if ((i + 1) % 8 == 0) fprintf(fp, "\n          ");
   }
   fprintf(fp, "\n        </DataArray>\n");
   /* Y */
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"y\" "
               "format=\"ascii\">\n          ");
   for (int j = 0; j < nyg; j++)
   {
      fprintf(fp, "%.15g ", (iy0 - ofj + j) * p->L[1] * gs[1]);
      if ((j + 1) % 8 == 0) fprintf(fp, "\n          ");
   }
   fprintf(fp, "\n        </DataArray>\n");
   /* Z */
   fprintf(fp, "        <DataArray type=\"Float64\" Name=\"z\" "
               "format=\"ascii\">\n          ");
   for (int k = 0; k < nzg; k++)
   {
      fprintf(fp, "%.15g ", (iz0 - ofk + k) * p->L[2] * gs[2]);
      if ((k + 1) % 8 == 0) fprintf(fp, "\n          ");
   }
   fprintf(fp, "\n        </DataArray>\n");
   fprintf(fp, "      </Coordinates>\n");

   fprintf(fp, "      <PointData Scalars=\"temperature\">\n");
   if (p->visualize & 0x2)
   {
      /* ASCII format (bit 1 set) */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"temperature\" "
                  "format=\"ascii\">\n          ");
      int cnt = 0;
      for (int k = 0; k < nzg; k++)
         for (int j = 0; j < nyg; j++)
            for (int i = 0; i < nxg; i++)
            {
               int idx = k * nyg * nxg + j * nxg + i;
               fprintf(fp, "%.15g ", ext_T[idx]);
               if ((++cnt) % 8 == 0) fprintf(fp, "\n          ");
            }
      fprintf(fp, "\n        </DataArray>\n");
      /* conductivity field */
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"conductivity\" "
                  "format=\"ascii\">\n          ");
      cnt = 0;
      for (int k = 0; k < nzg; k++)
         for (int j = 0; j < nyg; j++)
            for (int i = 0; i < nxg; i++)
            {
               int    idx = k * nyg * nxg + j * nxg + i;
               double kv  = p->k0 * safe_exp(p->beta * ext_T[idx]);
               fprintf(fp, "%.15g ", kv);
               if ((++cnt) % 8 == 0) fprintf(fp, "\n          ");
            }
      fprintf(fp, "\n        </DataArray>\n");
      /* Exact solution if requested */
      if (include_exact)
      {
         fprintf(fp, "        <DataArray type=\"Float64\" Name=\"temperature_exact\" "
                     "format=\"ascii\">\n          ");
         cnt = 0;
         for (int k = 0; k < nzg; k++)
            for (int j = 0; j < nyg; j++)
               for (int i = 0; i < nxg; i++)
               {
                  int idx = k * nyg * nxg + j * nxg + i;
                  fprintf(fp, "%.15g ", T_exact[idx]);
                  if ((++cnt) % 8 == 0) fprintf(fp, "\n          ");
               }
         fprintf(fp, "\n        </DataArray>\n");
      }
      /* Heat flux if requested */
      if (include_flux)
      {
         fprintf(fp, "        <DataArray type=\"Float64\" Name=\"heat_flux\" "
                     "NumberOfComponents=\"3\" format=\"ascii\">\n          ");
         cnt = 0;
         for (int k = 0; k < nzg; k++)
            for (int j = 0; j < nyg; j++)
               for (int i = 0; i < nxg; i++)
               {
                  int idx = k * nyg * nxg + j * nxg + i;
                  fprintf(fp, "%.15g %.15g %.15g ", qx[idx], qy[idx], qz[idx]);
                  if ((++cnt) % 3 == 0) fprintf(fp, "\n          ");
               }
         fprintf(fp, "\n        </DataArray>\n");
      }
      fprintf(fp, "      </PointData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </RectilinearGrid>\n");
      fprintf(fp, "</VTKFile>\n");
   }
   else
   {
      /* Binary format (bit 1 not set) */
      size_t bytes_T     = (size_t)npts * sizeof(double);
      size_t bytes_k     = (size_t)npts * sizeof(double);
      size_t bytes_exact = (size_t)npts * sizeof(double);
      size_t bytes_q     = (size_t)npts * 3 * sizeof(double);
      size_t off_T       = 0;
      size_t off_k       = sizeof(int) + bytes_T;
      size_t off_exact   = off_k + sizeof(int) + bytes_k;
      size_t off_q       = off_exact + (include_exact ? (sizeof(int) + bytes_exact) : 0);

      fprintf(fp,
              "        <DataArray type=\"Float64\" Name=\"temperature\" "
              "format=\"appended\" offset=\"%zu\">\n",
              off_T);
      fprintf(fp, "        </DataArray>\n");
      fprintf(fp,
              "        <DataArray type=\"Float64\" Name=\"conductivity\" "
              "format=\"appended\" offset=\"%zu\">\n",
              off_k);
      fprintf(fp, "        </DataArray>\n");
      if (include_exact)
      {
         fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"temperature_exact\" "
                 "format=\"appended\" offset=\"%zu\">\n",
                 off_exact);
         fprintf(fp, "        </DataArray>\n");
      }
      if (include_flux)
      {
         fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"heat_flux\" "
                 "NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\">\n",
                 off_q);
         fprintf(fp, "        </DataArray>\n");
      }
      fprintf(fp, "      </PointData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </RectilinearGrid>\n");
      fprintf(fp, "  <AppendedData encoding=\"raw\">\n   _");

      int len_T = (int)bytes_T;
      fwrite(&len_T, sizeof(int), 1, fp);
      for (int k = 0; k < nzg; k++)
         for (int j = 0; j < nyg; j++)
            for (int i = 0; i < nxg; i++)
            {
               int    idx = k * nyg * nxg + j * nxg + i;
               double tv  = ext_T[idx];
               fwrite(&tv, sizeof(double), 1, fp);
            }
      int len_k = (int)bytes_k;
      fwrite(&len_k, sizeof(int), 1, fp);
      for (int k = 0; k < nzg; k++)
         for (int j = 0; j < nyg; j++)
            for (int i = 0; i < nxg; i++)
            {
               int    idx = k * nyg * nxg + j * nxg + i;
               double kv  = p->k0 * safe_exp(p->beta * ext_T[idx]);
               fwrite(&kv, sizeof(double), 1, fp);
            }
      if (include_exact)
      {
         int len_exact = (int)bytes_exact;
         fwrite(&len_exact, sizeof(int), 1, fp);
         for (int k = 0; k < nzg; k++)
            for (int j = 0; j < nyg; j++)
               for (int i = 0; i < nxg; i++)
               {
                  int idx = k * nyg * nxg + j * nxg + i;
                  fwrite(&T_exact[idx], sizeof(double), 1, fp);
               }
      }
      if (include_flux)
      {
         int len_q = (int)bytes_q;
         fwrite(&len_q, sizeof(int), 1, fp);
         for (int k = 0; k < nzg; k++)
            for (int j = 0; j < nyg; j++)
               for (int i = 0; i < nxg; i++)
               {
                  int idx = k * nyg * nxg + j * nxg + i;
                  fwrite(&qx[idx], sizeof(double), 1, fp);
                  fwrite(&qy[idx], sizeof(double), 1, fp);
                  fwrite(&qz[idx], sizeof(double), 1, fp);
               }
      }
      fprintf(fp, "\n  </AppendedData>\n");
      fprintf(fp, "</VTKFile>\n");
   }
   fclose(fp);

   if (include_flux)
   {
      free(qx);
      free(qy);
      free(qz);
   }
   if (include_exact)
   {
      free(T_exact);
   }
   free(ext_T);

   return 0;
}

int
main(int argc, char *argv[])
{
   MPI_Comm       comm = MPI_COMM_WORLD;
   HYPREDRV_t     hypredrv;
   int            myid, nprocs;
   HeatParams     params;
   DistMesh      *mesh;
   GhostData3D   *g_T_k = NULL, *g_T_n = NULL;
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJVector vec_s[2];
   HYPRE_Real    *T_old;
   HYPRE_Real    *T_now;
   double         current_time, rnorm;
   int            newton_iter, t_step;
   double         energy_prev = NAN;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   if (ParseArguments(argc, argv, &params, myid, nprocs))
   {
      MPI_Finalize();
      return 1;
   }

   if (!myid)
   {
      printf("\n");
      printf("=====================================================\n");
      printf("       3D Nonlinear Heat Flow Problem Setup\n");
      printf("=====================================================\n");
      printf("Domain dimensions:       %.2f x %.2f x %.2f\n", params.L[0], params.L[1],
             params.L[2]);
      printf("Grid dimensions (nodes): %d x %d x %d\n", (int)params.N[0],
             (int)params.N[1], (int)params.N[2]);
      printf("Processor topology:      %d x %d x %d\n", (int)params.P[0],
             (int)params.P[1], (int)params.P[2]);
      printf("Material (rho, cp):      (%.2e, %.2e)\n", params.rho, params.cp);
      printf("Material (k0, beta):     (%.2e, %.2e)\n", params.k0, params.beta);
      /* Characteristic Fourier number (based on k0) using precomputed h_min */
      {
         HYPRE_Real Fo = params.alpha0 * params.dt / (params.hmin * params.hmin);
         printf("Fourier number (Fo):     %.2e\n", Fo);
      }
      printf("Boundary conditions:     T=0 at y=0, insulated elsewhere\n");
      printf("Time step:               %.2e\n", params.dt);
      printf("Final time:              %.2e\n", params.tf);
      printf("Adaptive time stepping:  %s\n", params.adaptive_dt ? "true" : "false");
      if (params.max_cfl > 0.0)
      {
         printf("Maximum CFL:             %.1f\n", params.max_cfl);
      }
      if (params.visualize)
      {
         const char *format    = (params.visualize & 0x2) ? "ASCII" : "binary";
         const char *timesteps = (params.visualize & 0x4) ? "all" : "last only";
         const char *flux      = (params.visualize & 0x8) ? ", with flux" : "";
         const char *exact     = (params.visualize & 0x10) ? ", with exact" : "";
         printf("Visualization:           enabled (%s, %s%s%s)\n", format, timesteps,
                flux, exact);
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
   if (params.verbose & 0x1)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   }
   if (params.verbose & 0x2)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));

   /* Configure solver using YAML input or default presets */
   if (params.yaml_file)
   {
      char *args[2] = {params.yaml_file, NULL};
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));
   }
   else
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetSolverPreset(hypredrv, "gmres"));
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(hypredrv, "poisson"));
   }

   /* Set HYPREDRV global options */
   HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(hypredrv));

   /* Create distributed mesh object */
   CreateDistMesh(comm, params.N[0], params.N[1], params.N[2], params.P[0], params.P[1],
                  params.P[2], &mesh);

   /* Create state vectors (scalar: 1 DOF per node) */
   for (int i = 0; i < 2; i++)
   {
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, mesh->ilower, mesh->iupper, &vec_s[i]);
      HYPRE_IJVectorSetObjectType(vec_s[i], HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(vec_s[i]);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorSet(hypredrv, 2, vec_s));

   /* Create ghost exchange objects */
   CreateGhostData3D(mesh, &g_T_k);
   CreateGhostData3D(mesh, &g_T_n);

   /* Initialize temperature field */
   HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 0, &T_now));
   InitializeTemperatureField(mesh, &params, T_now);
   ProjectDirichlet(mesh, &params, T_now);

   /* Copy initial state to vec_s[1] (T_old) */
   HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorCopy(hypredrv, 0, 1));

   /* Time stepping loop */
   for (current_time = 0, t_step = 0; current_time < params.tf; ++t_step)
   {
      if (current_time + params.dt > params.tf)
      {
         params.dt = params.tf - current_time;
      }
      current_time += params.dt;

      /* Begin timestep annotation (level 0) */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(0, "timestep", t_step));

      /* Initialize T_now with the solution from the previous time step */
      HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorCopy(hypredrv, 1, 0));

      /* Retrieve solution values from the old state (T^{n-1}) */
      HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 1, &T_old));
      ExchangeScalarGhosts(mesh, T_old, g_T_n);

      /* Newton iteration loop */
      for (newton_iter = 0; newton_iter < params.newton.max_iters; newton_iter++)
      {
         /* Begin Newton iteration annotation (level 1) */
         HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelBegin(1, "newton", newton_iter));

         /* Retrieve current iterate T^{k} */
         HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 0, &T_now));
         ProjectDirichlet(mesh, &params, T_now);
         ExchangeScalarGhosts(mesh, T_now, g_T_k);

         /* Assemble linear system: J dT = -R */
         BuildNonlinearSystem_Heat(mesh, &params, T_now, T_old, &A, &b, &rnorm,
                                   current_time, g_T_k, g_T_n);

         /* Check Newton convergence (after first iteration) */
         if (newton_iter > 0 && rnorm < params.newton.rtol)
         {
            HYPRE_IJMatrixDestroy(A);
            HYPRE_IJVectorDestroy(b);
            HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(1, "newton", newton_iter));
            break;
         }

#if defined(HYPRE_USING_GPU)
         if (!myid && (params.verbose & 0x2)) printf("Migrating linear system to GPU...");
         HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
         HYPRE_IJVectorMigrate(b, HYPRE_MEMORY_DEVICE);
         if (!myid && (params.verbose & 0x2)) printf(" Done!\n");
#endif

         /* Set up linear system in hypredrv */
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

         /* T = T + dT (correction applied to state vector 0) */
         HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorApplyCorrection(hypredrv));

         /* Report Newton iteration */
         if (params.verbose & 0x1)
         {
            /* Compute update norm ||dT||_inf BEFORE destroying solver or applying
             * correction */
            double delta_inf = 0.0;
            HYPREDRV_SAFE_CALL(
               HYPREDRV_LinearSystemGetSolutionNorm(hypredrv, "inf", &delta_inf));

            int num_iterations;
            HYPREDRV_SAFE_CALL(HYPREDRV_GetLastStat(hypredrv, "iter", &num_iterations));
            /* Compute energy and Fourier numbers for this (updated) iterate */
            HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 0, &T_now));
            double     energy_now = ComputeTotalEnergyLumped(mesh, &params, T_now);
            double     dE         = isnan(energy_prev) ? 0.0 : (energy_now - energy_prev);
            HYPRE_Real alpha0     = params.alpha0;
            HYPRE_Real hmin       = params.hmin;
            HYPRE_Real Fo         = alpha0 * params.dt / (hmin * hmin);
            /* MMS L2 error for current iterate */
            double L2_err_inline = 0.0, Linf_dummy = 0.0;
            ComputeMMSError(mesh, &params, T_now, current_time, &L2_err_inline,
                            &Linf_dummy);
            if (!myid)
            {
               printf("Time step: %3d | Time: %.4e [s] | Fo=%.2e | NL: %2d | Lin: %3d | "
                      "Linf(dT)=%.2e  L2(R)=%.2e  L2(Err)=%.6e | E=%.6e dE=%10.3e\n",
                      t_step + 1, current_time, Fo, newton_iter + 1, num_iterations,
                      delta_inf, rnorm, L2_err_inline, energy_now, dE);
            }
            energy_prev = energy_now;
         }

         /* Free linear system */
         HYPRE_IJMatrixDestroy(A);
         HYPRE_IJVectorDestroy(b);

         /* End Newton iteration annotation (level 1) */
         HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(1, "newton", newton_iter));
      }
      if (!myid && params.verbose > 0 && newton_iter > 1) printf("\n");

      /* Update timestep based on Newton convergence */
      UpdateTimeStep(&params, newton_iter);

      /* End timestep annotation (level 0) */
      HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateLevelEnd(0, "timestep", t_step));

      /* Save VTK output */
      if (params.visualize)
      {
         int is_last_timestep = (current_time >= params.tf - 1e-10);
         int write_all        = (params.visualize & 0x4) != 0;
         if (write_all || is_last_timestep)
         {
            HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorGetValues(hypredrv, 0, &T_now));
            ExchangeScalarGhosts(mesh, T_now,
                                 g_T_k); /* Ensure ghost data is fresh for VTK */
            WriteVTKsolutionScalar(mesh, &params, T_now, g_T_k, t_step, current_time);
         }
      }

      /* Update state vectors: T^{n} = T^{n+1} */
      HYPREDRV_SAFE_CALL(HYPREDRV_StateVectorUpdateAll(hypredrv));
   }

   /* Free memory */
   DestroyDistMesh(&mesh);
   DestroyGhostData3D(&g_T_k);
   DestroyGhostData3D(&g_T_n);
   for (int i = 0; i < 2; i++)
   {
      HYPRE_IJVectorDestroy(vec_s[i]);
   }

   /* Print timestep statistics summary */
   if (!myid && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsLevelPrint(0));
   }

   /* Write PVD collection file */
   if (!myid && params.visualize)
   {
      WritePVDCollectionFromStats(&params, nprocs, params.tf);
   }

   if (!myid && (params.verbose & 0x1)) HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
   if (params.verbose & 0x2) HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}
