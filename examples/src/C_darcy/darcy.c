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

typedef struct
{
   HYPRE_Int    n[3];
   HYPRE_Int    dim;
   HYPRE_Real   L[3];
   HYPRE_Real   h[3];
   HYPRE_Real   K[3];
   HYPRE_Real   Kinv[3];
   HYPRE_Real  *Kinv_cells;
   HYPRE_BigInt n_cells;
   HYPRE_BigInt n_x_faces;
   HYPRE_BigInt n_y_faces;
   HYPRE_BigInt n_z_faces;
   HYPRE_BigInt n_faces;
   HYPRE_BigInt n_total;
} DarcyMesh;

typedef struct
{
   HYPRE_Int  verbose;
   HYPRE_Int  n[3];
   HYPRE_Int  mpi_grid[3];
   HYPRE_Real L[3];
   HYPRE_Real K[3];
   HYPRE_Int  K_file_grid[3];
   HYPRE_Int  has_constant_K;
   HYPRE_Int  K_file_top_down;
   HYPRE_Int  drive_axis;
   HYPRE_Int  hypredrv_argc;
   char      *yaml_file;
   char      *K_file;
   char     **hypredrv_argv;
} DarcyParams;

typedef struct
{
   HYPRE_Int    P[3];
   HYPRE_Int   *x0, *x1, *y0, *y1, *z0, *z1;
   HYPRE_Int   *xpart, *ypart, *zpart;
   HYPRE_BigInt *offset;
   HYPRE_BigInt *count_xf, *count_yf, *count_zf, *count_cells, *total;
   HYPRE_BigInt N;
   int          nprocs;
} DarcyLayout;

typedef struct DarcyDiscretization DarcyDiscretization;
struct DarcyDiscretization
{
   const char *name;
   HYPRE_Int (*num_cell_flux_dofs)(const DarcyMesh *);
   void (*cell_flux_dofs)(const DarcyMesh *, HYPRE_BigInt, HYPRE_BigInt, HYPRE_BigInt,
                          HYPRE_BigInt *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);
   HYPRE_Real (*mass_entry)(const DarcyMesh *, HYPRE_BigInt, HYPRE_Int, HYPRE_Int,
                            const HYPRE_Int *, const HYPRE_Int *);
};

#define DEFAULT_MGR_CONFIG(statistics_value) \
   "general:\n"                              \
   "  statistics: " statistics_value "\n"    \
   "  exec_policy: host\n"                   \
   "linear_system:\n"                        \
   "  init_guess_mode: zeros\n"              \
   "solver:\n"                               \
   "  gmres:\n"                              \
   "    max_iter: 200\n"                     \
   "    krylov_dim: 60\n"                    \
   "    relative_tol: 1.0e-10\n"             \
   "    absolute_tol: 0.0\n"                 \
   "    print_level: 0\n"                    \
   "preconditioner:\n"                       \
   "  mgr:\n"                                \
   "    tolerance: 0.0\n"                    \
   "    max_iter: 1\n"                       \
   "    print_level: 0\n"                    \
   "    coarse_th: 0.0\n"                    \
   "    level:\n"                            \
   "      0:\n"                              \
   "        f_dofs: [1]\n"                   \
   "        f_relaxation: jacobi\n"          \
   "        g_relaxation: none\n"            \
   "        restriction_type: injection\n"   \
   "        prolongation_type: jacobi\n"     \
   "        coarse_level_type: rap\n"        \
   "    coarsest_level:\n"                   \
   "      amg:\n"                            \
   "        tolerance: 0.0\n"                \
   "        max_iter: 1\n"                   \
   "        print_level: 0\n"

#define LEGACY_HYPRE_CONFIG(statistics_value) \
   "general:\n"                               \
   "  statistics: " statistics_value "\n"     \
   "  exec_policy: host\n"                    \
   "linear_system:\n"                         \
   "  init_guess_mode: zeros\n"               \
   "solver:\n"                                \
   "  gmres:\n"                               \
   "    max_iter: 200\n"                      \
   "    krylov_dim: 60\n"                     \
   "    relative_tol: 1.0e-10\n"              \
   "    absolute_tol: 0.0\n"                  \
   "    print_level: 0\n"                     \
   "preconditioner: amg\n"

static const char *default_mgr_config        = DEFAULT_MGR_CONFIG("0");
static const char *default_mgr_stats_config  = DEFAULT_MGR_CONFIG("1");
static const char *legacy_hypre_config       = LEGACY_HYPRE_CONFIG("0");
static const char *legacy_hypre_stats_config = LEGACY_HYPRE_CONFIG("1");

static const char *
default_config(HYPRE_Int print_stats)
{
#if defined(HYPREDRV_HYPRE_RELEASE_NUMBER) && HYPREDRV_HYPRE_RELEASE_NUMBER < 30000
   return print_stats ? legacy_hypre_stats_config : legacy_hypre_config;
#else
   return print_stats ? default_mgr_stats_config : default_mgr_config;
#endif
}

static HYPRE_BigInt
prod3(const HYPRE_Int n[3])
{
   return (HYPRE_BigInt)n[0] * (HYPRE_BigInt)n[1] * (HYPRE_BigInt)n[2];
}

static HYPRE_Int
ceil_div_int(HYPRE_Int numerator, HYPRE_Int denominator)
{
   return (numerator + denominator - 1) / denominator;
}

static void
factor_mpi_grid(const DarcyMesh *mesh, int nprocs, HYPRE_Int parts[3])
{
   parts[0] = parts[1] = parts[2] = 1;

   int factors[64];
   int nfactors  = 0;
   int remaining = nprocs;
   for (int factor = 2; factor * factor <= remaining; factor++)
   {
      while (remaining % factor == 0 &&
             nfactors < (int)(sizeof(factors) / sizeof(factors[0])))
      {                                                                               \
         factors[nfactors++] = factor;
         remaining /= factor;
      }
   }
   if (remaining > 1 && nfactors < (int)(sizeof(factors) / sizeof(factors[0])))
   {
      factors[nfactors++] = remaining;
   }

   for (int f = nfactors - 1; f >= 0; f--)
   {
      HYPRE_Int best_dir       = 0;
      double    best_cell_span = -1.0;
      for (HYPRE_Int d = 0; d < mesh->dim; d++)
      {                                                                               \
         double cell_span = (double)mesh->n[d] / (double)parts[d];
         if (cell_span > best_cell_span)
         {                                                                            \
            best_cell_span = cell_span;
            best_dir       = d;
         }                                                                            \
      }
      parts[best_dir] *= factors[f];
   }
}

static int
layout_rank(const DarcyLayout *layout, HYPRE_Int rx, HYPRE_Int ry, HYPRE_Int rz)
{
   return rx + layout->P[0] * (ry + layout->P[1] * rz);
}

static void
layout_rank_coords(const DarcyLayout *layout, int rank, HYPRE_Int *rx, HYPRE_Int *ry,
                   HYPRE_Int *rz)
{
   *rx = rank % layout->P[0];
   *ry = (rank / layout->P[0]) % layout->P[1];
   *rz = rank / (layout->P[0] * layout->P[1]);
}

static int
layout_alloc_axis(HYPRE_Int n, HYPRE_Int parts, HYPRE_Int **starts_ptr,
                  HYPRE_Int **ends_ptr, HYPRE_Int **part_of_ptr)
{
   HYPRE_Int *starts  = (HYPRE_Int *)calloc((size_t)parts, sizeof(HYPRE_Int));
   HYPRE_Int *ends    = (HYPRE_Int *)calloc((size_t)parts, sizeof(HYPRE_Int));
   HYPRE_Int *part_of = (HYPRE_Int *)calloc((size_t)n, sizeof(HYPRE_Int));
   if (!starts || !ends || !part_of)
   {
      free(starts);
      free(ends);
      free(part_of);
      return 1;
   }

   HYPRE_Int base = n / parts;
   HYPRE_Int rem  = n - base * parts;
   HYPRE_Int s    = 0;
   for (HYPRE_Int p = 0; p < parts; p++)
   {
      HYPRE_Int len = base + (p < rem ? 1 : 0);
      if (len <= 0)
      {
         free(starts);
         free(ends);
         free(part_of);
         return 1;
      }
      starts[p] = s;
      ends[p]   = s + len;
      for (HYPRE_Int i = starts[p]; i < ends[p]; i++)
      {
         part_of[i] = p;
      }
      s += len;
   }

   *starts_ptr  = starts;
   *ends_ptr    = ends;
   *part_of_ptr = part_of;
   return 0;
}

static void
destroy_layout(DarcyLayout *layout)
{
   free(layout->x0);
   free(layout->x1);
   free(layout->y0);
   free(layout->y1);
   free(layout->z0);
   free(layout->z1);
   free(layout->xpart);
   free(layout->ypart);
   free(layout->zpart);
   free(layout->offset);
   free(layout->count_xf);
   free(layout->count_yf);
   free(layout->count_zf);
   free(layout->count_cells);
   free(layout->total);
   memset(layout, 0, sizeof(*layout));
}

static int
init_layout(const DarcyMesh *mesh, const HYPRE_Int P[3], DarcyLayout *layout)
{
   memset(layout, 0, sizeof(*layout));
   layout->P[0]  = P[0];
   layout->P[1]  = P[1];
   layout->P[2]  = P[2];
   layout->nprocs = P[0] * P[1] * P[2];

   if (layout_alloc_axis(mesh->n[0], P[0], &layout->x0, &layout->x1, &layout->xpart) ||
       layout_alloc_axis(mesh->n[1], P[1], &layout->y0, &layout->y1, &layout->ypart) ||
       layout_alloc_axis(mesh->n[2], P[2], &layout->z0, &layout->z1, &layout->zpart))
   {
      destroy_layout(layout);
      return 1;
   }

   size_t nprocs = (size_t)layout->nprocs;
   layout->offset      = (HYPRE_BigInt *)calloc(nprocs + 1, sizeof(HYPRE_BigInt));
   layout->count_xf    = (HYPRE_BigInt *)calloc(nprocs, sizeof(HYPRE_BigInt));
   layout->count_yf    = (HYPRE_BigInt *)calloc(nprocs, sizeof(HYPRE_BigInt));
   layout->count_zf    = (HYPRE_BigInt *)calloc(nprocs, sizeof(HYPRE_BigInt));
   layout->count_cells = (HYPRE_BigInt *)calloc(nprocs, sizeof(HYPRE_BigInt));
   layout->total       = (HYPRE_BigInt *)calloc(nprocs, sizeof(HYPRE_BigInt));
   if (!layout->offset || !layout->count_xf || !layout->count_yf || !layout->count_zf ||
       !layout->count_cells || !layout->total)
   {
      destroy_layout(layout);
      return 1;
   }

   for (int rank = 0; rank < layout->nprocs; rank++)
   {
      HYPRE_Int rx, ry, rz;
      layout_rank_coords(layout, rank, &rx, &ry, &rz);
      HYPRE_BigInt nx = layout->x1[rx] - layout->x0[rx];
      HYPRE_BigInt ny = layout->y1[ry] - layout->y0[ry];
      HYPRE_BigInt nz = layout->z1[rz] - layout->z0[rz];
      layout->count_xf[rank] = (nx + (rx == P[0] - 1 ? 1 : 0)) * ny * nz;
      if (mesh->dim >= 2)
      {
         layout->count_yf[rank] = nx * (ny + (ry == P[1] - 1 ? 1 : 0)) * nz;
      }
      if (mesh->dim >= 3)
      {
         layout->count_zf[rank] = nx * ny * (nz + (rz == P[2] - 1 ? 1 : 0));
      }
      layout->count_cells[rank] = nx * ny * nz;
      layout->total[rank] = layout->count_xf[rank] + layout->count_yf[rank] +
                            layout->count_zf[rank] + layout->count_cells[rank];
      layout->offset[rank + 1] = layout->offset[rank] + layout->total[rank];
   }
   layout->N = layout->offset[layout->nprocs];
   return layout->N == mesh->n_total ? 0 : 1;
}

static HYPRE_BigInt
layout_xface(const DarcyLayout *layout, const DarcyMesh *mesh, HYPRE_BigInt i,
             HYPRE_BigInt j, HYPRE_BigInt k)
{
   HYPRE_Int rx = (i == mesh->n[0]) ? layout->P[0] - 1 : layout->xpart[i];
   HYPRE_Int ry = layout->ypart[j];
   HYPRE_Int rz = layout->zpart[k];
   int       r  = layout_rank(layout, rx, ry, rz);
   HYPRE_BigInt nx = layout->x1[rx] - layout->x0[rx] + (rx == layout->P[0] - 1 ? 1 : 0);
   HYPRE_BigInt ny = layout->y1[ry] - layout->y0[ry];
   return layout->offset[r] + (i - layout->x0[rx]) +
          nx * ((j - layout->y0[ry]) + ny * (k - layout->z0[rz]));
}

static HYPRE_BigInt
layout_yface(const DarcyLayout *layout, const DarcyMesh *mesh, HYPRE_BigInt i,
             HYPRE_BigInt j, HYPRE_BigInt k)
{
   HYPRE_Int rx = layout->xpart[i];
   HYPRE_Int ry = (j == mesh->n[1]) ? layout->P[1] - 1 : layout->ypart[j];
   HYPRE_Int rz = layout->zpart[k];
   int       r  = layout_rank(layout, rx, ry, rz);
   HYPRE_BigInt nx = layout->x1[rx] - layout->x0[rx];
   HYPRE_BigInt ny = layout->y1[ry] - layout->y0[ry] + (ry == layout->P[1] - 1 ? 1 : 0);
   return layout->offset[r] + layout->count_xf[r] + (i - layout->x0[rx]) +
          nx * ((j - layout->y0[ry]) + ny * (k - layout->z0[rz]));
}

static HYPRE_BigInt
layout_zface(const DarcyLayout *layout, const DarcyMesh *mesh, HYPRE_BigInt i,
             HYPRE_BigInt j, HYPRE_BigInt k)
{
   HYPRE_Int rx = layout->xpart[i];
   HYPRE_Int ry = layout->ypart[j];
   HYPRE_Int rz = (k == mesh->n[2]) ? layout->P[2] - 1 : layout->zpart[k];
   int       r  = layout_rank(layout, rx, ry, rz);
   HYPRE_BigInt nx = layout->x1[rx] - layout->x0[rx];
   HYPRE_BigInt ny = layout->y1[ry] - layout->y0[ry];
   return layout->offset[r] + layout->count_xf[r] + layout->count_yf[r] +
          (i - layout->x0[rx]) + nx * ((j - layout->y0[ry]) +
                                       ny * (k - layout->z0[rz]));
}

static HYPRE_BigInt
layout_cell(const DarcyLayout *layout, const DarcyMesh *mesh, HYPRE_BigInt i,
            HYPRE_BigInt j, HYPRE_BigInt k)
{
   HYPRE_Int rx = layout->xpart[i];
   HYPRE_Int ry = layout->ypart[j];
   HYPRE_Int rz = layout->zpart[k];
   int       r  = layout_rank(layout, rx, ry, rz);
   HYPRE_BigInt nx = layout->x1[rx] - layout->x0[rx];
   HYPRE_BigInt ny = layout->y1[ry] - layout->y0[ry];
   return layout->offset[r] + layout->count_xf[r] + layout->count_yf[r] +
          layout->count_zf[r] + (i - layout->x0[rx]) +
          nx * ((j - layout->y0[ry]) + ny * (k - layout->z0[rz]));
}

static HYPRE_BigInt
cell_index(const DarcyMesh *mesh, HYPRE_BigInt i, HYPRE_BigInt j, HYPRE_BigInt k)
{
   return i + (HYPRE_BigInt)mesh->n[0] * (j + (HYPRE_BigInt)mesh->n[1] * k);
}

static void
cell_ijk(const DarcyMesh *mesh, HYPRE_BigInt cell, HYPRE_BigInt *i, HYPRE_BigInt *j,
         HYPRE_BigInt *k)
{
   *i = cell % mesh->n[0];
   *j = (cell / mesh->n[0]) % mesh->n[1];
   *k = cell / ((HYPRE_BigInt)mesh->n[0] * mesh->n[1]);
}

static HYPRE_BigInt
x_face_index(const DarcyMesh *mesh, HYPRE_BigInt i, HYPRE_BigInt j, HYPRE_BigInt k)
{
   return i + (HYPRE_BigInt)(mesh->n[0] + 1) * (j + (HYPRE_BigInt)mesh->n[1] * k);
}

static HYPRE_BigInt
y_face_index(const DarcyMesh *mesh, HYPRE_BigInt i, HYPRE_BigInt j, HYPRE_BigInt k)
{
   return mesh->n_x_faces + i +
          (HYPRE_BigInt)mesh->n[0] * (j + (HYPRE_BigInt)(mesh->n[1] + 1) * k);
}

static HYPRE_BigInt
z_face_index(const DarcyMesh *mesh, HYPRE_BigInt i, HYPRE_BigInt j, HYPRE_BigInt k)
{
   return mesh->n_x_faces + mesh->n_y_faces + i +
          (HYPRE_BigInt)mesh->n[0] * (j + (HYPRE_BigInt)mesh->n[1] * k);
}

static HYPRE_Real
face_area(const DarcyMesh *mesh, HYPRE_Int dir)
{
   if (dir == 0)
   {
      HYPRE_Real ay = mesh->n[1] > 1 ? mesh->h[1] : 1.0;
      HYPRE_Real az = mesh->n[2] > 1 ? mesh->h[2] : 1.0;
      return ay * az;
   }
   if (dir == 1)
   {
      HYPRE_Real az = mesh->n[2] > 1 ? mesh->h[2] : 1.0;
      return mesh->h[0] * az;
   }
   return mesh->h[0] * mesh->h[1];
}

static HYPRE_Real
cell_volume(const DarcyMesh *mesh)
{
   HYPRE_Real vy = mesh->n[1] > 1 ? mesh->h[1] : 1.0;
   HYPRE_Real vz = mesh->n[2] > 1 ? mesh->h[2] : 1.0;
   return mesh->h[0] * vy * vz;
}

static HYPRE_Int
rt0_num_cell_flux_dofs(const DarcyMesh *mesh)
{
   return 2 * mesh->dim;
}

static void
rt0_cell_flux_dofs(const DarcyMesh *mesh, HYPRE_BigInt i, HYPRE_BigInt j, HYPRE_BigInt k,
                   HYPRE_BigInt *faces, HYPRE_Int *dirs, HYPRE_Int *is_low,
                   HYPRE_Int *signs)
{
   faces[0] = x_face_index(mesh, i, j, k);
   faces[1] = x_face_index(mesh, i + 1, j, k);
   dirs[0] = dirs[1] = 0;
   is_low[0]         = 1;
   is_low[1]         = 0;
   signs[0]          = -1;
   signs[1]          = +1;
   if (mesh->dim >= 2)
   {
      faces[2] = y_face_index(mesh, i, j, k);
      faces[3] = y_face_index(mesh, i, j + 1, k);
      dirs[2] = dirs[3] = 1;
      is_low[2]         = 1;
      is_low[3]         = 0;
      signs[2]          = -1;
      signs[3]          = +1;
   }
   if (mesh->dim >= 3)
   {
      faces[4] = z_face_index(mesh, i, j, k);
      faces[5] = z_face_index(mesh, i, j, k + 1);
      dirs[4] = dirs[5] = 2;
      is_low[4]         = 1;
      is_low[5]         = 0;
      signs[4]          = -1;
      signs[5]          = +1;
   }
}

static void
rt0_cell_flux_dofs_layout(const DarcyLayout *layout, const DarcyMesh *mesh,
                          HYPRE_BigInt i, HYPRE_BigInt j, HYPRE_BigInt k,
                          HYPRE_BigInt *faces, HYPRE_Int *dirs, HYPRE_Int *is_low,
                          HYPRE_Int *signs)
{
   faces[0] = layout_xface(layout, mesh, i, j, k);
   faces[1] = layout_xface(layout, mesh, i + 1, j, k);
   dirs[0] = dirs[1] = 0;
   is_low[0]         = 1;
   is_low[1]         = 0;
   signs[0]          = -1;
   signs[1]          = +1;
   if (mesh->dim >= 2)
   {
      faces[2] = layout_yface(layout, mesh, i, j, k);
      faces[3] = layout_yface(layout, mesh, i, j + 1, k);
      dirs[2] = dirs[3] = 1;
      is_low[2]         = 1;
      is_low[3]         = 0;
      signs[2]          = -1;
      signs[3]          = +1;
   }
   if (mesh->dim >= 3)
   {
      faces[4] = layout_zface(layout, mesh, i, j, k);
      faces[5] = layout_zface(layout, mesh, i, j, k + 1);
      dirs[4] = dirs[5] = 2;
      is_low[4]         = 1;
      is_low[5]         = 0;
      signs[4]          = -1;
      signs[5]          = +1;
   }
}

static HYPRE_Real
cell_Kinv(const DarcyMesh *mesh, HYPRE_BigInt cell, HYPRE_Int dir)
{
   if (mesh->Kinv_cells)
   {
      return mesh->Kinv_cells[3 * (size_t)cell + (size_t)dir];
   }
   return mesh->Kinv[dir];
}

static HYPRE_Real
rt0_mass_entry(const DarcyMesh *mesh, HYPRE_BigInt cell, HYPRE_Int a, HYPRE_Int b,
               const HYPRE_Int *dirs, const HYPRE_Int *is_low)
{
   HYPRE_Int da = dirs[a];
   HYPRE_Int db = dirs[b];
   if (da != db)
   {
      return 0.0;
   }
   HYPRE_Real coef = is_low[a] == is_low[b] ? 1.0 / 3.0 : 1.0 / 6.0;
   return cell_Kinv(mesh, cell, da) * cell_volume(mesh) * coef /
          (face_area(mesh, da) * face_area(mesh, db));
}

static const DarcyDiscretization rt0_discretization = {
   "RT0/P0", rt0_num_cell_flux_dofs, rt0_cell_flux_dofs, rt0_mass_entry};

static int
is_pinned_neumann_physical(const DarcyMesh *mesh, HYPRE_Int axis, HYPRE_BigInt i,
                           HYPRE_BigInt j, HYPRE_BigInt k, HYPRE_Int drive_axis)
{
   (void)i;
   HYPRE_Int low = 0, high = 0;
   if (axis == 0)
   {
      low  = (i == 0);
      high = (i == mesh->n[0]);
   }
   else if (axis == 1)
   {
      low  = (j == 0);
      high = (j == mesh->n[1]);
   }
   else
   {
      low  = (k == 0);
      high = (k == mesh->n[2]);
   }
   return (low || high) && axis != drive_axis;
}

static HYPRE_Real
dirichlet_rhs_physical(const DarcyMesh *mesh, HYPRE_Int axis, HYPRE_BigInt i,
                       HYPRE_BigInt j, HYPRE_BigInt k, HYPRE_Int drive_axis)
{
   (void)j;
   (void)k;
   if (axis != drive_axis)
   {
      return 0.0;
   }
   if (axis == 0)
   {
      return i == 0 ? 1.0 : 0.0;
   }
   if (axis == 1)
   {
      return j == 0 ? 1.0 : 0.0;
   }
   return k == 0 ? 1.0 : 0.0;
}

static void
append_or_accumulate_entry(HYPRE_BigInt *cols, HYPRE_Real *vals, HYPRE_Int *nentries,
                           HYPRE_BigInt col, HYPRE_Real val)
{
   for (HYPRE_Int entry = 0; entry < *nentries; entry++)
   {
      if (cols[entry] == col)
      {
         vals[entry] += val;
         return;
      }
   }
   cols[*nentries]   = col;
   vals[*nentries]   = val;
   (*nentries)++;
}

static int
init_mesh(const DarcyParams *params, DarcyMesh *mesh)
{
   memset(mesh, 0, sizeof(*mesh));
   for (int d = 0; d < 3; d++)
   {
      mesh->n[d]    = params->n[d];
      mesh->L[d]    = params->L[d];
      mesh->h[d]    = params->L[d] / params->n[d];
      mesh->K[d]    = params->K[d];
      mesh->Kinv[d] = 1.0 / params->K[d];
   }
   mesh->dim       = (mesh->n[0] > 1) + (mesh->n[1] > 1) + (mesh->n[2] > 1);
   mesh->n_cells   = prod3(mesh->n);
   mesh->n_x_faces = (HYPRE_BigInt)(mesh->n[0] + 1) * mesh->n[1] * mesh->n[2];
   mesh->n_y_faces =
      mesh->dim >= 2 ? (HYPRE_BigInt)mesh->n[0] * (mesh->n[1] + 1) * mesh->n[2] : 0;
   mesh->n_z_faces =
      mesh->dim >= 3 ? (HYPRE_BigInt)mesh->n[0] * mesh->n[1] * (mesh->n[2] + 1) : 0;
   mesh->n_faces = mesh->n_x_faces + mesh->n_y_faces + mesh->n_z_faces;
   mesh->n_total = mesh->n_faces + mesh->n_cells;
   return 0;
}

static void
destroy_mesh(DarcyMesh *mesh)
{
   free(mesh->Kinv_cells);
   mesh->Kinv_cells = NULL;
}

static void
set_default_params(DarcyParams *params)
{
   params->verbose         = 1;
   params->n[0]            = 8;
   params->n[1]            = 8;
   params->n[2]            = 1;
   params->mpi_grid[0]     = 0;
   params->mpi_grid[1]     = 0;
   params->mpi_grid[2]     = 0;
   params->L[0]            = 1.0;
   params->L[1]            = 1.0;
   params->L[2]            = 1.0;
   params->K[0]            = 1.0;
   params->K[1]            = 1.0;
   params->K[2]            = 1.0;
   params->K_file_grid[0]  = 0;
   params->K_file_grid[1]  = 0;
   params->K_file_grid[2]  = 0;
   params->has_constant_K  = 0;
   params->K_file_top_down = 0;
   params->drive_axis      = 0;
   params->hypredrv_argc   = 0;
   params->yaml_file       = NULL;
   params->K_file          = NULL;
   params->hypredrv_argv   = NULL;
}

static int
print_usage(void)
{
   printf("\n");
   printf("Usage:\n");
   printf("  mpirun -np <ranks> ./darcy [options]\n\n");
   printf("Solves an RT0/P0 mixed Darcy problem on a prefix-active Cartesian mesh.\n");
   printf("Active dimensions must be x, x-y, or x-y-z.\n\n");
   printf("Active mesh examples:\n");
   printf("  -n 30 1 1       1D x-line\n");
   printf("  -n 30 110 1     2D x-y plane\n");
   printf("  -n 30 110 85    3D volume\n\n");
   printf("Options:\n");
   printf("  -i, --input <file>               YAML solver/preconditioner file\n");
   printf(
      "  -a, --args <key value>...        Override hypredrive YAML options; must be\n");
   printf("                                   the last Darcy option\n");
   printf("  -n <nx> <ny> <nz>                Cell counts; default: 8 8 1\n");
   printf("  -P, --procs <px> <py> <pz>       MPI rank grid; product must equal -np;\n");
   printf("                                   inactive dimensions must be 1\n");
   printf("  -L <Lx> <Ly> <Lz>                Domain lengths; default: 1 1 1\n");
   printf("  -K <Kx> <Ky> <Kz>                Constant diagonal permeability; default: 1 "
          "1 1\n");
   printf("  --K-file <path>                  Permeability text file; one value per "
          "source\n");
   printf("                                   cell, or three blocks Kx Ky Kz\n");
   printf("  --K-file-grid <nx> <ny> <nz>     Source grid for --K-file; omit when it\n");
   printf("                                   matches -n\n");
   printf("  --K-file-k-order <order>         Layer order: bottom-up or top-down;\n");
   printf("                                   default: bottom-up\n");
   printf("  -g, --gradient-direction <axis>  Pressure-drop direction: x, y, or z;\n");
   printf("                                   default: x\n");
   printf("  -v, --verbose <bits>             Verbosity bitset; default: 1\n");
   printf(
      "                                     0x1  library info and solver statistics\n");
   printf("                                     0x2  system and platform info\n");
   printf("                                     0x4  print assembled linear system\n");
   printf("  -h, --help                       Print this help and exit\n\n");
   printf("Notes:\n");
   printf("  -K and --K-file are mutually exclusive.\n");
   printf("  Hypredrive overrides after -a use path syntax, e.g.\n");
   printf("  --solver:gmres:max_iter 100.\n");
   printf("  Heterogeneous runs skip the analytic pressure-error check.\n\n");
   printf("Examples:\n");
   printf("  mpirun -np 1 ./darcy -n 4 3 1 -g x -v 1\n");
   printf("  mpirun -np 2 ./darcy -n 30 110 85 \\\n");
   printf("    -P 1 1 2 \\\n");
   printf("    --K-file data/spe10_case2a/spe_perm.dat \\\n");
   printf("    --K-file-grid 60 220 85 --K-file-k-order top-down -g x -v 1\n\n");
   printf("  mpirun -np 2 ./darcy -n 30 110 85 -v 1 \\\n");
   printf("    -a --solver:gmres:max_iter 100 --preconditioner:mgr:print_level 1\n\n");
   return 0;
}

static int
parse_args(int argc, char **argv, DarcyParams *params, int rank, int nprocs)
{
   set_default_params(params);
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
            if (!rank) printf("Error: -n requires three values\n");
            return 1;
         }
         for (int d = 0; d < 3; d++) params->n[d] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-P") || !strcmp(argv[i], "--procs"))
      {
         if (i + 3 >= argc)
         {
            if (!rank) printf("Error: -P requires three values\n");
            return 1;
         }
         for (int d = 0; d < 3; d++) params->mpi_grid[d] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-L"))
      {
         if (i + 3 >= argc)
         {
            if (!rank) printf("Error: -L requires three values\n");
            return 1;
         }
         for (int d = 0; d < 3; d++) params->L[d] = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-K"))
      {
         if (i + 3 >= argc)
         {
            if (!rank) printf("Error: -K requires three values\n");
            return 1;
         }
         for (int d = 0; d < 3; d++) params->K[d] = atof(argv[++i]);
         params->has_constant_K = 1;
      }
      else if (!strcmp(argv[i], "--K-file"))
      {
         if (++i >= argc)
         {
            if (!rank) printf("Error: --K-file requires a path\n");
            return 1;
         }
         params->K_file = argv[i];
      }
      else if (!strcmp(argv[i], "--K-file-grid"))
      {
         if (i + 3 >= argc)
         {
            if (!rank) printf("Error: --K-file-grid requires three values\n");
            return 1;
         }
         for (int d = 0; d < 3; d++) params->K_file_grid[d] = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "--K-file-k-order"))
      {
         if (++i >= argc)
         {
            if (!rank) printf("Error: --K-file-k-order requires bottom-up or top-down\n");
            return 1;
         }
         if (!strcmp(argv[i], "bottom-up"))
         {
            params->K_file_top_down = 0;
         }
         else if (!strcmp(argv[i], "top-down"))
         {
            params->K_file_top_down = 1;
         }
         else
         {
            if (!rank) printf("Error: --K-file-k-order requires bottom-up or top-down\n");
            return 1;
         }
      }
      else if (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--gradient-direction"))
      {
         if (++i >= argc)
         {
            if (!rank) printf("Error: -g requires x, y, or z\n");
            return 1;
         }
         params->drive_axis = (!strcmp(argv[i], "x"))   ? 0
                              : (!strcmp(argv[i], "y")) ? 1
                              : (!strcmp(argv[i], "z")) ? 2
                                                        : -1;
      }
      else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose"))
      {
         if (++i < argc) params->verbose = atoi(argv[i]);
      }
      else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         if (!rank) print_usage();
         return 2;
      }
   }

   if (params->drive_axis < 0 || params->drive_axis > 2)
   {
      if (!rank) printf("Error: -g must be x, y, or z\n");
      return 1;
   }
   if (params->n[0] < 1 || params->n[1] < 1 || params->n[2] < 1)
   {
      if (!rank) printf("Error: all cell counts must be positive\n");
      return 1;
   }
   if (params->n[0] <= 1 && (params->n[1] > 1 || params->n[2] > 1))
   {
      if (!rank) printf("Error: active dimensions must be a prefix of x,y,z\n");
      return 1;
   }
   if (params->n[1] <= 1 && params->n[2] > 1)
   {
      if (!rank) printf("Error: active dimensions must be a prefix of x,y,z\n");
      return 1;
   }
   if (params->n[0] <= 1 && params->n[1] <= 1 && params->n[2] <= 1)
   {
      if (!rank) printf("Error: at least one cell count must exceed 1\n");
      return 1;
   }
   HYPRE_Int dim = (params->n[0] > 1) + (params->n[1] > 1) + (params->n[2] > 1);
   if (params->drive_axis >= dim)
   {
      if (!rank) printf("Error: pressure-drop direction is not active for this mesh\n");
      return 1;
   }
   int has_mpi_grid =
      params->mpi_grid[0] || params->mpi_grid[1] || params->mpi_grid[2];
   if (has_mpi_grid)
   {
      HYPRE_BigInt mpi_grid_product = 1;
      for (int d = 0; d < 3; d++)
      {
         if (params->mpi_grid[d] <= 0)
         {
            if (!rank) printf("Error: -P entries must be positive\n");
            return 1;
         }
         if (params->n[d] == 1 && params->mpi_grid[d] != 1)
         {
            if (!rank) printf("Error: cannot partition inactive dimension %c\n", "xyz"[d]);
            return 1;
         }
         mpi_grid_product *= params->mpi_grid[d];
      }
      if (mpi_grid_product != nprocs)
      {
         if (!rank)
         {
            printf("Error: -P product must equal MPI ranks (%lld != %d)\n",
                   (long long)mpi_grid_product, nprocs);
         }
         return 1;
      }
   }
   for (int d = 0; d < 3; d++)
   {
      if (params->L[d] <= 0.0 || params->K[d] <= 0.0)
      {
         if (!rank) printf("Error: lengths and permeability entries must be positive\n");
         return 1;
      }
   }
   if (params->K_file && params->has_constant_K)
   {
      if (!rank) printf("Error: -K and --K-file are mutually exclusive\n");
      return 1;
   }
   if (!params->K_file &&
       (params->K_file_grid[0] || params->K_file_grid[1] || params->K_file_grid[2]))
   {
      if (!rank) printf("Error: --K-file-grid requires --K-file\n");
      return 1;
   }
   if (!params->K_file && params->K_file_top_down)
   {
      if (!rank) printf("Error: --K-file-k-order requires --K-file\n");
      return 1;
   }
   for (int d = 0; d < 3; d++)
   {
      if (params->K_file_grid[d] < 0)
      {
         if (!rank) printf("Error: --K-file-grid entries must be nonnegative\n");
         return 1;
      }
   }
   if (params->K_file &&
       ((params->K_file_grid[0] == 0) != (params->K_file_grid[1] == 0) ||
        (params->K_file_grid[0] == 0) != (params->K_file_grid[2] == 0)))
   {
      if (!rank) printf("Error: --K-file-grid must provide three positive values\n");
      return 1;
   }
   return 0;
}

static int
append_real(HYPRE_Real **values_ptr, size_t *count_ptr, size_t *capacity_ptr,
            HYPRE_Real value)
{
   if (*count_ptr == *capacity_ptr)
   {
      size_t      next_capacity = *capacity_ptr ? 2 * *capacity_ptr : 4096;
      HYPRE_Real *next_values =
         (HYPRE_Real *)realloc(*values_ptr, next_capacity * sizeof(HYPRE_Real));
      if (!next_values)
      {
         return 1;
      }
      *values_ptr   = next_values;
      *capacity_ptr = next_capacity;
   }
   (*values_ptr)[(*count_ptr)++] = value;
   return 0;
}

static int
read_permeability_values(const char *path, HYPRE_Real **values_ptr, size_t *count_ptr,
                         int rank)
{
   FILE *fp = fopen(path, "r");
   if (!fp)
   {
      if (!rank) printf("Error: failed to open permeability file '%s'\n", path);
      return 1;
   }

   HYPRE_Real *values   = NULL;
   size_t      count    = 0;
   size_t      capacity = 0;
   for (;;)
   {
      double value = 0.0;
      int    rc    = fscanf(fp, "%lf", &value);
      if (rc == EOF)
      {
         break;
      }
      if (rc != 1 || !isfinite(value))
      {
         if (!rank) printf("Error: invalid value in permeability file '%s'\n", path);
         free(values);
         fclose(fp);
         return 1;
      }
      if (append_real(&values, &count, &capacity, (HYPRE_Real)value))
      {
         if (!rank) printf("Error: failed to allocate permeability values\n");
         free(values);
         fclose(fp);
         return 1;
      }
   }
   fclose(fp);

   *values_ptr = values;
   *count_ptr  = count;
   return 0;
}

static HYPRE_BigInt
nearest_source_index(HYPRE_BigInt dst, HYPRE_BigInt ndst, HYPRE_BigInt nsrc)
{
   HYPRE_BigInt src = ((2 * dst + 1) * nsrc) / (2 * ndst);
   return src < nsrc ? src : nsrc - 1;
}

static int
load_permeability_file(const DarcyParams *params, DarcyMesh *mesh, int rank)
{
   if (!params->K_file)
   {
      return 0;
   }
   if (mesh->n_cells <= 0 || (HYPRE_BigInt)(size_t)mesh->n_cells != mesh->n_cells)
   {
      if (!rank) printf("Error: mesh is too large for permeability file storage\n");
      return 1;
   }

   HYPRE_Int source_n[3] = {params->K_file_grid[0], params->K_file_grid[1],
                            params->K_file_grid[2]};
   if (source_n[0] == 0)
   {
      source_n[0] = mesh->n[0];
      source_n[1] = mesh->n[1];
      source_n[2] = mesh->n[2];
   }
   for (int d = 0; d < 3; d++)
   {
      if (source_n[d] <= 0)
      {
         if (!rank) printf("Error: permeability source grid entries must be positive\n");
         return 1;
      }
   }

   HYPRE_BigInt source_cells =
      (HYPRE_BigInt)source_n[0] * (HYPRE_BigInt)source_n[1] * (HYPRE_BigInt)source_n[2];
   if ((HYPRE_BigInt)(size_t)source_cells != source_cells)
   {
      if (!rank) printf("Error: permeability source grid is too large\n");
      return 1;
   }

   HYPRE_Real *values = NULL;
   size_t      count  = 0;
   if (read_permeability_values(params->K_file, &values, &count, rank))
   {
      return 1;
   }

   size_t source_count = (size_t)source_cells;
   int    ncomponents  = 0;
   if (count == source_count)
   {
      ncomponents = 1;
   }
   else if (source_count <= (size_t)-1 / 3 && count == 3 * source_count)
   {
      ncomponents = 3;
   }
   else
   {
      if (!rank)
      {
         printf("Error: permeability file '%s' has %zu values; expected %zu or %zu\n",
                params->K_file, count, source_count, 3 * source_count);
      }
      free(values);
      return 1;
   }

   size_t n_cells = (size_t)mesh->n_cells;
   if (n_cells > (size_t)-1 / (3 * sizeof(HYPRE_Real)))
   {
      if (!rank) printf("Error: mesh is too large for permeability file storage\n");
      free(values);
      return 1;
   }
   mesh->Kinv_cells = (HYPRE_Real *)calloc(3 * n_cells, sizeof(HYPRE_Real));
   if (!mesh->Kinv_cells)
   {
      if (!rank) printf("Error: failed to allocate per-cell permeability field\n");
      free(values);
      return 1;
   }

   for (HYPRE_BigInt k = 0; k < mesh->n[2]; k++)
   {
      HYPRE_BigInt sk_physical = nearest_source_index(k, mesh->n[2], source_n[2]);
      HYPRE_BigInt sk_file     = params->K_file_top_down
                                    ? (HYPRE_BigInt)source_n[2] - 1 - sk_physical
                                    : sk_physical;
      for (HYPRE_BigInt j = 0; j < mesh->n[1]; j++)
      {
         HYPRE_BigInt sj = nearest_source_index(j, mesh->n[1], source_n[1]);
         for (HYPRE_BigInt i = 0; i < mesh->n[0]; i++)
         {
            HYPRE_BigInt si = nearest_source_index(i, mesh->n[0], source_n[0]);
            size_t       src_cell =
               (size_t)(si + (HYPRE_BigInt)source_n[0] *
                                (sj + (HYPRE_BigInt)source_n[1] * sk_file));
            size_t dst_cell = (size_t)cell_index(mesh, i, j, k);
            for (int d = 0; d < 3; d++)
            {
               HYPRE_Real K =
                  values[(ncomponents == 1 ? 0 : (size_t)d * source_count) + src_cell];
               if (K <= 0.0 || !isfinite(K))
               {
                  if (!rank)
                  {
                     printf("Error: permeability values must be positive and finite\n");
                  }
                  free(values);
                  destroy_mesh(mesh);
                  return 1;
               }
               mesh->Kinv_cells[3 * dst_cell + (size_t)d] = 1.0 / K;
            }
         }
      }
   }
   free(values);
   return 0;
}

static int
build_system_csr(MPI_Comm comm, const DarcyMesh *mesh, const DarcyLayout *layout,
                 const DarcyDiscretization *disc, HYPRE_Int drive_axis,
                 HYPRE_BigInt **indptr_ptr,
                 HYPRE_BigInt **col_indices_ptr, HYPRE_Real **data_ptr,
                 HYPRE_Real **rhs_ptr, int **dofmap_ptr, HYPRE_BigInt *ilower_ptr,
                 HYPRE_BigInt *iupper_ptr)
{
   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   HYPRE_BigInt ilower     = layout->offset[rank];
   HYPRE_BigInt local_rows = layout->total[rank];
   HYPRE_BigInt iupper     = ilower + local_rows - 1;
   if (local_rows <= 0)
   {
      if (!rank) printf("Error: more ranks than unknowns are not supported\n");
      return 1;
   }
   if ((HYPRE_BigInt)((int)local_rows) != local_rows)
   {
      if (!rank) printf("Error: local row count is too large for dofmap API\n");
      return 1;
   }

   int        *dofmap   = (int *)calloc((size_t)local_rows, sizeof(int));
   HYPRE_Real *rhs      = (HYPRE_Real *)calloc((size_t)local_rows, sizeof(HYPRE_Real));
   HYPRE_BigInt *indptr =
      (HYPRE_BigInt *)calloc((size_t)local_rows + 1, sizeof(HYPRE_BigInt));
   if (!dofmap || !rhs || !indptr)
   {
      free(dofmap);
      free(rhs);
      free(indptr);
      return 1;
   }

   HYPRE_Int nloc = disc->num_cell_flux_dofs(mesh);
   HYPRE_BigInt nnz_capacity = local_rows * (2 * (nloc + 1));
   if (nnz_capacity < 0 || (HYPRE_BigInt)(size_t)nnz_capacity != nnz_capacity)
   {
      if (!rank) printf("Error: local nonzero count is too large\n");
      free(dofmap);
      free(rhs);
      free(indptr);
      return 1;
   }

   HYPRE_BigInt *col_indices =
      (HYPRE_BigInt *)malloc((size_t)nnz_capacity * sizeof(HYPRE_BigInt));
   HYPRE_Real *data = (HYPRE_Real *)malloc((size_t)nnz_capacity * sizeof(HYPRE_Real));
   if ((nnz_capacity && (!col_indices || !data)))
   {
      if (!rank) printf("Error: failed to allocate local CSR matrix\n");
      free(dofmap);
      free(rhs);
      free(indptr);
      free(col_indices);
      free(data);
      return 1;
   }

   HYPRE_BigInt faces[6];
   HYPRE_Int    dirs[6], is_low[6], signs[6];
   HYPRE_BigInt cols[14];
   HYPRE_Real   vals[14];
   HYPRE_BigInt nnz = 0;
   HYPRE_BigInt lr  = 0;
   HYPRE_Int    rx, ry, rz;
   layout_rank_coords(layout, rank, &rx, &ry, &rz);

#define DARCY_APPEND_ROW()                                                     \
   do                                                                          \
   {                                                                           \
      if (nnz + nentries > nnz_capacity)                                       \
      {                                                                         \
         if (!rank) printf("Error: local CSR capacity was underestimated\n");   \
         free(dofmap);                                                         \
         free(rhs);                                                            \
         free(indptr);                                                         \
         free(col_indices);                                                    \
         free(data);                                                           \
         return 1;                                                             \
      }                                                                        \
      for (HYPRE_Int entry = 0; entry < nentries; entry++)                     \
      {                                                                         \
         col_indices[nnz] = cols[entry];                                       \
         data[nnz]        = vals[entry];                                       \
         nnz++;                                                                \
      }                                                                        \
      indptr[lr + 1] = nnz;                                                    \
      lr++;                                                                    \
   } while (0)

   HYPRE_Int x_start = layout->x0[rx];
   HYPRE_Int x_end   = layout->x1[rx];
   HYPRE_Int y_start = layout->y0[ry];
   HYPRE_Int y_end   = layout->y1[ry];
   HYPRE_Int z_start = layout->z0[rz];
   HYPRE_Int z_end   = layout->z1[rz];

#define DARCY_ADD_FLUX_ADJ_CELL(cell_i, cell_j, cell_k, local_face)            \
   do                                                                          \
   {                                                                           \
      HYPRE_BigInt c = cell_index(mesh, (cell_i), (cell_j), (cell_k));         \
      rt0_cell_flux_dofs_layout(layout, mesh, (cell_i), (cell_j), (cell_k),    \
                                faces, dirs, is_low, signs);                  \
      HYPRE_Int a = (local_face);                                              \
      for (HYPRE_Int bb = 0; bb < nloc; bb++)                                  \
      {                                                                         \
         HYPRE_Real m = disc->mass_entry(mesh, c, a, bb, dirs, is_low);        \
         if (m != 0.0)                                                         \
         {                                                                      \
            append_or_accumulate_entry(cols, vals, &nentries, faces[bb], m);   \
         }                                                                      \
      }                                                                        \
      append_or_accumulate_entry(cols, vals, &nentries,                        \
                                 layout_cell(layout, mesh, (cell_i), (cell_j), \
                                             (cell_k)),                        \
                                 -(HYPRE_Real)signs[a]);                       \
   } while (0)

   for (HYPRE_BigInt k = z_start; k < z_end; k++)
   {
      for (HYPRE_BigInt j = y_start; j < y_end; j++)
      {
         HYPRE_BigInt x_face_end = x_end + (rx == layout->P[0] - 1 ? 1 : 0);
         for (HYPRE_BigInt i = x_start; i < x_face_end; i++)
         {
            HYPRE_BigInt row = layout_xface(layout, mesh, i, j, k);
            HYPRE_Int    nentries = 0;
            dofmap[lr] = 1;
            rhs[lr]    = dirichlet_rhs_physical(mesh, 0, i, j, k, drive_axis);
            if (is_pinned_neumann_physical(mesh, 0, i, j, k, drive_axis))
            {
               cols[nentries]   = row;
               vals[nentries++] = 1.0;
            }
            else
            {
               if (i > 0) DARCY_ADD_FLUX_ADJ_CELL(i - 1, j, k, 1);
               if (i < mesh->n[0]) DARCY_ADD_FLUX_ADJ_CELL(i, j, k, 0);
            }
            DARCY_APPEND_ROW();
         }
      }
   }

   if (mesh->dim >= 2)
   {
      for (HYPRE_BigInt k = z_start; k < z_end; k++)
      {
         HYPRE_BigInt y_face_end = y_end + (ry == layout->P[1] - 1 ? 1 : 0);
         for (HYPRE_BigInt j = y_start; j < y_face_end; j++)
         {
            for (HYPRE_BigInt i = x_start; i < x_end; i++)
            {
               HYPRE_BigInt row = layout_yface(layout, mesh, i, j, k);
               HYPRE_Int    nentries = 0;
               dofmap[lr] = 1;
               rhs[lr]    = dirichlet_rhs_physical(mesh, 1, i, j, k, drive_axis);
               if (is_pinned_neumann_physical(mesh, 1, i, j, k, drive_axis))
               {
                  cols[nentries]   = row;
                  vals[nentries++] = 1.0;
               }
               else
               {
                  if (j > 0) DARCY_ADD_FLUX_ADJ_CELL(i, j - 1, k, 3);
                  if (j < mesh->n[1]) DARCY_ADD_FLUX_ADJ_CELL(i, j, k, 2);
               }
               DARCY_APPEND_ROW();
            }
         }
      }
   }

   if (mesh->dim >= 3)
   {
      HYPRE_BigInt z_face_end = z_end + (rz == layout->P[2] - 1 ? 1 : 0);
      for (HYPRE_BigInt k = z_start; k < z_face_end; k++)
      {
         for (HYPRE_BigInt j = y_start; j < y_end; j++)
         {
            for (HYPRE_BigInt i = x_start; i < x_end; i++)
            {
               HYPRE_BigInt row = layout_zface(layout, mesh, i, j, k);
               HYPRE_Int    nentries = 0;
               dofmap[lr] = 1;
               rhs[lr]    = dirichlet_rhs_physical(mesh, 2, i, j, k, drive_axis);
               if (is_pinned_neumann_physical(mesh, 2, i, j, k, drive_axis))
               {
                  cols[nentries]   = row;
                  vals[nentries++] = 1.0;
               }
               else
               {
                  if (k > 0) DARCY_ADD_FLUX_ADJ_CELL(i, j, k - 1, 5);
                  if (k < mesh->n[2]) DARCY_ADD_FLUX_ADJ_CELL(i, j, k, 4);
               }
               DARCY_APPEND_ROW();
            }
         }
      }
   }

   for (HYPRE_BigInt k = z_start; k < z_end; k++)
   {
      for (HYPRE_BigInt j = y_start; j < y_end; j++)
      {
         for (HYPRE_BigInt i = x_start; i < x_end; i++)
         {
            HYPRE_BigInt c = cell_index(mesh, i, j, k);
            HYPRE_Int    nentries = 0;
            dofmap[lr] = 0;
            rhs[lr]    = 0.0;
            rt0_cell_flux_dofs_layout(layout, mesh, i, j, k, faces, dirs, is_low, signs);
            for (HYPRE_Int a = 0; a < nloc; a++)
            {
               cols[nentries]   = faces[a];
               vals[nentries++] = -(HYPRE_Real)signs[a];
            }
            (void)c;
            DARCY_APPEND_ROW();
         }
      }
   }

#undef DARCY_ADD_FLUX_ADJ_CELL
#undef DARCY_APPEND_ROW

   if (lr != local_rows)
   {
      if (!rank)
      {
         printf("Error: layout row count mismatch (%lld != %lld)\n", (long long)lr,
                (long long)local_rows);
      }
      free(dofmap);
      free(rhs);
      free(indptr);
      free(col_indices);
      free(data);
      return 1;
   }

   *indptr_ptr      = indptr;
   *col_indices_ptr = col_indices;
   *data_ptr        = data;
   *rhs_ptr         = rhs;
   *dofmap_ptr      = dofmap;
   *ilower_ptr      = ilower;
   *iupper_ptr      = iupper;
   return 0;
}

static HYPRE_Real
pressure_l2_error(MPI_Comm comm, const DarcyMesh *mesh, const DarcyLayout *layout,
                  HYPRE_Int drive_axis, HYPRE_BigInt ilower, const HYPRE_Real *x)
{
   int rank;
   MPI_Comm_rank(comm, &rank);
   HYPRE_Int rx, ry, rz;
   layout_rank_coords(layout, rank, &rx, &ry, &rz);

   HYPRE_Real local_err = 0.0;
   HYPRE_Real local_ref = 0.0;
   for (HYPRE_BigInt k = layout->z0[rz]; k < layout->z1[rz]; k++)
   {
      for (HYPRE_BigInt j = layout->y0[ry]; j < layout->y1[ry]; j++)
      {
         for (HYPRE_BigInt i = layout->x0[rx]; i < layout->x1[rx]; i++)
         {
            HYPRE_BigInt row = layout_cell(layout, mesh, i, j, k);
            HYPRE_BigInt c   = cell_index(mesh, i, j, k);
            (void)c;
            HYPRE_Real coord[3] = {(i + 0.5) * mesh->h[0], (j + 0.5) * mesh->h[1],
                                   (k + 0.5) * mesh->h[2]};
            HYPRE_Real u_ref = 1.0 - coord[drive_axis] / mesh->L[drive_axis];
            HYPRE_Real diff  = x[row - ilower] - u_ref;
            local_err += diff * diff * cell_volume(mesh);
            local_ref += u_ref * u_ref * cell_volume(mesh);
         }
      }
   }

   HYPRE_Real global_err = 0.0;
   HYPRE_Real global_ref = 0.0;
   MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(&local_ref, &global_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
   return sqrt(global_err / global_ref);
}

int
main(int argc, char **argv)
{
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);

   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   DarcyParams params;
   int         parse_rc = parse_args(argc, argv, &params, rank, nprocs);
   if (parse_rc)
   {
      MPI_Finalize();
      return parse_rc == 2 ? 0 : 1;
   }

   DarcyMesh mesh;
   init_mesh(&params, &mesh);
   if (load_permeability_file(&params, &mesh, rank))
   {
      destroy_mesh(&mesh);
      MPI_Finalize();
      return 1;
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
   if (params.verbose & 0x1)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   }
   if (params.verbose & 0x2)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   HYPREDRV_t hypredrv;
   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_SetLibraryMode(hypredrv));
   char *config_arg =
      params.yaml_file ? params.yaml_file : (char *)default_config(params.verbose & 0x1);
   HYPRE_Int hypredrv_argc = 1 + params.hypredrv_argc;
   char     *hypredrv_argv[hypredrv_argc];
   hypredrv_argv[0] = config_arg;
   for (HYPRE_Int i = 0; i < params.hypredrv_argc; i++)
   {
      hypredrv_argv[i + 1] = params.hypredrv_argv[i];
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(hypredrv_argc, hypredrv_argv, hypredrv));

   HYPRE_Int mpi_grid[3];
   if (params.mpi_grid[0])
   {
      mpi_grid[0] = params.mpi_grid[0];
      mpi_grid[1] = params.mpi_grid[1];
      mpi_grid[2] = params.mpi_grid[2];
   }
   else
   {
      factor_mpi_grid(&mesh, nprocs, mpi_grid);
   }

   DarcyLayout layout;
   if (init_layout(&mesh, mpi_grid, &layout))
   {
      if (!rank)
      {
         printf("Error: failed to create MPI grid partition layout\n");
      }
      destroy_mesh(&mesh);
      HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
      MPI_Finalize();
      return 1;
   }

   if (!rank)
   {
      HYPRE_Int    cell_block[3]     = {ceil_div_int(mesh.n[0], mpi_grid[0]),
                                        ceil_div_int(mesh.n[1], mpi_grid[1]),
                                        ceil_div_int(mesh.n[2], mpi_grid[2])};
      HYPRE_BigInt rows_per_rank_min = layout.total[0];
      HYPRE_BigInt rows_per_rank_max = layout.total[0];
      for (int r = 1; r < nprocs; r++)
      {
         if (layout.total[r] < rows_per_rank_min) rows_per_rank_min = layout.total[r];
         if (layout.total[r] > rows_per_rank_max) rows_per_rank_max = layout.total[r];
      }

      printf("\n");
      printf("=====================================================\n");
      printf("              Darcy Mixed Problem Setup\n");
      printf("=====================================================\n");
      printf("Discretization:       %s\n", rt0_discretization.name);
      printf("Grid cells:           %d x %d x %d\n", params.n[0], params.n[1],
             params.n[2]);
      printf("Unknowns:             %lld flux + %lld pressure = %lld\n",
             (long long)mesh.n_faces, (long long)mesh.n_cells, (long long)mesh.n_total);
      printf("MPI ranks:            %d\n", nprocs);
      printf("MPI grid partition:   %d x %d x %d ranks\n", mpi_grid[0], mpi_grid[1],
             mpi_grid[2]);
      printf("Cell block target:    <= %d x %d x %d cells/rank\n", cell_block[0],
             cell_block[1], cell_block[2]);
      printf("Row partition:        rank-contiguous spatial DOFs (%lld",
             (long long)rows_per_rank_min);
      if (rows_per_rank_max != rows_per_rank_min)
      {
         printf("-%lld", (long long)rows_per_rank_max);
      }
      printf(" rows/rank)\n");
      printf("Drive direction:      %c\n", "xyz"[params.drive_axis]);
      if (params.K_file)
      {
         printf("Permeability file:    %s\n", params.K_file);
         if (params.K_file_grid[0])
         {
            printf("Permeability grid:    %d x %d x %d (%s)\n", params.K_file_grid[0],
                   params.K_file_grid[1], params.K_file_grid[2],
                   params.K_file_top_down ? "top-down" : "bottom-up");
         }
      }
      else
      {
         printf("Permeability diag:    %.3e %.3e %.3e\n", params.K[0], params.K[1],
                params.K[2]);
      }
      printf("=====================================================\n\n");
   }

   HYPRE_BigInt *indptr      = NULL;
   HYPRE_BigInt *col_indices = NULL;
   HYPRE_Real   *data        = NULL;
   HYPRE_Real   *rhs         = NULL;
   int          *dofmap      = NULL;
   HYPRE_BigInt  ilower, iupper;

   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(hypredrv, "system", -1));
   if (build_system_csr(comm, &mesh, &layout, &rt0_discretization, params.drive_axis,
                        &indptr, &col_indices, &data, &rhs, &dofmap, &ilower,
                        &iupper))
   {
      MPI_Abort(comm, 1);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrixFromCSR(
      hypredrv, ilower, iupper, indptr, col_indices, data));
   HYPREDRV_SAFE_CALL(
      HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, ilower, iupper, rhs));
   HYPREDRV_SAFE_CALL(
      HYPREDRV_LinearSystemSetDofmap(hypredrv, (int)(iupper - ilower + 1), dofmap));
   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(hypredrv, "system", -1));

   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
   free(indptr);
   free(col_indices);
   free(data);
   free(rhs);

   if (params.verbose & 0x4)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemPrint(hypredrv));
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hypredrv));

   HYPRE_Real *sol_data = NULL;
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &sol_data));
   HYPRE_Real rel_err = 0.0;
   if (!mesh.Kinv_cells)
   {
      rel_err = pressure_l2_error(comm, &mesh, &layout, params.drive_axis, ilower,
                                  sol_data);
   }
   if (!rank)
   {
      if (mesh.Kinv_cells)
      {
         printf("heterogeneous pressure solve completed\n");
      }
      else
      {
         printf("relative pressure L2 error : %.6e\n", rel_err);
      }
   }

   if (!rank && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
   }

   free(dofmap);
   int success = mesh.Kinv_cells || rel_err < 1.0e-6;
   destroy_layout(&layout);
   destroy_mesh(&mesh);
   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();
   return success ? 0 : 1;
}
