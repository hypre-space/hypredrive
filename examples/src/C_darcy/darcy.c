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
   HYPRE_Real L[3];
   HYPRE_Real K[3];
   HYPRE_Int  drive_axis;
   char      *yaml_file;
} DarcyParams;

typedef struct DarcyDiscretization DarcyDiscretization;
struct DarcyDiscretization
{
   const char *name;
   HYPRE_Int (*num_cell_flux_dofs)(const DarcyMesh *);
   void (*cell_flux_dofs)(const DarcyMesh *, HYPRE_BigInt, HYPRE_BigInt, HYPRE_BigInt,
                          HYPRE_BigInt *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);
   HYPRE_Real (*mass_entry)(const DarcyMesh *, HYPRE_Int, HYPRE_Int, const HYPRE_Int *,
                            const HYPRE_Int *);
};

static const char *default_config = "general:\n"
                                    "  statistics: 0\n"
                                    "  exec_policy: host\n"
                                    "linear_system:\n"
                                    "  init_guess_mode: zeros\n"
                                    "solver:\n"
                                    "  gmres:\n"
                                    "    max_iter: 200\n"
                                    "    krylov_dim: 60\n"
                                    "    relative_tol: 1.0e-10\n"
                                    "    absolute_tol: 0.0\n"
                                    "    print_level: 0\n"
                                    "preconditioner:\n"
                                    "  mgr:\n"
                                    "    tolerance: 0.0\n"
                                    "    max_iter: 1\n"
                                    "    print_level: 0\n"
                                    "    coarse_th: 0.0\n"
                                    "    level:\n"
                                    "      0:\n"
                                    "        f_dofs: [1]\n"
                                    "        f_relaxation: jacobi\n"
                                    "        g_relaxation: none\n"
                                    "        restriction_type: injection\n"
                                    "        prolongation_type: jacobi\n"
                                    "        coarse_level_type: rap\n"
                                    "    coarsest_level:\n"
                                    "      amg:\n"
                                    "        tolerance: 0.0\n"
                                    "        max_iter: 1\n"
                                    "        print_level: 0\n";

static HYPRE_BigInt
prod3(const HYPRE_Int n[3])
{
   return (HYPRE_BigInt)n[0] * (HYPRE_BigInt)n[1] * (HYPRE_BigInt)n[2];
}

static void
row_partition(HYPRE_BigInt nrows, int rank, int nprocs, HYPRE_BigInt *ilower,
              HYPRE_BigInt *iupper)
{
   HYPRE_BigInt base = nrows / nprocs;
   HYPRE_BigInt rem  = nrows - base * nprocs;
   HYPRE_BigInt len  = base + (rank < rem ? 1 : 0);
   *ilower           = rank * base + (rank < rem ? rank : rem);
   *iupper           = *ilower + len - 1;
}

static int
owned(HYPRE_BigInt row, HYPRE_BigInt ilower, HYPRE_BigInt iupper)
{
   return row >= ilower && row <= iupper;
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

static HYPRE_Real
rt0_mass_entry(const DarcyMesh *mesh, HYPRE_Int a, HYPRE_Int b, const HYPRE_Int *dirs,
               const HYPRE_Int *is_low)
{
   HYPRE_Int da = dirs[a];
   HYPRE_Int db = dirs[b];
   if (da != db)
   {
      return 0.0;
   }
   HYPRE_Real coef = is_low[a] == is_low[b] ? 1.0 / 3.0 : 1.0 / 6.0;
   return mesh->Kinv[da] * cell_volume(mesh) * coef /
          (face_area(mesh, da) * face_area(mesh, db));
}

static const DarcyDiscretization rt0_discretization = {
   "RT0/P0", rt0_num_cell_flux_dofs, rt0_cell_flux_dofs, rt0_mass_entry};

static HYPRE_Int
face_axis_high_low(const DarcyMesh *mesh, HYPRE_BigInt face, HYPRE_Int *axis,
                   HYPRE_Int *high, HYPRE_Int *low)
{
   *axis = -1;
   *high = *low = 0;
   if (face < mesh->n_x_faces)
   {
      HYPRE_BigInt i = face % (mesh->n[0] + 1);
      *axis          = 0;
      *low           = (i == 0);
      *high          = (i == mesh->n[0]);
      return *low || *high;
   }
   if (face < mesh->n_x_faces + mesh->n_y_faces)
   {
      HYPRE_BigInt f = face - mesh->n_x_faces;
      HYPRE_BigInt j = (f / mesh->n[0]) % (mesh->n[1] + 1);
      *axis          = 1;
      *low           = (j == 0);
      *high          = (j == mesh->n[1]);
      return *low || *high;
   }
   HYPRE_BigInt f = face - mesh->n_x_faces - mesh->n_y_faces;
   HYPRE_BigInt k = f / ((HYPRE_BigInt)mesh->n[0] * mesh->n[1]);
   *axis          = 2;
   *low           = (k == 0);
   *high          = (k == mesh->n[2]);
   return *low || *high;
}

static int
is_pinned_neumann_face(const DarcyMesh *mesh, HYPRE_BigInt face, HYPRE_Int drive_axis)
{
   HYPRE_Int axis, high, low;
   if (!face_axis_high_low(mesh, face, &axis, &high, &low))
   {
      return 0;
   }
   return axis >= 0 && axis < mesh->dim && axis != drive_axis;
}

static HYPRE_Real
dirichlet_rhs(const DarcyMesh *mesh, HYPRE_BigInt face, HYPRE_Int drive_axis)
{
   HYPRE_Int axis, high, low;
   if (!face_axis_high_low(mesh, face, &axis, &high, &low) || axis != drive_axis)
   {
      return 0.0;
   }
   return low ? 1.0 : 0.0;
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
set_default_params(DarcyParams *params)
{
   params->verbose    = 1;
   params->n[0]       = 8;
   params->n[1]       = 8;
   params->n[2]       = 1;
   params->L[0]       = 1.0;
   params->L[1]       = 1.0;
   params->L[2]       = 1.0;
   params->K[0]       = 1.0;
   params->K[1]       = 1.0;
   params->K[2]       = 1.0;
   params->drive_axis = 0;
   params->yaml_file  = NULL;
}

static int
print_usage(void)
{
   printf("\n");
   printf("Usage: ${MPIEXEC_COMMAND} <np> ./darcy [options]\n\n");
   printf("Options:\n");
   printf("  -i <file>         : YAML solver/preconditioner file (optional)\n");
   printf("  -n <nx> <ny> <nz> : Cell counts; active axes must be x, x-y, or x-y-z (8 8 "
          "1)\n");
   printf("  -L <Lx> <Ly> <Lz> : Domain lengths (1 1 1)\n");
   printf("  -K <Kx> <Ky> <Kz> : Constant diagonal permeability (1 1 1)\n");
   printf("  -g <x|y|z>        : Pressure-drop direction (x)\n");
   printf("  -v <n>            : Verbosity bitset; 0x1 stats, 0x2 system info, 0x4 print "
          "system\n");
   printf("  -h|--help         : Print this message\n\n");
   return 0;
}

static int
parse_args(int argc, char **argv, DarcyParams *params, int rank, int nprocs)
{
   (void)nprocs;
   set_default_params(params);
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
            if (!rank) printf("Error: -n requires three values\n");
            return 1;
         }
         for (int d = 0; d < 3; d++) params->n[d] = atoi(argv[++i]);
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
   for (int d = 0; d < 3; d++)
   {
      if (params->L[d] <= 0.0 || params->K[d] <= 0.0)
      {
         if (!rank) printf("Error: lengths and permeability entries must be positive\n");
         return 1;
      }
   }
   return 0;
}

static int
build_system(MPI_Comm comm, const DarcyMesh *mesh, const DarcyDiscretization *disc,
             HYPRE_Int drive_axis, HYPRE_IJMatrix *A_ptr, HYPRE_IJVector *b_ptr,
             int **dofmap_ptr, HYPRE_BigInt *ilower_ptr, HYPRE_BigInt *iupper_ptr)
{
   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   HYPRE_BigInt ilower, iupper;
   row_partition(mesh->n_total, rank, nprocs, &ilower, &iupper);
   HYPRE_BigInt local_rows = iupper - ilower + 1;
   if (local_rows <= 0)
   {
      if (!rank) printf("Error: more ranks than unknowns are not supported\n");
      return 1;
   }

   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorCreate(comm, ilower, iupper, &b);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);

   HYPRE_Int *row_sizes = (HYPRE_Int *)calloc((size_t)local_rows, sizeof(HYPRE_Int));
   int       *dofmap    = (int *)calloc((size_t)local_rows, sizeof(int));
   if (!row_sizes || !dofmap)
   {
      free(row_sizes);
      free(dofmap);
      return 1;
   }
   HYPRE_Int nloc = disc->num_cell_flux_dofs(mesh);
   for (HYPRE_BigInt lr = 0; lr < local_rows; lr++)
   {
      HYPRE_BigInt row = ilower + lr;
      if (row < mesh->n_faces)
      {
         row_sizes[lr] =
            is_pinned_neumann_face(mesh, row, drive_axis) ? 1 : 2 * (nloc + 1);
         dofmap[lr] = 1;
      }
      else
      {
         row_sizes[lr] = nloc;
         dofmap[lr]    = 0;
      }
   }
   HYPRE_IJMatrixSetRowSizes(A, row_sizes);
   HYPREDRV_IJ_MATRIX_INIT_HOST(A);
   HYPREDRV_IJ_VECTOR_INIT_HOST(b);
   free(row_sizes);

   for (HYPRE_BigInt lr = 0; lr < local_rows; lr++)
   {
      HYPRE_BigInt row = ilower + lr;
      HYPRE_Real   rhs = row < mesh->n_faces ? dirichlet_rhs(mesh, row, drive_axis) : 0.0;
      HYPRE_IJVectorSetValues(b, 1, &row, &rhs);
      if (row < mesh->n_faces && is_pinned_neumann_face(mesh, row, drive_axis))
      {
         HYPRE_Int  one = 1;
         HYPRE_Real val = 1.0;
         HYPRE_IJMatrixAddToValues(A, 1, &one, &row, &row, &val);
      }
   }

   HYPRE_BigInt faces[6];
   HYPRE_Int    dirs[6], is_low[6], signs[6];
   HYPRE_BigInt cols[7];
   HYPRE_Real   vals[7];

   for (HYPRE_BigInt c = 0; c < mesh->n_cells; c++)
   {
      HYPRE_BigInt i, j, k;
      cell_ijk(mesh, c, &i, &j, &k);
      disc->cell_flux_dofs(mesh, i, j, k, faces, dirs, is_low, signs);

      for (HYPRE_Int a = 0; a < nloc; a++)
      {
         HYPRE_BigInt row = faces[a];
         if (!owned(row, ilower, iupper) || is_pinned_neumann_face(mesh, row, drive_axis))
         {
            continue;
         }
         HYPRE_Int nentries = 0;
         for (HYPRE_Int bb = 0; bb < nloc; bb++)
         {
            HYPRE_Real m = disc->mass_entry(mesh, a, bb, dirs, is_low);
            if (m != 0.0)
            {
               cols[nentries]   = faces[bb];
               vals[nentries++] = m;
            }
         }
         cols[nentries]   = mesh->n_faces + c;
         vals[nentries++] = -(HYPRE_Real)signs[a];
         HYPRE_IJMatrixAddToValues(A, 1, &nentries, &row, cols, vals);
      }

      HYPRE_BigInt prow = mesh->n_faces + c;
      if (owned(prow, ilower, iupper))
      {
         HYPRE_Int nentries = 0;
         for (HYPRE_Int a = 0; a < nloc; a++)
         {
            cols[nentries]   = faces[a];
            vals[nentries++] = -(HYPRE_Real)signs[a];
         }
         HYPRE_IJMatrixAddToValues(A, 1, &nentries, &prow, cols, vals);
      }
   }
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJVectorAssemble(b);

   *A_ptr      = A;
   *b_ptr      = b;
   *dofmap_ptr = dofmap;
   *ilower_ptr = ilower;
   *iupper_ptr = iupper;
   return 0;
}

static HYPRE_Real
pressure_l2_error(MPI_Comm comm, const DarcyMesh *mesh, HYPRE_Int drive_axis,
                  HYPRE_BigInt ilower, HYPRE_BigInt iupper, const HYPRE_Real *x)
{
   HYPRE_Real local_err = 0.0;
   HYPRE_Real local_ref = 0.0;
   for (HYPRE_BigInt row = ilower; row <= iupper; row++)
   {
      if (row < mesh->n_faces)
      {
         continue;
      }
      HYPRE_BigInt c = row - mesh->n_faces;
      HYPRE_BigInt i, j, k;
      cell_ijk(mesh, c, &i, &j, &k);
      HYPRE_Real coord[3] = {(i + 0.5) * mesh->h[0], (j + 0.5) * mesh->h[1],
                             (k + 0.5) * mesh->h[2]};
      HYPRE_Real u_ref    = 1.0 - coord[drive_axis] / mesh->L[drive_axis];
      HYPRE_Real diff     = x[row - ilower] - u_ref;
      local_err += diff * diff * cell_volume(mesh);
      local_ref += u_ref * u_ref * cell_volume(mesh);
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
   char *args[1] = {params.yaml_file ? params.yaml_file : (char *)default_config};
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, args, hypredrv));

   if (!rank)
   {
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
      printf("Drive direction:      %c\n", "xyz"[params.drive_axis]);
      printf("Permeability diag:    %.3e %.3e %.3e\n", params.K[0], params.K[1],
             params.K[2]);
      printf("=====================================================\n\n");
   }

   HYPRE_IJMatrix A      = NULL;
   HYPRE_IJVector b      = NULL;
   int           *dofmap = NULL;
   HYPRE_BigInt   ilower, iupper;

   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(hypredrv, "system", -1));
   if (build_system(comm, &mesh, &rt0_discretization, params.drive_axis, &A, &b, &dofmap,
                    &ilower, &iupper))
   {
      MPI_Abort(comm, 1);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(hypredrv, "system", -1));

#if defined(HYPRE_USING_GPU)
   HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   HYPRE_IJVectorMigrate(b, HYPRE_MEMORY_DEVICE);
#endif

   HYPREDRV_SAFE_CALL(
      HYPREDRV_LinearSystemSetDofmap(hypredrv, (int)(iupper - ilower + 1), dofmap));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)A));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)b));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));

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
   HYPRE_Real rel_err =
      pressure_l2_error(comm, &mesh, params.drive_axis, ilower, iupper, sol_data);
   if (!rank)
   {
      printf("relative pressure L2 error : %.6e\n", rel_err);
   }

   if (!rank && (params.verbose & 0x1))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(hypredrv));
   }

   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   free(dofmap);
   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();
   return rel_err < 1.0e-6 ? 0 : 1;
}
