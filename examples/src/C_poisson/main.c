/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <HYPREDRV.h>

int main(int argc, char** argv)
{
   MPI_Comm      comm = MPI_COMM_WORLD;
   HYPRE_BigInt  Nx, Ny, Nz, N;
   HYPRE_Int     Px, Py, Pz;
   HYPRE_Int     myid, num_procs;
   HYPREDRV_t    obj;

   // Initialize
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   HYPREDRV_Initialize();
   HYPREDRV_Create(comm, &obj);
   HYPREDRV_InputArgsParse(argc - 1, argv + 1, obj);
   HYPREDRV_SetGlobalOptions(obj);
   HYPREDRV_SetLibraryMode(obj);

   // Problem size
   Nx = 10, Ny = 10, Nz = 10;
   Px = 1,  Py = 1,  Pz = 1;
   N = Nx * Ny * Nz;

   // Compute process grid coordinates
   HYPRE_Int        px = myid % Px;
   HYPRE_Int        py = (myid / Px) % Py;
   HYPRE_Int        pz = myid / (Px * Py);

   // Compute local grid dimensions
   HYPRE_BigInt     nx = Nx / Px;
   HYPRE_BigInt     ny = Ny / Py;
   HYPRE_BigInt     nz = Nz / Pz;

   // Compute global and local row ranges
   HYPRE_BigInt     ilower = (px * nx + py * ny * Nx + pz * nz * Nx * Ny);
   HYPRE_BigInt     iupper = ilower + nx * ny * nz - 1;

   // Create the matrix
   HYPRE_IJMatrix   A;
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);

   // Set up the matrix stencil
   HYPRE_Int        stencil_size = 7;
   HYPRE_Int        offsets[7]   = {0, -nx*ny, -nx, -1, 1, nx, nx*ny};
   HYPRE_Complex    values[7]    = {6.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

   // Allocate memory for a block of rows
   HYPRE_Int       block_size = nx * ny;
   HYPRE_BigInt   *rows  = (HYPRE_BigInt *)  malloc(block_size * sizeof(HYPRE_BigInt));
   HYPRE_BigInt   *cols  = (HYPRE_BigInt *)  malloc(block_size * stencil_size * sizeof(HYPRE_BigInt));
   HYPRE_Complex  *vals  = (HYPRE_Complex *) malloc(block_size * stencil_size * sizeof(HYPRE_Complex));
   HYPRE_Int      *ncols = (HYPRE_Int *)     malloc(block_size * sizeof(HYPRE_Int));
   HYPRE_Int       nnz;
   HYPRE_Int       row_index;
   HYPRE_Int       diag_index;

   for (HYPRE_BigInt iz = 0; iz < nz; iz++)
   {
      row_index = nnz = 0;
      for (HYPRE_BigInt iy = 0; iy < ny; iy++)
      {
         for (HYPRE_BigInt ix = 0; ix < nx; ix++)
         {
            HYPRE_BigInt row = ilower + iz * nx * ny + iy * nx + ix;

            rows[row_index]  = row;
            ncols[row_index] = 0;
            if (!ix || ix == nx - 1 || !iy || iy == ny - 1 || !iz || iz == nz - 1)
            {
               diag_index = nnz;

               // Boundary point
               HYPRE_Int conditions[7] = {1, iz, iy, ix, ix != nx - 1, iy != ny - 1, iz != nz - 1};
               for (HYPRE_Int i = 0; i < stencil_size; i++)
               {
                  if (conditions[i])
                  {
                     cols[nnz] = row + offsets[i];
                     vals[nnz] = values[i];
                     nnz++;
                  }
               }
               ncols[row_index] = nnz - diag_index;
            }
            else
            {
               // Interior point
               for (HYPRE_Int i = 0; i < stencil_size; i++)
               {
                  cols[nnz] = row + offsets[i];
                  vals[nnz] = values[i];
                  nnz++;
               }
               ncols[row_index] = stencil_size;
            }
            row_index++;
         }
      }

      HYPRE_IJMatrixSetValues(A, block_size, ncols, rows, cols, vals);
   }

   // Free work arrays
   free(rows);
   free(ncols);
   free(cols);
   free(vals);

   // Assemble the matrix
   HYPRE_IJMatrixAssemble(A);
   //HYPRE_IJMatrixPrint(A, "test");

   // Associate the matrix with the HYPREDRV object
   HYPREDRV_LinearSystemSetMatrix(obj, A);
   HYPREDRV_LinearSystemSetRHS(obj, NULL); // Replace NULL with vector if available
   HYPREDRV_LinearSystemSetInitialGuess(obj);
   HYPREDRV_LinearSystemSetPrecMatrix(obj);
   HYPREDRV_LinearSystemReadDofmap(obj);
   HYPREDRV_LinearSystemResetInitialGuess(obj);

   // Solver create phase
   HYPREDRV_PreconCreate(obj);
   HYPREDRV_LinearSolverCreate(obj);

   // Solver setup phase
   HYPREDRV_LinearSolverSetup(obj);

   // Solver application phase
   HYPREDRV_LinearSolverApply(obj);

   // Destroy phase
   HYPREDRV_PreconDestroy(obj);
   HYPREDRV_LinearSolverDestroy(obj);

   // Print statistics
   if (!myid) HYPREDRV_StatsPrint(obj);

   // Finalize program
   HYPREDRV_Destroy(&obj);
   HYPREDRV_Finalize();
   MPI_Finalize();

   return 0;
}
