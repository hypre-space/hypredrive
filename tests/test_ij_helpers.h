/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TEST_IJ_HELPERS_HEADER
#define TEST_IJ_HELPERS_HEADER

#include <mpi.h>

#include "HYPRE_IJ_mv.h"
#include "test_helpers.h"

static inline HYPRE_IJMatrix
create_test_ijmatrix_1x1(double diag)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   HYPRE_Int    nrows     = 1;
   HYPRE_Int    ncols[1]  = {1};
   HYPRE_BigInt rows[1]   = {0};
   HYPRE_BigInt cols[1]   = {0};
   double       values[1] = {diag};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static inline HYPRE_IJVector
create_test_ijvector_1x1(double value)
{
   HYPRE_IJVector vec = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(vec), 0);
   HYPRE_BigInt idx[1] = {0};
   double       val[1] = {value};
   ASSERT_EQ(HYPRE_IJVectorSetValues(vec, 1, idx, val), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vec), 0);
   return vec;
}

static inline double
get_test_ijmatrix_1x1(HYPRE_IJMatrix mat)
{
   HYPRE_Int     ncols = 1;
   HYPRE_BigInt  index = 0;
   HYPRE_Complex value = 0.0;
   ASSERT_EQ(HYPRE_IJMatrixGetValues(mat, 1, &ncols, &index, &index, &value), 0);
   return (double)value;
}

static inline double
get_test_ijvector_1x1(HYPRE_IJVector vec)
{
   HYPRE_BigInt  index = 0;
   HYPRE_Complex value = 0.0;
   ASSERT_EQ(HYPRE_IJVectorGetValues(vec, 1, &index, &value), 0);
   return (double)value;
}

static inline HYPRE_IJMatrix
create_test_ijmatrix_2x2(double a00, double a01, double a10, double a11)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 1, 0, 1, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   HYPRE_Int    nrows     = 2;
   HYPRE_Int    ncols[2]  = {2, 2};
   HYPRE_BigInt rows[2]   = {0, 1};
   HYPRE_BigInt cols[4]   = {0, 1, 0, 1};
   double       values[4] = {a00, a01, a10, a11};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static inline HYPRE_IJVector
create_test_ijvector_2x1(double v0, double v1)
{
   HYPRE_IJVector vec = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 1, &vec), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(vec), 0);
   HYPRE_BigInt idx[2] = {0, 1};
   double       val[2] = {v0, v1};
   ASSERT_EQ(HYPRE_IJVectorSetValues(vec, 2, idx, val), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vec), 0);
   return vec;
}

#endif /* TEST_IJ_HELPERS_HEADER */
