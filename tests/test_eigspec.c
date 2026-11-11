#include "HYPREDRV.h"
#include <string.h>

#include "_hypre_parcsr_mv.h"
#include "internal/eigspec.h"
#include "internal/error.h"
#include "internal/yaml.h"
#include "object.h"
#include "test_helpers.h"
#include "test_ij_helpers.h"

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

static void
test_EigSpecSetDefaultArgs(void)
{
   EigSpec_args args;
   memset(&args, 0, sizeof(args));
   hypredrv_EigSpecSetDefaultArgs(&args);
   ASSERT_EQ(args.enable, 0);
   ASSERT_EQ(args.vectors, 0);
   ASSERT_EQ(args.hermitian, 0);
   ASSERT_EQ(args.preconditioned, 0);
   ASSERT_STREQ(args.output_prefix, "eig");
}

static void
test_EigSpecSetArgsFromYAML_sets_fields(void)
{
   EigSpec_args args;
   hypredrv_EigSpecSetDefaultArgs(&args);

   YAMLnode *parent       = hypredrv_YAMLnodeCreate("eigspectrum", "", 0);
   YAMLnode *enable_node  = add_child(parent, "enable", "on", 1);
   YAMLnode *vectors_node = add_child(parent, "vectors", "on", 1);
   YAMLnode *herm_node    = add_child(parent, "hermitian", "on", 1);
   YAMLnode *precon_node  = add_child(parent, "preconditioned", "on", 1);
   YAMLnode *prefix_node  = add_child(parent, "output_prefix", "diag", 1);

   hypredrv_ErrorStateReset();
   hypredrv_EigSpecSetArgs(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(enable_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(vectors_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(herm_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(precon_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(prefix_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(args.enable, 1);
   ASSERT_EQ(args.vectors, 1);
   ASSERT_EQ(args.hermitian, 1);
   ASSERT_EQ(args.preconditioned, 1);
   ASSERT_STREQ(args.output_prefix, "diag");

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_EigSpecSetArgsFromYAML_invalid_toggle_sets_error(void)
{
   EigSpec_args args;
   hypredrv_EigSpecSetDefaultArgs(&args);

   YAMLnode *parent      = hypredrv_YAMLnodeCreate("eigspectrum", "", 0);
   YAMLnode *enable_node = add_child(parent, "enable", "maybe", 1);

   hypredrv_ErrorStateReset();
   hypredrv_EigSpecSetArgs(&args, parent);

   ASSERT_EQ(enable_node->valid, YAML_NODE_INVALID_VAL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.enable, 0);

   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorStateReset();
}

#ifdef HYPREDRV_ENABLE_EIGSPEC
static HYPREDRV_t
create_eigspec_object(const char *policy, const char *preconditioner)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char yaml[512];
   snprintf(yaml, sizeof(yaml),
            "general:\n  exec_policy: %s\n  statistics: off\n"
            "linear_system:\n  eigspec:\n    enable: on\n    vectors: on\n"
            "solver: pcg\npreconditioner: %s\n",
            policy, preconditioner);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_BigInt rowptr[] = {0, 1, 2}, columns[] = {0, 1};
   HYPRE_Real   values[] = {2.0, 3.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 1, rowptr, columns, values),
             ERROR_NONE);
   test_scratch_path(obj->iargs->ls.eigspec.output_prefix, MAX_FILENAME_LENGTH,
                     "spectrum");
   return obj;
}

static void
assert_diagonal_spectrum(const EigSpec_args *args, double a, double b)
{
   char path[MAX_FILENAME_LENGTH + 32];
   snprintf(path, sizeof(path), "%s.values.txt", args->output_prefix);
   FILE *fp = fopen(path, "r");
   ASSERT_NOT_NULL(fp);
   double values[2] = {0, 0}, imaginary = 0;
   for (int i = 0; i < 2; i++)
   {
      ASSERT_EQ(fscanf(fp, args->hermitian ? "%lf" : "%lf %lf", &values[i], &imaginary),
                args->hermitian ? 1 : 2);
      ASSERT_EQ_DOUBLE(imaginary, 0.0, 1e-12);
   }
   ASSERT_EQ(fclose(fp), 0);
   ASSERT_EQ_DOUBLE(values[0] + values[1], a + b, 1e-12);
   ASSERT_EQ_DOUBLE(values[0] * values[1], a * b, 1e-12);
   snprintf(path, sizeof(path), "%s.vectors.bin", args->output_prefix);
   fp = fopen(path, "rb");
   ASSERT_NOT_NULL(fp);
   double vectors[8] = {0};
   size_t count      = args->hermitian ? 4u : 8u;
   ASSERT_EQ_SIZE(fread(vectors, sizeof(double), 8, fp), count);
   ASSERT_EQ(fclose(fp), 0);
   for (int ev = 0; ev < 2; ev++)
   {
      double norm = 0.0;
      for (int row = 0; row < 2; row++)
      {
         size_t offset = (size_t)(2 * ev + row) * (args->hermitian ? 1u : 2u);
         double real   = vectors[offset];
         double imag   = args->hermitian ? 0.0 : vectors[offset + 1];
         norm += real * real + imag * imag;
         ASSERT_EQ_DOUBLE(((row ? b : a) - values[ev]) * real, 0.0, 1e-12);
         ASSERT_EQ_DOUBLE(imag, 0.0, 1e-12);
      }
      ASSERT_EQ_DOUBLE(norm, 1.0, 1e-12);
   }
}

static void
test_EigSpecCompute_own_status(void)
{
   HYPREDRV_t    obj  = create_eigspec_object("host", "amg");
   EigSpec_args *args = &obj->iargs->ls.eigspec;
   for (int symmetric = 0; symmetric < 2; symmetric++)
   {
      args->hermitian = symmetric;
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ASSERT_EQ(hypredrv_EigSpecCompute(obj), ERROR_NONE);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
      assert_diagonal_spectrum(args, 2.0, 3.0);
   }
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_EigSpecCompute_complex_eigenvectors(void)
{
   HYPREDRV_t   obj      = create_eigspec_object("host", "amg");
   HYPRE_BigInt rowptr[] = {0, 1, 2}, columns[] = {1, 0};
   HYPRE_Real   entries[] = {-1.0, 1.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 1, rowptr, columns, entries),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);
   char path[MAX_FILENAME_LENGTH + 32];
   snprintf(path, sizeof(path), "%s.values.txt", obj->iargs->ls.eigspec.output_prefix);
   FILE *fp = fopen(path, "r");
   ASSERT_NOT_NULL(fp);
   double real[2], imag[2];
   for (int ev = 0; ev < 2; ev++)
   {
      ASSERT_EQ(fscanf(fp, "%lf %lf", &real[ev], &imag[ev]), 2);
      ASSERT_EQ_DOUBLE(real[ev], 0.0, 1e-12);
      ASSERT_EQ_DOUBLE(fabs(imag[ev]), 1.0, 1e-12);
   }
   ASSERT_EQ_DOUBLE(imag[0], -imag[1], 1e-12);
   ASSERT_EQ(fclose(fp), 0);
   snprintf(path, sizeof(path), "%s.vectors.bin", obj->iargs->ls.eigspec.output_prefix);
   fp = fopen(path, "rb");
   ASSERT_NOT_NULL(fp);
   double vectors[8];
   ASSERT_EQ_SIZE(fread(vectors, sizeof(double), 8, fp), 8u);
   ASSERT_EQ(fgetc(fp), EOF);
   ASSERT_EQ(fclose(fp), 0);
   for (int ev = 0; ev < 2; ev++)
   {
      double *v = &vectors[4 * ev];
      /* A*v = i*imag[ev]*v for A = [0,-1; 1,0]. */
      ASSERT_EQ_DOUBLE(-v[2], -imag[ev] * v[1], 1e-12);
      ASSERT_EQ_DOUBLE(-v[3], imag[ev] * v[0], 1e-12);
      ASSERT_EQ_DOUBLE(v[0], -imag[ev] * v[3], 1e-12);
      ASSERT_EQ_DOUBLE(v[1], imag[ev] * v[2], 1e-12);
      ASSERT_EQ_DOUBLE(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3], 1.0, 1e-12);
   }
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_EigSpecCompute_preconditioned(void)
{
#ifdef HYPRE_USING_GPU
   const char *policies[] = {"host", "device"};
#else
   const char *policies[] = {"host"};
#endif
   for (size_t i = 0; i < sizeof(policies) / sizeof(policies[0]); i++)
   {
      HYPREDRV_t    obj    = create_eigspec_object(policies[i], "amg");
      EigSpec_args *args   = &obj->iargs->ls.eigspec;
      args->preconditioned = 1;
      void *matrix_object  = NULL;
      ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &matrix_object), 0);
      hypre_ParCSRMatrix  *matrix   = (hypre_ParCSRMatrix *)matrix_object;
      HYPRE_MemoryLocation location = hypre_ParCSRMatrixMemoryLocation(matrix);
      HYPRE_Complex       *data     = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(matrix));
#ifdef HYPRE_USING_GPU
      ASSERT_EQ(location, i ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST);
#endif
      for (int symmetric = 0; symmetric < 2; symmetric++)
      {
         args->hermitian = symmetric;
         ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);
         assert_diagonal_spectrum(args, 1.0, 1.0);
         ASSERT_EQ(hypre_ParCSRMatrixMemoryLocation(matrix), location);
         ASSERT_PTR_EQ(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(matrix)), data);
      }
      /* The same operator and preconditioner remain usable after analysis. */
      HYPRE_Real rhs[] = {2.0, 6.0};
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 1, rhs), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_PreconApply(obj, (HYPRE_Vector)obj->vec_b.handle,
                                     (HYPRE_Vector)obj->vec_x.handle),
                ERROR_NONE);
      void *solution_object = NULL;
      ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &solution_object), 0);
      hypre_Vector *solution =
         hypre_ParVectorLocalVector((hypre_ParVector *)solution_object);
      HYPRE_Complex values[2];
      hypre_TMemcpy(values, hypre_VectorData(solution), HYPRE_Complex, 2,
                    HYPRE_MEMORY_HOST, hypre_VectorMemoryLocation(solution));
      ASSERT_EQ_DOUBLE((double)hypre_creal(values[0]), 1.0, 1e-12);
      ASSERT_EQ_DOUBLE((double)hypre_creal(values[1]), 2.0, 1e-12);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_EigSpecCompute_rejects_invalid_input(void)
{
   HYPREDRV_t obj                        = create_eigspec_object("host", "ams");
   obj->iargs->ls.eigspec.preconditioned = 1;
   char path[MAX_FILENAME_LENGTH + 32];
   test_scratch_path(obj->iargs->ls.eigspec.output_prefix, MAX_FILENAME_LENGTH,
                     "failed-preconditioner");
   snprintf(path, sizeof(path), "%s.values.txt", obj->iargs->ls.eigspec.output_prefix);
   ASSERT_TRUE(HYPREDRV_LinearSystemComputeEigenspectrum(obj) & ERROR_MISSING_PRECON);
   ASSERT_NULL(fopen(path, "r"));
   ASSERT_FALSE(obj->precon_is_setup);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   obj                  = create_eigspec_object("host", "amg");
   HYPRE_IJMatrix large = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 20000, 0, 20000, &large), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(large, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(large), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(large), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)large), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_OUT_OF_BOUNDS);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_FILE_NOT_FOUND);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(large), 0);
}
#endif

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_EigSpecSetDefaultArgs);
   RUN_TEST(test_EigSpecSetArgsFromYAML_sets_fields);
   RUN_TEST(test_EigSpecSetArgsFromYAML_invalid_toggle_sets_error);
#ifdef HYPREDRV_ENABLE_EIGSPEC
   RUN_TEST(test_EigSpecCompute_own_status);
   RUN_TEST(test_EigSpecCompute_complex_eigenvectors);
   RUN_TEST(test_EigSpecCompute_preconditioned);
   RUN_TEST(test_EigSpecCompute_rejects_invalid_input);
#endif
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
