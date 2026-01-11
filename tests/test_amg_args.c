#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "amg.h"
#include "containers.h"
#include "error.h"
#include "test_helpers.h"
#include "yaml.h"

/* Forward declarations for internal AMG functions */
void           AMGSetFieldByName(AMG_args *, const YAMLnode *);
void           AMGintSetFieldByName(AMGint_args *, const YAMLnode *);
void           AMGintSetDefaultArgs(AMGint_args *);
StrIntMapArray AMGintGetValidValues(const char *);
void           AMGcsnSetFieldByName(AMGcsn_args *, const YAMLnode *);
void           AMGcsnSetDefaultArgs(AMGcsn_args *);
StrIntMapArray AMGcsnGetValidValues(const char *);
StrIntMapArray AMGaggGetValidValues(const char *);
StrIntMapArray AMGrlxGetValidValues(const char *);
StrIntMapArray AMGsmtGetValidValues(const char *);

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static void
test_AMGSetFieldByName_all_fields(void)
{
   AMG_args args;
   AMGSetDefaultArgs(&args);

   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "max_iter", .value = "5"},
      {.key = "print_level", .value = "2"},
      {.key = "tolerance", .value = "1.0e-6"},
   };

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      AMGSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 5);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ_DOUBLE(args.tolerance, 1.0e-6, 1e-12);
}

static void
test_AMGSetFieldByName_unknown_key(void)
{
   AMG_args args;
   AMGSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   AMGSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - it just doesn't match any field */
   /* Verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_AMGintGetValidValues_prolongation_type(void)
{
   StrIntMapArray map = AMGintGetValidValues("prolongation_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "mod_classical"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "extended+i"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "one_point"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "mod_classical"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "extended+i"), 6);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "one_point"), 100);
}

static void
test_AMGintGetValidValues_restriction_type(void)
{
   StrIntMapArray map = AMGintGetValidValues("restriction_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "p_transpose"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "air_1"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "air_1.5"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "p_transpose"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "air_1"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "air_1.5"), 15);
}

static void
test_AMGintGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGintGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGcsnGetValidValues_type(void)
{
   StrIntMapArray map = AMGcsnGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "cljp"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "falgout"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "hmis"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "cljp"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "falgout"), 6);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "hmis"), 10);
}

static void
test_AMGcsnGetValidValues_on_off_keys(void)
{
   StrIntMapArray map1 = AMGcsnGetValidValues("filter_functions");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map1, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map1, "off"));

   StrIntMapArray map2 = AMGcsnGetValidValues("nodal");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map2, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map2, "off"));

   StrIntMapArray map3 = AMGcsnGetValidValues("rap2");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map3, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map3, "off"));
}

static void
test_AMGcsnGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGcsnGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGaggGetValidValues_prolongation_type(void)
{
   StrIntMapArray map = AMGaggGetValidValues("prolongation_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "2_stage_extended+i"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "multipass"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "mm_extended+e"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "2_stage_extended+i"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "multipass"), 4);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "mm_extended+e"), 7);
}

static void
test_AMGaggGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGaggGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGrlxGetValidValues_down_type(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("down_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "jacobi_non_mv"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "chebyshev"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "l1sym-hgs"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "jacobi_non_mv"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "chebyshev"), 16);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "l1sym-hgs"), 89);
}

static void
test_AMGrlxGetValidValues_up_type(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("up_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "backward-hgs"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "cg"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "backward-hgs"), 4);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "cg"), 15);
}

static void
test_AMGrlxGetValidValues_coarse_type(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("coarse_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "lu_piv"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "lu_inv"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "lu_piv"), 99);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "lu_inv"), 199);
}

static void
test_AMGrlxGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGsmtGetValidValues_type(void)
{
   StrIntMapArray map = AMGsmtGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "fsai"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "ilu"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "euclid"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "fsai"), 4);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "ilu"), 5);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "euclid"), 9);
}

static void
test_AMGsmtGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGsmtGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGintSetFieldByName_all_fields(void)
{
   AMGint_args args;
   AMGintSetDefaultArgs(&args);

   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "prolongation_type", .value = "8"},
      {.key = "restriction_type", .value = "1"},
      {.key = "max_nnz_row", .value = "6"},
      {.key = "trunc_factor", .value = "0.5"},
   };

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      AMGintSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.prolongation_type, 8);
   ASSERT_EQ(args.restriction_type, 1);
   ASSERT_EQ(args.max_nnz_row, 6);
   ASSERT_EQ_DOUBLE(args.trunc_factor, 0.5, 1e-12);
}

static void
test_AMGcsnSetFieldByName_all_fields(void)
{
   AMGcsn_args args;
   AMGcsnSetDefaultArgs(&args);

   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "type", .value = "8"},
      {.key = "rap2", .value = "1"},
      {.key = "mod_rap2", .value = "0"},
      {.key = "keep_transpose", .value = "1"},
      {.key = "num_functions", .value = "2"},
      {.key = "filter_functions", .value = "1"},
      {.key = "nodal", .value = "0"},
      {.key = "seq_amg_th", .value = "1"},
      {.key = "min_coarse_size", .value = "10"},
      {.key = "max_coarse_size", .value = "100"},
      {.key = "max_levels", .value = "15"},
      {.key = "max_row_sum", .value = "0.8"},
      {.key = "strong_th", .value = "0.3"},
   };

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      AMGcsnSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.type, 8);
   ASSERT_EQ(args.rap2, 1);
   ASSERT_EQ(args.mod_rap2, 0);
   ASSERT_EQ(args.keep_transpose, 1);
   ASSERT_EQ(args.num_functions, 2);
   ASSERT_EQ(args.filter_functions, 1);
   ASSERT_EQ(args.nodal, 0);
   ASSERT_EQ(args.seq_amg_th, 1);
   ASSERT_EQ(args.min_coarse_size, 10);
   ASSERT_EQ(args.max_coarse_size, 100);
   ASSERT_EQ(args.max_levels, 15);
   ASSERT_EQ_DOUBLE(args.max_row_sum, 0.8, 1e-12);
   ASSERT_EQ_DOUBLE(args.strong_th, 0.3, 1e-12);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_AMGSetFieldByName_all_fields);
   RUN_TEST(test_AMGSetFieldByName_unknown_key);
   RUN_TEST(test_AMGintGetValidValues_prolongation_type);
   RUN_TEST(test_AMGintGetValidValues_restriction_type);
   RUN_TEST(test_AMGintGetValidValues_unknown_key);
   RUN_TEST(test_AMGcsnGetValidValues_type);
   RUN_TEST(test_AMGcsnGetValidValues_on_off_keys);
   RUN_TEST(test_AMGcsnGetValidValues_unknown_key);
   RUN_TEST(test_AMGaggGetValidValues_prolongation_type);
   RUN_TEST(test_AMGaggGetValidValues_unknown_key);
   RUN_TEST(test_AMGrlxGetValidValues_down_type);
   RUN_TEST(test_AMGrlxGetValidValues_up_type);
   RUN_TEST(test_AMGrlxGetValidValues_coarse_type);
   RUN_TEST(test_AMGrlxGetValidValues_unknown_key);
   RUN_TEST(test_AMGsmtGetValidValues_type);
   RUN_TEST(test_AMGsmtGetValidValues_unknown_key);
   RUN_TEST(test_AMGintSetFieldByName_all_fields);
   RUN_TEST(test_AMGcsnSetFieldByName_all_fields);

   MPI_Finalize();
   return 0;
}
