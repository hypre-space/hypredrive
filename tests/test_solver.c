#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "bicgstab.h"
#include "containers.h"
#include "gmres.h"
#include "pcg.h"
#include "test_helpers.h"
#include "yaml.h"

void           GMRESSetFieldByName(GMRES_args *, const YAMLnode *);
void           GMRESSetDefaultArgs(GMRES_args *);
StrArray       GMRESGetValidKeys(void);
StrIntMapArray GMRESGetValidValues(const char *);
void           PCGSetFieldByName(PCG_args *, const YAMLnode *);
void           PCGSetDefaultArgs(PCG_args *);
StrIntMapArray PCGGetValidValues(const char *);
void           BiCGSTABSetFieldByName(BiCGSTAB_args *, const YAMLnode *);
void           BiCGSTABSetDefaultArgs(BiCGSTAB_args *);

typedef struct
{
   const char *key;
   const char *value;
} keyval_pair;

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static void
test_GMRESSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "min_iter", .value = "2"},
      {.key = "max_iter", .value = "75"},
      {.key = "stop_crit", .value = "1"},
      {.key = "skip_real_res_check", .value = "1"},
      {.key = "krylov_dim", .value = "40"},
      {.key = "rel_change", .value = "1"},
      {.key = "logging", .value = "0"},
      {.key = "print_level", .value = "3"},
      {.key = "relative_tol", .value = "1.0e-7"},
      {.key = "absolute_tol", .value = "0.5"},
      {.key = "conv_fac_tol", .value = "0.25"},
   };

   GMRES_args args;
   GMRESSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      GMRESSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.min_iter, 2);
   ASSERT_EQ(args.max_iter, 75);
   ASSERT_EQ(args.stop_crit, 1);
   ASSERT_EQ(args.skip_real_res_check, 1);
   ASSERT_EQ(args.krylov_dim, 40);
   ASSERT_EQ(args.rel_change, 1);
   ASSERT_EQ(args.logging, 0);
   ASSERT_EQ(args.print_level, 3);
   ASSERT_EQ_DOUBLE(args.relative_tol, 1.0e-7, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.5, 1e-12);
   ASSERT_EQ_DOUBLE(args.conv_fac_tol, 0.25, 1e-12);

   StrArray keys = GMRESGetValidKeys();
   ASSERT_EQ(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(StrArrayEntryExists(keys, updates[i].key));
   }

   StrIntMapArray bool_map = GMRESGetValidValues("skip_real_res_check");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "off"));

   YAMLnode *parent = YAMLnodeCreate("gmres", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      YAMLnodeAddChild(parent, child);
   }

   GMRES_args args_from_yaml;
   GMRESSetArgs(&args_from_yaml, parent);
   YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.max_iter, 75);
   ASSERT_EQ_DOUBLE(args_from_yaml.conv_fac_tol, 0.25, 1e-12);
}

static void
test_PCGSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "max_iter", .value = "150"},
      {.key = "two_norm", .value = "0"},
      {.key = "stop_crit", .value = "1"},
      {.key = "rel_change", .value = "1"},
      {.key = "print_level", .value = "2"},
      {.key = "recompute_res", .value = "1"},
      {.key = "relative_tol", .value = "9.0e-8"},
      {.key = "absolute_tol", .value = "0.75"},
      {.key = "residual_tol", .value = "0.33"},
      {.key = "conv_fac_tol", .value = "0.12"},
   };

   PCG_args args;
   PCGSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      PCGSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 150);
   ASSERT_EQ(args.two_norm, 0);
   ASSERT_EQ(args.stop_crit, 1);
   ASSERT_EQ(args.rel_change, 1);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ(args.recompute_res, 1);
   ASSERT_EQ_DOUBLE(args.relative_tol, 9.0e-8, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.75, 1e-12);
   ASSERT_EQ_DOUBLE(args.residual_tol, 0.33, 1e-12);
   ASSERT_EQ_DOUBLE(args.conv_fac_tol, 0.12, 1e-12);

   StrIntMapArray bool_map = PCGGetValidValues("two_norm");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "off"));
}

static void
test_BiCGSTABSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "min_iter", .value = "3"},
      {.key = "max_iter", .value = "120"},
      {.key = "stop_crit", .value = "1"},
      {.key = "logging", .value = "0"},
      {.key = "print_level", .value = "4"},
      {.key = "relative_tol", .value = "5.0e-7"},
      {.key = "absolute_tol", .value = "0.21"},
      {.key = "conv_fac_tol", .value = "0.09"},
   };

   BiCGSTAB_args args;
   BiCGSTABSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      BiCGSTABSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.min_iter, 3);
   ASSERT_EQ(args.max_iter, 120);
   ASSERT_EQ(args.stop_crit, 1);
   ASSERT_EQ(args.logging, 0);
   ASSERT_EQ(args.print_level, 4);
   ASSERT_EQ_DOUBLE(args.relative_tol, 5.0e-7, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.21, 1e-12);
   ASSERT_EQ_DOUBLE(args.conv_fac_tol, 0.09, 1e-12);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_GMRESSetFieldByName_all_fields);
   RUN_TEST(test_PCGSetFieldByName_all_fields);
   RUN_TEST(test_BiCGSTABSetFieldByName_all_fields);

   MPI_Finalize();
   return 0;
}
