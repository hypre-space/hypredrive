#include <string.h>

#include "eigspec.h"
#include "test_helpers.h"

static void test_EigSpecSetDefaultArgs(void)
{
   EigSpec_args args;
   memset(&args, 0, sizeof(args));
   EigSpecSetDefaultArgs(&args);
   ASSERT_EQ(args.enable, 0);
   ASSERT_EQ(args.vectors, 0);
   ASSERT_EQ(args.hermitian, 0);
   ASSERT_EQ(args.preconditioned, 0);
   ASSERT_STREQ(args.output_prefix, "eig");
}

int main(void)
{
   RUN_TEST(test_EigSpecSetDefaultArgs);
   return 0;
}

