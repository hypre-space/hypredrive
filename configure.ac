AC_PREREQ([2.71])
AC_INIT([hypre-app], [0.1], [paludettomag1@llnl.gov])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

dnl use pretty build output with automake >= 1.11
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])],
  [AM_DEFAULT_VERBOSITY=1
   AC_SUBST(AM_DEFAULT_VERBOSITY)])

AC_PROG_CC([mpicc cc])

AC_ARG_WITH([hypre-include],
            [AS_HELP_STRING([--with-hypre-include=DIR], [Path to HYPRE include directory])],
            [CFLAGS+=" -I$withval"],
            [])

AC_ARG_WITH([hypre-lib],
            [AS_HELP_STRING([--with-hypre-lib=DIR], [Path to HYPRE library directory])],
            [LDFLAGS+=" -L$withval"],
            [])

AC_CHECK_LIB([m], [sin],
             [],
             [AC_MSG_ERROR([libm (math library) not found or not usable])])
AC_CHECK_LIB([HYPRE], [HYPRE_Initialize], [], [AC_MSG_ERROR([HYPRE library not found or not usable])])

AC_CHECK_FUNCS([strchr strcspn strdup])
AC_TYPE_SIZE_T
AC_FUNC_MALLOC
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([
    Makefile
])

AC_OUTPUT
