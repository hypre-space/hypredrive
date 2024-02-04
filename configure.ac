AC_PREREQ([2.69])
AC_INIT([hypredrive], [0.1], [paludettomag1@llnl.gov])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

dnl Include m4 macros
m4_include([m4/ax_prog_doxygen.m4])

dnl use pretty build output with automake >= 1.11
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])],
  [AM_DEFAULT_VERBOSITY=1
   AC_SUBST(AM_DEFAULT_VERBOSITY)])

dnl Generate documentation with doxygen
DX_DOXYGEN_FEATURE(ON)
DX_HTML_FEATURE(ON)
DX_MAN_FEATURE(ON)
DX_RTF_FEATURE(ON)
DX_PDF_FEATURE(ON)
DX_XML_FEATURE(ON)
DX_DOT_FEATURE(ON)
DX_INIT_DOXYGEN(hypredrive, [Doxyfile], [docs])

AC_PROG_CC([mpicc cc])

AC_ARG_WITH([hypre-include],
            [AS_HELP_STRING([--with-hypre-include=DIR], [Path to HYPRE include directory])],
            [CFLAGS+=" -I$withval"],
            [])

AC_ARG_WITH([hypre-lib],
            [AS_HELP_STRING([--with-hypre-lib=DIR], [Path to HYPRE library directory])],
            [LDFLAGS+=" -L$withval -Wl,-rpath,${withval}"],
            [])

AC_CHECK_LIB([m], [sin],
             [],
             [AC_MSG_ERROR([libm (math library) not found or not usable])])
AC_CHECK_LIB([HYPRE], [HYPRE_Initialize], [], [AC_MSG_ERROR([HYPRE library not found or not usable])])

AC_CONFIG_HEADERS([config.h])
AC_CHECK_HEADERS([stdint.h])
AC_CHECK_FUNCS([strchr strcspn strdup memset strstr strtol])
AC_CHECK_HEADER_STDBOOL
AC_TYPE_UINT32_T
AC_TYPE_SIZE_T
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_C_INLINE

AC_CONFIG_FILES([
    Makefile
    Doxyfile
])

AC_OUTPUT
