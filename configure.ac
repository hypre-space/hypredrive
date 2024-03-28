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
AC_ARG_ENABLE([doxygen],
              [AS_HELP_STRING([--enable-doxygen], [Enable Doxygen documentation (default: no)])],
              [enable_doxygen=$enableval],
              [enable_doxygen=no])

AS_IF([test "x$enable_doxygen" = "xyes"], [
    DX_DOXYGEN_FEATURE(ON)
    DX_HTML_FEATURE(ON)
    DX_MAN_FEATURE(ON)
    DX_RTF_FEATURE(ON)
    DX_PDF_FEATURE(ON)
    DX_XML_FEATURE(ON)
    DX_DOT_FEATURE(ON)
    DX_INIT_DOXYGEN(hypredrive, [Doxyfile], [docs])
])

AC_PROG_CC([mpicc])
AM_PROG_CC_C_O()
AM_PROG_AR
LT_INIT([disable-shared])

dnl Specify hypre include directory
AC_ARG_WITH([hypre-include],
            [AS_HELP_STRING([--with-hypre-include=DIR], [Required: Path to HYPRE include dir.])],
            [CFLAGS+=" -I$withval"],
            [AC_MSG_ERROR([--with-hypre-include=DIR is required. Please specify the path to the HYPRE include directory.])])

dnl Specify hypre lib directory
AC_ARG_WITH([hypre-lib],
            [AS_HELP_STRING([--with-hypre-lib=DIR], [Required: Path to HYPRE library dir.])],
            [LDFLAGS+=" -L$withval -Wl,-rpath,${withval}"],
            [AC_MSG_ERROR([--with-hypre-lib=DIR is required. Please specify the path to the HYPRE library directory.])])

dnl Check for CUDA support
AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda], [Enable CUDA support])],
            [with_cuda=yes],
            [with_cuda=no])

dnl Allow specifying CUDA home directory
AC_ARG_WITH([cuda-home],
            [AS_HELP_STRING([--with-cuda-home=DIR], [Specify CUDA home directory (default is taken from CUDA_HOME environment variable)])],
            [cuda_home="$withval"],
            [cuda_home="$CUDA_HOME"])

dnl Verify CUDA_HOME is defined if CUDA support is enabled
AS_IF([test "x$with_cuda" = "xyes"], [
    AS_IF([test -z "$cuda_home"], [
        AC_MSG_ERROR([CUDA home directory is not defined. Please define the CUDA_HOME environment variable or use --with-cuda-home=DIR.])
    ])
    LDFLAGS+=" -L$cuda_home/lib64 -Wl,-rpath,$cuda_home/lib64"
    LIBS+=" -lcudart -lcusparse -lcublas -lcurand -lcusolver -lstdc++"
])

dnl Check for HIP support
AC_ARG_WITH([hip],
            [AS_HELP_STRING([--with-hip], [Enable HIP support])],
            [with_hip=yes],
            [with_hip=no])

dnl Allow specifying ROCM home directory
AC_ARG_WITH([rocm-path],
            [AS_HELP_STRING([--with-rocm-path=DIR], [Specify ROCm installation directory])],
            [rocm_path="$withval"],
            [rocm_path="$ROCM_PATH"])

dnl Verify ROCM_PATH is defined if HIP support is enabled
AS_IF([test "x$with_hip" = "xyes"], [
    AS_IF([test -z "$rocm_path"], [
        AC_MSG_ERROR([ROCM home directory is not defined. Please define the ROCM_PATH environment variable or use --with-rocm-path=DIR.])
    ])
    LDFLAGS+=" -L$rocm_path/lib -Wl,-rpath,$rocm_path/lib"
    LIBS+=" -lamdhip64 -lrocsparse -lrocblas -lrocrand -lrocsolver -lstdc++"
])

dnl Avoid using CUDA and HIP
AS_IF([test "x$with_cuda" = "xyes" -a "x$with_hip" = "xyes"], [
    AC_MSG_ERROR([--with-cuda and --with-hip options are mutually exclusive. Please choose one.])
])

AC_CHECK_LIB([m], [sin],
             [],
             [AC_MSG_ERROR([libm (math library) not found or not usable])])

AC_CHECK_LIB([HYPRE], [HYPRE_IJMatrixCreate], [],
             [AC_MSG_ERROR([HYPRE library not found or not usable])])

AC_CHECK_HEADER([HYPRE_config.h], [],
                [AC_MSG_ERROR([HYPRE headers not found or not usable. Please ensure the correct path is specified with --with-hypre-include and the headers are installed.])])

AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_HEADERS([HYPREDRV_config.h])
AC_CHECK_HEADERS([stdint.h])
AC_CHECK_HEADERS([limits.h])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_FUNCS([strchr strcspn strdup memset strstr strtol sqrt])
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
