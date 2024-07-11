AC_PREREQ([2.69])
AC_INIT([hypredrive],
        [0.1],
        [https://github.com/hypre-space/hypredrive/issues],
        [hypredrive],
        [https://github.com/hypre-space/hypredrive])
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

dnl Check for --with-hypre-dir first
AC_ARG_WITH([hypre-dir],
            [AS_HELP_STRING([--with-hypre-dir=DIR], [Optional: Path to HYPRE directory. Specifies both include and library dirs.])],
            [
             CFLAGS+=" -I$withval/include"
             LDFLAGS+=" -L$withval/lib -Wl,-rpath,$withval/lib"
            ])

dnl Specify hypre include directory only if hypre-dir is not specified
AS_IF([test -z "$with_hypre_dir"], [
  AC_ARG_WITH([hypre-include],
              [AS_HELP_STRING([--with-hypre-include=DIR], [Required if --with-hypre-dir is not specified: Path to HYPRE include dir.])],
              [CFLAGS+=" -I$withval"],
              [AC_MSG_ERROR([--with-hypre-include=DIR or --with-hypre-dir=DIR is required. Please specify the path to the HYPRE include directory.])])
])

dnl Specify hypre lib directory only if hypre-dir is not specified
AS_IF([test -z "$with_hypre_dir"], [
  AC_ARG_WITH([hypre-lib],
              [AS_HELP_STRING([--with-hypre-lib=DIR], [Required if --with-hypre-dir is not specified: Path to HYPRE library dir.])],
              [LDFLAGS+=" -L$withval -Wl,-rpath,$withval"],
              [AC_MSG_ERROR([--with-hypre-lib=DIR or --with-hypre-dir=DIR is required. Please specify the path to the HYPRE library directory.])])
])

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

dnl Verify CUDA support
AS_IF([test "x$with_cuda" = "xyes"], [
    dnl Check for CUDA_HOME
    AS_IF([test -z "$cuda_home"], [
        AC_MSG_ERROR([CUDA home directory is not defined. Please define the CUDA_HOME environment variable or use --with-cuda-home=DIR.])
    ])

    dnl Update LDFLAGS and LIBS
    LDFLAGS+=" -L$cuda_home/lib64 -Wl,-rpath,$cuda_home/lib64"
    LIBS+=" -lcudart -lcusparse -lcublas -lcurand -lcusolver -lstdc++"

    dnl Check for libcudart
    AC_CHECK_LIB([cudart],
                 [cudaMalloc],
                 [AC_MSG_NOTICE([libcudart found.])],
                 [AC_MSG_ERROR([libcudart not found. Please ensure CUDA is installed correctly.])])

    dnl Check for libcudart
    AC_CHECK_LIB([cusparse],
                 [cusparseCreate],
                 [AC_MSG_NOTICE([libcusparse found.])],
                 [AC_MSG_ERROR([libcusparse not found. Please ensure CUDA is installed correctly.])])

    dnl Check for libcublas
    AC_CHECK_LIB([cublas],
                 [cublasInit],
                 [AC_MSG_NOTICE([libcublas found.])],
                 [AC_MSG_ERROR([libcublas not found. Please ensure CUDA is installed correctly.])])

    dnl Check for libcusolver
    AC_CHECK_LIB([cusolver],
                 [cusolverDnCreate],
                 [AC_MSG_NOTICE([libcusolver found.])],
                 [AC_MSG_ERROR([libcusolver not found. Please ensure CUDA is installed correctly.])])

    dnl Check for libcurand
    AC_CHECK_LIB([curand],
                 [curandCreateGenerator],
                 [AC_MSG_NOTICE([libcurand found.])],
                 [AC_MSG_ERROR([libcurand not found. Please ensure CUDA is installed correctly.])])
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
    dnl Check for ROCM_PATH
    AS_IF([test -z "$rocm_path"], [
        AC_MSG_ERROR([ROCM home directory is not defined. Please define the ROCM_PATH environment variable or use --with-rocm-path=DIR.])
    ])

    dnl Update LDFLAGS and LIBS
    LDFLAGS+=" -L$rocm_path/lib -Wl,-rpath,$rocm_path/lib"
    LIBS+=" -lamdhip64 -lrocsparse -lrocblas -lrocrand -lrocsolver -lstdc++"


    dnl Check for libamdhip64
    AC_CHECK_LIB([amdhip64],
                 [hipMalloc],
                 [AC_MSG_NOTICE([libamdhip64 found.])],
                 [AC_MSG_ERROR([libamdhip64 not found. Please ensure ROCm is installed correctly.])])

    dnl Check for librocsparse
    AC_CHECK_LIB([rocsparse],
                 [rocsparse_create_handle],
                 [AC_MSG_NOTICE([librocsparse found.])],
                 [AC_MSG_ERROR([librocsparse not found. Please ensure ROCm is installed correctly.])])

    dnl Check for librocblas
    AC_CHECK_LIB([rocblas],
                 [rocblas_create_handle],
                 [AC_MSG_NOTICE([librocblas found.])],
                 [AC_MSG_ERROR([librocblas not found. Please ensure ROCm is installed correctly.])])

    dnl Check for librocrand
    AC_CHECK_LIB([rocrand],
                 [rocrand_create_generator],
                 [AC_MSG_NOTICE([librocrand found.])],
                 [AC_MSG_ERROR([librocrand not found. Please ensure ROCm is installed correctly.])])

    dnl Check for librocsolver
    AC_CHECK_LIB([rocsolver],
                 [rocsolver_create_handle],
                 [AC_MSG_NOTICE([librocsolver found.])],
                 [AC_MSG_ERROR([librocsolver not found. Please ensure ROCm is installed correctly.])])
])

dnl Avoid using CUDA and HIP
AS_IF([test "x$with_cuda" = "xyes" -a "x$with_hip" = "xyes"], [
    AC_MSG_ERROR([--with-cuda and --with-hip options are mutually exclusive. Please choose one.])
])

dnl Check for libm
AC_CHECK_LIB([m], [sin], [],
             [AC_MSG_ERROR([libm (math library) not found or not usable])])

dnl Check for libHYPRE
AC_CHECK_LIB([HYPRE], [HYPRE_IJMatrixCreate], [],
             [AC_MSG_ERROR([HYPRE library not found or not usable])])

dnl Check for HYPRE headers
AC_CHECK_HEADER([HYPRE_config.h], [],
                [AC_MSG_ERROR([HYPRE headers not found or not usable. Please ensure the correct path is specified with --with-hypre-include and the headers are installed.])])

dnl Check for minimum HYPRE version
AC_LANG_PUSH([C])
AC_MSG_CHECKING([for HYPRE version])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([#include <HYPRE_config.h>],
                                   [#if HYPRE_RELEASE_NUMBER < 23100
                                    #error "HYPRE version too old"
                                    #endif])],
                  [AC_MSG_RESULT([sufficient])],
                  [AC_MSG_ERROR([HYPRE library version is too old. Use hypre >= 2.31.0])])
AC_LANG_POP([C])

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
