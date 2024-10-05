include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckFunctionExists)
include(CheckTypeSize)
include(CheckCSourceCompiles)
include(CheckSymbolExists)

# Check for libraries
if(TARGET HYPRE::HYPRE)
  set(HAVE_LIBHYPRE 1)
endif()
check_library_exists(m sqrt "" HAVE_LIBM)
if(HAVE_LIBM)
  set(CMAKE_REQUIRED_LIBRARIES m)
endif()

# Check for headers
check_include_file(dlfcn.h HAVE_DLFCN_H)
check_include_file(inttypes.h HAVE_INTTYPES_H)
check_include_file(limits.h HAVE_LIMITS_H)
check_include_file(stddef.h HAVE_STDDEF_H)
check_include_file(stdint.h HAVE_STDINT_H)
check_include_file(stdio.h HAVE_STDIO_H)
check_include_file(stdlib.h HAVE_STDLIB_H)
check_include_file(strings.h HAVE_STRINGS_H)
check_include_file(string.h HAVE_STRING_H)
check_include_file(sys/stat.h HAVE_SYS_STAT_H)
check_include_file(sys/types.h HAVE_SYS_TYPES_H)
check_include_file(unistd.h HAVE_UNISTD_H)

# Check for functions
check_function_exists(malloc HAVE_MALLOC)
check_function_exists(memset HAVE_MEMSET)
check_function_exists(realloc HAVE_REALLOC)
check_function_exists(strchr HAVE_STRCHR)
check_function_exists(strcspn HAVE_STRCSPN)
check_function_exists(strdup HAVE_STRDUP)
check_function_exists(strstr HAVE_STRSTR)
check_function_exists(strtol HAVE_STRTOL)

# Check for types and compiler features
check_type_size(_Bool HAVE__BOOL)
check_type_size(size_t SIZE_T)
check_type_size(uint32_t UINT32_T)

# Check for symbols
check_symbol_exists(sqrt "math.h" HAVE_SQRT)

# Set package information
set(PACKAGE ${PROJECT_NAME})
set(PACKAGE_NAME ${PROJECT_NAME})
set(PACKAGE_VERSION ${PROJECT_VERSION})
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME ${PROJECT_NAME})
set(PACKAGE_URL "${PROJECT_URL}")
set(PACKAGE_BUGREPORT "${PROJECT_BUGREPORT}")