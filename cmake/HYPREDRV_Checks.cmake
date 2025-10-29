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
check_include_file(sys/stat.h HAVE_SYS_STAT_H)

# Check for functions
check_function_exists(malloc HAVE_MALLOC)

# Check for types and compiler features
check_type_size(_Bool HAVE__BOOL)

# Check for symbols
check_symbol_exists(sqrt "math.h" HAVE_SQRT)