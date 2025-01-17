cmake_minimum_required(VERSION 3.12)
project(poisson C)

# Disallow in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

set(CMAKE_C_STANDARD 99)

# Add the cmake folder to the module path
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# Find HYPREDRV package
find_package(HYPREDRV REQUIRED)

# Add the executable
add_executable(poisson main.c)

# Link against HYPREDRV
target_link_libraries(poisson PRIVATE HYPREDRV::HYPREDRV)
