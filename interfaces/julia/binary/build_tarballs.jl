#!/usr/bin/env julia
# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using BinaryBuilder

name = "HYPREDRV"
version = VersionNumber(get(ENV, "HYPREDRV_BINARY_VERSION", "0.2.0"))
repo_url = get(ENV, "HYPREDRV_BINARY_REPOSITORY", "https://github.com/hypre-space/hypredrive.git")
default_hypre_ref = "341f9089807934407a52ea8324759f3af1e49a57"

function resolve_git_ref(repo::AbstractString, ref::AbstractString)
    if occursin(r"^[0-9a-fA-F]{40}$", ref)
        return String(ref)
    end

    for pattern in ("refs/heads/$ref", "refs/tags/$ref")
        line = readchomp(`git ls-remote $repo $pattern`)
        isempty(line) || return String(first(split(line)))
    end
    error("failed to resolve git ref '$ref' for $repo")
end

source_ref = resolve_git_ref(repo_url, get(ENV, "HYPREDRV_BINARY_GIT_SHA",
                                           get(ENV, "GITHUB_SHA", "master")))
hypre_ref = get(ENV, "HYPREDRV_BINARY_HYPRE_GIT_SHA", default_hypre_ref)

sources = [
    GitSource(repo_url, source_ref; unpack_target="hypredrive"),
    GitSource("https://github.com/hypre-space/hypre.git", hypre_ref; unpack_target="hypre"),
]

script = replace(raw"""
cd ${WORKSPACE}/srcdir/hypredrive
cmake -S . -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_PREFIX_PATH=${prefix} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DHYPRE_VERSION=@HYPRE_REF@ \
    -DFETCHCONTENT_SOURCE_DIR_HYPRE=${WORKSPACE}/srcdir/hypre \
    -DHYPREDRV_ENABLE_JULIA=ON \
    -DHYPREDRV_ENABLE_TESTING=OFF \
    -DHYPREDRV_ENABLE_EXAMPLES=OFF \
    -DHYPRE_ENABLE_MPI=ON \
    -DHYPRE_ENABLE_FORTRAN=OFF \
    -DHYPRE_ENABLE_CUDA=OFF \
    -DHYPRE_ENABLE_HIP=OFF \
    -DHYPRE_ENABLE_SYCL=OFF \
    -DCMAKE_C_COMPILER=${CC} \
    -DMPI_C_COMPILER=${CC} \
    -DMPI_C_HEADER_DIR=${prefix}/include \
    -DMPI_C_INCLUDE_DIRS=${prefix}/include \
    -DMPI_C_LIB_NAMES=mpi \
    -DMPI_mpi_LIBRARY=${prefix}/lib/libmpi.${dlext} \
    -DMPI_C_LIBRARIES=${prefix}/lib/libmpi.${dlext} \
    -DMPI_C_WORKS=TRUE
cmake --build build --target install --parallel ${nproc}
""", "@HYPRE_REF@" => hypre_ref)

platforms = [Platform("x86_64", "linux"; libc="glibc")]

products = [
    LibraryProduct("libHYPREDRV", :libHYPREDRV; dont_dlopen=true),
    LibraryProduct("libHYPREDRV_Julia", :libHYPREDRV_Julia; dont_dlopen=true),
]

dependencies = [
    BuildDependency("CMake_jll"),
    BuildDependency("Ninja_jll"),
    Dependency("MPItrampoline_jll"; compat="5.5"),
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.9")
