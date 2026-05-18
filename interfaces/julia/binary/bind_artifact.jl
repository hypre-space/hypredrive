#!/usr/bin/env julia
# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using Pkg.Artifacts
using Base.BinaryPlatforms
using SHA

include(normpath(joinpath(@__DIR__, "..", "src", "constants.jl")))

function usage()
    println(stderr, "usage: julia bind_artifact.jl <tarball> <download-url> [Artifacts.toml]")
    println(stderr, "       tarball filename must contain a BinaryBuilder platform triplet")
    println(stderr, "       or set HYPREDRV_ARTIFACT_PLATFORM to a triplet such as x86_64-linux-gnu")
    exit(2)
end

length(ARGS) in (2, 3) || usage()

tarball = abspath(ARGS[1])
url = ARGS[2]
toml = length(ARGS) == 3 ? abspath(ARGS[3]) : normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
isfile(tarball) || error("tarball does not exist: $tarball")

function platform_from_filename(path::AbstractString)
    if haskey(ENV, "HYPREDRV_ARTIFACT_PLATFORM")
        return parse(Platform, ENV["HYPREDRV_ARTIFACT_PLATFORM"])
    end

    name = basename(path)
    stem = replace(name, r"\.(tar\.gz|tgz)$" => "")
    tokens = [String(m.match) for m in eachmatch(r"[A-Za-z0-9_]+", stem)]
    # Enumerate bounded candidate triplets instead of using one greedy match, so
    # suffixes such as libgfortran5/cxx11 tags cannot consume the whole tail.
    for width in min(8, length(tokens)):-1:3
        for start in 1:(length(tokens) - width + 1)
            triplet_text = join(tokens[start:(start + width - 1)], "-")
            try
                platform = parse(Platform, triplet_text)
                triplet(platform) == triplet_text && return platform
            catch
                continue
            end
        end
    end
    error("could not derive BinaryBuilder platform triplet from tarball name: $name")
end

sha256_hash = bytes2hex(open(sha256, tarball))
tree_hash = create_artifact() do dir
    run(`tar -xzf $tarball -C $dir`)
end

platform = platform_from_filename(tarball)
existing_hash = artifact_hash(_HYPREDRV_BRIDGE_ARTIFACT_NAME, toml; platform=platform)
if existing_hash !== nothing
    if get(ENV, "HYPREDRV_ARTIFACT_REPLACE", "") != "1"
        error("artifact binding already exists for $(triplet(platform)); set HYPREDRV_ARTIFACT_REPLACE=1 to replace it")
    end
    @warn "Replacing existing HYPREDRV Julia artifact binding" artifact = _HYPREDRV_BRIDGE_ARTIFACT_NAME platform = triplet(platform) old_tree_hash = existing_hash new_tree_hash = tree_hash
end
bind_artifact!(toml, _HYPREDRV_BRIDGE_ARTIFACT_NAME, tree_hash;
               download_info=[(url, sha256_hash)], platform=platform,
               force=(existing_hash !== nothing))

println("Bound ", _HYPREDRV_BRIDGE_ARTIFACT_NAME)
println("  platform: ", triplet(platform))
println("  git-tree-sha1: ", tree_hash)
println("  sha256: ", sha256_hash)
println("  url: ", url)
println("  toml: ", toml)
