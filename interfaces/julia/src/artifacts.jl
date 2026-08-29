# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

include(normpath(joinpath(@__DIR__, "constants.jl")))

const _MPITRAMPOLINE_JLL_UUID = Base.UUID("f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748")
const _MICROSOFT_MPI_JLL_UUID = Base.UUID("9237b28f-5490-5468-be7b-bb81f5f5e6cf")
const _HYPREDRV_ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
const _mpi_runtime_handle = Ref{Union{Nothing,Ptr{Cvoid}}}(nothing)
const _mpi_runtime_handle_owned = Ref(false)

function _artifact_root()
    isfile(_HYPREDRV_ARTIFACTS_TOML) || return nothing
    try
        platform = Base.BinaryPlatforms.HostPlatform()
        LazyArtifacts.artifact_hash(_HYPREDRV_BRIDGE_ARTIFACT_NAME,
                                    _HYPREDRV_ARTIFACTS_TOML;
                                    platform=platform) === nothing && return nothing
        root = LazyArtifacts.ensure_artifact_installed(
            _HYPREDRV_BRIDGE_ARTIFACT_NAME, _HYPREDRV_ARTIFACTS_TOML; platform=platform)
        return isdir(root) ? root : nothing
    catch err
        @debug "HYPREDRV Julia artifact is unavailable" exception = (err, catch_backtrace())
        return nothing
    end
end

function _candidate_paths_from_artifact()
    root = _artifact_root()
    root === nothing && return String[]
    return _candidate_paths_from_dir(root)
end

function _mpi_runtime_jll_module()
    try
        if Sys.iswindows()
            return Base.require(Base.PkgId(_MICROSOFT_MPI_JLL_UUID, "MicrosoftMPI_jll"))
        end
        return Base.require(Base.PkgId(_MPITRAMPOLINE_JLL_UUID, "MPItrampoline_jll"))
    catch err
        @warn "The platform MPI runtime package is unavailable; artifact-backed HYPREDRV may fail to load MPI symbols" exception = (err, catch_backtrace())
        return nothing
    end
end

function _mpi_runtime_artifact_root()
    # MPItrampoline_jll is augmented by MPIPreferences.  If the user has not
    # selected MPItrampoline as the active MPI ABI, the JLL module remains
    # loadable but does not expose an artifact directory or library path.  The
    # HYPREDRV artifact is nevertheless linked against MPItrampoline, so resolve
    # that artifact directly for the preload step.  Windows uses Microsoft MPI
    # and is intentionally left to the normal JLL wrapper path above.
    Sys.iswindows() && return nothing

    try
        pkgpath = Base.locate_package(Base.PkgId(_MPITRAMPOLINE_JLL_UUID,
                                                  "MPItrampoline_jll"))
        pkgpath === nothing && return nothing

        artifacts_toml = normpath(joinpath(dirname(pkgpath), "..", "Artifacts.toml"))
        isfile(artifacts_toml) || return nothing

        platform = Base.BinaryPlatforms.HostPlatform()
        platform["mpi"] = "MPItrampoline"
        LazyArtifacts.artifact_hash("MPItrampoline", artifacts_toml;
                                    platform=platform) === nothing && return nothing
        root = LazyArtifacts.ensure_artifact_installed("MPItrampoline", artifacts_toml;
                                                       platform=platform)
        return isdir(root) ? root : nothing
    catch err
        @debug "MPItrampoline_jll artifact is unavailable" exception = (err, catch_backtrace())
        return nothing
    end
end

function _preload_mpi_runtime()
    mpi_runtime_jll = _mpi_runtime_jll_module()

    lock(_state_lock)
    try
        _mpi_runtime_handle[] !== nothing && return nothing

        candidates = String[]
        if mpi_runtime_jll !== nothing
            for property in (:libmpi, :libmpitrampoline, :libmpi_path, :libmpitrampoline_path)
                if isdefined(mpi_runtime_jll, property)
                    value = getproperty(mpi_runtime_jll, property)
                    value isa AbstractString && push!(candidates, String(value))
                end
            end
        end

        libdirs = String[]
        if mpi_runtime_jll !== nothing
            if isdefined(mpi_runtime_jll, :LIBPATH_list)
                append!(libdirs, mpi_runtime_jll.LIBPATH_list)
            end
            if isdefined(mpi_runtime_jll, :LIBPATH)
                libpath = mpi_runtime_jll.LIBPATH[]
                if !isempty(libpath)
                    append!(libdirs, split(libpath, Sys.iswindows() ? ';' : ':'))
                end
            end
            if isdefined(mpi_runtime_jll, :artifact_dir)
                root = mpi_runtime_jll.artifact_dir
                push!(libdirs, joinpath(root, "lib"))
                push!(libdirs, joinpath(root, "bin"))
                push!(libdirs, root)
            end
        end

        fallback_root = _mpi_runtime_artifact_root()
        if fallback_root !== nothing
            push!(libdirs, joinpath(fallback_root, "lib"))
            push!(libdirs, joinpath(fallback_root, "bin"))
            push!(libdirs, fallback_root)
        end
        unique!(libdirs)

        for libdir in libdirs
            isempty(libdir) && continue
            for name in ("libmpi.so", "libmpi.dylib", "mpi.dll",
                         "libmpitrampoline.so", "libmpitrampoline.so.5",
                         "libmpitrampoline.dylib", "libmpitrampoline.5.dylib",
                         "libmpitrampoline.5.0.0.dylib", "mpitrampoline.dll",
                         "msmpi.dll", "libmpi.dll")
                push!(candidates, joinpath(libdir, name))
            end
        end

        for path in unique(candidates)
            if isfile(path)
                # Keep this handle open for the process lifetime. Closing an MPI
                # runtime library while HYPREDRV or Julia still has live symbols is
                # more dangerous than the tiny intentional process-lifetime handle.
                handle = Libdl.dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL; throw_error=false)
                if handle == C_NULL
                    @warn "Failed to preload MPI runtime library" path
                    continue
                end
                _mpi_runtime_handle[] = handle
                _mpi_runtime_handle_owned[] = true
                return nothing
            end
        end

        if mpi_runtime_jll !== nothing && isdefined(mpi_runtime_jll, :libmpi_handle)
            handle = mpi_runtime_jll.libmpi_handle
            if handle isa Ptr && handle != C_NULL
                # This handle is owned by the runtime JLL. We cache it only as a
                # sentinel and must never dlclose it.
                _mpi_runtime_handle[] = handle
                _mpi_runtime_handle_owned[] = false
                return nothing
            end
        end

        @warn "Platform MPI runtime library was not found in its JLL artifact"
    finally
        unlock(_state_lock)
    end
    return nothing
end

function _preload_mpi_runtime_before_dlopen()
    try
        _preload_mpi_runtime()
    catch err
        @warn "MPI runtime preload failed; artifact-backed HYPREDRV may fail to load MPI symbols" exception = (err, catch_backtrace())
    end
    return nothing
end
