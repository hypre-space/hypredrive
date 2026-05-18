# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

include(normpath(joinpath(@__DIR__, "constants.jl")))

const _MPITRAMPOLINE_JLL_UUID = Base.UUID("f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748")
const _HYPREDRV_ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
const _mpi_trampoline_handle = Ref{Union{Nothing,Ptr{Cvoid}}}(nothing)
const _mpi_trampoline_handle_owned = Ref(false)

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

function _mpi_trampoline_jll_module()
    try
        return Base.require(Base.PkgId(_MPITRAMPOLINE_JLL_UUID, "MPItrampoline_jll"))
    catch err
        @warn "MPItrampoline_jll is unavailable; artifact-backed HYPREDRV may fail to load MPI symbols" exception = (err, catch_backtrace())
        return nothing
    end
end

function _preload_mpi_trampoline()
    mpi_trampoline_jll = _mpi_trampoline_jll_module()
    mpi_trampoline_jll === nothing && return nothing

    lock(_state_lock)
    try
        _mpi_trampoline_handle[] !== nothing && return nothing

        candidates = String[]
        for property in (:libmpi, :libmpitrampoline, :libmpi_path, :libmpitrampoline_path)
            if isdefined(mpi_trampoline_jll, property)
                value = getproperty(mpi_trampoline_jll, property)
                value isa AbstractString && push!(candidates, String(value))
            end
        end

        libdirs = String[]
        if isdefined(mpi_trampoline_jll, :LIBPATH_list)
            append!(libdirs, mpi_trampoline_jll.LIBPATH_list)
        end
        if isdefined(mpi_trampoline_jll, :LIBPATH)
            libpath = mpi_trampoline_jll.LIBPATH[]
            if !isempty(libpath)
                append!(libdirs, split(libpath, Sys.iswindows() ? ';' : ':'))
            end
        end
        if isdefined(mpi_trampoline_jll, :artifact_dir)
            root = mpi_trampoline_jll.artifact_dir
            push!(libdirs, joinpath(root, "lib"))
            push!(libdirs, root)
        end
        unique!(libdirs)

        for libdir in libdirs
            isempty(libdir) && continue
            for name in ("libmpi.so", "libmpi.dylib", "mpi.dll",
                         "libmpitrampoline.so", "libmpitrampoline.dylib", "mpitrampoline.dll")
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
                    @warn "Failed to preload MPItrampoline library" path
                    continue
                end
                _mpi_trampoline_handle[] = handle
                _mpi_trampoline_handle_owned[] = true
                return nothing
            end
        end

        if isdefined(mpi_trampoline_jll, :libmpi_handle)
            handle = mpi_trampoline_jll.libmpi_handle
            if handle isa Ptr && handle != C_NULL
                # This handle is owned by mpi_trampoline_jll. We cache it only as a
                # sentinel and must never dlclose it.
                _mpi_trampoline_handle[] = handle
                _mpi_trampoline_handle_owned[] = false
                return nothing
            end
        end

        @warn "MPItrampoline library was not found in MPItrampoline_jll artifact"
    finally
        unlock(_state_lock)
    end
    return nothing
end

function _preload_mpi_trampoline_before_dlopen()
    try
        _preload_mpi_trampoline()
    catch err
        @warn "MPItrampoline preload failed; artifact-backed HYPREDRV may fail to load MPI symbols" exception = (err, catch_backtrace())
    end
    return nothing
end
