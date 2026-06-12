# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

module HypreDrive

using LazyArtifacts
using Libdl
using SparseArrays

export hypredrive_error, SolveInfo, hypredrive_initialize, hypredrive_library_path,
       hypredrive_comm_rank, hypredrive_comm_size,
       hypredrive_mpi_world_rank, hypredrive_mpi_world_size, hypredrive_mpi_world_sum,
       hypredrive_options, hypredrive_shutdown, hypredrive_solve,
       hypredrive_solve_mpi, hypredrive_solve_mpi_csr,
       HypreDriveSession, set_matrix_csr!, update_matrix_csr!, set_rhs!, setup!, solve!,
       info

struct hypredrive_error <: Exception
    code::UInt32
    operation::String
end

Base.showerror(io::IO, err::hypredrive_error) =
    print(io, "HYPREDRV call failed in ", err.operation, " with error code ", err.code)

struct SolveInfo
    iterations::Int
    setup_time::Float64
    solve_time::Float64
    solution_norm::Float64
    converged::Bool
    final_res_norm::Float64
end

mutable struct HypreDriveSession
    drv_ref::Ref{Ptr{Cvoid}}
    comm::Symbol
    row_start::Int
    row_end::Int
    matrix_revision::Int
    setup_revision::Int
    matrix_set::Bool
    rhs_set::Bool
    solver_created::Bool
    closed::Bool
    last_info::SolveInfo
end

const _state_lock = ReentrantLock()
const _library = Ref{String}("")
const _initialized = Ref(false)
const _atexit_registered = Ref(false)
const _scalar_abi_checked = Ref(false)
const _index_type_cache = Ref{Union{Nothing,DataType}}(nothing)

include("artifacts.jl")

function _library_names()
    if Sys.iswindows()
        return ("HYPREDRV_Julia.dll", "libHYPREDRV_Julia.dll")
    elseif Sys.isapple()
        return ("libHYPREDRV_Julia.dylib", "HYPREDRV_Julia.dylib")
    else
        return ("libHYPREDRV_Julia.so", "HYPREDRV_Julia.so")
    end
end

function _candidate_paths_from_dir(root::AbstractString)
    paths = String[]
    for subdir in (joinpath("lib", "julia"), "lib", joinpath("lib64", "julia"), "lib64", "")
        dir = isempty(subdir) ? root : joinpath(root, subdir)
        for name in _library_names()
            push!(paths, joinpath(dir, name))
        end
    end
    return paths
end

function _candidate_paths_from_prefix_env()
    if haskey(ENV, "HYPREDRV_PREFIX")
        root = ENV["HYPREDRV_PREFIX"]
        return "HYPREDRV_PREFIX", root, _candidate_paths_from_dir(root)
    end
    if haskey(ENV, "HYPREDRV_DIR")
        root = ENV["HYPREDRV_DIR"]
        return "HYPREDRV_DIR", root, _candidate_paths_from_dir(root)
    end
    return nothing, nothing, String[]
end

function _candidate_paths_from_source_tree()
    paths = String[]
    package_dir = dirname(@__DIR__)
    repo_root = dirname(dirname(package_dir))
    isdir(repo_root) || return paths

    for entry in readdir(repo_root; join=true)
        isdir(entry) || continue
        startswith(basename(entry), "build") || continue
        for name in _library_names()
            push!(paths, joinpath(entry, "interfaces", "julia", "lib", name))
            push!(paths, joinpath(entry, "lib", "julia", name))
            push!(paths, joinpath(entry, "lib", name))
        end
    end
    return paths
end

function _discover_library()
    if haskey(ENV, "HYPREDRV_LIBRARY")
        path = ENV["HYPREDRV_LIBRARY"]
        isfile(path) && return path
        error("HYPREDRV_LIBRARY is set but does not point to a file: $path")
    end

    env_name, env_root, env_paths = _candidate_paths_from_prefix_env()
    if env_name !== nothing
        for path in env_paths
            isfile(path) && return path
        end
        error("$env_name is set, but libHYPREDRV_Julia was not found below $env_root")
    end

    for path in _candidate_paths_from_artifact()
        if isfile(path)
            # MPItrampoline preload is artifact-only by design. Source-tree and
            # install-prefix builds should use their platform MPI runtime
            # directly instead of forcing MPItrampoline into the process.
            _preload_mpi_trampoline_before_dlopen()
            return path
        end
    end

    for path in _candidate_paths_from_source_tree()
        isfile(path) && return path
    end

    found = Libdl.find_library(["HYPREDRV_Julia", "libHYPREDRV_Julia"])
    !isempty(found) && return found

    error("Could not locate libHYPREDRV_Julia. Build HYPREDRV with -DHYPREDRV_ENABLE_JULIA=ON, set HYPREDRV_PREFIX to the install prefix, or set HYPREDRV_LIBRARY to the full bridge-library path.")
end

function hypredrive_library_path()
    lock(_state_lock)
    try
        if isempty(_library[])
            _library[] = _discover_library()
        end
        return _library[]
    finally
        unlock(_state_lock)
    end
end

_lib() = hypredrive_library_path()

function _describe(code::UInt32)
    ccall((:HYPREDRV_JuliaErrorCodeDescribe, _lib()), Cvoid, (UInt32,), code)
end

function _check(code::UInt32, operation::AbstractString)
    if code != 0x00000000
        _describe(code)
        throw(hypredrive_error(code, String(operation)))
    end
    return nothing
end

function _warn_shutdown_failure(code::UInt32, operation::AbstractString)
    code == 0x00000000 && return nothing
    try
        @warn "HYPREDRV shutdown call failed" operation = operation code = code
    catch
        println(stderr, "HYPREDRV shutdown call failed in $operation with error code $code")
    end
    return nothing
end

function hypredrive_initialize()
    lock(_state_lock)
    try
        if !_atexit_registered[]
            atexit(() -> _hypredrive_shutdown(false))
            _atexit_registered[] = true
        end
        if !_initialized[]
            mpi_started = false
            runtime_started = false
            try
                _check(ccall((:HYPREDRV_JuliaMPIInitialize, _lib()), UInt32, ()), "MPI initialize")
                mpi_started = true
                _check(ccall((:HYPREDRV_JuliaInitialize, _lib()), UInt32, ()), "HYPREDRV initialize")
                runtime_started = true
                _check_scalar_abi()
                _initialized[] = true
            catch
                if runtime_started
                    _warn_shutdown_failure(ccall((:HYPREDRV_JuliaFinalize, _lib()), UInt32, ()),
                                           "HYPREDRV finalize after failed initialize")
                end
                if mpi_started
                    _warn_shutdown_failure(ccall((:HYPREDRV_JuliaMPIFinalize, _lib()), UInt32, ()),
                                           "MPI finalize after failed initialize")
                end
                rethrow()
            end
        end
    finally
        unlock(_state_lock)
    end
    return nothing
end

function _hypredrive_shutdown(throw_errors::Bool)
    lock(_state_lock)
    try
        if _initialized[] && !isempty(_library[])
            code1 = ccall((:HYPREDRV_JuliaFinalize, _lib()), UInt32, ())
            code2 = ccall((:HYPREDRV_JuliaMPIFinalize, _lib()), UInt32, ())
            ok = code1 == 0x00000000 && code2 == 0x00000000
            if ok
                _initialized[] = false
                _scalar_abi_checked[] = false
                _index_type_cache[] = nothing
            end
            if throw_errors
                _check(code1, "HYPREDRV finalize")
                _check(code2, "MPI finalize")
            elseif !ok
                _warn_shutdown_failure(code1, "HYPREDRV finalize")
                _warn_shutdown_failure(code2, "MPI finalize")
            end
        end
    finally
        unlock(_state_lock)
    end
    return nothing
end

hypredrive_shutdown() = _hypredrive_shutdown(true)

function hypredrive_mpi_world_rank()
    hypredrive_initialize()
    rank = Ref{Cint}(0)
    _check(ccall((:HYPREDRV_JuliaWorldCommRank, _lib()), UInt32, (Ref{Cint},), rank), "MPI rank")
    return Int(rank[])
end

function hypredrive_mpi_world_size()
    hypredrive_initialize()
    size = Ref{Cint}(1)
    _check(ccall((:HYPREDRV_JuliaWorldCommSize, _lib()), UInt32, (Ref{Cint},), size), "MPI size")
    return Int(size[])
end

function hypredrive_comm_rank(comm::Symbol=:world)
    if comm === :self
        hypredrive_initialize()
        rank = Ref{Cint}(0)
        _check(ccall((:HYPREDRV_JuliaSelfCommRank, _lib()), UInt32, (Ref{Cint},), rank),
               "MPI self rank")
        return Int(rank[])
    end
    comm === :world && return hypredrive_mpi_world_rank()
    throw(ArgumentError("comm must be :world or :self"))
end

function hypredrive_comm_size(comm::Symbol=:world)
    if comm === :self
        hypredrive_initialize()
        size = Ref{Cint}(1)
        _check(ccall((:HYPREDRV_JuliaSelfCommSize, _lib()), UInt32, (Ref{Cint},), size),
               "MPI self size")
        return Int(size[])
    end
    comm === :world && return hypredrive_mpi_world_size()
    throw(ArgumentError("comm must be :world or :self"))
end

function hypredrive_mpi_world_sum(value::Real)
    hypredrive_initialize()
    total_ref = Ref{Cdouble}(0)
    _check(ccall((:HYPREDRV_JuliaWorldAllreduceDoubleSum, _lib()), UInt32,
                 (Cdouble, Ref{Cdouble}), Cdouble(value), total_ref), "MPI allreduce")
    return Float64(total_ref[])
end

function _mpi_world_allgatherv(values::Vector{Float64}, counts::Vector{Cint},
                               displs::Vector{Cint})
    hypredrive_initialize()
    length(counts) == length(displs) || throw(ArgumentError("counts and displacements differ"))
    all(>=(0), counts) || throw(ArgumentError("counts must be nonnegative"))
    all(>=(0), displs) || throw(ArgumentError("displacements must be nonnegative"))
    length(values) <= typemax(Cint) ||
        throw(ArgumentError("MPI allgatherv send count exceeds Cint range"))
    total = isempty(counts) ? 0 : maximum(Int.(displs) .+ Int.(counts))
    gathered = Vector{Float64}(undef, total)
    GC.@preserve values counts displs gathered begin
        _check(ccall((:HYPREDRV_JuliaWorldAllgathervDouble, _lib()), UInt32,
                     (Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                     pointer(values), Cint(length(values)), pointer(gathered),
                     pointer(counts), pointer(displs)), "MPI allgatherv")
    end
    return gathered
end

function _index_type()
    lock(_state_lock)
    try
        _index_type_cache[] !== nothing && return _index_type_cache[]::DataType
        n = ccall((:HYPREDRV_JuliaBigIntSize, _lib()), Csize_t, ())
        T = n == 4 ? Int32 : n == 8 ? Int64 : nothing
        T === nothing && error("Unsupported HYPRE_BigInt size: $n bytes")
        _index_type_cache[] = T
        return T
    finally
        unlock(_state_lock)
    end
end

function _check_scalar_abi()
    lock(_state_lock)
    try
        _scalar_abi_checked[] && return nothing
        real_size = ccall((:HYPREDRV_JuliaRealSize, _lib()), Csize_t, ())
        entry_size = ccall((:HYPREDRV_JuliaSolutionEntrySize, _lib()), Csize_t, ())
        real_size == sizeof(Float64) || error("Julia interface requires HYPRE_Real to be double precision; got $real_size bytes")
        entry_size == sizeof(Float64) || error("Julia interface does not support complex HYPRE builds; solution entries are $entry_size bytes")
        _scalar_abi_checked[] = true
    finally
        unlock(_state_lock)
    end
    return nothing
end

function _to_string_key_dict(x)
    if x isa NamedTuple
        return Dict{String,Any}(String(k) => _normalize_value(v) for (k, v) in pairs(x))
    elseif x isa AbstractDict
        return Dict{String,Any}(String(k) => _normalize_value(v) for (k, v) in pairs(x))
    else
        error("expected a NamedTuple or Dict, got $(typeof(x))")
    end
end

_normalize_value(v) = (v isa NamedTuple || v isa AbstractDict) ? _to_string_key_dict(v) : v

function _emit_value(io::IO, value, indent::Int)
    if value isa AbstractDict
        if isempty(value)
            println(io, " {}")
        else
            println(io)
            _emit_mapping(io, value, indent + 2)
        end
    elseif value isa AbstractVector
        println(io)
        _emit_sequence(io, value, indent + 2)
    elseif value isa Bool
        println(io, " ", value ? "true" : "false")
    elseif value isa Symbol
        println(io, " ", String(value))
    else
        println(io, " ", value)
    end
end

function _emit_sequence(io::IO, values::AbstractVector, indent::Int)
    prefix = repeat(" ", indent)
    for value in values
        print(io, prefix, "-")
        if value isa AbstractDict
            if isempty(value)
                println(io, " {}")
            else
                println(io)
                _emit_mapping(io, value, indent + 2)
            end
        elseif value isa AbstractVector
            println(io)
            _emit_sequence(io, value, indent + 2)
        else
            _emit_value(io, value, indent)
        end
    end
end

function _emit_mapping(io::IO, mapping::AbstractDict, indent::Int)
    prefix = repeat(" ", indent)
    for key in sort(collect(keys(mapping)); by=String)
        print(io, prefix, key, ":")
        _emit_value(io, mapping[key], indent)
    end
end

function _emit_yaml(mapping::AbstractDict)
    io = IOBuffer()
    _emit_mapping(io, mapping, 0)
    return String(take!(io))
end

function _quiet_general_yaml()
    return _emit_yaml(Dict{String,Any}("general" => Dict{String,Any}("statistics" => 0)))
end

function _default_options()
    return Dict{String,Any}(
        "general" => Dict{String,Any}("statistics" => 0),
        "solver" => Dict{String,Any}(
            "pcg" => Dict{String,Any}("max_iter" => 200, "relative_tol" => 1.0e-8),
        ),
        "preconditioner" => Dict{String,Any}(
            "amg" => Dict{String,Any}("print_level" => 0),
        ),
    )
end

function hypredrive_options(; kwargs...)
    isempty(kwargs) && return _emit_yaml(_default_options())

    method_options = Dict{String,Any}()
    general = Dict{String,Any}()
    output = Dict{String,Any}("general" => general)

    solver = nothing
    preconditioner = nothing
    for (key, value) in pairs(kwargs)
        if key === :solver
            solver = String(value)
        elseif key === :preconditioner
            preconditioner = String(value)
        elseif key === :general
            merge!(general, _to_string_key_dict(value))
        else
            method_options[String(key)] = _normalize_value(value)
        end
    end
    get!(general, "statistics", 0)

    if solver !== nothing
        output["solver"] = Dict{String,Any}(solver => get(method_options, solver, Dict{String,Any}()))
    end
    if preconditioner !== nothing
        output["preconditioner"] = Dict{String,Any}(preconditioner => get(method_options, preconditioner, Dict{String,Any}()))
    end

    return _emit_yaml(output)
end

function hypredrive_options(options::AbstractString)
    text = String(options)
    return _ensure_quiet_statistics_yaml(text)
end

function _top_level_key_line(line::AbstractString, key::AbstractString)
    if startswith(line, " ") || startswith(line, "\t")
        return false
    end
    colon = findfirst(==(':'), line)
    colon === nothing && return false
    return strip(line[begin:prevind(line, colon)]) == key
end

function _has_top_level_key(text::AbstractString, key::AbstractString)
    for line in split(text, '\n'; keepempty=true)
        isempty(strip(line)) && continue
        _top_level_key_line(line, key) && return true
    end
    return false
end

function _ensure_quiet_statistics_yaml(text::AbstractString)
    text = String(text)
    lines = split(text, '\n'; keepempty=true)
    general_index = findfirst(line -> _top_level_key_line(line, "general"), lines)
    general_index === nothing && return _quiet_general_yaml() * text

    next_top = length(lines) + 1
    for i in (general_index + 1):length(lines)
        isempty(strip(lines[i])) && continue
        if !startswith(lines[i], " ") && !startswith(lines[i], "\t")
            next_top = i
            break
        end
    end

    for i in (general_index + 1):(next_top - 1)
        startswith(strip(lines[i]), "statistics:") && return text
    end

    if occursin(r"^general\s*:\s*\{\s*\}\s*$", lines[general_index])
        lines[general_index] = "general:"
        insert!(lines, general_index + 1, "  statistics: 0")
    elseif occursin(r"^general\s*:\s*$", lines[general_index])
        insert!(lines, general_index + 1, "  statistics: 0")
    else
        throw(ArgumentError("string options with a non-mapping top-level general: block cannot be made quiet automatically; use a mapping and add statistics: 0"))
    end
    return join(lines, '\n')
end

hypredrive_options(options::NamedTuple) = hypredrive_options(; options...)

function hypredrive_options(options::AbstractDict)
    output = _to_string_key_dict(options)
    if !haskey(output, "general")
        output["general"] = Dict{String,Any}("statistics" => 0)
    elseif output["general"] isa AbstractDict && !haskey(output["general"], "statistics")
        output["general"]["statistics"] = 0
    end
    return _emit_yaml(output)
end

function _csr_rows_from_sparse(A::SparseMatrixCSC{Tv,Ti}, ::Type{T}, row_first::Int, row_last::Int) where {Tv,Ti,T<:Union{Int32,Int64}}
    m, n = size(A)
    m == n || throw(ArgumentError("matrix must be square"))
    1 <= row_first <= row_last <= n || throw(ArgumentError("invalid local row range"))
    n <= Int(typemax(T)) || throw(ArgumentError("matrix dimension exceeds HYPRE_BigInt range"))

    local_n = row_last - row_first + 1
    row_counts = zeros(T, local_n)
    rows = rowvals(A)
    vals = nonzeros(A)
    for col in 1:n
        for p in nzrange(A, col)
            row = rows[p]
            if row_first <= row <= row_last
                row_counts[row - row_first + 1] += one(T)
            end
        end
    end

    indptr = Vector{T}(undef, local_n + 1)
    indptr[1] = zero(T)
    for i in 1:local_n
        indptr[i + 1] = indptr[i] + row_counts[i]
    end

    local_nnz = Int(indptr[end])
    col_indices = Vector{T}(undef, local_nnz)
    data = Vector{Float64}(undef, local_nnz)
    next = Vector{T}(undef, local_n)
    for i in 1:local_n
        next[i] = indptr[i]
    end
    for col in 1:n
        col_index = T(col - 1)
        for p in nzrange(A, col)
            row = rows[p]
            if row_first <= row <= row_last
                local_row = row - row_first + 1
                dest = Int(next[local_row]) + 1
                col_indices[dest] = col_index
                data[dest] = Float64(vals[p])
                next[local_row] += one(T)
            end
        end
    end

    return indptr, col_indices, data
end

function _as_sparse_float64(A::SparseMatrixCSC{Float64})
    return A
end

function _as_sparse_float64(A::SparseMatrixCSC)
    return SparseMatrixCSC(size(A, 1), size(A, 2), copy(A.colptr), copy(A.rowval), Float64.(A.nzval))
end

function _as_sparse_float64(A::AbstractMatrix)
    return sparse(Float64.(A))
end

_as_vector(::Type{T}, data::Vector{T}) where {T} = data
_as_vector(::Type{T}, data) where {T} = Vector{T}(data)

function _checked_cint_vector(data, name::AbstractString)
    length(data) <= typemax(Cint) || throw(ArgumentError("$name length exceeds Cint range"))
    out = Vector{Cint}(undef, length(data))
    for i in eachindex(data)
        value = data[i]
        value isa Integer || throw(ArgumentError("$name entries must be integers"))
        typemin(Cint) <= value <= typemax(Cint) ||
            throw(ArgumentError("$name entry $value exceeds Cint range"))
        out[i] = Cint(value)
    end
    return out
end

function _partition_rows(n::Int, rank::Int, size::Int)
    base = div(n, size)
    extra = rem(n, size)
    local_n = base + (rank < extra ? 1 : 0)
    start0 = rank * base + min(rank, extra)
    return start0, start0 + local_n - 1
end

function _parse_input_args!(drv::Ptr{Cvoid}, yaml::AbstractString,
                            input_args::Union{Nothing,Vector{String}})
    if input_args === nothing
        _check(ccall((:HYPREDRV_JuliaInputArgsParseYaml, _lib()), UInt32,
                     (Ptr{Cvoid}, Cstring), drv, yaml), "parse options")
        return nothing
    end

    !isempty(input_args) || throw(ArgumentError("input_args must not be empty"))
    cargs = Vector{Cstring}(undef, length(input_args))
    GC.@preserve input_args begin
        for i in eachindex(input_args)
            cargs[i] = Base.unsafe_convert(Cstring, input_args[i])
        end
        _check(ccall((:HYPREDRV_JuliaInputArgsParseArgv, _lib()), UInt32,
                     (Ptr{Cvoid}, Cint, Ptr{Cstring}), drv, Cint(length(cargs)),
                     pointer(cargs)), "parse options")
    end
    return nothing
end

function _create_driver(comm::Symbol)
    comm in (:world, :self) || throw(ArgumentError("comm must be :world or :self"))
    hypredrive_initialize()
    drv_ref = Ref{Ptr{Cvoid}}(C_NULL)
    if comm === :world
        _check(ccall((:HYPREDRV_JuliaCreateWithWorld, _lib()), UInt32,
                     (Ref{Ptr{Cvoid}},), drv_ref), "create driver")
    else
        _check(ccall((:HYPREDRV_JuliaCreateWithSelf, _lib()), UInt32,
                     (Ref{Ptr{Cvoid}},), drv_ref), "create driver")
    end
    _check(ccall((:HYPREDRV_JuliaSetLibraryMode, _lib()), UInt32, (Ptr{Cvoid},),
                 drv_ref[]), "set library mode")
    return drv_ref
end

function _destroy_solver!(session::HypreDriveSession; warn_only::Bool=false)
    if session.solver_created && session.drv_ref[] != C_NULL
        code = ccall((:HYPREDRV_JuliaLinearSolverDestroy, _lib()), UInt32,
                     (Ptr{Cvoid},), session.drv_ref[])
        session.solver_created = false
        session.setup_revision = -1
        if warn_only
            _warn_shutdown_failure(code, "destroy solver")
        else
            _check(code, "destroy solver")
        end
    end
    return nothing
end

"""
    HypreDriveSession(; comm=:self, options=nothing, input_args=nothing)

Create a reusable HYPREDRV driver session. Matrix/RHS data are still copied into
HYPREDRV on `set_matrix_csr!` and `set_rhs!`, but options and the driver handle
are reused across solves.
"""
function HypreDriveSession(; comm::Symbol=:self, options=nothing,
                           input_args::Union{Nothing,Vector{String}}=nothing)
    yaml = options === nothing ? hypredrive_options() : hypredrive_options(options)
    drv_ref = _create_driver(comm)
    ok = false
    try
        _parse_input_args!(drv_ref[], yaml, input_args)
        session = HypreDriveSession(drv_ref, comm, 0, -1, 0, -1, false, false,
                                    false, false, SolveInfo(0, 0.0, 0.0, 0.0, false, 0.0))
        ok = true
        return finalizer(Base.close, session)
    finally
        if !ok && drv_ref[] != C_NULL
            code = ccall((:HYPREDRV_JuliaDestroy, _lib()), UInt32,
                         (Ref{Ptr{Cvoid}},), drv_ref)
            _warn_shutdown_failure(code, "destroy driver")
        end
    end
end

function Base.isopen(session::HypreDriveSession)
    return !session.closed && session.drv_ref[] != C_NULL
end

function Base.close(session::HypreDriveSession)
    if session.drv_ref[] != C_NULL
        if _initialized[]
            _destroy_solver!(session, warn_only=true)
            code = ccall((:HYPREDRV_JuliaDestroy, _lib()), UInt32,
                         (Ref{Ptr{Cvoid}},), session.drv_ref)
            _warn_shutdown_failure(code, "destroy driver")
        end
        session.drv_ref[] = C_NULL
    end
    session.closed = true
    return nothing
end

function _require_open(session::HypreDriveSession)
    isopen(session) || throw(ArgumentError("HypreDriveSession is closed"))
    return session.drv_ref[]::Ptr{Cvoid}
end

function _set_matrix_from_csr!(drv::Ptr{Cvoid}, row_start::Integer, row_end::Integer,
                               indptr::Vector{T}, cols::Vector{T},
                               data::Vector{Float64}) where {T<:Union{Int32,Int64}}
    GC.@preserve indptr cols data begin
        _check(ccall((:HYPREDRV_JuliaSetMatrixFromCSR, _lib()), UInt32,
                     (Ptr{Cvoid}, Int64, Int64, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}),
                     drv, Int64(row_start), Int64(row_end),
                     Base.unsafe_convert(Ptr{Cvoid}, pointer(indptr)),
                     Base.unsafe_convert(Ptr{Cvoid}, pointer(cols)),
                     pointer(data)), "set matrix")
    end
end

function set_matrix_csr!(session::HypreDriveSession, indptr, col_indices, data,
                         row_start::Integer; dofmap=nothing)
    drv = _require_open(session)
    _destroy_solver!(session)
    T = _index_type()
    indptr_t = _as_vector(T, indptr)
    cols_t = _as_vector(T, col_indices)
    data64 = _as_vector(Float64, data)
    length(indptr_t) >= 1 || throw(ArgumentError("indptr must not be empty"))
    length(cols_t) == length(data64) || throw(ArgumentError("column and data lengths differ"))
    indptr_t[1] == zero(T) || throw(ArgumentError("indptr must start at zero"))
    indptr_t[end] == T(length(data64)) || throw(ArgumentError("indptr[end] must equal nnz"))
    local_n = length(indptr_t) - 1
    row_start >= 0 || throw(ArgumentError("row_start must be nonnegative"))
    row_end = Int(row_start) + local_n - 1
    if local_n == 0
        throw(ArgumentError("empty row ranges are not supported by HypreDriveSession"))
    end
    dofmap_cint = dofmap === nothing ? nothing : _checked_cint_vector(dofmap, "dofmap")
    if dofmap_cint !== nothing
        length(dofmap_cint) == local_n || throw(ArgumentError("dofmap length does not match row range"))
    end
    _set_matrix_from_csr!(drv, row_start, row_end, indptr_t, cols_t, data64)
    if dofmap_cint !== nothing
        GC.@preserve dofmap_cint begin
            _check(ccall((:HYPREDRV_JuliaSetDofmap, _lib()), UInt32,
                         (Ptr{Cvoid}, Cint, Ptr{Cint}), drv,
                         Cint(length(dofmap_cint)), pointer(dofmap_cint)),
                   "set dofmap")
        end
    end
    session.row_start = Int(row_start)
    session.row_end = row_end
    session.matrix_revision += 1
    session.matrix_set = true
    session.rhs_set = false
    return session
end

function update_matrix_csr!(session::HypreDriveSession, indptr, col_indices, data,
                            row_start::Integer; dofmap=nothing)
    return set_matrix_csr!(session, indptr, col_indices, data, row_start; dofmap=dofmap)
end

function set_rhs!(session::HypreDriveSession, rhs, row_start::Integer=session.row_start)
    drv = _require_open(session)
    session.matrix_set || throw(ArgumentError("set_matrix_csr! must be called before set_rhs!"))
    rhs64 = _as_vector(Float64, rhs)
    length(rhs64) == session.row_end - session.row_start + 1 ||
        throw(ArgumentError("rhs length does not match matrix row range"))
    Int(row_start) == session.row_start ||
        throw(ArgumentError("rhs row_start does not match matrix row_start"))
    GC.@preserve rhs64 begin
        _check(ccall((:HYPREDRV_JuliaSetRHSFromArray, _lib()), UInt32,
                     (Ptr{Cvoid}, Int64, Int64, Ptr{Float64}), drv,
                     Int64(session.row_start), Int64(session.row_end),
                     pointer(rhs64)), "set rhs")
    end
    _check(ccall((:HYPREDRV_JuliaSetInitialGuessZero, _lib()), UInt32,
                 (Ptr{Cvoid},), drv), "set initial guess")
    session.rhs_set = true
    return session
end

function setup!(session::HypreDriveSession)
    drv = _require_open(session)
    session.matrix_set || throw(ArgumentError("matrix has not been set"))
    session.rhs_set || throw(ArgumentError("rhs has not been set"))
    if !session.solver_created
        _check(ccall((:HYPREDRV_JuliaLinearSolverCreate, _lib()), UInt32,
                     (Ptr{Cvoid},), drv), "create solver")
        session.solver_created = true
    end
    if session.setup_revision != session.matrix_revision
        _check(ccall((:HYPREDRV_JuliaLinearSolverSetup, _lib()), UInt32,
                     (Ptr{Cvoid},), drv), "setup solver")
        session.setup_revision = session.matrix_revision
    end
    return session
end

function _read_solution!(x::AbstractVector{Float64}, session::HypreDriveSession)
    drv = _require_open(session)
    local_n = session.row_end - session.row_start + 1
    length(x) == local_n || throw(ArgumentError("solution output length does not match row range"))
    src = Ref{Ptr{Cvoid}}(C_NULL)
    len = Ref{Int64}(0)
    _check(ccall((:HYPREDRV_JuliaGetSolutionValues, _lib()), UInt32,
                 (Ptr{Cvoid}, Ref{Ptr{Cvoid}}, Ref{Int64}), drv, src, len),
           "get solution")
    len[] == local_n || error("solution length $(len[]) does not match local row count $local_n")
    GC.@preserve x begin
        unsafe_copyto!(pointer(x), Ptr{Float64}(src[]), local_n)
    end
    return x
end

function _get_info(session::HypreDriveSession)
    drv = _require_open(session)
    iterations = Ref{Cint}(0)
    setup_time = Ref{Cdouble}(0)
    solve_time = Ref{Cdouble}(0)
    solution_norm = Ref{Cdouble}(0)
    converged = Ref{Cint}(0)
    final_res_norm = Ref{Cdouble}(0)
    _check(ccall((:HYPREDRV_JuliaLinearSolverGetNumIter, _lib()), UInt32,
                 (Ptr{Cvoid}, Ref{Cint}), drv, iterations), "get iterations")
    _check(ccall((:HYPREDRV_JuliaLinearSolverGetConverged, _lib()), UInt32,
                 (Ptr{Cvoid}, Ref{Cint}), drv, converged), "get convergence flag")
    _check(ccall((:HYPREDRV_JuliaLinearSolverGetFinalRelativeResidualNorm, _lib()),
                 UInt32, (Ptr{Cvoid}, Ref{Cdouble}), drv, final_res_norm),
           "get final relative residual norm")
    _check(ccall((:HYPREDRV_JuliaLinearSolverGetSetupTime, _lib()), UInt32,
                 (Ptr{Cvoid}, Ref{Cdouble}), drv, setup_time), "get setup time")
    _check(ccall((:HYPREDRV_JuliaLinearSolverGetSolveTime, _lib()), UInt32,
                 (Ptr{Cvoid}, Ref{Cdouble}), drv, solve_time), "get solve time")
    _check(ccall((:HYPREDRV_JuliaGetSolutionNorm, _lib()), UInt32,
                 (Ptr{Cvoid}, Cstring, Ref{Cdouble}), drv, "L2", solution_norm),
           "get solution norm")
    return SolveInfo(Int(iterations[]), setup_time[], solve_time[], solution_norm[],
                     converged[] != 0, final_res_norm[])
end

function solve!(x::AbstractVector{Float64}, session::HypreDriveSession)
    drv = _require_open(session)
    setup!(session)
    _check(ccall((:HYPREDRV_JuliaResetInitialGuess, _lib()), UInt32,
                 (Ptr{Cvoid},), drv), "reset initial guess")
    _check(ccall((:HYPREDRV_JuliaLinearSolverApply, _lib()), UInt32,
                 (Ptr{Cvoid},), drv), "apply solver")
    _read_solution!(x, session)
    session.last_info = _get_info(session)
    return x, session.last_info
end

info(session::HypreDriveSession) = session.last_info

function _solve_csr_impl(indptr::Vector{T}, col_indices::Vector{T}, data::Vector{Float64}, rhs::Vector{Float64},
                         row_start::Integer, row_end::Integer, yaml::AbstractString;
                         comm::Symbol=:self, nsolve::Integer=1,
                         dofmap::Union{Nothing,Vector{Cint}}=nothing,
                         input_args::Union{Nothing,Vector{String}}=nothing) where {T<:Union{Int32,Int64}}
    hypredrive_initialize()
    nsolve >= 1 || throw(ArgumentError("nsolve must be positive"))
    row_start >= 0 || throw(ArgumentError("row_start must be nonnegative"))
    if row_start > row_end
        isempty(rhs) || throw(ArgumentError("empty row range requires empty rhs"))
        length(indptr) == 1 || throw(ArgumentError("empty row range requires a one-entry indptr"))
        isempty(col_indices) || throw(ArgumentError("empty row range requires empty column indices"))
        isempty(data) || throw(ArgumentError("empty row range requires empty data"))
        indptr[1] == zero(T) || throw(ArgumentError("indptr must start at zero"))
        if comm === :self
            return Float64[], SolveInfo(0, 0.0, 0.0, 0.0, false, 0.0)
        end
        throw(ArgumentError("empty MPI_COMM_WORLD row ranges are not supported by direct CSR solves; avoid launching more ranks than nonempty local CSR slabs"))
    end
    length(rhs) == row_end - row_start + 1 || throw(ArgumentError("rhs length does not match row range"))
    length(indptr) == length(rhs) + 1 || throw(ArgumentError("indptr length does not match row range"))
    length(col_indices) == length(data) || throw(ArgumentError("column and data lengths differ"))
    length(data) <= typemax(T) || throw(ArgumentError("nnz exceeds HYPRE_BigInt range"))
    indptr[1] == zero(T) || throw(ArgumentError("indptr must start at zero"))
    indptr[end] == T(length(data)) || throw(ArgumentError("indptr[end] must equal nnz"))
    if dofmap !== nothing
        length(dofmap) == length(rhs) || throw(ArgumentError("dofmap length does not match row range"))
        length(dofmap) <= typemax(Cint) || throw(ArgumentError("dofmap length exceeds Cint range"))
    end

    # Driver handles are intentionally scoped to this function and destroyed in
    # finally; no long-lived Julia wrapper is exposed that would need a finalizer.
    drv_ref = Ref{Ptr{Cvoid}}(C_NULL)
    solver_created = false
    try
        if comm === :world
            _check(ccall((:HYPREDRV_JuliaCreateWithWorld, _lib()), UInt32,
                         (Ref{Ptr{Cvoid}},), drv_ref), "create driver")
        else
            _check(ccall((:HYPREDRV_JuliaCreateWithSelf, _lib()), UInt32,
                         (Ref{Ptr{Cvoid}},), drv_ref), "create driver")
        end
        drv = drv_ref[]
        _check(ccall((:HYPREDRV_JuliaSetLibraryMode, _lib()), UInt32, (Ptr{Cvoid},), drv), "set library mode")
        _parse_input_args!(drv, yaml, input_args)
        GC.@preserve indptr col_indices data rhs begin
            _check(ccall((:HYPREDRV_JuliaSetMatrixFromCSR, _lib()), UInt32,
                         (Ptr{Cvoid}, Int64, Int64, Ptr{T}, Ptr{T}, Ptr{Float64}),
                         drv, Int64(row_start), Int64(row_end), pointer(indptr),
                         pointer(col_indices), pointer(data)), "set matrix")
            _check(ccall((:HYPREDRV_JuliaSetRHSFromArray, _lib()), UInt32,
                         (Ptr{Cvoid}, Int64, Int64, Ptr{Float64}),
                         drv, Int64(row_start), Int64(row_end), pointer(rhs)), "set rhs")
        end
        if dofmap !== nothing
            GC.@preserve dofmap begin
                _check(ccall((:HYPREDRV_JuliaSetDofmap, _lib()), UInt32,
                             (Ptr{Cvoid}, Cint, Ptr{Cint}), drv, Cint(length(dofmap)),
                             pointer(dofmap)), "set dofmap")
            end
        end

        local_n = length(rhs)
        x = Vector{Float64}(undef, local_n)
        info = SolveInfo(0, 0.0, 0.0, 0.0, false, 0.0)
        _check(ccall((:HYPREDRV_JuliaSetInitialGuessZero, _lib()), UInt32, (Ptr{Cvoid},), drv), "set initial guess")
        _check(ccall((:HYPREDRV_JuliaLinearSolverCreate, _lib()), UInt32, (Ptr{Cvoid},), drv), "create solver")
        solver_created = true
        _check(ccall((:HYPREDRV_JuliaLinearSolverSetup, _lib()), UInt32, (Ptr{Cvoid},), drv), "setup solver")
        for _ in 1:nsolve
            _check(ccall((:HYPREDRV_JuliaResetInitialGuess, _lib()), UInt32, (Ptr{Cvoid},), drv), "reset initial guess")
            _check(ccall((:HYPREDRV_JuliaLinearSolverApply, _lib()), UInt32, (Ptr{Cvoid},), drv), "apply solver")

            src = Ref{Ptr{Cvoid}}(C_NULL)
            len = Ref{Int64}(0)
            _check(ccall((:HYPREDRV_JuliaGetSolutionValues, _lib()), UInt32,
                         (Ptr{Cvoid}, Ref{Ptr{Cvoid}}, Ref{Int64}), drv, src, len), "get solution")
            len[] == local_n || error("solution length $(len[]) does not match local row count $local_n")
            # src is a borrowed pointer owned by HYPREDRV. Copy it before the
            # next solver call, reset, or driver destroy can invalidate it.
            GC.@preserve x begin
                unsafe_copyto!(pointer(x), Ptr{Float64}(src[]), local_n)
            end

            iterations = Ref{Cint}(0)
            setup_time = Ref{Cdouble}(0)
            solve_time = Ref{Cdouble}(0)
            solution_norm = Ref{Cdouble}(0)
            converged = Ref{Cint}(0)
            final_res_norm = Ref{Cdouble}(0)
            _check(ccall((:HYPREDRV_JuliaLinearSolverGetNumIter, _lib()), UInt32, (Ptr{Cvoid}, Ref{Cint}), drv, iterations), "get iterations")
            _check(ccall((:HYPREDRV_JuliaLinearSolverGetConverged, _lib()), UInt32, (Ptr{Cvoid}, Ref{Cint}), drv, converged), "get convergence flag")
            _check(ccall((:HYPREDRV_JuliaLinearSolverGetFinalRelativeResidualNorm, _lib()), UInt32, (Ptr{Cvoid}, Ref{Cdouble}), drv, final_res_norm), "get final relative residual norm")
            _check(ccall((:HYPREDRV_JuliaLinearSolverGetSetupTime, _lib()), UInt32, (Ptr{Cvoid}, Ref{Cdouble}), drv, setup_time), "get setup time")
            _check(ccall((:HYPREDRV_JuliaLinearSolverGetSolveTime, _lib()), UInt32, (Ptr{Cvoid}, Ref{Cdouble}), drv, solve_time), "get solve time")
            _check(ccall((:HYPREDRV_JuliaGetSolutionNorm, _lib()), UInt32, (Ptr{Cvoid}, Cstring, Ref{Cdouble}), drv, "L2", solution_norm), "get solution norm")
            info = SolveInfo(Int(iterations[]), setup_time[], solve_time[], solution_norm[], converged[] != 0, final_res_norm[])

        end
        _check(ccall((:HYPREDRV_JuliaLinearSolverDestroy, _lib()), UInt32, (Ptr{Cvoid},), drv), "destroy solver")
        solver_created = false
        return x, info
    finally
        if drv_ref[] != C_NULL
            if solver_created
                code = ccall((:HYPREDRV_JuliaLinearSolverDestroy, _lib()), UInt32, (Ptr{Cvoid},), drv_ref[])
                _warn_shutdown_failure(code, "destroy solver")
                if code != 0x00000000
                    @warn "Skipping driver destroy because solver destroy failed; driver handle intentionally leaked to avoid cascading cleanup through inconsistent HYPRE state"
                    drv_ref[] = C_NULL
                end
            end
            if drv_ref[] != C_NULL
                code = ccall((:HYPREDRV_JuliaDestroy, _lib()), UInt32, (Ref{Ptr{Cvoid}},), drv_ref)
                _warn_shutdown_failure(code, "destroy driver")
                drv_ref[] = C_NULL
            end
        end
    end
end

function _solve_impl(A::SparseMatrixCSC, b::AbstractVector, yaml::AbstractString;
                     comm::Symbol=:self, row_start::Union{Nothing,Int}=nothing,
                     row_end::Union{Nothing,Int}=nothing, nsolve::Integer=1,
                     dofmap=nothing,
                     input_args::Union{Nothing,Vector{String}}=nothing)
    n = size(A, 1)
    length(b) == n || throw(ArgumentError("rhs length $(length(b)) does not match matrix size $n"))
    T = _index_type()
    if row_start === nothing || row_end === nothing
        row_start = 0
        row_end = n - 1
    end
    indptr, col_indices, data = _csr_rows_from_sparse(A, T, row_start + 1, row_end + 1)
    rhs = Vector{Float64}(@view b[row_start + 1:row_end + 1])
    dofmap_cint = nothing
    if dofmap !== nothing
        length(dofmap) == n || throw(ArgumentError("dofmap length does not match matrix size"))
        dofmap_cint = _checked_cint_vector(@view(dofmap[row_start + 1:row_end + 1]), "dofmap")
    end
    return _solve_csr_impl(indptr, col_indices, data, rhs, row_start, row_end, yaml;
                           comm=comm, nsolve=nsolve, dofmap=dofmap_cint,
                           input_args=input_args)
end

function hypredrive_solve(A::AbstractMatrix, b::AbstractVector; options=nothing,
                          nsolve::Integer=1, dofmap=nothing,
                          input_args::Union{Nothing,Vector{String}}=nothing)
    sparse_A = _as_sparse_float64(A)
    yaml = options === nothing ? hypredrive_options() : hypredrive_options(options)
    return _solve_impl(sparse_A, b, yaml; nsolve=nsolve, dofmap=dofmap,
                       input_args=input_args)
end

function hypredrive_solve_mpi(A::AbstractMatrix, b::AbstractVector; options=nothing,
                              nsolve::Integer=1, dofmap=nothing,
                              input_args::Union{Nothing,Vector{String}}=nothing)
    sparse_A = _as_sparse_float64(A)
    yaml = options === nothing ? hypredrive_options() : hypredrive_options(options)
    rank = hypredrive_mpi_world_rank()
    nprocs = hypredrive_mpi_world_size()
    nprocs <= size(sparse_A, 1) || throw(ArgumentError("hypredrive_solve_mpi requires at least one row per MPI rank"))
    row_start, row_end = _partition_rows(size(sparse_A, 1), rank, nprocs)
    return _solve_impl(sparse_A, b, yaml; comm=:world, row_start=row_start,
                       row_end=row_end, nsolve=nsolve, dofmap=dofmap,
                       input_args=input_args)
end

function hypredrive_solve_mpi_csr(indptr, col_indices, data, rhs, row_start::Integer;
                                  options=nothing, nsolve::Integer=1, comm::Symbol=:self,
                                  dofmap=nothing,
                                  input_args::Union{Nothing,Vector{String}}=nothing)
    T = _index_type()
    comm in (:world, :self) || throw(ArgumentError("comm must be :world or :self"))
    row_start >= 0 || throw(ArgumentError("row_start must be nonnegative"))
    indptr_t = _as_vector(T, indptr)
    cols_t = _as_vector(T, col_indices)
    data64 = _as_vector(Float64, data)
    rhs64 = _as_vector(Float64, rhs)
    dofmap_cint = dofmap === nothing ? nothing : _checked_cint_vector(dofmap, "dofmap")
    row_end = Int(row_start) + length(rhs64) - 1
    yaml = options === nothing ? hypredrive_options() : hypredrive_options(options)
    return _solve_csr_impl(indptr_t, cols_t, data64, rhs64, row_start, row_end, yaml;
                           comm=comm, nsolve=nsolve, dofmap=dofmap_cint,
                           input_args=input_args)
end

# Qualified convenience aliases. They are intentionally not exported.
initialize() = hypredrive_initialize()
shutdown() = hypredrive_shutdown()
library_path() = hypredrive_library_path()
mpi_rank() = hypredrive_mpi_world_rank()
mpi_size() = hypredrive_mpi_world_size()
comm_rank(comm::Symbol=:world) = hypredrive_comm_rank(comm)
comm_size(comm::Symbol=:world) = hypredrive_comm_size(comm)
solve(A::AbstractMatrix, b::AbstractVector; kwargs...) = hypredrive_solve(A, b; kwargs...)
solve_mpi(A::AbstractMatrix, b::AbstractVector; kwargs...) = hypredrive_solve_mpi(A, b; kwargs...)

# Qualified convenience aliases for the MPI_COMM_WORLD helpers.
hypredrive_mpi_rank() = hypredrive_mpi_world_rank()
hypredrive_mpi_size() = hypredrive_mpi_world_size()
hypredrive_mpi_sum(value::Real) = hypredrive_mpi_world_sum(value)

end # module HypreDrive
