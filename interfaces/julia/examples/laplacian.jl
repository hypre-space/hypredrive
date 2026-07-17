#!/usr/bin/env julia
# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using HypreDrive
using LinearAlgebra

mutable struct Params
    input::Union{Nothing,String}
    n::NTuple{3,Int}
    c::NTuple{3,Float64}
    p::NTuple{3,Int}
    stencil::Int
    nsolve::Int
    verbose::Int
    hypredrv_args::Vector{String}
end

function usage()
    println("""
Usage: mpiexec -n <np> julia --project=interfaces/julia interfaces/julia/examples/laplacian.jl [options]

Options:
  -i, --input <file>       YAML configuration file for solver settings
  -a, --args <key value>...  Hypredrive YAML overrides, e.g.
                             -a --solver:pcg:max_iter 100 (must come last)
  -n <nx> <ny> <nz>        Global grid dimensions (default: 10 10 10)
  -c <cx> <cy> <cz>        Diffusion coefficients (default: 1.0 1.0 1.0)
  -P <px> <py> <pz>        Processor grid dimensions (default: 1 1 1)
  -s <val>                 Stencil type: 7, 19, or 27 (default: 7)
  -ns, --nsolve <n>        Number of solves using one assembled matrix (default: 5)
  -v, --verbose <n>        Verbosity bitset (default: 1)
                             0x1: print setup/solve summary
  -h, --help               Print this message
""")
end

function parse_args(args)
    params = Params(nothing, (10, 10, 10), (1.0, 1.0, 1.0), (1, 1, 1), 7, 5, 1, String[])
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif arg in ("-i", "--input")
            i += 1
            i <= length(args) || error("$arg requires a file")
            params.input = args[i]
        elseif arg in ("-a", "--args")
            params.hypredrv_args = args[i:end]
            break
        elseif arg == "-n"
            i + 3 <= length(args) || error("-n requires three values")
            params.n = (parse(Int, args[i + 1]), parse(Int, args[i + 2]), parse(Int, args[i + 3]))
            i += 3
        elseif arg == "-c"
            i + 3 <= length(args) || error("-c requires three values")
            params.c = (parse(Float64, args[i + 1]), parse(Float64, args[i + 2]), parse(Float64, args[i + 3]))
            i += 3
        elseif arg == "-P"
            i + 3 <= length(args) || error("-P requires three values")
            params.p = (parse(Int, args[i + 1]), parse(Int, args[i + 2]), parse(Int, args[i + 3]))
            i += 3
        elseif arg == "-s"
            i += 1
            i <= length(args) || error("-s requires a stencil value")
            params.stencil = parse(Int, args[i])
        elseif arg in ("-ns", "--nsolve")
            i += 1
            i <= length(args) || error("$arg requires a count")
            params.nsolve = parse(Int, args[i])
        elseif arg in ("-v", "--verbose")
            i += 1
            i <= length(args) || error("$arg requires a value")
            params.verbose = parse(Int, args[i])
        else
            error("unknown argument: $arg")
        end
        i += 1
    end
    return params
end

function starts_for(n::Int, p::Int)
    base = div(n, p)
    rest = n - base * p
    return [base * j + min(j, rest) for j in 0:p]
end

function rank_to_coords(rank::Int, p::NTuple{3,Int})
    px, py, pz = p
    x = div(rank, py * pz)
    rem1 = rank - x * py * pz
    y = div(rem1, pz)
    z = rem1 - y * pz
    return (x, y, z)
end

function owner_coord(coord0::Int, starts::Vector{Int})
    return searchsortedlast(starts, coord0) - 1
end

function block_index(g::NTuple{3,Int}, starts)
    return (owner_coord(g[1], starts[1]), owner_coord(g[2], starts[2]), owner_coord(g[3], starts[3]))
end

function block_size(b::NTuple{3,Int}, starts)
    bx, by, bz = b
    sx, sy, sz = starts
    lx = sx[bx + 2] - sx[bx + 1]
    ly = sy[by + 2] - sy[by + 1]
    lz = sz[bz + 2] - sz[bz + 1]
    return lx * ly * lz
end

function block_bases(p::NTuple{3,Int}, starts)
    bases = Array{Int,3}(undef, p...)
    next_base = 0
    for bx in 0:p[1]-1, by in 0:p[2]-1, bz in 0:p[3]-1
        bases[bx + 1, by + 1, bz + 1] = next_base
        next_base += block_size((bx, by, bz), starts)
    end
    return bases
end

function grid_to_index(g::NTuple{3,Int}, b::NTuple{3,Int}, starts, bases)
    bx, by, bz = b
    sx, sy, sz = starts
    lx = sx[bx + 2] - sx[bx + 1]
    ly = sy[by + 2] - sy[by + 1]
    return bases[bx + 1, by + 1, bz + 1] +
           ((g[3] - sz[bz + 1]) * ly + (g[2] - sy[by + 1])) * lx +
           (g[1] - sx[bx + 1])
end

function check_global_indexing(params::Params)
    starts = (starts_for(params.n[1], params.p[1]),
              starts_for(params.n[2], params.p[2]),
              starts_for(params.n[3], params.p[3]))
    bases = block_bases(params.p, starts)
    total = prod(params.n)
    seen = falses(total)
    for bx in 0:params.p[1]-1, by in 0:params.p[2]-1, bz in 0:params.p[3]-1
        block = (bx, by, bz)
        x0, x1 = starts[1][bx + 1], starts[1][bx + 2] - 1
        y0, y1 = starts[2][by + 1], starts[2][by + 2] - 1
        z0, z1 = starts[3][bz + 1], starts[3][bz + 2] - 1
        for z in z0:z1, y in y0:y1, x in x0:x1
            index = grid_to_index((x, y, z), block, starts, bases)
            0 <= index < total || error("global row index $index is out of range")
            !seen[index + 1] || error("duplicate global row index $index")
            seen[index + 1] = true
        end
    end
    all(seen) || error("global row indexing has gaps")
    return nothing
end

struct StencilOffset
    di::Int
    dj::Int
    dk::Int
    weight::Float64
end

function stencil_offsets(stencil::Int)
    offsets = StencilOffset[]
    append!(offsets, [StencilOffset(-1, 0, 0, 1.0), StencilOffset(1, 0, 0, 1.0),
                      StencilOffset(0, -1, 0, 1.0), StencilOffset(0, 1, 0, 1.0),
                      StencilOffset(0, 0, -1, 1.0), StencilOffset(0, 0, 1, 1.0)])
    if stencil >= 19
        for di in -1:1, dj in -1:1, dk in -1:1
            distance = abs(di) + abs(dj) + abs(dk)
            if distance == 2 || (stencil == 27 && distance == 3)
                push!(offsets, StencilOffset(di, dj, dk, 0.25 / distance))
            end
        end
    end
    return offsets
end

function build_local_laplacian(params::Params, rank::Int)
    params.stencil in (7, 19, 27) || error("supported Julia stencils are 7, 19, and 27")
    all(params.n .>= 2) || error("all grid dimensions must be at least 2")
    starts = (starts_for(params.n[1], params.p[1]),
              starts_for(params.n[2], params.p[2]),
              starts_for(params.n[3], params.p[3]))
    bases = block_bases(params.p, starts)
    coords = rank_to_coords(rank, params.p)
    x0, x1 = starts[1][coords[1] + 1], starts[1][coords[1] + 2] - 1
    y0, y1 = starts[2][coords[2] + 1], starts[2][coords[2] + 2] - 1
    z0, z1 = starts[3][coords[3] + 1], starts[3][coords[3] + 2] - 1

    row_start = grid_to_index((x0, y0, z0), coords, starts, bases)
    local_n = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
    indptr = Int64[0]
    cols = Int64[]
    vals = Float64[]
    rhs = Float64[]
    offsets = stencil_offsets(params.stencil)
    cx, cy, cz = params.c
    coeff_scale = (cx, cy, cz)

    for z in z0:z1, y in y0:y1, x in x0:x1
        boundary = x == 0 || x == params.n[1] - 1 || y == 0 || y == params.n[2] - 1 || z == 0 || z == params.n[3] - 1
        if boundary
            push!(cols, grid_to_index((x, y, z), coords, starts, bases))
            push!(vals, 1.0)
            push!(rhs, y == 0 ? 1.0 : 0.0)
            push!(indptr, length(cols))
            continue
        end

        diag = 0.0
        for offset in offsets
            ng = (x + offset.di, y + offset.dj, z + offset.dk)
            nb = block_index(ng, starts)
            direction_coeff = abs(offset.di) * coeff_scale[1] +
                              abs(offset.dj) * coeff_scale[2] +
                              abs(offset.dk) * coeff_scale[3]
            coeff = offset.weight * direction_coeff
            push!(cols, grid_to_index(ng, nb, starts, bases))
            push!(vals, -coeff)
            diag += coeff
        end
        push!(cols, grid_to_index((x, y, z), coords, starts, bases))
        push!(vals, diag)
        push!(rhs, 0.0)
        push!(indptr, length(cols))
    end

    return row_start, indptr, cols, vals, rhs
end

function options_from(params::Params)
    params.input !== nothing && return read(params.input, String)
    return hypredrive_options(solver=:pcg,
                              preconditioner=:amg,
                              pcg=(max_iter=200, relative_tol=1.0e-8),
                              amg=(print_level=0,))
end

function main()
    params = parse_args(ARGS)
    rank = hypredrive_mpi_world_rank()
    nprocs = hypredrive_mpi_world_size()

    prod(params.p) == nprocs || error("number of MPI ranks ($nprocs) must equal P product $(prod(params.p))")
    all(params.p .<= params.n) || error("processor grid dimensions must not exceed grid dimensions")
    if rank == 0 && get(ENV, "HYPREDRV_JULIA_CHECK_INDEXING", "0") == "1"
        check_global_indexing(params)
    end

    if rank == 0 && (params.verbose & 0x1) != 0
        println()
        println("=====================================================")
        println("              Julia Laplacian Problem Setup")
        println("=====================================================")
        println("Grid dimensions:      $(params.n[1]) x $(params.n[2]) x $(params.n[3])")
        println("Processor topology:   $(params.p[1]) x $(params.p[2]) x $(params.p[3])")
        println("Diffusion coeffs:     $(params.c)")
        println("Discretization:       $(params.stencil)-point stencil")
        println("Number of solves:     $(params.nsolve)")
        println("=====================================================")
        println()
    end

    row_start, indptr, cols, vals, rhs = build_local_laplacian(params, rank)
    if isempty(params.hypredrv_args)
        local_x, info = hypredrive_solve_mpi_csr(indptr, cols, vals, rhs, row_start;
                                                 options=options_from(params),
                                                 nsolve=params.nsolve, comm=:world)
    else
        # input_args[1] carries the configuration; normalize it the same way the
        # options= path does (e.g. inject a quiet general.statistics default).
        input_args = String[hypredrive_options(options_from(params))]
        append!(input_args, params.hypredrv_args)
        local_x, info = hypredrive_solve_mpi_csr(indptr, cols, vals, rhs, row_start;
                                                 options="", input_args=input_args,
                                                 nsolve=params.nsolve, comm=:world)
    end
    local_norm2 = dot(local_x, local_x)
    global_norm = sqrt(hypredrive_mpi_world_sum(local_norm2))

    if rank == 0 && (params.verbose & 0x1) != 0
        println("Iterations:           ", info.iterations)
        println("Setup time:           ", info.setup_time)
        println("Solve time:           ", info.solve_time)
        println("Global solution norm: ", global_norm)
    end
end

main()
