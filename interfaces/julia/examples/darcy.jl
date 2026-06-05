#!/usr/bin/env julia
# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using HypreDrive
using LinearAlgebra
using Printf

mutable struct DarcyParams
    input::Union{Nothing,String}
    n::NTuple{3,Int}
    p::NTuple{3,Int}
    p_set::Bool
    L::NTuple{3,Float64}
    K::NTuple{3,Float64}
    constant_K_set::Bool
    K_file::Union{Nothing,String}
    K_file_grid::NTuple{3,Int}
    K_file_top_down::Bool
    drive_axis::Int
    output_file::Union{Nothing,String}
    verbose::Int
    hypredrv_args::Vector{String}
end

struct DarcyMesh
    n::NTuple{3,Int}
    L::NTuple{3,Float64}
    h::NTuple{3,Float64}
    K::NTuple{3,Float64}
    Kinv::NTuple{3,Float64}
    dim::Int
    n_cells::Int64
    n_x_faces::Int64
    n_y_faces::Int64
    n_z_faces::Int64
    n_faces::Int64
    n_total::Int64
    Kinv_cells::Union{Nothing,Vector{Float64}}
end

struct DarcyLayout
    P::NTuple{3,Int}
    x0::Vector{Int}
    x1::Vector{Int}
    y0::Vector{Int}
    y1::Vector{Int}
    z0::Vector{Int}
    z1::Vector{Int}
    xpart::Vector{Int}
    ypart::Vector{Int}
    zpart::Vector{Int}
    offset::Vector{Int64}
    count_xf::Vector{Int64}
    count_yf::Vector{Int64}
    count_zf::Vector{Int64}
    count_cells::Vector{Int64}
    total::Vector{Int64}
    N::Int64
    nprocs::Int
end

function usage()
    println("""
Usage:
  mpirun -np <ranks> julia --project=interfaces/julia interfaces/julia/examples/darcy.jl [options]

Solves an RT0/P0 mixed Darcy problem on a prefix-active Cartesian mesh.
Active dimensions must be x, x-y, or x-y-z.

Options:
  -i, --input <file>               YAML solver/preconditioner file
  -a, --args <key value>...        Override hypredrive YAML options; must be
                                   the last Darcy option
  -n <nx> <ny> <nz>                Cell counts; default: 8 8 1
  -P, --procs <px> <py> <pz>       MPI rank grid; product must equal -np;
                                   inactive dimensions must be 1
  -L <Lx> <Ly> <Lz>                Domain lengths; default: 1 1 1
  -K <Kx> <Ky> <Kz>                Constant diagonal permeability; default: 1 1 1
  --K-file <path>                  Permeability text file; one value per source
                                   cell, or three blocks Kx Ky Kz
  --K-file-grid <nx> <ny> <nz>     Source grid for --K-file; omit when it
                                   matches -n
  --K-file-k-order <order>         Layer order: bottom-up or top-down;
                                   default: bottom-up
  -g, --gradient-direction <axis>  Pressure-drop direction: x, y, or z;
                                   default: x
  -o, --output <file>              Write VTK results. On 1 rank a .vti is
                                   written; on >1 rank a .pvti master plus
                                   one .vti piece per rank is written
  -v, --verbose <bits>             Verbosity bitset; default: 1
                                     0x1  setup and solve summary
                                     0x4  reserved for C parity
  -h, --help                       Print this help and exit
""")
end

default_params() = DarcyParams(nothing, (8, 8, 1), (0, 0, 0), false,
                               (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), false,
                               nothing, (0, 0, 0), false, 1, nothing, 1, String[])

function parse_axis(text::AbstractString)
    text == "x" && return 1
    text == "y" && return 2
    text == "z" && return 3
    error("-g must be x, y, or z")
end

function parse_args(args)
    params = default_params()
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
        elseif arg in ("-P", "--procs")
            i + 3 <= length(args) || error("-P requires three values")
            params.p = (parse(Int, args[i + 1]), parse(Int, args[i + 2]), parse(Int, args[i + 3]))
            params.p_set = true
            i += 3
        elseif arg == "-L"
            i + 3 <= length(args) || error("-L requires three values")
            params.L = (parse(Float64, args[i + 1]), parse(Float64, args[i + 2]), parse(Float64, args[i + 3]))
            i += 3
        elseif arg == "-K"
            i + 3 <= length(args) || error("-K requires three values")
            params.K = (parse(Float64, args[i + 1]), parse(Float64, args[i + 2]), parse(Float64, args[i + 3]))
            params.constant_K_set = true
            i += 3
        elseif arg == "--K-file"
            i += 1
            i <= length(args) || error("--K-file requires a path")
            params.K_file = args[i]
        elseif arg == "--K-file-grid"
            i + 3 <= length(args) || error("--K-file-grid requires three values")
            params.K_file_grid = (parse(Int, args[i + 1]), parse(Int, args[i + 2]), parse(Int, args[i + 3]))
            i += 3
        elseif arg == "--K-file-k-order"
            i += 1
            i <= length(args) || error("--K-file-k-order requires bottom-up or top-down")
            if args[i] == "bottom-up"
                params.K_file_top_down = false
            elseif args[i] == "top-down"
                params.K_file_top_down = true
            else
                error("--K-file-k-order requires bottom-up or top-down")
            end
        elseif arg in ("-g", "--gradient-direction")
            i += 1
            i <= length(args) || error("-g requires x, y, or z")
            params.drive_axis = parse_axis(args[i])
        elseif arg in ("-o", "--output")
            i += 1
            i <= length(args) || error("--output requires a path")
            params.output_file = args[i]
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

function validate_params(params::DarcyParams, nprocs::Int)
    all(params.n .>= 1) || error("all cell counts must be positive")
    params.n[1] > 1 || (params.n[2] == 1 && params.n[3] == 1) ||
        error("active dimensions must be a prefix of x,y,z")
    params.n[2] > 1 || params.n[3] == 1 || error("active dimensions must be a prefix of x,y,z")
    any(params.n .> 1) || error("at least one cell count must exceed 1")
    dim = count(>(1), params.n)
    params.drive_axis <= dim || error("pressure-drop direction is not active for this mesh")
    all(params.L .> 0.0) || error("length entries must be positive")
    all(params.K .> 0.0) || error("permeability entries must be positive")
    if params.p_set
        all(params.p .> 0) || error("-P entries must be positive")
        prod(params.p) == nprocs || error("-P product must equal MPI ranks ($(prod(params.p)) != $nprocs)")
        for d in 1:3
            params.n[d] == 1 && params.p[d] != 1 &&
                error("cannot partition inactive dimension $("xyz"[d])")
        end
    end
    params.K_file !== nothing && params.constant_K_set && error("-K and --K-file are mutually exclusive")
    params.K_file === nothing && any(params.K_file_grid .!= 0) &&
        error("--K-file-grid requires --K-file")
    params.K_file === nothing && params.K_file_top_down &&
        error("--K-file-k-order requires --K-file")
    all(params.K_file_grid .>= 0) || error("--K-file-grid entries must be nonnegative")
    params.K_file !== nothing && !in(count(==(0), params.K_file_grid), (0, 3)) &&
        error("--K-file-grid must provide three positive values")
end

function init_mesh(params::DarcyParams)
    h = ntuple(d -> params.L[d] / params.n[d], 3)
    Kinv = ntuple(d -> 1.0 / params.K[d], 3)
    dim = count(>(1), params.n)
    n_cells = Int64(params.n[1]) * params.n[2] * params.n[3]
    n_x_faces = Int64(params.n[1] + 1) * params.n[2] * params.n[3]
    n_y_faces = dim >= 2 ? Int64(params.n[1]) * (params.n[2] + 1) * params.n[3] : Int64(0)
    n_z_faces = dim >= 3 ? Int64(params.n[1]) * params.n[2] * (params.n[3] + 1) : Int64(0)
    n_faces = n_x_faces + n_y_faces + n_z_faces
    return DarcyMesh(params.n, params.L, h, params.K, Kinv, dim, n_cells,
                     n_x_faces, n_y_faces, n_z_faces, n_faces, n_faces + n_cells,
                     nothing)
end

cell_index(mesh::DarcyMesh, i::Integer, j::Integer, k::Integer) =
    Int64(i) + Int64(mesh.n[1]) * (Int64(j) + Int64(mesh.n[2]) * Int64(k))

x_face_index(mesh::DarcyMesh, i::Integer, j::Integer, k::Integer) =
    Int64(i) + Int64(mesh.n[1] + 1) * (Int64(j) + Int64(mesh.n[2]) * Int64(k))

y_face_index(mesh::DarcyMesh, i::Integer, j::Integer, k::Integer) =
    mesh.n_x_faces + Int64(i) +
    Int64(mesh.n[1]) * (Int64(j) + Int64(mesh.n[2] + 1) * Int64(k))

z_face_index(mesh::DarcyMesh, i::Integer, j::Integer, k::Integer) =
    mesh.n_x_faces + mesh.n_y_faces + Int64(i) +
    Int64(mesh.n[1]) * (Int64(j) + Int64(mesh.n[2]) * Int64(k))

function face_area(mesh::DarcyMesh, dir::Int)
    dir == 1 && return (mesh.n[2] > 1 ? mesh.h[2] : 1.0) * (mesh.n[3] > 1 ? mesh.h[3] : 1.0)
    dir == 2 && return mesh.h[1] * (mesh.n[3] > 1 ? mesh.h[3] : 1.0)
    return mesh.h[1] * mesh.h[2]
end

cell_volume(mesh::DarcyMesh) = mesh.h[1] * (mesh.n[2] > 1 ? mesh.h[2] : 1.0) *
                               (mesh.n[3] > 1 ? mesh.h[3] : 1.0)

function cell_Kinv(mesh::DarcyMesh, cell::Integer, dir::Int)
    mesh.Kinv_cells === nothing && return mesh.Kinv[dir]
    return mesh.Kinv_cells[3 * Int(cell) + dir]
end

function rt0_cell_flux_dofs_layout(layout::DarcyLayout, mesh::DarcyMesh, i::Int, j::Int, k::Int)
    faces = Int64[layout_xface(layout, mesh, i, j, k), layout_xface(layout, mesh, i + 1, j, k)]
    dirs = Int[1, 1]
    is_low = Bool[true, false]
    signs = Int[-1, 1]
    if mesh.dim >= 2
        append!(faces, [layout_yface(layout, mesh, i, j, k), layout_yface(layout, mesh, i, j + 1, k)])
        append!(dirs, [2, 2])
        append!(is_low, [true, false])
        append!(signs, [-1, 1])
    end
    if mesh.dim >= 3
        append!(faces, [layout_zface(layout, mesh, i, j, k), layout_zface(layout, mesh, i, j, k + 1)])
        append!(dirs, [3, 3])
        append!(is_low, [true, false])
        append!(signs, [-1, 1])
    end
    return faces, dirs, is_low, signs
end

function rt0_mass_entry(mesh::DarcyMesh, cell::Int64, a::Int, b::Int,
                        dirs::Vector{Int}, is_low::Vector{Bool})
    da = dirs[a]
    db = dirs[b]
    da == db || return 0.0
    coef = is_low[a] == is_low[b] ? 1.0 / 3.0 : 1.0 / 6.0
    return cell_Kinv(mesh, cell, da) * cell_volume(mesh) * coef /
           (face_area(mesh, da) * face_area(mesh, db))
end

function nearest_source_index(dst::Integer, ndst::Integer, nsrc::Integer)
    src = div((2 * Int64(dst) + 1) * Int64(nsrc), 2 * Int64(ndst))
    return min(src, Int64(nsrc) - 1)
end

function load_permeability_file(params::DarcyParams, mesh::DarcyMesh)
    params.K_file === nothing && return mesh
    source_n = params.K_file_grid[1] == 0 ? mesh.n : params.K_file_grid
    all(source_n .> 0) || error("permeability source grid entries must be positive")
    values = parse.(Float64, split(read(params.K_file, String)))
    source_cells = Int64(source_n[1]) * source_n[2] * source_n[3]
    ncomponents = length(values) == source_cells ? 1 :
                  length(values) == 3 * source_cells ? 3 : 0
    ncomponents != 0 ||
        error("permeability file '$(params.K_file)' has $(length(values)) values; expected $source_cells or $(3 * source_cells)")

    Kinv_cells = Vector{Float64}(undef, 3 * Int(mesh.n_cells))
    for k in 0:(mesh.n[3] - 1), j in 0:(mesh.n[2] - 1), i in 0:(mesh.n[1] - 1)
        sk_physical = nearest_source_index(k, mesh.n[3], source_n[3])
        sk_file = params.K_file_top_down ? Int64(source_n[3]) - 1 - sk_physical : sk_physical
        sj = nearest_source_index(j, mesh.n[2], source_n[2])
        si = nearest_source_index(i, mesh.n[1], source_n[1])
        src = Int(si + Int64(source_n[1]) * (sj + Int64(source_n[2]) * sk_file)) + 1
        dst = Int(cell_index(mesh, i, j, k))
        for d in 1:3
            K = values[(ncomponents == 1 ? 0 : (d - 1) * Int(source_cells)) + src]
            isfinite(K) && K > 0.0 || error("permeability values must be positive and finite")
            Kinv_cells[3 * dst + d] = 1.0 / K
        end
    end
    return DarcyMesh(mesh.n, mesh.L, mesh.h, mesh.K, mesh.Kinv, mesh.dim, mesh.n_cells,
                     mesh.n_x_faces, mesh.n_y_faces, mesh.n_z_faces, mesh.n_faces,
                     mesh.n_total, Kinv_cells)
end

function starts_for(n::Int, p::Int)
    base = div(n, p)
    rest = n - base * p
    return [base * j + min(j, rest) for j in 0:p]
end

function parts_for_axis(n::Int, p::Int)
    starts = starts_for(n, p)
    owners = Vector{Int}(undef, n)
    for r in 0:(p - 1)
        for i in starts[r + 1]:(starts[r + 2] - 1)
            owners[i + 1] = r
        end
    end
    return starts[1:end-1], starts[2:end], owners
end

function factor_mpi_grid(mesh::DarcyMesh, nprocs::Int)
    parts = [1, 1, 1]
    factors = Int[]
    remaining = nprocs
    factor = 2
    while factor * factor <= remaining
        while remaining % factor == 0
            push!(factors, factor)
            remaining = div(remaining, factor)
        end
        factor += 1
    end
    remaining > 1 && push!(factors, remaining)
    for f in reverse(factors)
        best_dir = 1
        best_span = -1.0
        for d in 1:mesh.dim
            span = mesh.n[d] / parts[d]
            if span > best_span
                best_span = span
                best_dir = d
            end
        end
        parts[best_dir] *= f
    end
    return Tuple(parts)::NTuple{3,Int}
end

layout_rank(layout::DarcyLayout, rx::Int, ry::Int, rz::Int) =
    rx * layout.P[2] * layout.P[3] + ry * layout.P[3] + rz

function layout_rank_coords(layout::DarcyLayout, rank::Int)
    rx = div(rank, layout.P[2] * layout.P[3])
    rem1 = rank - rx * layout.P[2] * layout.P[3]
    ry = div(rem1, layout.P[3])
    rz = rem1 - ry * layout.P[3]
    return rx, ry, rz
end

function init_layout(mesh::DarcyMesh, P::NTuple{3,Int})
    x0, x1, xpart = parts_for_axis(mesh.n[1], P[1])
    y0, y1, ypart = parts_for_axis(mesh.n[2], P[2])
    z0, z1, zpart = parts_for_axis(mesh.n[3], P[3])
    nprocs = prod(P)
    offset = zeros(Int64, nprocs + 1)
    count_xf = zeros(Int64, nprocs)
    count_yf = zeros(Int64, nprocs)
    count_zf = zeros(Int64, nprocs)
    count_cells = zeros(Int64, nprocs)
    total = zeros(Int64, nprocs)
    for rank in 0:(nprocs - 1)
        rx, ry, rz = div(rank, P[2] * P[3]), div(rank % (P[2] * P[3]), P[3]), rank % P[3]
        nx = Int64(x1[rx + 1] - x0[rx + 1])
        ny = Int64(y1[ry + 1] - y0[ry + 1])
        nz = Int64(z1[rz + 1] - z0[rz + 1])
        count_xf[rank + 1] = (nx + (rx == P[1] - 1 ? 1 : 0)) * ny * nz
        mesh.dim >= 2 && (count_yf[rank + 1] = nx * (ny + (ry == P[2] - 1 ? 1 : 0)) * nz)
        mesh.dim >= 3 && (count_zf[rank + 1] = nx * ny * (nz + (rz == P[3] - 1 ? 1 : 0)))
        count_cells[rank + 1] = nx * ny * nz
        total[rank + 1] = count_xf[rank + 1] + count_yf[rank + 1] + count_zf[rank + 1] + count_cells[rank + 1]
        offset[rank + 2] = offset[rank + 1] + total[rank + 1]
    end
    return DarcyLayout(P, x0, x1, y0, y1, z0, z1, xpart, ypart, zpart, offset,
                       count_xf, count_yf, count_zf, count_cells, total, offset[end],
                       nprocs)
end

function layout_xface(layout::DarcyLayout, mesh::DarcyMesh, i::Integer, j::Integer, k::Integer)
    rx = i == mesh.n[1] ? layout.P[1] - 1 : layout.xpart[Int(i) + 1]
    ry = layout.ypart[Int(j) + 1]
    rz = layout.zpart[Int(k) + 1]
    r = layout_rank(layout, rx, ry, rz)
    nx = layout.x1[rx + 1] - layout.x0[rx + 1] + (rx == layout.P[1] - 1 ? 1 : 0)
    ny = layout.y1[ry + 1] - layout.y0[ry + 1]
    return layout.offset[r + 1] + Int64(i - layout.x0[rx + 1]) +
           Int64(nx) * (Int64(j - layout.y0[ry + 1]) + Int64(ny) * Int64(k - layout.z0[rz + 1]))
end

function layout_yface(layout::DarcyLayout, mesh::DarcyMesh, i::Integer, j::Integer, k::Integer)
    rx = layout.xpart[Int(i) + 1]
    ry = j == mesh.n[2] ? layout.P[2] - 1 : layout.ypart[Int(j) + 1]
    rz = layout.zpart[Int(k) + 1]
    r = layout_rank(layout, rx, ry, rz)
    nx = layout.x1[rx + 1] - layout.x0[rx + 1]
    ny = layout.y1[ry + 1] - layout.y0[ry + 1] + (ry == layout.P[2] - 1 ? 1 : 0)
    return layout.offset[r + 1] + layout.count_xf[r + 1] + Int64(i - layout.x0[rx + 1]) +
           Int64(nx) * (Int64(j - layout.y0[ry + 1]) + Int64(ny) * Int64(k - layout.z0[rz + 1]))
end

function layout_zface(layout::DarcyLayout, mesh::DarcyMesh, i::Integer, j::Integer, k::Integer)
    rx = layout.xpart[Int(i) + 1]
    ry = layout.ypart[Int(j) + 1]
    rz = k == mesh.n[3] ? layout.P[3] - 1 : layout.zpart[Int(k) + 1]
    r = layout_rank(layout, rx, ry, rz)
    nx = layout.x1[rx + 1] - layout.x0[rx + 1]
    ny = layout.y1[ry + 1] - layout.y0[ry + 1]
    return layout.offset[r + 1] + layout.count_xf[r + 1] + layout.count_yf[r + 1] +
           Int64(i - layout.x0[rx + 1]) + Int64(nx) * (Int64(j - layout.y0[ry + 1]) +
           Int64(ny) * Int64(k - layout.z0[rz + 1]))
end

function layout_cell(layout::DarcyLayout, mesh::DarcyMesh, i::Integer, j::Integer, k::Integer)
    rx = layout.xpart[Int(i) + 1]
    ry = layout.ypart[Int(j) + 1]
    rz = layout.zpart[Int(k) + 1]
    r = layout_rank(layout, rx, ry, rz)
    nx = layout.x1[rx + 1] - layout.x0[rx + 1]
    ny = layout.y1[ry + 1] - layout.y0[ry + 1]
    return layout.offset[r + 1] + layout.count_xf[r + 1] + layout.count_yf[r + 1] +
           layout.count_zf[r + 1] + Int64(i - layout.x0[rx + 1]) +
           Int64(nx) * (Int64(j - layout.y0[ry + 1]) + Int64(ny) * Int64(k - layout.z0[rz + 1]))
end

function is_pinned_neumann(mesh::DarcyMesh, axis::Int, i::Int, j::Int, k::Int, drive_axis::Int)
    low = axis == 1 ? i == 0 : axis == 2 ? j == 0 : k == 0
    high = axis == 1 ? i == mesh.n[1] : axis == 2 ? j == mesh.n[2] : k == mesh.n[3]
    return (low || high) && axis != drive_axis
end

function dirichlet_rhs(mesh::DarcyMesh, axis::Int, i::Int, j::Int, k::Int, drive_axis::Int)
    axis == drive_axis || return 0.0
    axis == 1 && return i == 0 ? 1.0 : 0.0
    axis == 2 && return j == 0 ? 1.0 : 0.0
    return k == 0 ? 1.0 : 0.0
end

function append_or_accumulate!(cols::Vector{Int64}, vals::Vector{Float64}, col::Int64, val::Float64)
    for q in eachindex(cols)
        if cols[q] == col
            vals[q] += val
            return
        end
    end
    push!(cols, col)
    push!(vals, val)
end

function build_system_csr(mesh::DarcyMesh, layout::DarcyLayout, rank::Int, drive_axis::Int)
    ilower = layout.offset[rank + 1]
    local_rows = Int(layout.total[rank + 1])
    local_rows > 0 || error("more ranks than unknowns are not supported")
    indptr = Int64[0]
    cols = Int64[]
    vals = Float64[]
    rhs = Float64[]
    dofmap = Cint[]
    rx, ry, rz = layout_rank_coords(layout, rank)
    x_start, x_end = layout.x0[rx + 1], layout.x1[rx + 1]
    y_start, y_end = layout.y0[ry + 1], layout.y1[ry + 1]
    z_start, z_end = layout.z0[rz + 1], layout.z1[rz + 1]
    nloc = 2 * mesh.dim

    function add_flux_adj_cell!(rcols, rvals, ci, cj, ck, local_face)
        c = cell_index(mesh, ci, cj, ck)
        faces, dirs, is_low, signs = rt0_cell_flux_dofs_layout(layout, mesh, ci, cj, ck)
        for bb in 1:nloc
            m = rt0_mass_entry(mesh, c, local_face, bb, dirs, is_low)
            m != 0.0 && append_or_accumulate!(rcols, rvals, faces[bb], m)
        end
        append_or_accumulate!(rcols, rvals, layout_cell(layout, mesh, ci, cj, ck),
                              -Float64(signs[local_face]))
    end

    function append_row!(rcols, rvals, b, label)
        append!(cols, rcols)
        append!(vals, rvals)
        push!(rhs, b)
        push!(dofmap, Cint(label))
        push!(indptr, Int64(length(cols)))
    end

    for k in z_start:(z_end - 1), j in y_start:(y_end - 1)
        x_face_end = x_end + (rx == layout.P[1] - 1 ? 1 : 0)
        for i in x_start:(x_face_end - 1)
            row = layout_xface(layout, mesh, i, j, k)
            rcols = Int64[]
            rvals = Float64[]
            if is_pinned_neumann(mesh, 1, i, j, k, drive_axis)
                push!(rcols, row); push!(rvals, 1.0)
            else
                i > 0 && add_flux_adj_cell!(rcols, rvals, i - 1, j, k, 2)
                i < mesh.n[1] && add_flux_adj_cell!(rcols, rvals, i, j, k, 1)
            end
            append_row!(rcols, rvals, dirichlet_rhs(mesh, 1, i, j, k, drive_axis), 1)
        end
    end

    if mesh.dim >= 2
        for k in z_start:(z_end - 1)
            y_face_end = y_end + (ry == layout.P[2] - 1 ? 1 : 0)
            for j in y_start:(y_face_end - 1), i in x_start:(x_end - 1)
                row = layout_yface(layout, mesh, i, j, k)
                rcols = Int64[]
                rvals = Float64[]
                if is_pinned_neumann(mesh, 2, i, j, k, drive_axis)
                    push!(rcols, row); push!(rvals, 1.0)
                else
                    j > 0 && add_flux_adj_cell!(rcols, rvals, i, j - 1, k, 4)
                    j < mesh.n[2] && add_flux_adj_cell!(rcols, rvals, i, j, k, 3)
                end
                append_row!(rcols, rvals, dirichlet_rhs(mesh, 2, i, j, k, drive_axis), 1)
            end
        end
    end

    if mesh.dim >= 3
        z_face_end = z_end + (rz == layout.P[3] - 1 ? 1 : 0)
        for k in z_start:(z_face_end - 1), j in y_start:(y_end - 1), i in x_start:(x_end - 1)
            row = layout_zface(layout, mesh, i, j, k)
            rcols = Int64[]
            rvals = Float64[]
            if is_pinned_neumann(mesh, 3, i, j, k, drive_axis)
                push!(rcols, row); push!(rvals, 1.0)
            else
                k > 0 && add_flux_adj_cell!(rcols, rvals, i, j, k - 1, 6)
                k < mesh.n[3] && add_flux_adj_cell!(rcols, rvals, i, j, k, 5)
            end
            append_row!(rcols, rvals, dirichlet_rhs(mesh, 3, i, j, k, drive_axis), 1)
        end
    end

    for k in z_start:(z_end - 1), j in y_start:(y_end - 1), i in x_start:(x_end - 1)
        faces, _, _, signs = rt0_cell_flux_dofs_layout(layout, mesh, i, j, k)
        rcols = Int64[]
        rvals = Float64[]
        for a in 1:nloc
            push!(rcols, faces[a])
            push!(rvals, -Float64(signs[a]))
        end
        append_row!(rcols, rvals, 0.0, 0)
    end

    length(rhs) == local_rows || error("layout row count mismatch ($(length(rhs)) != $local_rows)")
    return ilower, indptr, cols, vals, rhs, dofmap
end

function default_config(print_stats::Bool)
    stats = print_stats ? "1" : "0"
    return """
general:
  statistics: $stats
  exec_policy: host
linear_system:
  init_guess_mode: zeros
solver:
  gmres:
    max_iter: 200
    krylov_dim: 60
    relative_tol: 1.0e-10
    absolute_tol: 0.0
    print_level: 0
preconditioner:
  mgr:
    tolerance: 0.0
    max_iter: 1
    print_level: 0
    coarse_th: 0.0
    level:
      0:
        f_dofs: [1]
        f_relaxation: jacobi
        g_relaxation: none
        restriction_type: injection
        prolongation_type: jacobi
        coarse_level_type: rap
    coarsest_level:
      amg:
        tolerance: 0.0
        max_iter: 1
        print_level: 0
"""
end

function pressure_l2_error(mesh::DarcyMesh, layout::DarcyLayout, rank::Int, drive_axis::Int,
                           ilower::Int64, x::Vector{Float64})
    rx, ry, rz = layout_rank_coords(layout, rank)
    local_err = 0.0
    local_ref = 0.0
    vol = cell_volume(mesh)
    for k in layout.z0[rz + 1]:(layout.z1[rz + 1] - 1),
        j in layout.y0[ry + 1]:(layout.y1[ry + 1] - 1),
        i in layout.x0[rx + 1]:(layout.x1[rx + 1] - 1)
        row = layout_cell(layout, mesh, i, j, k)
        coord = ((i + 0.5) * mesh.h[1], (j + 0.5) * mesh.h[2], (k + 0.5) * mesh.h[3])
        u_ref = 1.0 - coord[drive_axis] / mesh.L[drive_axis]
        diff = x[Int(row - ilower + 1)] - u_ref
        local_err += diff * diff * vol
        local_ref += u_ref * u_ref * vol
    end
    return sqrt(hypredrive_mpi_world_sum(local_err) / hypredrive_mpi_world_sum(local_ref))
end

function vtk_extent_for_rank(mesh::DarcyMesh, layout::DarcyLayout, rank::Int)
    rx, ry, rz = layout_rank_coords(layout, rank)
    return (layout.x0[rx + 1], layout.x1[rx + 1],
            mesh.n[2] > 1 ? layout.y0[ry + 1] : 0, mesh.n[2] > 1 ? layout.y1[ry + 1] : 1,
            mesh.n[3] > 1 ? layout.z0[rz + 1] : 0, mesh.n[3] > 1 ? layout.z1[rz + 1] : 1)
end

function write_array_ascii(io::IO, data)
    for (q, v) in enumerate(data)
        q > 1 && print(io, ' ')
        print(io, v)
    end
    println(io)
end

function write_vti_piece(path::String, mesh::DarcyMesh, extent, pressure, flux, perm, cell_ids)
    sx, sy, sz = mesh.h[1], mesh.n[2] > 1 ? mesh.h[2] : 1.0, mesh.n[3] > 1 ? mesh.h[3] : 1.0
    id_type = mesh.n_cells > typemax(UInt32) ? "UInt64" : "UInt32"
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">")
        @printf(io, "  <ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%.17g %.17g %.17g\">\n",
                extent..., sx, sy, sz)
        @printf(io, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", extent...)
        println(io, "      <CellData Scalars=\"pressure\" Vectors=\"flux\" Tensors=\"permeability\">")
        println(io, "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">")
        write_array_ascii(io, pressure)
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"Float64\" Name=\"flux\" NumberOfComponents=\"3\" format=\"ascii\">")
        write_array_ascii(io, flux)
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"Float64\" Name=\"permeability\" NumberOfComponents=\"6\" ComponentName0=\"xx\" ComponentName1=\"yy\" ComponentName2=\"zz\" ComponentName3=\"xy\" ComponentName4=\"yz\" ComponentName5=\"xz\" format=\"ascii\">")
        write_array_ascii(io, perm)
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"$id_type\" Name=\"GlobalCellId\" format=\"ascii\">")
        write_array_ascii(io, cell_ids)
        println(io, "        </DataArray>")
        println(io, "      </CellData>")
        println(io, "      <PointData></PointData>")
        println(io, "    </Piece>")
        println(io, "  </ImageData>")
        println(io, "</VTKFile>")
    end
end

function write_pvti_master(path::String, piece_base::String, mesh::DarcyMesh, layout::DarcyLayout)
    whole = (0, mesh.n[1], 0, mesh.n[2] > 1 ? mesh.n[2] : 1, 0, mesh.n[3] > 1 ? mesh.n[3] : 1)
    sx, sy, sz = mesh.h[1], mesh.n[2] > 1 ? mesh.h[2] : 1.0, mesh.n[3] > 1 ? mesh.h[3] : 1.0
    id_type = mesh.n_cells > typemax(UInt32) ? "UInt64" : "UInt32"
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"PImageData\" version=\"1.0\" byte_order=\"LittleEndian\">")
        @printf(io, "  <PImageData WholeExtent=\"%d %d %d %d %d %d\" GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"%.17g %.17g %.17g\">\n",
                whole..., sx, sy, sz)
        println(io, "    <PCellData Scalars=\"pressure\" Vectors=\"flux\" Tensors=\"permeability\">")
        println(io, "      <PDataArray type=\"Float64\" Name=\"pressure\"/>")
        println(io, "      <PDataArray type=\"Float64\" Name=\"flux\" NumberOfComponents=\"3\"/>")
        println(io, "      <PDataArray type=\"Float64\" Name=\"permeability\" NumberOfComponents=\"6\" ComponentName0=\"xx\" ComponentName1=\"yy\" ComponentName2=\"zz\" ComponentName3=\"xy\" ComponentName4=\"yz\" ComponentName5=\"xz\"/>")
        println(io, "      <PDataArray type=\"$id_type\" Name=\"GlobalCellId\"/>")
        println(io, "    </PCellData>")
        println(io, "    <PPointData></PPointData>")
        for r in 0:(layout.nprocs - 1)
            extent = vtk_extent_for_rank(mesh, layout, r)
            @printf(io, "    <Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s_p%d.vti\"/>\n",
                    extent..., basename(piece_base), r)
        end
        println(io, "  </PImageData>")
        println(io, "</VTKFile>")
    end
end

function write_vtk_output(mesh::DarcyMesh, layout::DarcyLayout, rank::Int, nprocs::Int,
                          output_file::String, local_solution::Vector{Float64})
    counts = Cint.(layout.total)
    displs = Cint.(layout.offset[1:end-1])
    solution = HypreDrive._mpi_world_allgatherv(local_solution, counts, displs)
    rx, ry, rz = layout_rank_coords(layout, rank)
    n_cells = Int(layout.count_cells[rank + 1])
    pressure = Vector{Float64}(undef, n_cells)
    flux = zeros(Float64, 3 * n_cells)
    perm = zeros(Float64, 6 * n_cells)
    cell_ids = Vector{Int64}(undef, n_cells)
    ax, ay, az = face_area(mesh, 1), face_area(mesh, 2), face_area(mesh, 3)
    p = 1
    for k in layout.z0[rz + 1]:(layout.z1[rz + 1] - 1),
        j in layout.y0[ry + 1]:(layout.y1[ry + 1] - 1),
        i in layout.x0[rx + 1]:(layout.x1[rx + 1] - 1)
        cell = cell_index(mesh, i, j, k)
        pressure[p] = solution[Int(layout_cell(layout, mesh, i, j, k) + 1)]
        flux[3 * p - 2] = 0.5 * (solution[Int(layout_xface(layout, mesh, i, j, k) + 1)] +
                                 solution[Int(layout_xface(layout, mesh, i + 1, j, k) + 1)]) / ax
        if mesh.dim >= 2
            flux[3 * p - 1] = 0.5 * (solution[Int(layout_yface(layout, mesh, i, j, k) + 1)] +
                                     solution[Int(layout_yface(layout, mesh, i, j + 1, k) + 1)]) / ay
        end
        if mesh.dim >= 3
            flux[3 * p] = 0.5 * (solution[Int(layout_zface(layout, mesh, i, j, k) + 1)] +
                                 solution[Int(layout_zface(layout, mesh, i, j, k + 1) + 1)]) / az
        end
        kx, ky, kz = 1.0 / cell_Kinv(mesh, cell, 1), 1.0 / cell_Kinv(mesh, cell, 2), 1.0 / cell_Kinv(mesh, cell, 3)
        perm[(6 * p - 5):(6 * p)] = [kx, ky, kz, 0.0, 0.0, 0.0]
        cell_ids[p] = cell
        p += 1
    end
    extent = vtk_extent_for_rank(mesh, layout, rank)
    if nprocs == 1
        write_vti_piece(output_file, mesh, extent, pressure, flux, perm, cell_ids)
        rank == 0 && println("wrote                : $output_file")
    else
        base, _ = splitext(output_file)
        piece = "$(base)_p$(rank).vti"
        write_vti_piece(piece, mesh, extent, pressure, flux, perm, cell_ids)
        if rank == 0
            master = "$(base).pvti"
            write_pvti_master(master, base, mesh, layout)
            println("wrote                : $master (+ $nprocs piece(s))")
        end
    end
end

function print_setup(params::DarcyParams, mesh::DarcyMesh, layout::DarcyLayout, mpi_grid::NTuple{3,Int}, nprocs::Int)
    cell_block = ntuple(d -> cld(mesh.n[d], mpi_grid[d]), 3)
    rows_min, rows_max = minimum(layout.total), maximum(layout.total)
    println()
    println("=====================================================")
    println("              Julia Darcy Mixed Problem Setup")
    println("=====================================================")
    println("Discretization:       RT0/P0")
    println("Grid cells:           $(params.n[1]) x $(params.n[2]) x $(params.n[3])")
    println("Unknowns:             $(mesh.n_faces) flux + $(mesh.n_cells) pressure = $(mesh.n_total)")
    println("MPI ranks:            $nprocs")
    println("MPI grid partition:   $(mpi_grid[1]) x $(mpi_grid[2]) x $(mpi_grid[3]) ranks")
    println("Cell block target:    <= $(cell_block[1]) x $(cell_block[2]) x $(cell_block[3]) cells/rank")
    if rows_min == rows_max
        println("Row partition:        rank-contiguous spatial DOFs ($rows_min rows/rank)")
    else
        println("Row partition:        rank-contiguous spatial DOFs ($rows_min-$rows_max rows/rank)")
    end
    println("Drive direction:      $("xyz"[params.drive_axis])")
    if params.K_file !== nothing
        println("Permeability file:    $(params.K_file)")
        params.K_file_grid[1] != 0 &&
            println("Permeability grid:    $(params.K_file_grid[1]) x $(params.K_file_grid[2]) x $(params.K_file_grid[3]) ($(params.K_file_top_down ? "top-down" : "bottom-up"))")
    else
        @printf("Permeability diag:    %.3e %.3e %.3e\n", params.K...)
    end
    params.output_file !== nothing && println("VTK output:           $(params.output_file)")
    println("=====================================================")
    println()
end

function main()
    params = parse_args(ARGS)
    rank = hypredrive_mpi_world_rank()
    nprocs = hypredrive_mpi_world_size()
    validate_params(params, nprocs)
    mesh = load_permeability_file(params, init_mesh(params))
    mpi_grid = params.p_set ? params.p : factor_mpi_grid(mesh, nprocs)
    layout = init_layout(mesh, mpi_grid)
    rank == 0 && (params.verbose & 0x1) != 0 && print_setup(params, mesh, layout, mpi_grid, nprocs)

    ilower, indptr, cols, vals, rhs, dofmap = build_system_csr(mesh, layout, rank, params.drive_axis)
    config_arg = params.input === nothing ? default_config((params.verbose & 0x1) != 0) : params.input
    input_args = String[config_arg]
    append!(input_args, params.hypredrv_args)
    x, info = hypredrive_solve_mpi_csr(indptr, cols, vals, rhs, ilower;
                                       options="", input_args=input_args, dofmap=dofmap,
                                       comm=:world)

    if mesh.Kinv_cells === nothing
        rel_err = pressure_l2_error(mesh, layout, rank, params.drive_axis, ilower, x)
        rank == 0 && @printf("relative pressure L2 error : %.6e\n", rel_err)
        rel_err < 1.0e-6 || error("relative pressure L2 error exceeds tolerance")
    else
        rank == 0 && println("heterogeneous pressure solve completed")
    end

    params.output_file !== nothing && write_vtk_output(mesh, layout, rank, nprocs, params.output_file, x)
    if rank == 0 && (params.verbose & 0x1) != 0
        println("Iterations:           $(info.iterations)")
        println("Setup time:           $(info.setup_time)")
        println("Solve time:           $(info.solve_time)")
        println("Solution norm:        $(info.solution_norm)")
    end
end

main()
