#!/usr/bin/env julia
# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using Test

mutable struct Args
    mode::String
    c_darcy::String
    julia_exe::String
    project::String
    example::String
    mpiexec::String
    np_flag::String
    perm_file::String
end

function parse_args(argv)
    args = Args("", "", "", "", "", "", "-n", "")
    i = 1
    while i <= length(argv)
        key = argv[i]
        if key == "--mode"
            i += 1; args.mode = argv[i]
        elseif key == "--c-darcy"
            i += 1; args.c_darcy = argv[i]
        elseif key == "--julia-exe"
            i += 1; args.julia_exe = argv[i]
        elseif key == "--project"
            i += 1; args.project = argv[i]
        elseif key == "--example"
            i += 1; args.example = argv[i]
        elseif key == "--mpiexec"
            i += 1; args.mpiexec = argv[i]
        elseif key == "--np-flag"
            i += 1; args.np_flag = argv[i]
        elseif key == "--perm-file"
            i += 1; args.perm_file = argv[i]
        else
            error("unknown argument: $key")
        end
        i += 1
    end
    isempty(args.mode) && error("--mode is required")
    isempty(args.julia_exe) && error("--julia-exe is required")
    isempty(args.project) && error("--project is required")
    isempty(args.example) && error("--example is required")
    return args
end

function capture(command::Vector{String})
    return read(pipeline(Cmd(command), stderr=stdout), String)
end

function julia_cmd(args::Args, extra::Vector{String})
    return [args.julia_exe, "--startup-file=no", "--project=$(args.project)", args.example, extra...]
end

function mpi_cmd(args::Args, ranks::Int, command::Vector{String})
    return [args.mpiexec, args.np_flag, string(ranks), command...]
end

function pressure_error(output::String)
    m = match(r"relative pressure L2 error\s*:\s*([0-9.eE+-]+)", output)
    m !== nothing || error("missing relative pressure L2 error in output:\n$output")
    return parse(Float64, m.captures[1])
end

function compare_errors(c_output::String, jl_output::String)
    c_err = pressure_error(c_output)
    jl_err = pressure_error(jl_output)
    @test c_err < 1.0e-8
    @test jl_err < 1.0e-8
    @test abs(c_err - jl_err) < 1.0e-10
end

args = parse_args(ARGS)

if args.mode == "help"
    out = capture(julia_cmd(args, ["--help"]))
    @test occursin("Usage:", out)
    @test occursin("--procs", out)
    @test occursin("--args", out)
    @test occursin("--K-file-grid", out)
    @test occursin("--output", out)
    @test occursin("Verbosity bitset", out)
elseif args.mode == "serial"
    !isempty(args.c_darcy) || error("--c-darcy is required")
    c_out = capture([args.c_darcy, "-n", "4", "3", "1", "-g", "x", "-v", "1"])
    jl_out = capture(julia_cmd(args, ["-n", "4", "3", "1", "-g", "x", "-v", "1"]))
    @test occursin("Darcy Mixed Problem Setup", c_out)
    @test occursin("Julia Darcy Mixed Problem Setup", jl_out)
    compare_errors(c_out, jl_out)
elseif args.mode == "mpi"
    !isempty(args.c_darcy) || error("--c-darcy is required")
    !isempty(args.mpiexec) || error("--mpiexec is required")
    c_out = capture(mpi_cmd(args, 2, [args.c_darcy, "-n", "4", "3", "1", "-P", "1", "2", "1", "-g", "x", "-v", "1"]))
    jl_out = capture(mpi_cmd(args, 2, julia_cmd(args, ["-n", "4", "3", "1", "-P", "1", "2", "1", "-g", "x", "-v", "1"])))
    @test occursin("MPI grid partition:   1 x 2 x 1 ranks", c_out)
    @test occursin("MPI grid partition:   1 x 2 x 1 ranks", jl_out)
    compare_errors(c_out, jl_out)
elseif args.mode == "heterogeneous"
    !isempty(args.c_darcy) || error("--c-darcy is required")
    !isempty(args.perm_file) || error("--perm-file is required")
    c_out = capture([args.c_darcy, "-n", "4", "3", "1", "--K-file", args.perm_file, "-g", "x", "-v", "0"])
    jl_out = capture(julia_cmd(args, ["-n", "4", "3", "1", "--K-file", args.perm_file, "-g", "x", "-v", "0"]))
    @test occursin("heterogeneous pressure solve completed", c_out)
    @test occursin("heterogeneous pressure solve completed", jl_out)
else
    error("unknown --mode $(args.mode)")
end
