# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using HypreDrive
using LinearAlgebra
using SparseArrays

function laplacian1d(n::Integer)
    main = fill(2.0, n)
    off = fill(-1.0, n - 1)
    return spdiagm(-1 => off, 0 => main, 1 => off)
end

# Hypredrive YAML overrides given after -a/--args, e.g.
# julia laplacian1d.jl 64 -a --solver:pcg:max_iter 100
overrides_at = findfirst(a -> a in ("-a", "--args"), ARGS)
positional = overrides_at === nothing ? ARGS : ARGS[1:overrides_at - 1]
n = length(positional) >= 1 ? parse(Int, positional[1]) : 128
A = laplacian1d(n)
b = ones(n)

input_args = overrides_at === nothing ? nothing :
             vcat(String[hypredrive_options()], ARGS[overrides_at:end])
x, info = hypredrive_solve(A, b; input_args=input_args)
relres = norm(A * x - b) / norm(b)

println("unknowns: ", n)
println("iterations: ", info.iterations)
println("relative residual: ", relres)
