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

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
A = laplacian1d(n)
b = ones(n)

x, info = hypredrive_solve(A, b)
relres = norm(A * x - b) / norm(b)

println("unknowns: ", n)
println("iterations: ", info.iterations)
println("relative residual: ", relres)
