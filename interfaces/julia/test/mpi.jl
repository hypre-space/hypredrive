# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

using HypreDrive
using LinearAlgebra
using SparseArrays
using Test

function laplacian1d(n::Integer)
    main = fill(2.0, n)
    off = fill(-1.0, n - 1)
    return spdiagm(-1 => off, 0 => main, 1 => off)
end

n = 48
A = laplacian1d(n)
exact = ones(n)
b = A * exact
rank = hypredrive_mpi_world_rank()
nprocs = hypredrive_mpi_world_size()

x_local, info = hypredrive_solve_mpi(A, b; options=hypredrive_options(solver=:pcg,
                                                                      preconditioner=:amg,
                                                                      pcg=(max_iter=100, relative_tol=1.0e-10),
                                                                      amg=(print_level=0,)))

row_start, row_end = HypreDrive._partition_rows(n, rank, nprocs)
local_n = row_end - row_start + 1
local_err2 = sum(abs2, x_local .- 1.0)
global_rms = sqrt(hypredrive_mpi_world_sum(local_err2) / n)

@test length(x_local) == local_n
@test global_rms < 1.0e-8
@test info.iterations >= 0

indptr = [0]
cols = Int[]
data = Float64[]
rhs = Float64[]
for row in row_start:row_end
    value = 2.0
    if row > 0
        push!(cols, row - 1)
        push!(data, -1.0)
        value -= 1.0
    end
    push!(cols, row)
    push!(data, 2.0)
    if row < n - 1
        push!(cols, row + 1)
        push!(data, -1.0)
        value -= 1.0
    end
    push!(rhs, value)
    push!(indptr, length(cols))
end
x_csr, info_csr = hypredrive_solve_mpi_csr(indptr, cols, data, rhs, row_start;
                                           options=hypredrive_options(solver=:pcg,
                                                                      preconditioner=:amg,
                                                                      pcg=(max_iter=20, relative_tol=1.0e-12),
                                                                      amg=(print_level=0,)),
                                           comm=:world)
@test x_csr ≈ ones(local_n)
@test info_csr.iterations >= 0
