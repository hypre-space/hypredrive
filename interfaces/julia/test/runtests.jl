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

@testset "options" begin
    yaml_kwargs = hypredrive_options(solver=:pcg, preconditioner=:amg,
                                     pcg=(max_iter=50, relative_tol=1.0e-10),
                                     amg=(print_level=0,))
    yaml_dict = hypredrive_options(Dict("solver" => Dict("pcg" => Dict("max_iter" => 50))))
    yaml_named = hypredrive_options((solver=:pcg, pcg=(max_iter=50,)))
    yaml_string = hypredrive_options("solver:\n  pcg: {}\n")

    for yaml in (yaml_kwargs, yaml_dict, yaml_named, yaml_string)
        @test occursin("general:", yaml)
        @test occursin("statistics: 0", yaml)
    end
    @test occursin("preconditioner:", yaml_kwargs)
    @test occursin("pcg:", yaml_named)
    @test occursin("statistics: 2", hypredrive_options(general=(statistics=2,)))
    @test occursin("statistics: 2", hypredrive_options(Dict("general" => Dict("statistics" => 2))))
    @test occursin("statistics: 2", hypredrive_options("general:\n  statistics: 2\nsolver:\n  pcg: {}\n"))
end

@testset "serial solve" begin
    n = 32
    A = laplacian1d(n)
    b = ones(n)
    x, info = hypredrive_solve(A, b; options=hypredrive_options(solver=:pcg,
                                                                preconditioner=:amg,
                                                                pcg=(max_iter=100, relative_tol=1.0e-10),
                                                                amg=(print_level=0,)))
    @test length(x) == n
    @test norm(A * x - b) / norm(b) < 1.0e-8
    @test info.iterations >= 0
    @test info.setup_time >= 0.0
    @test info.solve_time >= 0.0
    @test info.solution_norm > 0.0

    x_args, _ = hypredrive_solve(A, b; dofmap=zeros(Int, n),
                                 input_args=[hypredrive_options(solver=:pcg,
                                                                preconditioner=:amg,
                                                                pcg=(max_iter=100, relative_tol=1.0e-10),
                                                                amg=(print_level=0,))])
    @test norm(A * x_args - b) / norm(b) < 1.0e-8
end

@testset "input validation" begin
    A = laplacian1d(8)
    @test_throws ArgumentError hypredrive_solve(A, ones(7))
    @test_throws hypredrive_error hypredrive_solve(A, ones(8); options="solver: [")
    @test hypredrive_comm_rank(:self) == 0
    @test hypredrive_comm_size(:self) == 1
end

@testset "public CSR solve" begin
    indptr = [0, 1, 2]
    cols = [0, 1]
    data = [2.0, 3.0]
    rhs = [4.0, 9.0]
    # The public CSR path defaults to MPI_COMM_SELF, so it is safe without mpiexec.
    x, info = hypredrive_solve_mpi_csr(indptr, cols, data, rhs, 0;
                                       options=hypredrive_options(solver=:pcg,
                                                                  preconditioner=:amg,
                                                                  pcg=(max_iter=20, relative_tol=1.0e-12),
                                                                  amg=(print_level=0,)),
                                       comm=:self)
    @test x ≈ [2.0, 3.0]
    @test info.iterations >= 0
    @test_throws ArgumentError hypredrive_solve_mpi_csr(indptr, cols, data, rhs, 0;
                                                        options=hypredrive_options(solver=:pcg,
                                                                                   preconditioner=:amg),
                                                        comm=:self,
                                                        dofmap=[0, typemax(Int64)])
end

@testset "reusable session" begin
    opts = hypredrive_options(solver=:pcg,
                              preconditioner=:amg,
                              pcg=(max_iter=30, relative_tol=1.0e-12),
                              amg=(print_level=0,))
    session = HypreDriveSession(options=opts)
    try
        indptr = [0, 1, 2]
        cols = [0, 1]
        data = [2.0, 3.0]
        x = zeros(2)
        set_matrix_csr!(session, indptr, cols, data, 0)
        set_rhs!(session, [4.0, 9.0])
        solve!(x, session)
        @test x ≈ [2.0, 3.0]
        @test info(session).iterations >= 0

        set_rhs!(session, [8.0, 12.0])
        solve!(x, session)
        @test x ≈ [4.0, 4.0]

        set_matrix_csr!(session, indptr, cols, [4.0, 6.0], 0)
        set_rhs!(session, [8.0, 12.0])
        solve!(x, session)
        @test x ≈ [2.0, 2.0]
    finally
        HypreDrive.close(session)
    end
end
