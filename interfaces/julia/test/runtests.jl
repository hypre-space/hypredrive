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

    args_options = hypredrive_options(solver=:pcg, preconditioner=:amg,
                                      pcg=(max_iter=100, relative_tol=1.0e-10),
                                      amg=(print_level=0,))
    x_args, _ = hypredrive_solve(A, b; dofmap=zeros(Int, n),
                                 input_args=[args_options])
    @test norm(A * x_args - b) / norm(b) < 1.0e-8

    # A tolerated -a marker must not make a YAML-looking override value become
    # a second configuration file.
    x_yaml_value, _ = hypredrive_solve(A, b;
                                       input_args=[args_options, "-a", "--general:name",
                                                   "run.yml"])
    @test norm(A * x_yaml_value - b) / norm(b) < 1.0e-8
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

        set_relative_tol!(session, 1.0e-10)
        solve!(x, session)
        @test x ≈ [4.0, 4.0]
        @test_throws Exception set_relative_tol!(session, -1.0)

        set_matrix_csr!(session, indptr, cols, [4.0, 6.0], 0)
        set_rhs!(session, [8.0, 12.0])
        solve!(x, session)
        @test x ≈ [2.0, 2.0]
        @test info(session).setup_time > 0.0
    finally
        HypreDrive.close(session)
    end
end

@testset "extended session API" begin
    hypredrive_error_clear()
    @test_throws ArgumentError print_lib_info(:bogus)

    opts = hypredrive_options(solver=:pcg,
                              preconditioner=:amg,
                              pcg=(max_iter=30, relative_tol=1.0e-12),
                              amg=(print_level=0,))
    session = HypreDriveSession(options=opts)
    try
        set_object_name!(session, "julia-test")
        set_object_name!(session, nothing)

        @test input_warmup(session) isa Integer
        @test input_num_repetitions(session) >= 1
        @test input_num_linear_systems(session) >= 1
        @test input_num_precon_variants(session) >= 1

        set_matrix_csr!(session, [0, 1, 2], [0, 1], [2.0, 3.0], 0)
        set_rhs!(session, [4.0, 9.0])
        @test get_rhs_values(session) ≈ [4.0, 9.0]
        @test get_matrix_handle(session) != C_NULL
        @test get_rhs_handle(session) != C_NULL

        x = zeros(2)
        solve!(x, session)
        @test x ≈ [2.0, 3.0]
        @test get_solution_length(session) == 2
        @test get_solution_values(session) ≈ [2.0, 3.0]
        @test get_solution_handle(session) != C_NULL

        # The row count comes from the driver, not from a cached range: a
        # second session that only receives handles (no set_matrix_csr!) must
        # still report the right local length.
        other = HypreDriveSession(options=opts)
        try
            set_matrix_handle!(other, get_matrix_handle(session))
            set_rhs_handle!(other, get_rhs_handle(session))
            @test get_rhs_values(other) ≈ [4.0, 9.0]
            # Handle setters refresh the cached system state used by solve!.
            y = zeros(2)
            solve!(y, other)
            @test y ≈ [2.0, 3.0]
            set_matrix_handle!(other, C_NULL)  # clearing is allowed
            @test_throws ArgumentError solve!(y, other)

            # State-vector copies are sized from the registered vectors.
            three = HypreDriveSession(options=opts)
            try
                set_matrix_csr!(three, [0, 1, 2, 3], [0, 1, 2], [1.0, 1.0, 1.0], 0)
                set_rhs!(three, [1.0, 2.0, 3.0])
                @test_throws ArgumentError set_state_vectors!(
                    other, [get_rhs_handle(session), get_rhs_handle(three)])
                set_state_vectors!(other, [get_rhs_handle(three)])
                @test get_state_vector_values(other, 0) ≈ [1.0, 2.0, 3.0]
            finally
                HypreDrive.close(other)
                HypreDrive.close(three)
            end
        finally
            HypreDrive.close(other)
        end

    finally
        HypreDrive.close(session)
    end
    @test_throws ArgumentError set_object_name!(session, "closed")
end

const ps3d_dir = joinpath(@__DIR__, "..", "..", "..", "data", "ps3d10pt7", "np1")
if isfile(joinpath(ps3d_dir, "IJ.out.A.00000"))
    @testset "file-built system" begin
        hypredrive_error_clear()
        yaml = """
        linear_system:
          matrix_filename: $(joinpath(ps3d_dir, "IJ.out.A"))
          rhs_filename: $(joinpath(ps3d_dir, "IJ.out.b"))
          init_guess_mode: previous
        solver: pcg
        preconditioner: amg
        """
        session = HypreDriveSession(options=yaml)
        try
            # Build refreshes the cached rows and flags that solve! checks.
            linear_system_build!(session)
            x = zeros(1000)
            _, first = solve!(x, session)
            @test first.converged
            linear_system_build!(session)
            _, second = solve!(x, session)
            @test second.converged
            @test second.iterations < first.iterations
        finally
            HypreDrive.close(session)
        end
    end
end

@testset "presets, level stats and printing" begin
    hypredrive_error_clear()
    register_solver_preset("julia_pcg", "pcg:\n  max_iter: 50\n  relative_tol: 1.0e-8\n", "julia test")
    register_precon_preset("julia_amg", "amg:\n  print_level: 0\n", "julia test")

    opts = hypredrive_options(solver=:pcg, preconditioner=:amg,
                              pcg=(max_iter=30, relative_tol=1.0e-12),
                              amg=(print_level=0,))
    session = HypreDriveSession(options=opts)
    try
        set_solver_preset!(session, "julia_pcg")
        set_precon_preset!(session, "julia_amg")
        @test_throws Exception set_solver_preset!(session, "no_such_preset_xyz")
        hypredrive_error_clear()

        set_matrix_csr!(session, [0, 1, 2], [0, 1], [2.0, 3.0], 0)
        set_rhs!(session, [4.0, 9.0])

        annotate_level_begin(session, 0, "timestep", 1)
        solve!(zeros(2), session)
        annotate_level_end(session, 0, "timestep", 1)

        @test stats_level_count(session, 0) == 1
        entry = stats_level_entry(session, 0, 0)
        @test keys(entry) == (:entry_id, :num_solves, :linear_iters, :setup_time, :solve_time)
        @test entry.entry_id >= 1
        @test entry.num_solves >= 1
        @test_throws Exception stats_level_count(session, 999)
        hypredrive_error_clear()

        mktempdir() do dir
            print_dofmap_file = joinpath(dir, "dofmap")
            set_matrix_csr!(session, [0, 1, 2], [0, 1], [2.0, 3.0], 0; dofmap=[0, 1])
            print_dofmap(session, print_dofmap_file)
            @test any(startswith("dofmap"), readdir(dir))
        end

        redirect_stdout(devnull) do
            print_stats_level(session, 0)
            print_stats(session)
            print_linear_system(session)
        end
    finally
        HypreDrive.close(session)
    end

    redirect_stdout(devnull) do
        print_system_info(:self)
        print_exit_info("julia-test", :self)
    end
end
