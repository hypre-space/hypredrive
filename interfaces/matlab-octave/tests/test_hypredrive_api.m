% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT

function test_hypredrive_api
%TEST_HYPREDRIVE_API Exercise the handle-based hypredrive_mex command API.

n = 16;
A = build_laplacian(n);
b = ones(n, 1);

% Command table is introspectable and covers the core lifecycle.
cmds = hypredrive_mex('commands');
for name = {'create', 'destroy', 'set_matrix_csr', 'solver_apply', 'annotate_level_end'}
    if ~any(strcmp(cmds, name{1}))
        error('hypredrive:test', 'missing command %s', name{1});
    end
end

h0 = hypredrive_mex('num_handles');

% Full driver lifecycle on MATLAB-native data.
yaml = hypredrive_options('solver', 'pcg', 'preconditioner', 'amg', ...
                          'amg', struct('print_level', 0));
drv = hypredrive_mex('create', yaml);
if ~isa(drv, 'uint64') || ~hypredrive_mex('is_valid', drv)
    error('hypredrive:test', 'create did not return a valid handle');
end
hypredrive_mex('set_name', drv, 'api-test');
hypredrive_mex('set_matrix_csr', drv, A);
hypredrive_mex('set_rhs_array', drv, b);
hypredrive_mex('set_initial_guess', drv, []);
hypredrive_mex('reset_initial_guess', drv);
hypredrive_mex('solver_create', drv);
hypredrive_mex('solver_setup', drv);
hypredrive_mex('solver_apply', drv);
x = hypredrive_mex('copy_solution_values', drv);
if norm(b - A * x) / norm(b) > 1.0e-7
    error('hypredrive:test', 'driver solve residual too large');
end
x2 = hypredrive_mex('get_solution_values', drv);
if norm(x - x2) > 0 || hypredrive_mex('get_solution_length', drv) ~= n
    error('hypredrive:test', 'solution accessors disagree');
end
if ~hypredrive_mex('solver_converged', drv) || hypredrive_mex('solver_num_iter', drv) < 0
    error('hypredrive:test', 'solver status accessors failed');
end
info = hypredrive_mex('solver_info', drv);
if ~info.converged
    error('hypredrive:test', 'solver_info reports no convergence');
end
if abs(hypredrive_mex('get_solution_norm', drv, 'l2') - norm(x)) > 1.0e-10
    error('hypredrive:test', 'solution norm mismatch');
end

% Tolerance can be tightened and the solve repeated.
hypredrive_mex('solver_set_relative_tol', drv, 1.0e-12);
hypredrive_mex('reset_initial_guess', drv);
hypredrive_mex('solver_apply', drv);

% Borrowed handles are valid while the driver lives and die with it.
hm = hypredrive_mex('get_matrix', drv);
hv = hypredrive_mex('get_rhs', drv);
if ~hypredrive_mex('is_valid', hm) || norm(hypredrive_mex('vector_get_values', hv) - b) > 0
    error('hypredrive:test', 'borrowed handles are not usable');
end
hypredrive_mex('destroy', drv);
if hypredrive_mex('is_valid', drv) || hypredrive_mex('is_valid', hm)
    error('hypredrive:test', 'handles survived driver destruction');
end
expect_error(@() hypredrive_mex('solver_apply', drv), 'hypredrive:InvalidHandle');
expect_error(@() hypredrive_mex('destroy', drv), 'hypredrive:InvalidHandle');

% Owned IJ matrix/vector handles, attached through the raw-object commands.
drv = hypredrive_mex('create', yaml);
hA = hypredrive_mex('matrix_from_sparse', A);
hb = hypredrive_mex('vector_from_array', b);
hypredrive_mex('set_matrix', drv, hA);
hypredrive_mex('set_rhs', drv, hb);
expect_error(@() hypredrive_mex('destroy', hA), 'hypredrive:ObjectInUse');
expect_error(@() hypredrive_mex('set_rhs', drv, hA), 'hypredrive:InvalidHandle');
hypredrive_mex('solver_create', drv);
hypredrive_mex('solver_setup', drv);
hypredrive_mex('solver_apply', drv);
x3 = hypredrive_mex('get_solution_values', drv);
if norm(b - A * x3) / norm(b) > 1.0e-7
    error('hypredrive:test', 'IJ-handle solve residual too large');
end
hypredrive_mex('destroy', drv);
hypredrive_mex('destroy', hA);
hypredrive_mex('destroy', hb);

% Annotations, statistics levels, and presets.
drv = hypredrive_mex('create', yaml);
hypredrive_mex('annotate_level_begin', drv, 0, 'timestep', 0);
hypredrive_mex('annotate_level_end', drv, 0, 'timestep', 0);
hypredrive_mex('annotate_begin', drv, 'matrix');
hypredrive_mex('annotate_end', drv, 'matrix');
try
    hypredrive_mex('stats_level_count', drv, 0);
catch err
    if ~strcmp(err.identifier, 'hypredrive:HypreDriveError')
        rethrow(err);
    end
    hypredrive_mex('error_clear');
end
hypredrive_mex('destroy', drv);

hypredrive_mex('error_clear');
expect_error(@() hypredrive_mex('no_such_command'), 'hypredrive:UnknownCommand');
expect_error(@() hypredrive_mex('create', 1, 2), 'hypredrive:InvalidNumInputs');
expect_error(@() hypredrive_mex('solver_apply', uint64(123456)), 'hypredrive:InvalidHandle');
expect_error(@() hypredrive_mex('solver_apply', 1), 'hypredrive:InvalidHandle');

% The one-shot solve is unchanged and leaks no handles.
[xs, ~] = hypredrive_solve(A, b);
if norm(b - A * xs) / norm(b) > 1.0e-7
    error('hypredrive:test', 'one-shot solve regressed');
end

% destroy_all releases everything.
d1 = hypredrive_mex('create');
d2 = hypredrive_mex('create'); %#ok<NASGU>
hypredrive_mex('destroy_all');
if hypredrive_mex('is_valid', d1) || hypredrive_mex('num_handles') ~= 0
    error('hypredrive:test', 'destroy_all left handles behind');
end
if h0 ~= 0
    error('hypredrive:test', 'handles leaked before this test started');
end

fprintf('hypredrive MATLAB/Octave API test passed\n');

% A borrowed handle must not outlive the object it points to: replacing the
% driver's RHS or matrix invalidates handles borrowed before the replacement.
drv = hypredrive_mex('create', yaml);
hypredrive_mex('set_matrix_csr', drv, A);
hypredrive_mex('set_rhs_array', drv, b);
hv = hypredrive_mex('get_rhs', drv);
hm = hypredrive_mex('get_matrix', drv);
if ~hypredrive_mex('is_valid', hv) || ~hypredrive_mex('is_valid', hm)
    error('hypredrive:test', 'fresh borrowed handles should be valid');
end
hypredrive_mex('set_rhs_array', drv, 2 * b);
if hypredrive_mex('is_valid', hv)
    error('hypredrive:test', 'borrowed RHS handle survived the RHS being replaced');
end
expect_error(@() hypredrive_mex('vector_get_values', hv), 'hypredrive:InvalidHandle');
hv2 = hypredrive_mex('get_rhs', drv);
if norm(hypredrive_mex('vector_get_values', hv2) - 2 * b) > 0
    error('hypredrive:test', 'a fresh borrowed handle should see the new RHS');
end
hypredrive_mex('set_matrix_csr', drv, A);
if hypredrive_mex('is_valid', hm)
    error('hypredrive:test', 'borrowed matrix handle survived the matrix being replaced');
end
hypredrive_mex('destroy', drv);

% An owned matrix shared by two drivers must stay alive until both are gone:
% destroying one driver must not make the matrix look free while the other
% still holds it.
d1 = hypredrive_mex('create', yaml);
d2 = hypredrive_mex('create', yaml);
hA = hypredrive_mex('matrix_from_sparse', A);
hypredrive_mex('set_matrix', d1, hA);
hypredrive_mex('set_matrix', d2, hA);
hypredrive_mex('destroy', d2);
expect_error(@() hypredrive_mex('destroy', hA), 'hypredrive:ObjectInUse');
hypredrive_mex('destroy', d1);
hypredrive_mex('destroy', hA);

% A handle borrowed from one driver cannot be given to another: it would dangle
% once its own driver is destroyed.
d1 = hypredrive_mex('create', yaml);
d2 = hypredrive_mex('create', yaml);
hypredrive_mex('set_matrix_csr', d1, A);
hypredrive_mex('set_rhs_array', d1, b);
hb = hypredrive_mex('get_rhs', d1);
expect_error(@() hypredrive_mex('set_rhs', d2, hb), 'hypredrive:InvalidHandle');
hypredrive_mex('destroy', d1);
hypredrive_mex('destroy', d2);
end

function expect_error(fn, id)
try
    fn();
catch err
    if strcmp(err.identifier, id)
        return;
    end
    error('hypredrive:test', 'expected %s, got %s', id, err.identifier);
end
error('hypredrive:test', 'expected error %s', id);
end
