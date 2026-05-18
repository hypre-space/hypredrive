% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT

function test_hypredrive_mex
%TEST_HYPREDRIVE_MEX Smoke test for the hypredrive MATLAB MEX interface.

n = 16;
e = ones(n, 1);
A = spdiags([-e, 2 * e, -e], -1:1, n, n);
b = ones(n, 1);

opts = hypredrive_options( ...
    'solver', 'pcg', ...
    'preconditioner', 'amg', ...
    'pcg', struct('max_iter', 100, 'relative_tol', 1.0e-8), ...
    'amg', struct('max_iter', 1, 'tolerance', 0.0, 'print_level', 0));

[x, info] = hypredrive_solve(A, b, opts);
check_solution(A, b, x, info);

yaml = sprintf(['solver:\n', ...
                '  gmres:\n', ...
                '    max_iter: 100\n', ...
                '    relative_tol: 1.0e-8\n', ...
                'preconditioner:\n', ...
                '  amg:\n', ...
                '    max_iter: 1\n', ...
                '    tolerance: 0.0\n', ...
                '    print_level: 0\n', ...
                'general:\n', ...
                '  statistics: 0\n']);
[x, info] = hypredrive_solve(A, b, yaml);
check_solution(A, b, x, info);

opts_struct = struct();
opts_struct.solver.pcg.max_iter = 100;
opts_struct.solver.pcg.relative_tol = 1.0e-8;
opts_struct.preconditioner.amg.max_iter = 1;
opts_struct.preconditioner.amg.tolerance = 0.0;
opts_struct.preconditioner.amg.print_level = 0;
[x, info] = hypredrive_solve(A, b, opts_struct);
check_solution(A, b, x, info);

expect_error(@() hypredrive_solve(full(A), b, opts), 'hypredrive:InvalidMatrix');
expect_error(@() hypredrive_solve(A, b(1:end - 1), opts), 'hypredrive:InvalidRHS');
expect_error(@() hypredrive_options('solver', 'pcgg'), 'hypredrive:InvalidOptions');
expect_error(@() hypredrive_options('preconditioner', 'schwarz'), ...
             'hypredrive:InvalidOptions');

yaml_none = hypredrive_options('solver', 'pcg', 'preconditioner', 'none');
if ~isempty(strfind(yaml_none, 'preconditioner'))
    error('hypredrive:test', 'preconditioner none should omit the preconditioner block');
end
yaml_mgr = hypredrive_options('preconditioner', 'mgr', 'mgr', struct());
if isempty(strfind(yaml_mgr, 'preconditioner:')) || isempty(strfind(yaml_mgr, 'mgr: {}'))
    error('hypredrive:test', 'MGR preconditioner options were not emitted');
end
expect_error(@() hypredrive_options(struct('general', struct('statistics', Inf))), ...
             'hypredrive:InvalidOptions');

fprintf('hypredrive MATLAB/Octave smoke test passed\n');
end

function check_solution(A, b, x, info)
if ~iscolumn(x) || numel(x) ~= numel(b)
    error('hypredrive:test', 'solution has the wrong shape');
end
if norm(b - A * x) / norm(b) > 1.0e-7
    error('hypredrive:test', 'relative residual is too large');
end
if ~isfield(info, 'iterations') || info.iterations < 0
    error('hypredrive:test', 'missing iteration count');
end
if ~isfield(info, 'solution_norm') || info.solution_norm <= 0
    error('hypredrive:test', 'missing solution norm');
end
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
