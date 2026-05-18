% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT
%
%LAPLACIAN1D Solve a small 1D Poisson problem with hypredrive.
%
% Run from a build tree containing the hypredrive MEX file:
%
%   addpath('/path/to/build');
%   laplacian1d

n = 64;
e = ones(n, 1);
A = spdiags([-e, 2 * e, -e], -1:1, n, n);
b = ones(n, 1);

opts = hypredrive_options( ...
    'solver', 'pcg', ...
    'preconditioner', 'amg', ...
    'pcg', struct('max_iter', 100, 'relative_tol', 1.0e-8), ...
    'amg', struct('max_iter', 1, 'tolerance', 0.0, 'print_level', 0));

[x, info] = hypredrive_solve(A, b, opts);

fprintf('iterations: %d\n', info.iterations);
fprintf('setup time: %.6e s\n', info.setup_time);
fprintf('solve time: %.6e s\n', info.solve_time);
fprintf('solution l2 norm: %.6e\n', info.solution_norm);
fprintf('relative residual: %.6e\n', norm(b - A * x) / norm(b));
