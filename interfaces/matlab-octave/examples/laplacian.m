% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT
%
%LAPLACIAN Solve a 1D / 2D / 3D Poisson problem with hypredrive.
%
% The same script builds the standard finite-difference negative Laplacian
% (3-pt / 5-pt / 7-pt stencil) on a uniform grid with Dirichlet boundaries
% and grid spacing h = 1, then solves it with PCG preconditioned by AMG.
%
% Run from a build tree containing the hypredrive MEX file:
%
%   addpath('/path/to/build');
%   laplacian            % default 3D 16 x 16 x 16
%   laplacian(64)        % 1D, n = 64
%   laplacian(32, 32)    % 2D, 32 x 32
%   laplacian(16, 16, 8) % 3D, 16 x 16 x 8
%
% Compatible with MATLAB and GNU Octave.

function laplacian(nx, ny, nz)
    if nargin == 0
        nx = 16; ny = 16; nz = 16;
    else
        if nargin < 2, ny = 1; end
        if nargin < 3, nz = 1; end
    end

    A = build_laplacian(nx, ny, nz);
    n = size(A, 1);
    b = ones(n, 1);

    opts = hypredrive_options( ...
        'solver', 'pcg', ...
        'preconditioner', 'amg', ...
        'pcg', struct('max_iter', 100, 'relative_tol', 1.0e-8), ...
        'amg', struct('max_iter', 1, 'tolerance', 0.0, 'print_level', 0));

    [x, info] = hypredrive_solve(A, b, opts);

    dims = [nx, ny, nz];
    active = dims(dims > 1);
    parts = arrayfun(@(d) sprintf('%d', d), active, 'UniformOutput', false);
    grid_str = strjoin(parts, ' x ');
    ndim = numel(active);

    fprintf('grid:              %s (%dD)\n', grid_str, ndim);
    fprintf('unknowns:          %d\n', n);
    fprintf('iterations:        %d\n', info.iterations);
    fprintf('setup time:        %.6e s\n', info.setup_time);
    fprintf('solve time:        %.6e s\n', info.solve_time);
    fprintf('solution l2 norm:  %.6e\n', info.solution_norm);
    relres = norm(b - A * x) / norm(b);
    fprintf('relative residual: %.6e\n', relres);
    if ~isfinite(relres) || relres > 1.0e-6
        error('laplacian:convergence', ...
              'relative residual %.3e exceeds 1e-6', relres);
    end
end
