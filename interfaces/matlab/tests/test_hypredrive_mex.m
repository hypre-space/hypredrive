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

[x, info] = hypredrive(A, b, opts);

if ~iscolumn(x) || numel(x) ~= n
    error("hypredrive:test", "solution has the wrong shape");
end
if norm(b - A * x) / norm(b) > 1.0e-7
    error("hypredrive:test", "relative residual is too large");
end
if ~isfield(info, 'iterations') || info.iterations < 0
    error('hypredrive:test', 'missing iteration count');
end
if ~isfield(info, 'solution_norm') || info.solution_norm <= 0
    error('hypredrive:test', 'missing solution norm');
end

fprintf('hypredrive MATLAB/Octave smoke test passed\n');
end
