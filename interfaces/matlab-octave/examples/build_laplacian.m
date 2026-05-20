% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT
%
%BUILD_LAPLACIAN Assemble the negative finite-difference Laplacian.
%
% A = BUILD_LAPLACIAN(NX, NY, NZ) returns the (NX*NY*NZ)-by-(NX*NY*NZ)
% sparse matrix arising from the standard 3-pt / 5-pt / 7-pt stencil on a
% uniform grid with Dirichlet boundaries and unit spacing. A dimension
% equal to 1 is treated as inactive (so BUILD_LAPLACIAN(N, 1, 1) returns
% the 1D operator). At least one dimension must be greater than 1.
%
% Compatible with MATLAB and GNU Octave.

function A = build_laplacian(nx, ny, nz)
    if nargin < 1
        error('build_laplacian:badInput', 'nx is required');
    end
    if nargin < 2, ny = 1; end
    if nargin < 3, nz = 1; end

    dims = [nx, ny, nz];
    if any(dims < 1) || any(dims ~= floor(dims))
        error('build_laplacian:badInput', ...
              'grid dimensions must be positive integers');
    end
    if all(dims == 1)
        error('build_laplacian:badInput', ...
              'at least one grid dimension must be greater than 1');
    end

    N = prod(dims);
    A = sparse(N, N);
    for k = 1:3
        if dims(k) == 1
            continue;
        end
        e = ones(dims(k), 1);
        Lk = spdiags([-e, 2 * e, -e], -1:1, dims(k), dims(k));
        inner = prod(dims(1:k - 1));
        outer = prod(dims(k + 1:end));
        A = A + kron(speye(outer), kron(Lk, speye(inner)));
    end
end
