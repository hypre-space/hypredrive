% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT
%
%BUILD_ELASTICITY Assemble a Q1 small-strain linear-elasticity system.
%
% [K, b] = BUILD_ELASTICITY(NX, NY, NZ) returns the global stiffness matrix
% and body-force RHS for a unit-cube domain [0,1]^d, where d is the number
% of dimensions with more than one node. The discretization is Q1:
%   * 2-node bar in 1D, 4-node bilinear quad (plane strain) in 2D,
%   * 8-node trilinear hex in 3D.
% Material is isotropic with E = 1, nu = 0.3 (nu unused in 1D). The x = 0
% face is clamped (all displacement components fixed to 0). Loading is a
% unit body force along the last active axis (e.g., gravity in -y for 2D/3D,
% along -x in 1D). The returned K is SPD after clamp application.
%
% Compatible with MATLAB and GNU Octave.

function [K, b] = build_elasticity(nx, ny, nz)
    if nargin < 1
        error('build_elasticity:badInput', 'nx is required');
    end
    if nargin < 2, ny = 1; end
    if nargin < 3, nz = 1; end

    dims = [nx, ny, nz];
    if any(dims < 1) || any(dims ~= floor(dims))
        error('build_elasticity:badInput', ...
              'grid dimensions must be positive integers');
    end
    if all(dims == 1)
        error('build_elasticity:badInput', ...
              'at least one grid dimension must be greater than 1');
    end

    active = dims > 1;
    n = dims(active);
    ndim = numel(n);

    E = 1.0;
    nu = 0.3;
    h = 1.0 ./ (n - 1);

    [Ke, fe] = element_matrices(ndim, h, E, nu);

    elem_dofs = build_elem_dof_table(n, ndim);
    nelems = size(elem_dofs, 1);
    dofs_per_elem = size(elem_dofs, 2);
    nnodes = prod(n);
    ndof = ndim * nnodes;

    [I_loc, J_loc] = ndgrid(1:dofs_per_elem, 1:dofs_per_elem);
    II = elem_dofs(:, I_loc(:));
    JJ = elem_dofs(:, J_loc(:));
    VV = repmat(Ke(:)', nelems, 1);
    K = sparse(II(:), JJ(:), VV(:), ndof, ndof);

    fe_mat = repmat(fe(:)', nelems, 1);
    b = accumarray(elem_dofs(:), fe_mat(:), [ndof, 1]);

    clamp_dofs = clamped_dofs(n, ndim);
    K(clamp_dofs, :) = 0;
    K(:, clamp_dofs) = 0;
    K = K + sparse(clamp_dofs, clamp_dofs, ones(numel(clamp_dofs), 1), ndof, ndof);
    b(clamp_dofs) = 0;
end

function [Ke, fe] = element_matrices(ndim, h, E, nu)
    npe = 2 ^ ndim;
    dpe = ndim * npe;
    D = isotropic_D(ndim, E, nu);

    gp = [-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)];
    gw = [1.0, 1.0];
    gp_idx = gauss_index_table(ndim);
    ngp = size(gp_idx, 1);

    body = body_force(ndim);

    Ke = zeros(dpe, dpe);
    fe = zeros(dpe, 1);
    detJ = prod(h / 2);
    Jinv_diag = 2.0 ./ h;

    for q = 1:ngp
        xi = gp(gp_idx(q, :));
        w = prod(gw(gp_idx(q, :))) * detJ;

        [N, dN_dxi] = q1_shape(ndim, xi);
        dN_dx = dN_dxi .* Jinv_diag;
        B = strain_displacement(dN_dx, ndim);

        Ke = Ke + (B' * D * B) * w;

        for a = 1:npe
            base = (a - 1) * ndim;
            fe(base + (1:ndim)) = fe(base + (1:ndim)) + N(a) * body(:) * w;
        end
    end
end

function D = isotropic_D(ndim, E, nu)
    if ndim == 1
        D = E;
    elseif ndim == 2
        c = E / ((1 + nu) * (1 - 2 * nu));
        D = c * [1 - nu, nu,     0;
                 nu,     1 - nu, 0;
                 0,      0,      (1 - 2 * nu) / 2];
    else
        lam = E * nu / ((1 + nu) * (1 - 2 * nu));
        G = E / (2 * (1 + nu));
        D = zeros(6, 6);
        D(1:3, 1:3) = lam * ones(3, 3) + 2 * G * eye(3);
        D(4:6, 4:6) = G * eye(3);
    end
end

function [N, dN_dxi] = q1_shape(ndim, xi)
    npe = 2 ^ ndim;
    sgn = node_signs(ndim);
    xi_row = xi(:)';
    N = prod((1 + sgn .* xi_row) / 2, 2)';
    dN_dxi = zeros(npe, ndim);
    for k = 1:ndim
        other = ones(npe, 1);
        for d = 1:ndim
            if d == k, continue; end
            other = other .* ((1 + sgn(:, d) * xi_row(d)) / 2);
        end
        dN_dxi(:, k) = (sgn(:, k) / 2) .* other;
    end
end

function B = strain_displacement(dN_dx, ndim)
    npe = size(dN_dx, 1);
    dpe = ndim * npe;
    if ndim == 1
        B = dN_dx(:)';
        return;
    elseif ndim == 2
        B = zeros(3, dpe);
        for a = 1:npe
            ix = (a - 1) * 2 + 1;
            iy = (a - 1) * 2 + 2;
            B(1, ix) = dN_dx(a, 1);
            B(2, iy) = dN_dx(a, 2);
            B(3, ix) = dN_dx(a, 2);
            B(3, iy) = dN_dx(a, 1);
        end
    else
        B = zeros(6, dpe);
        for a = 1:npe
            ix = (a - 1) * 3 + 1;
            iy = (a - 1) * 3 + 2;
            iz = (a - 1) * 3 + 3;
            B(1, ix) = dN_dx(a, 1);
            B(2, iy) = dN_dx(a, 2);
            B(3, iz) = dN_dx(a, 3);
            B(4, ix) = dN_dx(a, 2); B(4, iy) = dN_dx(a, 1);
            B(5, iy) = dN_dx(a, 3); B(5, iz) = dN_dx(a, 2);
            B(6, ix) = dN_dx(a, 3); B(6, iz) = dN_dx(a, 1);
        end
    end
end

function sgn = node_signs(ndim)
    npe = 2 ^ ndim;
    sgn = zeros(npe, ndim);
    for a = 1:npe
        bits = a - 1;
        for d = 1:ndim
            sgn(a, d) = 2 * bitand(bits, 1) - 1;
            bits = bitshift(bits, -1);
        end
    end
end

function idx = gauss_index_table(ndim)
    if ndim == 1
        idx = [1; 2];
    elseif ndim == 2
        [i1, i2] = ndgrid(1:2, 1:2);
        idx = [i1(:), i2(:)];
    else
        [i1, i2, i3] = ndgrid(1:2, 1:2, 1:2);
        idx = [i1(:), i2(:), i3(:)];
    end
end

function bf = body_force(ndim)
    if ndim == 1
        bf = -1.0;
    elseif ndim == 2
        bf = [0.0, -1.0];
    else
        %bf = [0.0, -1.0, 0.0];
        bf = [0.0, 0.0, -1.0];
    end
end

function elem_dofs = build_elem_dof_table(n, ndim)
    ne = n - 1;
    npe = 2 ^ ndim;
    if ndim == 1
        starts = (1:ne)';
        [oa] = (0:1)';
        offsets = oa;
    elseif ndim == 2
        [si, sj] = ndgrid(1:ne(1), 1:ne(2));
        starts = [si(:), sj(:)];
        [oa, ob] = ndgrid(0:1, 0:1);
        offsets = [oa(:), ob(:)];
    else
        [si, sj, sk] = ndgrid(1:ne(1), 1:ne(2), 1:ne(3));
        starts = [si(:), sj(:), sk(:)];
        [oa, ob, oc] = ndgrid(0:1, 0:1, 0:1);
        offsets = [oa(:), ob(:), oc(:)];
    end
    nelems = size(starts, 1);

    node_idx = zeros(nelems, npe);
    for a = 1:npe
        coord = starts + offsets(a, :);
        if ndim == 1
            node_idx(:, a) = coord(:, 1);
        elseif ndim == 2
            node_idx(:, a) = coord(:, 1) + (coord(:, 2) - 1) * n(1);
        else
            node_idx(:, a) = coord(:, 1) ...
                           + (coord(:, 2) - 1) * n(1) ...
                           + (coord(:, 3) - 1) * n(1) * n(2);
        end
    end

    elem_dofs = zeros(nelems, ndim * npe);
    for a = 1:npe
        base = (a - 1) * ndim;
        for d = 1:ndim
            elem_dofs(:, base + d) = (node_idx(:, a) - 1) * ndim + d;
        end
    end
end

function dofs = clamped_dofs(n, ndim)
    if ndim == 1
        nodes = 1;
    elseif ndim == 2
        nodes = 1 + (0:n(2) - 1) * n(1);
    else
        [j, k] = ndgrid(1:n(2), 1:n(3));
        nodes = 1 + (j(:) - 1) * n(1) + (k(:) - 1) * n(1) * n(2);
    end
    nodes = nodes(:);
    base = (nodes - 1) * ndim;
    dofs = reshape(base + (1:ndim), [], 1);
end
