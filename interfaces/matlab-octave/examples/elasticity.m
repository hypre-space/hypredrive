% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT
%
%ELASTICITY Solve a 1D / 2D / 3D linear-elasticity problem with hypredrive.
%
% Q1 small-strain isotropic elasticity (E = 1, nu = 0.3) on a unit-cube
% domain. The x = 0 face is clamped; a unit body force acts along the last
% active axis (gravity-style loading). The system is solved with PCG
% preconditioned by AMG.
%
% Run from a build tree containing the hypredrive MEX file:
%
%   addpath('/path/to/build');
%   elasticity                          % default 3D 16 x 16 x 16
%   elasticity(64)                      % 1D bar with 64 nodes
%   elasticity(32, 32)                  % 2D plate, 32 x 32 nodes
%   elasticity(16, 16, 8)               % 3D block, 16 x 16 x 8 nodes
%
% Optional VTI binary XML output (loadable in ParaView). A trailing char
% argument is interpreted as the output path, so the dimensions can stay
% as compact as in the calls above:
%
%   elasticity(16, 16, 16, 'block.vti') % 3D + VTI
%   elasticity(32, 32, 'plate.vti')     % 2D + VTI
%   elasticity(64, 'bar.vti')           % 1D + VTI
%
% Compatible with MATLAB and GNU Octave.

function elasticity(varargin)
    args = varargin;
    vti_path = '';
    if ~isempty(args) && ischar(args{end})
        vti_path = args{end};
        args = args(1:end - 1);
    end
    if isempty(args)
        nx = 16; ny = 16; nz = 16;
    else
        nx = args{1};
        if numel(args) >= 2, ny = args{2}; else, ny = 1; end
        if numel(args) >= 3, nz = args{3}; else, nz = 1; end
    end

    [A, b] = build_elasticity(nx, ny, nz);
    n = size(A, 1);

    dims = [nx, ny, nz];
    active = dims(dims > 1);
    parts = arrayfun(@(d) sprintf('%d', d), active, 'UniformOutput', false);
    grid_str = strjoin(parts, ' x ');
    ndim = numel(active);

    % Match the built-in elasticity_{2d,3d} preconditioner preset:
    % BoomerAMG with num_functions = ndim and strong_th = 0.8.
    amg = struct('max_iter', 1, 'tolerance', 0.0, 'print_level', 0, ...
                 'coarsening', struct('num_functions', ndim, 'strong_th', 0.8));
    opts = hypredrive_options( ...
        'solver', 'pcg', ...
        'preconditioner', 'amg', ...
        'pcg', struct('max_iter', 500, 'relative_tol', 1.0e-8), ...
        'amg', amg);

    [x, info] = hypredrive_solve(A, b, opts);

    axis_names = 'xyz';
    fprintf('grid:              %s (%dD)\n', grid_str, ndim);
    fprintf('dofs per node:     %d\n', ndim);
    fprintf('loading axis:      -%c\n', axis_names(ndim));
    fprintf('unknowns:          %d\n', n);
    fprintf('iterations:        %d\n', info.iterations);
    fprintf('setup time:        %.6e s\n', info.setup_time);
    fprintf('solve time:        %.6e s\n', info.solve_time);
    fprintf('solution l2 norm:  %.6e\n', info.solution_norm);
    relres = norm(b - A * x) / norm(b);
    fprintf('relative residual: %.6e\n', relres);
    if ~isfinite(relres) || relres > 1.0e-6
        error('elasticity:convergence', ...
              'relative residual %.3e exceeds 1e-6', relres);
    end

    if ~isempty(vti_path)
        write_displacement_vti(vti_path, dims, ndim, x);
        fprintf('vti output:        %s\n', vti_path);
    end
end

function write_displacement_vti(path, dims, ndim, x)
%WRITE_DISPLACEMENT_VTI Emit a VTK ImageData (.vti) file with raw binary
% appended data. Displacements are written as a single 3-component Float64
% vector field padded with zeros for inactive dimensions, so the file can
% be opened directly in ParaView regardless of problem dimensionality.

    h = ones(1, 3);
    for d = 1:3
        if dims(d) > 1
            h(d) = 1.0 / (dims(d) - 1);
        end
    end
    nnodes = prod(dims);

    disp_field = zeros(3, nnodes);
    for d = 1:ndim
        disp_field(d, :) = x(d:ndim:end)';
    end

    ext = [0, dims(1) - 1, 0, dims(2) - 1, 0, dims(3) - 1];
    ext_str = sprintf('%d %d %d %d %d %d', ext);

    fid = fopen(path, 'wb');
    if fid < 0
        error('elasticity:vti', 'could not open %s for writing', path);
    end

    try
        fprintf(fid, '<?xml version="1.0"?>\n');
        fprintf(fid, ['<VTKFile type="ImageData" version="0.1" ', ...
                      'byte_order="LittleEndian" header_type="UInt32">\n']);
        fprintf(fid, ['  <ImageData WholeExtent="%s" Origin="0 0 0" ', ...
                      'Spacing="%.17g %.17g %.17g">\n'], ext_str, h(1), h(2), h(3));
        fprintf(fid, '    <Piece Extent="%s">\n', ext_str);
        fprintf(fid, '      <PointData Vectors="displacement">\n');
        fprintf(fid, ['        <DataArray type="Float64" Name="displacement" ', ...
                      'NumberOfComponents="3" format="appended" offset="0"/>\n']);
        fprintf(fid, '      </PointData>\n');
        fprintf(fid, '    </Piece>\n');
        fprintf(fid, '  </ImageData>\n');
        fprintf(fid, '  <AppendedData encoding="raw">\n');
        fprintf(fid, '_');

        nbytes = uint32(3 * nnodes * 8);
        fwrite(fid, nbytes, 'uint32');
        fwrite(fid, disp_field(:), 'double');

        fprintf(fid, '\n  </AppendedData>\n');
        fprintf(fid, '</VTKFile>\n');
    catch err
        fclose(fid);
        rethrow(err);
    end
    fclose(fid);
end
