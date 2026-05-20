% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT

function hypredrive_setup()
%HYPREDRIVE_SETUP Add an installed hypredrive MATLAB/Octave interface to path.
%
%   run('/path/to/install/lib/matlab/hypredrive_setup.m')
%   run('/path/to/install/lib/octave/hypredrive_setup.m')

root = fileparts(mfilename('fullpath'));
if ~isempty(root)
    addpath(root);
end
end
