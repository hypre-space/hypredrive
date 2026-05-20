% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT

function tf = hypredrive_is_text(value)
%HYPREDRIVE_IS_TEXT Return true for MATLAB/Octave text values.

tf = ischar(value) || isa(value, 'string');
end
