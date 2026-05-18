function varargout = hypredrive_solve(A, b, options, varargin)
%HYPREDRIVE_SOLVE Solve a sparse linear system with hypredrive.
%
%   x = HYPREDRIVE_SOLVE(A, b)
%   [x, info] = HYPREDRIVE_SOLVE(A, b, options)
%   [x, info] = HYPREDRIVE_SOLVE(A, b, "solver", "pcg", "preconditioner", "amg", ...)
%
% A must be a real double sparse matrix and b must be a real double dense
% vector. options may be a YAML character vector/string, a struct, or name-value
% pairs accepted by HYPREDRIVE_OPTIONS.

if nargin < 2
    error('hypredrive:InvalidInput', 'usage: hypredrive_solve(A, b, options)');
end

if nargin < 3
    yaml = '';
elseif nargin == 3 && hypredrive_is_text(options)
    yaml = char(options);
elseif nargin == 3 && isstruct(options)
    yaml = hypredrive_options(options);
else
    if nargin < 3
        args = {};
    else
        args = [{options}, varargin];
    end
    yaml = hypredrive_options(args{:});
end

if nargout <= 1
    varargout{1} = hypredrive_mex(A, b, yaml);
elseif nargout == 2
    [varargout{1}, varargout{2}] = hypredrive_mex(A, b, yaml);
else
    error('hypredrive:InvalidOutput', 'hypredrive_solve returns at most x and info');
end
end

function tf = hypredrive_is_text(value)
tf = ischar(value) || (exist('isstring', 'builtin') && isstring(value));
end
