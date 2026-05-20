% Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
% HYPRE Project Developers. See the top-level COPYRIGHT file for details.
%
% SPDX-License-Identifier: MIT

function yaml = hypredrive_options(varargin)
%HYPREDRIVE_OPTIONS Build hypredrive YAML from MATLAB/Octave data.
%
%   yaml = HYPREDRIVE_OPTIONS(opts)
%   yaml = HYPREDRIVE_OPTIONS('solver', 'pcg', 'preconditioner', 'amg', ...)
%
% Struct form mirrors hypredrive YAML:
%
%   opts.solver.pcg.max_iter = 200;
%   opts.solver.pcg.relative_tol = 1.0e-10;
%   opts.preconditioner.amg.print_level = 0;
%   yaml = hypredrive_options(opts);
%
% Name-value form is convenient for common cases:
%
%   yaml = hypredrive_options( ...
%       'solver', 'pcg', ...
%       'preconditioner', 'amg', ...
%       'pcg', struct('max_iter', 200, 'relative_tol', 1.0e-10), ...
%       'amg', struct('print_level', 0));

if nargin == 0
    opts = struct();
elseif nargin == 1 && isstruct(varargin{1})
    opts = varargin{1};
else
    opts = parse_name_value(varargin{:});
end
opts = apply_defaults(opts);

lines = emit_struct(opts, 0);
if isempty(lines)
    yaml = '';
else
    yaml = sprintf('%s\n', lines{:});
end
end

function opts = parse_name_value(varargin)
if mod(nargin, 2) ~= 0
    error('hypredrive:InvalidOptions', 'options must be name-value pairs');
end

opts = struct();
method_options = struct();
solver_method = '';
precon_method = '';

for i = 1:2:nargin
    key = char(varargin{i});
    value = varargin{i + 1};
    switch key
        case {'solver', 'preconditioner'}
            if ~hypredrive_is_text(value)
                error('hypredrive:InvalidOptions', '%s must be a string', key);
            end
            method = char(value);
            validate_method(key, method);
            if strcmp(key, 'solver')
                solver_method = method;
            else
                precon_method = method;
            end
        case {'general', 'linear_system'}
            if ~isstruct(value)
                error('hypredrive:InvalidOptions', '%s must be a struct', key);
            end
            opts.(key) = value;
        otherwise
            validate_method_block(key);
            if ~isstruct(value)
                error('hypredrive:InvalidOptions', '%s options must be a struct', key);
            end
            method_options.(key) = value;
    end
end

if ~isempty(solver_method)
    opts.solver.(solver_method) = method_block(method_options, solver_method);
end
if ~isempty(precon_method)
    if ~strcmp(precon_method, 'none')
        opts.preconditioner.(precon_method) = method_block(method_options, precon_method);
    end
end
end

function opts = apply_defaults(opts)
if ~isfield(opts, 'general')
    opts.general = struct();
end
if ~isfield(opts.general, 'statistics')
    % MATLAB/Octave calls are quiet by default. Set general.statistics
    % explicitly if the HYPREDRV statistics table should be printed.
    opts.general.statistics = 0;
end
end

function lines = emit_struct(s, indent)
names = fieldnames(s);
lines = {};
for i = 1:numel(names)
    key = names{i};
    value = s.(key);
    prefix = [repmat('  ', 1, indent), key];
    if isstruct(value)
        if numel(value) ~= 1
            error('hypredrive:InvalidOptions', 'struct arrays are not supported');
        end
        child = emit_struct(value, indent + 1);
        if isempty(child)
            lines{end + 1} = [prefix, ': {}']; %#ok<AGROW>
        else
            lines{end + 1} = [prefix, ':']; %#ok<AGROW>
            lines = [lines, child]; %#ok<AGROW>
        end
    else
        lines{end + 1} = [prefix, ': ', format_value(value)]; %#ok<AGROW>
    end
end
end

function text = format_value(value)
if islogical(value)
    if isscalar(value)
        if value
            text = 'true';
        else
            text = 'false';
        end
    else
        text = join_values(value);
    end
elseif isnumeric(value)
    text = join_values(value);
elseif hypredrive_is_text(value)
    text = quote_if_needed(char(value));
else
    error('hypredrive:InvalidOptions', 'unsupported option value type');
end
end

function text = join_values(values)
flat = values(:).';
parts = cell(1, numel(flat));
for i = 1:numel(flat)
    if islogical(flat(i))
        if flat(i)
            parts{i} = 'true';
        else
            parts{i} = 'false';
        end
    elseif isnumeric(flat(i))
        if ~isfinite(flat(i))
            error('hypredrive:InvalidOptions', 'Inf and NaN numeric options are not supported');
        end
        if abs(flat(i)) <= 2^53 && flat(i) == fix(flat(i))
            parts{i} = sprintf('%.0f', flat(i));
        else
            parts{i} = sprintf('%.17g', flat(i));
        end
    else
        error('hypredrive:InvalidOptions', 'unsupported option value type');
    end
end
text = strjoin(parts, ',');
end

function tf = needs_quotes(text)
first = text(1);
flow_indicators = '[]{},&*#?|-<>=!%@`:';
reserved = {'null', '~', 'yes', 'no', 'on', 'off', 'true', 'false'};
numeric_pattern = '^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$';
tf = isspace(first) || any(first == flow_indicators) || any(text == ':') || ...
      any(text == '#') || any(text == '''') || any(text == '"') || ...
      any(text == ',') || any(text == sprintf('\n')) || ...
      any(strcmpi(text, reserved)) || ~isempty(regexp(text, numeric_pattern, 'once'));
end

function text = quote_if_needed(text)
if isempty(text)
    text = '''''';
    return;
end
if needs_quotes(text)
    text = ['''', strrep(text, '''', ''''''), ''''];
end
end

function block = method_block(method_options, method)
if isfield(method_options, method)
    block = method_options.(method);
else
    block = struct();
end
end

function validate_method(kind, method)
switch kind
    case 'solver'
        allowed = {'pcg', 'gmres', 'fgmres', 'bicgstab'};
    otherwise
        allowed = {'amg', 'mgr', 'ilu', 'fsai', 'none'};
end
if ~any(strcmp(method, allowed))
    error('hypredrive:InvalidOptions', 'unsupported %s method: %s', kind, method);
end
end

function validate_method_block(method)
allowed = {'pcg', 'gmres', 'fgmres', 'bicgstab', 'amg', 'mgr', 'ilu', 'fsai'};
if ~any(strcmp(method, allowed))
    error('hypredrive:InvalidOptions', 'unsupported method options block: %s', method);
end
end
