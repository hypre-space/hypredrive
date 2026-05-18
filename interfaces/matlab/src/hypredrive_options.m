function yaml = hypredrive_options(varargin)
%HYPREDRIVE_OPTIONS Build hypredrive YAML from MATLAB/Octave data.
%
%   yaml = HYPREDRIVE_OPTIONS(opts)
%   yaml = HYPREDRIVE_OPTIONS("solver", "pcg", "preconditioner", "amg", ...)
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
%       "solver", "pcg", ...
%       "preconditioner", "amg", ...
%       "pcg", struct("max_iter", 200, "relative_tol", 1.0e-10), ...
%       "amg", struct("print_level", 0));

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

for i = 1:2:nargin
    key = char(varargin{i});
    value = varargin{i + 1};
    switch key
        case {'solver', 'preconditioner'}
            if ~hypredrive_is_text(value)
                error('hypredrive:InvalidOptions', '%s must be a string', key);
            end
            method = char(value);
            if isfield(method_options, method)
                block = method_options.(method);
            else
                block = struct();
            end
            opts.(key).(method) = block;
        case {'general', 'linear_system'}
            if ~isstruct(value)
                error('hypredrive:InvalidOptions', '%s must be a struct', key);
            end
            opts.(key) = value;
        otherwise
            if ~isstruct(value)
                error('hypredrive:InvalidOptions', '%s options must be a struct', key);
            end
            method_options.(key) = value;
            if isfield(opts, 'solver') && isfield(opts.solver, key)
                opts.solver.(key) = value;
            end
            if isfield(opts, 'preconditioner') && isfield(opts.preconditioner, key)
                opts.preconditioner.(key) = value;
            end
    end
end
end

function opts = apply_defaults(opts)
if ~isfield(opts, 'general')
    opts.general = struct();
end
if ~isfield(opts.general, 'statistics')
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
        lines{end + 1} = [prefix, ':']; %#ok<AGROW>
        child = emit_struct(value, indent + 1);
        lines = [lines, child]; %#ok<AGROW>
    else
        lines{end + 1} = [prefix, ': ', format_value(value)]; %#ok<AGROW>
    end
end
end

function text = format_value(value)
if islogical(value)
    if isscalar(value)
        if value
            text = 'on';
        else
            text = 'off';
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

function tf = hypredrive_is_text(value)
tf = ischar(value) || (exist('isstring', 'builtin') && isstring(value));
end

function text = join_values(values)
flat = values(:).';
parts = cell(1, numel(flat));
for i = 1:numel(flat)
    if islogical(flat(i))
        if flat(i)
            parts{i} = 'on';
        else
            parts{i} = 'off';
        end
    else
        parts{i} = sprintf('%.17g', flat(i));
    end
end
text = strjoin(parts, ',');
end

function text = quote_if_needed(text)
if isempty(text)
    text = '''''';
    return;
end
if any(text == ':') || any(text == '#') || any(text == '''') || any(text == '"') || ...
      any(text == sprintf('\n'))
    text = ['''', strrep(text, '''', ''''''), ''''];
end
end
