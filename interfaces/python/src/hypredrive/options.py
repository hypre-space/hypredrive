"""Helpers for translating Python-side options into hypredrive YAML.

We support three input shapes for solver/preconditioner configuration:

* a Python ``dict`` mirroring the YAML hierarchy
  (``{"general": {...}, "solver": {...}, ...}``);
* a YAML literal ``str``, passed through unchanged;
* a filesystem path (``str`` or ``pathlib.Path``) pointing to a YAML file.

The binding passes inline YAML or a file path to ``HYPREDRV_InputArgsParse``.
The C layer reads files on rank zero, broadcasts them, and resolves includes
relative to the configuration directory. It only recognizes a path when it
has no spaces and a ``.yml``/``.yaml`` extension; any other file is read here
and passed as text, so its includes resolve against the working directory.

We deliberately do not depend on PyYAML. The YAML hypredrive accepts is a
restricted subset (no anchors, no flow style, integer-aligned indents) and
hand-emitting the small subset the dict shape requires is both faster and
keeps the package wheel-friendly.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Union

OptionsLike = Union[Mapping[str, Any], str, os.PathLike[str], None]


def configure(
    *,
    solver: str | None = None,
    preconditioner: str | None = None,
    general: Mapping[str, Any] | None = None,
    linear_system: Mapping[str, Any] | None = None,
    **method_options: Any,
) -> dict[str, Any]:
    """Build a hypredrive options dictionary from common method names.

    ``solver`` and ``preconditioner`` select the method blocks under the
    matching top-level YAML sections. ``general`` and ``linear_system`` are
    copied into their matching top-level sections. Additional keyword arguments
    are matched by method name, e.g. ``pcg={...}`` or ``amg={...}``.
    """
    options: dict[str, Any] = {}
    for section, values in (
        ("general", general),
        ("linear_system", linear_system),
    ):
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(
                f"{section} options must be a Mapping, got {type(values).__name__}"
            )
        options[section] = dict(values)
    for section, method in (
        ("solver", solver),
        ("preconditioner", preconditioner),
    ):
        if method is None:
            continue
        if not isinstance(method, str):
            raise TypeError(
                f"{section} must be a string or None, got {type(method).__name__}"
            )
        values = method_options.pop(method, {})
        if isinstance(values, Mapping):
            values = dict(values)
        elif isinstance(values, (list, tuple)):
            values = list(values)
        else:
            raise TypeError(
                f"{method} options must be a Mapping, list, or tuple; "
                f"got {type(values).__name__}"
            )
        options[section] = {method: values}
    if method_options:
        names = ", ".join(sorted(method_options))
        raise ValueError(f"unused method option block(s): {names}")
    return options


def _format_scalar(value: Any) -> str:
    """Render a scalar in a form hypredrive's YAML parser accepts.

    Booleans become ``on``/``off`` because that is what the CLI surfaces
    for the existing toggle fields (``general.statistics``,
    ``general.use_millisec``, etc.). Numbers and strings are printed
    verbatim; the parser is tolerant of unquoted alphanumerics and
    integers/floats, which is the entire space of values that show up
    inside the documented option set.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None:
        return "''"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if not text:
        return "''"
    # Quote only if there's something the parser would mistake for syntax.
    if any(ch in text for ch in ":#'\"\n"):
        # Hypredrive's YAML parser treats single-quoted strings as literal.
        escaped = text.replace("'", "''")
        return f"'{escaped}'"
    return text


def _emit_value(
    prefix: str,
    key: str,
    value: Any,
    nested_indent: int,
    lines: list[str],
) -> None:
    if isinstance(value, Mapping):
        lines.append(f"{prefix}:")
        _emit(value, nested_indent, lines)
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"option sequence for {key!r} must not be empty")
        if value and all(isinstance(item, Mapping) for item in value):
            lines.append(f"{prefix}:")
            _emit_mapping_sequence(value, nested_indent, lines)
        else:
            joined = ",".join(_format_scalar(item) for item in value)
            lines.append(f"{prefix}: {joined}")
    else:
        lines.append(f"{prefix}: {_format_scalar(value)}")


def _emit_mapping_sequence(
    items: list[Any] | tuple[Any, ...],
    indent: int,
    lines: list[str],
) -> None:
    item_pad = "  " * indent
    child_pad = "  " * (indent + 1)
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("mapping sequence items must be mappings")
        if not item:
            raise ValueError("mapping sequence items must not be empty")
        first = True
        for key, value in item.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"options keys must be strings, got {type(key).__name__}: {key!r}"
                )
            if first:
                prefix = f"{item_pad}- {key}"
                first = False
            else:
                prefix = f"{child_pad}{key}"
            _emit_value(prefix, key, value, indent + 2, lines)


def _emit(node: Mapping[str, Any], indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for key, value in node.items():
        if not isinstance(key, str):
            raise TypeError(
                f"options keys must be strings, got {type(key).__name__}: {key!r}"
            )
        _emit_value(f"{pad}{key}", key, value, indent + 1, lines)


def options_to_yaml(options: Mapping[str, Any]) -> str:
    """Render a nested options ``dict`` as a hypredrive-flavored YAML string."""
    if not isinstance(options, Mapping):
        raise TypeError(
            f"options must be a Mapping, got {type(options).__name__}"
        )
    lines: list[str] = []
    _emit(options, 0, lines)
    if not lines:
        # The C parser tolerates an empty document, but produce a stub for
        # clarity when debugging captured YAML.
        return "general:\n"
    return "\n".join(lines) + "\n"


def _c_reads_config_path(path: str) -> bool:
    """Whether ``HYPREDRV_InputArgsParse`` reads ``path`` as a file.

    Mirrors ``hypredrv_IsYAMLFilename`` (no spaces, ``.yml``/``.yaml`` in any
    case) and the C rejection of resolved paths containing ``..``.
    """
    ext = os.path.splitext(path)[1].lower()
    return (
        " " not in path
        and ext in (".yml", ".yaml")
        and ".." not in os.path.realpath(path)
    )


def _config_file_option(path: str) -> str:
    """Return ``path`` when the C layer reads it, else the file's text."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if _c_reads_config_path(path):
        return path
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def normalize_options(options: OptionsLike) -> str:
    """Coerce ``options`` into inline YAML or a file path for the C layer.

    See module docstring for the accepted input shapes. ``None`` defaults
    to a minimal viable YAML that parses successfully but configures
    nothing beyond library defaults.
    """
    if options is None:
        return "general:\n  statistics: off\n"
    if isinstance(options, Mapping):
        return options_to_yaml(options)
    if isinstance(options, os.PathLike):
        return _config_file_option(os.fspath(options))
    if isinstance(options, str):
        if "\n" not in options:
            if os.path.isfile(options):
                return _config_file_option(options)
            if os.path.splitext(options)[1] or "/" in options or "\\" in options:
                raise FileNotFoundError(options)
        return options
    raise TypeError(
        "options must be a Mapping, str, PathLike, or None; "
        f"got {type(options).__name__}"
    )
