"""Helpers for translating Python-side options into hypredrive YAML.

We support three input shapes for solver/preconditioner configuration:

* a Python ``dict`` mirroring the YAML hierarchy
  (``{"general": {...}, "solver": {...}, ...}``);
* a YAML literal ``str``, passed through unchanged;
* a filesystem path (``str`` or ``pathlib.Path``) pointing to a YAML file.

In every case the binding ultimately hands a YAML *string* to
``HYPREDRV_InputArgsParse``: the C entry point already accepts a literal
YAML buffer in ``argv[0]``, so we sidestep temp files entirely.

We deliberately do not depend on PyYAML. The YAML hypredrive accepts is a
restricted subset (no anchors, no flow style, integer-aligned indents) and
hand-emitting the small subset the dict shape requires is both faster and
keeps the package wheel-friendly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Union

OptionsLike = Union[Mapping[str, Any], str, os.PathLike[str], None]


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


def _emit(node: Mapping[str, Any], indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for key, value in node.items():
        if not isinstance(key, str):
            raise TypeError(
                f"options keys must be strings, got {type(key).__name__}: {key!r}"
            )
        if isinstance(value, Mapping):
            lines.append(f"{pad}{key}:")
            _emit(value, indent + 1, lines)
        elif isinstance(value, (list, tuple)):
            # Hypredrive uses comma-separated scalars rather than YAML
            # sequences for fields like ``set_suffix``; mirror that.
            joined = ",".join(_format_scalar(item) for item in value)
            lines.append(f"{pad}{key}: {joined}")
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")


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


def normalize_options(options: OptionsLike) -> str:
    """Coerce ``options`` into a YAML *string* the C layer can consume.

    See module docstring for the accepted input shapes. ``None`` defaults
    to a minimal viable YAML that parses successfully but configures
    nothing beyond library defaults.
    """
    if options is None:
        return "general:\n  statistics: off\n"
    if isinstance(options, Mapping):
        return options_to_yaml(options)
    if isinstance(options, (str, os.PathLike)):
        text = os.fspath(options)
        # Heuristic: if it looks like a path that exists, read it; otherwise
        # treat the string as YAML literal. Both shapes are valid for the C
        # parser, but reading here lets us surface a clean Python-side
        # FileNotFoundError when the user clearly meant a path.
        if "\n" not in text and Path(text).is_file():
            return Path(text).read_text(encoding="utf-8")
        return text
    raise TypeError(
        "options must be a Mapping, str, PathLike, or None; "
        f"got {type(options).__name__}"
    )
