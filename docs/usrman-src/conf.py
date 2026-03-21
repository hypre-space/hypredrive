# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
from pathlib import Path


# -- Project information -----------------------------------------------------

project = u'hypredrive'
copyright = u'2024 Lawrence Livermore National Security, LLC and other HYPRE Project Developers. See the top-level COPYRIGHT file for details.'
author = 'Victor A. P. Magri'

# The full version, including alpha/beta/rc tags
release = os.environ.get('HYPREDRV_DOCS_RELEASE', '0.1')
version = os.environ.get('HYPREDRV_DOCS_VERSION', release)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['breathe', 'sphinx_copybutton']
c_id_attributes = ['HYPREDRV_EXPORT_SYMBOL']
cpp_id_attributes = ['HYPREDRV_EXPORT_SYMBOL']

latex_elements = {
    'preamble': r'''
\DeclareUnicodeCharacter{0394}{$\Delta$}
\DeclareUnicodeCharacter{00D7}{$\times$}
\DeclareUnicodeCharacter{2264}{$\le$}
\DeclareUnicodeCharacter{2265}{$\ge$}
''',
}

# Configure sphinx_copybutton
copybutton_prompt_text = r"\$ |>>> |\.\.\. "  # Regex for various prompts
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if os.path.exists('_static'):
    html_static_path = ['_static']
    # Ensure Sphinx includes custom JavaScript files
    html_js_files = ['remove-dollar.js']
else:
    html_static_path = []
    html_js_files = []

_docs_src_dir = Path(__file__).resolve().parent
_breathe_xml = os.environ.get("HYPREDRV_DOXYGEN_XML")
if not _breathe_xml:
    for candidate in (_docs_src_dir.parent / "xml", Path.cwd() / "docs" / "xml"):
        if (candidate / "index.xml").exists():
            _breathe_xml = str(candidate)
            break
if not _breathe_xml:
    _breathe_xml = str((_docs_src_dir.parent / "xml").resolve())

breathe_projects = {"hypredrive": _breathe_xml}
breathe_default_project = "hypredrive"
html_theme = "sphinx_rtd_theme"
