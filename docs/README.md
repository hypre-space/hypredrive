# hypredrive documentation

This folder contains the sources for hypredrive's Sphinx user manual and the
Doxygen API reference. Generated documentation is written to the build tree.

## Building docs with CMake

Install Doxygen and the Python packages in `usrman-src/requirements.txt` first.
The HTML build does not require LaTeX. Use the top-level CMake build:

```bash
cmake -S . -B build-docs -DHYPREDRV_ENABLE_DOCS=ON
cmake --build build-docs --target sphinx-doc
```

Useful targets:

- `sphinx-doc`: build Doxygen XML first, then the HTML user manual
- `sphinx-latexpdf`: build Doxygen XML first, then the PDF user manual (requires LaTeX)
- `docs`: build the Doxygen reference and PDF user manual (requires LaTeX)

The HTML output is written under `build-docs/docs/usrman-build/html/index.html`.

## Building the user's manual in HTML format

The HTML version of the documentation provides a better interactivity and navigation
experience than the PDF version.

For a direct Sphinx-only workflow from this folder, install the packages listed in
`usrman-src/requirements.txt`, then run:

```bash
make html
```

The output will be written to `usrman-build/html/index.html`.

If you need the API reference page populated, build docs through the CMake path above so
that Doxygen XML is generated first.
