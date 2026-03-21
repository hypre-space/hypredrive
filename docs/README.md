# hypredrive documentation

This folder contains the sources for hypredrive's Sphinx user manual plus generated PDF
artifacts when documentation is built.

## Building docs with CMake

The supported path is the top-level CMake build:

```bash
cmake -S . -B build-docs -DHYPREDRV_ENABLE_DOCS=ON
cmake --build build-docs --target docs
```

Useful targets:

- `docs`: build Doxygen first, then Sphinx
- `sphinx-doc`: build the HTML user manual
- `sphinx-latexpdf`: build the PDF user manual when LaTeX is available

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
