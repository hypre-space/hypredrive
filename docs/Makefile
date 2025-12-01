# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = usrman-src
BUILDDIR      = usrman-build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# Use -W to treat warnings as errors (e.g., missing images will fail the build)
%: Makefile
	@$(SPHINXBUILD) -W -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ "$@" = "latexpdf" ] && [ -f "$(BUILDDIR)/latex/hypredrive.pdf" ]; then \
		cp "$(BUILDDIR)/latex/hypredrive.pdf" usrman-hypredrive.pdf; \
	        echo "PDF copied to usrman-hypredrive.pdf"; \
	fi
