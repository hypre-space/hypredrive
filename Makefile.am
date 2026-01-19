bin_PROGRAMS = hypredrive laplacian
hypredrive_SOURCES = src/main.c
hypredrive_LDADD = libHYPREDRV.la

# Install the laplacian example binary
laplacian_SOURCES = examples/src/C_laplacian/laplacian.c
laplacian_LDADD = libHYPREDRV.la

lib_LTLIBRARIES = libHYPREDRV.la
libHYPREDRV_la_SOURCES = src/error.c src/info.c src/stats.c src/utils.c src/containers.c src/yaml.c\
                         src/field.c src/vector.c src/matrix.c src/linsys.c src/cheby.c src/amg.c\
                         src/fsai.c src/ilu.c src/mgr.c src/precon.c src/preset.c src/pcg.c \
	                 src/gmres.c src/fgmres.c src/bicgstab.c src/solver.c src/args.c \
                         src/HYPREDRV.c
libHYPREDRV_la_CFLAGS = $(AM_CPPFLAGS) -fvisibility=hidden
libHYPREDRV_la_LDFLAGS = -version-info 0:1:0

include_HEADERS = include/HYPREDRV.h
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/include

EXTRA_DIST = autogen.sh docs/Makefile docs/README.md docs/usrman-src docs/usrman-hypredrive.pdf docs/devman-hypredrive.pdf data examples
MOSTLYCLEANFILES =

@DX_RULES@
#MOSTLYCLEANFILES += $(DX_CLEANFILES)

docs: doxygen-doc
	@echo "Built developer's manual documentation (Doxygen)"
	@mv docs/hypredrive.pdf docs/devman-hypredrive.pdf
	@cd docs && make latexpdf
	@echo "Built user's manual documentation (Sphinx)"

check:
	@mpirun -np 1 ./hypredrive examples/ex1.yml > /dev/null 2>&1
	@if test $$? -eq 0 ; then \
		echo "Running ex1 with 1 MPI process... passed!"; \
	else \
		echo "Running ex1 with 1 MPI process... failed!"; \
		exit 1; \
	fi
	@mpirun -np 4 ./hypredrive examples/ex2.yml > /dev/null 2>&1
	@if test $$? -eq 0 ; then \
		echo "Running ex2 with 4 MPI processes... passed!"; \
	else \
		echo "Running ex2 with 4 MPI processes... failed!"; \
		exit 1; \
	fi
