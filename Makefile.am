bin_PROGRAMS = hypredrive-cli laplacian hypredrive-lsseq
hypredrive_cli_SOURCES = src/main.c
hypredrive_cli_LDADD = libHYPREDRV.la

# Install the laplacian example binary
laplacian_SOURCES = examples/src/C_laplacian/laplacian.c
laplacian_LDADD = libHYPREDRV.la

hypredrive_lsseq_SOURCES = utils/lsseq_driver.c
hypredrive_lsseq_LDADD = libHYPREDRV.la

lib_LTLIBRARIES = libHYPREDRV.la
libHYPREDRV_la_SOURCES = src/error.c src/eigspec.c src/info.c src/stats.c src/utils.c src/containers.c src/yaml.c\
                         src/field.c src/vector.c src/matrix.c src/linsys.c src/linsys_print.c src/cheby.c src/amg.c\
                         src/fsai.c src/ilu.c src/mgr.c src/precon.c src/presets.c src/krylov.c\
			 src/pcg.c src/gmres.c src/fgmres.c src/bicgstab.c src/solver.c src/args.c \
                         src/comp.c src/lsseq.c src/scaling.c src/runtime.c src/logging.c src/HYPREDRV.c
libHYPREDRV_la_CFLAGS = $(AM_CPPFLAGS) -fvisibility=hidden
libHYPREDRV_la_LDFLAGS = -version-info 0:1:0

include_HEADERS = include/HYPREDRV.h include/HYPREDRV_utils.h
nodist_include_HEADERS = HYPREDRV_config.h
INTERNAL_HEADER_FILES = include/internal/amg.h include/internal/args.h \
			include/internal/bicgstab.h include/internal/cheby.h \
			include/internal/compatibility.h include/internal/comp.h \
			include/internal/containers.h include/internal/eigspec.h \
			include/internal/error.h include/internal/fgmres.h \
			include/internal/field.h include/internal/fsai.h \
			include/internal/gen_macros.h include/internal/gmres.h \
			include/internal/ilu.h include/internal/info.h \
			include/internal/krylov.h include/internal/linsys.h \
			include/internal/lsseq.h include/internal/mgr.h \
			include/internal/pcg.h include/internal/precon.h \
			include/internal/presets.h include/internal/scaling.h \
			include/internal/solver.h include/internal/stats.h \
			include/internal/utils.h include/internal/yaml.h
noinst_HEADERS = src/object.h src/runtime.h src/logging.h $(INTERNAL_HEADER_FILES)
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)/include -I$(top_srcdir)/include/internal

EXTRA_DIST = autogen.sh docs/Makefile docs/README.md docs/usrman-src docs/usrman-hypredrive.pdf docs/devman-hypredrive.pdf data examples utils
MOSTLYCLEANFILES =

@DX_RULES@
#MOSTLYCLEANFILES += $(DX_CLEANFILES)

docs: doxygen-doc
	@echo "Built developer's manual documentation (Doxygen)"
	@mv docs/hypredrive.pdf docs/devman-hypredrive.pdf
	@cd docs && make latexpdf
	@echo "Built user's manual documentation (Sphinx)"

check:
	@mpirun -np 1 ./hypredrive-cli examples/ex1.yml > /dev/null 2>&1
	@if test $$? -eq 0 ; then \
		echo "Running ex1 with 1 MPI process... passed!"; \
	else \
		echo "Running ex1 with 1 MPI process... failed!"; \
		exit 1; \
	fi
	@mpirun -np 4 ./hypredrive-cli examples/ex2.yml > /dev/null 2>&1
	@if test $$? -eq 0 ; then \
		echo "Running ex2 with 4 MPI processes... passed!"; \
	else \
		echo "Running ex2 with 4 MPI processes... failed!"; \
		exit 1; \
	fi
