bin_PROGRAMS = hypredrive-cli laplacian hypredrive-lsseq
hypredrive_cli_SOURCES = src/internal/main.c
hypredrive_cli_CPPFLAGS = $(AM_CPPFLAGS) -DHYPREDRV_STATIC_DEFINE
hypredrive_cli_LDADD = libHYPREDRV_internal.la

# Install the laplacian example binary
laplacian_SOURCES = examples/src/C_laplacian/laplacian.c
laplacian_LDADD = libHYPREDRV.la

hypredrive_lsseq_SOURCES = utils/lsseq_driver.c
hypredrive_lsseq_CPPFLAGS = $(AM_CPPFLAGS) -DHYPREDRV_STATIC_DEFINE
hypredrive_lsseq_LDADD = libHYPREDRV_internal.la

noinst_LTLIBRARIES = libHYPREDRV_internal.la
lib_LTLIBRARIES = libHYPREDRV.la
libHYPREDRV_internal_la_SOURCES = src/internal/error.c src/internal/eigspec.c src/internal/info.c src/internal/stats.c src/internal/utils.c src/internal/containers.c src/internal/yaml.c\
                         src/internal/field.c src/internal/vector.c src/internal/matrix.c src/internal/linsys.c src/internal/linsys_print.c src/internal/cheby.c src/internal/amg.c\
                         src/internal/ams.c src/internal/ads.c src/internal/fsai.c src/internal/ilu.c src/internal/schwarz.c src/internal/mgr.c src/internal/precon.c \
			 src/internal/precon_reuse.c src/internal/presets.c src/internal/krylov.c src/internal/pcg.c src/internal/gmres.c src/internal/fgmres.c \
			 src/internal/bicgstab.c src/internal/solver.c src/internal/args.c src/internal/comp.c src/internal/lsseq.c src/internal/scaling.c src/internal/runtime.c \
			 src/internal/logging.c src/internal/help.c src/HYPREDRV.c
AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir)/include -I$(top_srcdir)/include/internal -I$(top_srcdir)/src/internal
libHYPREDRV_internal_la_CFLAGS = $(AM_CPPFLAGS) -fvisibility=hidden
if HYPREDRV_BUILD_SHARED
libHYPREDRV_internal_la_CFLAGS += -DHYPREDRV_EXPORTS
else
AM_CPPFLAGS += -DHYPREDRV_STATIC_DEFINE
endif
libHYPREDRV_la_SOURCES =
libHYPREDRV_la_LIBADD = libHYPREDRV_internal.la
libHYPREDRV_la_LDFLAGS = -version-info 0:1:0 \
	-export-symbols-regex '^HYPREDRV_.*$$'

include_HEADERS = include/HYPREDRV.h include/HYPREDRV_utils.h
nodist_include_HEADERS = HYPREDRV_config.h
INTERNAL_HEADER_FILES = include/internal/amg.h include/internal/args.h \
			include/internal/ads.h include/internal/ams.h \
			include/internal/bicgstab.h include/internal/cheby.h \
			include/internal/compatibility.h include/internal/comp.h \
			include/internal/containers.h include/internal/eigspec.h \
			include/internal/error.h include/internal/fgmres.h \
			include/internal/field.h include/internal/fsai.h \
			include/internal/gen_macros.h include/internal/gmres.h \
			include/internal/ilu.h include/internal/info.h \
			include/internal/help.h \
			include/internal/krylov.h include/internal/linsys.h \
			include/internal/lsseq.h include/internal/mgr.h \
			include/internal/pcg.h include/internal/precon.h \
			include/internal/precon_reuse.h include/internal/presets.h \
			include/internal/scaling.h include/internal/schwarz.h \
			include/internal/solver.h include/internal/stats.h \
			include/internal/utils.h include/internal/yaml.h
noinst_HEADERS = src/internal/object.h src/internal/runtime.h src/internal/logging.h $(INTERNAL_HEADER_FILES)
ACLOCAL_AMFLAGS = -I m4

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
