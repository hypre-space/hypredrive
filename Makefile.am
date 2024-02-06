bin_PROGRAMS = hypredrive
hypredrive_SOURCES = src/error.c src/info.c src/stats.c src/utils.c src/containers.c src/yaml.c src/field.c src/linsys.c src/cheby.c src/amg.c src/fsai.c src/ilu.c src/mgr.c src/precon.c src/pcg.c src/gmres.c src/fgmres.c src/bicgstab.c src/solver.c src/args.c src/hypredrive.c src/main.c
include_HEADERS = include/amg.h include/args.h include/bicgstab.h include/cheby.h include/containers.h include/error.h include/fgmres.h include/field.h include/fsai.h include/gen_macros.h include/gmres.h include/hypredrive.h include/ilu.h include/info.h include/linsys.h include/mgr.h include/pcg.h include/precon.h include/solver.h include/stats.h include/utils.h include/yaml.h
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
		echo "Running with 1 MPI process... passed!"; \
	else \
		echo "Running with 1 MPI process... failed!"; \
		exit 1; \
	fi
	@mpirun -np 4 ./hypredrive examples/ex2.yml > /dev/null 2>&1
	@if test $$? -eq 0 ; then \
		echo "Running with 4 MPI process... passed!"; \
	else \
		echo "Running with 4 MPI process... failed!"; \
		exit 1; \
	fi
