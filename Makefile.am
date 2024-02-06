bin_PROGRAMS = hypredrive
hypredrive_SOURCES = src/error.c src/info.c src/stats.c src/utils.c src/containers.c src/yaml.c src/field.c src/linsys.c src/cheby.c src/amg.c src/fsai.c src/ilu.c src/mgr.c src/precon.c src/pcg.c src/gmres.c src/fgmres.c src/bicgstab.c src/solver.c src/args.c src/hypredrive.c src/main.c
AM_CPPFLAGS = -I$(top_srcdir)/include

EXTRA_DIST =
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
