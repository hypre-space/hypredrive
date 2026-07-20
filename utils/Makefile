# Standalone utils: build without the main hypredrive library.
# (hypredrive-lsseq is built from the top-level CMake/Makefile.)

CC     = gcc
CFLAGS = -O3 -g -Wall

BINARIES = mat2ijbin

all: $(BINARIES)

mat2ijbin: mat2ijbin.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(BINARIES)

.PHONY: all clean
