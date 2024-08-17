# Compiler and Flags
CC = gcc
CFLAGS = -O3 -g

# Binaries to be generated
BINARIES = lossless_comp mat2ijbin

# Default target: build all binaries
all: $(BINARIES)

# Rule for lossless_comp
lossless_comp: lossless_comp.c
	$(CC) $(CFLAGS) $< -o $@ -llz4 -lzstd -lz -lblosc

# Rule for mat2ijbin
mat2ijbin: mat2ijbin.c
	$(CC) $(CFLAGS) $< -o $@

# Clean up generated files
clean:
	rm -f $(BINARIES)

# Marking targets that are not files
.PHONY: all clean
