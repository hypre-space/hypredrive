# hypredrive Rust interface

The `hypredrive` Cargo crate solves distributed sparse systems from rank-local
CSR data and returns owned solution values and solver statistics.

## Quick start

Use Rust 1.82 or newer, Cargo, CMake, libclang, and an MPI C compiler. The crate
requires shared hypredrive with real, double-precision HYPRE.

From the repository root, build the Rust tests in a dedicated build directory.

```bash
rust_build_dir=$(mktemp -d /tmp/hypredrive-rust-build-XXXXXXXX)
cmake -S . -B "$rust_build_dir" \
  -DBUILD_SHARED_LIBS=ON \
  -DHYPREDRV_ENABLE_RUST=ON \
  -DHYPREDRV_ENABLE_TESTING=ON \
  -DHYPREDRV_DATASETS= \
  -DCMAKE_C_COMPILER=mpicc
cmake --build "$rust_build_dir" --target rust-test --parallel
```

## Minimal example

[`minimal.rs`](examples/minimal.rs) solves a two-row system with solution
`[1, 1]`. [`laplacian.rs`](examples/laplacian.rs) builds each rank's share of a
distributed system and solves for two right-hand sides.

See the [Rust user manual](../../docs/usrman-src/interfaces_rust.rst) for
running the examples, using an installed library from Cargo, and the API and
MPI contracts.
