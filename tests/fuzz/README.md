# Hypredrive Fuzzing

Fuzzing is opt-in through `HYPREDRV_ENABLE_FUZZING`. The default engine is
`replay`, which registers deterministic CTest tests under the `fuzz-replay`
label. Detailed workflow notes live in
`docs/usrman-src/developer_notes.rst`.

```bash
cmake -S . -B build-fuzz -G Ninja \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DHYPREDRV_ENABLE_FUZZING=ON \
      -DHYPREDRV_ENABLE_TESTING=ON
cmake --build build-fuzz -j
ctest --test-dir build-fuzz -L fuzz-replay --output-on-failure
```

Live fuzzing uses the same targets with another engine:

```bash
cmake -S . -B build-fuzz -G Ninja \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DHYPREDRV_ENABLE_FUZZING=ON \
      -DHYPREDRV_FUZZ_ENGINE=libfuzzer
cmake --build build-fuzz --target hypredrv-fuzz-parse -j
tests/fuzz/fuzzing.sh parse 300 libfuzzer
```

Supported modes are `parse`, `solve`, `lsseq`, `matrix`, and `vector`.
Place minimized fixed-bug inputs in `tests/fuzz/regressions/<mode>/`.
Parse seeds come from `examples/*.yml`; keep checked-in examples replay-clean.
