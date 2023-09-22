# hypre_app
Driver application for solving linear systems read from file with hypre.

Currently, the driver only supports linear systems written in hypre's IJ conceptual interface format.

# Build instructions

```
 $ ./autogen.sh
 $ ./configure --with-hypre-include=${HYPRE_INSTALL_DIR}/include --with-hypre-lib=${HYPRE_INSTALL_DIR}/lib
 $ make all
 $ make install
```

Note:
1. The first step `./autogen.sh` must be executed only once after cloning this repository.
2. An installation prefix can be passed to `./configure` such as `--prefix=${INSTALL_PATH}`.  
   For more configure options, type `./configure -help`.

# Example

```
 $ cd test/mgr-sysL3
 $ mpirun -np 1 ../../hypre_app mgr-sysL3.yml
```
