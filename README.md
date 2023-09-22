# hypre_app
Driver application for solving linear systems read from file with hypre.

Currently, the driver only supports linear systems written in hypre's IJ conceptual interface format.

# Build instructions

```
 $ ./autogen.sh
 $ ./configure --prefix=${DEST_INSTALL_PATH} --with-hypre-include=${HYPRE_INSTALL_DIR}/include --with-hypre-lib=${HYPRE_INSTALL_DIR}/lib
 $ make all
 $ make install
```

Note: the first step `./autogen.sh` must be executed only once after cloning this repository.

# Example

```
 $ cd test/mgr-sysL3
 $ mpirun -np 1 ../../hypre_app mgr-sysL3.yml
```
