# hypre-ij-app
Driver application for solving linear systems with hypre through the IJ interface

# Build instructions

```
 $ ./autogen.sh
 $ ./configure --prefix=${DEST_INSTALL_PATH} --with-hypre-include=${HYPRE_INSTALL_DIR}/include --with-hypre-lib=${HYPRE_INSTALL_DIR}/lib
 $ make all
 $ make install
```

# Example

```
 $ cd test/mgr-sysL3
 $ mpirun -np 1 ../../hypre_app mgr-sysL3.yml
```
