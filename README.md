# hypre-ij-app
Driver application for solving linear systems with hypre through the IJ interface

# Build instructions

Run:
```
 $ ./configure --prefix=${DEST_INSTALL_PATH} --with-hypre-include=${HYPRE_INSTALL_DIR}/include --with-hypre-lib=${HYPRE_INSTALL_DIR}/lib
 $ make all
 $ make install
```
