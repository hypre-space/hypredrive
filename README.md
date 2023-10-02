# hypre_app
High-level interface for solving linear systems with hypre, providing a user-friendly way to leverage its functionalities. Key features are:
1. **YAML Input**: Accepts configuration parameters written in the structured and human-readable YAML format.
2. **Intuitive Interface**: Offers a clear and concise API, encapsulating the functionalities of *hypre* to ensure user-friendly interactions.
3. **Prototyping**: Establishes a quick prototyping framework, facilitating the exploration of various solver/preconditioner setups.
4. **Testing**: Enables the construction of an integrated testing framework, accommodating problems applications built on hypre.

Currently, the driver only supports linear systems written in hypre's IJ conceptual interface format.

# Build instructions

```
 $ ./autogen.sh
 $ ./configure --with-hypre-include=${HYPRE_INSTALL_DIR}/include \
               --with-hypre-lib=${HYPRE_INSTALL_DIR}/lib
 $ make all
 $ make install
```

Note:
1. These instructions assume that [hypre](https://github.com/hypre-space/hypre) has been installed at `${HYPRE_INSTALL_DIR}`
2. The first step `./autogen.sh` must be executed only once after cloning this repository.
3. An installation prefix can be passed to `./configure` such as `--prefix=${INSTALL_PATH}`.  
   For more configure options, type `./configure -help`.

# Example

```
 $ cd test/mgr-sysL3
 $ mpirun -np 1 ../../hypre_app mgr-sysL3.yml
```
