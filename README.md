# hypredrive

High-level interface for solving linear systems with hypre, providing a user-friendly way to leverage its functionalities. Key features are:

1. **YAML Input**: Accepts configuration parameters written in the structured and human-readable YAML format.
2. **Intuitive Interface**: Offers a clear and concise API, encapsulating the functionalities of *hypre* to ensure user-friendly interactions.
3. **Prototyping**: Establishes a quick prototyping framework, facilitating the exploration of various solver/preconditioner setups.
4. **Testing**: Enables the construction of an integrated testing framework, accommodating problems from applications built on hypre.

## Getting Started

The instructions for building `hypredrive` are given below:

```
 $ ./autogen.sh
 $ ./configure --with-hypre-include=${HYPRE_INSTALL_DIR}/include \
               --with-hypre-lib=${HYPRE_INSTALL_DIR}/lib
 $ make all
 $ make check
 $ make install
```

Note:
1. The first step `./autogen.sh` must be executed only once after cloning this repository.
2. [hypre](https://github.com/hypre-space/hypre) needs to be installed at
   `${HYPRE_INSTALL_DIR}`.
3. For GPU support, add `--with-cuda` for NVIDIA GPUs or `--with-hip` for AMD GPUs in the
   `./configure` line.
4. An installation prefix can be passed to `./configure` such as `--prefix=${INSTALL_PATH}`.
   For more configure options, type `./configure -help`.

## Examples

The user's manual has a detailed section about running examples. For a quick-start, try
running the first example from the top-level folder:

```
 $ mpirun -np 1 ./hypredrive example/ex1.yml
```

## Documentation

Check the user's manual
[here](https://github.com/hypre-space/hypredrive/blob/master/docs/usrman-hypredrive.pdf). Check the developer's manual
[here](https://github.com/hypre-space/hypredrive/blob/master/docs/devman-hypredrive.pdf).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING) for details on contributing to `hypredrive`,
including the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
details. All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-861860
