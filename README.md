# *hypredrive*

High-level interface for solving linear systems with hypre, providing a user-friendly way to leverage its functionalities. Key features are:

1. **YAML Input**: Accepts configuration parameters written in the structured and human-readable YAML format.
2. **Intuitive Interface**: Offers a clear and concise API, encapsulating the functionalities of *hypre* to ensure user-friendly interactions.
3. **Prototyping**: Establishes a quick prototyping framework, facilitating the exploration of various solver/preconditioner setups.
4. **Testing**: Enables the construction of an integrated testing framework, accommodating problems from applications built on hypre.

## Getting Started

The instructions for building *hypredrive* are given below:

```
 $ autoreconf -i
 $ ./configure --prefix=${PWD}/install --with-hypre-dir=${HYPRE_INSTALL_DIR}
 $ make all
 $ make check
 $ make install
```

Note:
1. The first step must be executed only once after cloning this repository.
2. [hypre](https://github.com/hypre-space/hypre) needs to be installed at
   `${HYPRE_INSTALL_DIR}`.
3. For GPU support, add `--with-cuda` (NVIDIA GPUs) or `--with-hip` (AMD GPUs) to
   `./configure`.

## Examples

The user's manual has a detailed section about running examples. For a quick-start, try
running the first example from the top-level folder:

```
 $ mpirun -np 1 ./hypredrive examples/ex1.yml
```

## Documentation

Check *hypredrive*'s manual [here](https://hypredrive.readthedocs.io/en/latest/).

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING.md) for details on contributing to *hypredrive*,
including the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
details. All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-861860
