#!/bin/bash

set -euxo pipefail

SUFFIX=cuda-debug
HYPREDRV_BUILD=$(pwd)/build-${SUFFIX}
HYPREDRV_INSTALL=$(pwd)/install-${SUFFIX}
rm -rf ${HYPREDRV_BUILD} ${HYPREDRV_INSTALL}
#cmake -DHYPRE_ROOT=$(pwd)/hypre/install-${SUFFIX} \
cmake -DCMAKE_INSTALL_PREFIX=${HYPREDRV_INSTALL} \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DBUILD_SHARED_LIBS=OFF \
      -DHYPREDRV_ENABLE_TESTING=ON \
      -DHYPREDRV_ENABLE_EXAMPLES=ON \
      -DHYPREDRV_ENABLE_COVERAGE=ON \
      -DHYPREDRV_ENABLE_ANALYSIS=ON \
      -DHYPRE_ENABLE_CUDA=ON \
      -DHYPRE_ENABLE_UMPIRE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCLANG_FORMAT=clang-format-18 \
      -S . -B ${HYPREDRV_BUILD} #--debug-trycompile
cmake --build ${HYPREDRV_BUILD} --parallel
cmake --install ${HYPREDRV_BUILD}
cmake --build ${HYPREDRV_BUILD} --target test
#cmake --build ${HYPREDRV_BUILD} --target coverage
#cmake --build ${HYPREDRV_BUILD} --target clang-tidy
#cmake --build ${HYPREDRV_BUILD} --target cppcheck
#cmake --build ${HYPREDRV_BUILD} --target format
