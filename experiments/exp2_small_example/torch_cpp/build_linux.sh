#!/usr/bin/env bash

#set -o xtrace

LIBTORCH_PATH="/home/<USER>/Downloads/libtorch-shared-with-deps-2.5.1+cpu"

g++-11 -x c++ --std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-I${LIBTORCH_PATH}/libtorch/include \
-I${LIBTORCH_PATH}/libtorch/include/torch/csrc/api/include \
-L${LIBTORCH_PATH}/libtorch/lib \
-O3 -flto \
main.cpp \
-o test_app_linux \
-ltorch -lc10 -ltorch_cpu -lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK

export LD_LIBRARY_PATH=${LIBTORCH_PATH}/libtorch/lib

./test_app_linux
