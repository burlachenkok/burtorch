#!/usr/bin/env bash

LIBTORCH_PATH="/Users/XXX/Downloads"

g++ -x c++ --std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-I${LIBTORCH_PATH}/libtorch/include \
-I${LIBTORCH_PATH}/libtorch/include/torch/csrc/api/include \
-L${LIBTORCH_PATH}/libtorch/lib \
-O3 -flto \
main.cpp \
-o test_app \
-ltorch -lc10 -ltorch_cpu -lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK

# export DYLD_LIBRARY_PATH=${LIBTORCH_PATH}/libtorch/lib
