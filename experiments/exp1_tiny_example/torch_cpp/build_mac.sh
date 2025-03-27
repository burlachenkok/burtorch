#!/usr/bin/env bash

#https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.2.2.zip
LIBTORCH_PATH="/Users/burlack/Downloads/libtorch22"

g++ -x c++ --std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-I${LIBTORCH_PATH}/include \
-I${LIBTORCH_PATH}/include/torch/csrc/api/include \
-L${LIBTORCH_PATH}/lib \
-O3 -flto \
main.cpp \
-o test_app_macos \
-ltorch -lc10 -ltorch_cpu -lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK

export DYLD_LIBRARY_PATH=${LIBTORCH_PATH}/lib
./test_app_macos
