#!/usr/bin/env bash

g++ -x c++ --std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-I/Users/XXX/Downloads/libtorch/include \
-I/Users/XXX/Downloads/libtorch/include/torch/csrc/api/include \
-L/Users/XXX/Downloads/libtorch/lib \
-O3 -flto \
main.cpp \
-o test_app \
-ltorch -lc10 -ltorch_cpu -lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK

# export DYLD_LIBRARY_PATH=/home/XXX/YYY/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/lib

