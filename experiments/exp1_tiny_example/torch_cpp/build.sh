#!/usr/bin/env bash

g++-11 -x c++ --std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-I/home/XXX/YYY/burt/bin_micrograd/orig/tiny_example/torch_cpp/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/include \
-I/home/XXX/YYY/burt/bin_micrograd/orig/tiny_example/torch_cpp/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/include/torch/csrc/api/include \
-L/home/XXX/YYY/burt/bin_micrograd/orig/tiny_example/torch_cpp/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/lib \
-O3 -flto \
main.cpp \
-o test_app \
-ltorch -lc10 -ltorch_cpu -lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK

#-ltorch -lc10 -ltorch_cpu
#-lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK \
#-O3 \
#-ltorch -ltorch_cpu -lc10 -lgomp -lpthread \
#-lc10 -ltorch_cpu -ltorch \
#-lc10 -ltorch_cpu -ltorch -lasmjit -lcpuinfo -ldnnl -lfbgemm -lfmt -lkineto -lprotobuf -lprotoc -lpthreadpool -lXNNPACK \

# export LD_LIBRARY_PATH=/home/XXX/YYY/burt/bin_micrograd/orig/tiny_example/torch_cpp/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/lib
