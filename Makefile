CC = g++
CCFLAGS = -g
CXX = nvcc
CXXFLAGS = -g

all: memcpy add add_cuda add_block add_grid add_reduced add_cuda_dual-gpu add_memcpy cuda_test hello

memcpy: CUDA_memcpy.cu 
	$(CXX) CUDA_memcpy.cu $(CXXFLAGS) -o CUDA_memcpy

add: add.cpp
	$(CC) $(CCFLAGS) -o add add.cpp

add_cuda: add.cu
	$(CXX) add.cu $(CXXFLAGS) -o add_cuda

add_block: add_block.cu
	$(CXX) add_block.cu $(CXXFLAGS) -o add_cuda_block

add_grid: add_grid.cu
	$(CXX) add_grid.cu $(CXXFLAGS) -o add_cuda_grid

add_reduced: add_reduced.cu
	$(CXX) add_reduced.cu $(CXXFLAGS) -o add_cuda_reduced

add_cuda_dual-gpu: add_reduced_multi-gpu.cu
	$(CXX) add_reduced_multi-gpu.cu $(CXXFLAGS) -o add_cuda_dual-gpu

add_memcpy: add_memcpy.cu
	$(CXX) add_memcpy.cu $(CXXFLAGS) -o add_cuda_memcpy

cuda_test: cuda_test.cu
	$(CXX) cuda_test.cu $(CXXFLAGS) -o cuda_test

hello: hello.cu
	$(CXX) hello.cu $(CXXFLAGS) -o hello

