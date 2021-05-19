#include <iostream>
#include <math.h>

// Makes this a cuda function

__global__
void init(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}
__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void){
  //Calling cudaMalloc for memory accessibile by the gpu
  int N = 1<<20; // 1M elements

  float *x;
  float *y;
  int blockSize = 256;
  int numBlocks = (N + blockSize -1) / blockSize;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host

  init<<<numBlocks,blockSize>>>(N ,x, y);
  // Run kernel on 1M elements on the CPU
  add<<<numBlocks,blockSize>>>(N, x, y);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;

  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}