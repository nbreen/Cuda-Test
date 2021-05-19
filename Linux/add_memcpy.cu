#include <iostream>

__global__ void add(int* devVal, int addEnd){
    *devVal= *devVal + addEnd;
    printf("Value after kernel is %d\n", *devVal);
}

int main(void){
    int* p0;
    int* p1;
    int* h1;
    int currentDev;
    size_t size = sizeof(int);

    cudaSetDevice(0);

    cudaMallocManaged(&p0, size);
    *p0 = 2;

    cudaGetDevice(&currentDev);
    printf("Calling add on device: %d\n", currentDev);
    add<<<1,1>>>(p0, 2);

    cudaDeviceSynchronize();

    cudaSetDevice(1);

    cudaMallocManaged(&p1, size);

    cudaError_t memErr = cudaMemcpyPeer(p1, 1, p0, 0, size);

    cudaGetDevice(&currentDev);
    printf("Calling add on device: %d\n", currentDev);
    add<<<1,1>>>(p1, 3);

    cudaDeviceSynchronize();

    printf("The final value is %d\n", *p1);
}