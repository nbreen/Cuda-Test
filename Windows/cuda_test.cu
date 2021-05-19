#include <iostream>

int main(void){
    int devcount;
    int deviceInUse;
    int attrVal;
    int canAccess;

    cudaGetDeviceCount(&devcount);
    printf("This machine has %d CUDA capable GPUs\n", devcount);
    printf("------------------------\n");
    
    int deviceGrid[devcount][8];
    cudaSetDevice(0);
    cudaGetDevice(&deviceInUse);
    printf("This machine is currently using device %d\n", deviceInUse);
    printf("------------------------\n");

    if (devcount > 1) {
        printf("Changing device in use.....\n");
        for (int i  = 1; i < devcount; i++) {
            cudaSetDevice(i);
            cudaGetDevice(&deviceInUse);
            printf("This machine is currently using device %d\n", deviceInUse);
            printf("------------------------\n");
        }

        printf("Checking cross talk between devices.....\n");

        for (int i = 0; i < devcount; i++) {
            if (i + 1 < devcount) {
                cudaDeviceCanAccessPeer(&canAccess, i, i + 1);

                if (canAccess == 1) {
                    printf("CUDA Device %d can access CUDA Device %d\n", i, i + 1);
                } else {
                    printf("ERROR CUDA Device %d can't access CUDA Device %d ensure that this devices supports P2P communication\n", i, i + 1);
                }
            } else {
                cudaDeviceCanAccessPeer(&canAccess, i, i - 1);

                if (canAccess == 1){
                    printf("CUDA Device %d can access CUDA Device %d\n", i, i - 1);
                } else {
                    printf("ERROR CUDA Device %d can't access CUDA Device %d ensure that this devices supports P2P communication\n", i, i - 1);
                }
            }
        }        

    }

    printf("------------------------\n");
    printf("Building block matrix...\n");
    printf("------------------------\n");

    for (int i  = 0; i < devcount; i++){
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxThreadsPerBlock, i);
        deviceGrid[i][0] = attrVal;
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxBlockDimX, i);
        deviceGrid[i][1] = attrVal;
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxBlockDimY, i);
        deviceGrid[i][2] = attrVal;
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxBlockDimZ, i);
        deviceGrid[i][3] = attrVal;
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxGridDimX, i);
        deviceGrid[i][4] = attrVal;
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxGridDimY, i);
        deviceGrid[i][5] = attrVal;
        cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxGridDimZ, i);
        deviceGrid[i][6] = attrVal;
    }

    for (int i = 0; i < devcount; i++) {
        printf("Device %d has the following properties\n", i);
        printf("Max Threads Per Block: %d\n", deviceGrid[i][0]);
        printf("Max Blocks X: %d\n", deviceGrid[i][1]);
        printf("Max Blocks Y: %d\n", deviceGrid[i][2]);
        printf("Max Blocks Z: %d\n", deviceGrid[i][3]);
        printf("Max Grid X: %d\n", deviceGrid[i][4]);
        printf("Max Grid Y: %d\n", deviceGrid[i][5]);
        printf("Max Grid Z: %d\n", deviceGrid[i][6]);
        printf("----------------------------------\n");
    }
}