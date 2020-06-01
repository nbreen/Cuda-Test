# Cuda
This repo constains my work with Nvidia CUDA on my personal server.
Currently just tinkering with CUDA on a dual GPU configuration.

My personal server is equiped with 2x Nvidia GeForce GTX 680.
The 680 runs Nvidia's Kepler Architecture and is capable of CUDA 3.0.
Since unified addressings (UVA) for multi-GPU machines is not supported in CUDA 3.0 you must us cudaMemcpyPeer to copy data between the GPU's.
However, this system does benefit from Unified memory addressing via cudaMallocManaged
