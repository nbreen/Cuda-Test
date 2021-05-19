# Cuda
This repo constains my work with Nvidia CUDA.
Currently just tinkering with CUDA on single as well as dual GPU configuration.

My personal server is equiped with 2x Nvidia GeForce GTX 680.
The 680 runs Nvidia's Kepler Architecture and is capable of CUDA 3.0.
Since unified addressings (UVA) for multi-GPU machines is not supported in CUDA 3.0 you must us cudaMemcpyPeer to copy data between the GPU's.
However, this system does benefit from Unified memory addressing via cudaMallocManaged

My personal machine is equipped with an Nvidia Geforce GTX 1080
The 1080 runs Nvidia's Pascal Architecture and is capabale of CUDA 6.1
