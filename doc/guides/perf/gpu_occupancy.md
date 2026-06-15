# GPU occupancy and SM utilization

Pynerve auto-tunes kernel launch parameters for each target GPU architecture. On Volta (SM70), Tensor Cores v1 are probed and blockSize with gridSize are auto-tuned. Turing (SM75) uses INT8 Tensor Cores with tileSize and prefetch tuning. Ampere (SM80) probes BF16 and WGMMA, tuning clusterSize and sharedMem. Ada Lovelace (SM89) probes FP8 E4M3/E5M2 with useTCgen05 tuning. Hopper (SM90) probes TMA, DPX, and cluster features, tuning useTMA, clusterSize, and useWGMMA. Blackwell (SM100) probes FP4 and 4th-gen Tensor Cores, tuning useFP4 and TMA multicast.

The NVIDIA auto-tuner runs microbenchmarks at startup for each unique GPU type and caches tuned parameters to `~/.cache/pynerve/gpu_tuning.json`.

### Occupancy targets

Occupancy targets vary by kernel type. The distance matrix kernel targets 75-100% occupancy as it is memory bandwidth bound. Matrix reduction targets 50-75% as it is register and L1 bound. Apparent pairs target 75-100% as a compute-bound kernel. The persistence image kernel targets 50% as it is shared memory bound.

On Turing, the matrix reduction kernel uses a block size of 256 with a few kilobytes of shared memory and achieves 50% occupancy. The FP16 distance kernel uses a block size of 128 with tens of kilobytes of shared memory and 75% occupancy. Apparent pairs use a block size of 256 with a few kilobytes of shared memory and full 100% occupancy. The persistence image kernel uses a block size of 128 with tens of kilobytes of shared memory and 50% occupancy.

On Ampere, the matrix reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The BF16 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy. The FP32 distance kernel uses a block size of 128 with tens of kilobytes of shared memory and 75% occupancy. The WGMMA distance kernel uses a block size of 128 with tens of kilobytes of shared memory and 50% occupancy.

On Hopper, the TMA reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The warp-specialized kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The FP8 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy. The cluster reduce kernel uses a block size of 512 with hundreds of kilobytes of shared memory and 50% occupancy.

On Blackwell, the TMA reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The FP4 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy. The warp-specialized kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The cluster reduce kernel uses a block size of 512 with hundreds of kilobytes of shared memory and 50% occupancy.

[Back to index](index.md)
