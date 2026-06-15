# Occupancy analysis by GPU architecture

Pynerve auto-tunes kernel launch parameters for each architecture. The tuner (`src/gpu/tuner_nvidia_auto.cpp`) runs microbenchmarks at startup and caches results to `~/.cache/nerve/gpu_tuning.json`.

## Turing (SM75, compute capability 7.5)

On Turing, the matrix reduction kernel uses a block size of 256 with a few kilobytes of shared memory and achieves 50% occupancy. The FP16 distance kernel uses a block size of 128 with tens of kilobytes of shared memory and 75% occupancy. Apparent pairs use a block size of 256 with a few kilobytes of shared memory and full 100% occupancy. The persistence image kernel uses a block size of 128 with tens of kilobytes of shared memory and 50% occupancy.

## Ampere (SM80/SM86, compute capability 8.0/8.6)

On Ampere, the matrix reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The BF16 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy. The FP32 distance kernel uses a block size of 128 with tens of kilobytes of shared memory and 75% occupancy. WGMMA distance uses a block size of 128 with tens of kilobytes of shared memory and 50% occupancy. Apparent pairs use a block size of 256 with a few kilobytes of shared memory and full 100% occupancy. The persistence image kernel uses a block size of 128 with tens of kilobytes of shared memory and 33% occupancy.

## Hopper (SM90, compute capability 9.0)

On Hopper, the TMA reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy, employing TMA for data movement. The warp-specialized reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy via persistent thread partitioning. The FP8 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy through the TCgen05 Tensor Core path. The FP16 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy via the WGMMA path. The cluster reduce kernel uses block 512 with hundreds of kilobytes of shared memory and 50% occupancy via cluster-level reduction. The DPX reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy via DPX acceleration. Apparent pairs use a block size of 256 with a few kilobytes of shared memory and full 100% occupancy. The persistence image kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy.

## Blackwell (SM100, compute capability 10.0)

On Blackwell, the TMA reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy via enhanced TMA multicast. The FP4 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy using 4th-generation Tensor Cores. The FP8 distance kernel uses a block size of 256 with tens of kilobytes of shared memory and 75% occupancy. The warp-specialized reduction kernel uses a block size of 256 with tens of kilobytes of shared memory and 50% occupancy. The cluster reduce kernel uses a block size of 512 with hundreds of kilobytes of shared memory and 50% occupancy with larger clusters.

## Tuning cache format

```json
{
  "sm_90": {
    "reduction": {"block_size": 256, "shared_mem": 32768, "use_tma": true},
    "distance": {"block_size": 256, "shared_mem": 65536, "use_tcgen05": true,
                 "precision": "fp8", "tile_size": 16}
  },
  "sm_80": {
    "reduction": {"block_size": 256, "shared_mem": 12288, "use_atomics": false},
    "distance": {"block_size": 128, "shared_mem": 16384, "precision": "fp16"}
  }
}
```


<- [Back to GPU Acceleration index](index.md)
