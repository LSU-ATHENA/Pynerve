# CUDA kernel inventory

92 CUDA kernel files organized by domain. The largest group is persistence with 40 files including cuda_matrix_reduction variants, kernel clearing, apparent pairs, warp_specialized, tile, tma, hypha_scan, distance filtration, cohomology clearing, GPU persistence reduction, and multi-GPU kernels. Distance computation contributes 12 files spanning distance kernels, extended kernels, Tensor Core kernels, fasted, tedjoin, and matrix distance variants. Spectral operations have 4 files covering Laplacian, Dirac operator, Dirac Clifford product, and eigensolver. Graph algorithms have 6 files for graph algorithms, attention, GNN, graph engine, graph zigzag, and message passing. Sheaf operations have 2 files for standard and Tensor Core sheaf Laplacian. The remaining 28 files cover encoder, autoencoder, autodiff, DMT, mapper, optimization, streaming, regularization, calibration, cup product, Reeb graph, zigzag, probabilistic, NN, metrics, Bottleneck, and Wasserstein.

## Complete kernel file listing by subdirectory

`src/persistence/cuda/` (34 files):
- `adaptive_selector_gpu.cu` -- runtime heuristic to select reduction strategy
- `cluster_16_block.cu` -- 16-block cluster reduction for small complexes
- `cluster_distributed_l2.cu` -- distributed L2 cluster reduction
- `cluster_tma_multicast.cu` -- TMA multicast cluster reduction (Hopper+)
- `cohomology_clearing_cuda.cu` -- cohomology clearing with natural clearing
- `cuda_blackwell_benchmark.cu` -- Blackwell architecture benchmark suite
- `cuda_blackwell_tma.cu` -- Blackwell TMA kernel experiments
- `cuda_matrix_reduction_apparent_pairs.cu` -- apparent pair detection during reduction
- `cuda_matrix_reduction_compute.cu` -- core matrix reduction compute kernel
- `cuda_matrix_reduction_diagram.cu` -- pair extraction from reduced matrix
- `cuda_matrix_reduction_kernels.cu` -- main reduction kernel dispatch
- `cuda_matrix_reduction_warp.cu` -- warp-level reduction primitives
- `distance_filtration_cuda.cu` -- combined distance + filtration on GPU
- `distance_filtration_wrappers_cuda.cu` -- wrapper kernels for distance filtration
- `kernel_apparent_pairs_cuda.cu` -- apparent pair detection kernel
- `kernel_clearing_cuda.cu` -- clearing optimization kernel
- `kernel_edge_extraction_cuda.cu` -- edge extraction from distance matrix
- `kernel_hypha_scan.cu` -- hypha scan (column skipping) kernel
- `kernel_ptx_micro_ops_cuda.cu` -- PTX-level micro-operations
- `kernel_tile_cuda.cu` -- tiled matrix reduction
- `kernel_tma_cuda.cu` -- Tensor Memory Accelerator reduction (Hopper+)
- `kernel_warp_specialized_cuda.cu` -- warp-specialized reduction (Hopper+)
- `matrix_distance_api_cuda.cu` -- distance API dispatch
- `matrix_distance_api_entrypoints.cu` -- API entry points for distance
- `matrix_distance_config_cuda.cu` -- distance kernel configuration
- `matrix_distance_cuda.cu` -- main distance computation kernel
- `matrix_distance_tiled_cuda.cu` -- tiled distance computation
- `matrix_reduction_launch_cuda.cu` -- reduction launch configuration
- `multi_gpu_cuda.cu` -- multi-GPU coordination kernels
- `tensor_core_benchmark.cu` -- Tensor Core benchmark suite
- `tensor_core_cuda.cu` -- Tensor Core matrix operations
- `tensor_core_wrappers_cuda.cu` -- Tensor Core wrapper kernels
- `tuning_tma_cuda.cu` -- TMA tuning parameters

`src/cuda/kernels/` (12 files):
- `bottleneck_distance.cu`
- `distance_fasted.cu`
- `distance_kernels.cu`
- `distance_kernels_ext.cu`
- `distance_tedjoin.cu`
- `gpu_persistence_launcher.cu`
- `gpu_persistence_reduction.cu`
- `mapper_gpu.cu`
- `persistence_image.cu`
- `reduction_kernels.cu`
- `specseq_reduction.cu`
- `wasserstein_distance.cu`

`src/streaming/gpu/` (2 files):
- `streaming_persistence_cuda.cu` -- chunk-based streaming on GPU
- `windowed_ph_cuda.cu` -- sliding window persistence on GPU


<- [Back to GPU Acceleration index](index.md)
