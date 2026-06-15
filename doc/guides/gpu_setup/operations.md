# Available operations

Pynerve supports a wide range of GPU-accelerated operations. Cohomology reduction uses the CUDA_HYBRID backend with warp-shuffle matrix reduction and zero atomics. Distance computation is accelerated with FP16, BF16, and FP8 via Tensor Cores. Spectral decomposition uses a Krylov-shift GPU eigensolver. Graph algorithms cover connected components, MST, and shortest paths. Sheaf Laplacian assembly and solve run on sparse GPU. Deterministic GPU PRNG with seeded streams handles RNG. Discrete Morse Theory computes gradients and reductions on GPU.

Bottleneck and Wasserstein distances use sparse matching and Hungarian algorithm on GPU respectively. Persistence image rendering, mapper cover construction and clustering, and streaming persistence (windowed PH) are all GPU-accelerated. Autodiff provides a differentiable persistence backward pass with GPU topological autoencoders and regularizers. Spectral sequence reduction, cup product computation, Reeb graph construction, zigzag persistence, GNN layers (message passing and attention), GPU-based ML encoders, optimization sweeps, probabilistic TDA reduction, and GPU runtime calibration round out the available operations.


<- [Back to GPU Acceleration index](index.md)
