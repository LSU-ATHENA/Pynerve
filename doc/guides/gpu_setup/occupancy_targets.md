# Occupancy targets by kernel type

Occupancy targets vary by kernel type. The distance matrix kernel targets 75-100% occupancy as it is memory bandwidth bound. Matrix reduction targets 50-75% as it is register and L1 bound. Apparent pairs target 75-100% as a compute-bound kernel. The persistence image kernel targets 50% as it is shared memory bound. Spectral operations target 50-75% as they are compute and memory bound. Graph algorithms target 75-100% as memory bandwidth bound. The sheaf Laplacian targets 50-75% as compute bound.


<- [Back to GPU Acceleration index](index.md)
