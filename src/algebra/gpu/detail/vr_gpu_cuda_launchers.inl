// Forward declarations from cuda_vr_complex.cu and companion CUDA kernels.
extern void launchVrEdgeDetection(
    const double* d_points,
    int* d_adjacency,
    int n_points,
    int point_dim,
    double max_radius_sq,
    cudaStream_t stream
);

extern void launchOptimizedEdgeDetection(
    const double* d_points,
    int* d_adjacency,
    int* d_edge_count,
    int n_points,
    int point_dim,
    double max_radius,
    cudaStream_t stream
);

extern void launchOptimizedDistanceMatrix(
    const double* d_points,
    double* d_distance_matrix,
    int n_points,
    int point_dim,
    double max_radius,
    cudaStream_t stream
);

extern void launchSimplexExpansion(
    const int* d_adjacency,
    const int* d_current_simplices,
    int* d_new_simplices,
    int* d_new_count,
    int n_current,
    int current_dim,
    int n_points,
    int max_new,
    cudaStream_t stream
);

extern void launchVrFiltrationValues(
    const double* d_points,
    const int* d_simplices,
    const int* d_simplex_sizes,
    double* d_filtration_values,
    int n_simplices,
    int point_dim,
    int n_points,
    cudaStream_t stream
);

extern void launchCliqueExpansion(
    const int* d_adjacency,
    const int* d_edges,
    int n_edges,
    int* d_out_simplices,
    int* d_out_simplex_count,
    int max_dimension,
    int n_points,
    cudaStream_t stream
);
