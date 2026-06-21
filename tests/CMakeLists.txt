cmake_minimum_required(VERSION 3.20)

# Test Organization
# C++ tests are listed individually below with descriptive target names
# and LABELS for ctest filtering. Categories:
#   cpu / cuda / xpu        -- hardware backend
#   regression              -- regression tests (the bulk of C++ tests)
#   generated               -- contract/generated tests
#   distributed / mpi       -- multi-node tests
#   persistence / algorithms / algebra / nn / etc. -- domain area
#   quality / static        -- code quality checks
#
# Run with: ctest --test-dir build -L "cpu|generated" --output-on-failure

find_package(Python3 COMPONENTS Interpreter QUIET)
option(NERVE_ALLOW_MISSING_PYTEST
    "Allow BUILD_TESTS=ON to skip Python pytest CTest entries when pytest is missing"
    OFF)

function(nerve_add_cpp_test target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(TARGET nerve_core)
        target_link_libraries(${target} PRIVATE nerve_core)
    endif()
    add_test(NAME ${target} COMMAND ${target})
    # Group tests by category for IDE navigation
    if(source MATCHES "^cpp/")
        string(REGEX REPLACE "^cpp/(.+)_[^_]+$" "C++/\\1" group_id ${source})
        set_property(TARGET ${target} PROPERTY FOLDER ${group_id})
    endif()
endfunction()

nerve_add_cpp_test(nerve_cpp_generated_numeric cpp/generated_numeric_contracts.cpp)
set_tests_properties(nerve_cpp_generated_numeric PROPERTIES LABELS "generated;cpu;operators;gradient")

nerve_add_cpp_test(nerve_cpp_kernel_methods_contracts cpp/kernel_methods_contracts.cpp)
target_sources(nerve_cpp_kernel_methods_contracts PRIVATE ${CMAKE_SOURCE_DIR}/src/algorithms/kernel_methods.cpp)
set_tests_properties(nerve_cpp_kernel_methods_contracts PROPERTIES LABELS "algorithms;cpu;operators")

nerve_add_cpp_test(nerve_cpp_build_features cpp/build_feature_contracts.cpp)
target_compile_definitions(nerve_cpp_build_features PRIVATE
    NERVE_EXPECT_MPI_REQUIRED=0
    NERVE_EXPECT_CUDA_VERSION_MAJOR=${NERVE_CUDA_REQUIRED_VERSION_MAJOR}
    NERVE_EXPECT_CUDA_VERSION_MINOR=${NERVE_CUDA_REQUIRED_VERSION_MINOR}
)
set_tests_properties(nerve_cpp_build_features PROPERTIES LABELS "build;cpu")

nerve_add_cpp_test(nerve_cpp_algorithm_metrics_regressions cpp/algorithm_metrics_regressions.cpp)
set_tests_properties(nerve_cpp_algorithm_metrics_regressions PROPERTIES LABELS "algorithms;metrics;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algorithm_wasserstein_regressions cpp/algorithm_wasserstein_regressions.cpp)
set_tests_properties(nerve_cpp_algorithm_wasserstein_regressions PROPERTIES LABELS "algorithms;wasserstein;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algorithm_edge_collapse_regressions cpp/algorithm_edge_collapse_regressions.cpp)
set_tests_properties(nerve_cpp_algorithm_edge_collapse_regressions PROPERTIES LABELS "algorithms;edge-collapse;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algorithm_acceleration_regressions cpp/algorithm_acceleration_regressions.cpp)
set_tests_properties(nerve_cpp_algorithm_acceleration_regressions PROPERTIES LABELS "algorithms;acceleration;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algorithm_lockfree_regressions cpp/algorithm_lockfree_regressions.cpp)
set_tests_properties(nerve_cpp_algorithm_lockfree_regressions PROPERTIES LABELS "algorithms;lockfree;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algorithm_serialization_regressions cpp/algorithm_serialization_regressions.cpp)
set_tests_properties(nerve_cpp_algorithm_serialization_regressions PROPERTIES LABELS "algorithms;serialization;cpu;regression")

nerve_add_cpp_test(nerve_cpp_adaptive_selector_regressions cpp/adaptive_selector_regressions.cpp)
set_tests_properties(nerve_cpp_adaptive_selector_regressions PROPERTIES LABELS "algorithms;cpu;regression")

nerve_add_cpp_test(nerve_cpp_adaptive_acceleration_regressions cpp/adaptive_acceleration_regressions.cpp)
set_tests_properties(nerve_cpp_adaptive_acceleration_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_ph4_certificate_regressions cpp/ph4_certificate_regressions.cpp)
set_tests_properties(nerve_cpp_ph4_certificate_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_sheaf_validation_regressions cpp/sheaf_validation_regressions.cpp)
set_tests_properties(nerve_cpp_sheaf_validation_regressions PROPERTIES LABELS "sheaf;cpu;regression")

nerve_add_cpp_test(nerve_cpp_field_aware_vr_regressions cpp/field_aware_vr_regressions.cpp)
set_tests_properties(nerve_cpp_field_aware_vr_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_vr_backend_regressions cpp/vr_backend_regressions.cpp)
set_tests_properties(nerve_cpp_vr_backend_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_rng_determinism_regressions cpp/rng_determinism_regressions.cpp)
set_tests_properties(nerve_cpp_rng_determinism_regressions PROPERTIES LABELS "rng;determinism;cpu;regression")

nerve_add_cpp_test(nerve_cpp_thread_pool_regressions cpp/thread_pool_regressions.cpp)
set_tests_properties(nerve_cpp_thread_pool_regressions PROPERTIES LABELS "threading;cpu;regression")

nerve_add_cpp_test(nerve_cpp_enhanced_thread_pool_regressions cpp/enhanced_thread_pool_regressions.cpp)
set_tests_properties(nerve_cpp_enhanced_thread_pool_regressions PROPERTIES LABELS "threading;core;cpu;regression")

nerve_add_cpp_test(nerve_cpp_encoder_validation_regressions cpp/encoder_validation_regressions.cpp)
set_tests_properties(nerve_cpp_encoder_validation_regressions PROPERTIES LABELS "encoders;cpu;regression")

nerve_add_cpp_test(nerve_cpp_diagram_conv_regressions cpp/diagram_conv_regressions.cpp)
set_tests_properties(nerve_cpp_diagram_conv_regressions PROPERTIES LABELS "nn;cpu;regression")

nerve_add_cpp_test(nerve_cpp_autodiff_tensor_regressions cpp/autodiff_tensor_regressions.cpp)
set_tests_properties(nerve_cpp_autodiff_tensor_regressions PROPERTIES LABELS "autodiff;cpu;regression")

nerve_add_cpp_test(nerve_cpp_approx_distilled_vr_regressions cpp/approx_distilled_vr_regressions.cpp)
set_tests_properties(nerve_cpp_approx_distilled_vr_regressions PROPERTIES LABELS "persistence;approximation;cpu;regression")

nerve_add_cpp_test(nerve_cpp_precision_distance_regressions cpp/precision_distance_regressions.cpp)
set_tests_properties(nerve_cpp_precision_distance_regressions PROPERTIES LABELS "precision;cpu;regression")

nerve_add_cpp_test(nerve_cpp_tolerance_regressions cpp/tolerance_regressions.cpp)
set_tests_properties(nerve_cpp_tolerance_regressions PROPERTIES LABELS "math;cpu;regression")

nerve_add_cpp_test(nerve_cpp_ph_gradient_regressions cpp/ph_gradient_regressions.cpp)
set_tests_properties(nerve_cpp_ph_gradient_regressions PROPERTIES LABELS "persistence;gradient;cpu;regression")

nerve_add_cpp_test(nerve_cpp_persistence_metrics_regressions cpp/persistence_metrics_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_metrics_regressions PROPERTIES LABELS "metrics;persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_bit_packed_regressions cpp/bit_packed_regressions.cpp)
set_tests_properties(nerve_cpp_bit_packed_regressions PROPERTIES LABELS "algebra;persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_clear_compress_regressions cpp/clear_compress_regressions.cpp)
set_tests_properties(nerve_cpp_clear_compress_regressions PROPERTIES LABELS "persistence;memory;cpu;regression")

nerve_add_cpp_test(nerve_cpp_incremental_updates_regressions cpp/incremental_updates_regressions.cpp)
set_tests_properties(nerve_cpp_incremental_updates_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_compact_summary_regressions cpp/compact_summary_regressions.cpp)
set_tests_properties(nerve_cpp_compact_summary_regressions PROPERTIES LABELS "summary;cpu;regression")

nerve_add_cpp_test(nerve_cpp_per_dimension_regressions cpp/per_dimension_regressions.cpp)
set_tests_properties(nerve_cpp_per_dimension_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_vr_distance_tiled_regressions cpp/vr_distance_tiled_regressions.cpp)
set_tests_properties(nerve_cpp_vr_distance_tiled_regressions PROPERTIES LABELS "persistence;distance;cpu;regression")

nerve_add_cpp_test(nerve_cpp_vr_fast_input_regressions cpp/vr_fast_input_regressions.cpp)
set_tests_properties(nerve_cpp_vr_fast_input_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_vr_fast_compat_regressions cpp/vr_fast_compat_regressions.cpp)
set_tests_properties(nerve_cpp_vr_fast_compat_regressions PROPERTIES LABELS "persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_witness_regressions cpp/witness_regressions.cpp)
set_tests_properties(nerve_cpp_witness_regressions PROPERTIES LABELS "persistence;witness;cpu;regression")

nerve_add_cpp_test(nerve_cpp_streaming_column_regressions cpp/streaming_column_regressions.cpp)
set_tests_properties(nerve_cpp_streaming_column_regressions PROPERTIES LABELS "persistence;streaming;cpu;regression")

nerve_add_cpp_test(nerve_cpp_windowed_ph_regressions cpp/windowed_ph_regressions.cpp)
set_tests_properties(nerve_cpp_windowed_ph_regressions PROPERTIES LABELS "streaming;persistence;cpu;regression")

nerve_add_cpp_test(nerve_cpp_probabilistic_regressions cpp/probabilistic_regressions.cpp)
set_tests_properties(nerve_cpp_probabilistic_regressions PROPERTIES LABELS "probabilistic;cpu;regression")

nerve_add_cpp_test(nerve_cpp_distance_matrix_regressions cpp/distance_matrix_regressions.cpp)
set_tests_properties(nerve_cpp_distance_matrix_regressions PROPERTIES LABELS "distance;cpu;regression")

nerve_add_cpp_test(nerve_cpp_simd_distance_regressions cpp/simd_distance_regressions.cpp)
set_tests_properties(nerve_cpp_simd_distance_regressions PROPERTIES LABELS "algebra;simd;cpu;regression")

nerve_add_cpp_test(nerve_cpp_simd_metrics_regressions cpp/simd_metrics_regressions.cpp)
set_tests_properties(nerve_cpp_simd_metrics_regressions PROPERTIES LABELS "algebra;simd;metrics;cpu;regression")

nerve_add_cpp_test(nerve_cpp_simd_mapper_vectorization_regressions cpp/simd_mapper_vectorization_regressions.cpp)
set_tests_properties(nerve_cpp_simd_mapper_vectorization_regressions PROPERTIES LABELS "algorithms;mapper;vectorization;cpu;regression")

nerve_add_cpp_test(nerve_cpp_hnsw_index_tests cpp/hnsw_index_tests.cpp)
set_tests_properties(nerve_cpp_hnsw_index_tests PROPERTIES LABELS "algorithms;hnsw;nn;cpu;regression")

nerve_add_cpp_test(nerve_cpp_lockfree_structures_tests cpp/lockfree_structures_tests.cpp)
if(TARGET Threads::Threads)
    target_link_libraries(nerve_cpp_lockfree_structures_tests PRIVATE Threads::Threads)
endif()
set_tests_properties(nerve_cpp_lockfree_structures_tests PROPERTIES LABELS "concurrency;lockfree;cpu;regression")

nerve_add_cpp_test(nerve_cpp_persistence_invariant_contracts cpp/persistence_invariant_contracts.cpp)
set_tests_properties(nerve_cpp_persistence_invariant_contracts PROPERTIES LABELS "persistence;invariant;generated;cpu")

nerve_add_cpp_test(nerve_cpp_cohomology_persistent_regressions cpp/cohomology_persistent_regressions.cpp)
set_tests_properties(nerve_cpp_cohomology_persistent_regressions PROPERTIES LABELS "persistence;cohomology;cpu;regression")

nerve_add_cpp_test(nerve_cpp_reduction_control_regressions cpp/reduction_control_regressions.cpp)
set_tests_properties(nerve_cpp_reduction_control_regressions PROPERTIES LABELS "persistence;reduction;cpu;regression")

nerve_add_cpp_test(nerve_cpp_exact_engine_regressions cpp/exact_engine_regressions.cpp)
set_tests_properties(nerve_cpp_exact_engine_regressions PROPERTIES LABELS "persistence;exact;cpu;regression")

nerve_add_cpp_test(nerve_cpp_matrix_reduction_contracts cpp/matrix_reduction_contracts.cpp)
set_tests_properties(nerve_cpp_matrix_reduction_contracts PROPERTIES LABELS "persistence;reduction;generated;cpu")

nerve_add_cpp_test(nerve_cpp_accelerated_regressions cpp/accelerated_regressions.cpp)
set_tests_properties(nerve_cpp_accelerated_regressions PROPERTIES LABELS "persistence;accelerated;cpu;regression")

nerve_add_cpp_test(nerve_cpp_distance_metric_axiom_contracts cpp/distance_metric_axiom_contracts.cpp)
set_tests_properties(nerve_cpp_distance_metric_axiom_contracts PROPERTIES LABELS "distance;metrics;axiom;generated;cpu")

nerve_add_cpp_test(nerve_cpp_numerical_correctness_contracts cpp/numerical_correctness_contracts.cpp)
set_tests_properties(nerve_cpp_numerical_correctness_contracts PROPERTIES LABELS "persistence;numerical;correctness;cpu")

nerve_add_cpp_test(nerve_cpp_algorithm_cross_validation_contracts cpp/algorithm_cross_validation_contracts.cpp)
set_tests_properties(nerve_cpp_algorithm_cross_validation_contracts PROPERTIES LABELS "persistence;algorithm;cross-validation;cpu")

if(ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        add_executable(nerve_cpp_mpi_smoke cpp/mpi_smoke.cpp)
        target_compile_features(nerve_cpp_mpi_smoke PRIVATE cxx_std_20)
        target_link_libraries(nerve_cpp_mpi_smoke PRIVATE MPI::MPI_CXX)
        if(TARGET nerve_core)
            target_link_libraries(nerve_cpp_mpi_smoke PRIVATE nerve_core)
        endif()
        add_test(
            NAME nerve_cpp_mpi_smoke
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:nerve_cpp_mpi_smoke>
                    ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(nerve_cpp_mpi_smoke PROPERTIES LABELS "distributed;mpi;cpu")

        add_executable(nerve_cpp_mpi_distributed_tests cpp/mpi_distributed_tests.cpp)
        target_compile_features(nerve_cpp_mpi_distributed_tests PRIVATE cxx_std_20)
        target_link_libraries(nerve_cpp_mpi_distributed_tests PRIVATE MPI::MPI_CXX)
        if(TARGET nerve_core)
            target_link_libraries(nerve_cpp_mpi_distributed_tests PRIVATE nerve_core)
        endif()
        add_test(
            NAME nerve_cpp_mpi_distributed_tests
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:nerve_cpp_mpi_distributed_tests>
                    ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(nerve_cpp_mpi_distributed_tests PROPERTIES LABELS "distributed;mpi;cpu;regression")
    endif()
endif()

nerve_add_cpp_test(nerve_cpp_persistence_cohomology_regressions cpp/persistence_cohomology_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_cohomology_regressions PROPERTIES LABELS "persistence;cohomology;cpu;regression")

nerve_add_cpp_test(nerve_cpp_persistence_reduction_regressions cpp/persistence_reduction_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_reduction_regressions PROPERTIES LABELS "persistence;reduction;cpu;regression")

nerve_add_cpp_test(nerve_cpp_persistence_kernels_contracts cpp/persistence_kernels_contracts.cpp)
set_tests_properties(nerve_cpp_persistence_kernels_contracts PROPERTIES LABELS "persistence;kernels;generated;cpu")

nerve_add_cpp_test(nerve_cpp_persistence_differentiable_regressions cpp/persistence_differentiable_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_differentiable_regressions PROPERTIES LABELS "persistence;differentiable;cpu;regression")

nerve_add_cpp_test(nerve_cpp_persistence_core_regressions cpp/persistence_core_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_core_regressions PROPERTIES LABELS "persistence;core;cpu;regression")

nerve_add_cpp_test(nerve_cpp_graphs_regressions cpp/graphs_regressions.cpp)
set_tests_properties(nerve_cpp_graphs_regressions PROPERTIES LABELS "graphs;cpu;regression")

nerve_add_cpp_test(nerve_cpp_spectral_regressions cpp/spectral_regressions.cpp)
set_tests_properties(nerve_cpp_spectral_regressions PROPERTIES LABELS "spectral;cpu;regression")

nerve_add_cpp_test(nerve_cpp_filtration_regressions cpp/filtration_regressions.cpp)
set_tests_properties(nerve_cpp_filtration_regressions PROPERTIES LABELS "filtration;cpu;regression")

nerve_add_cpp_test(nerve_cpp_serialization_regressions cpp/serialization_regressions.cpp)
set_tests_properties(nerve_cpp_serialization_regressions PROPERTIES LABELS "serialization;cpu;regression")

nerve_add_cpp_test(nerve_cpp_runtime_regressions cpp/runtime_regressions.cpp)
set_tests_properties(nerve_cpp_runtime_regressions PROPERTIES LABELS "runtime;cpu;regression")

nerve_add_cpp_test(nerve_cpp_optimization_regressions cpp/optimization_regressions.cpp)
set_tests_properties(nerve_cpp_optimization_regressions PROPERTIES LABELS "optimization;cpu;regression")

nerve_add_cpp_test(nerve_cpp_compression_regressions cpp/compression_regressions.cpp)
set_tests_properties(nerve_cpp_compression_regressions PROPERTIES LABELS "compression;cpu;regression")

nerve_add_cpp_test(nerve_cpp_dmt_regressions cpp/dmt_regressions.cpp)
set_tests_properties(nerve_cpp_dmt_regressions PROPERTIES LABELS "dmt;cpu;regression")

nerve_add_cpp_test(nerve_cpp_io_regressions cpp/io_regressions.cpp)
set_tests_properties(nerve_cpp_io_regressions PROPERTIES LABELS "io;cpu;regression")

nerve_add_cpp_test(nerve_cpp_metrics_bottleneck_regressions cpp/metrics_bottleneck_regressions.cpp)
set_tests_properties(nerve_cpp_metrics_bottleneck_regressions PROPERTIES LABELS "metrics;bottleneck;cpu;regression")

nerve_add_cpp_test(nerve_cpp_metrics_diagram_regressions cpp/metrics_diagram_regressions.cpp)
set_tests_properties(nerve_cpp_metrics_diagram_regressions PROPERTIES LABELS "metrics;diagram;cpu;regression")

nerve_add_cpp_test(nerve_cpp_memory_regressions cpp/memory_regressions.cpp)
set_tests_properties(nerve_cpp_memory_regressions PROPERTIES LABELS "memory;cpu;regression")

nerve_add_cpp_test(nerve_cpp_anomaly_regressions cpp/anomaly_regressions.cpp)
set_tests_properties(nerve_cpp_anomaly_regressions PROPERTIES LABELS "anomaly;cpu;regression")

nerve_add_cpp_test(nerve_cpp_batching_regressions cpp/batching_regressions.cpp)
set_tests_properties(nerve_cpp_batching_regressions PROPERTIES LABELS "batching;cpu;regression")

nerve_add_cpp_test(nerve_cpp_cache_regressions cpp/cache_regressions.cpp)
set_tests_properties(nerve_cpp_cache_regressions PROPERTIES LABELS "cache;cpu;regression")

nerve_add_cpp_test(nerve_cpp_approximation_regressions cpp/approximation_regressions.cpp)
set_tests_properties(nerve_cpp_approximation_regressions PROPERTIES LABELS "approximation;cpu;regression")

nerve_add_cpp_test(nerve_cpp_differentiable_regressions cpp/differentiable_regressions.cpp)
set_tests_properties(nerve_cpp_differentiable_regressions PROPERTIES LABELS "differentiable;cpu;regression")

nerve_add_cpp_test(nerve_cpp_mapper_ops_regressions cpp/mapper_ops_regressions.cpp)
set_tests_properties(nerve_cpp_mapper_ops_regressions PROPERTIES LABELS "mapper;cpu;regression")

nerve_add_cpp_test(nerve_cpp_determinism_contracts cpp/determinism_contracts.cpp)
set_tests_properties(nerve_cpp_determinism_contracts PROPERTIES LABELS "determinism;generated;cpu")

nerve_add_cpp_test(nerve_cpp_torch_regressions cpp/torch_regressions.cpp)
set_tests_properties(nerve_cpp_torch_regressions PROPERTIES LABELS "torch;cpu;regression")

nerve_add_cpp_test(nerve_cpp_gpu_column_ops_regressions cpp/gpu_column_ops_regressions.cpp)
set_tests_properties(nerve_cpp_gpu_column_ops_regressions PROPERTIES LABELS "persistence;gpu;cuda;regression")

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    add_executable(nerve_cuda_smoke cpp/cuda_smoke.cu)
    set_target_properties(nerve_cuda_smoke PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    if(TARGET CUDA::cudart)
        target_link_libraries(nerve_cuda_smoke PRIVATE CUDA::cudart)
    endif()
    add_test(NAME nerve_cuda_smoke COMMAND nerve_cuda_smoke)
    set_tests_properties(nerve_cuda_smoke PROPERTIES LABELS "cuda;cuda-hardware;gpu")

    add_executable(nerve_cuda_gpu_persistence_reduction_tests cpp/gpu_persistence_reduction_tests.cu)
    set_target_properties(nerve_cuda_gpu_persistence_reduction_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    if(TARGET CUDA::cudart)
        target_link_libraries(nerve_cuda_gpu_persistence_reduction_tests PRIVATE CUDA::cudart)
    endif()
    if(TARGET nerve_core)
        target_link_libraries(nerve_cuda_gpu_persistence_reduction_tests PRIVATE nerve_core)
    endif()
    add_test(NAME nerve_cuda_gpu_persistence_reduction_tests COMMAND nerve_cuda_gpu_persistence_reduction_tests)
    set_tests_properties(nerve_cuda_gpu_persistence_reduction_tests PROPERTIES LABELS "cuda;cuda-hardware;gpu;persistence;regression")

    add_executable(nerve_cuda_device_array_tests cpp/device_array_tests.cu)
    set_target_properties(nerve_cuda_device_array_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    if(TARGET CUDA::cudart)
        target_link_libraries(nerve_cuda_device_array_tests PRIVATE CUDA::cudart)
    endif()
    if(TARGET nerve_core)
        target_link_libraries(nerve_cuda_device_array_tests PRIVATE nerve_core)
    endif()
    add_test(NAME nerve_cuda_device_array_tests COMMAND nerve_cuda_device_array_tests)
    set_tests_properties(nerve_cuda_device_array_tests PROPERTIES LABELS "cuda;cuda-hardware;gpu;memory;regression")

    add_executable(nerve_cuda_gpu_distance_tests cpp/gpu_distance_tests.cu)
    set_target_properties(nerve_cuda_gpu_distance_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    if(TARGET CUDA::cudart)
        target_link_libraries(nerve_cuda_gpu_distance_tests PRIVATE CUDA::cudart)
    endif()
    if(TARGET nerve_core)
        target_link_libraries(nerve_cuda_gpu_distance_tests PRIVATE nerve_core)
    endif()
    add_test(NAME nerve_cuda_gpu_distance_tests COMMAND nerve_cuda_gpu_distance_tests)
    set_tests_properties(nerve_cuda_gpu_distance_tests PROPERTIES LABELS "cuda;cuda-hardware;gpu;metrics;regression")
endif()

# Integration tests
nerve_add_cpp_test(nerve_cpp_integration_vr_vs_cohomology cpp/integration_vr_vs_cohomology_consistency.cpp)
set_tests_properties(nerve_cpp_integration_vr_vs_cohomology PROPERTIES LABELS "cpu;integration;persistence")
nerve_add_cpp_test(nerve_cpp_integration_persistence_pipeline cpp/integration_persistence_pipeline.cpp)
set_tests_properties(nerve_cpp_integration_persistence_pipeline PROPERTIES LABELS "cpu;integration;persistence")
nerve_add_cpp_test(nerve_cpp_integration_filtration_persistence cpp/integration_filtration_persistence.cpp)
set_tests_properties(nerve_cpp_integration_filtration_persistence PROPERTIES LABELS "cpu;integration;filtration")
nerve_add_cpp_test(nerve_cpp_integration_distance_persistence cpp/integration_distance_persistence.cpp)
set_tests_properties(nerve_cpp_integration_distance_persistence PROPERTIES LABELS "cpu;integration;distance")
nerve_add_cpp_test(nerve_cpp_integration_serialization_persistence cpp/integration_serialization_persistence.cpp)
set_tests_properties(nerve_cpp_integration_serialization_persistence PROPERTIES LABELS "cpu;integration;serialization")

# Stress tests
nerve_add_cpp_test(nerve_cpp_stress_large_random_clouds cpp/stress_large_random_clouds.cpp)
set_tests_properties(nerve_cpp_stress_large_random_clouds PROPERTIES LABELS "cpu;stress;persistence")
nerve_add_cpp_test(nerve_cpp_stress_determinism cpp/stress_determinism.cpp)
set_tests_properties(nerve_cpp_stress_determinism PROPERTIES LABELS "cpu;stress;determinism")
nerve_add_cpp_test(nerve_cpp_stress_memory_stability cpp/stress_memory_stability.cpp)
set_tests_properties(nerve_cpp_stress_memory_stability PROPERTIES LABELS "cpu;stress;memory")

# Edge case tests
nerve_add_cpp_test(nerve_cpp_edge_case_zero_dimension cpp/edge_case_zero_dimension.cpp)
set_tests_properties(nerve_cpp_edge_case_zero_dimension PROPERTIES LABELS "cpu;edge;persistence")
nerve_add_cpp_test(nerve_cpp_edge_case_extreme_values cpp/edge_case_extreme_values.cpp)
set_tests_properties(nerve_cpp_edge_case_extreme_values PROPERTIES LABELS "cpu;edge;persistence")
nerve_add_cpp_test(nerve_cpp_edge_case_duplicate_points cpp/edge_case_duplicate_points.cpp)
set_tests_properties(nerve_cpp_edge_case_duplicate_points PROPERTIES LABELS "cpu;edge;persistence")
nerve_add_cpp_test(nerve_cpp_edge_case_degenerate_input cpp/edge_case_degenerate_input.cpp)
set_tests_properties(nerve_cpp_edge_case_degenerate_input PROPERTIES LABELS "cpu;edge;persistence")

# Algebra tests
nerve_add_cpp_test(nerve_cpp_algebra_boundary_regressions cpp/algebra_boundary_regressions.cpp)
set_tests_properties(nerve_cpp_algebra_boundary_regressions PROPERTIES LABELS "algebra;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algebra_complex_regressions cpp/algebra_complex_regressions.cpp)
set_tests_properties(nerve_cpp_algebra_complex_regressions PROPERTIES LABELS "algebra;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algebra_geometry_regressions cpp/algebra_geometry_regressions.cpp)
set_tests_properties(nerve_cpp_algebra_geometry_regressions PROPERTIES LABELS "algebra;geometry;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algebra_operations_regressions cpp/algebra_operations_regressions.cpp)
set_tests_properties(nerve_cpp_algebra_operations_regressions PROPERTIES LABELS "algebra;operations;cpu;regression")
nerve_add_cpp_test(nerve_cpp_algebra_gpu_regressions cpp/algebra_gpu_regressions.cpp)
set_tests_properties(nerve_cpp_algebra_gpu_regressions PROPERTIES LABELS "algebra;gpu;cpu;regression")

# Sheaf tests
nerve_add_cpp_test(nerve_cpp_sheaf_regressions cpp/sheaf_regressions.cpp)
set_tests_properties(nerve_cpp_sheaf_regressions PROPERTIES LABELS "sheaf;cpu;regression")
nerve_add_cpp_test(nerve_cpp_sheaf_morphism_regressions cpp/sheaf_morphism_regressions.cpp)
set_tests_properties(nerve_cpp_sheaf_morphism_regressions PROPERTIES LABELS "sheaf;morphism;cpu;regression")

# Streaming detail tests
nerve_add_cpp_test(nerve_cpp_streaming_core_regressions cpp/streaming_core_regressions.cpp)
set_tests_properties(nerve_cpp_streaming_core_regressions PROPERTIES LABELS "streaming;core;cpu;regression")
nerve_add_cpp_test(nerve_cpp_streaming_zigzag_regressions cpp/streaming_zigzag_regressions.cpp)
set_tests_properties(nerve_cpp_streaming_zigzag_regressions PROPERTIES LABELS "streaming;zigzag;cpu;regression")
nerve_add_cpp_test(nerve_cpp_streaming_laplacian_regressions cpp/streaming_laplacian_regressions.cpp)
set_tests_properties(nerve_cpp_streaming_laplacian_regressions PROPERTIES LABELS "streaming;laplacian;cpu;regression")
nerve_add_cpp_test(nerve_cpp_streaming_lockfree_regressions cpp/streaming_lockfree_regressions.cpp)
set_tests_properties(nerve_cpp_streaming_lockfree_regressions PROPERTIES LABELS "streaming;lockfree;cpu;regression")
nerve_add_cpp_test(nerve_cpp_streaming_windowed_ph_detail_regressions cpp/streaming_windowed_ph_detail_regressions.cpp)
set_tests_properties(nerve_cpp_streaming_windowed_ph_detail_regressions PROPERTIES LABELS "streaming;windowed;cpu;regression")

# Persistence detail tests
nerve_add_cpp_test(nerve_cpp_persistence_utils_regressions cpp/persistence_utils_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_utils_regressions PROPERTIES LABELS "persistence;utils;cpu;regression")
nerve_add_cpp_test(nerve_cpp_avx512_optimizer_regressions cpp/avx512_optimizer_regressions.cpp)
set_tests_properties(nerve_cpp_avx512_optimizer_regressions PROPERTIES LABELS "persistence;avx512;cpu;regression")
nerve_add_cpp_test(nerve_cpp_exact_engine_fast_regressions cpp/exact_engine_fast_regressions.cpp)
set_tests_properties(nerve_cpp_exact_engine_fast_regressions PROPERTIES LABELS "persistence;exact;fast;cpu;regression")
nerve_add_cpp_test(nerve_cpp_parallel_ph_regression cpp/parallel_ph_regression.cpp)
set_tests_properties(nerve_cpp_parallel_ph_regression PROPERTIES LABELS "persistence;parallel;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_core_vram_regressions cpp/persistence_core_vram_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_core_vram_regressions PROPERTIES LABELS "persistence;vram;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_core_flood_complex_regressions cpp/persistence_core_flood_complex_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_core_flood_complex_regressions PROPERTIES LABELS "persistence;flood;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_core_per_dimension_regressions cpp/persistence_core_per_dimension_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_core_per_dimension_regressions PROPERTIES LABELS "persistence;per-dimension;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_memory_regressions cpp/persistence_memory_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_memory_regressions PROPERTIES LABELS "persistence;memory;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_streaming_vr_regressions cpp/persistence_streaming_vr_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_streaming_vr_regressions PROPERTIES LABELS "persistence;streaming;vr;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_accelerated_determinism_regressions cpp/persistence_accelerated_determinism_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_accelerated_determinism_regressions PROPERTIES LABELS "persistence;accelerated;determinism;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_accelerated_allocator_regressions cpp/persistence_accelerated_allocator_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_accelerated_allocator_regressions PROPERTIES LABELS "persistence;accelerated;allocator;cpu;regression")
nerve_add_cpp_test(nerve_cpp_persistence_accelerated_validation_regressions cpp/persistence_accelerated_validation_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_accelerated_validation_regressions PROPERTIES LABELS "persistence;accelerated;validation;cpu;regression")

# Metrics detail tests
nerve_add_cpp_test(nerve_cpp_metrics_frechet_regressions cpp/metrics_frechet_regressions.cpp)
set_tests_properties(nerve_cpp_metrics_frechet_regressions PROPERTIES LABELS "metrics;frechet;cpu;regression")
nerve_add_cpp_test(nerve_cpp_metrics_general_regressions cpp/metrics_general_regressions.cpp)
set_tests_properties(nerve_cpp_metrics_general_regressions PROPERTIES LABELS "metrics;general;cpu;regression")
nerve_add_cpp_test(nerve_cpp_metrics_matrix_regressions cpp/metrics_matrix_regressions.cpp)
set_tests_properties(nerve_cpp_metrics_matrix_regressions PROPERTIES LABELS "metrics;matrix;cpu;regression")
nerve_add_cpp_test(nerve_cpp_metrics_wasserstein_detail_regressions cpp/metrics_wasserstein_detail_regressions.cpp)
set_tests_properties(nerve_cpp_metrics_wasserstein_detail_regressions PROPERTIES LABELS "metrics;wasserstein;cpu;regression")

# IO detail tests
nerve_add_cpp_test(nerve_cpp_io_detail_regressions cpp/io_detail_regressions.cpp)
set_tests_properties(nerve_cpp_io_detail_regressions PROPERTIES LABELS "io;detail;cpu;regression")

# NN detail tests
nerve_add_cpp_test(nerve_cpp_nn_detail_regressions cpp/nn_detail_regressions.cpp)
set_tests_properties(nerve_cpp_nn_detail_regressions PROPERTIES LABELS "nn;detail;cpu;regression")

# Probabilistic detail tests
nerve_add_cpp_test(nerve_cpp_probabilistic_detail_regressions cpp/probabilistic_detail_regressions.cpp)
set_tests_properties(nerve_cpp_probabilistic_detail_regressions PROPERTIES LABELS "probabilistic;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_probabilistic_simd_regressions cpp/probabilistic_simd_regressions.cpp)
set_tests_properties(nerve_cpp_probabilistic_simd_regressions PROPERTIES LABELS "probabilistic;simd;cpu;regression")

# Runtime detail tests
nerve_add_cpp_test(nerve_cpp_runtime_detail_regressions cpp/runtime_detail_regressions.cpp)
set_tests_properties(nerve_cpp_runtime_detail_regressions PROPERTIES LABELS "runtime;detail;cpu;regression")

# Filtration detail tests
nerve_add_cpp_test(nerve_cpp_filtration_level_set_regressions cpp/filtration_level_set_regressions.cpp)
set_tests_properties(nerve_cpp_filtration_level_set_regressions PROPERTIES LABELS "filtration;level-set;cpu;regression")
nerve_add_cpp_test(nerve_cpp_filtration_vr_detail_regressions cpp/filtration_vr_detail_regressions.cpp)
set_tests_properties(nerve_cpp_filtration_vr_detail_regressions PROPERTIES LABELS "filtration;vr;detail;cpu;regression")

# Distributed detail tests
nerve_add_cpp_test(nerve_cpp_distributed_detail_regressions cpp/distributed_detail_regressions.cpp)
set_tests_properties(nerve_cpp_distributed_detail_regressions PROPERTIES LABELS "distributed;detail;cpu;regression")

# Determinism tests
nerve_add_cpp_test(nerve_cpp_determinism_detail_regressions cpp/determinism_detail_regressions.cpp)
set_tests_properties(nerve_cpp_determinism_detail_regressions PROPERTIES LABELS "determinism;detail;cpu;regression")

# Core/error tests
nerve_add_cpp_test(nerve_cpp_thread_affinity_regressions cpp/thread_affinity_regressions.cpp)
set_tests_properties(nerve_cpp_thread_affinity_regressions PROPERTIES LABELS "core;threading;cpu;regression")
nerve_add_cpp_test(nerve_cpp_compact_summary_extensions_regressions cpp/compact_summary_extensions_regressions.cpp)
set_tests_properties(nerve_cpp_compact_summary_extensions_regressions PROPERTIES LABELS "summary;extensions;cpu;regression")
nerve_add_cpp_test(nerve_cpp_error_event_extensions_regressions cpp/error_event_extensions_regressions.cpp)
set_tests_properties(nerve_cpp_error_event_extensions_regressions PROPERTIES LABELS "errors;events;cpu;regression")
nerve_add_cpp_test(nerve_cpp_configurable_error_system_regressions cpp/configurable_error_system_regressions.cpp)
set_tests_properties(nerve_cpp_configurable_error_system_regressions PROPERTIES LABELS "errors;configurable;cpu;regression")
nerve_add_cpp_test(nerve_cpp_precision_policy_regressions cpp/precision_policy_regressions.cpp)
set_tests_properties(nerve_cpp_precision_policy_regressions PROPERTIES LABELS "precision;policy;cpu;regression")

# Cohomology detail tests
nerve_add_cpp_test(nerve_cpp_cohomology_involuted_regressions cpp/cohomology_involuted_regressions.cpp)
set_tests_properties(nerve_cpp_cohomology_involuted_regressions PROPERTIES LABELS "persistence;cohomology;involuted;cpu;regression")
nerve_add_cpp_test(nerve_cpp_cohomology_persistent_ops_detail_regressions cpp/cohomology_persistent_ops_detail_regressions.cpp)
set_tests_properties(nerve_cpp_cohomology_persistent_ops_detail_regressions PROPERTIES LABELS "persistence;cohomology;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_cohomology_rref_regressions cpp/cohomology_rref_regressions.cpp)
set_tests_properties(nerve_cpp_cohomology_rref_regressions PROPERTIES LABELS "persistence;cohomology;rref;cpu;regression")

# Reduction detail tests
nerve_add_cpp_test(nerve_cpp_reduction_distributed_regressions cpp/reduction_distributed_regressions.cpp)
set_tests_properties(nerve_cpp_reduction_distributed_regressions PROPERTIES LABELS "persistence;reduction;distributed;cpu;regression")
nerve_add_cpp_test(nerve_cpp_reduction_hypha_regressions cpp/reduction_hypha_regressions.cpp)
set_tests_properties(nerve_cpp_reduction_hypha_regressions PROPERTIES LABELS "persistence;reduction;hypha;cpu;regression")
nerve_add_cpp_test(nerve_cpp_reduction_mpi_detail_regressions cpp/reduction_mpi_detail_regressions.cpp)
set_tests_properties(nerve_cpp_reduction_mpi_detail_regressions PROPERTIES LABELS "persistence;reduction;mpi;cpu;regression")

# VR detail tests
nerve_add_cpp_test(nerve_cpp_vr_h1_reduction_regressions cpp/vr_h1_reduction_regressions.cpp)
set_tests_properties(nerve_cpp_vr_h1_reduction_regressions PROPERTIES LABELS "persistence;vr;h1;cpu;regression")

# Accelerated detail tests
nerve_add_cpp_test(nerve_cpp_accelerated_gpu_reduction_regressions cpp/accelerated_gpu_reduction_regressions.cpp)
set_tests_properties(nerve_cpp_accelerated_gpu_reduction_regressions PROPERTIES LABELS "persistence;accelerated;gpu;cpu;regression")
nerve_add_cpp_test(nerve_cpp_accelerated_error_handling_regressions cpp/accelerated_error_handling_regressions.cpp)
set_tests_properties(nerve_cpp_accelerated_error_handling_regressions PROPERTIES LABELS "persistence;accelerated;errors;cpu;regression")
nerve_add_cpp_test(nerve_cpp_acceleration_runtime_engine_regressions cpp/acceleration_runtime_engine_regressions.cpp)
set_tests_properties(nerve_cpp_acceleration_runtime_engine_regressions PROPERTIES LABELS "persistence;acceleration;runtime;cpu;regression")
nerve_add_cpp_test(nerve_cpp_gpu_memory_monitor_regressions cpp/gpu_memory_monitor_regressions.cpp)
set_tests_properties(nerve_cpp_gpu_memory_monitor_regressions PROPERTIES LABELS "gpu;memory;monitor;cpu;regression")
nerve_add_cpp_test(nerve_cpp_gpu_hybrid_tuning_regressions cpp/gpu_hybrid_tuning_regressions.cpp)
set_tests_properties(nerve_cpp_gpu_hybrid_tuning_regressions PROPERTIES LABELS "gpu;hybrid;tuning;cpu;regression")

# Adaptive acceleration detail tests
nerve_add_cpp_test(nerve_cpp_adaptive_selector_calibration_regressions cpp/adaptive_selector_calibration_regressions.cpp)
set_tests_properties(nerve_cpp_adaptive_selector_calibration_regressions PROPERTIES LABELS "persistence;adaptive;calibration;cpu;regression")
nerve_add_cpp_test(nerve_cpp_representative_cycles_detail_regressions cpp/representative_cycles_detail_regressions.cpp)
set_tests_properties(nerve_cpp_representative_cycles_detail_regressions PROPERTIES LABELS "persistence;adaptive;cycles;cpu;regression")

# Kernels detail tests
nerve_add_cpp_test(nerve_cpp_ph4_summary_regressions cpp/ph4_summary_regressions.cpp)
set_tests_properties(nerve_cpp_ph4_summary_regressions PROPERTIES LABELS "persistence;kernels;ph4;cpu;regression")
nerve_add_cpp_test(nerve_cpp_thread_block_cluster_regressions cpp/thread_block_cluster_regressions.cpp)
set_tests_properties(nerve_cpp_thread_block_cluster_regressions PROPERTIES LABELS "persistence;kernels;cluster;cpu;regression")

# Gradient tests
nerve_add_cpp_test(nerve_cpp_persistence_gradient_detail_regressions cpp/persistence_gradient_detail_regressions.cpp)
set_tests_properties(nerve_cpp_persistence_gradient_detail_regressions PROPERTIES LABELS "persistence;gradient;detail;cpu;regression")

# Torch detail tests
nerve_add_cpp_test(nerve_cpp_torch_boundary_matrix_detail_regressions cpp/torch_boundary_matrix_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_boundary_matrix_detail_regressions PROPERTIES LABELS "torch;boundary;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_torch_filtration_detail_regressions cpp/torch_filtration_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_filtration_detail_regressions PROPERTIES LABELS "torch;filtration;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_torch_mapper_detail_regressions cpp/torch_mapper_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_mapper_detail_regressions PROPERTIES LABELS "torch;mapper;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_torch_simplex_tree_detail_regressions cpp/torch_simplex_tree_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_simplex_tree_detail_regressions PROPERTIES LABELS "torch;simplex;tree;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_torch_ml_detail_regressions cpp/torch_ml_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_ml_detail_regressions PROPERTIES LABELS "torch;ml;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_torch_autograd_detail_regressions cpp/torch_autograd_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_autograd_detail_regressions PROPERTIES LABELS "torch;autograd;detail;cpu;regression")
nerve_add_cpp_test(nerve_cpp_torch_library_detail_regressions cpp/torch_library_detail_regressions.cpp)
set_tests_properties(nerve_cpp_torch_library_detail_regressions PROPERTIES LABELS "torch;library;detail;cpu;regression")

# Encoder / serialization / distributed detail tests
nerve_add_cpp_test(nerve_cpp_encoder_simd_regressions cpp/encoder_simd_regressions.cpp)
set_tests_properties(nerve_cpp_encoder_simd_regressions PROPERTIES LABELS "encoders;simd;cpu;regression")
nerve_add_cpp_test(nerve_cpp_ph5_ph6_schema_registry_regressions cpp/ph5_ph6_schema_registry_regressions.cpp)
set_tests_properties(nerve_cpp_ph5_ph6_schema_registry_regressions PROPERTIES LABELS "serialization;schema;cpu;regression")
nerve_add_cpp_test(nerve_cpp_distributed_nvshmem_regressions cpp/distributed_nvshmem_regressions.cpp)
set_tests_properties(nerve_cpp_distributed_nvshmem_regressions PROPERTIES LABELS "distributed;nvshmem;cpu;regression")
nerve_add_cpp_test(nerve_cpp_numa_completion_regressions cpp/numa_completion_regressions.cpp)
set_tests_properties(nerve_cpp_numa_completion_regressions PROPERTIES LABELS "memory;numa;cpu;regression")
nerve_add_cpp_test(nerve_cpp_error_cuda_detail_regressions cpp/error_cuda_detail_regressions.cpp)
set_tests_properties(nerve_cpp_error_cuda_detail_regressions PROPERTIES LABELS "cuda;errors;detail;cpu;regression")

# Internal implementation tests

nerve_add_cpp_test(nerve_cpp_internal_persistence_vr cpp/internal_persistence_vr_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_persistence_core cpp/internal_persistence_core_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_persistence_kernels cpp/internal_persistence_kernels_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_persistence_accelerated cpp/internal_persistence_accelerated_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_persistence_cuda cpp/internal_persistence_cuda_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_gpu cpp/internal_gpu_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_streaming cpp/internal_streaming_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_graphs cpp/internal_graphs_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_algebra cpp/internal_algebra_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_metrics cpp/internal_metrics_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_filtration cpp/internal_filtration_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_core cpp/internal_core_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_serialization cpp/internal_serialization_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_encoders cpp/internal_encoders_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_autodiff cpp/internal_autodiff_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_sheaf cpp/internal_sheaf_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_spectral cpp/internal_spectral_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_probabilistic cpp/internal_probabilistic_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_distributed_extra cpp/internal_distributed_extra_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_optimization cpp/internal_optimization_test.cpp)
nerve_add_cpp_test(nerve_cpp_internal_torch_extra cpp/internal_torch_extra_test.cpp)

if(Python3_Interpreter_FOUND)
    function(nerve_add_python_test target)
        cmake_parse_arguments(ARG "" "MARKER;SHARD_INDEX;SHARD_COUNT" "LABELS;ENVIRONMENT" ${ARGN})
        set(NERVE_PYTEST_COMMAND
            ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_SOURCE_DIR}/tests/python
        )
        if(ARG_MARKER)
            list(APPEND NERVE_PYTEST_COMMAND -m ${ARG_MARKER})
        endif()
        if(DEFINED ARG_SHARD_INDEX)
            list(APPEND NERVE_PYTEST_COMMAND --nerve-shard-index=${ARG_SHARD_INDEX})
        endif()
        if(DEFINED ARG_SHARD_COUNT)
            list(APPEND NERVE_PYTEST_COMMAND --nerve-shard-count=${ARG_SHARD_COUNT})
        endif()
        add_test(NAME ${target} COMMAND ${NERVE_PYTEST_COMMAND})
        set_tests_properties(${target} PROPERTIES
            LABELS "${ARG_LABELS}"
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;${ARG_ENVIRONMENT}"
        )
    endfunction()

    function(nerve_add_matrix_manifest target backend labels)
        set(NERVE_MATRIX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${target}.json")
        add_test(
            NAME ${target}
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/tools/test_matrix.py
                    --shard-index 0
                    --shard-count 1
                    --label ${backend}
                    --output ${NERVE_MATRIX_OUTPUT}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(${target} PROPERTIES LABELS "${labels}")
    endfunction()

    add_test(
        NAME nerve_quality_all
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/quality.py --check all
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(nerve_quality_all PROPERTIES LABELS "quality;static;operators")

    nerve_add_matrix_manifest(nerve_generated_matrix_manifest cpu "generated;cpu")
    nerve_add_matrix_manifest(nerve_generated_matrix_cuda cuda "generated;cuda")
    nerve_add_matrix_manifest(nerve_generated_matrix_xpu xpu "generated;xpu")

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE NERVE_PYTEST_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NERVE_PYTEST_IMPORT_RESULT EQUAL 0)
        nerve_add_python_test(
            nerve_pytest_generated
            MARKER "generated"
            SHARD_INDEX 0
            SHARD_COUNT 1
            LABELS "generated;python"
        )
        nerve_add_python_test(
            nerve_pytest_gradient
            MARKER "gradient or autograd"
            SHARD_INDEX 0
            SHARD_COUNT 1
            LABELS "gradient;autograd;python"
        )
        nerve_add_python_test(
            nerve_pytest_distributed
            MARKER "distributed"
            SHARD_INDEX 0
            SHARD_COUNT 1
            LABELS "distributed;python"
        )
        nerve_add_python_test(
            nerve_pytest_performance
            MARKER "performance"
            SHARD_INDEX 0
            SHARD_COUNT 1
            LABELS "performance;python"
        )
        nerve_add_python_test(
            nerve_pytest_quality
            MARKER "quality"
            SHARD_INDEX 0
            SHARD_COUNT 1
            LABELS "quality;python"
        )
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch"
            RESULT_VARIABLE NERVE_TORCH_IMPORT_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(NERVE_TORCH_IMPORT_RESULT EQUAL 0)
            nerve_add_python_test(
                nerve_pytest_torch
                MARKER "torch"
                SHARD_INDEX 0
                SHARD_COUNT 1
                LABELS "torch;gradient;python"
            )
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
                RESULT_VARIABLE NERVE_TORCH_CUDA_AVAILABLE
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(NERVE_TORCH_CUDA_AVAILABLE EQUAL 0)
                nerve_add_python_test(
                    nerve_pytest_cuda
                    MARKER "cuda"
                    SHARD_INDEX 0
                    SHARD_COUNT 1
                    LABELS "cuda;cuda-hardware;gradient;python"
                    ENVIRONMENT "NERVE_TEST_CUDA=1"
                )
            endif()
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch; raise SystemExit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() else 1)"
                RESULT_VARIABLE NERVE_TORCH_XPU_AVAILABLE
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(NERVE_TORCH_XPU_AVAILABLE EQUAL 0)
                nerve_add_python_test(
                    nerve_pytest_xpu
                    MARKER "xpu"
                    SHARD_INDEX 0
                    SHARD_COUNT 1
                    LABELS "xpu;gradient;python"
                    ENVIRONMENT "NERVE_TEST_XPU=1"
                )
            endif()
        endif()
    else()
        if(NERVE_ALLOW_MISSING_PYTEST)
            message(STATUS "Skipping Python generated tests because pytest is missing")
        else()
            message(FATAL_ERROR
                "BUILD_TESTS=ON requires pytest for the selected Python interpreter "
                "(${Python3_EXECUTABLE}). Install pytest or set Python3_EXECUTABLE to an "
                "interpreter with pytest. Set NERVE_ALLOW_MISSING_PYTEST=ON only for "
                "explicit C++-only test jobs."
            )
        endif()
    endif()
endif()
