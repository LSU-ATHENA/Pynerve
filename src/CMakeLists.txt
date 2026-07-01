cmake_minimum_required(VERSION 3.20)
project(NerveCore VERSION 0.9.7 LANGUAGES CXX)

# Early checks for first-time developers
if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "No C++ compiler found. Nerve requires a C++20 compiler: GCC >= 11 or Clang >= 14.
  Install GCC: g++ (GCC) 11 or later
  Install Clang: clang++ 14 or later
  On macOS: xcode-select --install")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11")
    message(FATAL_ERROR "GCC >= 11 required (found ${CMAKE_CXX_COMPILER_VERSION}). Nerve uses C++20 features not available in older versions.")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
    message(FATAL_ERROR "Clang >= 14 required (found ${CMAKE_CXX_COMPILER_VERSION}). Nerve uses C++20 features not available in older versions.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ninja is strongly recommended
if(NOT DEFINED CMAKE_GENERATOR OR CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    message(STATUS "Tip: For faster builds, use Ninja: cmake -G Ninja ...")
endif()
if(ENABLE_PYTORCH)
    find_package(Torch QUIET)
    set(NERVE_HAS_TORCH ${Torch_FOUND})
else()
    set(NERVE_HAS_TORCH OFF)
endif()
if(NERVE_HAS_TORCH)
    message(STATUS "PyTorch found - Enabling ML integration")
    set(NERVE_TORCH_SOURCES
        torch/torch_library.cpp
        torch/vietoris_rips_torch.cpp
        torch/diagram_operations_torch.cpp
        torch/autograd_torch.cpp
        torch/simplex_tree.cpp
        torch/simplex_tree_ops.cpp
        torch/simplex_tree_queries.cpp
        torch/boundary_matrix.cpp
        torch/filtration.cpp
        torch/persistence_diagram.cpp
        torch/ml_vectorization.cpp
        torch/ml_vectorization_ops.cpp
        torch/ml_statistics.cpp
        torch/ml_kernels.cpp
        torch/mapper.cpp
    )
else()
    message(STATUS "PyTorch not found - Building without ML integration")
    set(NERVE_TORCH_SOURCES "")
endif()
set(NERVE_TORCH_SOURCES ${NERVE_TORCH_SOURCES} PARENT_SCOPE)
if(DEFINED ENABLE_CUDA AND NOT ENABLE_CUDA)
    set(NERVE_HAS_CUDA OFF)
    message(STATUS "CUDA disabled by ENABLE_CUDA=OFF")
else()
    find_package(CUDAToolkit QUIET)
    set(NERVE_HAS_CUDA ${CUDAToolkit_FOUND})
    if(NERVE_HAS_CUDA)
        if(DEFINED NERVE_CUDA_REQUIRED_VERSION_MAJOR AND DEFINED NERVE_CUDA_REQUIRED_VERSION_MINOR)
            if(CUDAToolkit_VERSION VERSION_LESS "${NERVE_CUDA_REQUIRED_VERSION_MAJOR}.${NERVE_CUDA_REQUIRED_VERSION_MINOR}")
                message(FATAL_ERROR "Nerve requires CUDA >= ${NERVE_CUDA_REQUIRED_VERSION_MAJOR}.${NERVE_CUDA_REQUIRED_VERSION_MINOR}; found ${CUDAToolkit_VERSION}")
            endif()
        endif()
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "${NERVE_GPU_BASE_ARCHS};${NERVE_GPU_OPT_ARCHS}" CACHE STRING
                "CUDA architectures for Nerve targets" FORCE)
        endif()
        # Validate that the installed nvcc supports the requested architectures
        # BEFORE enabling the CUDA language to avoid unsupported-arch compile errors.
        execute_process(
            COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
            OUTPUT_VARIABLE _NERVE_SRC_CUDA_AVAIL
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_NERVE_SRC_CUDA_AVAIL)
            string(REPLACE "\n" ";" _NERVE_SRC_CUDA_AVAIL "${_NERVE_SRC_CUDA_AVAIL}")
            set(_NERVE_SRC_FILTERED "")
            foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
                string(STRIP "${_arch}" _arch)
                if("compute_${_arch}" IN_LIST _NERVE_SRC_CUDA_AVAIL)
                    list(APPEND _NERVE_SRC_FILTERED "${_arch}")
                endif()
            endforeach()
            if(_NERVE_SRC_FILTERED)
                set(CMAKE_CUDA_ARCHITECTURES "${_NERVE_SRC_FILTERED}" CACHE STRING
                    "CUDA architectures (auto-validated)" FORCE)
                enable_language(CUDA)
                message(STATUS "CUDA found - Enabling GPU support (${CUDAToolkit_VERSION})")
                message(STATUS "CUDA architectures: ${_NERVE_SRC_FILTERED}")
            else()
                message(WARNING
                    "No GPU architecture in ${CMAKE_CUDA_ARCHITECTURES} is supported by "
                    "the installed CUDA toolkit. Skipping CUDA sources. "
                    "Set NERVE_GPU_BASE_ARCHS to a supported value.")
                set(NERVE_HAS_CUDA OFF CACHE BOOL "CUDA available" FORCE)
            endif()
        else()
            enable_language(CUDA)
            message(STATUS "CUDA found - Enabling GPU support (${CUDAToolkit_VERSION})")
            message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        message(STATUS "CUDA not found - Building without GPU acceleration")
    endif()
endif()
if(NOT DEFINED ENABLE_MPI)
    option(ENABLE_MPI "Enable distributed/MPI reduction features" OFF)
endif()
if(ENABLE_MPI)
    find_package(MPI QUIET)
    set(NERVE_HAS_MPI ${MPI_FOUND})
    if(NERVE_HAS_MPI)
        message(STATUS "MPI found - Enabling distributed computing")
    else()
        message(STATUS "MPI requested but not found - Building without distributed support")
    endif()
else()
    set(NERVE_HAS_MPI OFF)
    message(STATUS "MPI disabled - Building without distributed support")
endif()
if(NOT DEFINED ENABLE_EIGEN)
    option(ENABLE_EIGEN "Enable Eigen3-backed gradient and sheaf Laplacian components" OFF)
endif()
if(ENABLE_EIGEN)
    find_package(Eigen3 QUIET)
    set(NERVE_HAS_EIGEN3 ${Eigen3_FOUND})
    if(NERVE_HAS_EIGEN3)
        message(STATUS "Eigen3 found - Enabling Eigen-backed gradient and sheaf Laplacian components")
    else()
        message(STATUS "Eigen3 requested but not found - Building without Eigen-backed components")
    endif()
else()
    set(NERVE_HAS_EIGEN3 OFF)
    message(STATUS "Eigen3 disabled - Building without Eigen-backed components")
endif()

find_package(OpenMP QUIET)
set(NERVE_HAS_OPENMP ${OpenMP_CXX_FOUND})
if(NERVE_HAS_OPENMP)
    message(STATUS "OpenMP found - Enabling OpenMP-parallelized CPU paths")
else()
    message(STATUS "OpenMP not found - Building CPU paths without OpenMP pragmas")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/simd.cmake)
nerve_configure_x86_simd_support()

if(NOT DEFINED ENABLE_NUMA)
    option(ENABLE_NUMA "Enable NUMA memory optimizations" ON)
endif()
if(ENABLE_NUMA)
    find_library(NUMA_LIB numa)
    find_path(NUMA_INCLUDE numa.h)
    if(NUMA_LIB AND NUMA_INCLUDE)
        set(NERVE_HAS_NUMA ON)
        message(STATUS "NUMA found - Enabling memory optimizations")
    else()
        set(NERVE_HAS_NUMA OFF)
        message(STATUS "NUMA requested but not found - Building without NUMA optimizations")
    endif()
else()
    set(NERVE_HAS_NUMA OFF)
    message(STATUS "NUMA disabled - Building without NUMA optimizations")
endif()

nerve_detect_avx512_codegen()

option(NERVE_ENABLE_VALIDATION_BENCHMARKS
    "Build validation benchmark sources in nerve_core"
    OFF)
option(NERVE_ENABLE_BENCHMARK_SOURCES
    "Build benchmark harness sources in nerve_core"
    OFF)
option(NERVE_ENABLE_CUDNN_COMPONENTS
    "Build cuDNN-dependent compression components in nerve_core"
    OFF)
option(NERVE_ENABLE_CUDA_COMPONENTS
    "Build CUDA kernel/component sources in nerve_core"
    OFF)
option(NERVE_ENABLE_EXTENDED_CUDA_COMPONENTS
    "Build extended CUDA graph/filtration/encoder components in nerve_core"
    OFF)
option(NERVE_ENABLE_ADVANCED_DIFFERENTIABLE_COMPONENTS
    "Build advanced differentiable persistence components in nerve_core"
    OFF)
option(NERVE_ENABLE_AVX512_COMPONENTS
    "Build AVX-512 optimized component sources in nerve_core"
    OFF)
option(NERVE_ENABLE_AVX512_STAGED_COMPONENTS
    "Build staged AVX-512 component sources in nerve_core"
    OFF)

# Production builds should not compile legacy benchmark/validation modules
# unless benchmark mode is explicitly enabled at the top level.
if(DEFINED BUILD_BENCHMARKS AND NOT BUILD_BENCHMARKS)
    if(NERVE_ENABLE_VALIDATION_BENCHMARKS OR NERVE_ENABLE_BENCHMARK_SOURCES)
        message(STATUS "BUILD_BENCHMARKS=OFF - forcing validation/benchmark source groups OFF")
    endif()
    set(NERVE_ENABLE_VALIDATION_BENCHMARKS OFF CACHE BOOL
        "Build validation benchmark sources in nerve_core" FORCE)
    set(NERVE_ENABLE_BENCHMARK_SOURCES OFF CACHE BOOL
        "Build benchmark harness sources in nerve_core" FORCE)
endif()

# Keep unstable advanced groups out of default production builds unless
# top-level advanced mode is explicitly enabled.
if(DEFINED ENABLE_ADVANCED_FEATURES AND NOT ENABLE_ADVANCED_FEATURES)
    if(NERVE_ENABLE_ADVANCED_DIFFERENTIABLE_COMPONENTS
       OR NERVE_ENABLE_AVX512_STAGED_COMPONENTS)
        message(STATUS "ENABLE_ADVANCED_FEATURES=OFF - forcing advanced/staged source groups OFF")
    endif()
    set(NERVE_ENABLE_ADVANCED_DIFFERENTIABLE_COMPONENTS OFF CACHE BOOL
        "Build advanced differentiable persistence components in nerve_core" FORCE)
    set(NERVE_ENABLE_AVX512_STAGED_COMPONENTS OFF CACHE BOOL
        "Build staged AVX-512 component sources in nerve_core" FORCE)
endif()

nerve_validate_simd_mode()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/source_groups.cmake)
set(NERVE_ALL_SOURCES ${NERVE_CORE_SOURCES})
if(NOT NERVE_HAS_CUDA)
    list(REMOVE_ITEM NERVE_ALL_SOURCES
        persistence/adaptive_acceleration/cuda/hybrid_algorithms.cpp
        persistence/adaptive_acceleration/cuda/tensor_core_optimizer.cpp
        persistence/kernels/thread_block_cluster.cpp
        persistence/core/vram_efficient_algorithms.cpp
        metrics/hungarian/assignment_hungarian_gpu_impl.cpp
        spectral/laplacian_gpu_impl.cpp
        streaming/gpu/streaming_gpu_impl.cpp
        encoders/encoder_tensor_cores_benchmark.cpp
        gpu/tuner_nvidia_auto.cpp
        gpu/tuner_nvidia_auto_reporting.cpp
    )
endif()
if(NOT NERVE_HAS_MPI)
    list(REMOVE_ITEM NERVE_ALL_SOURCES
        persistence/distributed/mpi_distributed_ph.cpp
    )
endif()
if(NOT NERVE_HAS_CUDA OR NOT NERVE_HAS_MPI)
    list(REMOVE_ITEM NERVE_ALL_SOURCES
        formats/gpu_mpi_overlap.cpp
    )
endif()
if(NERVE_HAS_CUDA AND NERVE_ENABLE_CUDA_COMPONENTS)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_CUDA_SOURCES})
    list(LENGTH NERVE_CUDA_SOURCES NERVE_CUDA_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_CUDA_SOURCE_COUNT} CUDA source files")
    if(NERVE_ENABLE_EXTENDED_CUDA_COMPONENTS)
        list(APPEND NERVE_ALL_SOURCES ${NERVE_CUDA_EXTENDED_SOURCES})
        list(LENGTH NERVE_CUDA_EXTENDED_SOURCES NERVE_CUDA_EXTENDED_SOURCE_COUNT)
        message(STATUS "Added ${NERVE_CUDA_EXTENDED_SOURCE_COUNT} extended CUDA source files")
    else()
        message(STATUS "Extended CUDA sources disabled (NERVE_ENABLE_EXTENDED_CUDA_COMPONENTS=OFF)")
    endif()
elseif(NERVE_HAS_CUDA)
    message(STATUS "CUDA sources disabled (NERVE_ENABLE_CUDA_COMPONENTS=OFF)")
endif()

# Tile API depends on distance-kernel launch wrappers for device execution.
# Keep this backend compiled whenever CUDA is available, regardless of the
# broader optional CUDA component set.
if(NERVE_HAS_CUDA)
    list(FIND NERVE_ALL_SOURCES cuda/kernels/distance_kernels.cu NERVE_DISTANCE_KERNEL_IDX)
    if(NERVE_DISTANCE_KERNEL_IDX EQUAL -1)
        list(APPEND NERVE_ALL_SOURCES cuda/kernels/distance_kernels.cu)
        message(STATUS "Added mandatory CUDA distance kernel backend source")
    endif()
    foreach(NERVE_GRAPH_BACKEND_SOURCE
            graphs/graph_algorithms_gpu.cu
            graphs/graph_engine_gpu.cu
            graphs/graph_zigzag_gpu.cu)
        list(FIND NERVE_ALL_SOURCES ${NERVE_GRAPH_BACKEND_SOURCE} NERVE_GRAPH_KERNEL_IDX)
        if(NERVE_GRAPH_KERNEL_IDX EQUAL -1)
            list(APPEND NERVE_ALL_SOURCES ${NERVE_GRAPH_BACKEND_SOURCE})
            message(STATUS "Added mandatory CUDA graph backend source: ${NERVE_GRAPH_BACKEND_SOURCE}")
        endif()
    endforeach()
    foreach(NERVE_CLUSTER_BACKEND_SOURCE
            persistence/cuda/cluster_16_block.cu
            persistence/cuda/cluster_distributed_l2.cu
            persistence/cuda/cluster_tma_multicast.cu)
        list(FIND NERVE_ALL_SOURCES ${NERVE_CLUSTER_BACKEND_SOURCE} NERVE_CLUSTER_KERNEL_IDX)
        if(NERVE_CLUSTER_KERNEL_IDX EQUAL -1)
            list(APPEND NERVE_ALL_SOURCES ${NERVE_CLUSTER_BACKEND_SOURCE})
            message(STATUS "Added mandatory CUDA cluster backend source: ${NERVE_CLUSTER_BACKEND_SOURCE}")
        endif()
    endforeach()
    list(FIND NERVE_ALL_SOURCES persistence/cuda/error_mapping_cuda.cpp NERVE_CUDA_ERROR_MAPPING_IDX)
    if(NERVE_CUDA_ERROR_MAPPING_IDX EQUAL -1)
        list(APPEND NERVE_ALL_SOURCES persistence/cuda/error_mapping_cuda.cpp)
        message(STATUS "Added mandatory CUDA error mapping source")
    endif()
endif()

if(NERVE_HAS_TORCH)
    list(LENGTH NERVE_TORCH_SOURCES NERVE_TORCH_SOURCE_COUNT)
    list(LENGTH NERVE_PYTORCH_SOURCES NERVE_PYTORCH_SOURCE_COUNT)
    math(EXPR NERVE_TORCH_TOTAL_SOURCE_COUNT
        "${NERVE_TORCH_SOURCE_COUNT} + ${NERVE_PYTORCH_SOURCE_COUNT}")
    message(STATUS "Configured ${NERVE_TORCH_TOTAL_SOURCE_COUNT} PyTorch source files outside nerve_core")
endif()

if(NERVE_HAS_MPI)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_MPI_SOURCES})
    list(LENGTH NERVE_MPI_SOURCES NERVE_MPI_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_MPI_SOURCE_COUNT} MPI source files")
endif()
list(FIND NERVE_ALL_SOURCES distributed/mpi_communicator.cpp MPI_COMM_IDX)
if(MPI_COMM_IDX EQUAL -1)
    list(APPEND NERVE_ALL_SOURCES distributed/mpi_communicator.cpp)
    message(STATUS "MPI communicator source added (always compiled)")
endif()

if(NERVE_HAS_EIGEN3)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_EIGEN3_SOURCES})
    list(LENGTH NERVE_EIGEN3_SOURCES NERVE_EIGEN3_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_EIGEN3_SOURCE_COUNT} Eigen3 source files")
else()
    list(APPEND NERVE_ALL_SOURCES ${NERVE_NON_EIGEN_SOURCES})
    list(LENGTH NERVE_NON_EIGEN_SOURCES NERVE_NON_EIGEN_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_NON_EIGEN_SOURCE_COUNT} non-Eigen basic source files")
endif()

if(NERVE_HAS_NUMA)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_NUMA_SOURCES})
    list(LENGTH NERVE_NUMA_SOURCES NERVE_NUMA_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_NUMA_SOURCE_COUNT} NUMA source files")
endif()

if(NERVE_ENABLE_AVX512_CODEGEN AND NERVE_ENABLE_AVX512_COMPONENTS)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_AVX512_SOURCES})
    list(LENGTH NERVE_AVX512_SOURCES NERVE_AVX512_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_AVX512_SOURCE_COUNT} AVX-512 source files")
    if(NERVE_ENABLE_AVX512_STAGED_COMPONENTS)
        list(APPEND NERVE_ALL_SOURCES ${NERVE_AVX512_STAGED_SOURCES})
        list(LENGTH NERVE_AVX512_STAGED_SOURCES NERVE_AVX512_STAGED_SOURCE_COUNT)
        message(STATUS "Added ${NERVE_AVX512_STAGED_SOURCE_COUNT} staged AVX-512 source files")
    else()
        message(STATUS "Staged AVX-512 sources disabled (NERVE_ENABLE_AVX512_STAGED_COMPONENTS=OFF)")
    endif()
elseif(NERVE_HAS_AVX512 AND NOT NERVE_ENABLE_AVX512_CODEGEN)
    message(STATUS "AVX-512 detected but disabled for codegen (NERVE_SIMD=${NERVE_SIMD})")
elseif(NERVE_HAS_AVX512)
    message(STATUS "AVX-512 component sources disabled (NERVE_ENABLE_AVX512_COMPONENTS=OFF)")
endif()

if(NERVE_ENABLE_VALIDATION_BENCHMARKS)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_VALIDATION_SOURCES})
    list(LENGTH NERVE_VALIDATION_SOURCES NERVE_VALIDATION_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_VALIDATION_SOURCE_COUNT} validation benchmark source files")
else()
    message(STATUS "Validation benchmark sources disabled (NERVE_ENABLE_VALIDATION_BENCHMARKS=OFF)")
endif()

if(NERVE_ENABLE_BENCHMARK_SOURCES)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_BENCHMARK_SOURCES})
    list(LENGTH NERVE_BENCHMARK_SOURCES NERVE_BENCHMARK_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_BENCHMARK_SOURCE_COUNT} benchmark source files")
else()
    message(STATUS "Benchmark sources disabled (NERVE_ENABLE_BENCHMARK_SOURCES=OFF)")
endif()

if(NERVE_ENABLE_ADVANCED_DIFFERENTIABLE_COMPONENTS)
    list(APPEND NERVE_ALL_SOURCES ${NERVE_ADVANCED_DIFFERENTIABLE_SOURCES})
    list(LENGTH NERVE_ADVANCED_DIFFERENTIABLE_SOURCES NERVE_ADVANCED_DIFFERENTIABLE_SOURCE_COUNT)
    message(STATUS "Added ${NERVE_ADVANCED_DIFFERENTIABLE_SOURCE_COUNT} advanced differentiable source files")
else()
    message(STATUS "Advanced differentiable sources disabled (NERVE_ENABLE_ADVANCED_DIFFERENTIABLE_COMPONENTS=OFF)")
endif()

if(NERVE_ENABLE_CUDNN_COMPONENTS AND NOT NERVE_HAS_CUDA)
    set(NERVE_HAS_CUDNN OFF)
    message(WARNING "NERVE_ENABLE_CUDNN_COMPONENTS=ON requires CUDA; cuDNN sources are skipped")
elseif(NERVE_ENABLE_CUDNN_COMPONENTS)
    find_path(NERVE_CUDNN_INCLUDE_DIR cudnn.h)
    if(NERVE_CUDNN_INCLUDE_DIR)
        set(NERVE_HAS_CUDNN ON)
        list(APPEND NERVE_ALL_SOURCES ${NERVE_CUDNN_SOURCES})
        list(LENGTH NERVE_CUDNN_SOURCES NERVE_CUDNN_SOURCE_COUNT)
        message(STATUS "Added ${NERVE_CUDNN_SOURCE_COUNT} cuDNN source files")
    else()
        set(NERVE_HAS_CUDNN OFF)
        message(WARNING "NERVE_ENABLE_CUDNN_COMPONENTS=ON but cudnn.h was not found; cuDNN sources are skipped")
    endif()
else()
    set(NERVE_HAS_CUDNN OFF)
    message(STATUS "cuDNN component sources disabled (NERVE_ENABLE_CUDNN_COMPONENTS=OFF)")
endif()

set(HAS_CUDA ${NERVE_HAS_CUDA})
set(HAS_PYTORCH ${NERVE_HAS_TORCH})
set(HAS_MPI ${NERVE_HAS_MPI})
set(HAS_EIGEN ${NERVE_HAS_EIGEN3})
set(HAS_NUMA ${NERVE_HAS_NUMA})
set(NERVE_CONFIG_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/include/nerve/config.hpp
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nerve/config.hpp.in
    ${NERVE_CONFIG_HEADER}
    @ONLY
)

foreach(NERVE_FEATURE_VAR
        NERVE_HAS_CUDA
        NERVE_HAS_TORCH
        NERVE_HAS_MPI
        NERVE_HAS_EIGEN3
        NERVE_HAS_OPENMP
        NERVE_HAS_NUMA)
    set(${NERVE_FEATURE_VAR} "${${NERVE_FEATURE_VAR}}" PARENT_SCOPE)
endforeach()
set(NERVE_CONFIG_HEADER "${NERVE_CONFIG_HEADER}" PARENT_SCOPE)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets.cmake)

message(STATUS "Nerve Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
list(LENGTH NERVE_ALL_SOURCES NERVE_TOTAL_SOURCE_COUNT)
message(STATUS "  Sources Included: ${NERVE_TOTAL_SOURCE_COUNT} files")
message(STATUS "  Features:")
message(STATUS "    CUDA: ${NERVE_HAS_CUDA}")
message(STATUS "    PyTorch: ${NERVE_HAS_TORCH}")
message(STATUS "    MPI: ${NERVE_HAS_MPI}")
message(STATUS "    Eigen3: ${NERVE_HAS_EIGEN3}")
message(STATUS "    NUMA: ${NERVE_HAS_NUMA}")
message(STATUS "    AVX-512 codegen available: ${NERVE_HAS_AVX512}")
message(STATUS "    AVX-512 codegen enabled: ${NERVE_ENABLE_AVX512_CODEGEN}")
