cmake_minimum_required(VERSION 3.20)
project(Nerve VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin|Windows")
    message(FATAL_ERROR "Nerve currently supports Linux, macOS, and Windows builds only")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Windows build detected - using nerve::sys platform abstraction layer")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|aarch64|ARM64")
    message(WARNING "Nerve Linux builds are tested on x86_64 and ARM64; other architectures may not work")
endif()

set(NERVE_CUDA_REQUIRED_VERSION_MAJOR 12 CACHE STRING "Required CUDA major version")
set(NERVE_CUDA_REQUIRED_VERSION_MINOR 4 CACHE STRING "Required CUDA minor version")
set(NERVE_GPU_BASE_ARCHS "75" CACHE STRING "Minimum supported NVIDIA GPU architectures")
set(NERVE_GPU_OPT_ARCHS "80;86;89;90" CACHE STRING "Additional NVIDIA GPU architectures")

find_program(NERVE_CCACHE_PROGRAM ccache)
if(NERVE_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${NERVE_CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${NERVE_CCACHE_PROGRAM})
endif()

option(NERVE_USE_MIMALLOC "Use mimalloc as the global allocator (recommended)" ON)
option(NERVE_USE_HUGEPAGES "Use hugepages for large memory pools" ON)
option(NERVE_USE_IO_URING "Use io_uring for async file IO (Linux 5.1+)" ON)

if(NERVE_USE_MIMALLOC)
    find_package(mimalloc 2.1 QUIET)
    if(mimalloc_FOUND)
        message(STATUS "mimalloc found - Using as global allocator")
        set(NERVE_HAS_MIMALLOC ON)
        add_compile_definitions(NERVE_HAS_MIMALLOC=1)
        if(TARGET mimalloc-static)
            set(NERVE_MIMALLOC_TARGET mimalloc-static)
        elseif(TARGET mimalloc)
            set(NERVE_MIMALLOC_TARGET mimalloc)
        endif()
    else()
        message(STATUS "mimalloc not found - Using system allocator")
        set(NERVE_HAS_MIMALLOC OFF)
    endif()
endif()

if(NERVE_USE_HUGEPAGES AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NERVE_HAS_HUGEPAGES ON)
    add_compile_definitions(NERVE_HAS_HUGEPAGES=1)
    message(STATUS "Hugepage support enabled")
else()
    set(NERVE_HAS_HUGEPAGES OFF)
endif()

if(NERVE_USE_IO_URING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LIBURING liburing QUIET)
    endif()
    if(LIBURING_FOUND)
        set(NERVE_HAS_IO_URING ON)
        add_compile_definitions(NERVE_HAS_IO_URING=1)
        message(STATUS "io_uring support enabled via liburing")
    else()
        set(NERVE_HAS_IO_URING OFF)
        message(STATUS "liburing not found - Using mmap for async IO")
    endif()
else()
    set(NERVE_HAS_IO_URING OFF)
endif()

# Build configuration
option(ENABLE_CUDA "Enable CUDA-accelerated features (GPU memory, kernels)" OFF)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NERVE_NATIVE_ARCH
    "Tune CPU code generation for the build host; disables portable binary output"
    OFF)

option(ENABLE_PH5 "Enable PH5 advanced features" OFF)
option(ENABLE_PH6 "Enable PH6 advanced features" OFF)
option(ENABLE_ADVANCED_FEATURES "Enable all advanced features" OFF)
option(ENABLE_GREEN_CONTEXTS "Enable CUDA Green Contexts (CUDA 13.1+)" OFF)

# Optional External Dependencies
option(ENABLE_PYTORCH "Enable PyTorch differentiable operations" ON)
option(ENABLE_MPI "Enable distributed/MPI reduction features" OFF)
option(ENABLE_EIGEN "Enable Eigen3-backed gradient and sheaf Laplacian components" ON)
option(ENABLE_NUMA "Enable NUMA memory optimizations" ON)
option(ENABLE_ALL "Enable all optional dependencies (CUDA, PyTorch, MPI, Eigen, NUMA)" OFF)

# Auto-validate CUDA architectures against installed toolkit before enabling.
find_program(NERVE_NVCC_PROGRAM nvcc)
if(NERVE_NVCC_PROGRAM AND ENABLE_CUDA)
    execute_process(
        COMMAND ${NERVE_NVCC_PROGRAM} --list-gpu-arch
        OUTPUT_VARIABLE _NERVE_CUDA_SUPPORTED_ARCHS
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_NERVE_CUDA_SUPPORTED_ARCHS)
        string(REPLACE "\n" ";" _NERVE_CUDA_AVAILABLE "${_NERVE_CUDA_SUPPORTED_ARCHS}")
        # Normalise: nvcc reports "compute_75", our list uses bare "75"
        set(_NERVE_CUDA_AVAILABLE_NORM "")
        foreach(_a IN LISTS _NERVE_CUDA_AVAILABLE)
            string(STRIP "${_a}" _a)
            string(REGEX REPLACE "^compute_" "" _a_norm "${_a}")
            list(APPEND _NERVE_CUDA_AVAILABLE_NORM "${_a_norm}")
        endforeach()
        set(_NERVE_CUDA_AVAILABLE "${_NERVE_CUDA_AVAILABLE_NORM}")
        set(_NERVE_CUDA_FILTERED "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(STRIP "${_arch}" _arch)
            if(_arch IN_LIST _NERVE_CUDA_AVAILABLE)
                list(APPEND _NERVE_CUDA_FILTERED "${_arch}")
            endif()
        endforeach()
        if(NOT _NERVE_CUDA_FILTERED AND NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "${NERVE_GPU_BASE_ARCHS};${NERVE_GPU_OPT_ARCHS}" CACHE STRING
                "CUDA architectures for Nerve" FORCE)
            foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
                string(STRIP "${_arch}" _arch)
                if(_arch IN_LIST _NERVE_CUDA_AVAILABLE)
                    list(APPEND _NERVE_CUDA_FILTERED "${_arch}")
                endif()
            endforeach()
        endif()
        if(_NERVE_CUDA_FILTERED)
            set(CMAKE_CUDA_ARCHITECTURES "${_NERVE_CUDA_FILTERED}" CACHE STRING
                "CUDA architectures for Nerve (auto-validated)" FORCE)
            message(STATUS "CUDA architectures: ${_NERVE_CUDA_FILTERED}")
        else()
            message(WARNING
                "No compatible CUDA GPU architectures found. "
                "Installed toolkit supports: ${_NERVE_CUDA_AVAILABLE}. "
                "Set NERVE_GPU_BASE_ARCHS to a supported value or install a newer CUDA toolkit. "
                "Disabling CUDA build.")
            set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA-accelerated features" FORCE)
        endif()
    endif()
elseif(NOT NERVE_NVCC_PROGRAM AND ENABLE_CUDA)
    message(WARNING "nvcc not found - disabling CUDA build")
    set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA-accelerated features" FORCE)
endif()

# Enable CUDA language only if still requested.
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Compiler flags
# GCC 15+ requires additional compatibility flags for C++20 template deduction
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "15.0")
    add_compile_options(-Wno-template-id-cdtor)
    message(STATUS "GCC >= 15 detected: adding -Wno-template-id-cdtor for C++20 compatibility")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -finline-functions")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xptxas=-O3 --expt-relaxed-constexpr")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} -fno-fast-math -fno-associative-math -fno-unsafe-math-optimizations -ffp-contract=off")
    endif()
    set(CMAKE_CUDA_FLAGS_RELEASE
        "${CMAKE_CUDA_FLAGS_RELEASE} --fmad=false --prec-div=true --prec-sqrt=true")
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS_RELEASE
            "${CMAKE_CUDA_FLAGS_RELEASE} --ftz=false")
        add_compile_definitions(NERVE_GPU_DETERMINISM=1)
        message(STATUS "GPU determinism enabled: RFA reductions, no fast-math, ftz disabled")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Dependencies
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# CUDA setup
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA not found, disabling CUDA support")
        set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA-accelerated features" FORCE)
    else()
        if(CUDAToolkit_VERSION VERSION_LESS "${NERVE_CUDA_REQUIRED_VERSION_MAJOR}.${NERVE_CUDA_REQUIRED_VERSION_MINOR}")
            message(FATAL_ERROR "Nerve requires CUDA >= ${NERVE_CUDA_REQUIRED_VERSION_MAJOR}.${NERVE_CUDA_REQUIRED_VERSION_MINOR}; found ${CUDAToolkit_VERSION}")
        endif()
        set(CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()

# ENABLE_ALL overrides individual settings
if(ENABLE_ALL)
    set(ENABLE_CUDA ON)
    set(ENABLE_PYTORCH ON)
    set(ENABLE_MPI ON)
    set(ENABLE_EIGEN ON)
    set(ENABLE_NUMA ON)
endif()

# Detect optional dependencies
if(ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_FOUND)
        message(STATUS "MPI found - Enabling distributed reduction features")
    else()
        message(WARNING "MPI not found, disabling MPI features")
        set(ENABLE_MPI OFF)
    endif()
endif()

if(ENABLE_EIGEN)
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Eigen3 found - Enabling Eigen-backed gradient and sheaf Laplacian components")
    else()
        message(WARNING "Eigen3 not found, disabling Eigen-backed components")
        set(ENABLE_EIGEN OFF)
    endif()
endif()

if(ENABLE_NUMA)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(WARNING "NUMA is only available on Linux, disabling NUMA features")
        set(ENABLE_NUMA OFF)
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(NUMA numa QUIET)
        endif()
        if(NOT NUMA_FOUND)
            find_library(NUMA_LIBRARY NAMES numa PATHS /usr/lib /usr/local/lib)
            find_path(NUMA_INCLUDE_DIR NAMES numa.h PATHS /usr/include /usr/local/include)
            if(NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
                set(NUMA_FOUND TRUE)
                set(NUMA_LIBRARIES ${NUMA_LIBRARY})
                set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})
            endif()
        endif()
        if(NUMA_FOUND)
            message(STATUS "NUMA library found - Enabling NUMA optimizations")
        else()
            message(WARNING "NUMA library not found, disabling NUMA features")
            set(ENABLE_NUMA OFF)
        endif()
    endif()
endif()

# Feature summary
message(STATUS "")
message(STATUS "Nerve Feature Summary:")
message(STATUS "  CUDA support:          ${ENABLE_CUDA}")
message(STATUS "  PyTorch support:       ${ENABLE_PYTORCH}")
message(STATUS "  MPI support:           ${ENABLE_MPI}")
message(STATUS "  Eigen3 support:        ${ENABLE_EIGEN}")
message(STATUS "  NUMA support:          ${ENABLE_NUMA}")
message(STATUS "")

# Core library
add_subdirectory(src)

# Configure feature flags after core target exists.
set(NERVE_FEATURE_DEFINITIONS "")
if(ENABLE_PH5)
    list(APPEND NERVE_FEATURE_DEFINITIONS ENABLE_PH5=1)
endif()
if(ENABLE_PH6)
    list(APPEND NERVE_FEATURE_DEFINITIONS ENABLE_PH6=1)
endif()
if(ENABLE_ADVANCED_FEATURES)
    list(APPEND NERVE_FEATURE_DEFINITIONS ENABLE_ADVANCED_FEATURES=1)
endif()
if(ENABLE_GREEN_CONTEXTS)
    list(APPEND NERVE_FEATURE_DEFINITIONS ENABLE_GREEN_CONTEXTS=1)
endif()
if(NERVE_FEATURE_DEFINITIONS)
    target_compile_definitions(nerve_core PRIVATE ${NERVE_FEATURE_DEFINITIONS})
endif()

# Tests
if(BUILD_TESTS)
    include(CTest)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
        set(NERVE_TESTS_ADDED TRUE CACHE INTERNAL "Tests subdirectory already added")
    endif()
endif()

# Benchmarks (reserved for future use)
if(BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
        add_subdirectory(benchmarks)
    endif()
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

include(cmake/tooling.cmake)

# Install rules
include(GNUInstallDirs)

install(TARGETS nerve_core
    EXPORT NerveTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(EXPORT NerveTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nerve
    NAMESPACE Nerve::
)

set(NERVE_INSTALL_HEADER_EXCLUDES)
list(APPEND NERVE_INSTALL_HEADER_EXCLUDES
    PATTERN "nerve/compression/model_aware_compression.hpp" EXCLUDE
    PATTERN "nerve/compression/gpu_autoencoder.hpp" EXCLUDE
    PATTERN "nerve/persistence/accelerated/gpu_apparent_pairs.hpp" EXCLUDE
    PATTERN "nerve/persistence/cuda/gpu_apparent_pairs.hpp" EXCLUDE
    PATTERN "nerve/persistence/gpu_apparent_pairs.hpp" EXCLUDE
    PATTERN "nerve/persistence/cuda_matrix_reduction_private.cuh" EXCLUDE
)
if(NOT NERVE_HAS_TORCH)
    list(APPEND NERVE_INSTALL_HEADER_EXCLUDES
        PATTERN "nerve/torch" EXCLUDE
        PATTERN "nerve/torch/*" EXCLUDE
    )
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
        PATTERN "*.inl"
        PATTERN "*.inc"
        ${NERVE_INSTALL_HEADER_EXCLUDES}
)
install(FILES "${NERVE_CONFIG_HEADER}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nerve
)

include(cmake/package_config.cmake)
