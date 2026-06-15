# Adaptive Acceleration Persistent Homology CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# Adaptive Acceleration library options
option(NERVE_ENABLE_ADAPTIVE_ACCELERATION "Enable Adaptive Acceleration persistent homology" ON)
option(NERVE_ENABLE_CUDA "Enable CUDA support for Adaptive Acceleration" OFF)
option(NERVE_ENABLE_STREAMING "Enable streaming support" ON)
option(NERVE_ENABLE_APPROXIMATION "Enable approximation methods" ON)
option(NERVE_ENABLE_REPRESENTATIVE_CYCLES "Enable representative cycles" ON)
option(NERVE_ENABLE_PERFORMANCE_MONITORING "Enable performance monitoring" ON)
option(NERVE_NATIVE_ARCH
    "Tune CPU code generation for the build host; disables portable binary output"
    OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/simd.cmake)
nerve_configure_x86_simd_support()

# Adaptive Acceleration library sources
set(ADAPTIVE_ACCELERATION_SOURCES
    matrix_multiplication_framework.cpp
    sparsification_engine.cpp
    lockfree_multicore.cpp
    adaptive_algorithm_selector.cpp
    adaptive_acceleration_engine.cpp
    adaptive_acceleration_problem_analysis.cpp
    adaptive_acceleration_system_capabilities.cpp
    sparse_matrix.cpp
)

# Streaming sources
if(NERVE_ENABLE_STREAMING)
    list(APPEND ADAPTIVE_ACCELERATION_SOURCES
        streaming/streaming_processor.cpp
        streaming/approximate_processor.cpp
        streaming/representative_cycles.cpp
    )
endif()

# CUDA sources
if(NERVE_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA not found, disabling Adaptive Acceleration CUDA support")
        set(NERVE_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support for Adaptive Acceleration" FORCE)
    else()
        list(APPEND ADAPTIVE_ACCELERATION_SOURCES
            cuda/distance_kernels.cu
            cuda/reduction_kernels.cu
            cuda/tensor_core_optimizer.cpp
            cuda/hybrid_algorithms.cpp
        )
        set(CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()

# Create Adaptive Acceleration library
add_library(nerve_persistence_adaptive_acceleration STATIC ${ADAPTIVE_ACCELERATION_SOURCES})

# Include directories
target_include_directories(nerve_persistence_adaptive_acceleration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

# Link with core library
target_link_libraries(nerve_persistence_adaptive_acceleration PRIVATE
    nerve_core
    nerve_persistence
)

# Compiler flags
target_compile_features(nerve_persistence_adaptive_acceleration PRIVATE
    cxx_std_20
)
nerve_apply_simd(nerve_persistence_adaptive_acceleration PRIVATE)

# CUDA integration
if(NERVE_ENABLE_CUDA AND CUDAToolkit_FOUND)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90" CACHE STRING
            "CUDA architectures for Nerve adaptive acceleration targets" FORCE)
    endif()

    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NERVE_ENABLE_CUDA=1
    )

    target_link_libraries(nerve_persistence_adaptive_acceleration PRIVATE
        ${CUDA_LIBRARIES}
    )

    # CUDA compile flags
    set_target_properties(nerve_persistence_adaptive_acceleration PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    # CUDA sources
    set(CUDA_SOURCES
        cuda/distance_kernels.cu
        cuda/reduction_kernels.cu
    )

    if(CUDA_SOURCES)
        cuda_add_library(nerve_persistence_adaptive_acceleration_cuda STATIC ${CUDA_SOURCES})
        target_link_libraries(nerve_persistence_adaptive_acceleration PRIVATE
            nerve_persistence_adaptive_acceleration_cuda
            ${CUDA_LIBRARIES}
        )

        # CUDA compile flags
        set_target_properties(nerve_persistence_adaptive_acceleration_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()
endif()

# Streaming integration
if(NERVE_ENABLE_STREAMING)
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NERVE_ENABLE_STREAMING=1
    )
endif()

# Approximation integration
if(NERVE_ENABLE_APPROXIMATION)
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NERVE_ENABLE_APPROXIMATION=1
    )
endif()

# Representative cycles integration
if(NERVE_ENABLE_REPRESENTATIVE_CYCLES)
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NERVE_ENABLE_REPRESENTATIVE_CYCLES=1
    )
endif()

# Performance monitoring integration
if(NERVE_ENABLE_PERFORMANCE_MONITORING)
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NERVE_ENABLE_PERFORMANCE_MONITORING=1
    )
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(nerve_persistence_adaptive_acceleration PRIVATE Threads)

# Platform-specific optimizations
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
        -pthread
    )
    target_link_libraries(nerve_persistence_adaptive_acceleration PRIVATE rt)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        DEBUG
        _DEBUG
        NERVE_DEBUG=1
    )
    target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
        -g
        -O0
        -DDEBUG
    )
endif()

# Release configuration
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NDEBUG
        NERVE_RELEASE=1
    )
    target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
        -O3
        -DNDEBUG
        -flto
        -finline-functions
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
            -fno-fast-math
            -fno-associative-math
            -fno-unsafe-math-optimizations
            -ffp-contract=off
        )
    endif()

    # Enable Link-Time Optimization
    set_target_properties(nerve_persistence_adaptive_acceleration PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )

    # Additional optimization flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
            -fwhole-program
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
            -Rpass=loop-vectorize
            -Rpass-missed=loop-vectorize
        )
    endif()
endif()

# RelWithDebInfo configuration
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_definitions(nerve_persistence_adaptive_acceleration PRIVATE
        NDEBUG
        NERVE_RELWITHDEBINFO=1
    )
    target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
        -O2
        -DNDEBUG
        -g
    )
endif()

if(NERVE_NATIVE_ARCH
   AND CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
        -march=native
    )
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(nerve_persistence_adaptive_acceleration PRIVATE
            -mtune=native
        )
    endif()
endif()

# Installation
install(TARGETS nerve_persistence_adaptive_acceleration
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/nerve/persistence/adaptive_acceleration
)

# Install headers
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/nerve/persistence/adaptive_acceleration
    FILES_MATCHING PATTERN "*.hpp"
)

# Install streaming headers
if(NERVE_ENABLE_STREAMING)
    install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/streaming/
        DESTINATION include/nerve/persistence/adaptive_acceleration/streaming
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()

# Install CUDA headers
if(NERVE_ENABLE_CUDA)
    install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/
        DESTINATION include/nerve/persistence/adaptive_acceleration/cuda
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()

# Testing
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NerveSOTAConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NerveSOTAConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NerveSOTAConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NerveAdaptiveAcceleration
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NerveSOTAConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NerveSOTAConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NerveAdaptiveAcceleration
)

install(EXPORT NerveAdaptiveAccelerationTargets
    FILE NerveAdaptiveAccelerationTargets.cmake
    NAMESPACE Nerve::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NerveAdaptiveAcceleration
)

# Print configuration summary
message(STATUS "Nerve Adaptive Acceleration Configuration:")
message(STATUS "  Enable Adaptive Acceleration: ${NERVE_ENABLE_ADAPTIVE_ACCELERATION}")
message(STATUS "  Enable CUDA: ${NERVE_ENABLE_CUDA}")
message(STATUS "  Enable Streaming: ${NERVE_ENABLE_STREAMING}")
message(STATUS "  Enable Approximation: ${NERVE_ENABLE_APPROXIMATION}")
message(STATUS "  Enable Representative Cycles: ${NERVE_ENABLE_REPRESENTATIVE_CYCLES}")
message(STATUS "  Enable Performance Monitoring: ${NERVE_ENABLE_PERFORMANCE_MONITORING}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(NERVE_ENABLE_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
