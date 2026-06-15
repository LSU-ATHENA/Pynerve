# CUDA Components

# CUDA Error Handling Component
add_library(nerve_cuda_error INTERFACE)
target_include_directories(nerve_cuda_error
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Heterogeneous Computing Engine Component
add_library(nerve_heterogeneous INTERFACE)
target_include_directories(nerve_heterogeneous
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Common CUDA dependencies
if(BUILD_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(nerve_cuda_error
        INTERFACE
            CUDA::cudart
    )
    target_link_libraries(nerve_heterogeneous
        INTERFACE
            CUDA::cudart
            CUDA::cusparse
            CUDA::cusolver
    )
    target_compile_definitions(nerve_cuda_error
        INTERFACE
            NERVE_ENABLE_CUDA
    )
    target_compile_definitions(nerve_heterogeneous
        INTERFACE
            NERVE_ENABLE_HETEROGENEOUS_GPU
    )
endif()

# Threading support
target_link_libraries(nerve_heterogeneous
    INTERFACE
        Threads::Threads
)

# OpenMP for CPU proxy
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nerve_heterogeneous
        INTERFACE
            OpenMP::OpenMP_CXX
    )
endif()

# Export for parent CMakeLists.txt
set(NERVE_CUDA_ERROR_TARGET nerve_cuda_error PARENT_SCOPE)
set(NERVE_HETEROGENEOUS_TARGET nerve_heterogeneous PARENT_SCOPE)