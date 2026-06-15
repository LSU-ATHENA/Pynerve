add_library(nerve_core STATIC ${NERVE_ALL_SOURCES})
set_target_properties(nerve_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

function(nerve_apply_project_warnings target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
       OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
        )
        if(NERVE_HAS_CUDA)
            target_compile_options(${target} PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
            )
        endif()
    endif()
endfunction()

if(NERVE_HAS_CUDA AND NERVE_ENABLE_CUDA_COMPONENTS)
    set_source_files_properties(
        cuda/kernels/reduction_kernels_launcher.cpp
        PROPERTIES LANGUAGE CUDA
    )
endif()

nerve_apply_simd(nerve_core)
target_link_libraries(nerve_core PRIVATE Threads::Threads)
if(NERVE_HAS_MIMALLOC AND DEFINED NERVE_MIMALLOC_TARGET)
    target_link_libraries(nerve_core PUBLIC ${NERVE_MIMALLOC_TARGET})
    target_compile_definitions(nerve_core PRIVATE NERVE_USE_MIMALLOC=1)
    message(STATUS "nerve_core linked against mimalloc")
endif()
if(NERVE_HAS_OPENMP)
    target_link_libraries(nerve_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(nerve_core PRIVATE NERVE_USE_OPENMP=1)
endif()
if(NERVE_ENABLE_AVX512_CODEGEN)
    target_compile_definitions(nerve_core PRIVATE NERVE_USE_SIMD=1 NERVE_HAS_AVX512=1)
endif()
if(NERVE_HAS_TORCH AND TORCH_CXX_FLAGS)
    separate_arguments(NERVE_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(nerve_core PUBLIC ${NERVE_TORCH_CXX_FLAGS})
endif()

if(NERVE_ENABLE_ADVANCED_DIFFERENTIABLE_COMPONENTS)
    target_compile_definitions(nerve_core PUBLIC NERVE_ENABLE_ADVANCED_DIFFERENTIABLE=1)
endif()

if(NERVE_HAS_CUDA)
    target_link_libraries(nerve_core PRIVATE CUDA::cudart CUDA::cusparse CUDA::cusolver CUDA::cublas)
    target_compile_definitions(nerve_core PRIVATE NERVE_HAS_CUDA=1)
endif()

if(NERVE_HAS_CUDNN)
    target_include_directories(nerve_core PRIVATE ${NERVE_CUDNN_INCLUDE_DIR})
    target_compile_definitions(nerve_core PRIVATE NERVE_HAS_CUDNN=1)
endif()

if(NERVE_HAS_MPI)
    target_link_libraries(nerve_core PRIVATE MPI::MPI_CXX)
    target_compile_definitions(nerve_core PRIVATE NERVE_HAS_MPI=1)
endif()

if(NERVE_HAS_EIGEN3 AND TARGET Eigen3::Eigen)
    target_link_libraries(nerve_core PUBLIC Eigen3::Eigen)
    target_compile_definitions(nerve_core PRIVATE NERVE_HAS_EIGEN=1 NERVE_ENABLE_ADVANCED_SHEAF_LAPLACIANS=1)
endif()

if(NERVE_HAS_NUMA)
    target_link_libraries(nerve_core PRIVATE ${NUMA_LIB})
    target_include_directories(nerve_core PRIVATE ${NUMA_INCLUDE})
    target_compile_definitions(nerve_core PRIVATE NERVE_HAS_NUMA=1)
endif()

if(NERVE_ENABLE_AVX512_CODEGEN)
    target_compile_options(nerve_core PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-mavx512f;-mavx512dq;-mavx512bw;-mfma>"
    )
endif()

nerve_apply_project_warnings(nerve_core)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(nerve_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-O3;-DNDEBUG>")
        if(NERVE_NATIVE_ARCH)
            target_compile_options(nerve_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
            if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
                target_compile_options(nerve_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-mtune=native>")
            endif()
        endif()
        target_compile_options(nerve_core PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-fno-associative-math;-fno-unsafe-math-optimizations;-ffp-contract=off>"
        )
    endif()
endif()

target_include_directories(nerve_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(nerve_core PRIVATE NERVE_EXPORTS)
if(NERVE_HAS_TORCH)
    add_library(nerve_torch STATIC ${NERVE_TORCH_SOURCES})
    set_target_properties(nerve_torch PROPERTIES
        POSITION_INDEPENDENT_CODE ON

        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(nerve_torch
        PUBLIC
            ${TORCH_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(nerve_torch
        PUBLIC ${TORCH_LIBRARIES}
        PRIVATE nerve_core
    )
    message(STATUS "Created nerve_torch library")
endif()

add_library(nerve_full INTERFACE)
target_link_libraries(nerve_full INTERFACE nerve_core)
if(NERVE_HAS_TORCH)
    target_link_libraries(nerve_full INTERFACE nerve_torch)
endif()
