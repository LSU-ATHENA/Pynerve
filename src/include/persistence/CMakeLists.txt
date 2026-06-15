# Persistence Component

# Header-only library for unified reduction engine
add_library(nerve_persistence INTERFACE)

target_include_directories(nerve_persistence
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Dependencies
target_link_libraries(nerve_persistence
    INTERFACE
        Threads::Threads
)

# OpenMP support for parallel reduction
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nerve_persistence
        INTERFACE
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(nerve_persistence
        INTERFACE
            NERVE_ENABLE_OPENMP
    )
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/simd.cmake)
nerve_apply_simd(nerve_persistence INTERFACE)

# Export for parent CMakeLists.txt
set(NERVE_PERSISTENCE_TARGET nerve_persistence PARENT_SCOPE)
