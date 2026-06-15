# Distance Computing Component

# Header-only library for SIMD distance computing
add_library(nerve_distance INTERFACE)

target_include_directories(nerve_distance
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/simd.cmake)
nerve_apply_simd(nerve_distance INTERFACE)

# Threading support
target_link_libraries(nerve_distance
    INTERFACE
        Threads::Threads
)

# Export for parent CMakeLists.txt
set(NERVE_DISTANCE_TARGET nerve_distance PARENT_SCOPE)
