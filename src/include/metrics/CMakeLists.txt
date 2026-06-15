# Metrics Component

# Header-only library for exact bottleneck and Wasserstein distances
add_library(nerve_metrics INTERFACE)

target_include_directories(nerve_metrics
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Dependencies
target_link_libraries(nerve_metrics
    INTERFACE
        Threads::Threads
)

# No special optimization flags needed - algorithmic complexity dominates

# Export for parent CMakeLists.txt
set(NERVE_METRICS_TARGET nerve_metrics PARENT_SCOPE)