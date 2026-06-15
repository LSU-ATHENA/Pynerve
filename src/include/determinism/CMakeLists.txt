# Determinism Component

# Header-only library for determinism contract and verification
add_library(nerve_determinism INTERFACE)

target_include_directories(nerve_determinism
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Dependencies
target_link_libraries(nerve_determinism
    INTERFACE
        Threads::Threads
)

# Compile definitions for determinism checking
target_compile_definitions(nerve_determinism
    INTERFACE
        NERVE_ENABLE_DETERMINISM_CHECKS
)

# Export for parent CMakeLists.txt
set(NERVE_DETERMINISM_TARGET nerve_determinism PARENT_SCOPE)