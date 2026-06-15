# Mathematical Components

# Header-only libraries for tolerance and precision systems
add_library(nerve_math INTERFACE)

target_include_directories(nerve_math
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compile definitions for precision control
target_compile_definitions(nerve_math
    INTERFACE
        NERVE_MATH_ENABLE_PRECISION_CONTROL
        NERVE_MATH_ENABLE_TOLERANCE_SYSTEM
)

# No special dependencies - pure C++20

# Export for parent CMakeLists.txt
set(NERVE_MATH_TARGET nerve_math PARENT_SCOPE)