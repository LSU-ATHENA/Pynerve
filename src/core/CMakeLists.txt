# Ownership tests CMake configuration

cmake_minimum_required(VERSION 3.16)
project(OwnershipTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(../../src/include)

# Ownership test executables
add_executable(ownership_integration_test
    ownership_integration_test.cpp
    ownership_policy.cpp
)

add_executable(ownership_checks_test
    ownership_checks_test.cpp
    ownership_policy.cpp
)

add_executable(ownership_performance_test
    ownership_performance_test.cpp
    ownership_policy.cpp
)

add_executable(ownership_zero_copy_test
    ownership_zero_copy_test.cpp
    ownership_policy.cpp
)

# Link with threading library if needed
if(THREADS_HAVE_PTHREAD)
    target_link_libraries(ownership_integration_test Threads::Threads)
    target_link_libraries(ownership_checks_test Threads::Threads)
    target_link_libraries(ownership_performance_test Threads::Threads)
    target_link_libraries(ownership_zero_copy_test Threads::Threads)
endif()

# Add compiler flags for better error checking
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ownership_integration_test PRIVATE -Wall -Wextra -Werror)
    target_compile_options(ownership_checks_test PRIVATE -Wall -Wextra -Werror)
    target_compile_options(ownership_performance_test PRIVATE -Wall -Wextra -Werror)
    target_compile_options(ownership_zero_copy_test PRIVATE -Wall -Wextra -Werror)
endif()

# Enable testing
enable_testing()

# Add tests
add_test(NAME ownership_integration
         COMMAND ownership_integration_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME ownership_checks
         COMMAND ownership_checks_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME ownership_performance
         COMMAND ownership_performance_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME ownership_zero_copy
         COMMAND ownership_zero_copy_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Test suite
set_tests_properties(ownership_tests PROPERTIES
    TIMEOUT 300
    PASS_REGULAR_EXPRESSION "OK|passed"
    FAIL_REGULAR_EXPRESSION "FAILED|failed"
)
