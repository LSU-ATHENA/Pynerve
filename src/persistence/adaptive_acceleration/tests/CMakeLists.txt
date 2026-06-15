# Adaptive Acceleration Persistent Homology Tests CMakeLists.txt

# Find required packages
find_package(GTest REQUIRED)

# Test sources
set(ADAPTIVE_ACCELERATION_TEST_SOURCES
    adaptive_acceleration_performance_tests.cpp
    adaptive_acceleration_integration_tests.cpp
    adaptive_acceleration_correctness_tests.cpp
    adaptive_acceleration_memory_tests.cpp
    adaptive_acceleration_thread_safety_tests.cpp
    adaptive_acceleration_streaming_tests.cpp
    adaptive_acceleration_approximation_tests.cpp
    adaptive_acceleration_representative_cycles_tests.cpp
    adaptive_acceleration_cuda_tests.cpp
    adaptive_acceleration_performance_benchmarks.cpp
)

# Test executable
add_executable(nerve_adaptive_acceleration_tests ${ADAPTIVE_ACCELERATION_TEST_SOURCES})

# Link with libraries
target_link_libraries(nerve_adaptive_acceleration_tests PRIVATE
    nerve_persistence_adaptive_acceleration
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Include directories
target_include_directories(nerve_adaptive_acceleration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compiler flags
target_compile_features(nerve_adaptive_acceleration_tests PRIVATE
    cxx_std_20
)

# CUDA integration for tests
if(NERVE_ENABLE_CUDA)
    target_compile_definitions(nerve_adaptive_acceleration_tests PRIVATE
        NERVE_ENABLE_CUDA=1
    )

    target_link_libraries(nerve_adaptive_acceleration_tests PRIVATE
        ${CUDA_LIBRARIES}
    )
endif()

# Streaming integration for tests
if(NERVE_ENABLE_STREAMING)
    target_compile_definitions(nerve_adaptive_acceleration_tests PRIVATE
        NERVE_ENABLE_STREAMING=1
    )
endif()

# Approximation integration for tests
if(NERVE_ENABLE_APPROXIMATION)
    target_compile_definitions(nerve_adaptive_acceleration_tests PRIVATE
        NERVE_ENABLE_APPROXIMATION=1
    )
endif()

# Representative cycles integration for tests
if(NERVE_ENABLE_REPRESENTATIVE_CYCLES)
    target_compile_definitions(nerve_adaptive_acceleration_tests PRIVATE
        NERVE_ENABLE_REPRESENTATIVE_CYCLES=1
    )
endif()

# Performance monitoring integration for tests
if(NERVE_ENABLE_PERFORMANCE_MONITORING)
    target_compile_definitions(nerve_adaptive_acceleration_tests PRIVATE
        NERVE_ENABLE_PERFORMANCE_MONITORING=1
    )
endif()

# Add tests to CTest
add_test(NAME AdaptiveAccelerationPerformanceTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="AdaptiveAccelerationPerformanceTest.*")
add_test(NAME AdaptiveAccelerationIntegrationTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTAIntegrationTest.*")
add_test(NAME AdaptiveAccelerationCorrectnessTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTACorrectnessTest.*")
add_test(NAME AdaptiveAccelerationMemoryTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTAMemoryTest.*")
add_test(NAME SAThreadSafetyTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SAThreadSafetyTest.*")
add_test(NAME AdaptiveAccelerationStreamingTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTAStreamingTest.*")
add_test(NAME AdaptiveAccelerationApproximationTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTAApproximationTest.*")
add_test(NAME AdaptiveAccelerationRepresentativeCyclesTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTARepresentativeCyclesTest.*")
add_test(NAME AdaptiveAccelerationCudaTests COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTACudaTest.*")
add_test(NAME AdaptiveAccelerationPerformanceBenchmarks COMMAND nerve_adaptive_acceleration_tests --gtest_filter="SOTAPerformanceBenchmark.*")

# Performance benchmark executable
add_executable(nerve_adaptive_acceleration_benchmarks adaptive_acceleration_performance_benchmarks.cpp)

target_link_libraries(nerve_adaptive_acceleration_benchmarks PRIVATE
    nerve_persistence_adaptive_acceleration
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

target_include_directories(nerve_adaptive_acceleration_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(nerve_adaptive_acceleration_benchmarks PRIVATE
    cxx_std_20
)

# CUDA integration for benchmarks
if(NERVE_ENABLE_CUDA)
    target_compile_definitions(nerve_adaptive_acceleration_benchmarks PRIVATE
        NERVE_ENABLE_CUDA=1
    )

    target_link_libraries(nerve_adaptive_acceleration_benchmarks PRIVATE
        ${CUDA_LIBRARIES}
    )
endif()

# Add benchmarks to CTest
add_test(NAME AdaptiveAccelerationPerformanceBenchmarks COMMAND nerve_adaptive_acceleration_benchmarks)

# Memory test executable
add_executable(nerve_adaptive_acceleration_memory_tests adaptive_acceleration_memory_tests.cpp)

target_link_libraries(nerve_adaptive_acceleration_memory_tests PRIVATE
    nerve_persistence_adaptive_acceleration
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

target_include_directories(nerve_adaptive_acceleration_memory_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(nerve_adaptive_acceleration_memory_tests PRIVATE
    cxx_std_20
)

# Add memory tests to CTest
add_test(NAME AdaptiveAccelerationMemoryLeakTests COMMAND nerve_adaptive_acceleration_memory_tests)

# Thread safety test executable
add_executable(nerve_adaptive_acceleration_thread_tests adaptive_acceleration_thread_safety_tests.cpp)

target_link_libraries(nerve_adaptive_acceleration_thread_tests PRIVATE
    nerve_persistence_adaptive_acceleration
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

target_include_directories(nerve_adaptive_acceleration_thread_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(nerve_adaptive_acceleration_thread_tests PRIVATE
    cxx_std_20
)

# Add thread safety tests to CTest
add_test(NAME SAThreadSafetyValidationTests COMMAND nerve_adaptive_acceleration_thread_tests)

# Installation
install(TARGETS nerve_adaptive_acceleration_tests
    RUNTIME DESTINATION bin
)

install(TARGETS nerve_adaptive_acceleration_benchmarks
    RUNTIME DESTINATION bin
)

install(TARGETS nerve_adaptive_acceleration_memory_tests
    RUNTIME DESTINATION bin
)

install(TARGETS nerve_adaptive_acceleration_thread_tests
    RUNTIME DESTINATION bin
)