# Python bindings CMake configuration

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NervePython VERSION 1.0.0 LANGUAGES CXX)
    option(BUILD_PYTHON "Build Python bindings" ON)
    if(NOT BUILD_PYTHON)
        message(FATAL_ERROR "The standalone Nerve Python package build requires BUILD_PYTHON=ON")
    endif()
endif()

set(NERVE_SOURCE_ROOT "")
foreach(NERVE_SOURCE_ROOT_CANDIDATE
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(EXISTS "${NERVE_SOURCE_ROOT_CANDIDATE}/src/CMakeLists.txt")
        get_filename_component(NERVE_SOURCE_ROOT "${NERVE_SOURCE_ROOT_CANDIDATE}" REALPATH)
        break()
    endif()
endforeach()
if(NOT NERVE_SOURCE_ROOT)
    message(FATAL_ERROR "Nerve Python bindings require the Nerve src tree")
endif()
set(NERVE_SRC_ROOT "${NERVE_SOURCE_ROOT}/src")
set(NERVE_TOOLS_ROOT "${NERVE_SOURCE_ROOT}/tools")
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NERVE_PYTHON_BUILD_ROOT "${CMAKE_CURRENT_BINARY_DIR}/python")
else()
    set(NERVE_PYTHON_BUILD_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(NOT TARGET nerve_core)
    if(NOT DEFINED ENABLE_CUDA)
        option(ENABLE_CUDA "Enable CUDA-accelerated features (GPU memory, kernels)" OFF)
    endif()
    if(NOT DEFINED BUILD_TESTS)
        option(BUILD_TESTS "Build tests" OFF)
    endif()
    if(NOT DEFINED BUILD_BENCHMARKS)
        option(BUILD_BENCHMARKS "Build benchmarks" OFF)
    endif()
    if(NOT DEFINED ENABLE_PYTORCH)
        option(ENABLE_PYTORCH "Enable PyTorch differentiable operations" ON)
    endif()
    if(NOT DEFINED ENABLE_MPI)
        option(ENABLE_MPI "Enable distributed/MPI reduction features" OFF)
    endif()
    if(NOT DEFINED ENABLE_EIGEN)
        option(ENABLE_EIGEN "Enable Eigen3-backed gradient and sheaf Laplacian components" OFF)
    endif()
    if(NOT DEFINED ENABLE_NUMA)
        option(ENABLE_NUMA "Enable NUMA memory optimizations" OFF)
    endif()
    if(NOT DEFINED ENABLE_ADVANCED_FEATURES)
        option(ENABLE_ADVANCED_FEATURES "Enable all advanced features" OFF)
    endif()
    if(NOT DEFINED NERVE_NATIVE_ARCH)
        option(NERVE_NATIVE_ARCH
            "Tune CPU code generation for the build host; disables portable binary output"
            OFF)
    endif()
    find_package(Threads REQUIRED)
    if(ENABLE_PYTORCH)
        find_package(Torch QUIET)
    endif()
    add_subdirectory("${NERVE_SRC_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/nerve-src")
endif()

if(BUILD_TESTS AND NOT NERVE_TESTS_ADDED)
    include(CTest)
    if(EXISTS "${NERVE_SOURCE_ROOT}/tests/CMakeLists.txt")
        add_subdirectory("${NERVE_SOURCE_ROOT}/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests")
    endif()
endif()

# Find Python and pybind11.
find_package(Python 3.10...<3.14 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(PYBIND11_CMAKE_DIR_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

# Python binding sources.
set(PYTHON_API_SOURCES
    bindings/nerve_api_bindings.cpp
    ${NERVE_SRC_ROOT}/algorithms/distance.cpp
    ${NERVE_SRC_ROOT}/persistence/utils/diagram_statistics.cpp
)

# Create Python extension module with correct name
pybind11_add_module(pynerve_internal ${PYTHON_API_SOURCES})

pybind11_add_module(nerve_extras
    bindings/nerve_extras_bindings.cpp
)

target_link_libraries(nerve_extras PRIVATE pybind11::headers nerve_core Python::NumPy)
target_include_directories(nerve_extras PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    ${NERVE_SRC_ROOT}/include
)
target_compile_features(nerve_extras PRIVATE cxx_std_20)
set_target_properties(nerve_extras PROPERTIES
    OUTPUT_NAME nerve_extras
    SUFFIX ${PYTHON_MODULE_EXTENSION}
    LIBRARY_OUTPUT_DIRECTORY "${NERVE_PYTHON_BUILD_ROOT}"
)

pybind11_add_module(algorithms_bindings
    bindings/nerve_algorithms_bindings.cpp
)

# PyTorch-specific bindings (if Torch is available)
if(ENABLE_PYTORCH)
    find_package(Torch QUIET)
else()
    set(Torch_FOUND OFF)
endif()

if(Torch_FOUND)
    message(STATUS "PyTorch found - Building torch-native bindings")

    set(TORCH_BINDING_SOURCES
        bindings/nerve_torch_bindings.cpp
    )
    foreach(NERVE_TORCH_SOURCE IN LISTS NERVE_TORCH_SOURCES)
        list(APPEND TORCH_BINDING_SOURCES
            "${NERVE_SRC_ROOT}/${NERVE_TORCH_SOURCE}"
        )
    endforeach()

    pybind11_add_module(nerve_torch_internal ${TORCH_BINDING_SOURCES})

    target_include_directories(nerve_torch_internal PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings
        ${NERVE_SRC_ROOT}/include
    )

    target_link_libraries(nerve_torch_internal
        PRIVATE ${TORCH_LIBRARIES}
        PRIVATE nerve_core
        PRIVATE nerve_torch
    )
    get_filename_component(NERVE_TORCH_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
    find_library(NERVE_TORCH_PYTHON_LIBRARY
        NAMES torch_python
        PATHS "${NERVE_TORCH_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    if(NOT NERVE_TORCH_PYTHON_LIBRARY)
        message(FATAL_ERROR "PyTorch-native bindings require libtorch_python")
    endif()
    target_link_libraries(nerve_torch_internal PRIVATE ${NERVE_TORCH_PYTHON_LIBRARY})

    target_compile_definitions(nerve_torch_internal PRIVATE NERVE_TORCH_AVAILABLE)
    target_compile_features(nerve_torch_internal PRIVATE cxx_std_20)
    set_target_properties(nerve_torch_internal PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${NERVE_PYTHON_BUILD_ROOT}"
    )

    message(STATUS "nerve_torch_internal module will be built")
else()
    message(STATUS "PyTorch not found - Skipping torch-native bindings")
endif()

# Link libraries
target_link_libraries(pynerve_internal PRIVATE pybind11::headers nerve_core)
target_link_libraries(algorithms_bindings PRIVATE pybind11::headers nerve_core)
target_link_libraries(pynerve_internal PRIVATE Python::NumPy)
target_link_libraries(algorithms_bindings PRIVATE Python::NumPy)

# Include directories
target_include_directories(pynerve_internal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    ${NERVE_SRC_ROOT}/include
)
target_include_directories(algorithms_bindings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    ${NERVE_SRC_ROOT}/include
)

# Compiler flags
target_compile_features(pynerve_internal PRIVATE cxx_std_20)
target_compile_features(algorithms_bindings PRIVATE cxx_std_20)

if(Torch_FOUND AND TORCH_CXX_FLAGS)
    separate_arguments(NERVE_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(pynerve_internal PRIVATE ${NERVE_TORCH_CXX_FLAGS})
    target_compile_options(algorithms_bindings PRIVATE ${NERVE_TORCH_CXX_FLAGS})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(pynerve_internal PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(algorithms_bindings PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Module properties
set_target_properties(pynerve_internal PROPERTIES
    OUTPUT_NAME pynerve_internal
    SUFFIX ${PYTHON_MODULE_EXTENSION}
    LIBRARY_OUTPUT_DIRECTORY "${NERVE_PYTHON_BUILD_ROOT}"
)
set_target_properties(algorithms_bindings PROPERTIES
    OUTPUT_NAME algorithms_bindings
    SUFFIX ${PYTHON_MODULE_EXTENSION}
    LIBRARY_OUTPUT_DIRECTORY "${NERVE_PYTHON_BUILD_ROOT}"
)

set(NERVE_BINDING_SMOKE_COMMAND
    ${Python_EXECUTABLE}
    ${NERVE_TOOLS_ROOT}/binding_smoke.py
    --build-dir ${CMAKE_BINARY_DIR}
)
add_custom_target(pynerve_python_binding_smoke
    COMMAND ${NERVE_BINDING_SMOKE_COMMAND}
    DEPENDS pynerve_internal algorithms_bindings nerve_extras
    WORKING_DIRECTORY ${NERVE_SOURCE_ROOT}
    COMMENT "Smoke-testing Python extension modules"
)
if(BUILD_TESTING)
    add_test(
        NAME pynerve_python_binding_smoke
        COMMAND ${NERVE_BINDING_SMOKE_COMMAND}
        WORKING_DIRECTORY ${NERVE_SOURCE_ROOT}
    )
    set_tests_properties(pynerve_python_binding_smoke PROPERTIES
        LABELS "python;bindings;operators"
    )
endif()

if(Torch_FOUND)
    set(NERVE_TORCH_BINDING_SMOKE_COMMAND
        ${Python_EXECUTABLE}
        ${NERVE_TOOLS_ROOT}/torch_binding_smoke.py
        --build-dir ${CMAKE_BINARY_DIR}
    )
    add_custom_target(pynerve_torch_binding_smoke
        COMMAND ${NERVE_TORCH_BINDING_SMOKE_COMMAND}
        DEPENDS pynerve_internal algorithms_bindings nerve_torch_internal
        WORKING_DIRECTORY ${NERVE_SOURCE_ROOT}
        COMMENT "Smoke-testing PyTorch-native extension modules"
    )
    if(BUILD_TESTING)
        add_test(
            NAME pynerve_torch_binding_smoke
            COMMAND ${NERVE_TORCH_BINDING_SMOKE_COMMAND}
            WORKING_DIRECTORY ${NERVE_SOURCE_ROOT}
        )
        set_tests_properties(pynerve_torch_binding_smoke PROPERTIES
            LABELS "torch;python;bindings;gradient"
        )
    endif()
endif()

# Copy the source Python package into the build tree.
file(GLOB_RECURSE NERVE_PYTHON_PACKAGE_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/pynerve/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/pynerve/*.pyi"
)
foreach(NERVE_PYTHON_PACKAGE_FILE IN LISTS NERVE_PYTHON_PACKAGE_FILES)
    file(RELATIVE_PATH NERVE_PYTHON_PACKAGE_RELATIVE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${NERVE_PYTHON_PACKAGE_FILE}"
    )
    configure_file(
        "${NERVE_PYTHON_PACKAGE_FILE}"
        "${NERVE_PYTHON_BUILD_ROOT}/${NERVE_PYTHON_PACKAGE_RELATIVE}"
        COPYONLY
    )
endforeach()

# Install Python package
install(TARGETS pynerve_internal
    DESTINATION .
    COMPONENT python
)
install(TARGETS algorithms_bindings
    DESTINATION .
    COMPONENT python
)
install(TARGETS nerve_extras
    DESTINATION .
    COMPONENT python
)
install(DIRECTORY pynerve/
    DESTINATION pynerve
    COMPONENT python
    FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.pyi"
)

# Install torch bindings if built
if(Torch_FOUND)
    install(TARGETS nerve_torch_internal
        DESTINATION .
        COMPONENT python
    )
endif()
