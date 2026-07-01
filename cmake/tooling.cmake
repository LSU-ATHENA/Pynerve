# Tooling and quality targets for Nerve
# Included from root CMakeLists.txt

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(nerve_generate_test_matrix
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/test_matrix.py
                --output ${CMAKE_CURRENT_BINARY_DIR}/nerve-test-matrix.json
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Nerve test matrix"
    )
    add_custom_target(nerve_generate_pyi
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/generate_pyi.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Python API type files from public signatures"
    )
endif()
