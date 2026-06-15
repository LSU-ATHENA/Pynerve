# Package configuration for Nerve
# Included from root CMakeLists.txt

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NerveConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/NerveConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NerveConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nerve
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NerveConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NerveConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nerve
)

install(EXPORT NerveTargets
    FILE NerveTargets.cmake
    NAMESPACE Nerve::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nerve
)
