# Install rules for Nerve
# Included from root CMakeLists.txt

include(GNUInstallDirs)

install(TARGETS nerve_core
    EXPORT NerveTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

set(NERVE_INSTALL_HEADER_EXCLUDES)
list(APPEND NERVE_INSTALL_HEADER_EXCLUDES
    PATTERN "nerve/compression/model_aware_compression.hpp" EXCLUDE
    PATTERN "nerve/compression/gpu_autoencoder.hpp" EXCLUDE
    PATTERN "nerve/persistence/accelerated/gpu_apparent_pairs.hpp" EXCLUDE
    PATTERN "nerve/persistence/cuda/gpu_apparent_pairs.hpp" EXCLUDE
    PATTERN "nerve/persistence/gpu_apparent_pairs.hpp" EXCLUDE
    PATTERN "nerve/persistence/cuda_matrix_reduction_private.cuh" EXCLUDE
)
if(NOT NERVE_HAS_TORCH)
    list(APPEND NERVE_INSTALL_HEADER_EXCLUDES
        PATTERN "nerve/torch" EXCLUDE
        PATTERN "nerve/torch/*" EXCLUDE
    )
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
        PATTERN "*.inl"
        PATTERN "*.inc"
        ${NERVE_INSTALL_HEADER_EXCLUDES}
)
install(FILES "${NERVE_CONFIG_HEADER}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nerve
)
