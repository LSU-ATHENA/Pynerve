include_guard(GLOBAL)

# x86 SIMD Support

macro(nerve_configure_x86_simd_support)
    set(NERVE_TARGET_SUPPORTS_X86_SIMD OFF)
    if(CMAKE_OSX_ARCHITECTURES)
        set(NERVE_TARGET_SUPPORTS_X86_SIMD ON)
        foreach(NERVE_OSX_ARCH IN LISTS CMAKE_OSX_ARCHITECTURES)
            string(TOLOWER "${NERVE_OSX_ARCH}" NERVE_OSX_ARCH_LOWER)
            if(NOT NERVE_OSX_ARCH_LOWER MATCHES "^(x86_64|amd64|i[3-6]86)$")
                set(NERVE_TARGET_SUPPORTS_X86_SIMD OFF)
            endif()
        endforeach()
    else()
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" NERVE_SYSTEM_PROCESSOR_LOWER)
        if(NERVE_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|i[3-6]86)$")
            set(NERVE_TARGET_SUPPORTS_X86_SIMD ON)
        endif()
    endif()

    set(NERVE_CAN_USE_X86_SIMD_FLAGS OFF)
    if(NERVE_TARGET_SUPPORTS_X86_SIMD
       AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|MSVC")
        set(NERVE_CAN_USE_X86_SIMD_FLAGS ON)
    endif()
    if(NERVE_TARGET_SUPPORTS_X86_SIMD AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # MSVC on x64 always supports SSE2; detect higher via _M_AMD64
        set(NERVE_CAN_USE_X86_SIMD_FLAGS ON)
    endif()
endmacro()

macro(nerve_detect_avx512_codegen)
    if(NOT DEFINED NERVE_CAN_USE_X86_SIMD_FLAGS)
        nerve_configure_x86_simd_support()
    endif()

    if(NERVE_CAN_USE_X86_SIMD_FLAGS)
        include(CheckCXXSourceCompiles)
        include(CheckCXXSourceRuns)
        set(NERVE_SAVED_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS "-mavx512f")
        check_cxx_source_compiles("#include <immintrin.h>\nint main() { __m512i a; return 0; }" NERVE_HAS_AVX512)
        if(NERVE_HAS_AVX512 AND NOT CMAKE_CROSSCOMPILING)
            check_cxx_source_runs("#include <immintrin.h>\nint main() { __m512i a = _mm512_set1_epi32(1); return _mm512_cvtsi512_si32(a) == 1 ? 0 : 1; }" NERVE_HAS_AVX512_RUNTIME)
        elseif(NERVE_HAS_AVX512)
            set(NERVE_HAS_AVX512_RUNTIME UNKNOWN)
        else()
            set(NERVE_HAS_AVX512_RUNTIME OFF)
        endif()
        set(CMAKE_REQUIRED_FLAGS "${NERVE_SAVED_CMAKE_REQUIRED_FLAGS}")
        if(NERVE_HAS_AVX512)
            if(NERVE_HAS_AVX512_RUNTIME)
                message(STATUS "AVX-512 codegen and runtime execution are available")
            elseif(NERVE_HAS_AVX512_RUNTIME STREQUAL "UNKNOWN")
                message(STATUS "AVX-512 codegen is available; runtime execution was not checked while cross-compiling")
            else()
                message(STATUS "AVX-512 codegen is available, but this CPU cannot execute AVX-512 instructions")
            endif()
        else()
            message(STATUS "AVX-512 codegen missing - Building AVX-512 kernels out")
        endif()
    else()
        set(NERVE_HAS_AVX512 OFF)
        if(NERVE_TARGET_SUPPORTS_X86_SIMD)
            message(STATUS "AVX-512 detection skipped - Unsupported compiler")
        else()
            message(STATUS "AVX-512 detection skipped - Non-x86 target architecture")
        endif()
    endif()
endmacro()

# ARM SIMD Support

macro(nerve_configure_arm_simd_support)
    set(NERVE_TARGET_SUPPORTS_ARM_SIMD OFF)
    set(NERVE_HAS_NEON OFF)
    set(NERVE_HAS_SVE OFF)

    if(CMAKE_OSX_ARCHITECTURES)
        foreach(NERVE_OSX_ARCH IN LISTS CMAKE_OSX_ARCHITECTURES)
            string(TOLOWER "${NERVE_OSX_ARCH}" NERVE_OSX_ARCH_LOWER)
            if(NERVE_OSX_ARCH_LOWER MATCHES "^(arm64|aarch64)$")
                set(NERVE_TARGET_SUPPORTS_ARM_SIMD ON)
            endif()
        endforeach()
    else()
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" NERVE_SYSTEM_PROCESSOR_LOWER)
        if(NERVE_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$")
            set(NERVE_TARGET_SUPPORTS_ARM_SIMD ON)
        endif()
    endif()

    if(NERVE_TARGET_SUPPORTS_ARM_SIMD
       AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        include(CheckCXXSourceCompiles)
        set(NERVE_SAVED_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")

        # Check NEON (available on all ARMv8-A)
        set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+fp+simd")
        check_cxx_source_compiles(
            "#include <arm_neon.h>\nint main() { float64x2_t v = vdupq_n_f64(1.0); return (int)vgetq_lane_f64(v, 0); }"
            NERVE_HAS_NEON)

        # Check SVE (optional, ARMv8.2+ with SVE extension)
        set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+sve")
        check_cxx_source_compiles(
            "#include <arm_sve.h>\nint main() { svfloat64_t v = svdupq_n_f64(1.0, 2.0); return (int)svaddv_f64(svptrue_b64(), v); }"
            NERVE_HAS_SVE)
        # NERVE_HAS_SVE already set from the check above

        set(CMAKE_REQUIRED_FLAGS "${NERVE_SAVED_CMAKE_REQUIRED_FLAGS}")

        if(NERVE_HAS_NEON)
            message(STATUS "ARM NEON detected")
        endif()
        if(NERVE_HAS_SVE)
            message(STATUS "ARM SVE detected")
        endif()
    endif()
endmacro()

# SIMD Mode Validation

macro(nerve_validate_simd_mode)
    set(NERVE_SIMD "auto" CACHE STRING "SIMD mode: auto|sse41|avx2|avx512|neon|sve|scalar")
    set_property(CACHE NERVE_SIMD PROPERTY STRINGS auto sse41 avx2 avx512 neon sve scalar)

    if(NOT DEFINED NERVE_CAN_USE_X86_SIMD_FLAGS)
        nerve_configure_x86_simd_support()
    endif()
    if(NOT DEFINED NERVE_HAS_AVX512)
        nerve_detect_avx512_codegen()
    endif()
    if(NOT DEFINED NERVE_HAS_NEON)
        nerve_configure_arm_simd_support()
    endif()

    set(NERVE_ENABLE_AVX512_CODEGEN OFF)
    set(NERVE_ENABLE_NEON_CODEGEN OFF)
    set(NERVE_ENABLE_SVE_CODEGEN OFF)

    if(NERVE_SIMD STREQUAL "sse41")
        if(NOT NERVE_CAN_USE_X86_SIMD_FLAGS)
            message(FATAL_ERROR "NERVE_SIMD=sse41 requires an x86/x86_64 target and a GNU/Clang-compatible compiler")
        endif()
    elseif(NERVE_SIMD STREQUAL "avx2")
        if(NOT NERVE_CAN_USE_X86_SIMD_FLAGS)
            message(FATAL_ERROR "NERVE_SIMD=avx2 requires an x86/x86_64 target and a GNU/Clang-compatible compiler")
        endif()
    elseif(NERVE_SIMD STREQUAL "avx512")
        if(NOT NERVE_CAN_USE_X86_SIMD_FLAGS)
            message(FATAL_ERROR "NERVE_SIMD=avx512 requires an x86/x86_64 target and a GNU/Clang-compatible compiler")
        endif()
        if(NOT NERVE_HAS_AVX512)
            message(FATAL_ERROR "NERVE_SIMD=avx512 was requested, but the compiler does not accept AVX-512 codegen flags")
        endif()
        if(NOT CMAKE_CROSSCOMPILING AND NOT NERVE_HAS_AVX512_RUNTIME)
            message(FATAL_ERROR "NERVE_SIMD=avx512 was requested, but this CPU cannot execute AVX-512 instructions")
        endif()
        set(NERVE_ENABLE_AVX512_CODEGEN ON)
    elseif(NERVE_SIMD STREQUAL "neon")
        if(NOT NERVE_HAS_NEON)
            message(FATAL_ERROR "NERVE_SIMD=neon requires an ARM64 target with NEON")
        endif()
        set(NERVE_ENABLE_NEON_CODEGEN ON)
    elseif(NERVE_SIMD STREQUAL "sve")
        if(NOT NERVE_HAS_SVE)
            message(FATAL_ERROR "NERVE_SIMD=sve requires an ARM64 target with SVE")
        endif()
        set(NERVE_ENABLE_SVE_CODEGEN ON)
    elseif(NERVE_SIMD STREQUAL "auto" OR NERVE_SIMD STREQUAL "scalar")
        # Auto and scalar builds keep target-safe compiler flags.
    else()
        message(FATAL_ERROR "NERVE_SIMD must be auto, sse41, avx2, avx512, neon, sve, or scalar")
    endif()
endmacro()

# Apply SIMD to a target (x86 paths + ARM paths)

function(nerve_apply_simd target)
    set(NERVE_SIMD_VISIBILITY PRIVATE)
    if(ARGC GREATER 1)
        set(NERVE_SIMD_VISIBILITY "${ARGV1}")
    endif()

    if(NOT DEFINED NERVE_CAN_USE_X86_SIMD_FLAGS)
        nerve_configure_x86_simd_support()
    endif()
    if(NOT DEFINED NERVE_ENABLE_AVX512_CODEGEN)
        nerve_validate_simd_mode()
    endif()

    if(NERVE_SIMD STREQUAL "scalar")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_FORCE_SCALAR)
    elseif(NERVE_SIMD STREQUAL "sse41")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_FORCE_SSE41)
        target_compile_options(${target} ${NERVE_SIMD_VISIBILITY}
            "$<$<COMPILE_LANGUAGE:CXX>:-msse4.1>"
        )
    elseif(NERVE_SIMD STREQUAL "avx2")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_FORCE_AVX2)
        target_compile_options(${target} ${NERVE_SIMD_VISIBILITY}
            "$<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>"
        )
    elseif(NERVE_SIMD STREQUAL "avx512")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_FORCE_AVX512)
        target_compile_options(${target} ${NERVE_SIMD_VISIBILITY}
            "$<$<COMPILE_LANGUAGE:CXX>:-mavx512f;-mfma>"
        )
    elseif(NERVE_SIMD STREQUAL "neon")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_FORCE_NEON)
        target_compile_options(${target} ${NERVE_SIMD_VISIBILITY}
            "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a+fp+simd>"
        )
    elseif(NERVE_SIMD STREQUAL "sve")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_FORCE_SVE)
        target_compile_options(${target} ${NERVE_SIMD_VISIBILITY}
            "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+sve>"
        )
    elseif(NERVE_SIMD STREQUAL "auto")
        target_compile_definitions(${target} ${NERVE_SIMD_VISIBILITY} NERVE_SIMD_AUTO)
    else()
        message(FATAL_ERROR "NERVE_SIMD must be auto, sse41, avx2, avx512, neon, sve, or scalar")
    endif()
endfunction()

# Apply ARM SIMD compile flags to a specific target

function(nerve_apply_neon target)
    target_compile_definitions(${target} PRIVATE NERVE_HAS_NEON=1)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a+fp+simd>"
    )
endfunction()

function(nerve_apply_sve target)
    target_compile_definitions(${target} PRIVATE NERVE_HAS_SVE=1)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+sve>"
    )
endfunction()

# Arch-independent SIMD dispatch initialization

function(nerve_record_simd_capabilities)
    if(NERVE_ENABLE_AVX512_CODEGEN)
        target_compile_definitions(nerve_core PRIVATE
            NERVE_CMAKE_HAS_AVX512=1
            NERVE_USE_SIMD=1
            NERVE_HAS_AVX512=1
        )
    endif()
    if(NERVE_ENABLE_NEON_CODEGEN)
        target_compile_definitions(nerve_core PRIVATE NERVE_CMAKE_HAS_NEON=1)
    endif()
    if(NERVE_ENABLE_SVE_CODEGEN)
        target_compile_definitions(nerve_core PRIVATE NERVE_CMAKE_HAS_SVE=1)
    endif()
    if(NERVE_CAN_USE_X86_SIMD_FLAGS
       OR NERVE_HAS_NEON
       OR NERVE_HAS_SVE)
        target_compile_definitions(nerve_core PRIVATE NERVE_USE_SIMD=1)
    endif()
endfunction()
