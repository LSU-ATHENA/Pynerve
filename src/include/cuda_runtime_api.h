#pragma once

#if defined(__has_include_next)
#if __has_include_next(<cuda_runtime_api.h>)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include_next <cuda_runtime_api.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
#include "cuda_runtime.h"
#endif
#else
#include "cuda_runtime.h"
#endif
