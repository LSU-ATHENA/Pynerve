#pragma once

#if defined(__has_include_next)
#if __has_include_next(<cuda_fp16.h>)
#include_next <cuda_fp16.h>
#define NERVE_CUDA_FP16_AVAILABLE 1
#else
#define NERVE_CUDA_FP16_AVAILABLE 0
struct __half
{
    unsigned short x;
};
#endif
#else
struct __half
{
    unsigned short x;
};
#endif
