#pragma once

#if __has_include(<torch/torch.h>)
#include <torch/torch.h>
#define NERVE_TORCH_AVAILABLE 1
#else
#define NERVE_TORCH_AVAILABLE 0
#endif
